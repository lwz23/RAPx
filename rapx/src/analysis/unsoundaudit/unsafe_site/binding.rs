use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::PathBuf;

#[rustversion::since(1.96)]
use rustc_abi::{FieldIdx, VariantIdx};
#[rustversion::since(1.96)]
use rustc_attr_ir::LangItem;
use rustc_hir::Safety;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{
    AggregateKind, Body, CastKind, Location, Place, PlaceElem, ProjectionElem, Rvalue,
    StatementKind, TerminatorKind,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::symbol::sym;
use rustc_span::Span;
#[rustversion::before(1.96)]
use rustc_target::abi::{FieldIdx, VariantIdx};
use serde_json::json;

use super::super::summary::PlaceKey;
use super::{
    BindingStatus, FunctionKey, OperandRole, RawSpanAnchor, UnsafeSiteKind, UnsafeSiteSeed,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum MirStage {
    Built,
    Solver,
}

fn place_key(place: Place<'_>) -> PlaceKey {
    PlaceKey::new(format!("{place:?}"))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MirEventLocation {
    pub stage: MirStage,
    pub block: u32,
    pub statement: u32,
    pub event: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MirOperandAnchor {
    pub role: OperandRole,
    pub type_fingerprint: String,
    pub place: Option<PlaceKey>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MirEventAnchor {
    pub owner: FunctionKey,
    pub kind: UnsafeSiteKind,
    pub raw_span: RawSpanAnchor,
    pub hir_context: String,
    pub callee: Option<String>,
    pub indirect_call: bool,
    pub operands: Vec<MirOperandAnchor>,
    pub result_type: Option<String>,
    pub result_place: Option<PlaceKey>,
    pub shape: String,
    pub location: MirEventLocation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SiteBindingRecord {
    pub site_id: String,
    pub status: BindingStatus,
    pub built_candidate_count: usize,
    pub solver_candidate_count: usize,
    pub built: Option<MirEventLocation>,
    pub solver: Option<MirEventLocation>,
    pub(crate) solver_event: Option<MirEventAnchor>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BoundUnsafeSite {
    pub seed: UnsafeSiteSeed,
    pub built: MirEventLocation,
    pub solver: MirEventLocation,
    pub operands: Vec<MirOperandAnchor>,
    pub result_place: Option<PlaceKey>,
}

impl BoundUnsafeSite {
    pub fn from_record(seed: &UnsafeSiteSeed, record: &SiteBindingRecord) -> Option<Self> {
        (record.status == BindingStatus::BoundExact).then(|| Self {
            seed: seed.clone(),
            built: record
                .built
                .clone()
                .expect("bound site has built MIR location"),
            solver: record
                .solver
                .clone()
                .expect("bound site has solver MIR location"),
            operands: record
                .solver_event
                .as_ref()
                .expect("bound site has solver MIR event")
                .operands
                .clone(),
            result_place: record
                .solver_event
                .as_ref()
                .expect("bound site has solver MIR event")
                .result_place
                .clone(),
        })
    }
}

pub(crate) fn maybe_write_binding_sidecar(
    records: &BTreeMap<String, SiteBindingRecord>,
    built_events: &BTreeMap<FunctionKey, Vec<MirEventAnchor>>,
    solver_events: &BTreeMap<FunctionKey, Vec<MirEventAnchor>>,
) -> Result<(), String> {
    let Some(output_path) = env::var_os("UNSOUND_SCANNER_BINDING_JSON_OUT") else {
        return Ok(());
    };
    let mir_events = [("built", built_events), ("solver", solver_events)]
        .into_iter()
        .flat_map(|(stage, owners)| {
            owners.values().flatten().map(move |event| {
                json!({
                    "stage": stage,
                    "owner": event.owner.0,
                    "kind": event.kind.as_str(),
                    "span": {
                        "path": event.raw_span.stable.path,
                        "start_line": event.raw_span.stable.start.line,
                        "start_column": event.raw_span.stable.start.column,
                        "end_line": event.raw_span.stable.end.line,
                        "end_column": event.raw_span.stable.end.column,
                        "lo_byte": event.raw_span.lo_byte,
                        "hi_byte": event.raw_span.hi_byte,
                        "syntax_context": event.raw_span.syntax_context,
                    },
                    "hir_context": event.hir_context,
                    "callee": event.callee,
                    "indirect_call": event.indirect_call,
                    "operands": event.operands.iter().map(|operand| json!({
                        "role": operand.role.canonical_token(),
                        "type_fingerprint": operand.type_fingerprint,
                    })).collect::<Vec<_>>(),
                    "result_type": event.result_type,
                    "shape": event.shape,
                    "location": location_json(&event.location),
                })
            })
        })
        .collect::<Vec<_>>();
    let payload = json!({
        "schema_version": "unsafe-site-binding-v1",
        "sites": records.values().map(|record| json!({
            "site_id": record.site_id,
            "status": binding_status_name(record.status),
            "built_candidate_count": record.built_candidate_count,
            "solver_candidate_count": record.solver_candidate_count,
            "built": record.built.as_ref().map(location_json),
            "solver": record.solver.as_ref().map(location_json),
        })).collect::<Vec<_>>(),
        "mir_events": mir_events,
    });
    let mut bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|error| format!("failed to serialize unsafe-site binding sidecar: {error}"))?;
    bytes.push(b'\n');
    crate::utils::unit_output::atomic_write(&PathBuf::from(output_path), &bytes)
        .map_err(|error| format!("failed to write unsafe-site binding sidecar: {error}"))
}

fn binding_status_name(status: BindingStatus) -> &'static str {
    match status {
        BindingStatus::BoundExact => "bound_exact",
        BindingStatus::Unbound => "unbound",
        BindingStatus::Ambiguous => "ambiguous",
        BindingStatus::Elided => "elided",
    }
}

fn location_json(location: &MirEventLocation) -> serde_json::Value {
    json!({
        "stage": match location.stage {
            MirStage::Built => "built",
            MirStage::Solver => "solver",
        },
        "block": location.block,
        "statement": location.statement,
        "event": location.event,
    })
}

pub(crate) fn bind_owner(
    sites: &[UnsafeSiteSeed],
    built: &[MirEventAnchor],
    solver: &[MirEventAnchor],
) -> Result<Vec<SiteBindingRecord>, String> {
    if sites.is_empty() {
        return Ok(Vec::new());
    }
    let owner = &sites[0].owner;
    let mut site_ids = BTreeSet::new();
    let mut ordinals = BTreeSet::new();
    for site in sites {
        if &site.owner != owner {
            return Err("binding input contains multiple THIR owners".to_string());
        }
        if !site_ids.insert(site.site_id()) || !ordinals.insert(site.ordinal) {
            return Err("binding input contains duplicate THIR site identity".to_string());
        }
    }
    validate_events(owner, MirStage::Built, built)?;
    validate_events(owner, MirStage::Solver, solver)?;

    let mut ordered_sites = sites.iter().collect::<Vec<_>>();
    ordered_sites.sort_by_key(|site| (site.ordinal, site.site_id()));
    let mut ordered_built = built.iter().collect::<Vec<_>>();
    ordered_built.sort();
    let mut ordered_solver = solver.iter().collect::<Vec<_>>();
    ordered_solver.sort();

    let built_candidates = candidate_matrix(&ordered_sites, &ordered_built);
    let built_resolution = resolve_components(&built_candidates, ordered_built.len());
    let solver_candidates = candidate_matrix(&ordered_sites, &ordered_solver);
    let solver_resolution = resolve_components(&solver_candidates, ordered_solver.len());

    let mut records = ordered_sites
        .iter()
        .enumerate()
        .map(|(index, site)| SiteBindingRecord {
            site_id: site.site_id(),
            status: if built_candidates[index].is_empty() {
                BindingStatus::Unbound
            } else {
                BindingStatus::Ambiguous
            },
            built_candidate_count: built_candidates[index].len(),
            solver_candidate_count: solver_candidates[index].len(),
            built: None,
            solver: None,
            solver_event: None,
        })
        .collect::<Vec<_>>();

    for index in 0..records.len() {
        match (built_resolution[index], solver_resolution[index]) {
            (ComponentResolution::NoCandidate, _) => {
                records[index].status = BindingStatus::Unbound;
            }
            (ComponentResolution::Ambiguous, _) => {
                records[index].status = BindingStatus::Ambiguous;
            }
            (ComponentResolution::Unique(built), ComponentResolution::NoCandidate) => {
                records[index].status = BindingStatus::Elided;
                records[index].built = Some(ordered_built[built].location.clone());
            }
            (ComponentResolution::Unique(_), ComponentResolution::Ambiguous) => {
                records[index].status = BindingStatus::Ambiguous;
            }
            (ComponentResolution::Unique(built), ComponentResolution::Unique(solver)) => {
                records[index].status = BindingStatus::BoundExact;
                records[index].built = Some(ordered_built[built].location.clone());
                records[index].solver = Some(ordered_solver[solver].location.clone());
                records[index].solver_event = Some((*ordered_solver[solver]).clone());
            }
        }
    }
    Ok(records)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComponentResolution {
    NoCandidate,
    Ambiguous,
    Unique(usize),
}

fn resolve_components(candidates: &[Vec<usize>], right_len: usize) -> Vec<ComponentResolution> {
    let mut result = vec![ComponentResolution::NoCandidate; candidates.len()];
    let mut visited = vec![false; candidates.len()];
    for start in 0..candidates.len() {
        if visited[start] || candidates[start].is_empty() {
            continue;
        }
        let mut left = BTreeSet::from([start]);
        let mut right = BTreeSet::new();
        let mut frontier = vec![start];
        visited[start] = true;
        while let Some(site) = frontier.pop() {
            for &event in &candidates[site] {
                if event >= right_len || !right.insert(event) {
                    continue;
                }
                for (other, other_candidates) in candidates.iter().enumerate() {
                    if !visited[other] && other_candidates.contains(&event) {
                        visited[other] = true;
                        left.insert(other);
                        frontier.push(other);
                    }
                }
            }
        }
        let left = left.into_iter().collect::<Vec<_>>();
        let right = right.into_iter().collect::<Vec<_>>();
        let right_positions = right
            .iter()
            .enumerate()
            .map(|(position, event)| (*event, position))
            .collect::<BTreeMap<_, _>>();
        let local = left
            .iter()
            .map(|site| {
                candidates[*site]
                    .iter()
                    .map(|event| right_positions[event])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        match resolve_unique(&local, right.len()) {
            UniqueResolution::Unique(matching) => {
                for (position, site) in left.iter().copied().enumerate() {
                    result[site] = ComponentResolution::Unique(right[matching[position]]);
                }
            }
            UniqueResolution::NoComplete | UniqueResolution::Ambiguous => {
                for site in left {
                    result[site] = ComponentResolution::Ambiguous;
                }
            }
        }
    }
    result
}

fn validate_events(
    owner: &FunctionKey,
    expected_stage: MirStage,
    events: &[MirEventAnchor],
) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    for event in events {
        if &event.owner != owner {
            return Err("MIR binding event owner differs from THIR owner".to_string());
        }
        if event.location.stage != expected_stage {
            return Err("MIR binding event is in the wrong stage".to_string());
        }
        if !unique.insert(event) {
            return Err("duplicate MIR binding event".to_string());
        }
    }
    Ok(())
}

fn candidate_matrix(sites: &[&UnsafeSiteSeed], events: &[&MirEventAnchor]) -> Vec<Vec<usize>> {
    sites
        .iter()
        .map(|site| {
            events
                .iter()
                .enumerate()
                .filter_map(|(index, event)| compatible(site, event).then_some(index))
                .collect()
        })
        .collect()
}

fn compatible(site: &UnsafeSiteSeed, event: &MirEventAnchor) -> bool {
    let lowered_transmute = event.shape == "unsafe_call:lowered_transmute";
    if site.owner != event.owner
        || site.kind != event.kind
        || !span_compatible(&site.raw_span, &event.raw_span)
        || (!lowered_transmute && event.shape != site.kind.as_str())
        || site.result.as_ref().map(|result| &result.type_fingerprint) != event.result_type.as_ref()
    {
        return false;
    }
    match site.kind {
        UnsafeSiteKind::UnsafeCall => match &site.callee {
            Some(callee) if event.indirect_call || event.callee.as_ref() != Some(callee) => {
                return false
            }
            None if !event.indirect_call || event.callee.is_some() => return false,
            _ => {}
        },
        _ if site.callee != event.callee => return false,
        _ => {}
    }
    let site_operands = site
        .operands
        .iter()
        .filter(|operand| !lowered_transmute || operand.role != OperandRole::Callee)
        .collect::<Vec<_>>();
    site_operands.len() == event.operands.len()
        && site_operands
            .into_iter()
            .zip(&event.operands)
            .all(|(site, event)| {
                site.role == event.role && site.type_fingerprint == event.type_fingerprint
            })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum UniqueResolution {
    NoComplete,
    Ambiguous,
    Unique(Vec<usize>),
}

fn resolve_unique(candidates: &[Vec<usize>], right_len: usize) -> UniqueResolution {
    let Some(matching) = complete_matching(candidates, right_len, None) else {
        return UniqueResolution::NoComplete;
    };
    for (left, right) in matching.iter().copied().enumerate() {
        if complete_matching(candidates, right_len, Some((left, right))).is_some() {
            return UniqueResolution::Ambiguous;
        }
    }
    UniqueResolution::Unique(matching)
}

fn complete_matching(
    candidates: &[Vec<usize>],
    right_len: usize,
    banned: Option<(usize, usize)>,
) -> Option<Vec<usize>> {
    let mut right_to_left = vec![None; right_len];
    for left in 0..candidates.len() {
        let mut seen = vec![false; right_len];
        if !augment(left, candidates, banned, &mut seen, &mut right_to_left) {
            return None;
        }
    }
    let mut left_to_right = vec![usize::MAX; candidates.len()];
    for (right, left) in right_to_left.into_iter().enumerate() {
        if let Some(left) = left {
            left_to_right[left] = right;
        }
    }
    left_to_right
        .iter()
        .all(|right| *right != usize::MAX)
        .then_some(left_to_right)
}

fn augment(
    left: usize,
    candidates: &[Vec<usize>],
    banned: Option<(usize, usize)>,
    seen: &mut [bool],
    right_to_left: &mut [Option<usize>],
) -> bool {
    for &right in &candidates[left] {
        if banned == Some((left, right)) || seen[right] {
            continue;
        }
        seen[right] = true;
        if right_to_left[right].is_none()
            || augment(
                right_to_left[right].expect("checked Some"),
                candidates,
                banned,
                seen,
                right_to_left,
            )
        {
            right_to_left[right] = Some(left);
            return true;
        }
    }
    false
}

pub(crate) fn enumerate_mir_events<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_id: LocalDefId,
    body: &Body<'tcx>,
    stage: MirStage,
) -> Result<Vec<MirEventAnchor>, String> {
    let owner = FunctionKey::new(tcx.def_path_str(owner_id.to_def_id()));
    let static_locals = static_pointer_locals(tcx, body);
    let mut collector = PlaceEventCollector {
        tcx,
        body,
        owner: owner.clone(),
        stage,
        static_locals,
        events: Vec::new(),
        errors: Vec::new(),
    };
    collector.visit_body(body);
    if !collector.errors.is_empty() {
        return Err(collector.errors.join("; "));
    }
    let mut events = collector.events;
    for (block, data) in body.basic_blocks.iter_enumerated() {
        for (statement_index, statement) in data.statements.iter().enumerate() {
            let StatementKind::Assign(box (destination, rvalue)) = &statement.kind else {
                continue;
            };
            let location = Location {
                block,
                statement_index,
            };
            if let Rvalue::Aggregate(kind, _) = rvalue {
                if let AggregateKind::Adt(def, variant, ..) = &**kind {
                    if is_restricted_initialization(tcx, *def) {
                        push_event(
                            tcx,
                            body,
                            &owner,
                            stage,
                            location,
                            statement.source_info.span,
                            UnsafeSiteKind::RestrictedValidityInitialization,
                            Some(tcx.def_path_str(*def)),
                            false,
                            Vec::new(),
                            Some(destination.ty(&body.local_decls, tcx).ty),
                            Some(place_key(*destination)),
                            &mut events,
                        )?;
                    }
                    if variant_has_unsafe_fields(tcx, *def, *variant) {
                        push_event(
                            tcx,
                            body,
                            &owner,
                            stage,
                            location,
                            statement.source_info.span,
                            UnsafeSiteKind::UnsafeFieldInitialization,
                            Some(tcx.def_path_str(*def)),
                            false,
                            Vec::new(),
                            Some(destination.ty(&body.local_decls, tcx).ty),
                            Some(place_key(*destination)),
                            &mut events,
                        )?;
                    }
                }
            }
            if is_unsafe_binder_rvalue(rvalue) {
                push_event(
                    tcx,
                    body,
                    &owner,
                    stage,
                    location,
                    statement.source_info.span,
                    UnsafeSiteKind::UnsafeBinderCast,
                    None,
                    false,
                    Vec::new(),
                    Some(destination.ty(&body.local_decls, tcx).ty),
                    Some(place_key(*destination)),
                    &mut events,
                )?;
            }
            if let Rvalue::Cast(CastKind::Transmute, operand, target) = rvalue {
                let def = tcx
                    .get_diagnostic_item(sym::transmute)
                    .ok_or_else(|| "rustc did not expose the transmute intrinsic".to_string())?;
                push_event(
                    tcx,
                    body,
                    &owner,
                    stage,
                    location,
                    statement.source_info.span,
                    UnsafeSiteKind::UnsafeCall,
                    Some(tcx.def_path_str(def)),
                    false,
                    vec![(
                        OperandRole::Argument(0),
                        operand.ty(&body.local_decls, tcx),
                        operand.place().map(place_key),
                    )],
                    Some(*target),
                    Some(place_key(*destination)),
                    &mut events,
                )?;
                events
                    .last_mut()
                    .expect("transmute event was just inserted")
                    .shape = "unsafe_call:lowered_transmute".to_string();
            }
        }
        let location = Location {
            block,
            statement_index: data.statements.len(),
        };
        let terminator = data.terminator();
        match &terminator.kind {
            TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } => {
                let function_ty = func.ty(&body.local_decls, tcx);
                let raw = func.const_fn_def().map(|(def, _)| def);
                let (kind, callee, indirect) = classify_call(tcx, owner_id, function_ty, raw);
                if let Some(kind) = kind {
                    let mut operands = vec![(
                        OperandRole::Callee,
                        function_ty,
                        func.place().map(place_key),
                    )];
                    operands.extend(args.iter().enumerate().map(|(index, argument)| {
                        (
                            OperandRole::Argument(index as u32),
                            argument.node.ty(&body.local_decls, tcx),
                            argument.node.place().map(place_key),
                        )
                    }));
                    push_event(
                        tcx,
                        body,
                        &owner,
                        stage,
                        location,
                        terminator.source_info.span,
                        kind,
                        callee,
                        indirect,
                        operands,
                        Some(destination.ty(&body.local_decls, tcx).ty),
                        Some(place_key(*destination)),
                        &mut events,
                    )?;
                }
                if raw.is_some_and(|def| is_explicit_drop(tcx, def)) {
                    let operands = std::iter::once((
                        OperandRole::Callee,
                        function_ty,
                        func.place().map(place_key),
                    ))
                    .chain(args.iter().enumerate().map(|(index, argument)| {
                        (
                            OperandRole::Argument(index as u32),
                            argument.node.ty(&body.local_decls, tcx),
                            argument.node.place().map(place_key),
                        )
                    }))
                    .collect();
                    push_event(
                        tcx,
                        body,
                        &owner,
                        stage,
                        location,
                        terminator.source_info.span,
                        UnsafeSiteKind::ExplicitDropCall,
                        raw.map(|def| tcx.def_path_str(def)),
                        false,
                        operands,
                        Some(destination.ty(&body.local_decls, tcx).ty),
                        Some(place_key(*destination)),
                        &mut events,
                    )?;
                }
            }
            TerminatorKind::InlineAsm { .. } if is_counted_inline_asm(&terminator.kind) => {
                push_event(
                    tcx,
                    body,
                    &owner,
                    stage,
                    location,
                    terminator.source_info.span,
                    UnsafeSiteKind::InlineAssembly,
                    None,
                    false,
                    Vec::new(),
                    Some(tcx.types.unit),
                    None,
                    &mut events,
                )?;
            }
            _ => {}
        }
    }
    events.sort();
    Ok(events)
}

fn push_event<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    owner: &FunctionKey,
    stage: MirStage,
    location: Location,
    span: Span,
    kind: UnsafeSiteKind,
    callee: Option<String>,
    indirect_call: bool,
    operands: Vec<(OperandRole, Ty<'tcx>, Option<PlaceKey>)>,
    result_type: Option<Ty<'tcx>>,
    result_place: Option<PlaceKey>,
    events: &mut Vec<MirEventAnchor>,
) -> Result<(), String> {
    let hir_context = body
        .source_info(location)
        .scope
        .lint_root(&body.source_scopes);
    let event = u32::try_from(events.len())
        .map_err(|_| "owner MIR unsafe-event count exceeds u32".to_string())?;
    events.push(MirEventAnchor {
        owner: owner.clone(),
        kind,
        raw_span: super::raw_span_anchor(tcx, span)?,
        hir_context: hir_context.map_or_else(String::new, |id| format!("{id:?}")),
        callee,
        indirect_call,
        operands: operands
            .into_iter()
            .map(|(role, ty, place)| MirOperandAnchor {
                role,
                type_fingerprint: super::type_fingerprint(tcx, ty),
                place,
            })
            .collect(),
        result_type: result_type.map(|ty| super::type_fingerprint(tcx, ty)),
        result_place,
        shape: kind.as_str().to_string(),
        location: MirEventLocation {
            stage,
            block: location.block.index() as u32,
            statement: location.statement_index as u32,
            event,
        },
    });
    Ok(())
}

struct PlaceEventCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    owner: FunctionKey,
    stage: MirStage,
    static_locals: BTreeMap<rustc_middle::mir::Local, DefId>,
    events: Vec<MirEventAnchor>,
    errors: Vec<String>,
}

impl<'tcx> Visitor<'tcx> for PlaceEventCollector<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if !context.is_use() || skip_raw_borrow_context(context) {
            return;
        }
        let source_info = self.body.source_info(location);
        let mut place_ty = Place::from(place.local).ty(&self.body.local_decls, self.tcx);
        for (base, element) in place.as_ref().iter_projections() {
            let result_ty = place_ty.projection_ty(self.tcx, element);
            let event = match element {
                ProjectionElem::Deref if is_raw_pointer(place_ty.ty) => {
                    if let Some(def) = self.static_locals.get(&place.local).copied() {
                        let kind = if self.tcx.is_mutable_static(def) {
                            UnsafeSiteKind::MutableStaticAccess
                        } else if self.tcx.is_foreign_item(def) {
                            UnsafeSiteKind::ExternStaticAccess
                        } else {
                            UnsafeSiteKind::RawPointerDereference
                        };
                        Some((kind, Some(self.tcx.def_path_str(def)), OperandRole::Place))
                    } else {
                        Some((
                            UnsafeSiteKind::RawPointerDereference,
                            None,
                            OperandRole::Address,
                        ))
                    }
                }
                ProjectionElem::Field(field, _)
                    if matches!(place_ty.ty.kind(), ty::Adt(def, _) if def.is_union())
                        && !context.is_place_assignment() =>
                {
                    let _ = field;
                    Some((UnsafeSiteKind::UnionFieldAccess, None, OperandRole::Place))
                }
                ProjectionElem::Field(field, _)
                    if unsafe_field(self.tcx, place_ty, field)
                        && !context.is_place_assignment() =>
                {
                    Some((UnsafeSiteKind::UnsafeFieldUse, None, OperandRole::Place))
                }
                ProjectionElem::Field(_, _)
                    if layout_constrained(self.tcx, place_ty.ty)
                        && (context.is_place_assignment() || context.is_borrow()) =>
                {
                    let kind = if context.is_borrow() && !context.is_mutating_use() {
                        UnsafeSiteKind::LayoutConstrainedBorrow
                    } else {
                        UnsafeSiteKind::LayoutConstrainedMutation
                    };
                    Some((kind, None, OperandRole::Place))
                }
                _ if is_unsafe_binder_projection(element) => {
                    Some((UnsafeSiteKind::UnsafeBinderCast, None, OperandRole::Place))
                }
                _ => None,
            };
            if let Some((kind, callee, role)) = event {
                let (operand_ty, result_ty) = if matches!(
                    kind,
                    UnsafeSiteKind::LayoutConstrainedMutation
                        | UnsafeSiteKind::LayoutConstrainedBorrow
                ) {
                    (
                        result_ty.ty,
                        layout_event_result_type(self.tcx, self.body, context, location)
                            .unwrap_or(result_ty.ty),
                    )
                } else {
                    (place_ty.ty, result_ty.ty)
                };
                let operands = if kind == UnsafeSiteKind::UnsafeBinderCast {
                    Vec::new()
                } else {
                    vec![(role, operand_ty, Some(place_key(base.to_place(self.tcx))))]
                };
                if let Err(error) = push_event(
                    self.tcx,
                    self.body,
                    &self.owner,
                    self.stage,
                    location,
                    source_info.span,
                    kind,
                    callee,
                    false,
                    operands,
                    Some(result_ty),
                    Some(place_key(*place)),
                    &mut self.events,
                ) {
                    self.errors.push(error);
                }
            }
            place_ty = result_ty;
        }
    }
}

#[rustversion::before(1.96)]
fn is_unsafe_binder_projection(_element: PlaceElem<'_>) -> bool {
    false
}

#[rustversion::since(1.96)]
fn is_unsafe_binder_projection(element: PlaceElem<'_>) -> bool {
    matches!(element, ProjectionElem::UnwrapUnsafeBinder(_))
}

fn layout_event_result_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    context: PlaceContext,
    location: Location,
) -> Option<Ty<'tcx>> {
    if context.is_place_assignment() {
        return Some(tcx.types.unit);
    }
    let statement = body
        .basic_blocks
        .get(location.block)?
        .statements
        .get(location.statement_index)?;
    let StatementKind::Assign(box (destination, Rvalue::Ref(..))) = &statement.kind else {
        return None;
    };
    Some(destination.ty(&body.local_decls, tcx).ty)
}

fn static_pointer_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> BTreeMap<rustc_middle::mir::Local, DefId> {
    let mut known = BTreeMap::new();
    for data in body.basic_blocks.iter() {
        for statement in &data.statements {
            let StatementKind::Assign(box (destination, rvalue)) = &statement.kind else {
                continue;
            };
            if !destination.projection.is_empty() {
                continue;
            }
            if let Some(def) = rvalue_use_operand(rvalue)
                .and_then(|operand| operand.constant())
                .and_then(|constant| constant.check_static_ptr(tcx))
            {
                known.insert(destination.local, def);
            }
        }
    }
    known
}

fn classify_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: LocalDefId,
    function_ty: Ty<'tcx>,
    raw: Option<DefId>,
) -> (Option<UnsafeSiteKind>, Option<String>, bool) {
    let signature = function_ty.fn_sig(tcx);
    let safety = signature.safety();
    if unsafe_call_requires_unsafe(tcx, raw, safety) {
        return (
            Some(UnsafeSiteKind::UnsafeCall),
            raw.map(|def| tcx.def_path_str(def)),
            raw.is_none(),
        );
    }
    if raw.is_some_and(|def| target_feature_call_requires_unsafe(tcx, owner, def)) {
        return (
            Some(UnsafeSiteKind::TargetFeatureCall),
            raw.map(|def| tcx.def_path_str(def)),
            false,
        );
    }
    (None, None, false)
}

#[rustversion::before(1.96)]
fn unsafe_call_requires_unsafe(_tcx: TyCtxt<'_>, _raw: Option<DefId>, safety: Safety) -> bool {
    safety == Safety::Unsafe
}

#[rustversion::since(1.96)]
fn unsafe_call_requires_unsafe(tcx: TyCtxt<'_>, raw: Option<DefId>, safety: Safety) -> bool {
    safety.is_unsafe() && !raw.is_some_and(|def| tcx.codegen_fn_attrs(def).safe_target_features)
}

#[rustversion::before(1.96)]
fn target_feature_call_requires_unsafe(tcx: TyCtxt<'_>, owner: LocalDefId, def: DefId) -> bool {
    let callee = &tcx.codegen_fn_attrs(def).target_features;
    let body = &tcx.body_codegen_attrs(owner.to_def_id()).target_features;
    !tcx.sess.target.options.is_like_wasm
        && !callee
            .iter()
            .all(|feature| body.iter().any(|active| active.name == feature.name))
}

#[rustversion::since(1.96)]
fn target_feature_call_requires_unsafe(tcx: TyCtxt<'_>, owner: LocalDefId, def: DefId) -> bool {
    tcx.def_kind(def).is_fn_like()
        && !tcx.is_target_feature_call_safe(
            &tcx.codegen_fn_attrs(def).target_features,
            &tcx.body_codegen_attrs(owner.to_def_id()).target_features,
        )
}

#[rustversion::before(1.96)]
fn is_restricted_initialization(tcx: TyCtxt<'_>, def: DefId) -> bool {
    use std::ops::Bound;
    tcx.layout_scalar_valid_range(def) != (Bound::Unbounded, Bound::Unbounded)
}

#[rustversion::since(1.96)]
fn is_restricted_initialization(_tcx: TyCtxt<'_>, _def: DefId) -> bool {
    false
}

#[rustversion::before(1.96)]
fn variant_has_unsafe_fields(_tcx: TyCtxt<'_>, _def: DefId, _variant: VariantIdx) -> bool {
    false
}

#[rustversion::since(1.96)]
fn variant_has_unsafe_fields(tcx: TyCtxt<'_>, def: DefId, variant: VariantIdx) -> bool {
    tcx.adt_def(def).variant(variant).has_unsafe_fields()
}

#[rustversion::before(1.96)]
fn unsafe_field(
    _tcx: TyCtxt<'_>,
    _place_ty: rustc_middle::mir::tcx::PlaceTy<'_>,
    _field: FieldIdx,
) -> bool {
    false
}

#[rustversion::since(1.96)]
fn unsafe_field(
    _tcx: TyCtxt<'_>,
    place_ty: rustc_middle::mir::PlaceTy<'_>,
    field: FieldIdx,
) -> bool {
    let ty::Adt(def, _) = place_ty.ty.kind() else {
        return false;
    };
    let variant = place_ty
        .variant_index
        .map_or_else(|| def.non_enum_variant(), |variant| def.variant(variant));
    variant.fields[field].safety.is_unsafe()
}

#[rustversion::before(1.96)]
fn layout_constrained(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    use std::ops::Bound;
    matches!(ty.kind(), ty::Adt(def, _) if tcx.layout_scalar_valid_range(def.did()) != (Bound::Unbounded, Bound::Unbounded))
}

#[rustversion::since(1.96)]
fn layout_constrained(_tcx: TyCtxt<'_>, _ty: Ty<'_>) -> bool {
    false
}

#[rustversion::before(1.96)]
fn is_raw_pointer(ty: Ty<'_>) -> bool {
    ty.is_unsafe_ptr()
}

#[rustversion::since(1.96)]
fn is_raw_pointer(ty: Ty<'_>) -> bool {
    ty.is_raw_ptr()
}

#[rustversion::before(1.96)]
fn is_unsafe_binder_rvalue(_rvalue: &Rvalue<'_>) -> bool {
    false
}

#[rustversion::since(1.96)]
fn is_unsafe_binder_rvalue(rvalue: &Rvalue<'_>) -> bool {
    matches!(rvalue, Rvalue::WrapUnsafeBinder(..))
}

#[rustversion::before(1.96)]
fn is_explicit_drop(_tcx: TyCtxt<'_>, _def: DefId) -> bool {
    false
}

#[rustversion::since(1.96)]
fn is_explicit_drop(tcx: TyCtxt<'_>, def: DefId) -> bool {
    tcx.trait_of_assoc(def)
        .is_some_and(|trait_def| tcx.is_lang_item(trait_def, LangItem::Drop))
}

#[rustversion::before(1.96)]
fn skip_raw_borrow_context(_context: PlaceContext) -> bool {
    false
}

#[rustversion::since(1.96)]
fn skip_raw_borrow_context(context: PlaceContext) -> bool {
    context.is_address_of()
}

#[rustversion::before(1.96)]
fn is_counted_inline_asm(_terminator: &TerminatorKind<'_>) -> bool {
    true
}

#[rustversion::since(1.96)]
fn is_counted_inline_asm(terminator: &TerminatorKind<'_>) -> bool {
    matches!(
        terminator,
        TerminatorKind::InlineAsm {
            asm_macro: rustc_middle::mir::InlineAsmMacro::Asm,
            ..
        }
    )
}

#[rustversion::before(1.96)]
fn rvalue_use_operand<'a, 'tcx>(
    rvalue: &'a Rvalue<'tcx>,
) -> Option<&'a rustc_middle::mir::Operand<'tcx>> {
    match rvalue {
        Rvalue::Use(operand) => Some(operand),
        _ => None,
    }
}

#[rustversion::since(1.96)]
fn rvalue_use_operand<'a, 'tcx>(
    rvalue: &'a Rvalue<'tcx>,
) -> Option<&'a rustc_middle::mir::Operand<'tcx>> {
    match rvalue {
        Rvalue::Use(operand, _) => Some(operand),
        _ => None,
    }
}

fn span_contains(outer: &RawSpanAnchor, inner: &RawSpanAnchor) -> bool {
    outer.stable.path == inner.stable.path
        && outer.syntax_context == inner.syntax_context
        && outer.lo_byte <= inner.lo_byte
        && outer.hi_byte >= inner.hi_byte
}

fn span_compatible(site: &RawSpanAnchor, event: &RawSpanAnchor) -> bool {
    site == event
        || span_contains(site, event)
        || span_contains(event, site)
        || stable_span_contains(&site.stable, &event.stable)
        || stable_span_contains(&event.stable, &site.stable)
}

fn stable_span_contains(
    outer: &crate::analysis::unsoundaudit::summary::StableSpan,
    inner: &crate::analysis::unsoundaudit::summary::StableSpan,
) -> bool {
    outer.path == inner.path && outer.start <= inner.start && outer.end >= inner.end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::unsoundaudit::summary::{StablePosition, StableSpan};
    use crate::analysis::unsoundaudit::unsafe_site::{OperandAnchor, ResultAnchor};

    fn span(lo: u32, hi: u32) -> RawSpanAnchor {
        RawSpanAnchor::new(
            StableSpan::new(
                "src/lib.rs",
                StablePosition::new(10, lo + 1),
                StablePosition::new(10, hi + 1),
            ),
            lo,
            hi,
            "#0",
        )
    }

    fn site(
        ordinal: u32,
        kind: UnsafeSiteKind,
        callee: Option<&str>,
        operands: &[(OperandRole, &str)],
        result: Option<&str>,
    ) -> UnsafeSiteSeed {
        UnsafeSiteSeed {
            adapter_id: "adapter".to_string(),
            owner: FunctionKey::new("crate::owner"),
            owner_def_path_hash: "owner-hash".to_string(),
            ordinal,
            kind,
            raw_span: span(ordinal * 10, ordinal * 10 + 8),
            hir_context: "HirId(1)".to_string(),
            callee: callee.map(str::to_string),
            operands: operands
                .iter()
                .enumerate()
                .map(|(index, (role, ty))| OperandAnchor {
                    role: role.clone(),
                    span: span(100 + index as u32, 101 + index as u32),
                    type_fingerprint: (*ty).to_string(),
                    shape: "operand".to_string(),
                })
                .collect(),
            result: result.map(|ty| ResultAnchor {
                span: span(200, 201),
                type_fingerprint: ty.to_string(),
                shape: "result".to_string(),
            }),
        }
    }

    fn event(site: &UnsafeSiteSeed, stage: MirStage, block: u32, statement: u32) -> MirEventAnchor {
        MirEventAnchor {
            owner: site.owner.clone(),
            kind: site.kind,
            raw_span: site.raw_span.clone(),
            hir_context: site.hir_context.clone(),
            callee: site.callee.clone(),
            indirect_call: site.kind == UnsafeSiteKind::UnsafeCall && site.callee.is_none(),
            operands: site
                .operands
                .iter()
                .map(|operand| MirOperandAnchor {
                    role: operand.role.clone(),
                    type_fingerprint: operand.type_fingerprint.clone(),
                    place: None,
                })
                .collect(),
            result_type: site
                .result
                .as_ref()
                .map(|result| result.type_fingerprint.clone()),
            result_place: None,
            shape: site.kind.as_str().to_string(),
            location: MirEventLocation {
                stage,
                block,
                statement,
                event: 0,
            },
        }
    }

    #[test]
    fn binds_direct_and_function_pointer_calls_with_actuals_and_destination() {
        let direct = site(
            0,
            UnsafeSiteKind::UnsafeCall,
            Some("core::slice::get_unchecked"),
            &[
                (OperandRole::Callee, "fn"),
                (OperandRole::Argument(0), "&[u8]"),
            ],
            Some("&u8"),
        );
        let indirect = site(
            1,
            UnsafeSiteKind::UnsafeCall,
            None,
            &[
                (OperandRole::Callee, "unsafe fn(*const u8)"),
                (OperandRole::Argument(0), "*const u8"),
            ],
            Some("u8"),
        );
        let sites = vec![direct.clone(), indirect.clone()];
        let built = vec![
            event(&indirect, MirStage::Built, 2, 0),
            event(&direct, MirStage::Built, 1, 0),
        ];
        let mut direct_solver = event(&direct, MirStage::Solver, 3, 1);
        direct_solver.operands[1].place = Some(PlaceKey::new("_1"));
        direct_solver.result_place = Some(PlaceKey::new("_2"));
        let solver = vec![direct_solver, event(&indirect, MirStage::Solver, 4, 1)];
        let records = bind_owner(&sites, &built, &solver).unwrap();
        assert_eq!(records.len(), 2);
        assert!(records
            .iter()
            .all(|record| record.status == BindingStatus::BoundExact));
        assert_eq!(records[0].built.as_ref().unwrap().block, 1);
        assert_eq!(records[0].solver.as_ref().unwrap().block, 3);
        assert_eq!(records[1].built.as_ref().unwrap().block, 2);
        assert_eq!(records[1].solver.as_ref().unwrap().block, 4);
        let bound = BoundUnsafeSite::from_record(&direct, &records[0]).unwrap();
        assert_eq!(bound.operands[1].place, Some(PlaceKey::new("_1")));
        assert_eq!(bound.result_place, Some(PlaceKey::new("_2")));
    }

    #[test]
    fn lowered_transmute_binds_without_fabricating_a_call_event() {
        let transmute = site(
            0,
            UnsafeSiteKind::UnsafeCall,
            Some("core::intrinsics::transmute"),
            &[
                (OperandRole::Callee, "unsafe fn(&str) -> &str"),
                (OperandRole::Argument(0), "&str"),
            ],
            Some("&str"),
        );
        let lowered = |stage| {
            let mut event = event(&transmute, stage, 1, 2);
            event.operands.remove(0);
            event.shape = "unsafe_call:lowered_transmute".to_string();
            event
        };
        let records = bind_owner(
            &[transmute.clone()],
            &[lowered(MirStage::Built)],
            &[lowered(MirStage::Solver)],
        )
        .unwrap();
        assert_eq!(records[0].status, BindingStatus::BoundExact);
    }

    #[test]
    fn same_span_sites_use_kind_and_roles_not_insertion_order() {
        let raw = site(
            0,
            UnsafeSiteKind::RawPointerDereference,
            None,
            &[(OperandRole::Address, "*const u8")],
            Some("u8"),
        );
        let mut union = site(
            1,
            UnsafeSiteKind::UnionFieldAccess,
            None,
            &[(OperandRole::Place, "U")],
            Some("u8"),
        );
        union.raw_span = raw.raw_span.clone();
        let sites = vec![raw.clone(), union.clone()];
        let mut built = vec![
            event(&union, MirStage::Built, 1, 0),
            event(&raw, MirStage::Built, 1, 0),
        ];
        let mut solver = vec![
            event(&raw, MirStage::Solver, 2, 0),
            event(&union, MirStage::Solver, 2, 0),
        ];
        let first = bind_owner(&sites, &built, &solver).unwrap();
        built.reverse();
        solver.reverse();
        let second = bind_owner(&sites, &built, &solver).unwrap();
        assert_eq!(first, second);
        assert!(first
            .iter()
            .all(|record| record.status == BindingStatus::BoundExact));
    }

    #[test]
    fn macro_expansion_can_bind_by_unique_source_callsite_structure() {
        let site = site(0, UnsafeSiteKind::UnsafeBinderCast, None, &[], Some("&u8"));
        let mut built = event(&site, MirStage::Built, 1, 0);
        built.raw_span = RawSpanAnchor::new(
            StableSpan::new(
                "src/lib.rs",
                StablePosition::new(10, 1),
                StablePosition::new(10, 9),
            ),
            1_000,
            1_020,
            "#9",
        );
        let mut solver = built.clone();
        solver.location.stage = MirStage::Solver;
        let records = bind_owner(&[site], &[built], &[solver]).unwrap();
        assert_eq!(records[0].status, BindingStatus::BoundExact);
        assert_eq!(records[0].built_candidate_count, 1);
        assert_eq!(records[0].solver_candidate_count, 1);
    }

    #[test]
    fn zero_candidates_are_unbound_and_solver_elision_is_explicit() {
        let site = site(
            0,
            UnsafeSiteKind::RawPointerDereference,
            None,
            &[(OperandRole::Address, "*const u8")],
            Some("u8"),
        );
        let unbound = bind_owner(std::slice::from_ref(&site), &[], &[]).unwrap();
        assert_eq!(unbound[0].status, BindingStatus::Unbound);
        assert_eq!(unbound[0].built_candidate_count, 0);

        let built = vec![event(&site, MirStage::Built, 1, 0)];
        let elided = bind_owner(std::slice::from_ref(&site), &built, &[]).unwrap();
        assert_eq!(elided[0].status, BindingStatus::Elided);
        assert_eq!(elided[0].built_candidate_count, 1);
        assert_eq!(elided[0].solver_candidate_count, 0);
    }

    #[test]
    fn an_unbound_site_does_not_hide_a_disjoint_exact_binding() {
        let exact = site(0, UnsafeSiteKind::InlineAssembly, None, &[], Some("()"));
        let missing = site(
            1,
            UnsafeSiteKind::RawPointerDereference,
            None,
            &[(OperandRole::Address, "*const u8")],
            Some("u8"),
        );
        let records = bind_owner(
            &[exact.clone(), missing],
            &[event(&exact, MirStage::Built, 1, 0)],
            &[event(&exact, MirStage::Solver, 2, 0)],
        )
        .unwrap();
        assert_eq!(records[0].status, BindingStatus::BoundExact);
        assert_eq!(records[1].status, BindingStatus::Unbound);
    }

    #[test]
    fn multiple_complete_matchings_are_ambiguous_not_arbitrarily_selected() {
        let first = site(0, UnsafeSiteKind::InlineAssembly, None, &[], Some("()"));
        let mut second = first.clone();
        second.ordinal = 1;
        let sites = vec![first.clone(), second.clone()];
        let mut built_second = event(&first, MirStage::Built, 2, 0);
        built_second.location.event = 1;
        let built = vec![event(&first, MirStage::Built, 1, 0), built_second];
        let mut solver_second = event(&first, MirStage::Solver, 4, 0);
        solver_second.location.event = 1;
        let solver = vec![event(&first, MirStage::Solver, 3, 0), solver_second];
        let records = bind_owner(&sites, &built, &solver).unwrap();
        assert!(records
            .iter()
            .all(|record| record.status == BindingStatus::Ambiguous));
        assert!(records
            .iter()
            .all(|record| record.built_candidate_count == 2));
    }

    #[test]
    fn wrong_owner_stage_and_duplicate_locations_fail_closed() {
        let site = site(0, UnsafeSiteKind::InlineAssembly, None, &[], Some("()"));
        let mut wrong_owner = event(&site, MirStage::Built, 1, 0);
        wrong_owner.owner = FunctionKey::new("crate::other");
        assert!(bind_owner(std::slice::from_ref(&site), &[wrong_owner], &[]).is_err());

        let built = event(&site, MirStage::Built, 1, 0);
        assert!(bind_owner(std::slice::from_ref(&site), &[built.clone(), built], &[]).is_err());

        let solver_as_built = event(&site, MirStage::Solver, 1, 0);
        assert!(bind_owner(std::slice::from_ref(&site), &[solver_as_built], &[]).is_err());
    }
}
