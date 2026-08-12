use super::summary::{
    classify_primary_with_secondary, map_call_outputs, scc_cycle_token, solve_summaries,
    AbstractOrigin, AbstractValue, CallBoundary, CallMapping, CanonicalWitness,
    ContractRequirement, FailureClass, Finding, FunctionKey, FunctionSummary, Obligation,
    Operation, OperationKind, OriginKey, Pattern, PlaceKey, Predicate, ProgramPoint, RuleId,
    SinkObligation, Source, SourceKind, StablePosition, StableSpan, WitnessStep, WitnessStepKind,
    WriteEffect,
};
use rustc_hir::{
    def::DefKind,
    def_id::{DefId, LocalDefId, LOCAL_CRATE},
    LangItem, Mutability, Safety,
};
use rustc_middle::{
    middle::privacy::Level,
    mir::{
        AggregateKind, BasicBlock, BinOp, Body, CastKind, Location, Operand, Place, ProjectionElem,
        Rvalue, StatementKind, TerminatorKind, UnOp, RETURN_PLACE,
    },
    ty::{self, Instance, InstanceKind, TyCtxt, TypeVisitableExt},
};
use rustc_span::{sym, FileNameDisplayPreference, Span};
use rustc_target::spec::abi::Abi;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ValueFacts {
    value: AbstractValue,
    dependencies: BTreeSet<PlaceKey>,
    value_flow: BTreeSet<PlaceKey>,
    len_of: Option<PlaceKey>,
    range_end: Option<PlaceKey>,
    pointer_base: Option<PlaceKey>,
    pointer_offset: Option<PlaceKey>,
    maybe_uninit_initialized: Option<bool>,
    generic_capability: bool,
    ffi_output: bool,
    open_behavior: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CallOperand {
    Place(PlaceKey),
    Constant(u64),
    Unknown,
}

#[derive(Clone, Debug)]
struct CallSite {
    callee: Option<FunctionKey>,
    raw_def: Option<DefId>,
    disposition: BoundaryDisposition,
    args: Vec<CallOperand>,
    destination: PlaceKey,
    point: ProgramPoint,
}

#[derive(Clone, Debug)]
struct CompareFact {
    result: PlaceKey,
    op: BinOp,
    left: CallOperand,
    right: CallOperand,
    point: ProgramPoint,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LengthFact {
    result: PlaceKey,
    collection: PlaceKey,
}

#[derive(Clone, Debug)]
enum PredicateFact {
    IsEmpty { result: PlaceKey, slice: PlaceKey },
    IsNull { result: PlaceKey, pointer: PlaceKey },
    Utf8Result { result: PlaceKey, bytes: PlaceKey },
    IsErr { result: PlaceKey, checked: PlaceKey },
}

#[derive(Clone, Debug)]
struct BranchFact {
    result: PlaceKey,
    success: BasicBlock,
    point: ProgramPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValidationRegion {
    branch: Location,
    success: BasicBlock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum WriteKind {
    Assignment,
    CallReturn,
    OpaqueMutable,
    ForeignOut,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WriteFact {
    place: PlaceKey,
    point: ProgramPoint,
    kind: WriteKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryDisposition {
    ResolvedLocal,
    ExactFfiOut,
    SelectedOpenBehavior,
    ModeledRegistry,
    OpaqueDirect,
    OpaqueIndirect,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BoundaryInputs {
    resolved_local: bool,
    exact_ffi_out: bool,
    selected_open_behavior: bool,
    modeled_registry: bool,
    direct: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CfgPoint {
    block: u32,
    statement: u32,
}

#[derive(Clone, Debug)]
struct BodyFacts {
    def_id: LocalDefId,
    function: FunctionKey,
    root_span: StableSpan,
    arg_count: usize,
    has_self: bool,
    summary_seed: FunctionSummary,
    values: BTreeMap<PlaceKey, ValueFacts>,
    calls: Vec<CallSite>,
    compares: Vec<CompareFact>,
    lengths: BTreeSet<LengthFact>,
    predicates: Vec<PredicateFact>,
    branches: Vec<BranchFact>,
    writes: Vec<WriteFact>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PreoptTransmute {
    first_failure: Operation,
    sink: Operation,
}

#[derive(Clone, Debug)]
struct SelectedRequirement {
    requirement: ContractRequirement,
    source: Source,
    witness: CanonicalWitness,
}

struct Engine<'tcx> {
    tcx: TyCtxt<'tcx>,
    project_root: PathBuf,
    crate_name: String,
    bodies: BTreeMap<FunctionKey, BodyFacts>,
    preopt_transmutes: BTreeMap<FunctionKey, BTreeSet<PreoptTransmute>>,
}

pub fn analyze(tcx: TyCtxt<'_>) -> Result<Vec<Finding>, String> {
    Engine::new(tcx).run()
}

impl<'tcx> Engine<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        let project_root = env::var_os("UNSOUND_SCANNER_PROJECT_ROOT")
            .map(PathBuf::from)
            .and_then(|path| path.canonicalize().ok())
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let crate_name = env::var("UNSOUND_SCANNER_RAP_CRATE_NAME")
            .unwrap_or_else(|_| tcx.crate_name(LOCAL_CRATE).to_string());
        Self {
            tcx,
            project_root,
            crate_name,
            bodies: BTreeMap::new(),
            preopt_transmutes: BTreeMap::new(),
        }
    }

    fn run(mut self) -> Result<Vec<Finding>, String> {
        let ids = self.local_function_ids();
        self.preopt_transmutes = self.collect_preopt_transmutes(&ids)?;
        self.collect_bodies(&ids);
        let graph = self.call_graph();
        let seeds = self
            .bodies
            .iter()
            .map(|(key, facts)| (key.clone(), facts.summary_seed.clone()))
            .collect::<BTreeMap<_, _>>();
        let solved = solve_summaries(&graph, &seeds, |node, current| {
            let mut derived = FunctionSummary::empty(node.clone());
            let Some(caller) = self.bodies.get(node) else {
                return derived;
            };
            for call in &caller.calls {
                let Some(callee) = &call.callee else {
                    continue;
                };
                let Some(callee_summary) = current.get(callee) else {
                    continue;
                };
                let actuals = call
                    .args
                    .iter()
                    .map(|operand| self.operand_value(caller, operand))
                    .collect::<Vec<_>>();
                let mapping = caller
                    .summary_seed
                    .calls
                    .iter()
                    .find(|boundary| boundary.callee == *callee && boundary.point == call.point)
                    .map(|boundary| boundary.mapping.clone())
                    .unwrap_or_default();
                let out_values = BTreeMap::new();
                let mapped_outputs = map_call_outputs(
                    &mapping,
                    &callee_summary.return_value,
                    &out_values,
                    &actuals,
                );
                let reaches_return = self.place_reaches_return(caller, &call.destination);
                for requirement in &callee_summary.requirements {
                    if requirement.return_exposure && !reaches_return {
                        continue;
                    }
                    let mut requirement = requirement.clone();
                    requirement.subject = requirement.subject.substitute(&actuals);
                    requirement.collection = requirement
                        .collection
                        .as_ref()
                        .map(|value| value.substitute(&actuals));
                    derived.requirements.insert(requirement);
                }
                if reaches_return {
                    if let Some(returned) = mapped_outputs.get(&call.destination) {
                        derived
                            .return_value
                            .origins
                            .extend(returned.origins.iter().cloned());
                    }
                }
                derived
                    .cycle_tokens
                    .extend(callee_summary.cycle_tokens.iter().cloned());
            }
            derived
        });

        let components = super::summary::strongly_connected_components(&graph);
        let recursive = components
            .iter()
            .filter(|component| {
                component.len() > 1
                    || component.first().is_some_and(|node| {
                        graph
                            .get(node)
                            .is_some_and(|callees| callees.contains(node))
                    })
            })
            .flat_map(|component| {
                let token = scc_cycle_token(component);
                component
                    .iter()
                    .cloned()
                    .map(move |node| (node, token.clone()))
            })
            .collect::<BTreeMap<_, _>>();

        let mut findings = Vec::new();
        for (root, facts) in &self.bodies {
            if !self.is_public_safe_root(facts.def_id) {
                continue;
            }
            let Some(summary) = solved.get(root) else {
                continue;
            };
            let paths = self.shortest_paths(root, &graph, &recursive);
            for requirement in &summary.requirements {
                if requirement.return_exposure
                    && summary
                        .return_value
                        .origins
                        .is_disjoint(&requirement.subject.origins)
                {
                    continue;
                }
                let Some(source) = self.classify_source(facts, requirement) else {
                    continue;
                };
                if self.is_discharged(facts, requirement, &paths) {
                    continue;
                }
                let witness = self.build_witness(facts, requirement, &paths, &recursive);
                let selected = SelectedRequirement {
                    requirement: requirement.clone(),
                    source,
                    witness,
                };
                findings.push(self.finding(root, facts, selected));
            }
        }
        Ok(findings)
    }

    fn local_function_ids(&self) -> Vec<LocalDefId> {
        let mut ids = self
            .tcx
            .iter_local_def_id()
            .filter(|id| {
                let did = id.to_def_id();
                matches!(self.tcx.def_kind(did), DefKind::Fn | DefKind::AssocFn)
                    && self.tcx.is_mir_available(did)
            })
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| self.tcx.def_path_str(id.to_def_id()));
        ids
    }

    fn collect_preopt_transmutes(
        &self,
        ids: &[LocalDefId],
    ) -> Result<BTreeMap<FunctionKey, BTreeSet<PreoptTransmute>>, String> {
        for id in ids {
            if self
                .tcx
                .mir_drops_elaborated_and_const_checked(*id)
                .is_stolen()
            {
                return Err(format!(
                    "UnsoundAudit v2 requires unconsumed pre-optimization MIR; '{}' was already consumed",
                    self.tcx.def_path_str(id.to_def_id())
                ));
            }
        }
        let mut result = BTreeMap::new();
        for id in ids {
            let function = FunctionKey::new(self.tcx.def_path_str(id.to_def_id()));
            let signature_output = self
                .tcx
                .fn_sig(id.to_def_id())
                .skip_binder()
                .output()
                .skip_binder();
            let returns_static_reference =
                matches!(signature_output.kind(), ty::Ref(region, ..) if region.is_static());
            let slot = self.tcx.mir_drops_elaborated_and_const_checked(*id);
            let guard = slot.borrow();
            let body = &*guard;
            let mut items = BTreeSet::new();
            for (bb, data) in body.basic_blocks.iter_enumerated() {
                for (statement_index, statement) in data.statements.iter().enumerate() {
                    let StatementKind::Assign(box (
                        destination,
                        Rvalue::Cast(CastKind::Transmute, operand, target_ty),
                    )) = &statement.kind
                    else {
                        continue;
                    };
                    let source_ty = operand.ty(&body.local_decls, self.tcx);
                    let source_is_internal = operand.place().is_some_and(|place| {
                        place.local != RETURN_PLACE && place.local.index() > body.arg_count
                    });
                    if destination.local != RETURN_PLACE
                        || !destination.projection.is_empty()
                        || !source_ty.is_ref()
                        || !target_ty.is_ref()
                        || !returns_static_reference
                        || !source_is_internal
                    {
                        continue;
                    }
                    let location = Location {
                        block: bb,
                        statement_index,
                    };
                    let point = self.point(&function, location, statement.source_info.span);
                    items.insert(PreoptTransmute {
                        first_failure: Operation {
                            kind: OperationKind::LifetimeTransmute,
                            point: point.clone(),
                        },
                        sink: Operation {
                            kind: OperationKind::InvalidValueExposure,
                            point,
                        },
                    });
                }
            }
            if !items.is_empty() {
                result.insert(function, items);
            }
        }
        Ok(result)
    }

    fn collect_bodies(&mut self, ids: &[LocalDefId]) {
        for id in ids {
            let facts = self.extract_body(*id);
            self.bodies.insert(facts.function.clone(), facts);
        }
    }

    fn extract_body(&self, id: LocalDefId) -> BodyFacts {
        let did = id.to_def_id();
        let body = self.tcx.optimized_mir(did);
        let function = FunctionKey::new(self.tcx.def_path_str(did));
        let has_self = self
            .tcx
            .opt_associated_item(did)
            .is_some_and(|item| item.fn_has_self_parameter);
        let mut facts = BodyFacts {
            def_id: id,
            function: function.clone(),
            root_span: self.stable_span(self.tcx.def_span(did)),
            arg_count: body.arg_count,
            has_self,
            summary_seed: FunctionSummary::empty(function.clone()),
            values: BTreeMap::new(),
            calls: Vec::new(),
            compares: Vec::new(),
            lengths: BTreeSet::new(),
            predicates: Vec::new(),
            branches: Vec::new(),
            writes: Vec::new(),
        };
        for arg in body.args_iter() {
            facts.values.insert(
                PlaceKey::new(format!("_{0}", arg.index())),
                ValueFacts {
                    value: AbstractValue::new([AbstractOrigin::Formal(arg.index() as u32)]),
                    dependencies: BTreeSet::from([PlaceKey::new(format!("_{0}", arg.index()))]),
                    value_flow: BTreeSet::from([PlaceKey::new(format!("_{0}", arg.index()))]),
                    ..ValueFacts::default()
                },
            );
        }

        for (bb, data) in body.basic_blocks.iter_enumerated() {
            for (statement_index, statement) in data.statements.iter().enumerate() {
                let location = Location {
                    block: bb,
                    statement_index,
                };
                let point = self.point(&function, location, statement.source_info.span);
                if let StatementKind::Assign(box (destination, rvalue)) = &statement.kind {
                    self.extract_assignment(body, *destination, rvalue, point, &mut facts);
                }
            }
            let location = Location {
                block: bb,
                statement_index: data.statements.len(),
            };
            let terminator = data.terminator();
            let point = self.point(&function, location, terminator.source_info.span);
            match &terminator.kind {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } => self.extract_call(did, body, func, args, *destination, point, &mut facts),
                TerminatorKind::SwitchInt { discr, targets } => {
                    if let Some(place) = discr.place() {
                        facts.branches.push(BranchFact {
                            result: place_key(place),
                            success: targets.target_for_value(0),
                            point,
                        });
                    }
                }
                _ => {}
            }
        }

        self.add_local_requirements(body, &mut facts);
        if let Some(returned) = facts.values.get(&PlaceKey::new("_0")) {
            facts
                .summary_seed
                .return_value
                .origins
                .extend(returned.value.origins.iter().cloned());
        }
        facts
    }

    fn extract_assignment(
        &self,
        body: &Body<'tcx>,
        destination: Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        point: ProgramPoint,
        facts: &mut BodyFacts,
    ) {
        let destination_key = place_key(destination);
        let assignment_write = destination_write_invalidates(
            !destination.projection.is_empty(),
            facts.values.contains_key(&destination_key),
        );
        let mut value = match rvalue {
            Rvalue::Use(operand) => self.operand_facts(body, facts, operand, &point.span),
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
                let mut result = self.place_facts(body, facts, *place, point.span.clone());
                result.dependencies.insert(place_key(*place));
                result
            }
            Rvalue::Cast(_, operand, _) => self.operand_facts(body, facts, operand, &point.span),
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let mut result = self.operand_facts(body, facts, operand, &point.span);
                if let Some(place) = operand.place() {
                    let collection = place_key(place);
                    result.len_of = Some(collection.clone());
                    facts.lengths.insert(LengthFact {
                        result: destination_key.clone(),
                        collection,
                    });
                }
                result.value_flow.clear();
                result
            }
            Rvalue::Len(place) => {
                let collection = place_key(*place);
                let mut result = self.place_facts(body, facts, *place, point.span.clone());
                result.len_of = Some(collection.clone());
                result.value_flow.clear();
                facts.lengths.insert(LengthFact {
                    result: destination_key.clone(),
                    collection,
                });
                result
            }
            Rvalue::BinaryOp(op, operands) => {
                let left = self.call_operand(body, &operands.0);
                let right = self.call_operand(body, &operands.1);
                facts.compares.push(CompareFact {
                    result: destination_key.clone(),
                    op: *op,
                    left: left.clone(),
                    right: right.clone(),
                    point: point.clone(),
                });
                let mut result = self.join_fact_values([
                    self.call_operand_facts(facts, &left),
                    self.call_operand_facts(facts, &right),
                ]);
                result.value_flow.clear();
                result
            }
            Rvalue::Aggregate(kind, operands) => {
                let mut result = self.join_fact_values(
                    operands
                        .iter()
                        .map(|operand| self.operand_facts(body, facts, operand, &point.span)),
                );
                if matches!(&**kind, AggregateKind::Adt(did, ..) if self.tcx.item_name(*did).as_str() == "RangeTo")
                {
                    if let Some(end) = operands.iter().next().and_then(|operand| operand.place()) {
                        value_range_end(&mut result, place_key(end));
                    }
                }
                result
            }
            _ => ValueFacts::default(),
        };

        if let Some((public, def_path)) = self.field_info(body, destination) {
            value
                .value
                .origins
                .retain(|origin| !matches!(origin, AbstractOrigin::Formal(_)));
            value.value.origins.insert(if public {
                AbstractOrigin::PublicField {
                    def_path,
                    span: point.span.clone(),
                }
            } else {
                AbstractOrigin::PrivateField {
                    def_path,
                    span: point.span.clone(),
                }
            });
        }
        facts.values.insert(destination_key.clone(), value);
        if assignment_write {
            record_write(facts, destination_key, point, WriteKind::Assignment);
        }
    }

    fn extract_call(
        &self,
        caller: DefId,
        body: &Body<'tcx>,
        func: &Operand<'tcx>,
        args: &[rustc_span::source_map::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        point: ProgramPoint,
        facts: &mut BodyFacts,
    ) {
        let raw = func.const_fn_def();
        let raw_def = raw.map(|(did, _)| did);
        let callee =
            raw.and_then(|(did, generic_args)| self.resolve_local(caller, did, generic_args));
        let destination_key = place_key(destination);
        let operands = args
            .iter()
            .map(|argument| self.call_operand(body, &argument.node))
            .collect::<Vec<_>>();
        let exact_ffi_out = raw_def.is_some_and(|did| {
            self.is_exact_foreign_c(did)
                && args.iter().any(|argument| {
                    is_exact_ffi_out_ty(argument.node.ty(&body.local_decls, self.tcx))
                })
        });
        let selected_open_behavior = raw_def
            .is_some_and(|did| self.is_open_behavior_call(did, &callee, body, args, destination));
        let modeled_registry = raw_def.is_some_and(|did| {
            self.is_modeled_registry_call(did)
                || self.is_associated_slice_as_ref(did, body, args, destination)
        });
        let disposition = select_boundary(BoundaryInputs {
            resolved_local: callee.is_some(),
            exact_ffi_out,
            selected_open_behavior,
            modeled_registry,
            direct: raw_def.is_some(),
        });
        facts.calls.push(CallSite {
            callee: callee.clone(),
            raw_def,
            disposition,
            args: operands.clone(),
            destination: destination_key.clone(),
            point: point.clone(),
        });
        if let Some(ref callee) = callee {
            let mut mapping = BTreeSet::new();
            for (index, operand) in operands.iter().enumerate() {
                if let CallOperand::Place(actual) = operand {
                    mapping.insert(CallMapping::ActualToFormal {
                        actual: actual.clone(),
                        formal_index: index as u32 + 1,
                    });
                }
            }
            mapping.insert(CallMapping::ReturnToDestination {
                destination: destination_key.clone(),
            });
            facts.summary_seed.calls.insert(CallBoundary {
                caller: facts.function.clone(),
                callee: callee.clone(),
                point: point.clone(),
                mapping,
            });
        }

        if destination_write_invalidates(
            !destination.projection.is_empty(),
            facts.values.contains_key(&destination_key),
        ) {
            record_write(
                facts,
                destination_key.clone(),
                point.clone(),
                WriteKind::CallReturn,
            );
        }

        if boundary_mutation_write_kind(disposition) == Some(WriteKind::OpaqueMutable) {
            self.record_opaque_mutable_writes(body, args, &operands, &point, facts);
        }

        let Some((did, generic_args)) = raw else {
            facts.values.insert(destination_key, ValueFacts::default());
            return;
        };
        // A local callee's return provenance is supplied by its solved summary.
        // Starting from the actual arguments would merge unrelated values.
        let mut result = ValueFacts::default();

        if self.is_maybe_uninit_constructor(did, "uninit") {
            result.maybe_uninit_initialized = Some(false);
        } else if self.is_maybe_uninit_constructor(did, "new") {
            result.maybe_uninit_initialized = Some(true);
        } else if self.is_slice_len(did) && !operands.is_empty() {
            if let CallOperand::Place(collection) = &operands[0] {
                result = self.call_operand_facts(facts, &operands[0]);
                result.len_of = Some(collection.clone());
                result.value_flow.clear();
                facts.lengths.insert(LengthFact {
                    result: destination_key.clone(),
                    collection: collection.clone(),
                });
            }
        } else if disposition == BoundaryDisposition::SelectedOpenBehavior {
            result.open_behavior = true;
            result.value = AbstractValue::new([AbstractOrigin::OpenBehaviorOutput {
                token: format!("trait-return:{}", self.tcx.def_path_str(did)),
                span: point.span.clone(),
            }]);
        } else if self.is_associated_slice_as_ref(did, body, args, destination) {
            result.generic_capability = true;
            result
                .value
                .origins
                .insert(AbstractOrigin::GenericCapability {
                    token: format!("associated-slice:{}", self.tcx.def_path_str(did)),
                    span: point.span.clone(),
                });
        } else if disposition == BoundaryDisposition::ExactFfiOut {
            for (index, operand) in operands.iter().enumerate() {
                let CallOperand::Place(actual) = operand else {
                    continue;
                };
                let Some(argument) = args.get(index) else {
                    continue;
                };
                let argument_ty = argument.node.ty(&body.local_decls, self.tcx);
                if !is_exact_ffi_out_ty(argument_ty) {
                    continue;
                }
                let targets = self.alias_closure(facts, actual);
                for target in targets {
                    {
                        let entry = facts.values.entry(target.clone()).or_default();
                        entry.ffi_output = true;
                        entry.value.origins.insert(AbstractOrigin::FfiOutput {
                            token: format!("ffi-out:{}", point.span.token()),
                            span: point.span.clone(),
                        });
                    }
                    record_write(
                        facts,
                        target,
                        point.clone(),
                        boundary_mutation_write_kind(disposition)
                            .expect("exact FFI out boundary has a frozen write kind"),
                    );
                }
            }
        }

        if self.is_saturating_sub(did) && operands.len() >= 2 {
            result = self.join_fact_values(
                operands
                    .iter()
                    .map(|operand| self.call_operand_facts(facts, operand)),
            );
            result.value_flow.clear();
        } else if self.tcx.is_diagnostic_item(sym::mem_size_of, did) {
            if let Ok(layout) = self
                .tcx
                .layout_of(self.tcx.param_env(caller).and(generic_args.type_at(0)))
            {
                let width = layout.size.bytes();
                result.value.origins.insert(AbstractOrigin::Constant(width));
            }
        } else if self.is_slice_prefix_index(did) && operands.len() >= 2 {
            result = self.call_operand_facts(facts, &operands[0]);
            if let CallOperand::Place(range) = &operands[1] {
                result.range_end = self
                    .resolve_value(facts, range)
                    .range_end
                    .clone()
                    .or_else(|| Some(range.clone()));
            }
        } else if self.is_slice_as_ptr(did) && !operands.is_empty() {
            result = self.call_operand_facts(facts, &operands[0]);
            result.pointer_base = operands[0].place().cloned();
        } else if self.is_wrapping_add(did) && operands.len() >= 2 {
            result = self.call_operand_facts(facts, &operands[0]);
            result.pointer_base = result
                .pointer_base
                .clone()
                .or_else(|| operands[0].place().cloned());
            result.pointer_offset = operands[1].place().cloned();
            result
                .value
                .origins
                .extend(self.operand_value(facts, &operands[1]).origins);
        } else if self.is_pointer_cast(did) && !operands.is_empty() {
            result = self.call_operand_facts(facts, &operands[0]);
        }

        if self.tcx.is_diagnostic_item(sym::str_from_utf8, did) && !operands.is_empty() {
            if let CallOperand::Place(bytes) = &operands[0] {
                facts.predicates.push(PredicateFact::Utf8Result {
                    result: destination_key.clone(),
                    bytes: bytes.clone(),
                });
            }
        } else if self.is_result_is_err(did) && !operands.is_empty() {
            if let CallOperand::Place(checked) = &operands[0] {
                facts.predicates.push(PredicateFact::IsErr {
                    result: destination_key.clone(),
                    checked: checked.clone(),
                });
            }
        } else if self.is_slice_is_empty(did) && !operands.is_empty() {
            if let CallOperand::Place(slice) = &operands[0] {
                facts.predicates.push(PredicateFact::IsEmpty {
                    result: destination_key.clone(),
                    slice: slice.clone(),
                });
            }
        } else if self.is_pointer_is_null(did) && !operands.is_empty() {
            if let CallOperand::Place(pointer) = &operands[0] {
                facts.predicates.push(PredicateFact::IsNull {
                    result: destination_key.clone(),
                    pointer: pointer.clone(),
                });
            }
        }

        facts.values.insert(destination_key, result);
    }

    fn record_opaque_mutable_writes(
        &self,
        body: &Body<'tcx>,
        args: &[rustc_span::source_map::Spanned<Operand<'tcx>>],
        operands: &[CallOperand],
        point: &ProgramPoint,
        facts: &mut BodyFacts,
    ) {
        for (argument, operand) in args.iter().zip(operands) {
            if !is_mutable_call_actual(argument.node.ty(&body.local_decls, self.tcx)) {
                continue;
            }
            let CallOperand::Place(place) = operand else {
                continue;
            };
            record_write(
                facts,
                place.clone(),
                point.clone(),
                WriteKind::OpaqueMutable,
            );
        }
    }

    fn add_local_requirements(&self, body: &Body<'tcx>, facts: &mut BodyFacts) {
        if let Some(items) = self.preopt_transmutes.get(&facts.function) {
            for item in items {
                let origin = AbstractOrigin::InternalLocal {
                    function: facts.function.clone(),
                    place: PlaceKey::new("_0"),
                };
                let requirement = ContractRequirement {
                    seed_id: format!("P3.lifetime:{}", item.first_failure.point.span.token()),
                    collection: None,
                    subject: AbstractValue::new([origin.clone()]),
                    source_hint: Some(SourceKind::InternalUnsafeOrigin),
                    internal_derivation: false,
                    source_span: item.first_failure.point.span.clone(),
                    first_failure: item.first_failure.clone(),
                    sink: item.sink.clone(),
                    predicates: BTreeSet::from([Predicate::ReferentOutlivesReference]),
                    access_width: None,
                    rule: RuleId::P3LifetimeTransmute,
                    return_exposure: true,
                    sink_function: facts.function.clone(),
                };
                facts.summary_seed.requirements.insert(requirement);
                facts.summary_seed.return_value.origins.insert(origin);
            }
        }

        for call in facts.calls.clone() {
            let Some(did) = call.raw_def else {
                continue;
            };
            if self.is_get_unchecked(did) && call.args.len() >= 2 {
                let collection = self.operand_value(facts, &call.args[0]);
                let index_subject = self.operand_value(facts, &call.args[1]);
                let constant_zero = matches!(call.args[1], CallOperand::Constant(0));
                let p5_subject = generic_nonempty_subject(&collection, constant_zero);
                let source_hint = if p5_subject.is_some() {
                    Some(SourceKind::GenericNonEmptyCapability)
                } else if index_subject
                    .origins
                    .iter()
                    .any(|origin| matches!(origin, AbstractOrigin::OpenBehaviorOutput { .. }))
                {
                    Some(SourceKind::OpenBehaviorOutput)
                } else {
                    None
                };
                let subject = p5_subject.unwrap_or(index_subject);
                let internal_derivation = source_hint.is_none()
                    && !subject.origins.is_empty()
                    && subject.origins.iter().all(|origin| {
                        matches!(
                            origin,
                            AbstractOrigin::PrivateField { .. }
                                | AbstractOrigin::InternalLocal { .. }
                        )
                    });
                let rule = match source_hint {
                    Some(SourceKind::GenericNonEmptyCapability) => RuleId::P51Nonempty,
                    Some(SourceKind::OpenBehaviorOutput) => RuleId::P6OpenTraitIndex,
                    _ => RuleId::P1GetUnchecked,
                };
                facts.summary_seed.requirements.insert(ContractRequirement {
                    seed_id: format!("{}:{}", rule.as_str(), call.point.span.token()),
                    collection: Some(collection),
                    subject,
                    source_hint,
                    internal_derivation,
                    source_span: call.point.span.clone(),
                    first_failure: Operation {
                        kind: OperationKind::GetUnchecked,
                        point: call.point.clone(),
                    },
                    sink: Operation {
                        kind: OperationKind::GetUnchecked,
                        point: call.point.clone(),
                    },
                    predicates: BTreeSet::from([if constant_zero {
                        Predicate::NonEmpty
                    } else {
                        Predicate::InBounds
                    }]),
                    access_width: None,
                    rule,
                    return_exposure: false,
                    sink_function: facts.function.clone(),
                });
            } else if self.tcx.is_diagnostic_item(sym::assume_init, did) && !call.args.is_empty() {
                let receiver = call.args[0].place().cloned();
                let initialization = receiver
                    .as_ref()
                    .and_then(|place| self.resolve_value(facts, place).maybe_uninit_initialized);
                let destination_ty = self.place_ty(body, &call.destination);
                if initialization == Some(false)
                    && destination_ty.is_some_and(|ty| ty.is_bool())
                    && self.place_reaches_return(facts, &call.destination)
                {
                    let origin = AbstractOrigin::InternalLocal {
                        function: facts.function.clone(),
                        place: call.destination.clone(),
                    };
                    facts.summary_seed.requirements.insert(ContractRequirement {
                        seed_id: format!("P3.assume_init:{}", call.point.span.token()),
                        collection: None,
                        subject: AbstractValue::new([origin.clone()]),
                        source_hint: Some(SourceKind::InternalUnsafeOrigin),
                        internal_derivation: false,
                        source_span: call.point.span.clone(),
                        first_failure: Operation {
                            kind: OperationKind::AssumeInitBool,
                            point: call.point.clone(),
                        },
                        sink: Operation {
                            kind: OperationKind::InvalidValueExposure,
                            point: call.point.clone(),
                        },
                        predicates: BTreeSet::from([Predicate::Initialized, Predicate::ValidBool]),
                        access_width: None,
                        rule: RuleId::P3AssumeInitBool,
                        return_exposure: true,
                        sink_function: facts.function.clone(),
                    });
                    facts.summary_seed.return_value.origins.insert(origin);
                }
            } else if self
                .tcx
                .is_diagnostic_item(sym::str_from_utf8_unchecked, did)
                && !call.args.is_empty()
            {
                if !self.utf8_discharged(body, facts, &call.args[0], call.point.point_location()) {
                    let origin = AbstractOrigin::InternalLocal {
                        function: facts.function.clone(),
                        place: call.destination.clone(),
                    };
                    facts.summary_seed.requirements.insert(ContractRequirement {
                        seed_id: format!("P3.utf8:{}", call.point.span.token()),
                        collection: None,
                        subject: AbstractValue::new([origin.clone()]),
                        source_hint: Some(SourceKind::InternalUnsafeOrigin),
                        internal_derivation: false,
                        source_span: call.point.span.clone(),
                        first_failure: Operation {
                            kind: OperationKind::FromUtf8Unchecked,
                            point: call.point.clone(),
                        },
                        sink: Operation {
                            kind: OperationKind::InvalidValueExposure,
                            point: call.point.clone(),
                        },
                        predicates: BTreeSet::from([Predicate::ValidUtf8]),
                        access_width: None,
                        rule: RuleId::P3UncheckedUtf8,
                        return_exposure: true,
                        sink_function: facts.function.clone(),
                    });
                    if self.place_reaches_return(facts, &call.destination) {
                        facts.summary_seed.return_value.origins.insert(origin);
                    }
                }
            } else if self.is_nonnull_new_unchecked(did) && !call.args.is_empty() {
                let subject = self.operand_value(facts, &call.args[0]);
                if subject
                    .origins
                    .iter()
                    .any(|origin| matches!(origin, AbstractOrigin::FfiOutput { .. }))
                {
                    facts.summary_seed.requirements.insert(ContractRequirement {
                        seed_id: format!("P6.ffi:{}", call.point.span.token()),
                        collection: None,
                        subject,
                        source_hint: Some(SourceKind::FfiOutput),
                        internal_derivation: false,
                        source_span: call.point.span.clone(),
                        first_failure: Operation {
                            kind: OperationKind::NonNullNewUnchecked,
                            point: call.point.clone(),
                        },
                        sink: Operation {
                            kind: OperationKind::NonNullNewUnchecked,
                            point: call.point.clone(),
                        },
                        predicates: BTreeSet::from([Predicate::NonNull]),
                        access_width: None,
                        rule: RuleId::P6FfiOutParam,
                        return_exposure: false,
                        sink_function: facts.function.clone(),
                    });
                }
            } else if self.is_read_unaligned(did) && !call.args.is_empty() {
                let pointer = self.operand_value(facts, &call.args[0]);
                if internal_only_origins(&pointer, facts.has_self) {
                    facts.summary_seed.requirements.insert(ContractRequirement {
                        seed_id: format!("P4.offset:{}", call.point.span.token()),
                        collection: None,
                        subject: pointer,
                        source_hint: Some(SourceKind::InternalDerived),
                        internal_derivation: true,
                        source_span: call.point.span.clone(),
                        first_failure: Operation {
                            kind: OperationKind::ReadUnaligned,
                            point: call.point.clone(),
                        },
                        sink: Operation {
                            kind: OperationKind::ReadUnaligned,
                            point: call.point.clone(),
                        },
                        predicates: BTreeSet::from([Predicate::RangeInBounds]),
                        access_width: self
                            .place_ty(body, &call.destination)
                            .and_then(|ty| {
                                self.tcx
                                    .layout_of(self.tcx.param_env(facts.def_id).and(ty))
                                    .ok()
                            })
                            .map(|layout| layout.size.bytes()),
                        rule: RuleId::P4Offset,
                        return_exposure: false,
                        sink_function: facts.function.clone(),
                    });
                }
            }
        }

        for (bb, data) in body.basic_blocks.iter_enumerated() {
            for (statement_index, statement) in data.statements.iter().enumerate() {
                let StatementKind::Assign(box (_, Rvalue::Use(operand))) = &statement.kind else {
                    continue;
                };
                let Some(place) = operand.place() else {
                    continue;
                };
                if !place
                    .projection
                    .iter()
                    .any(|element| matches!(element, ProjectionElem::Deref))
                    || !matches!(body.local_decls[place.local].ty.kind(), ty::RawPtr(..))
                {
                    continue;
                }
                let source_place = PlaceKey::new(format!("_{}", place.local.index()));
                let point = self.point(
                    &facts.function,
                    Location {
                        block: bb,
                        statement_index,
                    },
                    statement.source_info.span,
                );
                facts.summary_seed.requirements.insert(ContractRequirement {
                    seed_id: format!("P1.raw:{}", point.span.token()),
                    collection: None,
                    subject: self.resolve_value(facts, &source_place).value.clone(),
                    source_hint: None,
                    internal_derivation: false,
                    source_span: point.span.clone(),
                    first_failure: Operation {
                        kind: OperationKind::RawRead,
                        point: point.clone(),
                    },
                    sink: Operation {
                        kind: OperationKind::RawRead,
                        point,
                    },
                    predicates: BTreeSet::from([Predicate::ValidForRead]),
                    access_width: None,
                    rule: RuleId::P1RawRead,
                    return_exposure: false,
                    sink_function: facts.function.clone(),
                });
            }
        }
    }

    fn finding(
        &self,
        root: &FunctionKey,
        facts: &BodyFacts,
        selected: SelectedRequirement,
    ) -> Finding {
        let failure = if selected.requirement.sink.kind == OperationKind::InvalidValueExposure {
            FailureClass::InternalInvalidValue
        } else {
            FailureClass::SinkPrecondition
        };
        let mut all_sources = source_kinds(facts, &selected.requirement);
        all_sources.insert(selected.source.kind);
        let (primary, secondary_sources) =
            classify_primary_with_secondary(failure, selected.source.kind, all_sources);
        let rule = adjusted_rule(selected.requirement.rule, primary);
        let obligation = Obligation::new(
            selected.requirement.predicates.iter().copied(),
            selected.source.origin.clone(),
        );
        Finding::new(
            self.crate_name.clone(),
            root.clone(),
            facts.root_span.clone(),
            SinkObligation {
                source: selected.source,
                first_failure: selected.requirement.first_failure,
                sink: selected.requirement.sink,
                obligation,
            },
            primary,
            rule,
            selected.witness,
            secondary_sources,
        )
    }

    fn is_selected_open_behavior(&self, did: DefId) -> bool {
        self.tcx
            .trait_of_item(did)
            .is_some_and(|trait_id| self.tcx.item_name(trait_id).as_str() == "IndexSource")
            && self
                .tcx
                .opt_item_name(did)
                .is_some_and(|name| name.as_str() == "index")
    }

    fn classify_source(
        &self,
        root: &BodyFacts,
        requirement: &ContractRequirement,
    ) -> Option<Source> {
        if let Some(kind) = requirement.source_hint {
            if kind == SourceKind::GenericNonEmptyCapability {
                let collection = requirement.collection.as_ref()?;
                let origin = matching_generic_capability(collection, &requirement.subject)?.clone();
                return Some(Source {
                    kind,
                    origin: OriginKey::new(origin_token(&origin)),
                    span: origin_span(&origin).unwrap_or_else(|| requirement.source_span.clone()),
                });
            }
            let origin = requirement
                .subject
                .origins
                .iter()
                .find(|origin| origin_matches_kind(origin, kind))
                .cloned()?;
            return Some(Source {
                kind,
                origin: OriginKey::new(origin_token(&origin)),
                span: origin_span(&origin).unwrap_or_else(|| requirement.source_span.clone()),
            });
        }

        if let Some(origin) = requirement
            .subject
            .origins
            .iter()
            .find(|origin| matches!(origin, AbstractOrigin::PublicField { .. }))
        {
            return Some(Source {
                kind: SourceKind::LiteralPublicField,
                origin: OriginKey::new(origin_token(origin)),
                span: origin_span(origin).unwrap_or_else(|| requirement.source_span.clone()),
            });
        }
        let selected_internal_derivation = requirement.internal_derivation
            || (requirement.rule == RuleId::P1GetUnchecked
                && internal_only_origins(&requirement.subject, root.has_self));
        if selected_internal_derivation {
            let origin = requirement.subject.origins.iter().find(|origin| {
                matches!(
                    origin,
                    AbstractOrigin::PrivateField { .. } | AbstractOrigin::InternalLocal { .. }
                )
            })?;
            return Some(Source {
                kind: SourceKind::InternalDerived,
                origin: OriginKey::new(origin_token(origin)),
                span: origin_span(origin).unwrap_or_else(|| requirement.source_span.clone()),
            });
        }
        if let Some(AbstractOrigin::Formal(index)) = requirement
            .subject
            .origins
            .iter()
            .find(|origin| {
                matches!(origin, AbstractOrigin::Formal(index)
                    if *index > 0
                        && (*index as usize) <= root.arg_count
                        && !(root.has_self && *index == 1))
            })
        {
            return Some(Source {
                kind: SourceKind::PublicParameter,
                origin: OriginKey::new(format!("arg:{index}")),
                span: root.root_span.clone(),
            });
        }
        None
    }

    fn is_discharged(
        &self,
        root: &BodyFacts,
        requirement: &ContractRequirement,
        paths: &BTreeMap<FunctionKey, Vec<CallSite>>,
    ) -> bool {
        if requirement.sink_function == root.function {
            return self.local_validation(
                root,
                requirement,
                requirement.sink.point.point_location(),
            );
        }
        let Some(path) = paths.get(&requirement.sink_function) else {
            return false;
        };
        let Some(first_call) = path.first() else {
            return false;
        };
        self.local_validation(root, requirement, first_call.point.point_location())
    }

    fn local_validation(
        &self,
        facts: &BodyFacts,
        requirement: &ContractRequirement,
        sink: Location,
    ) -> bool {
        let body = self.tcx.optimized_mir(facts.def_id.to_def_id());
        let predicate = requirement.predicates.iter().next().copied();
        let subject_places = self.places_for_value(facts, &requirement.subject);
        let collection_places = requirement
            .collection
            .as_ref()
            .map(|value| self.places_for_value(facts, value))
            .unwrap_or_default();
        match predicate {
            Some(Predicate::InBounds) => self.bounds_validation(
                body,
                facts,
                &subject_places,
                &collection_places,
                sink,
            ),
            Some(Predicate::NonEmpty) => self.predicate_validation(
                body,
                facts,
                &collection_places,
                sink,
                |fact, candidate| {
                    matches!(fact, PredicateFact::IsEmpty { slice, .. } if candidate.contains(slice))
                },
            ),
            Some(Predicate::NonNull) => self.predicate_validation(
                body,
                facts,
                &subject_places,
                sink,
                |fact, candidate| {
                    matches!(fact, PredicateFact::IsNull { pointer, .. } if candidate.contains(pointer))
                },
            ),
            Some(Predicate::RangeInBounds) => {
                self.range_validation(body, facts, &subject_places, requirement.access_width, sink)
            }
            _ => false,
        }
    }

    fn bounds_validation(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        subjects: &BTreeSet<PlaceKey>,
        collections: &BTreeSet<PlaceKey>,
        sink: Location,
    ) -> bool {
        for compare in &facts.compares {
            if !is_in_bounds_guard(compare.op) {
                continue;
            }
            let left = self.operand_places(facts, &compare.left);
            if !same_binding(subjects, &left) {
                continue;
            }
            let right = self.operand_places(facts, &compare.right);
            let right_has_len = facts.lengths.iter().any(|length| {
                same_binding(&right, &self.alias_closure(facts, &length.result))
                    && same_binding(collections, &self.alias_closure(facts, &length.collection))
            });
            if !right_has_len {
                continue;
            }
            if let Some(region) = self.validation_branch(
                body,
                facts,
                &compare.result,
                compare.point.point_location(),
                sink,
            ) {
                let mut watched = subjects.clone();
                watched.extend(collections.iter().cloned());
                if !self.validation_invalidated(
                    body,
                    facts,
                    &watched,
                    compare.point.point_location(),
                    region,
                    sink,
                ) {
                    return true;
                }
            }
        }
        false
    }

    fn predicate_validation<F>(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        candidates: &BTreeSet<PlaceKey>,
        sink: Location,
        matches_fact: F,
    ) -> bool
    where
        F: Fn(&PredicateFact, &BTreeSet<PlaceKey>) -> bool,
    {
        for predicate in &facts.predicates {
            if !matches_fact(predicate, candidates) {
                continue;
            }
            let result = predicate_result(predicate);
            let Some(call) = facts.calls.iter().find(|call| call.destination == *result) else {
                continue;
            };
            if let Some(region) =
                self.validation_branch(body, facts, result, call.point.point_location(), sink)
            {
                if !self.validation_invalidated(
                    body,
                    facts,
                    candidates,
                    call.point.point_location(),
                    region,
                    sink,
                ) {
                    return true;
                }
            }
        }
        false
    }

    fn range_validation(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        pointer_places: &BTreeSet<PlaceKey>,
        access_width: Option<u64>,
        sink: Location,
    ) -> bool {
        let Some(width) = access_width else {
            return false;
        };
        let offsets = pointer_places
            .iter()
            .filter_map(|place| self.resolve_value(facts, place).pointer_offset.clone())
            .flat_map(|place| self.alias_closure(facts, &place))
            .collect::<BTreeSet<_>>();
        if offsets.is_empty() {
            return false;
        }
        for compare in &facts.compares {
            if compare.op != BinOp::Gt
                || offsets.is_disjoint(&self.operand_places(facts, &compare.left))
            {
                continue;
            }
            let right_places = self.operand_places(facts, &compare.right);
            let right_matches = right_places.iter().any(|place| {
                let value = self.resolve_value(facts, place);
                value.dependencies.iter().any(|dependency| {
                    matches!(self.resolve_value(facts, dependency).len_of, Some(_))
                }) && value.dependencies.iter().any(|dependency| {
                    self.resolve_value(facts, dependency)
                        .value
                        .origins
                        .contains(&AbstractOrigin::Constant(width))
                })
            });
            if right_matches {
                if let Some(region) = self.validation_branch(
                    body,
                    facts,
                    &compare.result,
                    compare.point.point_location(),
                    sink,
                ) {
                    let watched = self.range_watched_places(facts, pointer_places, &offsets);
                    if !self.validation_invalidated(
                        body,
                        facts,
                        &watched,
                        compare.point.point_location(),
                        region,
                        sink,
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn utf8_discharged(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        bytes: &CallOperand,
        sink: Location,
    ) -> bool {
        let candidates = self.operand_places(facts, bytes);
        for predicate in &facts.predicates {
            let PredicateFact::IsErr { result, checked } = predicate else {
                continue;
            };
            let checked_aliases = self.alias_closure(facts, checked);
            let same_bytes = facts.predicates.iter().any(|origin| {
                matches!(origin, PredicateFact::Utf8Result { result: checked_result, bytes }
                    if checked_aliases.contains(checked_result)
                        && !candidates.is_disjoint(&self.alias_closure(facts, bytes)))
            });
            if !same_bytes {
                continue;
            }
            let Some(call) = facts.calls.iter().find(|call| call.destination == *result) else {
                continue;
            };
            if let Some(region) =
                self.validation_branch(body, facts, result, call.point.point_location(), sink)
            {
                if !self.validation_invalidated(
                    body,
                    facts,
                    &candidates,
                    call.point.point_location(),
                    region,
                    sink,
                ) {
                    return true;
                }
            }
        }
        false
    }

    fn validation_branch(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        result: &PlaceKey,
        established: Location,
        sink: Location,
    ) -> Option<ValidationRegion> {
        let dominators = body.basic_blocks.dominators();
        facts.branches.iter().find_map(|branch| {
            validation_success_decision(
                branch.result == *result,
                branch.success == sink.block || dominators.dominates(branch.success, sink.block),
                location_precedes(established, branch.point.point_location(), body),
            )
            .then_some(ValidationRegion {
                branch: branch.point.point_location(),
                success: branch.success,
            })
        })
    }

    fn validation_invalidated(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        candidates: &BTreeSet<PlaceKey>,
        established: Location,
        region: ValidationRegion,
        sink: Location,
    ) -> bool {
        let successors = cfg_successors(body);
        facts.writes.iter().any(|write| {
            related_to_watched(facts, candidates, &write.place)
                && (write_is_between_points(
                    &successors,
                    CfgPoint {
                        block: established.block.index() as u32,
                        statement: established.statement_index as u32,
                    },
                    cfg_point(&write.point),
                    CfgPoint {
                        block: region.branch.block.index() as u32,
                        statement: region.branch.statement_index as u32,
                    },
                ) || write_is_between(
                    &successors,
                    region.success.index() as u32,
                    cfg_point(&write.point),
                    CfgPoint {
                        block: sink.block.index() as u32,
                        statement: sink.statement_index as u32,
                    },
                ))
        })
    }

    fn range_watched_places(
        &self,
        facts: &BodyFacts,
        pointers: &BTreeSet<PlaceKey>,
        offsets: &BTreeSet<PlaceKey>,
    ) -> BTreeSet<PlaceKey> {
        let mut watched = pointers.clone();
        watched.extend(offsets.iter().cloned());
        for pointer in pointers {
            let value = self.resolve_value(facts, pointer);
            watched.extend(value.dependencies.iter().cloned());
            watched.extend(value.pointer_base.iter().cloned());
            watched.extend(value.pointer_offset.iter().cloned());
        }
        watched
    }

    fn shortest_paths(
        &self,
        root: &FunctionKey,
        graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
        _recursive: &BTreeMap<FunctionKey, String>,
    ) -> BTreeMap<FunctionKey, Vec<CallSite>> {
        let mut paths: BTreeMap<FunctionKey, Vec<CallSite>> =
            BTreeMap::from([(root.clone(), Vec::new())]);
        loop {
            let snapshot = paths.clone();
            let mut changed = false;
            for (node, path) in snapshot {
                let Some(facts) = self.bodies.get(&node) else {
                    continue;
                };
                for call in &facts.calls {
                    let Some(callee) = &call.callee else {
                        continue;
                    };
                    if path.iter().any(|step| step.callee.as_ref() == Some(callee)) {
                        continue;
                    }
                    let mut candidate = path.clone();
                    candidate.push(call.clone());
                    let prefer = paths.get(callee).is_none_or(|current| {
                        candidate.len() < current.len()
                            || (candidate.len() == current.len()
                                && path_token(&candidate) < path_token(current))
                    });
                    if prefer {
                        paths.insert(callee.clone(), candidate);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let _ = graph;
        paths
    }

    fn build_witness(
        &self,
        root: &BodyFacts,
        requirement: &ContractRequirement,
        paths: &BTreeMap<FunctionKey, Vec<CallSite>>,
        recursive: &BTreeMap<FunctionKey, String>,
    ) -> CanonicalWitness {
        let mut steps = vec![WitnessStep {
            kind: WitnessStepKind::Entry,
            function: root.function.clone(),
            span: root.root_span.clone(),
        }];
        let path = paths
            .get(&requirement.sink_function)
            .cloned()
            .unwrap_or_default();
        let mut boundaries = Vec::new();
        for call in &path {
            let callee = call
                .callee
                .clone()
                .unwrap_or_else(|| requirement.sink_function.clone());
            boundaries.push(callee.clone());
            steps.push(WitnessStep {
                kind: WitnessStepKind::LocalCall,
                function: callee,
                span: call.point.span.clone(),
            });
        }
        if let Some(cycle_token) = recursive.get(&requirement.sink_function) {
            steps.push(WitnessStep {
                kind: WitnessStepKind::SccCycle,
                function: cycle_step_function(cycle_token),
                span: requirement.sink.point.span.clone(),
            });
        }
        steps.push(WitnessStep {
            kind: if requirement.sink.kind == OperationKind::InvalidValueExposure {
                WitnessStepKind::Exposure
            } else {
                WitnessStepKind::Sink
            },
            function: requirement.sink_function.clone(),
            span: requirement.sink.point.span.clone(),
        });
        CanonicalWitness::new(steps, boundaries)
    }

    fn call_graph(&self) -> BTreeMap<FunctionKey, BTreeSet<FunctionKey>> {
        self.bodies
            .iter()
            .map(|(function, facts)| {
                (
                    function.clone(),
                    facts
                        .calls
                        .iter()
                        .filter(|call| call.disposition == BoundaryDisposition::ResolvedLocal)
                        .filter_map(|call| call.callee.clone())
                        .collect(),
                )
            })
            .collect()
    }

    fn operand_facts(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        operand: &Operand<'tcx>,
        span: &StableSpan,
    ) -> ValueFacts {
        if let Some(place) = operand.place() {
            return self.place_facts(body, facts, place, span.clone());
        }
        self.usize_constant(body, operand)
            .map(|value| ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Constant(value)]),
                ..ValueFacts::default()
            })
            .unwrap_or_default()
    }

    fn place_facts(
        &self,
        body: &Body<'tcx>,
        facts: &BodyFacts,
        place: Place<'tcx>,
        span: StableSpan,
    ) -> ValueFacts {
        let key = place_key(place);
        let mut result = facts.values.get(&key).cloned().unwrap_or_else(|| {
            facts
                .values
                .get(&PlaceKey::new(format!("_{}", place.local.index())))
                .cloned()
                .unwrap_or_default()
        });
        result.dependencies.insert(key);
        result.value_flow.insert(place_key(place));
        if let Some((public, def_path)) = self.field_info(body, place) {
            result.value.origins.insert(if public {
                AbstractOrigin::PublicField { def_path, span }
            } else {
                AbstractOrigin::PrivateField { def_path, span }
            });
        }
        result
    }

    fn resolve_value<'a>(&self, facts: &'a BodyFacts, place: &PlaceKey) -> &'a ValueFacts {
        facts.values.get(place).unwrap_or_else(|| {
            static EMPTY: std::sync::OnceLock<ValueFacts> = std::sync::OnceLock::new();
            EMPTY.get_or_init(ValueFacts::default)
        })
    }

    fn operand_value(&self, facts: &BodyFacts, operand: &CallOperand) -> AbstractValue {
        match operand {
            CallOperand::Place(place) => self.resolve_value(facts, place).value.clone(),
            CallOperand::Constant(value) => AbstractValue::new([AbstractOrigin::Constant(*value)]),
            CallOperand::Unknown => AbstractValue::default(),
        }
    }

    fn operand_places(&self, facts: &BodyFacts, operand: &CallOperand) -> BTreeSet<PlaceKey> {
        match operand {
            CallOperand::Place(place) => self.alias_closure(facts, place),
            _ => BTreeSet::new(),
        }
    }

    fn places_for_value(&self, facts: &BodyFacts, value: &AbstractValue) -> BTreeSet<PlaceKey> {
        let mut places = BTreeSet::new();
        for origin in &value.origins {
            match origin {
                AbstractOrigin::Formal(index) if *index > 0 => {
                    places.insert(PlaceKey::new(format!("_{index}")));
                }
                AbstractOrigin::Constant(_) => {}
                _ => {
                    places.extend(
                        facts
                            .values
                            .iter()
                            .filter(|(_, facts)| facts.value.origins.contains(origin))
                            .map(|(place, _)| place.clone()),
                    );
                }
            }
        }
        places
    }

    fn alias_closure(&self, facts: &BodyFacts, seed: &PlaceKey) -> BTreeSet<PlaceKey> {
        let mut result = BTreeSet::from([seed.clone()]);
        loop {
            let before = result.len();
            for (place, value) in &facts.values {
                if !value.value_flow.is_disjoint(&result) || result.contains(place) {
                    result.insert(place.clone());
                    result.extend(value.value_flow.iter().cloned());
                }
            }
            if result.len() == before {
                break;
            }
        }
        result
    }

    fn place_reaches_return(&self, facts: &BodyFacts, place: &PlaceKey) -> bool {
        place.0 == "_0"
            || self
                .alias_closure(facts, place)
                .contains(&PlaceKey::new("_0"))
    }

    fn call_operand_facts(&self, facts: &BodyFacts, operand: &CallOperand) -> ValueFacts {
        match operand {
            CallOperand::Place(place) => {
                let mut result = self.resolve_value(facts, place).clone();
                result.dependencies.insert(place.clone());
                result.value_flow.insert(place.clone());
                result
            }
            CallOperand::Constant(value) => ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Constant(*value)]),
                ..ValueFacts::default()
            },
            CallOperand::Unknown => ValueFacts::default(),
        }
    }

    fn join_fact_values(&self, values: impl IntoIterator<Item = ValueFacts>) -> ValueFacts {
        let mut result = ValueFacts::default();
        for value in values {
            result.value.origins.extend(value.value.origins);
            result.dependencies.extend(value.dependencies);
            result.value_flow.extend(value.value_flow);
            result.len_of = result.len_of.or(value.len_of);
            result.range_end = result.range_end.or(value.range_end);
            result.pointer_base = result.pointer_base.or(value.pointer_base);
            result.pointer_offset = result.pointer_offset.or(value.pointer_offset);
            result.maybe_uninit_initialized = match (
                result.maybe_uninit_initialized,
                value.maybe_uninit_initialized,
            ) {
                (None, other) => other,
                (some, None) => some,
                (Some(left), Some(right)) if left == right => Some(left),
                _ => None,
            };
            result.generic_capability |= value.generic_capability;
            result.ffi_output |= value.ffi_output;
            result.open_behavior |= value.open_behavior;
        }
        result
    }

    fn call_operand(&self, body: &Body<'tcx>, operand: &Operand<'tcx>) -> CallOperand {
        operand
            .place()
            .map(|place| CallOperand::Place(place_key(place)))
            .or_else(|| {
                self.usize_constant(body, operand)
                    .map(CallOperand::Constant)
            })
            .unwrap_or(CallOperand::Unknown)
    }

    fn usize_constant(&self, body: &Body<'tcx>, operand: &Operand<'tcx>) -> Option<u64> {
        if !matches!(
            operand.ty(&body.local_decls, self.tcx).kind(),
            ty::Uint(ty::UintTy::Usize)
        ) {
            return None;
        }
        operand
            .constant()?
            .const_
            .try_eval_target_usize(self.tcx, ty::ParamEnv::empty())
    }

    fn field_info(&self, body: &Body<'tcx>, place: Place<'tcx>) -> Option<(bool, String)> {
        let mut place_ty =
            rustc_middle::mir::tcx::PlaceTy::from_ty(body.local_decls[place.local].ty);
        for element in place.projection.iter() {
            if let ProjectionElem::Field(index, _) = element {
                if let ty::Adt(adt, _) = place_ty.ty.kind() {
                    let variant = place_ty
                        .variant_index
                        .map(|variant| adt.variant(variant))
                        .unwrap_or_else(|| adt.non_enum_variant());
                    let field = &variant.fields[index];
                    let parent_public = adt.did().is_local()
                        && self
                            .tcx
                            .effective_visibilities(())
                            .is_public_at_level(adt.did().expect_local(), Level::Reachable);
                    return Some((
                        field.vis.is_public() && parent_public,
                        self.tcx.def_path_str(field.did),
                    ));
                }
            }
            place_ty = place_ty.projection_ty(self.tcx, element);
        }
        None
    }

    fn place_ty(&self, body: &Body<'tcx>, place: &PlaceKey) -> Option<ty::Ty<'tcx>> {
        let local = parse_local(place)?;
        body.local_decls
            .get(rustc_middle::mir::Local::from_usize(local))
            .map(|decl| decl.ty)
    }

    fn resolve_local(
        &self,
        caller: DefId,
        raw: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Option<FunctionKey> {
        if let Ok(Some(instance)) =
            Instance::try_resolve(self.tcx, self.tcx.param_env(caller), raw, args)
        {
            if let InstanceKind::Item(target) = instance.def {
                if target.is_local() && self.tcx.is_mir_available(target) {
                    return Some(FunctionKey::new(self.tcx.def_path_str(target)));
                }
            }
        }
        None
    }

    fn is_public_safe_root(&self, id: LocalDefId) -> bool {
        let did = id.to_def_id();
        self.tcx
            .effective_visibilities(())
            .is_public_at_level(id, Level::Reachable)
            && self.tcx.fn_sig(did).skip_binder().safety() == Safety::Safe
    }

    fn is_exact_foreign_c(&self, did: DefId) -> bool {
        self.tcx.is_foreign_item(did)
            && matches!(
                self.tcx.fn_sig(did).skip_binder().abi(),
                Abi::C { unwind: false }
            )
    }

    fn is_modeled_registry_call(&self, did: DefId) -> bool {
        self.is_get_unchecked(did)
            || self.is_nonnull_new_unchecked(did)
            || self.is_maybe_uninit_constructor(did, "uninit")
            || self.is_maybe_uninit_constructor(did, "new")
            || self.is_slice_len(did)
            || self.tcx.is_diagnostic_item(sym::assume_init, did)
            || self.tcx.is_diagnostic_item(sym::str_from_utf8, did)
            || self
                .tcx
                .is_diagnostic_item(sym::str_from_utf8_unchecked, did)
            || self.tcx.is_diagnostic_item(sym::mem_size_of, did)
            || self.is_slice_is_empty(did)
            || self.is_pointer_is_null(did)
            || self.is_result_is_err(did)
            || self.is_saturating_sub(did)
            || self.is_slice_prefix_index(did)
            || self.is_slice_as_ptr(did)
            || self.is_wrapping_add(did)
            || self.is_pointer_cast(did)
            || self.is_read_unaligned(did)
    }

    fn is_get_unchecked(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| matches!(name.as_str(), "get_unchecked" | "get_unchecked_mut"))
            && self
                .tcx
                .impl_of_method(did)
                .is_some_and(|impl_id| self.tcx.type_of(impl_id).skip_binder().is_slice())
    }

    fn is_nonnull_new_unchecked(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "new_unchecked")
            && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::NonNull, adt.did()))
            })
    }

    fn is_maybe_uninit_constructor(&self, did: DefId, expected: &str) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == expected)
            && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_lang_item(adt.did(), LangItem::MaybeUninit))
            })
    }

    fn is_slice_is_empty(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "is_empty")
            && self
                .tcx
                .impl_of_method(did)
                .is_some_and(|impl_id| self.tcx.type_of(impl_id).skip_binder().is_slice())
    }

    fn is_slice_len(&self, did: DefId) -> bool {
        self.tcx.lang_items().slice_len_fn() == Some(did)
    }

    fn is_pointer_is_null(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "is_null")
            && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::RawPtr(..)
                )
            })
    }

    fn is_result_is_err(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "is_err")
            && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::Result, adt.did()))
            })
    }

    fn is_saturating_sub(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "saturating_sub")
            && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::Uint(ty::UintTy::Usize)
                )
            })
    }

    fn is_slice_prefix_index(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "index")
            && self
                .tcx
                .trait_of_item(did)
                .is_some_and(|trait_id| self.tcx.item_name(trait_id).as_str() == "Index")
    }

    fn is_slice_as_ptr(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "as_ptr")
            && self
                .tcx
                .impl_of_method(did)
                .is_some_and(|impl_id| self.tcx.type_of(impl_id).skip_binder().is_slice())
    }

    fn is_wrapping_add(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "wrapping_add")
            && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::RawPtr(..)
                )
            })
    }

    fn is_pointer_cast(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "cast")
            && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::RawPtr(..)
                )
            })
    }

    fn is_read_unaligned(&self, did: DefId) -> bool {
        self.tcx.is_diagnostic_item(sym::ptr_read_unaligned, did)
            || (self
                .tcx
                .opt_item_name(did)
                .is_some_and(|name| name.as_str() == "read_unaligned")
                && self.tcx.impl_of_method(did).is_some_and(|impl_id| {
                    matches!(
                        self.tcx.type_of(impl_id).skip_binder().kind(),
                        ty::RawPtr(..)
                    )
                }))
    }

    fn is_associated_slice_as_ref(
        &self,
        did: DefId,
        body: &Body<'tcx>,
        args: &[rustc_span::source_map::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
    ) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "as_ref")
            && self
                .tcx
                .trait_of_item(did)
                .is_some_and(|trait_id| self.tcx.item_name(trait_id).as_str() == "AsRef")
            && destination
                .ty(&body.local_decls, self.tcx)
                .ty
                .is_array_slice()
            && args.first().is_some_and(|argument| {
                argument
                    .node
                    .ty(&body.local_decls, self.tcx)
                    .has_non_region_param()
            })
    }

    fn is_open_behavior_call(
        &self,
        did: DefId,
        callee: &Option<FunctionKey>,
        body: &Body<'tcx>,
        args: &[rustc_span::source_map::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
    ) -> bool {
        callee.is_none()
            && self.is_selected_open_behavior(did)
            && args.first().is_some_and(|argument| {
                argument
                    .node
                    .ty(&body.local_decls, self.tcx)
                    .has_non_region_param()
            })
            && matches!(
                destination.ty(&body.local_decls, self.tcx).ty.kind(),
                ty::Uint(ty::UintTy::Usize)
            )
    }

    fn stable_span(&self, span: Span) -> StableSpan {
        let span = span.source_callsite();
        let source_map = self.tcx.sess.source_map();
        let (file, lo_line, lo_column, hi_line, hi_column) = source_map.span_to_location_info(span);
        let relative = file
            .as_ref()
            .and_then(|file| {
                let local = file
                    .name
                    .display(FileNameDisplayPreference::Local)
                    .to_string();
                Path::new(&local)
                    .strip_prefix(&self.project_root)
                    .ok()
                    .map(Path::to_path_buf)
                    .or_else(|| {
                        let remapped = file
                            .name
                            .display(FileNameDisplayPreference::Remapped)
                            .to_string();
                        let remapped = PathBuf::from(remapped);
                        (!remapped.is_absolute()).then_some(remapped)
                    })
            })
            .unwrap_or_else(|| PathBuf::from("src/lib.rs"));
        let path = relative.to_string_lossy().replace('\\', "/");
        StableSpan::new(
            path,
            StablePosition::new(lo_line.max(1) as u32, lo_column.max(1) as u32),
            StablePosition::new(hi_line.max(lo_line).max(1) as u32, hi_column.max(1) as u32),
        )
    }

    fn point(&self, function: &FunctionKey, location: Location, span: Span) -> ProgramPoint {
        ProgramPoint {
            function: function.clone(),
            block: location.block.index() as u32,
            statement: location.statement_index as u32,
            span: self.stable_span(span),
        }
    }
}

impl From<AbstractValue> for ValueFacts {
    fn from(value: AbstractValue) -> Self {
        Self {
            value,
            ..Self::default()
        }
    }
}

trait CallOperandExt {
    fn place(&self) -> Option<&PlaceKey>;
}

impl CallOperandExt for CallOperand {
    fn place(&self) -> Option<&PlaceKey> {
        match self {
            Self::Place(place) => Some(place),
            _ => None,
        }
    }
}

trait ProgramPointLocation {
    fn point_location(&self) -> Location;
}

impl ProgramPointLocation for ProgramPoint {
    fn point_location(&self) -> Location {
        Location {
            block: BasicBlock::from_usize(self.block as usize),
            statement_index: self.statement as usize,
        }
    }
}

fn place_key(place: Place<'_>) -> PlaceKey {
    PlaceKey::new(format!("{place:?}"))
}

fn parse_local(place: &PlaceKey) -> Option<usize> {
    place
        .0
        .strip_prefix('_')?
        .split(|character: char| !character.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn value_range_end(value: &mut ValueFacts, range_end: PlaceKey) {
    value.range_end = Some(range_end);
}

fn predicate_result(predicate: &PredicateFact) -> &PlaceKey {
    match predicate {
        PredicateFact::IsEmpty { result, .. }
        | PredicateFact::IsNull { result, .. }
        | PredicateFact::Utf8Result { result, .. }
        | PredicateFact::IsErr { result, .. } => result,
    }
}

fn select_boundary(inputs: BoundaryInputs) -> BoundaryDisposition {
    if inputs.resolved_local {
        BoundaryDisposition::ResolvedLocal
    } else if inputs.exact_ffi_out {
        BoundaryDisposition::ExactFfiOut
    } else if inputs.selected_open_behavior {
        BoundaryDisposition::SelectedOpenBehavior
    } else if inputs.modeled_registry {
        BoundaryDisposition::ModeledRegistry
    } else if inputs.direct {
        BoundaryDisposition::OpaqueDirect
    } else {
        BoundaryDisposition::OpaqueIndirect
    }
}

fn boundary_mutation_write_kind(disposition: BoundaryDisposition) -> Option<WriteKind> {
    match disposition {
        BoundaryDisposition::ExactFfiOut => Some(WriteKind::ForeignOut),
        BoundaryDisposition::OpaqueDirect | BoundaryDisposition::OpaqueIndirect => {
            Some(WriteKind::OpaqueMutable)
        }
        _ => None,
    }
}

fn is_exact_ffi_out_ty(ty: ty::Ty<'_>) -> bool {
    matches!(ty.kind(), ty::RawPtr(inner, Mutability::Mut)
        if matches!(inner.kind(), ty::RawPtr(_, Mutability::Mut)))
}

fn is_mutable_call_actual(ty: ty::Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        ty::Ref(_, _, Mutability::Mut) | ty::RawPtr(_, Mutability::Mut)
    )
}

fn record_write(facts: &mut BodyFacts, place: PlaceKey, point: ProgramPoint, kind: WriteKind) {
    facts.summary_seed.writes.insert(WriteEffect {
        place: place.clone(),
        point: point.clone(),
    });
    facts.writes.push(WriteFact { place, point, kind });
}

fn destination_write_invalidates(projected_destination: bool, already_defined: bool) -> bool {
    projected_destination || already_defined
}

fn internal_only_origins(value: &AbstractValue, has_self: bool) -> bool {
    value.origins.iter().any(|origin| {
        matches!(
            origin,
            AbstractOrigin::PrivateField { .. } | AbstractOrigin::InternalLocal { .. }
        )
    }) && value.origins.iter().all(|origin| {
        matches!(
            origin,
            AbstractOrigin::PrivateField { .. }
                | AbstractOrigin::InternalLocal { .. }
                | AbstractOrigin::Constant(_)
        ) || matches!(origin, AbstractOrigin::Formal(1) if has_self)
    })
}

fn validation_success_decision(
    same_result: bool,
    success_dominates_sink: bool,
    established_precedes_branch: bool,
) -> bool {
    same_result && success_dominates_sink && established_precedes_branch
}

fn same_binding(left: &BTreeSet<PlaceKey>, right: &BTreeSet<PlaceKey>) -> bool {
    !left.is_disjoint(right)
}

fn cfg_point(point: &ProgramPoint) -> CfgPoint {
    CfgPoint {
        block: point.block,
        statement: point.statement,
    }
}

fn cfg_successors(body: &Body<'_>) -> BTreeMap<u32, BTreeSet<u32>> {
    body.basic_blocks
        .iter_enumerated()
        .map(|(block, data)| {
            (
                block.index() as u32,
                data.terminator()
                    .successors()
                    .map(|successor| successor.index() as u32)
                    .collect(),
            )
        })
        .collect()
}

fn block_reachable(successors: &BTreeMap<u32, BTreeSet<u32>>, start: u32, target: u32) -> bool {
    let mut pending = vec![start];
    let mut visited = BTreeSet::new();
    while let Some(block) = pending.pop() {
        if block == target {
            return true;
        }
        if !visited.insert(block) {
            continue;
        }
        pending.extend(successors.get(&block).into_iter().flatten().copied());
    }
    false
}

fn write_is_between(
    successors: &BTreeMap<u32, BTreeSet<u32>>,
    success: u32,
    write: CfgPoint,
    sink: CfgPoint,
) -> bool {
    if !block_reachable(successors, success, write.block) {
        return false;
    }
    if write.block == sink.block {
        write.statement < sink.statement
    } else {
        block_reachable(successors, write.block, sink.block)
    }
}

fn write_is_between_points(
    successors: &BTreeMap<u32, BTreeSet<u32>>,
    start: CfgPoint,
    write: CfgPoint,
    end: CfgPoint,
) -> bool {
    let start_reaches_write = if start.block == write.block {
        start.statement < write.statement
    } else {
        block_reachable(successors, start.block, write.block)
    };
    let write_reaches_end = if write.block == end.block {
        write.statement < end.statement
    } else {
        block_reachable(successors, write.block, end.block)
    };
    start_reaches_write && write_reaches_end
}

fn alias_closure_for(facts: &BodyFacts, seed: &PlaceKey) -> BTreeSet<PlaceKey> {
    let mut result = BTreeSet::from([seed.clone()]);
    loop {
        let before = result.len();
        for (place, value) in &facts.values {
            if !value.value_flow.is_disjoint(&result) || result.contains(place) {
                result.insert(place.clone());
                result.extend(value.value_flow.iter().cloned());
            }
        }
        if result.len() == before {
            return result;
        }
    }
}

fn related_to_watched(facts: &BodyFacts, watched: &BTreeSet<PlaceKey>, written: &PlaceKey) -> bool {
    let written_aliases = alias_closure_for(facts, written);
    watched.iter().any(|candidate| {
        !written_aliases.is_disjoint(&alias_closure_for(facts, candidate))
            || matches!(
                (base_local(candidate), base_local(written)),
                (Some(candidate), Some(written)) if candidate == written
            )
    })
}

fn base_local(place: &PlaceKey) -> Option<usize> {
    let start = place.0.find('_')? + 1;
    let digits = place.0[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn generic_nonempty_subject(
    collection: &AbstractValue,
    constant_zero: bool,
) -> Option<AbstractValue> {
    if !constant_zero {
        return None;
    }
    let origins = collection
        .origins
        .iter()
        .filter(|origin| matches!(origin, AbstractOrigin::GenericCapability { .. }))
        .cloned()
        .collect::<BTreeSet<_>>();
    (!origins.is_empty()).then(|| AbstractValue::new(origins))
}

fn matching_generic_capability<'a>(
    collection: &'a AbstractValue,
    subject: &AbstractValue,
) -> Option<&'a AbstractOrigin> {
    collection.origins.iter().find(|origin| {
        matches!(origin, AbstractOrigin::GenericCapability { .. })
            && subject.origins.contains(*origin)
    })
}

fn is_in_bounds_guard(op: BinOp) -> bool {
    op == BinOp::Ge
}

fn is_internal_invalid_value_rule(rule: RuleId) -> bool {
    matches!(
        rule,
        RuleId::P3LifetimeTransmute | RuleId::P3AssumeInitBool | RuleId::P3UncheckedUtf8
    )
}

fn source_kind(
    arg_count: usize,
    has_self: bool,
    source_hint: Option<SourceKind>,
    rule: RuleId,
    origin: &AbstractOrigin,
) -> Option<SourceKind> {
    match origin {
        AbstractOrigin::Formal(index)
            if *index > 0 && (*index as usize) <= arg_count && !(has_self && *index == 1) =>
        {
            Some(SourceKind::PublicParameter)
        }
        AbstractOrigin::Formal(_) => None,
        AbstractOrigin::PublicField { .. } => Some(SourceKind::LiteralPublicField),
        AbstractOrigin::PrivateField { .. } => Some(SourceKind::InternalDerived),
        AbstractOrigin::InternalLocal { .. }
            if source_hint == Some(SourceKind::InternalUnsafeOrigin)
                || is_internal_invalid_value_rule(rule) =>
        {
            Some(SourceKind::InternalUnsafeOrigin)
        }
        AbstractOrigin::InternalLocal { .. } => Some(SourceKind::InternalDerived),
        AbstractOrigin::GenericCapability { .. } => Some(SourceKind::GenericNonEmptyCapability),
        AbstractOrigin::FfiOutput { .. } => Some(SourceKind::FfiOutput),
        AbstractOrigin::OpenBehaviorOutput { .. } => Some(SourceKind::OpenBehaviorOutput),
        AbstractOrigin::Constant(_) => None,
    }
}

fn source_kinds(root: &BodyFacts, requirement: &ContractRequirement) -> BTreeSet<SourceKind> {
    origin_source_kinds(
        root.arg_count,
        root.has_self,
        requirement.source_hint,
        requirement.rule,
        &requirement.subject,
        requirement.collection.as_ref(),
    )
}

fn origin_source_kinds(
    arg_count: usize,
    has_self: bool,
    source_hint: Option<SourceKind>,
    rule: RuleId,
    subject: &AbstractValue,
    collection: Option<&AbstractValue>,
) -> BTreeSet<SourceKind> {
    subject
        .origins
        .iter()
        .chain(
            collection
                .into_iter()
                .flat_map(|value| value.origins.iter()),
        )
        .filter_map(|origin| source_kind(arg_count, has_self, source_hint, rule, origin))
        .collect()
}

fn cycle_step_function(token: &str) -> FunctionKey {
    FunctionKey::new(token)
}

fn origin_matches_kind(origin: &AbstractOrigin, kind: SourceKind) -> bool {
    matches!(
        (origin, kind),
        (AbstractOrigin::Formal(_), SourceKind::PublicParameter)
            | (
                AbstractOrigin::PublicField { .. },
                SourceKind::LiteralPublicField
            )
            | (
                AbstractOrigin::InternalLocal { .. },
                SourceKind::InternalUnsafeOrigin
            )
            | (
                AbstractOrigin::PrivateField { .. },
                SourceKind::InternalDerived
            )
            | (
                AbstractOrigin::GenericCapability { .. },
                SourceKind::GenericNonEmptyCapability
            )
            | (AbstractOrigin::FfiOutput { .. }, SourceKind::FfiOutput)
            | (
                AbstractOrigin::OpenBehaviorOutput { .. },
                SourceKind::OpenBehaviorOutput
            )
    )
}

fn origin_token(origin: &AbstractOrigin) -> String {
    match origin {
        AbstractOrigin::Formal(index) => format!("arg:{index}"),
        AbstractOrigin::PublicField { def_path, .. } => format!("field:{def_path}"),
        AbstractOrigin::PrivateField { def_path, .. } => format!("private-field:{def_path}"),
        AbstractOrigin::InternalLocal { function, place } => {
            format!("internal:{}:{}", function.0, place.0)
        }
        AbstractOrigin::GenericCapability { token, .. }
        | AbstractOrigin::FfiOutput { token, .. }
        | AbstractOrigin::OpenBehaviorOutput { token, .. } => token.clone(),
        AbstractOrigin::Constant(value) => format!("const:{value}"),
    }
}

fn origin_span(origin: &AbstractOrigin) -> Option<StableSpan> {
    match origin {
        AbstractOrigin::PublicField { span, .. }
        | AbstractOrigin::PrivateField { span, .. }
        | AbstractOrigin::GenericCapability { span, .. }
        | AbstractOrigin::FfiOutput { span, .. }
        | AbstractOrigin::OpenBehaviorOutput { span, .. } => Some(span.clone()),
        _ => None,
    }
}

fn adjusted_rule(original: RuleId, primary: Pattern) -> RuleId {
    match (original, primary) {
        (RuleId::P1RawRead, Pattern::P2) => RuleId::P2RawRead,
        (RuleId::P1GetUnchecked, Pattern::P2) => RuleId::P2GetUnchecked,
        (RuleId::P1GetUnchecked, Pattern::P4) => RuleId::P4Bounds,
        _ => original,
    }
}

fn path_token(path: &[CallSite]) -> String {
    path.iter()
        .map(|call| {
            format!(
                "{}@{}",
                call.callee
                    .as_ref()
                    .map(|callee| callee.0.as_str())
                    .unwrap_or("opaque"),
                call.point.span.token()
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn location_precedes(left: Location, right: Location, body: &Body<'_>) -> bool {
    if left.block == right.block {
        return left.statement_index <= right.statement_index;
    }
    body.basic_blocks
        .dominators()
        .dominates(left.block, right.block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> StableSpan {
        StableSpan::new(
            "src/lib.rs",
            StablePosition::new(1, 1),
            StablePosition::new(1, 2),
        )
    }

    #[test]
    fn actual_to_formal_substitution_is_structural() {
        let formal = AbstractValue::new([AbstractOrigin::Formal(2)]);
        let actuals = vec![
            AbstractValue::new([AbstractOrigin::Constant(0)]),
            AbstractValue::new([AbstractOrigin::PublicField {
                def_path: "crate::S::index".into(),
                span: span(),
            }]),
        ];
        assert!(matches!(
            formal.substitute(&actuals).origins.iter().next(),
            Some(AbstractOrigin::PublicField { .. })
        ));
    }

    #[test]
    fn primary_mapping_is_mutually_exclusive() {
        assert_eq!(
            classify_primary_with_secondary(
                FailureClass::SinkPrecondition,
                SourceKind::LiteralPublicField,
                [SourceKind::LiteralPublicField],
            )
            .0,
            Pattern::P2
        );
        assert_eq!(
            classify_primary_with_secondary(
                FailureClass::SinkPrecondition,
                SourceKind::GenericNonEmptyCapability,
                [SourceKind::GenericNonEmptyCapability],
            )
            .0,
            Pattern::P5
        );
        assert_eq!(
            classify_primary_with_secondary(
                FailureClass::SinkPrecondition,
                SourceKind::OpenBehaviorOutput,
                [SourceKind::OpenBehaviorOutput],
            )
            .0,
            Pattern::P6
        );
    }

    #[test]
    fn validation_value_binding_rejects_a_different_origin() {
        let checked = AbstractValue::new([AbstractOrigin::Formal(1)]);
        let sink = AbstractValue::new([AbstractOrigin::Formal(2)]);
        assert!(checked.origins.is_disjoint(&sink.origins));
    }

    #[test]
    fn p5_subject_and_source_bind_to_collection_capability_not_constant_zero() {
        let capability = AbstractOrigin::GenericCapability {
            token: "associated-slice:crate::Provider::Results".into(),
            span: span(),
        };
        let collection = AbstractValue::new([capability.clone(), AbstractOrigin::Formal(1)]);
        let constant_index = AbstractValue::new([AbstractOrigin::Constant(0)]);

        let subject = generic_nonempty_subject(&collection, true).expect("exact P5.1 subject");
        assert_eq!(subject.origins, BTreeSet::from([capability.clone()]));
        assert!(!subject.origins.contains(&AbstractOrigin::Constant(0)));
        assert_eq!(
            matching_generic_capability(&collection, &subject),
            Some(&capability)
        );
        assert!(matching_generic_capability(&collection, &constant_index).is_none());
        assert!(generic_nonempty_subject(&collection, false).is_none());
    }

    #[test]
    fn in_bounds_guard_accepts_ge_but_rejects_gt() {
        assert!(is_in_bounds_guard(BinOp::Ge));
        assert!(!is_in_bounds_guard(BinOp::Gt));
    }

    #[test]
    fn boundary_precedence_is_closed_and_opaque_calls_do_not_carry_p6_origins() {
        let all = BoundaryInputs {
            resolved_local: true,
            exact_ffi_out: true,
            selected_open_behavior: true,
            modeled_registry: true,
            direct: true,
        };
        assert_eq!(select_boundary(all), BoundaryDisposition::ResolvedLocal);
        assert_eq!(
            select_boundary(BoundaryInputs {
                resolved_local: false,
                ..all
            }),
            BoundaryDisposition::ExactFfiOut
        );
        assert_eq!(
            select_boundary(BoundaryInputs {
                resolved_local: false,
                exact_ffi_out: false,
                ..all
            }),
            BoundaryDisposition::SelectedOpenBehavior
        );
        let opaque = select_boundary(BoundaryInputs {
            direct: true,
            ..BoundaryInputs::default()
        });
        assert_eq!(opaque, BoundaryDisposition::OpaqueDirect);
        assert_eq!(
            boundary_mutation_write_kind(opaque),
            Some(WriteKind::OpaqueMutable)
        );
        assert_eq!(
            select_boundary(BoundaryInputs::default()),
            BoundaryDisposition::OpaqueIndirect
        );
    }

    #[test]
    fn exact_binding_helper_accepts_overlap_and_rejects_wrong_places() {
        let checked = BTreeSet::from([PlaceKey::new("_1"), PlaceKey::new("_2")]);
        assert!(same_binding(
            &checked,
            &BTreeSet::from([PlaceKey::new("_2")])
        ));
        assert!(!same_binding(
            &checked,
            &BTreeSet::from([PlaceKey::new("_3")])
        ));
        assert_eq!(base_local(&PlaceKey::new("(*_7).0")), Some(7));
    }

    #[test]
    fn validation_success_requires_binding_dominance_and_order() {
        assert!(validation_success_decision(true, true, true));
        assert!(!validation_success_decision(false, true, true));
        assert!(!validation_success_decision(true, false, true));
        assert!(!validation_success_decision(true, true, false));
    }

    #[test]
    fn diamond_reachability_invalidates_only_writes_on_a_success_to_sink_path() {
        let graph = BTreeMap::from([
            (0, BTreeSet::from([1, 2])),
            (1, BTreeSet::from([3])),
            (2, BTreeSet::from([3])),
            (3, BTreeSet::new()),
            (4, BTreeSet::new()),
        ]);
        let sink = CfgPoint {
            block: 3,
            statement: 5,
        };
        assert!(write_is_between(
            &graph,
            0,
            CfgPoint {
                block: 1,
                statement: 0,
            },
            sink,
        ));
        assert!(!write_is_between(
            &graph,
            0,
            CfgPoint {
                block: 4,
                statement: 0,
            },
            sink,
        ));
        assert!(write_is_between(
            &graph,
            0,
            CfgPoint {
                block: 3,
                statement: 4,
            },
            sink,
        ));
        assert!(!write_is_between(
            &graph,
            0,
            CfgPoint {
                block: 3,
                statement: 5,
            },
            sink,
        ));
    }

    #[test]
    fn write_effect_kinds_keep_assignment_call_and_boundary_mutations_distinct() {
        assert_eq!(
            BTreeSet::from([
                WriteKind::Assignment,
                WriteKind::CallReturn,
                WriteKind::OpaqueMutable,
                WriteKind::ForeignOut,
            ])
            .len(),
            4
        );
        assert_eq!(
            boundary_mutation_write_kind(BoundaryDisposition::ExactFfiOut),
            Some(WriteKind::ForeignOut)
        );
        assert_eq!(
            boundary_mutation_write_kind(BoundaryDisposition::ModeledRegistry),
            None
        );
        assert!(!destination_write_invalidates(false, false));
        assert!(destination_write_invalidates(false, true));
        assert!(destination_write_invalidates(true, false));
    }

    #[test]
    fn internal_derived_requires_internal_only_provenance_and_real_self() {
        let private = AbstractOrigin::PrivateField {
            def_path: "crate::S::field".into(),
            span: span(),
        };
        assert!(internal_only_origins(
            &AbstractValue::new([private.clone(), AbstractOrigin::Formal(1)]),
            true,
        ));
        assert!(!internal_only_origins(
            &AbstractValue::new([private.clone(), AbstractOrigin::Formal(1)]),
            false,
        ));
        assert!(!internal_only_origins(
            &AbstractValue::new([private, AbstractOrigin::Formal(2)]),
            true,
        ));
    }

    #[test]
    fn write_between_validation_and_branch_invalidates_the_fact() {
        let graph = BTreeMap::from([(0, BTreeSet::new())]);
        assert!(write_is_between_points(
            &graph,
            CfgPoint {
                block: 0,
                statement: 2,
            },
            CfgPoint {
                block: 0,
                statement: 3,
            },
            CfgPoint {
                block: 0,
                statement: 4,
            },
        ));
        assert!(!write_is_between_points(
            &graph,
            CfgPoint {
                block: 0,
                statement: 2,
            },
            CfgPoint {
                block: 0,
                statement: 4,
            },
            CfgPoint {
                block: 0,
                statement: 4,
            },
        ));
    }

    #[test]
    fn secondary_sources_are_stable_and_do_not_replace_the_selected_primary() {
        let subject = AbstractValue::new([
            AbstractOrigin::Formal(2),
            AbstractOrigin::PublicField {
                def_path: "crate::S::index".into(),
                span: span(),
            },
        ]);
        let collection = AbstractValue::new([AbstractOrigin::GenericCapability {
            token: "associated-slice:crate::Provider::Results".into(),
            span: span(),
        }]);
        let kinds = origin_source_kinds(
            2,
            false,
            None,
            RuleId::P1GetUnchecked,
            &subject,
            Some(&collection),
        );
        let (primary, secondary) = classify_primary_with_secondary(
            FailureClass::SinkPrecondition,
            SourceKind::PublicParameter,
            kinds,
        );
        assert_eq!(primary, Pattern::P1);
        assert_eq!(
            secondary,
            BTreeSet::from([
                SourceKind::LiteralPublicField,
                SourceKind::GenericNonEmptyCapability,
            ])
        );
    }

    #[test]
    fn secondary_extraction_excludes_self_and_classifies_internal_by_failure() {
        let internal = AbstractOrigin::InternalLocal {
            function: FunctionKey::new("crate::method"),
            place: PlaceKey::new("_3"),
        };
        let subject = AbstractValue::new([
            AbstractOrigin::Formal(1),
            AbstractOrigin::Formal(2),
            internal.clone(),
        ]);
        assert_eq!(
            origin_source_kinds(2, true, None, RuleId::P1GetUnchecked, &subject, None,),
            BTreeSet::from([SourceKind::PublicParameter, SourceKind::InternalDerived,])
        );
        assert_eq!(
            source_kind(
                2,
                true,
                Some(SourceKind::InternalUnsafeOrigin),
                RuleId::P3AssumeInitBool,
                &internal,
            ),
            Some(SourceKind::InternalUnsafeOrigin)
        );
        assert_eq!(
            source_kind(
                2,
                true,
                None,
                RuleId::P1GetUnchecked,
                &AbstractOrigin::Formal(1),
            ),
            None
        );
    }

    #[test]
    fn recursive_witness_uses_the_stable_scc_token_as_its_function() {
        let token = scc_cycle_token(&[FunctionKey::new("b"), FunctionKey::new("a")]);
        assert_eq!(token, "scc:[a,b]");
        assert_eq!(cycle_step_function(&token), FunctionKey::new("scc:[a,b]"));
    }
}
