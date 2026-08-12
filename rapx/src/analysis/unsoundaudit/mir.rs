use super::dataflow::{
    solve_cfg, BlockState, BoundValidation, DataflowResult, MayJoin, StaticWriteToken,
};
use super::summary::{
    classify_primary_with_secondary, map_call_outputs, scc_cycle_token, solve_summaries,
    AbstractOrigin, AbstractValue, BoundarySlot, CallBoundary, CallMapping, CanonicalWitness,
    ContractRequirement, FailureClass, Finding, FunctionKey, FunctionSummary, Obligation,
    Operation, OperationKind, OriginKey, Pattern, PlaceKey, Predicate, ProgramPoint, RuleId,
    SinkObligation, Source, SourceKind, StablePosition, StableSpan, ValidationFact, WitnessStep,
    WitnessStepKind, WriteEffect,
};
use rustc_hir::{
    def::DefKind,
    def_id::{DefId, LocalDefId, LOCAL_CRATE},
    LangItem, Mutability, Safety,
};
use rustc_index::IndexVec;
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LengthAtom {
    collection: PlaceKey,
    definition: StaticWriteToken,
    captured_versions: BTreeSet<StaticWriteToken>,
    invalidated: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ValueFacts {
    value: AbstractValue,
    dependencies: BTreeSet<PlaceKey>,
    value_flow: BTreeSet<PlaceKey>,
    exact_roots: BTreeSet<PlaceKey>,
    semantic_unknown: bool,
    len_of: BTreeSet<PlaceKey>,
    length_atoms: BTreeSet<LengthAtom>,
    length_relation_unknown: bool,
    may_be_non_length: bool,
    range_end: BTreeSet<PlaceKey>,
    pointer_base: BTreeSet<PlaceKey>,
    pointer_offset: BTreeSet<PlaceKey>,
    maybe_uninit_initialized: BTreeSet<bool>,
    generic_capability: bool,
    ffi_output: bool,
    open_behavior: bool,
    havoced: bool,
}

impl MayJoin for ValueFacts {
    fn join_may(&mut self, other: &Self) -> bool {
        let before = self.clone();
        self.value
            .origins
            .extend(other.value.origins.iter().cloned());
        self.dependencies.extend(other.dependencies.iter().cloned());
        self.value_flow.extend(other.value_flow.iter().cloned());
        self.exact_roots.extend(other.exact_roots.iter().cloned());
        self.semantic_unknown |= other.semantic_unknown;
        self.len_of.extend(other.len_of.iter().cloned());
        for atom in &other.length_atoms {
            if let Some(current) = self
                .length_atoms
                .iter()
                .find(|current| {
                    current.collection == atom.collection && current.definition == atom.definition
                })
                .cloned()
            {
                self.length_atoms.remove(&current);
                let mut merged = current;
                merged
                    .captured_versions
                    .extend(atom.captured_versions.iter().copied());
                merged.invalidated |= atom.invalidated;
                self.length_atoms.insert(merged);
            } else {
                self.length_atoms.insert(atom.clone());
            }
        }
        self.length_relation_unknown |= other.length_relation_unknown;
        self.may_be_non_length |= other.may_be_non_length;
        self.range_end.extend(other.range_end.iter().cloned());
        self.pointer_base.extend(other.pointer_base.iter().cloned());
        self.pointer_offset
            .extend(other.pointer_offset.iter().cloned());
        self.maybe_uninit_initialized
            .extend(other.maybe_uninit_initialized.iter().copied());
        self.generic_capability |= other.generic_capability;
        self.ffi_output |= other.ffi_output;
        self.open_behavior |= other.open_behavior;
        self.havoced |= other.havoced;
        *self != before
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ValidationBinding {
    predicate: Predicate,
    subject: PlaceKey,
    collection: Option<PlaceKey>,
}

impl ValidationBinding {
    fn storage_places(&self) -> Vec<PlaceKey> {
        let mut places = vec![canonical_storage(&self.subject)];
        places.extend(self.collection.iter().map(canonical_storage));
        places.sort();
        places.dedup();
        places
    }
}

#[derive(Clone, Debug)]
struct BodyProgram {
    operations: IndexVec<BasicBlock, Vec<ProgramOp>>,
    edges: IndexVec<BasicBlock, EdgeTerm>,
    successors: IndexVec<BasicBlock, BTreeSet<BasicBlock>>,
}

#[derive(Clone, Debug)]
struct ProgramOp {
    point: ProgramPoint,
    kind: ProgramOpKind,
}

#[derive(Clone, Debug)]
enum ProgramOpKind {
    Assign(AssignmentModel),
    Call(CallModel),
    Nop,
}

#[derive(Clone, Debug)]
struct AssignmentModel {
    destination: PlaceKey,
    value: ValueModel,
    field_origin: Option<AbstractOrigin>,
    raw_read_source: Option<OperandModel>,
    legacy_write: bool,
}

#[derive(Clone, Debug)]
struct OperandModel {
    operand: CallOperand,
    field_origin: Option<AbstractOrigin>,
    pure_deref_base: Option<PlaceKey>,
    exact_projection: Option<PlaceKey>,
    projection_unknown: bool,
}

#[derive(Clone, Debug)]
enum ValueModel {
    Operand(OperandModel),
    RefOrRaw(OperandModel),
    Length(OperandModel),
    Compare {
        op: BinOp,
        left: OperandModel,
        right: OperandModel,
    },
    Aggregate {
        operands: Vec<OperandModel>,
        range_end: Option<PlaceKey>,
    },
    Unknown,
}

#[derive(Clone, Debug)]
struct CallModel {
    descriptor: CallDescriptor,
    value: RegistryValueModel,
    predicate: Option<PredicateModel>,
    mutable_actuals: Vec<PlaceKey>,
    ffi_out_actuals: Vec<PlaceKey>,
    destination_is_bool: bool,
    access_width: Option<u64>,
    legacy_write: bool,
}

#[derive(Clone, Debug)]
struct CallDescriptor {
    callee: Option<FunctionKey>,
    raw_def: Option<DefId>,
    disposition: BoundaryDisposition,
    args: Vec<OperandModel>,
    destination: PlaceKey,
}

#[derive(Clone, Debug)]
enum RegistryValueModel {
    Empty,
    MaybeUninit(bool),
    Length(OperandModel),
    SelectedOpenBehavior {
        token: String,
    },
    GenericCapability {
        token: String,
    },
    SaturatingSub(Vec<OperandModel>),
    Constant(u64),
    SlicePrefix {
        base: OperandModel,
        range: PlaceKey,
    },
    SliceAsPtr(OperandModel),
    WrappingAdd {
        base: OperandModel,
        offset: OperandModel,
    },
    PointerCast(OperandModel),
}

#[derive(Clone, Debug)]
enum PredicateModel {
    Utf8Result { bytes: PlaceKey },
    IsErr { checked: PlaceKey },
    IsEmpty { slice: PlaceKey },
    IsNull { pointer: PlaceKey },
}

#[derive(Clone, Debug)]
enum EdgeTerm {
    CallNormal {
        normal_target: Option<BasicBlock>,
        point: ProgramPoint,
        call: CallModel,
    },
    Switch {
        condition: PlaceKey,
        false_target: BasicBlock,
        true_target: BasicBlock,
    },
    AssertPlumbingOnly {
        condition: PlaceKey,
        expected: bool,
        target: BasicBlock,
    },
    Return,
    None,
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
    arg_values: Vec<ValueFacts>,
    destination: PlaceKey,
    point: ProgramPoint,
    destination_is_bool: bool,
    access_width: Option<u64>,
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
    normal_successor: Option<BasicBlock>,
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
    states: BTreeMap<(BasicBlock, usize), BlockState<PlaceKey, ValueFacts, ValidationBinding>>,
    local_bindings: BTreeMap<ProgramPoint, ValidationBinding>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum ValidationRoute {
    Satisfied,
    Conditional(ValidationFact),
    Hard(ContractRequirement),
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
                    .arg_values
                    .iter()
                    .map(|facts| facts.value.clone())
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
                for validation in &callee_summary.validations {
                    if validation.requirement.return_exposure && !reaches_return {
                        continue;
                    }
                    let Some(state) = caller.states.get(&(
                        call.point.point_location().block,
                        call.point.statement as usize,
                    )) else {
                        derived
                            .requirements
                            .insert(validation.instantiate_requirement(&actuals));
                        continue;
                    };
                    match compose_validation_route(validation, call, state, caller.arg_count) {
                        ValidationRoute::Satisfied => {}
                        ValidationRoute::Conditional(validation) => {
                            derived.validations.insert(validation);
                        }
                        ValidationRoute::Hard(requirement) => {
                            derived.requirements.insert(requirement);
                        }
                    }
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
            let unresolved = summary
                .requirements
                .iter()
                .cloned()
                .chain(
                    summary
                        .validations
                        .iter()
                        .map(|validation| validation.requirement.clone()),
                )
                .collect::<BTreeSet<_>>();
            for requirement in &unresolved {
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
                if self.is_discharged(facts, requirement) {
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
            states: BTreeMap::new(),
            local_bindings: BTreeMap::new(),
        };
        for arg in body.args_iter() {
            facts.values.insert(
                PlaceKey::new(format!("_{0}", arg.index())),
                ValueFacts {
                    value: AbstractValue::new([AbstractOrigin::Formal(arg.index() as u32)]),
                    dependencies: BTreeSet::from([PlaceKey::new(format!("_{0}", arg.index()))]),
                    value_flow: BTreeSet::from([PlaceKey::new(format!("_{0}", arg.index()))]),
                    exact_roots: BTreeSet::from([PlaceKey::new(format!("_{0}", arg.index()))]),
                    may_be_non_length: true,
                    ..ValueFacts::default()
                },
            );
        }

        let program = self.normalize_body(did, body, &function);
        let operation_counts = program.operations.iter().map(Vec::len).collect();
        let mut entry = BlockState::empty();
        for (place, value) in &facts.values {
            entry.set_value(place.clone(), value.clone());
        }
        let solved = solve_cfg(
            &program.successors,
            &operation_counts,
            entry,
            |block, index, state| {
                apply_program_op(&program.operations[block][index], block, index, state)
            },
            |block, target, state| apply_program_edge(&program.edges[block], target, state),
        );
        facts.states = solved.before.clone();
        self.materialize_program(&program, &solved, &mut facts);
        self.add_local_requirements(body, &mut facts);
        self.partition_validation_contracts(&mut facts);
        for (block, edge) in program.edges.iter_enumerated() {
            if !matches!(edge, EdgeTerm::Return) {
                continue;
            }
            if let Some(returned) = solved.exit[block]
                .as_ref()
                .and_then(|state| state.may_values().get(&PlaceKey::new("_0")))
            {
                facts
                    .summary_seed
                    .return_value
                    .origins
                    .extend(returned.value.origins.iter().cloned());
            }
        }
        facts
    }

    fn partition_validation_contracts(&self, facts: &mut BodyFacts) {
        let requirements = facts
            .summary_seed
            .requirements
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for requirement in requirements {
            let Some(predicate) = migrated_validation_predicate(&requirement) else {
                continue;
            };
            let point = requirement.sink.point.point_location();
            let Some(state) = facts.states.get(&(point.block, point.statement_index)) else {
                continue;
            };
            let Some(binding) = facts.local_bindings.get(&requirement.sink.point) else {
                continue;
            };
            if binding.predicate != predicate {
                continue;
            }
            if state_proves_local_binding(state, Some(binding)) {
                facts.summary_seed.requirements.remove(&requirement);
                continue;
            }
            let subject = entry_formal_slot(facts.arg_count, state, &binding.subject);
            let collection = match &binding.collection {
                Some(place) => {
                    let Some(slot) = entry_formal_slot(facts.arg_count, state, place) else {
                        continue;
                    };
                    Some(slot)
                }
                None => None,
            };
            let Some(validation) = ValidationFact::export_entry_contract(
                requirement.clone(),
                subject,
                collection,
                true,
                true,
            ) else {
                continue;
            };
            facts.summary_seed.requirements.remove(&requirement);
            facts.summary_seed.validations.insert(validation);
        }
    }

    fn materialize_program(
        &self,
        program: &BodyProgram,
        solved: &DataflowResult<PlaceKey, ValueFacts, ValidationBinding>,
        facts: &mut BodyFacts,
    ) {
        for (block, operations) in program.operations.iter_enumerated() {
            for (index, operation) in operations.iter().enumerate() {
                let Some(state) = solved.before.get(&(block, index)) else {
                    continue;
                };
                for (place, value) in state.may_values() {
                    facts
                        .values
                        .entry(place.clone())
                        .or_default()
                        .join_may(value);
                }
                match &operation.kind {
                    ProgramOpKind::Assign(assignment) => {
                        let destination = assignment.destination.clone();
                        match &assignment.value {
                            ValueModel::Length(collection) => {
                                if let Some(collection) =
                                    unique_semantic_value(&eval_operand(state, collection))
                                {
                                    facts.lengths.insert(LengthFact {
                                        result: destination.clone(),
                                        collection,
                                    });
                                }
                            }
                            ValueModel::Compare { op, left, right } => {
                                facts.compares.push(CompareFact {
                                    result: destination.clone(),
                                    op: *op,
                                    left: left.operand.clone(),
                                    right: right.operand.clone(),
                                    point: operation.point.clone(),
                                });
                            }
                            _ => {}
                        }
                        if assignment.legacy_write {
                            record_write(
                                facts,
                                destination,
                                operation.point.clone(),
                                WriteKind::Assignment,
                            );
                        }
                        if let Some(source) = &assignment.raw_read_source {
                            let subject = eval_operand(state, source).value;
                            facts.summary_seed.requirements.insert(ContractRequirement {
                                seed_id: format!("P1.raw:{}", operation.point.span.token()),
                                collection: None,
                                subject,
                                source_hint: None,
                                internal_derivation: false,
                                source_span: operation.point.span.clone(),
                                first_failure: Operation {
                                    kind: OperationKind::RawRead,
                                    point: operation.point.clone(),
                                },
                                sink: Operation {
                                    kind: OperationKind::RawRead,
                                    point: operation.point.clone(),
                                },
                                predicates: BTreeSet::from([Predicate::ValidForRead]),
                                access_width: None,
                                rule: RuleId::P1RawRead,
                                return_exposure: false,
                                sink_function: facts.function.clone(),
                            });
                        }
                    }
                    ProgramOpKind::Call(call) => {
                        let descriptor = &call.descriptor;
                        let normal_successor = match &program.edges[block] {
                            EdgeTerm::CallNormal {
                                normal_target: Some(target),
                                ..
                            } if solved.edge.contains_key(&(block, *target)) => Some(*target),
                            _ => None,
                        };
                        let normal_edge_state =
                            normal_successor.and_then(|target| solved.edge.get(&(block, target)));
                        let arg_values = descriptor
                            .args
                            .iter()
                            .map(|operand| eval_operand(state, operand))
                            .collect::<Vec<_>>();
                        let args = descriptor
                            .args
                            .iter()
                            .map(|operand| operand.operand.clone())
                            .collect::<Vec<_>>();
                        facts.calls.push(CallSite {
                            callee: descriptor.callee.clone(),
                            raw_def: descriptor.raw_def,
                            disposition: descriptor.disposition,
                            args: args.clone(),
                            arg_values,
                            destination: descriptor.destination.clone(),
                            point: operation.point.clone(),
                            destination_is_bool: call.destination_is_bool,
                            access_width: call.access_width,
                        });
                        if let Some(callee) = &descriptor.callee {
                            let mut mapping = BTreeSet::new();
                            for (index, operand) in args.iter().enumerate() {
                                if let CallOperand::Place(actual) = operand {
                                    mapping.insert(CallMapping::ActualToFormal {
                                        actual: actual.clone(),
                                        formal_index: index as u32 + 1,
                                    });
                                }
                            }
                            mapping.insert(CallMapping::ReturnToDestination {
                                destination: descriptor.destination.clone(),
                            });
                            facts.summary_seed.calls.insert(CallBoundary {
                                caller: facts.function.clone(),
                                callee: callee.clone(),
                                point: operation.point.clone(),
                                mapping,
                            });
                        }
                        if call.legacy_write && normal_edge_state.is_some() {
                            record_edge_write(
                                facts,
                                descriptor.destination.clone(),
                                operation.point.clone(),
                                WriteKind::CallReturn,
                                normal_successor,
                            );
                        }
                        if normal_edge_state.is_some() {
                            for target in call
                                .mutable_actuals
                                .iter()
                                .flat_map(|actual| state_alias_closure(state, actual).into_iter())
                            {
                                record_edge_write(
                                    facts,
                                    target,
                                    operation.point.clone(),
                                    WriteKind::OpaqueMutable,
                                    normal_successor,
                                );
                            }
                            for target in call
                                .ffi_out_actuals
                                .iter()
                                .flat_map(|actual| state_alias_closure(state, actual).into_iter())
                            {
                                record_edge_write(
                                    facts,
                                    target,
                                    operation.point.clone(),
                                    WriteKind::ForeignOut,
                                    normal_successor,
                                );
                            }
                        }
                        if normal_edge_state.is_some() {
                            if let RegistryValueModel::Length(collection) = &call.value {
                                if let Some(collection) =
                                    unique_semantic_value(&eval_operand(state, collection))
                                {
                                    facts.lengths.insert(LengthFact {
                                        result: descriptor.destination.clone(),
                                        collection,
                                    });
                                }
                            }
                        }
                        if normal_edge_state.is_some() {
                            if let Some(predicate) = &call.predicate {
                                facts.predicates.push(match predicate {
                                    PredicateModel::Utf8Result { bytes } => {
                                        PredicateFact::Utf8Result {
                                            result: descriptor.destination.clone(),
                                            bytes: bytes.clone(),
                                        }
                                    }
                                    PredicateModel::IsErr { checked } => PredicateFact::IsErr {
                                        result: descriptor.destination.clone(),
                                        checked: checked.clone(),
                                    },
                                    PredicateModel::IsEmpty { slice } => PredicateFact::IsEmpty {
                                        result: descriptor.destination.clone(),
                                        slice: slice.clone(),
                                    },
                                    PredicateModel::IsNull { pointer } => PredicateFact::IsNull {
                                        result: descriptor.destination.clone(),
                                        pointer: pointer.clone(),
                                    },
                                });
                            }
                        }
                    }
                    ProgramOpKind::Nop => {}
                }
            }
            if let EdgeTerm::Switch {
                condition,
                false_target,
                ..
            } = &program.edges[block]
            {
                let point = operations
                    .last()
                    .expect("each MIR block has a terminator operation")
                    .point
                    .clone();
                facts.branches.push(BranchFact {
                    result: condition.clone(),
                    success: *false_target,
                    point,
                });
            }
        }
        for state in solved.exit.iter().flatten() {
            for (place, value) in state.may_values() {
                facts
                    .values
                    .entry(place.clone())
                    .or_default()
                    .join_may(value);
            }
        }
    }

    fn normalize_body(
        &self,
        caller: DefId,
        body: &Body<'tcx>,
        function: &FunctionKey,
    ) -> BodyProgram {
        let mut operations = IndexVec::new();
        let mut edges = IndexVec::new();
        let mut successors = IndexVec::new();
        let mut definition_counts = BTreeMap::<PlaceKey, usize>::new();
        for data in body.basic_blocks.iter() {
            for statement in &data.statements {
                if let StatementKind::Assign(box (destination, _)) = &statement.kind {
                    *definition_counts
                        .entry(place_key(*destination))
                        .or_default() += 1;
                }
            }
            if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
                *definition_counts
                    .entry(place_key(*destination))
                    .or_default() += 1;
            }
        }
        for (block, data) in body.basic_blocks.iter_enumerated() {
            let mut block_ops = Vec::with_capacity(data.statements.len() + 1);
            for (statement_index, statement) in data.statements.iter().enumerate() {
                let point = self.point(
                    function,
                    Location {
                        block,
                        statement_index,
                    },
                    statement.source_info.span,
                );
                let kind = match &statement.kind {
                    StatementKind::Assign(box (destination, rvalue)) => {
                        let destination_key = place_key(*destination);
                        let legacy_write = !destination.projection.is_empty()
                            || definition_counts
                                .get(&destination_key)
                                .copied()
                                .unwrap_or(0)
                                > 1;
                        ProgramOpKind::Assign(self.normalize_assignment(
                            body,
                            *destination,
                            rvalue,
                            &point,
                            legacy_write,
                        ))
                    }
                    _ => ProgramOpKind::Nop,
                };
                block_ops.push(ProgramOp { point, kind });
            }
            let statement_index = data.statements.len();
            let terminator = data.terminator();
            let point = self.point(
                function,
                Location {
                    block,
                    statement_index,
                },
                terminator.source_info.span,
            );
            let kind = match &terminator.kind {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } => {
                    let destination_key = place_key(*destination);
                    let legacy_write = !destination.projection.is_empty()
                        || definition_counts
                            .get(&destination_key)
                            .copied()
                            .unwrap_or(0)
                            > 1;
                    ProgramOpKind::Call(self.normalize_call(
                        caller,
                        body,
                        func,
                        args,
                        *destination,
                        &point,
                        legacy_write,
                    ))
                }
                _ => ProgramOpKind::Nop,
            };
            block_ops.push(ProgramOp {
                point: point.clone(),
                kind: kind.clone(),
            });
            operations.push(block_ops);
            successors.push(terminator.successors().collect());
            edges.push(match &terminator.kind {
                TerminatorKind::Call { target, .. } => match kind {
                    ProgramOpKind::Call(call) => EdgeTerm::CallNormal {
                        normal_target: *target,
                        point,
                        call,
                    },
                    _ => unreachable!("a normalized Call has a CallModel"),
                },
                TerminatorKind::SwitchInt { discr, targets } => {
                    discr
                        .place()
                        .map_or(EdgeTerm::None, |place| EdgeTerm::Switch {
                            condition: place_key(place),
                            false_target: targets.target_for_value(0),
                            true_target: targets.target_for_value(1),
                        })
                }
                TerminatorKind::Assert {
                    cond,
                    expected,
                    target,
                    ..
                } => cond
                    .place()
                    .map_or(EdgeTerm::None, |place| EdgeTerm::AssertPlumbingOnly {
                        condition: place_key(place),
                        expected: *expected,
                        target: *target,
                    }),
                TerminatorKind::Return => EdgeTerm::Return,
                _ => EdgeTerm::None,
            });
        }
        BodyProgram {
            operations,
            edges,
            successors,
        }
    }

    fn normalize_assignment(
        &self,
        body: &Body<'tcx>,
        destination: Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        point: &ProgramPoint,
        legacy_write: bool,
    ) -> AssignmentModel {
        let destination_key = place_key(destination);
        let value = match rvalue {
            Rvalue::Use(operand) => {
                ValueModel::Operand(self.normalize_operand(body, operand, point))
            }
            Rvalue::Cast(
                CastKind::PointerCoercion(ty::adjustment::PointerCoercion::Unsize, _),
                operand,
                _,
            ) => ValueModel::Operand(self.normalize_operand(body, operand, point)),
            Rvalue::Cast(_, _, _) => ValueModel::Unknown,
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
                let mut operand = self.normalize_place(body, *place, point);
                if matches!(place.projection.as_ref(), [ProjectionElem::Deref]) {
                    operand.pure_deref_base = Some(place_key(Place::from(place.local)));
                    operand.exact_projection = None;
                    operand.projection_unknown = false;
                }
                ValueModel::RefOrRaw(operand)
            }
            Rvalue::Len(place) => ValueModel::Length(self.normalize_place(body, *place, point)),
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => operand
                .place()
                .map(|place| ValueModel::Length(self.normalize_place(body, place, point)))
                .unwrap_or(ValueModel::Unknown),
            Rvalue::BinaryOp(op, operands) => ValueModel::Compare {
                op: *op,
                left: self.normalize_operand(body, &operands.0, point),
                right: self.normalize_operand(body, &operands.1, point),
            },
            Rvalue::Aggregate(kind, operands) => ValueModel::Aggregate {
                operands: operands
                    .iter()
                    .map(|operand| self.normalize_operand(body, operand, point))
                    .collect(),
                range_end: matches!(&**kind, AggregateKind::Adt(did, ..)
                    if self.tcx.item_name(*did).as_str() == "RangeTo")
                .then(|| {
                    operands
                        .iter()
                        .next()
                        .and_then(|operand| operand.place())
                        .map(place_key)
                })
                .flatten(),
            },
            _ => ValueModel::Unknown,
        };
        let field_origin = self
            .field_info(body, destination)
            .map(|(public, def_path)| {
                if public {
                    AbstractOrigin::PublicField {
                        def_path,
                        span: point.span.clone(),
                    }
                } else {
                    AbstractOrigin::PrivateField {
                        def_path,
                        span: point.span.clone(),
                    }
                }
            });
        let raw_read_source = match rvalue {
            Rvalue::Use(operand) => operand
                .place()
                .filter(|place| {
                    place
                        .projection
                        .iter()
                        .any(|element| matches!(element, ProjectionElem::Deref))
                        && matches!(body.local_decls[place.local].ty.kind(), ty::RawPtr(..))
                })
                .map(|place| self.normalize_place(body, Place::from(place.local), point)),
            _ => None,
        };
        AssignmentModel {
            destination: destination_key,
            value,
            field_origin,
            raw_read_source,
            legacy_write,
        }
    }

    fn normalize_operand(
        &self,
        body: &Body<'tcx>,
        operand: &Operand<'tcx>,
        point: &ProgramPoint,
    ) -> OperandModel {
        operand.place().map_or_else(
            || OperandModel {
                operand: self.call_operand(body, operand),
                field_origin: None,
                pure_deref_base: None,
                exact_projection: None,
                projection_unknown: false,
            },
            |place| self.normalize_place(body, place, point),
        )
    }

    fn normalize_place(
        &self,
        body: &Body<'tcx>,
        place: Place<'tcx>,
        point: &ProgramPoint,
    ) -> OperandModel {
        let field_origin = self.field_info(body, place).map(|(public, def_path)| {
            if public {
                AbstractOrigin::PublicField {
                    def_path,
                    span: point.span.clone(),
                }
            } else {
                AbstractOrigin::PrivateField {
                    def_path,
                    span: point.span.clone(),
                }
            }
        });
        let exact = place_key(place);
        let pure_deref_base = None;
        let projection_unknown = place.projection.iter().any(|element| {
            matches!(
                element,
                ProjectionElem::Index(_) | ProjectionElem::Subslice { .. }
            )
        });
        let exact_projection =
            (!place.projection.is_empty() && !projection_unknown).then_some(exact.clone());
        OperandModel {
            operand: CallOperand::Place(exact.clone()),
            field_origin,
            pure_deref_base,
            exact_projection,
            projection_unknown,
        }
    }

    fn normalize_call(
        &self,
        caller: DefId,
        body: &Body<'tcx>,
        func: &Operand<'tcx>,
        args: &[rustc_span::source_map::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        point: &ProgramPoint,
        legacy_write: bool,
    ) -> CallModel {
        let raw = func.const_fn_def();
        let raw_def = raw.map(|(did, _)| did);
        let callee =
            raw.and_then(|(did, generic_args)| self.resolve_local(caller, did, generic_args));
        let operands = args
            .iter()
            .map(|argument| self.normalize_operand(body, &argument.node, point))
            .collect::<Vec<_>>();
        let destination_key = place_key(destination);
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

        let mutable_actuals = if disposition == BoundaryDisposition::ResolvedLocal
            || disposition == BoundaryDisposition::OpaqueDirect
            || disposition == BoundaryDisposition::OpaqueIndirect
        {
            args.iter()
                .zip(&operands)
                .filter(|(argument, _)| {
                    is_mutable_call_actual(argument.node.ty(&body.local_decls, self.tcx))
                })
                .filter_map(|(_, operand)| operand.operand.place().cloned())
                .collect()
        } else {
            Vec::new()
        };
        let ffi_out_actuals = if disposition == BoundaryDisposition::ExactFfiOut {
            args.iter()
                .zip(&operands)
                .filter(|(argument, _)| {
                    is_exact_ffi_out_ty(argument.node.ty(&body.local_decls, self.tcx))
                })
                .filter_map(|(_, operand)| operand.operand.place().cloned())
                .collect()
        } else {
            Vec::new()
        };

        let mut value = RegistryValueModel::Empty;
        let mut predicate = None;
        if let Some((did, generic_args)) = raw {
            if self.is_maybe_uninit_constructor(did, "uninit") {
                value = RegistryValueModel::MaybeUninit(false);
            } else if self.is_maybe_uninit_constructor(did, "new") {
                value = RegistryValueModel::MaybeUninit(true);
            } else if self.is_slice_len(did) && !operands.is_empty() {
                value = RegistryValueModel::Length(operands[0].clone());
            } else if disposition == BoundaryDisposition::SelectedOpenBehavior {
                value = RegistryValueModel::SelectedOpenBehavior {
                    token: format!("trait-return:{}", self.tcx.def_path_str(did)),
                };
            } else if self.is_associated_slice_as_ref(did, body, args, destination) {
                value = RegistryValueModel::GenericCapability {
                    token: format!("associated-slice:{}", self.tcx.def_path_str(did)),
                };
            }

            if self.is_saturating_sub(did) && operands.len() >= 2 {
                value = RegistryValueModel::SaturatingSub(operands.clone());
            } else if self.tcx.is_diagnostic_item(sym::mem_size_of, did) {
                if let Ok(layout) = self
                    .tcx
                    .layout_of(self.tcx.param_env(caller).and(generic_args.type_at(0)))
                {
                    value = RegistryValueModel::Constant(layout.size.bytes());
                }
            } else if self.is_slice_prefix_index(did) && operands.len() >= 2 {
                if let Some(range) = operands[1].operand.place().cloned() {
                    value = RegistryValueModel::SlicePrefix {
                        base: operands[0].clone(),
                        range,
                    };
                }
            } else if self.is_slice_as_ptr(did) && !operands.is_empty() {
                value = RegistryValueModel::SliceAsPtr(operands[0].clone());
            } else if self.is_wrapping_add(did) && operands.len() >= 2 {
                value = RegistryValueModel::WrappingAdd {
                    base: operands[0].clone(),
                    offset: operands[1].clone(),
                };
            } else if self.is_pointer_cast(did) && !operands.is_empty() {
                value = RegistryValueModel::PointerCast(operands[0].clone());
            }

            predicate =
                if self.tcx.is_diagnostic_item(sym::str_from_utf8, did) && !operands.is_empty() {
                    operands[0]
                        .operand
                        .place()
                        .cloned()
                        .map(|bytes| PredicateModel::Utf8Result { bytes })
                } else if self.is_result_is_err(did) && !operands.is_empty() {
                    operands[0]
                        .operand
                        .place()
                        .cloned()
                        .map(|checked| PredicateModel::IsErr { checked })
                } else if self.is_slice_is_empty(did) && !operands.is_empty() {
                    operands[0]
                        .operand
                        .place()
                        .cloned()
                        .map(|slice| PredicateModel::IsEmpty { slice })
                } else if self.is_pointer_is_null(did) && !operands.is_empty() {
                    operands[0]
                        .operand
                        .place()
                        .cloned()
                        .map(|pointer| PredicateModel::IsNull { pointer })
                } else {
                    None
                };
        }

        let destination_ty = destination.ty(&body.local_decls, self.tcx).ty;
        let access_width = self
            .tcx
            .layout_of(self.tcx.param_env(caller).and(destination_ty))
            .ok()
            .map(|layout| layout.size.bytes());
        CallModel {
            descriptor: CallDescriptor {
                callee,
                raw_def,
                disposition,
                args: operands,
                destination: destination_key,
            },
            value,
            predicate,
            mutable_actuals,
            ffi_out_actuals,
            destination_is_bool: destination_ty.is_bool(),
            access_width,
            legacy_write,
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
                let collection = call.arg_values[0].value.clone();
                let index_subject = call.arg_values[1].value.clone();
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
                let seed_id = format!("{}:{}", rule.as_str(), call.point.span.token());
                let binding = match (&call.args[0], &call.args[1], constant_zero) {
                    (CallOperand::Place(_), CallOperand::Place(_), false) => {
                        unique_semantic_value(&call.arg_values[1]).and_then(|subject| {
                            unique_semantic_value(&call.arg_values[0]).map(|collection| {
                                ValidationBinding {
                                    predicate: Predicate::InBounds,
                                    subject,
                                    collection: Some(collection),
                                }
                            })
                        })
                    }
                    (CallOperand::Place(_), _, true) => unique_semantic_value(&call.arg_values[0])
                        .map(|collection| ValidationBinding {
                            predicate: Predicate::NonEmpty,
                            subject: collection,
                            collection: None,
                        }),
                    _ => None,
                };
                if let Some(binding) = binding {
                    facts.local_bindings.insert(call.point.clone(), binding);
                }
                facts.summary_seed.requirements.insert(ContractRequirement {
                    seed_id,
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
                let initialization = &call.arg_values[0].maybe_uninit_initialized;
                if initialization == &BTreeSet::from([false])
                    && call.destination_is_bool
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
                let subject = call.arg_values[0].value.clone();
                if subject
                    .origins
                    .iter()
                    .any(|origin| matches!(origin, AbstractOrigin::FfiOutput { .. }))
                {
                    let seed_id = format!("P6.ffi:{}", call.point.span.token());
                    if matches!(&call.args[0], CallOperand::Place(_)) {
                        if let Some(subject) = unique_semantic_value(&call.arg_values[0]) {
                            facts.local_bindings.insert(
                                call.point.clone(),
                                ValidationBinding {
                                    predicate: Predicate::NonNull,
                                    subject,
                                    collection: None,
                                },
                            );
                        }
                    }
                    facts.summary_seed.requirements.insert(ContractRequirement {
                        seed_id,
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
                let pointer = call.arg_values[0].value.clone();
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
                        access_width: call.access_width,
                        rule: RuleId::P4Offset,
                        return_exposure: false,
                        sink_function: facts.function.clone(),
                    });
                }
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
        if let Some(AbstractOrigin::Formal(index)) =
            requirement.subject.origins.iter().find(|origin| {
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

    fn is_discharged(&self, root: &BodyFacts, requirement: &ContractRequirement) -> bool {
        if requirement.sink_function == root.function {
            return self.local_validation(
                root,
                requirement,
                requirement.sink.point.point_location(),
            );
        }
        false
    }

    fn local_validation(
        &self,
        facts: &BodyFacts,
        requirement: &ContractRequirement,
        sink: Location,
    ) -> bool {
        let predicate = requirement.predicates.iter().next().copied();
        if matches!(
            predicate,
            Some(Predicate::InBounds | Predicate::NonEmpty | Predicate::NonNull)
        ) && requirement.sink_function == facts.function
        {
            if let Some(state) = facts.states.get(&(sink.block, sink.statement_index)) {
                return state_proves_local_binding(
                    state,
                    facts.local_bindings.get(&requirement.sink.point),
                );
            }
        }
        let body = self.tcx.optimized_mir(facts.def_id.to_def_id());
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
            .flat_map(|place| {
                self.resolve_value(facts, place)
                    .pointer_offset
                    .iter()
                    .cloned()
            })
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
                value
                    .dependencies
                    .iter()
                    .any(|dependency| !self.resolve_value(facts, dependency).len_of.is_empty())
                    && value.dependencies.iter().any(|dependency| {
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
            if !related_to_watched(facts, candidates, &write.place) {
                return false;
            }
            let before_branch =
                write_is_between_points(
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
                ) && write_normal_edge_reaches(&successors, write, region.branch.block);
            let before_sink = write_is_between(
                &successors,
                region.success.index() as u32,
                cfg_point(&write.point),
                CfgPoint {
                    block: sink.block.index() as u32,
                    statement: sink.statement_index as u32,
                },
            ) && write_normal_edge_reaches(&successors, write, sink.block);
            before_branch || before_sink
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

    fn resolve_value<'a>(&self, facts: &'a BodyFacts, place: &PlaceKey) -> &'a ValueFacts {
        facts.values.get(place).unwrap_or_else(|| {
            static EMPTY: std::sync::OnceLock<ValueFacts> = std::sync::OnceLock::new();
            EMPTY.get_or_init(ValueFacts::default)
        })
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

fn canonical_storage(place: &PlaceKey) -> PlaceKey {
    let start = place.0.find('_').map(|index| index + 1);
    let Some(start) = start else {
        return place.clone();
    };
    let digits = place.0[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        place.clone()
    } else {
        PlaceKey::new(format!("_{digits}"))
    }
}

fn state_value(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    place: &PlaceKey,
) -> ValueFacts {
    let mut value = state.may_values().get(place).cloned().unwrap_or_default();
    let storage = canonical_storage(place);
    if storage != *place {
        if let Some(storage_value) = state.may_values().get(&storage) {
            value.join_may(storage_value);
        }
    }
    value
}

fn exact_semantics(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    place: &PlaceKey,
) -> (BTreeSet<PlaceKey>, bool) {
    state
        .may_values()
        .get(place)
        .map(|facts| (facts.exact_roots.clone(), facts.semantic_unknown))
        .unwrap_or_default()
}

fn semantic_value_at(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    place: &PlaceKey,
) -> ValueFacts {
    let (exact_roots, semantic_unknown) = exact_semantics(state, place);
    ValueFacts {
        exact_roots,
        semantic_unknown,
        ..ValueFacts::default()
    }
}

fn current_storage_versions(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    place: &PlaceKey,
) -> BTreeSet<StaticWriteToken> {
    state
        .versions()
        .get(&canonical_storage(place))
        .cloned()
        .unwrap_or_default()
}

fn semantic_place_candidates(value: &ValueFacts) -> BTreeSet<PlaceKey> {
    value.exact_roots.clone()
}

fn unique_semantic_value(value: &ValueFacts) -> Option<PlaceKey> {
    if value.semantic_unknown {
        return None;
    }
    let mut roots = semantic_place_candidates(value).into_iter();
    match (roots.next(), roots.next()) {
        (Some(root), None) => Some(root),
        _ => None,
    }
}

fn length_relation_is_current(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    value: &ValueFacts,
    collection: &PlaceKey,
) -> bool {
    let mut atoms = value.length_atoms.iter();
    let Some(atom) = atoms.next() else {
        return false;
    };
    atoms.next().is_none()
        && atom.collection == *collection
        && !atom.invalidated
        && atom.captured_versions == current_storage_versions(state, collection)
}

fn eval_operand(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    operand: &OperandModel,
) -> ValueFacts {
    let mut result = match &operand.operand {
        CallOperand::Place(place) => {
            let mut value = state_value(state, place);
            let (exact_roots, semantic_unknown) = operand
                .pure_deref_base
                .as_ref()
                .map(|base| exact_semantics(state, base))
                .unwrap_or_else(|| exact_semantics(state, place));
            value.exact_roots = exact_roots;
            value.semantic_unknown = semantic_unknown;
            value.exact_roots.extend(operand.exact_projection.clone());
            value.semantic_unknown |= operand.projection_unknown;
            value.dependencies.insert(place.clone());
            value.value_flow.insert(place.clone());
            value
        }
        CallOperand::Constant(value) => ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Constant(*value)]),
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
        CallOperand::Unknown => ValueFacts {
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
    };
    if let Some(origin) = &operand.field_origin {
        result
            .value
            .origins
            .retain(|origin| !matches!(origin, AbstractOrigin::Formal(_)));
        result.value.origins.insert(origin.clone());
    }
    result
}

fn join_value_facts(values: impl IntoIterator<Item = ValueFacts>) -> ValueFacts {
    let mut result = ValueFacts::default();
    for value in values {
        result.join_may(&value);
    }
    result
}

fn eval_value_model(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    model: &ValueModel,
    definition: StaticWriteToken,
) -> ValueFacts {
    match model {
        ValueModel::Operand(operand) => eval_operand(state, operand),
        ValueModel::RefOrRaw(place) => eval_operand(state, place),
        ValueModel::Length(collection) => {
            let mut result = eval_operand(state, collection);
            let candidates = semantic_place_candidates(&result);
            result.length_relation_unknown = result.semantic_unknown;
            result.len_of.clear();
            result.length_atoms.clear();
            for collection in candidates {
                result.len_of.insert(collection.clone());
                result.length_atoms.insert(LengthAtom {
                    collection: collection.clone(),
                    definition,
                    captured_versions: current_storage_versions(state, &collection),
                    invalidated: false,
                });
            }
            result.exact_roots.clear();
            result.value_flow.clear();
            result.may_be_non_length = false;
            result.havoced = false;
            result
        }
        ValueModel::Compare { left, right, .. } => {
            let mut result =
                join_value_facts([eval_operand(state, left), eval_operand(state, right)]);
            result.value_flow.clear();
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.len_of.clear();
            result.length_atoms.clear();
            result.may_be_non_length = true;
            result
        }
        ValueModel::Aggregate {
            operands,
            range_end,
        } => {
            let mut result =
                join_value_facts(operands.iter().map(|operand| eval_operand(state, operand)));
            if let Some(range_end) = range_end {
                result.range_end.insert(range_end.clone());
            }
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result
        }
        ValueModel::Unknown => ValueFacts {
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
    }
}

fn eval_registry_value(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    model: &RegistryValueModel,
    point: &ProgramPoint,
) -> ValueFacts {
    let definition = StaticWriteToken::new(point.block, point.statement);
    match model {
        RegistryValueModel::Empty => ValueFacts {
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::MaybeUninit(initialized) => ValueFacts {
            maybe_uninit_initialized: BTreeSet::from([*initialized]),
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::Length(collection) => {
            let mut result = eval_operand(state, collection);
            let candidates = semantic_place_candidates(&result);
            result.length_relation_unknown = result.semantic_unknown;
            result.len_of.clear();
            result.length_atoms.clear();
            for collection in candidates {
                result.len_of.insert(collection.clone());
                result.length_atoms.insert(LengthAtom {
                    collection: collection.clone(),
                    definition,
                    captured_versions: current_storage_versions(state, &collection),
                    invalidated: false,
                });
            }
            result.exact_roots.clear();
            result.value_flow.clear();
            result.may_be_non_length = false;
            result.havoced = false;
            result
        }
        RegistryValueModel::SelectedOpenBehavior { token } => ValueFacts {
            value: AbstractValue::new([AbstractOrigin::OpenBehaviorOutput {
                token: token.clone(),
                span: point.span.clone(),
            }]),
            open_behavior: true,
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::GenericCapability { token } => ValueFacts {
            value: AbstractValue::new([AbstractOrigin::GenericCapability {
                token: token.clone(),
                span: point.span.clone(),
            }]),
            generic_capability: true,
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::SaturatingSub(operands) => {
            let mut result =
                join_value_facts(operands.iter().map(|operand| eval_operand(state, operand)));
            result.value_flow.clear();
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.len_of.clear();
            result.length_atoms.clear();
            result.may_be_non_length = true;
            result
        }
        RegistryValueModel::Constant(value) => ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Constant(*value)]),
            may_be_non_length: true,
            semantic_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::SlicePrefix { base, range } => {
            let mut result = eval_operand(state, base);
            result.range_end.extend(state_value(state, range).range_end);
            result.range_end.insert(range.clone());
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result
        }
        RegistryValueModel::SliceAsPtr(base) => {
            let mut result = eval_operand(state, base);
            result.pointer_base.extend(base.operand.place().cloned());
            result
        }
        RegistryValueModel::WrappingAdd { base, offset } => {
            let mut result = eval_operand(state, base);
            result.pointer_base.extend(base.operand.place().cloned());
            result
                .pointer_offset
                .extend(offset.operand.place().cloned());
            result
                .value
                .origins
                .extend(eval_operand(state, offset).value.origins);
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result
        }
        RegistryValueModel::PointerCast(base) => {
            let mut result = eval_operand(state, base);
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result
        }
    }
}

fn state_alias_closure(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    seed: &PlaceKey,
) -> BTreeSet<PlaceKey> {
    let mut aliases = BTreeSet::from([seed.clone(), canonical_storage(seed)]);
    loop {
        let before = aliases.len();
        for (place, value) in state.may_values() {
            if aliases.contains(place) || !value.value_flow.is_disjoint(&aliases) {
                aliases.insert(place.clone());
                aliases.extend(value.value_flow.iter().cloned());
            }
        }
        if aliases.len() == before {
            break;
        }
    }
    aliases
        .into_iter()
        .map(|place| canonical_storage(&place))
        .collect()
}

fn invalidate_length_evidence(
    state: &mut BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    written_storage: &PlaceKey,
) {
    let updates = state
        .may_values()
        .iter()
        .filter_map(|(place, value)| {
            let mut updated = value.clone();
            let mut changed = false;
            let atoms = updated.length_atoms.iter().cloned().collect::<Vec<_>>();
            for atom in atoms {
                if canonical_storage(&atom.collection) != *written_storage || atom.invalidated {
                    continue;
                }
                updated.length_atoms.remove(&atom);
                let mut invalidated = atom;
                invalidated.invalidated = true;
                updated.length_atoms.insert(invalidated);
                changed = true;
            }
            changed.then(|| (place.clone(), updated))
        })
        .collect::<Vec<_>>();
    for (place, value) in updates {
        state.set_value(place, value);
    }
}

fn apply_program_op(
    operation: &ProgramOp,
    block: BasicBlock,
    index: usize,
    state: &mut BlockState<PlaceKey, ValueFacts, ValidationBinding>,
) {
    let token = StaticWriteToken::new(block.index() as u32, index as u32);
    match &operation.kind {
        ProgramOpKind::Assign(assignment) => {
            let copied_pending = match &assignment.value {
                ValueModel::Operand(operand) => operand.operand.place().map(|source| {
                    state
                        .pending_must()
                        .get(source)
                        .into_iter()
                        .flat_map(|facts| facts.keys())
                        .filter(|(binding, polarity)| {
                            state.pending_is_current(source, binding, *polarity)
                        })
                        .map(|(binding, polarity)| {
                            (binding.clone(), *polarity, binding.storage_places())
                        })
                        .collect::<Vec<_>>()
                }),
                _ => None,
            };
            let mut value = eval_value_model(state, &assignment.value, token);
            let pending = match &assignment.value {
                ValueModel::Compare {
                    op: BinOp::Ge,
                    left,
                    right,
                } => {
                    let right_value = eval_operand(state, right);
                    let left_value = eval_operand(state, left);
                    match (
                        unique_semantic_value(&left_value),
                        right_value.len_of.iter().next(),
                    ) {
                        (Some(subject), Some(collection))
                            if right_value.len_of.len() == 1
                                && !right_value.may_be_non_length
                                && !right_value.length_relation_unknown
                                && length_relation_is_current(state, &right_value, collection) =>
                        {
                            if left_value.havoced || right_value.havoced {
                                Vec::new()
                            } else {
                                let binding = ValidationBinding {
                                    predicate: Predicate::InBounds,
                                    subject,
                                    collection: Some(collection.clone()),
                                };
                                vec![(binding.clone(), false, binding.storage_places())]
                            }
                        }
                        _ => Vec::new(),
                    }
                }
                _ => copied_pending.unwrap_or_default(),
            };
            if let Some(origin) = &assignment.field_origin {
                value
                    .value
                    .origins
                    .retain(|origin| !matches!(origin, AbstractOrigin::Formal(_)));
                value.value.origins.insert(origin.clone());
            }
            let defines_fresh_identity = matches!(
                &assignment.value,
                ValueModel::Operand(OperandModel {
                    operand: CallOperand::Constant(_),
                    ..
                }) | ValueModel::Length(_)
                    | ValueModel::Compare { .. }
                    | ValueModel::Aggregate { .. }
            );
            if defines_fresh_identity {
                value.exact_roots.clear();
                value.exact_roots.insert(assignment.destination.clone());
                value.semantic_unknown = false;
            }
            let storage = canonical_storage(&assignment.destination);
            state.record_write(storage.clone(), token);
            invalidate_length_evidence(state, &storage);
            state.set_value(assignment.destination.clone(), value);

            if !pending.is_empty() {
                state.set_pending(assignment.destination.clone(), pending);
            }
        }
        // Call effects belong to the normal edge, never the unwind edge.
        ProgramOpKind::Call(_) => {}
        ProgramOpKind::Nop => {}
    }
}

fn apply_program_edge(
    edge: &EdgeTerm,
    target: BasicBlock,
    state: &mut BlockState<PlaceKey, ValueFacts, ValidationBinding>,
) {
    if let EdgeTerm::CallNormal {
        normal_target,
        point,
        call,
    } = edge
    {
        if *normal_target != Some(target) {
            return;
        }
        let return_value = eval_registry_value(state, &call.value, point);
        apply_call_side_effects(call, point, state);
        apply_call_return(call, point, return_value, state);
        return;
    }
    if let EdgeTerm::AssertPlumbingOnly {
        condition,
        expected,
        target: success,
    } = edge
    {
        let _ = (condition, expected, success, target);
        // Task 9 owns Assert equivalence; Task 5 intentionally establishes nothing.
        return;
    }
    let EdgeTerm::Switch {
        condition,
        false_target,
        true_target,
    } = edge
    else {
        // Task 9 owns Lt equivalence.
        return;
    };
    let truth = if target == *true_target && target != *false_target {
        Some(true)
    } else if target == *false_target && target != *true_target {
        Some(false)
    } else {
        None
    };
    let Some(truth) = truth else {
        return;
    };
    let bindings = state
        .pending_must()
        .get(condition)
        .into_iter()
        .flat_map(|facts| facts.keys())
        .filter(|(binding, polarity)| {
            *polarity == truth && state.pending_is_current(condition, binding, *polarity)
        })
        .map(|(binding, _)| binding.clone())
        .collect::<Vec<_>>();
    for binding in bindings {
        state.establish(BoundValidation::new(
            binding.clone(),
            binding.storage_places(),
        ));
    }
}

fn state_proves_local_binding(
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    binding: Option<&ValidationBinding>,
) -> bool {
    binding.is_some_and(|binding| state.has_current_validation(binding, binding.storage_places()))
}

fn migrated_validation_predicate(requirement: &ContractRequirement) -> Option<Predicate> {
    let mut predicates = requirement.predicates.iter().copied();
    let predicate = predicates.next()?;
    if predicates.next().is_some()
        || !matches!(
            predicate,
            Predicate::InBounds | Predicate::NonEmpty | Predicate::NonNull
        )
    {
        return None;
    }
    Some(predicate)
}

fn entry_formal_slot(
    arg_count: usize,
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    place: &PlaceKey,
) -> Option<BoundarySlot> {
    if canonical_storage(place) != *place {
        return None;
    }
    let index = place.0.strip_prefix('_')?.parse::<u32>().ok()?;
    if index == 0 || index as usize > arg_count {
        return None;
    }
    if state
        .versions()
        .get(place)
        .is_some_and(|versions| !versions.is_empty())
        || state_value(state, place).havoced
        || unique_semantic_value(&semantic_value_at(state, place)).as_ref() != Some(place)
    {
        return None;
    }
    Some(BoundarySlot::Formal(index))
}

fn compose_validation_route(
    validation: &ValidationFact,
    call: &CallSite,
    state: &BlockState<PlaceKey, ValueFacts, ValidationBinding>,
    caller_arg_count: usize,
) -> ValidationRoute {
    let requirement = validation.instantiate_requirement(
        &call
            .arg_values
            .iter()
            .map(|value| value.value.clone())
            .collect::<Vec<_>>(),
    );
    let Some(binding) = instantiate_validation_binding(validation, call) else {
        return ValidationRoute::Hard(requirement);
    };
    if state.has_current_validation(&binding, binding.storage_places()) {
        return ValidationRoute::Satisfied;
    }
    let subject = entry_formal_slot(caller_arg_count, state, &binding.subject);
    let collection = match &binding.collection {
        Some(place) => match entry_formal_slot(caller_arg_count, state, place) {
            Some(slot) => Some(slot),
            None => return ValidationRoute::Hard(requirement),
        },
        None => None,
    };
    ValidationFact::export_entry_contract(requirement.clone(), subject, collection, true, true)
        .map_or(
            ValidationRoute::Hard(requirement),
            ValidationRoute::Conditional,
        )
}

fn instantiate_validation_binding(
    validation: &ValidationFact,
    call: &CallSite,
) -> Option<ValidationBinding> {
    let predicate = migrated_validation_predicate(&validation.requirement)?;
    let place_for_slot = |slot: BoundarySlot| match slot {
        BoundarySlot::Formal(index) if index > 0 => call
            .arg_values
            .get(index as usize - 1)
            .and_then(unique_semantic_value),
        BoundarySlot::Formal(_) | BoundarySlot::Return | BoundarySlot::Out(_) => None,
    };
    let subject = place_for_slot(validation.subject)?;
    let collection = validation.collection.and_then(place_for_slot);
    if predicate == Predicate::InBounds && collection.is_none() {
        return None;
    }
    Some(ValidationBinding {
        predicate,
        subject,
        collection,
    })
}

fn apply_call_side_effects(
    call: &CallModel,
    point: &ProgramPoint,
    state: &mut BlockState<PlaceKey, ValueFacts, ValidationBinding>,
) {
    let token = StaticWriteToken::new(point.block, point.statement);
    for actual in &call.mutable_actuals {
        for target in state_alias_closure(state, actual) {
            let mut havoced = state_value(state, &target);
            havoced.havoced = true;
            state.record_may_write(target.clone(), token);
            state.set_value(target.clone(), havoced);
            invalidate_length_evidence(state, &target);
        }
    }
    for actual in &call.ffi_out_actuals {
        for target in state_alias_closure(state, actual) {
            let mut output = state_value(state, &target);
            output.exact_roots.insert(target.clone());
            output.ffi_output = true;
            output.value.origins.insert(AbstractOrigin::FfiOutput {
                token: format!("ffi-out:{}", point.span.token()),
                span: point.span.clone(),
            });
            state.record_may_write(target.clone(), token);
            state.set_value(target.clone(), output);
            invalidate_length_evidence(state, &target);
        }
    }
}

fn apply_call_return(
    call: &CallModel,
    point: &ProgramPoint,
    mut value: ValueFacts,
    state: &mut BlockState<PlaceKey, ValueFacts, ValidationBinding>,
) {
    let token = StaticWriteToken::new(point.block, point.statement);
    let destination = call.descriptor.destination.clone();
    value.exact_roots.clear();
    value.exact_roots.insert(destination.clone());
    value.semantic_unknown = false;
    let storage = canonical_storage(&destination);
    state.record_write(storage.clone(), token);
    invalidate_length_evidence(state, &storage);
    state.set_value(destination.clone(), value);
    let pending = match &call.predicate {
        Some(PredicateModel::IsEmpty { slice })
            if !state_value(state, slice).havoced
                && unique_semantic_value(&semantic_value_at(state, slice)).is_some() =>
        {
            let slice = unique_semantic_value(&semantic_value_at(state, slice))
                .expect("the match guard established one semantic place");
            let binding = ValidationBinding {
                predicate: Predicate::NonEmpty,
                subject: slice,
                collection: None,
            };
            vec![(binding.clone(), false, binding.storage_places())]
        }
        Some(PredicateModel::IsNull { pointer })
            if !state_value(state, pointer).havoced
                && unique_semantic_value(&semantic_value_at(state, pointer)).is_some() =>
        {
            let pointer = unique_semantic_value(&semantic_value_at(state, pointer))
                .expect("the match guard established one semantic place");
            let binding = ValidationBinding {
                predicate: Predicate::NonNull,
                subject: pointer,
                collection: None,
            };
            vec![(binding.clone(), false, binding.storage_places())]
        }
        _ => Vec::new(),
    };
    if !pending.is_empty() {
        state.set_pending(destination, pending);
    }
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
    record_edge_write(facts, place, point, kind, None);
}

fn record_edge_write(
    facts: &mut BodyFacts,
    place: PlaceKey,
    point: ProgramPoint,
    kind: WriteKind,
    normal_successor: Option<BasicBlock>,
) {
    facts.summary_seed.writes.insert(WriteEffect {
        place: place.clone(),
        point: point.clone(),
    });
    facts.writes.push(WriteFact {
        place,
        point,
        kind,
        normal_successor,
    });
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

fn write_normal_edge_reaches(
    successors: &BTreeMap<u32, BTreeSet<u32>>,
    write: &WriteFact,
    target: BasicBlock,
) -> bool {
    write.normal_successor.is_none_or(|normal| {
        block_reachable(successors, normal.index() as u32, target.index() as u32)
    })
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

    fn point(block: u32, statement: u32) -> ProgramPoint {
        ProgramPoint {
            function: FunctionKey::new("crate::f"),
            block,
            statement,
            span: span(),
        }
    }

    fn place_operand(name: &str) -> OperandModel {
        OperandModel {
            operand: CallOperand::Place(PlaceKey::new(name)),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: None,
            projection_unknown: false,
        }
    }

    fn exact_value(place: &PlaceKey) -> ValueFacts {
        ValueFacts {
            exact_roots: BTreeSet::from([place.clone()]),
            ..ValueFacts::default()
        }
    }

    fn seed_exact(
        state: &mut BlockState<PlaceKey, ValueFacts, ValidationBinding>,
        place: &PlaceKey,
    ) {
        state.set_value(place.clone(), exact_value(place));
    }

    fn task6_contract() -> ValidationFact {
        let sink = Operation {
            kind: OperationKind::GetUnchecked,
            point: point(4, 0),
        };
        ValidationFact::export_entry_contract(
            ContractRequirement {
                seed_id: "task6-bounds".into(),
                collection: Some(AbstractValue::new([AbstractOrigin::Formal(1)])),
                subject: AbstractValue::new([AbstractOrigin::Formal(2)]),
                source_hint: None,
                internal_derivation: false,
                source_span: span(),
                first_failure: sink.clone(),
                sink,
                predicates: BTreeSet::from([Predicate::InBounds]),
                access_width: None,
                rule: RuleId::P1GetUnchecked,
                return_exposure: false,
                sink_function: FunctionKey::new("crate::sink"),
            },
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Formal(1)),
            true,
            true,
        )
        .unwrap()
    }

    fn task6_call(collection: &PlaceKey, subject: &PlaceKey) -> CallSite {
        let value = |place: &PlaceKey, formal| ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Formal(formal)]),
            exact_roots: BTreeSet::from([place.clone()]),
            ..ValueFacts::default()
        };
        CallSite {
            callee: Some(FunctionKey::new("crate::sink")),
            raw_def: None,
            disposition: BoundaryDisposition::ResolvedLocal,
            args: vec![
                CallOperand::Place(collection.clone()),
                CallOperand::Place(subject.clone()),
            ],
            arg_values: vec![value(collection, 1), value(subject, 2)],
            destination: PlaceKey::new("_0"),
            point: point(1, 0),
            destination_is_bool: false,
            access_width: None,
        }
    }

    #[test]
    fn task6_call_contract_requires_the_exact_subject_collection_pair() {
        let collection = PlaceKey::new("_1");
        let subject = PlaceKey::new("_2");
        let other = PlaceKey::new("_3");
        let call = task6_call(&collection, &subject);
        let contract = task6_contract();

        let mut guarded = BlockState::empty();
        for place in [&collection, &subject, &other] {
            seed_exact(&mut guarded, place);
        }
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: subject.clone(),
            collection: Some(collection.clone()),
        };
        guarded.establish(BoundValidation::new(
            binding.clone(),
            binding.storage_places(),
        ));
        assert_eq!(
            compose_validation_route(&contract, &call, &guarded, 3),
            ValidationRoute::Satisfied
        );

        let mut wrong_collection = BlockState::empty();
        for place in [&collection, &subject, &other] {
            seed_exact(&mut wrong_collection, place);
        }
        let wrong = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: subject.clone(),
            collection: Some(other),
        };
        wrong_collection.establish(BoundValidation::new(wrong.clone(), wrong.storage_places()));
        assert!(matches!(
            compose_validation_route(&contract, &call, &wrong_collection, 3),
            ValidationRoute::Conditional(_)
        ));
    }

    #[test]
    fn task6_unmet_routes_union_and_writes_prevent_rebasing() {
        let collection = PlaceKey::new("_1");
        let subject = PlaceKey::new("_2");
        let call = task6_call(&collection, &subject);
        let contract = task6_contract();

        let mut guarded = BlockState::empty();
        seed_exact(&mut guarded, &collection);
        seed_exact(&mut guarded, &subject);
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: subject.clone(),
            collection: Some(collection.clone()),
        };
        guarded.establish(BoundValidation::new(
            binding.clone(),
            binding.storage_places(),
        ));

        let mut unguarded = BlockState::empty();
        seed_exact(&mut unguarded, &collection);
        seed_exact(&mut unguarded, &subject);
        let routes = [
            compose_validation_route(&contract, &call, &guarded, 2),
            compose_validation_route(&contract, &call, &unguarded, 2),
        ];
        assert!(routes.contains(&ValidationRoute::Satisfied));
        assert!(routes
            .iter()
            .any(|route| matches!(route, ValidationRoute::Conditional(_))));

        unguarded.record_write(subject.clone(), StaticWriteToken::new(1, 1));
        assert!(matches!(
            compose_validation_route(&contract, &call, &unguarded, 2),
            ValidationRoute::Hard(_)
        ));
        assert_eq!(entry_formal_slot(2, &unguarded, &subject), None);
    }

    #[test]
    fn task6_resolved_local_mutable_actual_havocs_only_on_the_normal_edge() {
        let referent = PlaceKey::new("_2");
        let reference = PlaceKey::new("_3");
        let destination = PlaceKey::new("_4");
        let mut before = BlockState::empty();
        seed_exact(&mut before, &referent);
        before.set_value(
            reference.clone(),
            ValueFacts {
                value_flow: BTreeSet::from([referent.clone()]),
                exact_roots: BTreeSet::from([referent.clone()]),
                ..ValueFacts::default()
            },
        );
        let binding = ValidationBinding {
            predicate: Predicate::NonNull,
            subject: referent.clone(),
            collection: None,
        };
        before.establish(BoundValidation::new(
            binding.clone(),
            binding.storage_places(),
        ));
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::mutate")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand("_3")],
                destination,
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![reference],
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };

        let mut normal = before.clone();
        apply_call_side_effects(&call, &point(1, 0), &mut normal);
        assert!(state_value(&normal, &referent).havoced);
        assert!(!normal.has_current_validation(&binding, binding.storage_places()));
        assert!(before.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn length_joined_with_other_does_not_create_a_pending_bound() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_3");
        let condition = PlaceKey::new("_4");
        let mut state = BlockState::empty();
        state.set_value(
            length.clone(),
            ValueFacts {
                len_of: BTreeSet::from([collection]),
                may_be_non_length: true,
                ..ValueFacts::default()
            },
        );
        let operation = ProgramOp {
            point: point(0, 0),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: condition.clone(),
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
            }),
        };
        apply_program_op(&operation, BasicBlock::from_usize(0), 0, &mut state);
        assert!(state.pending_must().get(&condition).is_none());
    }

    #[test]
    fn stale_length_after_collection_write_cannot_establish_in_bounds() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_3");
        let length_op = ProgramOp {
            point: point(0, 0),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: length.clone(),
                value: ValueModel::Length(place_operand(&collection.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
            }),
        };
        let compare = |destination: PlaceKey| ProgramOp {
            point: point(0, 2),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination,
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
            }),
        };
        let mut fresh = BlockState::empty();
        seed_exact(&mut fresh, &collection);
        seed_exact(&mut fresh, &index);
        apply_program_op(&length_op, BasicBlock::from_usize(0), 0, &mut fresh);
        let fresh_condition = PlaceKey::new("_4");
        apply_program_op(
            &compare(fresh_condition.clone()),
            BasicBlock::from_usize(0),
            2,
            &mut fresh,
        );
        assert!(fresh.pending_must().contains_key(&fresh_condition));

        let mut stale = BlockState::empty();
        seed_exact(&mut stale, &collection);
        seed_exact(&mut stale, &index);
        apply_program_op(&length_op, BasicBlock::from_usize(0), 0, &mut stale);
        stale.record_may_write(canonical_storage(&collection), StaticWriteToken::new(0, 1));
        let stale_condition = PlaceKey::new("_5");
        apply_program_op(
            &compare(stale_condition.clone()),
            BasicBlock::from_usize(0),
            2,
            &mut stale,
        );
        assert!(!stale.pending_must().contains_key(&stale_condition));

        let first = StaticWriteToken::new(1, 0);
        let second = StaticWriteToken::new(2, 0);
        let mut merged_length = ValueFacts {
            len_of: BTreeSet::from([collection.clone()]),
            length_atoms: BTreeSet::from([LengthAtom {
                collection: collection.clone(),
                definition: StaticWriteToken::new(0, 0),
                captured_versions: BTreeSet::from([first]),
                invalidated: false,
            }]),
            ..ValueFacts::default()
        };
        merged_length.join_may(&ValueFacts {
            len_of: BTreeSet::from([collection.clone()]),
            length_atoms: BTreeSet::from([LengthAtom {
                collection: collection.clone(),
                definition: StaticWriteToken::new(0, 9),
                captured_versions: BTreeSet::from([second]),
                invalidated: false,
            }]),
            ..ValueFacts::default()
        });
        let mut conflated = BlockState::empty();
        conflated.record_may_write(canonical_storage(&collection), first);
        conflated.record_may_write(canonical_storage(&collection), second);
        conflated.set_value(length.clone(), merged_length);
        let conflated_condition = PlaceKey::new("_6");
        apply_program_op(
            &compare(conflated_condition.clone()),
            BasicBlock::from_usize(0),
            2,
            &mut conflated,
        );
        assert!(!conflated.pending_must().contains_key(&conflated_condition));
    }

    #[test]
    fn length_transfer_is_monotone_for_one_static_definition() {
        let collection = PlaceKey::new("_1");
        let definition = StaticWriteToken::new(0, 2);
        let first = StaticWriteToken::new(1, 0);
        let second = StaticWriteToken::new(2, 0);
        let mut small = BlockState::empty();
        seed_exact(&mut small, &collection);
        small.record_may_write(canonical_storage(&collection), first);
        let mut expanded = small.clone();
        expanded.record_may_write(canonical_storage(&collection), second);
        let model = ValueModel::Length(place_operand(&collection.0));
        let small_output = eval_value_model(&small, &model, definition);
        let expanded_output = eval_value_model(&expanded, &model, definition);
        let mut joined = small_output;
        joined.join_may(&expanded_output);
        assert_eq!(joined, expanded_output);
    }

    #[test]
    fn loop_reused_write_token_leaves_a_joinable_length_tombstone() {
        let collection = PlaceKey::new("_1");
        let length = PlaceKey::new("_3");
        let definition = StaticWriteToken::new(0, 0);
        let write = StaticWriteToken::new(1, 0);
        let mut entry = BlockState::empty();
        seed_exact(&mut entry, &collection);
        entry.record_may_write(canonical_storage(&collection), write);
        entry.set_value(
            length.clone(),
            eval_value_model(
                &entry,
                &ValueModel::Length(place_operand(&collection.0)),
                definition,
            ),
        );
        let unchanged = entry.clone();
        let mut written = entry.clone();
        written.record_may_write(canonical_storage(&collection), write);
        invalidate_length_evidence(&mut written, &canonical_storage(&collection));
        assert!(written.may_values()[&length]
            .length_atoms
            .iter()
            .all(|atom| atom.invalidated));
        let joined = BlockState::join_predecessors([&unchanged, &written]).unwrap();
        let joined_length = &joined.may_values()[&length];
        assert!(joined_length
            .length_atoms
            .iter()
            .all(|atom| atom.invalidated));
        assert!(!length_relation_is_current(
            &joined,
            joined_length,
            &collection
        ));
    }

    #[test]
    fn recomputing_length_after_havoc_restores_only_the_fresh_observation() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_3");
        let condition = PlaceKey::new("_4");
        let mut state = BlockState::empty();
        state.record_may_write(canonical_storage(&collection), StaticWriteToken::new(0, 0));
        state.set_value(
            collection.clone(),
            ValueFacts {
                havoced: true,
                exact_roots: BTreeSet::from([collection.clone()]),
                ..ValueFacts::default()
            },
        );
        seed_exact(&mut state, &index);
        let length_op = ProgramOp {
            point: point(0, 1),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: length.clone(),
                value: ValueModel::Length(place_operand(&collection.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
            }),
        };
        apply_program_op(&length_op, BasicBlock::from_usize(0), 1, &mut state);
        assert!(!state.may_values()[&length].havoced);
        let compare = ProgramOp {
            point: point(0, 2),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: condition.clone(),
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
            }),
        };
        apply_program_op(&compare, BasicBlock::from_usize(0), 2, &mut state);
        assert!(state.pending_must().contains_key(&condition));
        state.record_may_write(canonical_storage(&collection), StaticWriteToken::new(0, 3));
        invalidate_length_evidence(&mut state, &canonical_storage(&collection));
        let stale_condition = PlaceKey::new("_5");
        let mut stale_compare = compare;
        if let ProgramOpKind::Assign(assignment) = &mut stale_compare.kind {
            assignment.destination = stale_condition.clone();
        }
        apply_program_op(&stale_compare, BasicBlock::from_usize(0), 4, &mut state);
        assert!(!state.pending_must().contains_key(&stale_condition));
    }

    #[test]
    fn length_atom_join_is_commutative_associative_idempotent_and_keyed_by_definition() {
        let collection = PlaceKey::new("_1");
        let facts = |definition, version| ValueFacts {
            len_of: BTreeSet::from([collection.clone()]),
            length_atoms: BTreeSet::from([LengthAtom {
                collection: collection.clone(),
                definition,
                captured_versions: BTreeSet::from([version]),
                invalidated: false,
            }]),
            ..ValueFacts::default()
        };
        let first = facts(StaticWriteToken::new(0, 0), StaticWriteToken::new(1, 0));
        let second = facts(StaticWriteToken::new(0, 0), StaticWriteToken::new(2, 0));
        let third = facts(StaticWriteToken::new(0, 9), StaticWriteToken::new(3, 0));
        let join = |values: &[ValueFacts]| {
            let mut result = ValueFacts::default();
            for value in values {
                result.join_may(value);
            }
            result
        };
        assert_eq!(
            join(&[first.clone(), second.clone()]),
            join(&[second.clone(), first.clone()])
        );
        assert_eq!(
            join(&[join(&[first.clone(), second.clone()]), third.clone()]),
            join(&[first.clone(), join(&[second.clone(), third.clone()])])
        );
        assert_eq!(join(&[first.clone(), first.clone()]), first);
        assert_eq!(join(&[second, third]).length_atoms.len(), 2);
    }

    #[test]
    fn copied_condition_preserves_current_pending_and_write_kills_it() {
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: PlaceKey::new("_2"),
            collection: Some(PlaceKey::new("_1")),
        };
        let source = PlaceKey::new("_3");
        let copied = PlaceKey::new("_4");
        let mut state = BlockState::empty();
        state.set_pending(
            source.clone(),
            [(binding.clone(), false, binding.storage_places())],
        );
        let operation = ProgramOp {
            point: point(0, 1),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: copied.clone(),
                value: ValueModel::Operand(place_operand(&source.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
            }),
        };
        apply_program_op(&operation, BasicBlock::from_usize(0), 1, &mut state);
        assert!(state.pending_is_current(&copied, &binding, false));
        state.record_write(PlaceKey::new("_2"), StaticWriteToken::new(0, 2));
        assert!(!state.pending_is_current(&copied, &binding, false));
    }

    #[test]
    fn value_models_preserve_operand_ref_cast_length_and_field_origin() {
        let source = PlaceKey::new("_1");
        let mut state = BlockState::empty();
        state.set_value(
            source.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                value_flow: BTreeSet::from([source.clone()]),
                exact_roots: BTreeSet::from([source.clone()]),
                may_be_non_length: true,
                ..ValueFacts::default()
            },
        );
        let field = OperandModel {
            operand: CallOperand::Place(source.clone()),
            field_origin: Some(AbstractOrigin::PublicField {
                def_path: "crate::S::field".into(),
                span: span(),
            }),
            pure_deref_base: None,
            exact_projection: None,
            projection_unknown: false,
        };
        let definition = StaticWriteToken::new(0, 0);
        let copied = eval_value_model(&state, &ValueModel::Operand(field.clone()), definition);
        let referenced = eval_value_model(&state, &ValueModel::RefOrRaw(field.clone()), definition);
        assert_eq!(copied.value, referenced.value);
        assert!(copied
            .value
            .origins
            .iter()
            .any(|origin| matches!(origin, AbstractOrigin::PublicField { .. })));
        let length = eval_value_model(&state, &ValueModel::Length(field), definition);
        assert_eq!(length.len_of, BTreeSet::from([source]));
        assert!(!length.may_be_non_length);
    }

    #[test]
    fn copied_guard_paths_and_sink_share_only_one_semantic_storage_binding() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let expected = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: index.clone(),
            collection: Some(collection.clone()),
        };
        let guarded_path =
            |slice_temp: &str, index_temp: &str, length: &str, condition: &str, block: usize| {
                let slice_temp = PlaceKey::new(slice_temp);
                let index_temp = PlaceKey::new(index_temp);
                let length = PlaceKey::new(length);
                let condition = PlaceKey::new(condition);
                let mut state = BlockState::empty();
                state.set_value(
                    slice_temp.clone(),
                    ValueFacts {
                        value_flow: BTreeSet::from([collection.clone(), PlaceKey::new("(*_1)")]),
                        exact_roots: BTreeSet::from([collection.clone()]),
                        ..ValueFacts::default()
                    },
                );
                state.set_value(
                    index_temp.clone(),
                    ValueFacts {
                        value_flow: BTreeSet::from([index.clone()]),
                        exact_roots: BTreeSet::from([index.clone()]),
                        ..ValueFacts::default()
                    },
                );
                apply_program_op(
                    &ProgramOp {
                        point: point(block as u32, 0),
                        kind: ProgramOpKind::Assign(AssignmentModel {
                            destination: length.clone(),
                            value: ValueModel::Length(place_operand(&slice_temp.0)),
                            field_origin: None,
                            raw_read_source: None,
                            legacy_write: false,
                        }),
                    },
                    BasicBlock::from_usize(block),
                    0,
                    &mut state,
                );
                apply_program_op(
                    &ProgramOp {
                        point: point(block as u32, 1),
                        kind: ProgramOpKind::Assign(AssignmentModel {
                            destination: condition.clone(),
                            value: ValueModel::Compare {
                                op: BinOp::Ge,
                                left: place_operand(&index_temp.0),
                                right: place_operand(&length.0),
                            },
                            field_origin: None,
                            raw_read_source: None,
                            legacy_write: false,
                        }),
                    },
                    BasicBlock::from_usize(block),
                    1,
                    &mut state,
                );
                let false_target = BasicBlock::from_usize(9);
                apply_program_edge(
                    &EdgeTerm::Switch {
                        condition,
                        false_target,
                        true_target: BasicBlock::from_usize(10),
                    },
                    false_target,
                    &mut state,
                );
                assert!(state.has_current_validation(&expected, expected.storage_places()));
                state
            };

        let first = guarded_path("_9", "_7", "_8", "_6", 1);
        let second = guarded_path("_14", "_12", "_13", "_11", 5);
        let mut sink = BlockState::join_predecessors([&first, &second]).unwrap();
        sink.set_value(
            PlaceKey::new("_17"),
            ValueFacts {
                value_flow: BTreeSet::from([collection.clone(), PlaceKey::new("(*_1)")]),
                exact_roots: BTreeSet::from([collection.clone()]),
                ..ValueFacts::default()
            },
        );
        sink.set_value(
            PlaceKey::new("_18"),
            ValueFacts {
                value_flow: BTreeSet::from([index.clone()]),
                exact_roots: BTreeSet::from([index.clone()]),
                ..ValueFacts::default()
            },
        );
        let sink_binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: unique_semantic_value(&semantic_value_at(&sink, &PlaceKey::new("_18")))
                .unwrap(),
            collection: Some(
                unique_semantic_value(&semantic_value_at(&sink, &PlaceKey::new("_17"))).unwrap(),
            ),
        };
        assert_eq!(sink_binding, expected);
        assert!(state_proves_local_binding(&sink, Some(&sink_binding)));

        sink.set_value(
            PlaceKey::new("_19"),
            ValueFacts {
                value_flow: BTreeSet::from([index, PlaceKey::new("_3")]),
                exact_roots: BTreeSet::from([PlaceKey::new("_2"), PlaceKey::new("_3")]),
                ..ValueFacts::default()
            },
        );
        assert!(unique_semantic_value(&semantic_value_at(&sink, &PlaceKey::new("_19"))).is_none());
    }

    #[test]
    fn semantic_candidates_are_monotone_and_keep_sibling_projections_distinct() {
        let local = PlaceKey::new("_3");
        let first_field = PlaceKey::new("(_1.0: usize)");
        let second_field = PlaceKey::new("(_1.1: usize)");
        let mut small = BlockState::empty();
        small.set_value(
            local.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([first_field.clone()]),
                ..ValueFacts::default()
            },
        );
        let mut expanded = small.clone();
        expanded.set_value(
            local.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([first_field.clone(), second_field.clone()]),
                ..ValueFacts::default()
            },
        );
        let model = ValueModel::Length(place_operand(&local.0));
        let small_output = eval_value_model(&small, &model, StaticWriteToken::new(0, 0));
        let expanded_output = eval_value_model(&expanded, &model, StaticWriteToken::new(0, 0));
        let mut joined = small_output;
        joined.join_may(&expanded_output);
        assert_eq!(joined, expanded_output);
        assert_eq!(expanded_output.len_of.len(), 2);
        assert!(unique_semantic_value(&semantic_value_at(&expanded, &local)).is_none());

        let projection = |place: PlaceKey| OperandModel {
            operand: CallOperand::Place(place.clone()),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: Some(place),
            projection_unknown: false,
        };
        let empty = BlockState::empty();
        assert_eq!(
            unique_semantic_value(&eval_operand(&empty, &projection(first_field.clone()))),
            Some(first_field)
        );
        assert_eq!(
            unique_semantic_value(&eval_operand(&empty, &projection(second_field.clone()))),
            Some(second_field)
        );
        assert!(unique_semantic_value(&semantic_value_at(&empty, &local)).is_none());
    }

    #[test]
    fn semantic_unknown_preserves_length_atoms_and_only_reborrow_propagates_root() {
        let source = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_4");
        let deref = PlaceKey::new("(*_4)");
        let definition = StaticWriteToken::new(0, 0);
        let mut known = BlockState::empty();
        known.set_value(source.clone(), exact_value(&source));
        let known_length = eval_value_model(
            &known,
            &ValueModel::Length(place_operand(&source.0)),
            definition,
        );
        let mut expanded = known.clone();
        expanded.set_value(
            source.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([source.clone()]),
                semantic_unknown: true,
                ..ValueFacts::default()
            },
        );
        let expanded_length = eval_value_model(
            &expanded,
            &ValueModel::Length(place_operand(&source.0)),
            definition,
        );
        let mut joined = known_length;
        joined.join_may(&expanded_length);
        assert_eq!(joined, expanded_length);
        assert_eq!(expanded_length.len_of, BTreeSet::from([source.clone()]));
        assert!(unique_semantic_value(&expanded_length).is_none());

        let mut pointer_state = BlockState::empty();
        pointer_state.set_value(pointer.clone(), exact_value(&source));
        let ordinary_load = OperandModel {
            operand: CallOperand::Place(deref.clone()),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: Some(deref.clone()),
            projection_unknown: false,
        };
        assert_eq!(
            unique_semantic_value(&eval_operand(&pointer_state, &ordinary_load)),
            Some(deref)
        );
        let reborrow = OperandModel {
            operand: ordinary_load.operand,
            field_origin: None,
            pure_deref_base: Some(pointer),
            exact_projection: None,
            projection_unknown: false,
        };
        assert_eq!(
            unique_semantic_value(&eval_operand(&pointer_state, &reborrow)),
            Some(source)
        );

        let dynamic_index = OperandModel {
            operand: CallOperand::Place(PlaceKey::new("_6[_7]")),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: None,
            projection_unknown: true,
        };
        assert!(unique_semantic_value(&eval_operand(&pointer_state, &dynamic_index)).is_none());
    }

    #[test]
    fn cast_and_slice_prefix_are_derived_not_exact_equivalences() {
        let source = PlaceKey::new("_1");
        let range = PlaceKey::new("_2");
        let mut state = BlockState::empty();
        state.set_value(source.clone(), exact_value(&source));
        state.set_value(range.clone(), exact_value(&range));
        let cast = eval_value_model(&state, &ValueModel::Unknown, StaticWriteToken::new(0, 0));
        assert!(unique_semantic_value(&cast).is_none());
        let prefix = eval_registry_value(
            &state,
            &RegistryValueModel::SlicePrefix {
                base: place_operand(&source.0),
                range,
            },
            &point(0, 1),
        );
        assert!(unique_semantic_value(&prefix).is_none());
    }

    #[test]
    fn whitelisted_unsize_temporaries_converge_but_other_casts_remain_unknown() {
        let field = PlaceKey::new("((*_1).1: [u8; 8])");
        let first_ref = PlaceKey::new("_6");
        let first_slice = PlaceKey::new("_5");
        let second_ref = PlaceKey::new("_10");
        let second_slice = PlaceKey::new("_9");
        let mut state = BlockState::empty();
        state.set_value(field.clone(), exact_value(&field));
        for (reference, slice) in [
            (first_ref.clone(), first_slice.clone()),
            (second_ref.clone(), second_slice.clone()),
        ] {
            state.set_value(
                reference.clone(),
                eval_value_model(
                    &state,
                    &ValueModel::RefOrRaw(place_operand(&field.0)),
                    StaticWriteToken::new(0, 0),
                ),
            );
            // The production normalizer maps only PointerCoercion::Unsize to
            // this identity-preserving Operand form.
            state.set_value(
                slice,
                eval_value_model(
                    &state,
                    &ValueModel::Operand(place_operand(&reference.0)),
                    StaticWriteToken::new(0, 1),
                ),
            );
        }
        assert_eq!(
            unique_semantic_value(&semantic_value_at(&state, &first_slice)),
            Some(field.clone())
        );
        assert_eq!(
            unique_semantic_value(&semantic_value_at(&state, &second_slice)),
            Some(field)
        );
        assert!(unique_semantic_value(&eval_value_model(
            &state,
            &ValueModel::Unknown,
            StaticWriteToken::new(0, 2),
        ))
        .is_none());
    }

    #[test]
    fn switch_refines_only_the_matching_edge_and_exact_binding() {
        let condition = PlaceKey::new("_3");
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: PlaceKey::new("_2"),
            collection: Some(PlaceKey::new("_1")),
        };
        let false_target = BasicBlock::from_usize(1);
        let true_target = BasicBlock::from_usize(2);
        let edge = EdgeTerm::Switch {
            condition: condition.clone(),
            false_target,
            true_target,
        };
        let mut false_state = BlockState::empty();
        false_state.set_pending(
            condition.clone(),
            [(binding.clone(), false, binding.storage_places())],
        );
        apply_program_edge(&edge, false_target, &mut false_state);
        assert!(false_state.has_current_validation(&binding, binding.storage_places()));
        let wrong = ValidationBinding {
            subject: PlaceKey::new("_9"),
            ..binding.clone()
        };
        assert!(!false_state.has_current_validation(&wrong, wrong.storage_places()));
        let mut true_state = BlockState::empty();
        true_state.set_pending(
            condition,
            [(binding.clone(), false, binding.storage_places())],
        );
        apply_program_edge(&edge, true_target, &mut true_state);
        assert!(!true_state.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn assert_plumbing_does_not_establish_task9_validation() {
        let condition = PlaceKey::new("_3");
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: PlaceKey::new("_2"),
            collection: Some(PlaceKey::new("_1")),
        };
        let target = BasicBlock::from_usize(1);
        let edge = EdgeTerm::AssertPlumbingOnly {
            condition: condition.clone(),
            expected: true,
            target,
        };
        let mut state = BlockState::empty();
        state.set_pending(
            condition,
            [(binding.clone(), true, binding.storage_places())],
        );
        let before = state.clone();
        apply_program_edge(&edge, target, &mut state);
        assert_eq!(state, before);
        assert!(!state.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn call_effects_are_applied_only_on_the_normal_edge() {
        let normal = BasicBlock::from_usize(1);
        let unwind = BasicBlock::from_usize(2);
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::OpaqueIndirect,
                args: Vec::new(),
                destination: PlaceKey::new("_0"),
            },
            value: RegistryValueModel::Constant(7),
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: point(0, 0),
            call,
        };
        let mut unwind_state = BlockState::empty();
        apply_program_edge(&edge, unwind, &mut unwind_state);
        assert!(!unwind_state.may_values().contains_key(&PlaceKey::new("_0")));
        let mut normal_state = BlockState::empty();
        apply_program_edge(&edge, normal, &mut normal_state);
        assert!(normal_state.may_values().contains_key(&PlaceKey::new("_0")));
    }

    #[test]
    fn resolved_local_empty_return_does_not_copy_actual_provenance() {
        let actual = PlaceKey::new("_1");
        let destination = PlaceKey::new("_0");
        let normal = BasicBlock::from_usize(1);
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::callee")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand(&actual.0)],
                destination: destination.clone(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: point(0, 0),
            call,
        };
        let mut state = BlockState::empty();
        state.set_value(
            actual.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                ..ValueFacts::default()
            },
        );
        apply_program_edge(&edge, normal, &mut state);
        assert!(state.may_values()[&destination].value.origins.is_empty());
        assert!(state.may_values()[&actual]
            .value
            .origins
            .contains(&AbstractOrigin::Formal(1)));
    }

    #[test]
    fn modeled_call_result_and_ffi_output_have_fresh_exact_storage_identities() {
        let result = PlaceKey::new("_3");
        let normal = BasicBlock::from_usize(1);
        let modeled = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::SelectedOpenBehavior,
                args: Vec::new(),
                destination: result.clone(),
            },
            value: RegistryValueModel::SelectedOpenBehavior {
                token: "trait-return".into(),
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let mut state = BlockState::empty();
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: modeled,
            },
            normal,
            &mut state,
        );
        assert_eq!(
            unique_semantic_value(&semantic_value_at(&state, &result)),
            Some(result.clone())
        );

        let actual = PlaceKey::new("_5");
        let output = PlaceKey::new("_1");
        state.set_value(
            actual.clone(),
            ValueFacts {
                value_flow: BTreeSet::from([output.clone()]),
                ..ValueFacts::default()
            },
        );
        let ffi = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ExactFfiOut,
                args: Vec::new(),
                destination: PlaceKey::new("_0"),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: vec![actual],
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        apply_call_side_effects(&ffi, &point(0, 1), &mut state);
        assert_eq!(
            unique_semantic_value(&semantic_value_at(&state, &output)),
            Some(output.clone())
        );
        let condition = PlaceKey::new("_6");
        let is_null = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&output.0)],
                destination: condition.clone(),
            },
            value: RegistryValueModel::Empty,
            predicate: Some(PredicateModel::IsNull {
                pointer: output.clone(),
            }),
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: true,
            access_width: None,
            legacy_write: false,
        };
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 2),
                call: is_null,
            },
            normal,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::NonNull,
            subject: output,
            collection: None,
        };
        assert!(state.pending_is_current(&condition, &binding, false));
    }

    #[test]
    fn generic_result_identity_allows_is_empty_to_establish_nonempty() {
        let slice = PlaceKey::new("_3");
        let condition = PlaceKey::new("_4");
        let normal = BasicBlock::from_usize(1);
        let generic = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: Vec::new(),
                destination: slice.clone(),
            },
            value: RegistryValueModel::GenericCapability {
                token: "associated-slice".into(),
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let mut state = BlockState::empty();
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: generic,
            },
            normal,
            &mut state,
        );
        let is_empty = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&slice.0)],
                destination: condition.clone(),
            },
            value: RegistryValueModel::Empty,
            predicate: Some(PredicateModel::IsEmpty {
                slice: slice.clone(),
            }),
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: true,
            access_width: None,
            legacy_write: false,
        };
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 1),
                call: is_empty,
            },
            normal,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::NonEmpty,
            subject: slice,
            collection: None,
        };
        assert!(state.pending_is_current(&condition, &binding, false));
    }

    #[test]
    fn selected_open_index_identity_allows_length_guard_pending() {
        let data = PlaceKey::new("_1");
        let index = PlaceKey::new("_3");
        let length = PlaceKey::new("_4");
        let condition = PlaceKey::new("_5");
        let normal = BasicBlock::from_usize(1);
        let selected = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::SelectedOpenBehavior,
                args: Vec::new(),
                destination: index.clone(),
            },
            value: RegistryValueModel::SelectedOpenBehavior {
                token: "trait-index".into(),
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let mut state = BlockState::empty();
        seed_exact(&mut state, &data);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: selected,
            },
            normal,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                point: point(1, 0),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length.clone(),
                    value: ValueModel::Length(place_operand(&data.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                }),
            },
            normal,
            0,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                point: point(1, 1),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    value: ValueModel::Compare {
                        op: BinOp::Ge,
                        left: place_operand(&index.0),
                        right: place_operand(&length.0),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                }),
            },
            normal,
            1,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: index,
            collection: Some(data),
        };
        assert!(state.pending_is_current(&condition, &binding, false));
    }

    #[test]
    fn modeled_length_keeps_relation_unknown_while_defining_its_destination() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_3");
        let normal = BasicBlock::from_usize(1);
        let length_call = || CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&collection.0)],
                destination: length.clone(),
            },
            value: RegistryValueModel::Length(place_operand(&collection.0)),
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let compare = |destination: PlaceKey| ProgramOp {
            point: point(1, 1),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination,
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
            }),
        };

        let mut known = BlockState::empty();
        seed_exact(&mut known, &collection);
        seed_exact(&mut known, &index);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: length_call(),
            },
            normal,
            &mut known,
        );
        assert_eq!(
            unique_semantic_value(&semantic_value_at(&known, &length)),
            Some(length.clone())
        );
        let known_condition = PlaceKey::new("_4");
        apply_program_op(&compare(known_condition.clone()), normal, 1, &mut known);
        assert!(known.pending_must().contains_key(&known_condition));

        let mut unknown = BlockState::empty();
        unknown.set_value(
            collection.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([collection.clone()]),
                semantic_unknown: true,
                ..ValueFacts::default()
            },
        );
        seed_exact(&mut unknown, &index);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: length_call(),
            },
            normal,
            &mut unknown,
        );
        let known_length = &known.may_values()[&length];
        let unknown_length = &unknown.may_values()[&length];
        let mut joined = known_length.clone();
        joined.join_may(unknown_length);
        assert_eq!(&joined, unknown_length);
        assert!(unknown_length.length_relation_unknown);
        assert_eq!(
            unique_semantic_value(&semantic_value_at(&unknown, &length)),
            Some(length.clone())
        );
        let unknown_condition = PlaceKey::new("_5");
        apply_program_op(&compare(unknown_condition.clone()), normal, 1, &mut unknown);
        assert!(!unknown.pending_must().contains_key(&unknown_condition));
    }

    #[test]
    fn ffi_alias_expansion_only_grows_identity_and_versions() {
        let actual = PlaceKey::new("_5");
        let first = PlaceKey::new("_1");
        let second = PlaceKey::new("_2");
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ExactFfiOut,
                args: Vec::new(),
                destination: PlaceKey::new("_0"),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: vec![actual.clone()],
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let mut small = BlockState::empty();
        small.set_value(
            actual.clone(),
            ValueFacts {
                value_flow: BTreeSet::from([first.clone()]),
                ..ValueFacts::default()
            },
        );
        small.set_value(first.clone(), exact_value(&first));
        let mut expanded = small.clone();
        expanded.set_value(
            actual,
            ValueFacts {
                value_flow: BTreeSet::from([first.clone(), second.clone()]),
                ..ValueFacts::default()
            },
        );
        expanded.set_value(
            first.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([first.clone(), PlaceKey::new("_9")]),
                semantic_unknown: true,
                ..ValueFacts::default()
            },
        );
        expanded.set_value(second.clone(), exact_value(&second));
        apply_call_side_effects(&call, &point(0, 3), &mut small);
        apply_call_side_effects(&call, &point(0, 3), &mut expanded);
        for (place, small_value) in small.may_values() {
            let expanded_value = &expanded.may_values()[place];
            let mut joined = small_value.clone();
            joined.join_may(expanded_value);
            assert_eq!(&joined, expanded_value);
        }
        for (place, versions) in small.versions() {
            assert!(expanded.versions()[place].is_superset(versions));
        }
        assert!(expanded.may_values()[&first].semantic_unknown);
        assert!(expanded.may_values()[&first]
            .exact_roots
            .is_superset(&small.may_values()[&first].exact_roots));
        assert!(expanded.versions().contains_key(&second));
    }

    #[test]
    fn authoritative_local_binding_rejects_missing_and_wrong_bindings() {
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject: PlaceKey::new("_2"),
            collection: Some(PlaceKey::new("_1")),
        };
        let mut state = BlockState::empty();
        state.establish(BoundValidation::new(
            binding.clone(),
            binding.storage_places(),
        ));
        assert!(state_proves_local_binding(&state, Some(&binding)));
        assert!(!state_proves_local_binding(&state, None));
        let wrong = ValidationBinding {
            subject: PlaceKey::new("_9"),
            ..binding
        };
        assert!(!state_proves_local_binding(&state, Some(&wrong)));
    }

    #[test]
    fn edge_bound_write_only_reaches_its_normal_continuation() {
        let successors = BTreeMap::from([
            (0, BTreeSet::from([1, 2])),
            (1, BTreeSet::from([3])),
            (2, BTreeSet::from([4])),
            (3, BTreeSet::new()),
            (4, BTreeSet::new()),
        ]);
        let write = WriteFact {
            place: PlaceKey::new("_1"),
            point: point(0, 0),
            kind: WriteKind::CallReturn,
            normal_successor: Some(BasicBlock::from_usize(1)),
        };
        assert!(write_normal_edge_reaches(
            &successors,
            &write,
            BasicBlock::from_usize(3)
        ));
        assert!(!write_normal_edge_reaches(
            &successors,
            &write,
            BasicBlock::from_usize(4)
        ));
    }

    #[test]
    fn opaque_may_alias_write_only_grows_value_and_versions() {
        let actual = PlaceKey::new("_1");
        let referent = PlaceKey::new("_2");
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::OpaqueDirect,
                args: Vec::new(),
                destination: PlaceKey::new("_0"),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![actual.clone()],
            ffi_out_actuals: Vec::new(),
            destination_is_bool: false,
            access_width: None,
            legacy_write: false,
        };
        let mut state = BlockState::empty();
        state.set_value(
            actual,
            ValueFacts {
                value_flow: BTreeSet::from([referent.clone()]),
                ..ValueFacts::default()
            },
        );
        state.set_value(
            referent.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                ..ValueFacts::default()
            },
        );
        apply_call_side_effects(&call, &point(0, 3), &mut state);
        let result = state.may_values().get(&referent).unwrap();
        assert!(result.havoced);
        assert!(result.value.origins.contains(&AbstractOrigin::Formal(1)));
        assert!(state.versions()[&referent].contains(&StaticWriteToken::new(0, 3)));

        let mut small = BlockState::empty();
        small.set_value(PlaceKey::new("_1"), ValueFacts::default());
        let mut expanded = small.clone();
        expanded.set_value(
            PlaceKey::new("_1"),
            ValueFacts {
                value_flow: BTreeSet::from([PlaceKey::new("_2")]),
                ..ValueFacts::default()
            },
        );
        apply_call_side_effects(&call, &point(0, 3), &mut small);
        apply_call_side_effects(&call, &point(0, 3), &mut expanded);
        for (place, small_value) in small.may_values() {
            let expanded_value = &expanded.may_values()[place];
            let mut joined = small_value.clone();
            joined.join_may(expanded_value);
            assert_eq!(&joined, expanded_value);
        }
        for (place, small_versions) in small.versions() {
            assert!(expanded.versions()[place].is_superset(small_versions));
        }
        assert!(expanded
            .versions()
            .get(&PlaceKey::new("_2"))
            .is_some_and(|versions| versions.contains(&StaticWriteToken::new(0, 3))));
    }

    #[test]
    fn projected_value_cannot_hide_canonical_havoc() {
        let storage = PlaceKey::new("_1");
        let projection = PlaceKey::new("_1.0");
        let mut small = BlockState::empty();
        small.set_value(
            storage,
            ValueFacts {
                havoced: true,
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                ..ValueFacts::default()
            },
        );
        let mut expanded = small.clone();
        expanded.set_value(
            projection.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Constant(7)]),
                ..ValueFacts::default()
            },
        );
        let small_value = state_value(&small, &projection);
        let expanded_value = state_value(&expanded, &projection);
        assert!(expanded_value.havoced);
        let mut joined = small_value;
        joined.join_may(&expanded_value);
        assert_eq!(joined, expanded_value);
        assert!(expanded_value
            .value
            .origins
            .contains(&AbstractOrigin::Constant(7)));
        assert!(expanded_value
            .value
            .origins
            .contains(&AbstractOrigin::Formal(1)));
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
