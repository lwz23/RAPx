use super::candidate::{
    build_structural_candidate_set_with_context_traced, reachable_public_roots, CallRouteIndex,
    CandidateCallEdge, CandidateCallKind, CandidateOwner, StructuralCandidateSet,
    StructuralFindingCandidate, SuppressedStructuralCandidate, UnestablishedStructuralCandidate,
};
use super::contract_catalog::{
    structural_candidate, structural_candidate_with_contract, ContractCatalog, ContractDescriptor,
    DischargeModel, KnownContractModel, StaticContractGap, StaticContractImplication,
    StructuralSiteCandidate,
};
use super::dataflow::{
    replay_final_entries, solve_cfg, BlockState, BoundValidation, DataflowResult, MayJoin,
    StaticWriteToken,
};
use super::endpoint_flow::{
    AccessExtent, BehaviorController, BehaviorOpenness, CallSiteId, CallSiteOutput,
    CapabilityRequirement, CapabilityTerm, CapabilityTermForm, CapabilityTermRole,
    CapabilityTypeFacts, CapabilityUnestablishedReason, CapabilityViolation, CapabilityWitness,
    ContextActualBinding, ContextProofIndex, ContextRelationKey, ContextUnknownReason,
    DerivationKind, EndpointEdge, EndpointEdgeKind, EndpointFlowGraph, EndpointNode,
    ExternalSourceKind, FiniteTypeWitness, GenericArgBinding, GenericCallSubstitution,
    GenericCapabilityEvidence, GenericCapabilityKind, GenericParamId, GenericParamKind,
    GenericSubstitutionStatus, OpenBehaviorReceiver, PublicFieldAdtKind, StorageProvenance,
    TypedWitnessTransportTemplate,
};
use super::influence::{
    InfluenceCapabilities, InfluenceController, InfluenceOrigin, InfluenceSeedIndex,
    InfluenceToken, InfluenceTypeClass, UnestablishedReason,
};
use super::missing_sanitizer::{classify_sink, MissingSanitizerWarning, WarningPredicate};
use super::perf_trace::PerfTrace;
use super::relations::{
    ProjectionKey, RelationFact, RelationKind, RelationProof, RelationTerm,
    RelationalPostcondition, ReturnCondition, SummaryRelationKind, SummaryRelationTerm,
};
use super::summary::{
    classify_primary_with_secondary, map_call_outputs, out_values_from_dependencies,
    solve_summaries, AbstractOrigin, AbstractValue, BoundarySlot, CallBoundary, CallMapping,
    CanonicalWitness, ContractRequirement, FailureClass, FunctionKey, FunctionSummary, Operation,
    OperationKind, OutDependency, Pattern, PlaceKey, Predicate, ProgramPoint, RuleId,
    SelectedUnresolvedRoute, SourceKind, StablePosition, StableSpan, UnresolvedPredecessor,
    UnresolvedRouteCycle, UnresolvedRouteFact, UnresolvedRouteNode, ValidationFact, WitnessStep,
    WitnessStepKind, WriteEffect,
};
use super::unsafe_site::{
    self,
    binding::{
        bind_owner, enumerate_mir_events, BoundUnsafeSite, MirEventAnchor, MirEventLocation,
        MirOperandAnchor, MirStage, MirTypeValidity, SiteBindingRecord, TypeValidityClass,
    },
    BindingStatus, CoverageFailure, CoverageFailureKind,
};
#[rustversion::since(1.96)]
use rustc_abi::ExternAbi;
#[rustversion::since(1.96)]
use rustc_attr_ir::lang_items::LangItem;
#[rustversion::since(1.96)]
use rustc_hir::ConstContext;
#[rustversion::before(1.96)]
use rustc_hir::LangItem;
#[rustversion::before(1.96)]
use rustc_hir::WherePredicate;
#[rustversion::since(1.96)]
use rustc_hir::WherePredicateKind;
use rustc_hir::{
    def::DefKind,
    def_id::{DefId, LocalDefId},
    GenericBound, Mutability, QPath, Safety, TyKind,
};
use rustc_index::IndexVec;
use rustc_infer::infer::TyCtxtInferExt;
#[rustversion::before(1.96)]
use rustc_middle::mir::tcx::PlaceTy;
#[rustversion::since(1.96)]
use rustc_middle::mir::PlaceTy;
use rustc_middle::{
    middle::{codegen_fn_attrs::CodegenFnAttrFlags, privacy::Level},
    mir::{
        AggregateKind, BasicBlock, BinOp, Body, CastKind, Const as MirConst, Location, Operand,
        Place, ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp, RETURN_PLACE,
    },
    ty::{self, Instance, InstanceKind, Ty, TyCtxt, TypeVisitableExt, Visibility},
};
#[rustversion::before(1.96)]
use rustc_span::source_map::Spanned;
use rustc_span::{sym, Span};
#[rustversion::before(1.96)]
use rustc_span::{FileName, FileNameDisplayPreference};
#[rustversion::since(1.96)]
use rustc_span::{FileName, Spanned, Symbol};
#[rustversion::before(1.96)]
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::infer::InferCtxtExt;
use std::collections::{btree_map::Entry, BTreeMap, BTreeSet};
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const CANDIDATE_LIFECYCLE_TRACE_ENV: &str = "UNSOUND_SCANNER_CANDIDATE_LIFECYCLE_TRACE";

type CandidateLifecycleEvent = BTreeMap<String, serde_json::Value>;

fn candidate_lifecycle_event(
    stage: &str,
    owner: &FunctionKey,
    requirement: &ContractRequirement,
) -> CandidateLifecycleEvent {
    BTreeMap::from([
        ("stage".into(), stage.into()),
        ("owner".into(), owner.0.clone().into()),
        ("rule_id".into(), requirement.rule.as_str().into()),
        ("seed_id".into(), requirement.seed_id.clone().into()),
        (
            "subject".into(),
            format!("{:?}", requirement.subject.origins).into(),
        ),
        ("return_exposure".into(), requirement.return_exposure.into()),
    ])
}

fn write_candidate_lifecycle_trace(events: &[CandidateLifecycleEvent]) -> Result<(), String> {
    let Some(path) = env::var_os(CANDIDATE_LIFECYCLE_TRACE_ENV) else {
        return Ok(());
    };
    if path.is_empty() {
        return Err(format!("{CANDIDATE_LIFECYCLE_TRACE_ENV} is empty"));
    }
    let path = PathBuf::from(path);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| {
            format!(
                "create candidate lifecycle trace {}: {error}",
                path.display()
            )
        })?;
    for event in events {
        serde_json::to_writer(&mut output, event)
            .map_err(|error| format!("serialize candidate lifecycle trace: {error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("write candidate lifecycle trace: {error}"))?;
    }
    output
        .flush()
        .map_err(|error| format!("flush candidate lifecycle trace: {error}"))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LengthAtom {
    collection: PlaceKey,
    definition: StaticWriteToken,
    captured_versions: BTreeSet<StaticWriteToken>,
    invalidated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Utf8Atom {
    bytes: PlaceKey,
    definition: StaticWriteToken,
    captured_versions: BTreeSet<StaticWriteToken>,
    invalidated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CheckedEndAtom {
    offset: PlaceKey,
    width: u64,
    definition: StaticWriteToken,
    captured_versions: BTreeSet<StaticWriteToken>,
    branch_ready: bool,
    invalidated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RangeLimitAtom {
    collection: PlaceKey,
    width: u64,
    definition: StaticWriteToken,
    captured_versions: BTreeSet<StaticWriteToken>,
    invalidated: bool,
}

/// The externally meaningful view of the finite MaybeUninit lattice.
///
/// An empty internal fact set remains the lattice bottom used while solving;
/// callers must never interpret it as initialized.  Likewise, a joined fact,
/// an opaque write, or an FFI output is conservatively Unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TypedInitializationState {
    Initialized,
    Uninitialized,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum InitializationWitnessKind {
    MaybeUninitNew,
    MaybeUninitWrite,
    PointerWrite,
    PointerCopy,
    PointerCopyNonoverlapping,
}

impl InitializationWitnessKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::MaybeUninitNew => "maybe_uninit_new",
            Self::MaybeUninitWrite => "maybe_uninit_write",
            Self::PointerWrite => "ptr_write",
            Self::PointerCopy => "ptr_copy",
            Self::PointerCopyNonoverlapping => "ptr_copy_nonoverlapping",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InitializationWitness {
    kind: InitializationWitnessKind,
    point: ProgramPoint,
}

impl InitializationWitness {
    fn stable_identity(&self) -> String {
        format!("{}@{}", self.kind.as_str(), self.point.span.token())
    }
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
    utf8_atoms: BTreeSet<Utf8Atom>,
    utf8_relation_unknown: bool,
    checked_end_atoms: BTreeSet<CheckedEndAtom>,
    checked_end_unknown: bool,
    range_limit_atoms: BTreeSet<RangeLimitAtom>,
    range_limit_unknown: bool,
    known_lengths: BTreeSet<u64>,
    known_length_unknown: bool,
    may_be_non_length: bool,
    private_aggregate_transform_unknown: bool,
    range_end: BTreeSet<PlaceKey>,
    pointer_base: BTreeSet<PlaceKey>,
    pointer_offset: BTreeSet<PlaceKey>,
    maybe_uninit_initialized: BTreeSet<bool>,
    whole_init_targets: BTreeSet<PlaceKey>,
    whole_init_target_tracked: bool,
    whole_init_target_unknown: bool,
    initialization_witnesses: BTreeSet<InitializationWitness>,
    generic_capability: bool,
    ffi_output: bool,
    open_behavior: bool,
    behavior_controllers: BTreeSet<BehaviorController>,
    behavior_controller_unknown: bool,
    behavior_control: BehaviorControlProvenance,
    may_hold_entry_public_field: bool,
    stored_public_field_slots: BTreeSet<PublicFieldStorageSlot>,
    storage_provenance: StorageProvenance,
    havoced: bool,
    endpoint_sources: Arc<BTreeSet<EndpointNode>>,
    endpoint_edges: Arc<BTreeSet<EndpointEdge>>,
    influence_seeds: InfluenceSeedIndex,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BehaviorControlProvenance {
    #[default]
    Untracked,
    InternalOnly,
    MaybeExternal,
}

impl BehaviorControlProvenance {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Untracked, value) | (value, Self::Untracked) => value,
            (left, right) if left == right => left,
            _ => Self::MaybeExternal,
        }
    }
}

fn extend_fact_set<T: Clone + Ord>(current: &mut BTreeSet<T>, incoming: &BTreeSet<T>) -> bool {
    let before = current.len();
    current.extend(incoming.iter().cloned());
    current.len() != before
}

fn extend_shared_fact_set<T: Clone + Ord>(
    current: &mut Arc<BTreeSet<T>>,
    incoming: &Arc<BTreeSet<T>>,
) -> bool {
    if incoming.is_subset(current) {
        return false;
    }
    Arc::make_mut(current).extend(incoming.iter().cloned());
    true
}

fn join_fact_flag(current: &mut bool, incoming: bool) -> bool {
    let changed = incoming && !*current;
    *current |= incoming;
    changed
}

fn merge_length_atoms(current: &mut BTreeSet<LengthAtom>, incoming: &BTreeSet<LengthAtom>) -> bool {
    let mut changed = false;
    for atom in incoming {
        if let Some(existing) = current.iter().find(|existing| {
            existing.collection == atom.collection && existing.definition == atom.definition
        }) {
            if atom
                .captured_versions
                .is_subset(&existing.captured_versions)
                && (!atom.invalidated || existing.invalidated)
            {
                continue;
            }
            let existing = existing.clone();
            current.remove(&existing);
            let mut merged = existing;
            merged
                .captured_versions
                .extend(atom.captured_versions.iter().copied());
            merged.invalidated |= atom.invalidated;
            current.insert(merged);
            changed = true;
        } else {
            changed |= current.insert(atom.clone());
        }
    }
    changed
}

fn merge_utf8_atoms(current: &mut BTreeSet<Utf8Atom>, incoming: &BTreeSet<Utf8Atom>) -> bool {
    let mut changed = false;
    for atom in incoming {
        if let Some(existing) = current
            .iter()
            .find(|existing| existing.bytes == atom.bytes && existing.definition == atom.definition)
        {
            if atom
                .captured_versions
                .is_subset(&existing.captured_versions)
                && (!atom.invalidated || existing.invalidated)
            {
                continue;
            }
            let existing = existing.clone();
            current.remove(&existing);
            let mut merged = existing;
            merged
                .captured_versions
                .extend(atom.captured_versions.iter().copied());
            merged.invalidated |= atom.invalidated;
            current.insert(merged);
            changed = true;
        } else {
            changed |= current.insert(atom.clone());
        }
    }
    changed
}

fn merge_checked_end_atoms(
    current: &mut BTreeSet<CheckedEndAtom>,
    incoming: &BTreeSet<CheckedEndAtom>,
) -> bool {
    let mut changed = false;
    for atom in incoming {
        if let Some(existing) = current.iter().find(|existing| {
            existing.offset == atom.offset
                && existing.width == atom.width
                && existing.definition == atom.definition
                && existing.branch_ready == atom.branch_ready
        }) {
            if atom
                .captured_versions
                .is_subset(&existing.captured_versions)
                && (!atom.invalidated || existing.invalidated)
            {
                continue;
            }
            let existing = existing.clone();
            current.remove(&existing);
            let mut merged = existing;
            merged
                .captured_versions
                .extend(atom.captured_versions.iter().copied());
            merged.invalidated |= atom.invalidated;
            current.insert(merged);
            changed = true;
        } else {
            changed |= current.insert(atom.clone());
        }
    }
    changed
}

fn merge_range_limit_atoms(
    current: &mut BTreeSet<RangeLimitAtom>,
    incoming: &BTreeSet<RangeLimitAtom>,
) -> bool {
    let mut changed = false;
    for atom in incoming {
        if let Some(existing) = current.iter().find(|existing| {
            existing.collection == atom.collection
                && existing.width == atom.width
                && existing.definition == atom.definition
        }) {
            if atom
                .captured_versions
                .is_subset(&existing.captured_versions)
                && (!atom.invalidated || existing.invalidated)
            {
                continue;
            }
            let existing = existing.clone();
            current.remove(&existing);
            let mut merged = existing;
            merged
                .captured_versions
                .extend(atom.captured_versions.iter().copied());
            merged.invalidated |= atom.invalidated;
            current.insert(merged);
            changed = true;
        } else {
            changed |= current.insert(atom.clone());
        }
    }
    changed
}

impl MayJoin for ValueFacts {
    fn join_may(&mut self, other: &Self) -> bool {
        let mut changed = false;
        changed |= extend_fact_set(&mut self.value.origins, &other.value.origins);
        changed |= extend_fact_set(&mut self.dependencies, &other.dependencies);
        changed |= extend_fact_set(&mut self.value_flow, &other.value_flow);
        changed |= extend_fact_set(&mut self.exact_roots, &other.exact_roots);
        changed |= join_fact_flag(&mut self.semantic_unknown, other.semantic_unknown);
        changed |= extend_fact_set(&mut self.len_of, &other.len_of);
        changed |= merge_length_atoms(&mut self.length_atoms, &other.length_atoms);
        changed |= join_fact_flag(
            &mut self.length_relation_unknown,
            other.length_relation_unknown,
        );
        changed |= merge_utf8_atoms(&mut self.utf8_atoms, &other.utf8_atoms);
        changed |= join_fact_flag(&mut self.utf8_relation_unknown, other.utf8_relation_unknown);
        changed |= merge_checked_end_atoms(&mut self.checked_end_atoms, &other.checked_end_atoms);
        changed |= join_fact_flag(&mut self.checked_end_unknown, other.checked_end_unknown);
        changed |= merge_range_limit_atoms(&mut self.range_limit_atoms, &other.range_limit_atoms);
        changed |= join_fact_flag(&mut self.range_limit_unknown, other.range_limit_unknown);
        changed |= extend_fact_set(&mut self.known_lengths, &other.known_lengths);
        changed |= join_fact_flag(&mut self.known_length_unknown, other.known_length_unknown);
        changed |= join_fact_flag(&mut self.may_be_non_length, other.may_be_non_length);
        changed |= join_fact_flag(
            &mut self.private_aggregate_transform_unknown,
            other.private_aggregate_transform_unknown,
        );
        changed |= extend_fact_set(&mut self.range_end, &other.range_end);
        changed |= extend_fact_set(&mut self.pointer_base, &other.pointer_base);
        changed |= extend_fact_set(&mut self.pointer_offset, &other.pointer_offset);
        changed |= extend_fact_set(
            &mut self.maybe_uninit_initialized,
            &other.maybe_uninit_initialized,
        );
        let whole_target_tracking_disagrees =
            self.whole_init_target_tracked != other.whole_init_target_tracked;
        changed |= extend_fact_set(&mut self.whole_init_targets, &other.whole_init_targets);
        changed |= join_fact_flag(
            &mut self.whole_init_target_unknown,
            other.whole_init_target_unknown || whole_target_tracking_disagrees,
        );
        changed |= join_fact_flag(
            &mut self.whole_init_target_tracked,
            other.whole_init_target_tracked,
        );
        changed |= extend_fact_set(
            &mut self.initialization_witnesses,
            &other.initialization_witnesses,
        );
        changed |= join_fact_flag(&mut self.generic_capability, other.generic_capability);
        changed |= join_fact_flag(&mut self.ffi_output, other.ffi_output);
        changed |= join_fact_flag(&mut self.open_behavior, other.open_behavior);
        changed |= extend_fact_set(&mut self.behavior_controllers, &other.behavior_controllers);
        changed |= join_fact_flag(
            &mut self.behavior_controller_unknown,
            other.behavior_controller_unknown,
        );
        let joined_behavior_control = self.behavior_control.join(other.behavior_control);
        if joined_behavior_control != self.behavior_control {
            self.behavior_control = joined_behavior_control;
            changed = true;
        }
        changed |= join_fact_flag(
            &mut self.may_hold_entry_public_field,
            other.may_hold_entry_public_field,
        );
        changed |= extend_fact_set(
            &mut self.stored_public_field_slots,
            &other.stored_public_field_slots,
        );
        let joined_storage = self.storage_provenance.join(&other.storage_provenance);
        if joined_storage != self.storage_provenance {
            self.storage_provenance = joined_storage;
            changed = true;
        }
        changed |= join_fact_flag(&mut self.havoced, other.havoced);
        changed |= extend_shared_fact_set(&mut self.endpoint_sources, &other.endpoint_sources);
        changed |= extend_shared_fact_set(&mut self.endpoint_edges, &other.endpoint_edges);
        let influence_before = self.influence_seeds.clone();
        self.influence_seeds.join(&other.influence_seeds);
        changed |= self.influence_seeds != influence_before;
        changed
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ValidationBinding {
    predicate: Predicate,
    subject: PlaceKey,
    collection: Option<PlaceKey>,
    access_width: Option<u64>,
    proof_definition: Option<StaticWriteToken>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangeIndexKind {
    Full,
    From,
    To,
    Between,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RelationalObligation {
    Within {
        base: PlaceKey,
        start: RelationTerm,
        end: RelationTerm,
    },
}

impl ValidationBinding {
    fn range(
        subject: PlaceKey,
        collection: Option<PlaceKey>,
        access_width: u64,
        proof_definition: Option<StaticWriteToken>,
    ) -> Self {
        Self {
            predicate: Predicate::RangeInBounds,
            subject,
            collection,
            access_width: Some(access_width),
            proof_definition,
        }
    }

    fn storage_places(&self) -> Vec<PlaceKey> {
        let mut places = vec![canonical_storage(&self.subject)];
        places.extend(self.collection.iter().map(canonical_storage));
        places.sort();
        places.dedup();
        places
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ProofBinding {
    Validation(ValidationBinding),
    Relation(RelationFact),
    ReturnVariant(ReturnVariantFact),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReturnVariantFact {
    destination: PlaceKey,
    discriminant: u128,
    payload_path: Vec<ProjectionKey>,
}

impl From<ValidationBinding> for ProofBinding {
    fn from(binding: ValidationBinding) -> Self {
        Self::Validation(binding)
    }
}

impl PartialEq<ValidationBinding> for ProofBinding {
    fn eq(&self, other: &ValidationBinding) -> bool {
        self.validation() == Some(other)
    }
}

impl ProofBinding {
    fn storage_places(&self) -> Vec<PlaceKey> {
        match self {
            Self::Validation(binding) => binding.storage_places(),
            Self::Relation(relation) => relation_storage_places(relation),
            Self::ReturnVariant(variant) => vec![canonical_storage(&variant.destination)],
        }
    }

    fn validation(&self) -> Option<&ValidationBinding> {
        match self {
            Self::Validation(binding) => Some(binding),
            Self::Relation(_) | Self::ReturnVariant(_) => None,
        }
    }

    fn relation(&self) -> Option<&RelationFact> {
        match self {
            Self::Validation(_) | Self::ReturnVariant(_) => None,
            Self::Relation(relation) => Some(relation),
        }
    }

    fn return_variant(&self) -> Option<&ReturnVariantFact> {
        match self {
            Self::ReturnVariant(variant) => Some(variant),
            Self::Validation(_) | Self::Relation(_) => None,
        }
    }
}

type ProgramState = BlockState<PlaceKey, ValueFacts, ProofBinding>;
type ProgramDataflowResult = DataflowResult<PlaceKey, ValueFacts, ProofBinding>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtractionPhase {
    Outputs,
    Final,
}

impl ExtractionPhase {
    fn tracks_endpoints(self) -> bool {
        self == Self::Final
    }
}

#[cfg(target_os = "macos")]
fn release_allocator_pressure() {
    use std::ffi::c_void;

    unsafe extern "C" {
        fn malloc_zone_pressure_relief(zone: *mut c_void, goal: usize) -> usize;
    }
    unsafe {
        let _ = malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn release_allocator_pressure() {}

#[derive(Clone, Debug)]
struct BodyProgram {
    def_id: LocalDefId,
    function: FunctionKey,
    root_span: StableSpan,
    is_closure: bool,
    arg_count: usize,
    has_self: bool,
    argument_influence: BTreeMap<u32, InfluenceToken>,
    argument_utf8_capability: BTreeMap<u32, Utf8FormalCapability>,
    out_formals: BTreeMap<PlaceKey, u32>,
    operations: IndexVec<BasicBlock, Vec<ProgramOp>>,
    edges: IndexVec<BasicBlock, EdgeTerm>,
    successors: IndexVec<BasicBlock, BTreeSet<BasicBlock>>,
}

/// Type-level UTF-8 capability attached to one MIR entry formal.  This is a
/// closed compiler-type classification, not an API/name heuristic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Utf8FormalCapability {
    ValidUtf8,
    UnrestrictedBytes,
    Unknown,
}

fn utf8_formal_capability(mut value: ty::Ty<'_>) -> Utf8FormalCapability {
    while let ty::Ref(_, inner, _) = value.kind() {
        value = *inner;
    }
    match value.kind() {
        ty::Str => Utf8FormalCapability::ValidUtf8,
        ty::Slice(element) | ty::Array(element, _)
            if matches!(element.kind(), ty::Uint(ty::UintTy::U8)) =>
        {
            Utf8FormalCapability::UnrestrictedBytes
        }
        _ => Utf8FormalCapability::Unknown,
    }
}

impl BodyProgram {
    fn attached_unsafe_site_count(&self) -> usize {
        self.operations
            .iter()
            .flatten()
            .map(|operation| operation.unsafe_sites.len())
            .sum()
    }
}

fn influence_type_class_for_ty(ty: ty::Ty<'_>) -> InfluenceTypeClass {
    match ty.kind() {
        ty::Bool => InfluenceTypeClass::Boolean,
        ty::Int(_) | ty::Uint(_) | ty::Char => InfluenceTypeClass::Integer,
        ty::RawPtr(..) => InfluenceTypeClass::Pointer,
        ty::Ref(..) => InfluenceTypeClass::Reference,
        ty::FnDef(..) | ty::FnPtr(..) | ty::Closure(..) => InfluenceTypeClass::Callable,
        ty::Array(..) | ty::Slice(..) | ty::Str => InfluenceTypeClass::Sequence,
        ty::Tuple(..) | ty::Adt(..) | ty::Float(_) => InfluenceTypeClass::Aggregate,
        _ => InfluenceTypeClass::Opaque,
    }
}

fn is_private_aggregate_type(tcx: TyCtxt<'_>, value: ty::Ty<'_>) -> bool {
    matches!(value.kind(), ty::Adt(definition, _) if tcx.visibility(definition.did()) != Visibility::Public)
}

fn result_type_arguments<'tcx>(
    tcx: TyCtxt<'tcx>,
    value: ty::Ty<'tcx>,
) -> Option<(ty::Ty<'tcx>, ty::Ty<'tcx>)> {
    let ty::Adt(definition, arguments) = value.kind() else {
        return None;
    };
    tcx.is_diagnostic_item(sym::Result, definition.did())
        .then(|| (arguments.type_at(0), arguments.type_at(1)))
}

#[rustversion::before(1.96)]
fn erase_type_regions<'tcx>(tcx: TyCtxt<'tcx>, value: ty::Ty<'tcx>) -> ty::Ty<'tcx> {
    tcx.erase_regions(value)
}

#[rustversion::since(1.96)]
fn erase_type_regions<'tcx>(tcx: TyCtxt<'tcx>, value: ty::Ty<'tcx>) -> ty::Ty<'tcx> {
    tcx.erase_and_anonymize_regions(value)
}

#[rustversion::before(1.96)]
fn erased_identity_fn_sig<'tcx>(tcx: TyCtxt<'tcx>, did: DefId) -> ty::FnSig<'tcx> {
    tcx.instantiate_bound_regions_with_erased(tcx.fn_sig(did).instantiate_identity())
}

#[rustversion::since(1.96)]
fn erased_identity_fn_sig<'tcx>(tcx: TyCtxt<'tcx>, did: DefId) -> ty::FnSig<'tcx> {
    let signature = tcx.fn_sig(did).instantiate_identity().skip_norm_wip();
    tcx.instantiate_bound_regions_with_erased(signature)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Utf8ValidationCallShape {
    caller_is_str_from_utf8: bool,
    resolved_local: bool,
    one_argument: bool,
    actual_matches_caller_input: bool,
    callee_input_matches_actual: bool,
    destination_matches_callee_output: bool,
    validator_ok_is_unit: bool,
    error_matches: bool,
}

fn exact_utf8_validation_call_shape(shape: Utf8ValidationCallShape) -> bool {
    shape.caller_is_str_from_utf8
        && shape.resolved_local
        && shape.one_argument
        && shape.actual_matches_caller_input
        && shape.callee_input_matches_actual
        && shape.destination_matches_callee_output
        && shape.validator_ok_is_unit
        && shape.error_matches
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct MappedCallOutputs {
    returned: Option<(PlaceKey, AbstractValue)>,
    outs: BTreeMap<PlaceKey, MappedOutValue>,
    relations: BTreeSet<RelationFact>,
    conditional_relations: BTreeSet<MappedConditionalRelation>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MappedConditionalRelation {
    condition: ReturnCondition,
    relation: RelationFact,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct MappedOutValue {
    value: AbstractValue,
    may_skip_write: bool,
}

type LocalCallOutputs = BTreeMap<ProgramPoint, MappedCallOutputs>;

fn out_slot(formal_index: u32) -> PlaceKey {
    PlaceKey::new(format!("$out:{formal_index}"))
}

fn join_local_outputs(
    current: &mut BTreeMap<FunctionKey, LocalCallOutputs>,
    incoming: BTreeMap<FunctionKey, LocalCallOutputs>,
) -> BTreeSet<FunctionKey> {
    let mut changed_owners = BTreeSet::new();
    for (function, calls) in incoming {
        let mut changed = false;
        for (point, outputs) in calls {
            let has_effect = outputs
                .returned
                .as_ref()
                .is_some_and(|(_, value)| !value.origins.is_empty())
                || outputs
                    .outs
                    .values()
                    .any(|value| !value.value.origins.is_empty() || value.may_skip_write)
                || !outputs.relations.is_empty()
                || !outputs.conditional_relations.is_empty();
            if !has_effect {
                continue;
            }
            let destination = current
                .entry(function.clone())
                .or_default()
                .entry(point)
                .or_default();
            if let Some((place, mut value)) = outputs.returned {
                if !value.origins.is_empty() {
                    match destination.returned.as_mut() {
                        Some(returned) => {
                            debug_assert_eq!(returned.0, place);
                            let before = returned.1.origins.len();
                            returned.1.origins.append(&mut value.origins);
                            changed |= returned.1.origins.len() != before;
                        }
                        None => {
                            destination.returned = Some((place, value));
                            changed = true;
                        }
                    }
                }
            }
            for (place, mut value) in outputs.outs {
                if value.value.origins.is_empty() && !value.may_skip_write {
                    continue;
                }
                match destination.outs.entry(place) {
                    Entry::Occupied(mut entry) => {
                        let output = entry.get_mut();
                        let before = output.value.origins.len();
                        output.value.origins.append(&mut value.value.origins);
                        changed |= output.value.origins.len() != before;
                        changed |= !output.may_skip_write && value.may_skip_write;
                        output.may_skip_write |= value.may_skip_write;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(value);
                        changed = true;
                    }
                }
            }
            let before = destination.relations.len();
            let mut relations = outputs.relations;
            destination.relations.append(&mut relations);
            changed |= destination.relations.len() != before;
            let before = destination.conditional_relations.len();
            let mut conditional_relations = outputs.conditional_relations;
            destination
                .conditional_relations
                .append(&mut conditional_relations);
            changed |= destination.conditional_relations.len() != before;
        }
        if changed {
            changed_owners.insert(function);
        }
    }
    changed_owners
}

fn map_structured_local_outputs(
    mappings: &BTreeSet<CallMapping>,
    summary: &FunctionSummary,
    actuals: &[AbstractValue],
    caller_out_formals: &BTreeMap<PlaceKey, u32>,
    definite_outs: BTreeSet<u32>,
) -> MappedCallOutputs {
    let return_mappings = mappings
        .iter()
        .filter(|mapping| matches!(mapping, CallMapping::ReturnToDestination { .. }))
        .cloned()
        .collect::<BTreeSet<_>>();
    let out_mappings = mappings
        .iter()
        .filter(|mapping| matches!(mapping, CallMapping::OutToCaller { .. }))
        .cloned()
        .collect::<BTreeSet<_>>();
    let returned = map_call_outputs(
        &return_mappings,
        &summary.return_value,
        &BTreeMap::new(),
        actuals,
    )
    .into_iter()
    .next();
    let mut outs = BTreeMap::<PlaceKey, MappedOutValue>::new();
    let out_values = out_values_from_dependencies(&summary.out_dependencies);
    for mapping in out_mappings {
        let CallMapping::OutToCaller {
            formal_index,
            caller_place,
        } = mapping
        else {
            unreachable!("out_mappings contains only OutToCaller entries");
        };
        let mapped = map_call_outputs(
            &BTreeSet::from([CallMapping::OutToCaller {
                formal_index,
                caller_place: caller_place.clone(),
            }]),
            &AbstractValue::default(),
            &out_values,
            actuals,
        );
        let mapped = mapped.into_iter().next().or_else(|| {
            definite_outs
                .contains(&formal_index)
                .then(|| (caller_place, AbstractValue::default()))
        });
        if let Some((place, value)) = mapped {
            let must_write = definite_outs.contains(&formal_index);
            let output = outs.entry(place.clone()).or_default();
            output.value.origins.extend(value.origins.iter().cloned());
            output.may_skip_write |= !must_write;
            if let Some(slot) = caller_out_formals.get(&place).copied().map(out_slot) {
                let output = outs.entry(slot).or_default();
                output.value.origins.extend(value.origins);
                output.may_skip_write |= !must_write;
            }
        }
    }
    MappedCallOutputs {
        returned,
        outs,
        relations: BTreeSet::new(),
        conditional_relations: BTreeSet::new(),
    }
}

fn instantiate_summary_term(
    term: &SummaryRelationTerm,
    condition: Option<&ReturnCondition>,
    call: &CallSite,
    mappings: &BTreeSet<CallMapping>,
) -> Option<RelationTerm> {
    let actual = |index: u32| {
        let value = call.arg_values.get(index.checked_sub(1)? as usize)?;
        (!value.havoced)
            .then(|| unique_semantic_value(value))
            .flatten()
    };
    let actual_projection = |index: u32, path: &[ProjectionKey]| {
        let base = actual(index)?;
        let mut path = path.to_vec();
        let raw = call
            .args
            .get(index.checked_sub(1)? as usize)?
            .place()
            .map(canonical_storage);
        if matches!(path.first(), Some(ProjectionKey::Deref))
            && raw
                .as_ref()
                .is_some_and(|raw| *raw != canonical_storage(&base))
        {
            path.remove(0);
        }
        Some(RelationTerm::Projection { base, path })
    };
    Some(match term {
        SummaryRelationTerm::Formal(index) => RelationTerm::Place(actual(*index)?),
        SummaryRelationTerm::FormalProjection { formal, path } => actual_projection(*formal, path)?,
        SummaryRelationTerm::Return => match condition {
            Some(condition) => RelationTerm::Projection {
                base: canonical_storage(&call.destination),
                path: condition.payload_path.clone(),
            },
            None => RelationTerm::Place(call.destination.clone()),
        },
        SummaryRelationTerm::ReturnProjection(path) => RelationTerm::Projection {
            base: canonical_storage(&call.destination),
            path: path.clone(),
        },
        SummaryRelationTerm::Out(index) => {
            let place = mappings.iter().find_map(|mapping| match mapping {
                CallMapping::OutToCaller {
                    formal_index,
                    caller_place,
                } if formal_index == index => Some(caller_place),
                _ => None,
            })?;
            RelationTerm::Place(place.clone())
        }
        SummaryRelationTerm::OutProjection { formal, path } => {
            let place = mappings.iter().find_map(|mapping| match mapping {
                CallMapping::OutToCaller {
                    formal_index,
                    caller_place,
                } if formal_index == formal => Some(caller_place),
                _ => None,
            })?;
            RelationTerm::Projection {
                base: place.clone(),
                path: path.clone(),
            }
        }
        SummaryRelationTerm::LengthFormal(index) => RelationTerm::Length(actual(*index)?),
        SummaryRelationTerm::CapacityFormal(index) => RelationTerm::Capacity(actual(*index)?),
        SummaryRelationTerm::Constant(value) => RelationTerm::Constant(*value),
    })
}

fn instantiate_summary_relation(
    relation: &SummaryRelationKind,
    condition: Option<&ReturnCondition>,
    call: &CallSite,
    mappings: &BTreeSet<CallMapping>,
) -> Option<RelationKind> {
    let term = |term| instantiate_summary_term(term, condition, call, mappings);
    let base = |formal| {
        instantiate_summary_term(
            &SummaryRelationTerm::Formal(formal),
            condition,
            call,
            mappings,
        )?
        .storage_place()
        .cloned()
    };
    Some(match relation {
        SummaryRelationKind::Eq(left, right) => RelationKind::Eq(term(left)?, term(right)?),
        SummaryRelationKind::Le(left, right) => RelationKind::Le(term(left)?, term(right)?),
        SummaryRelationKind::Lt(left, right) => RelationKind::Lt(term(left)?, term(right)?),
        SummaryRelationKind::AddConst {
            result,
            base,
            addend,
        } => RelationKind::AddConst {
            result: term(result)?,
            base: term(base)?,
            addend: *addend,
        },
        SummaryRelationKind::Sub {
            result,
            upper,
            lower,
        } => RelationKind::Sub {
            result: term(result)?,
            upper: term(upper)?,
            lower: term(lower)?,
        },
        SummaryRelationKind::SumEq { total, left, right } => RelationKind::SumEq {
            total: term(total)?,
            left: term(left)?,
            right: term(right)?,
        },
        SummaryRelationKind::Within {
            base_formal,
            start,
            end,
        } => RelationKind::Within {
            base: base(*base_formal)?,
            start: term(start)?,
            end: term(end)?,
        },
        SummaryRelationKind::InitializedWithin {
            base_formal,
            start,
            end,
        } => RelationKind::InitializedWithin {
            base: base(*base_formal)?,
            start: term(start)?,
            end: term(end)?,
        },
        SummaryRelationKind::CapacityFor { base_formal, end } => RelationKind::CapacityFor {
            base: base(*base_formal)?,
            end: term(end)?,
        },
        SummaryRelationKind::SameObject(left, right) => {
            RelationKind::SameObject(term(left)?, term(right)?)
        }
    })
}

fn instantiate_relational_postconditions(
    postconditions: &BTreeSet<RelationalPostcondition>,
    call: &CallSite,
    mappings: &BTreeSet<CallMapping>,
) -> (BTreeSet<RelationFact>, BTreeSet<MappedConditionalRelation>) {
    let mut unconditional = BTreeSet::new();
    let mut conditional = BTreeSet::new();
    for postcondition in postconditions {
        let Some(relation) = instantiate_summary_relation(
            &postcondition.relation,
            postcondition.condition.as_ref(),
            call,
            mappings,
        ) else {
            continue;
        };
        let relation = RelationFact::new(relation);
        if let Some(condition) = &postcondition.condition {
            conditional.insert(MappedConditionalRelation {
                condition: condition.clone(),
                relation,
            });
        } else {
            unconditional.insert(relation);
        }
    }
    (unconditional, conditional)
}

fn reachable_out_dependencies(
    edges: &IndexVec<BasicBlock, EdgeTerm>,
    formals: impl IntoIterator<Item = u32>,
    solved: &ProgramDataflowResult,
    definite_outs: &BTreeSet<u32>,
) -> BTreeSet<OutDependency> {
    let formals = formals.into_iter().collect::<BTreeSet<_>>();
    let mut values = BTreeMap::<u32, AbstractValue>::new();
    for (block, edge) in edges.iter_enumerated() {
        if !matches!(edge, EdgeTerm::Return) {
            continue;
        }
        let Some(state) = solved.exit[block].as_ref() else {
            continue;
        };
        for formal_index in &formals {
            let Some(output) = state.may_values().get(&out_slot(*formal_index)) else {
                continue;
            };
            values
                .entry(*formal_index)
                .or_default()
                .origins
                .extend(output.value.origins.iter().cloned());
        }
    }
    formals
        .into_iter()
        .filter_map(|formal_index| {
            let value = values.remove(&formal_index).unwrap_or_default();
            (!value.origins.is_empty() || definite_outs.contains(&formal_index)).then_some(
                OutDependency {
                    formal_index,
                    value,
                    may_skip_write: !definite_outs.contains(&formal_index),
                },
            )
        })
        .collect()
}

fn solve_out_must(
    operations: &IndexVec<BasicBlock, Vec<ProgramOp>>,
    edges: &IndexVec<BasicBlock, EdgeTerm>,
    successors: &IndexVec<BasicBlock, BTreeSet<BasicBlock>>,
    local_outputs: &LocalCallOutputs,
) -> IndexVec<BasicBlock, Option<BTreeSet<u32>>> {
    let mut entry = IndexVec::from_elem_n(None, operations.len());
    let mut exit = IndexVec::from_elem_n(None, operations.len());
    let mut edge = BTreeMap::<(BasicBlock, BasicBlock), BTreeSet<u32>>::new();
    loop {
        let before = (entry.clone(), exit.clone(), edge.clone());
        for block in operations.indices() {
            let incoming = if block.index() == 0 {
                Some(BTreeSet::new())
            } else {
                let mut predecessors = edge
                    .iter()
                    .filter(|((_, target), _)| *target == block)
                    .map(|(_, state)| state);
                predecessors.next().map(|first| {
                    let mut common = first.clone();
                    for state in predecessors {
                        common.retain(|formal| state.contains(formal));
                    }
                    common
                })
            };
            entry[block] = incoming.clone();
            let Some(mut state) = incoming else {
                exit[block] = None;
                continue;
            };
            for operation in &operations[block] {
                if let ProgramOpKind::Assign(AssignmentModel {
                    out_formal: Some(formal_index),
                    ..
                }) = &operation.kind
                {
                    state.insert(*formal_index);
                }
            }
            exit[block] = Some(state.clone());
            for target in &successors[block] {
                let mut edge_state = state.clone();
                if let EdgeTerm::CallNormal {
                    normal_target: Some(normal_target),
                    point,
                    ..
                } = &edges[block]
                {
                    if target == normal_target {
                        if let EdgeTerm::CallNormal { call, .. } = &edges[block] {
                            if let Some(formal_index) = call.destination_out_formal {
                                edge_state.insert(formal_index);
                            }
                        }
                        if let Some(outputs) = local_outputs.get(point) {
                            for (place, output) in &outputs.outs {
                                if !output.may_skip_write {
                                    if let Some(formal_index) = out_slot_index(place) {
                                        edge_state.insert(formal_index);
                                    }
                                }
                            }
                        }
                    }
                }
                edge.insert((block, *target), edge_state);
            }
        }
        if (entry.clone(), exit.clone(), edge.clone()) == before {
            return exit;
        }
    }
}

fn out_slot_index(place: &PlaceKey) -> Option<u32> {
    place.0.strip_prefix("$out:")?.parse().ok()
}

#[derive(Clone, Debug)]
struct ProgramOp {
    point: Option<ProgramPoint>,
    kind: ProgramOpKind,
    unsafe_sites: Vec<BoundUnsafeSite>,
}

impl ProgramOp {
    fn semantic_point(&self) -> &ProgramPoint {
        self.point
            .as_ref()
            .expect("semantic MIR operations require a stable source point")
    }
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
    destination_projection: Option<PlaceProjectionModel>,
    value: ValueModel,
    field_origin: Option<FieldAccessModel>,
    raw_read_source: Option<OperandModel>,
    legacy_write: bool,
    out_formal: Option<u32>,
}

#[derive(Clone, Debug)]
struct OperandModel {
    operand: CallOperand,
    field_origin: Option<FieldAccessModel>,
    pure_deref_base: Option<PlaceKey>,
    exact_projection: Option<PlaceKey>,
    projection_path: Option<Vec<ProjectionKey>>,
    range_index_kind: Option<RangeIndexKind>,
    projection_unknown: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FieldAccessModel {
    access_owner: FunctionKey,
    full_place: PlaceKey,
    base: PlaceKey,
    projection_path: Option<Vec<ProjectionKey>>,
    projection_unknown: bool,
    def_path: String,
    adt_kind: PublicFieldAdtKind,
    path_public: bool,
    type_class: InfluenceTypeClass,
    span: StableSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PlaceProjectionModel {
    base: PlaceKey,
    path: Vec<ProjectionKey>,
    projection_unknown: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PublicFieldStorageSlot {
    full_place: PlaceKey,
    base: PlaceKey,
    path: Vec<ProjectionKey>,
}

#[derive(Clone, Debug)]
enum ValueModel {
    Operand(OperandModel),
    Unsize {
        operand: OperandModel,
        known_length: Option<u64>,
    },
    RefOrRaw(OperandModel),
    LocalRefOrRaw(OperandModel),
    IdentityPointerView {
        operand: OperandModel,
        generic_view: Option<GenericCapabilityEvidence>,
    },
    AddressPointerView {
        operand: OperandModel,
        generic_view: Option<GenericCapabilityEvidence>,
    },
    PointerCast {
        operand: OperandModel,
        obligation: ConcreteLayoutObligation,
    },
    GenericPointerCast {
        operand: OperandModel,
        evidence: GenericCapabilityEvidence,
    },
    Discriminant(OperandModel),
    Length(OperandModel),
    Compare {
        op: BinOp,
        left: OperandModel,
        right: OperandModel,
    },
    CheckedBinary {
        op: BinOp,
        left: OperandModel,
        right: OperandModel,
    },
    Aggregate {
        operands: Vec<OperandModel>,
        range_end: Option<PlaceKey>,
        variant: Option<u128>,
        closure: Option<FunctionKey>,
    },
    BehaviorPointer {
        operand: OperandModel,
        seed: Option<BehaviorController>,
    },
    Unknown,
}

#[derive(Clone, Debug)]
struct CallModel {
    descriptor: CallDescriptor,
    value: RegistryValueModel,
    predicate: Option<PredicateModel>,
    mutable_actuals: Vec<PlaceKey>,
    ffi_out_actuals: Vec<FfiOutActual>,
    typed_initialization_effect: Option<TypedInitializationEffect>,
    destination_is_bool: bool,
    destination_restricted: bool,
    access_width: Option<u64>,
    legacy_write: bool,
    destination_out_formal: Option<u32>,
}

#[derive(Clone, Debug)]
struct FfiOutActual {
    place: PlaceKey,
    type_class: InfluenceTypeClass,
}

#[derive(Clone, Debug)]
enum TypedInitializationEffect {
    MaybeUninitWrite {
        destination: OperandModel,
    },
    PointerWrite {
        destination: OperandModel,
        source: OperandModel,
    },
    PointerCopy {
        source: OperandModel,
        destination: OperandModel,
        count: CallOperand,
        nonoverlapping: bool,
    },
    OpaqueIntrinsicMutation {
        destinations: Vec<OperandModel>,
    },
}

#[derive(Clone, Debug)]
struct CallDescriptor {
    callee: Option<FunctionKey>,
    raw_def: Option<DefId>,
    disposition: BoundaryDisposition,
    args: Vec<OperandModel>,
    destination: PlaceKey,
    generic_substitution: GenericCallSubstitution,
    #[cfg(not(test))]
    typed_witness_templates: BTreeSet<TypedWitnessTransportTemplate>,
}

#[derive(Clone)]
struct ResolvedLocalCall<'tcx> {
    key: FunctionKey,
    target: DefId,
    args: ty::GenericArgsRef<'tcx>,
}

#[derive(Clone, Debug)]
enum RegistryValueModel {
    Empty,
    ExternalBehavior {
        kind: ExternalSourceKind,
        controller: BehaviorController,
        token: String,
        type_class: InfluenceTypeClass,
    },
    MaybeUninit(bool),
    TypedInitializationPointer {
        receiver: OperandModel,
        generic_view: Option<GenericCapabilityEvidence>,
    },
    AssumeInit {
        source: OperandModel,
        restricted_target: bool,
    },
    Length(OperandModel),
    SliceIterator(OperandModel),
    SlicePosition(OperandModel),
    RangeIntoIterator(OperandModel),
    RangeIteratorNext(OperandModel),
    PrivateAggregateTransform,
    SelectedOpenBehavior {
        token: String,
        controller: BehaviorController,
        inputs: Vec<OperandModel>,
        type_class: InfluenceTypeClass,
    },
    IndirectBehavior {
        source: OperandModel,
        type_class: InfluenceTypeClass,
    },
    GenericCapability {
        inputs: Vec<OperandModel>,
        evidence: GenericCapabilityEvidence,
    },
    SaturatingSub(Vec<OperandModel>),
    CheckedAdd {
        offset: OperandModel,
        width: OperandModel,
    },
    CheckedTryBranch(OperandModel),
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
    IdentityPointerView {
        operand: OperandModel,
        generic_view: Option<GenericCapabilityEvidence>,
    },
    AddressPointerView {
        operand: OperandModel,
        generic_view: Option<GenericCapabilityEvidence>,
    },
    PointerCast {
        operand: OperandModel,
        obligation: ConcreteLayoutObligation,
    },
    GenericPointerCast {
        base: OperandModel,
        evidence: GenericCapabilityEvidence,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PointerTransformEvidence {
    same_pointee: bool,
    sequence_element_view: bool,
    transparent_wrapper_view: bool,
    unit_erasure: bool,
    compiler_layout_implies_target: bool,
    source_validity_implies_target: bool,
    has_generic: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointerTransformKind {
    IdentityView,
    AddressView,
    Reinterpretation,
    GenericLayoutRisk,
}

/// Compiler-derived finite facts for one concrete raw-pointer
/// reinterpretation.  This deliberately records only violations that have a
/// concrete source/target type witness; missing layout or validity facts stay
/// outside the solver domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ConcreteLayoutObligation {
    size_violation: bool,
    alignment_violation: bool,
    validity_violation: bool,
    complete: bool,
    externally_controlled_bits: bool,
}

fn classify_pointer_transform(evidence: PointerTransformEvidence) -> PointerTransformKind {
    if evidence.same_pointee {
        PointerTransformKind::IdentityView
    } else if evidence.sequence_element_view
        || evidence.transparent_wrapper_view
        || evidence.unit_erasure
        || (evidence.compiler_layout_implies_target && evidence.source_validity_implies_target)
    {
        PointerTransformKind::AddressView
    } else if evidence.has_generic {
        PointerTransformKind::GenericLayoutRisk
    } else {
        PointerTransformKind::Reinterpretation
    }
}

#[derive(Clone, Debug)]
enum PredicateModel {
    Utf8Result { bytes: OperandModel },
    IsErr { checked: OperandModel },
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
    ArrayLengths(BTreeSet<PlaceKey>),
    Unknown,
}

fn entry_field_reads(program: &BodyProgram) -> BTreeMap<PlaceKey, FieldAccessModel> {
    let mut fields = BTreeMap::new();
    for operation in program.operations.iter().flatten() {
        match &operation.kind {
            ProgramOpKind::Assign(assignment) => {
                collect_value_field_reads(&assignment.value, &mut fields);
                if let Some(source) = &assignment.raw_read_source {
                    collect_operand_field_read(source, &mut fields);
                }
            }
            ProgramOpKind::Call(call) => {
                for operand in &call.descriptor.args {
                    collect_operand_field_read(operand, &mut fields);
                }
                collect_registry_field_reads(&call.value, &mut fields);
            }
            ProgramOpKind::Nop => {}
        }
    }
    fields
}

fn collect_operand_field_read(
    operand: &OperandModel,
    fields: &mut BTreeMap<PlaceKey, FieldAccessModel>,
) {
    let Some(field) = &operand.field_origin else {
        return;
    };
    fields
        .entry(field.full_place.clone())
        .or_insert_with(|| field.clone());
}

fn collect_value_field_reads(
    value: &ValueModel,
    fields: &mut BTreeMap<PlaceKey, FieldAccessModel>,
) {
    match value {
        ValueModel::Operand(operand)
        | ValueModel::RefOrRaw(operand)
        | ValueModel::LocalRefOrRaw(operand)
        | ValueModel::PointerCast { operand, .. }
        | ValueModel::Discriminant(operand)
        | ValueModel::Length(operand)
        | ValueModel::BehaviorPointer { operand, .. }
        | ValueModel::Unsize { operand, .. }
        | ValueModel::IdentityPointerView { operand, .. }
        | ValueModel::AddressPointerView { operand, .. }
        | ValueModel::GenericPointerCast { operand, .. } => {
            collect_operand_field_read(operand, fields);
        }
        ValueModel::Compare { left, right, .. } | ValueModel::CheckedBinary { left, right, .. } => {
            collect_operand_field_read(left, fields);
            collect_operand_field_read(right, fields);
        }
        ValueModel::Aggregate { operands, .. } => {
            for operand in operands {
                collect_operand_field_read(operand, fields);
            }
        }
        ValueModel::Unknown => {}
    }
}

fn collect_registry_field_reads(
    value: &RegistryValueModel,
    fields: &mut BTreeMap<PlaceKey, FieldAccessModel>,
) {
    match value {
        RegistryValueModel::TypedInitializationPointer {
            receiver: source, ..
        }
        | RegistryValueModel::AssumeInit { source, .. }
        | RegistryValueModel::Length(source)
        | RegistryValueModel::SliceIterator(source)
        | RegistryValueModel::SlicePosition(source)
        | RegistryValueModel::RangeIntoIterator(source)
        | RegistryValueModel::RangeIteratorNext(source)
        | RegistryValueModel::IndirectBehavior { source, .. }
        | RegistryValueModel::CheckedTryBranch(source)
        | RegistryValueModel::SliceAsPtr(source)
        | RegistryValueModel::PointerCast {
            operand: source, ..
        }
        | RegistryValueModel::IdentityPointerView {
            operand: source, ..
        }
        | RegistryValueModel::AddressPointerView {
            operand: source, ..
        }
        | RegistryValueModel::GenericPointerCast { base: source, .. } => {
            collect_operand_field_read(source, fields);
        }
        RegistryValueModel::SelectedOpenBehavior { inputs, .. }
        | RegistryValueModel::GenericCapability { inputs, .. }
        | RegistryValueModel::SaturatingSub(inputs) => {
            for operand in inputs {
                collect_operand_field_read(operand, fields);
            }
        }
        RegistryValueModel::CheckedAdd { offset, width } => {
            collect_operand_field_read(offset, fields);
            collect_operand_field_read(width, fields);
        }
        RegistryValueModel::SlicePrefix { base, .. } => {
            collect_operand_field_read(base, fields);
        }
        RegistryValueModel::WrappingAdd { base, offset } => {
            collect_operand_field_read(base, fields);
            collect_operand_field_read(offset, fields);
        }
        RegistryValueModel::Empty
        | RegistryValueModel::PrivateAggregateTransform
        | RegistryValueModel::ExternalBehavior { .. }
        | RegistryValueModel::MaybeUninit(_)
        | RegistryValueModel::Constant(_) => {}
    }
}

#[derive(Clone, Debug)]
struct CallSite {
    callee: Option<FunctionKey>,
    raw_def: Option<DefId>,
    disposition: BoundaryDisposition,
    args: Vec<CallOperand>,
    arg_values: Vec<ValueFacts>,
    actual_endpoint_sources: Vec<BTreeSet<EndpointNode>>,
    destination: PlaceKey,
    point: ProgramPoint,
    destination_is_bool: bool,
    destination_restricted: bool,
    access_width: Option<u64>,
    index_range_kind: Option<RangeIndexKind>,
    generic_substitution: GenericCallSubstitution,
    #[cfg(not(test))]
    typed_witness_templates: BTreeSet<TypedWitnessTransportTemplate>,
}

impl CallSite {
    fn typed_witness_templates(&self) -> BTreeSet<TypedWitnessTransportTemplate> {
        #[cfg(not(test))]
        {
            self.typed_witness_templates.clone()
        }
        #[cfg(test)]
        {
            BTreeSet::new()
        }
    }
}

/// Finite key exported by one function's endpoint summary.  It deliberately
/// contains endpoint atoms only; the call-chain witness is recorded outside
/// this lattice so recursive expansion cannot make the fixed point infinite.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EndpointRelation {
    input: EndpointNode,
    terminal: EndpointNode,
}

#[derive(Clone, Debug)]
struct CompareFact {
    result: PlaceKey,
    op: BinOp,
    left: CallOperand,
    right: CallOperand,
    left_formal: Option<u32>,
    right_formal: Option<u32>,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ValidationSnapshot {
    current_validations: BTreeSet<ValidationBinding>,
    current_relations: BTreeSet<RelationFact>,
    entry_formals: BTreeMap<PlaceKey, BoundarySlot>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BodyValueFacts {
    value_flow: BTreeSet<PlaceKey>,
}

impl BodyValueFacts {
    fn join_value_facts(&mut self, incoming: &ValueFacts) {
        self.value_flow.extend(incoming.value_flow.iter().cloned());
    }
}

impl ValidationSnapshot {
    fn capture(state: &ProgramState, arg_count: usize) -> Self {
        let current_validations = state
            .must_validations()
            .iter()
            .filter_map(|validation| {
                let binding = validation.predicate().validation()?;
                state
                    .has_current_validation(binding, binding.storage_places())
                    .then(|| binding.clone())
            })
            .collect();
        let current_relations = state
            .must_validations()
            .iter()
            .filter_map(|validation| {
                let relation = validation.predicate().relation()?;
                state
                    .has_current_validation(
                        &ProofBinding::Relation(relation.clone()),
                        relation_storage_places(relation),
                    )
                    .then(|| relation.clone())
            })
            .collect();
        let entry_formals = state
            .may_values()
            .keys()
            .filter_map(|place| {
                entry_formal_slot(arg_count, state, place).map(|slot| (place.clone(), slot))
            })
            .collect();
        Self {
            current_validations,
            current_relations,
            entry_formals,
        }
    }

    fn proves(&self, binding: &ValidationBinding) -> bool {
        if self.current_validations.contains(binding) {
            return true;
        }
        let proof = RelationProof::from_facts(self.current_relations.iter().cloned());
        match binding.predicate {
            Predicate::InBounds => binding.collection.as_ref().is_some_and(|collection| {
                proof.proves_in_bounds(&RelationTerm::Place(binding.subject.clone()), collection)
            }),
            _ => false,
        }
    }

    fn proves_relational(&self, obligation: &RelationalObligation) -> bool {
        let proof = RelationProof::from_facts(self.current_relations.iter().cloned());
        match obligation {
            RelationalObligation::Within { base, start, end } => {
                (matches!(start, RelationTerm::Constant(0))
                    && proof.proves_le(end, &RelationTerm::Length(base.clone())))
                    || proof.proves_within(base, start, end)
            }
        }
    }

    fn entry_formal(&self, place: &PlaceKey) -> Option<BoundarySlot> {
        self.entry_formals.get(place).copied()
    }
}

#[derive(Clone, Debug)]
struct BodyFacts {
    def_id: LocalDefId,
    function: FunctionKey,
    root_span: StableSpan,
    arg_count: usize,
    has_self: bool,
    argument_utf8_capability: BTreeMap<u32, Utf8FormalCapability>,
    summary_seed: FunctionSummary,
    values: BTreeMap<PlaceKey, BodyValueFacts>,
    calls: Vec<CallSite>,
    callback_calls: BTreeSet<CandidateCallEdge>,
    compares: Vec<CompareFact>,
    lengths: BTreeSet<LengthFact>,
    predicates: Vec<PredicateFact>,
    branches: Vec<BranchFact>,
    writes: Vec<WriteFact>,
    validation_snapshots: BTreeMap<(BasicBlock, usize), ValidationSnapshot>,
    local_bindings: BTreeMap<ProgramPoint, ValidationBinding>,
    local_relational_obligations: BTreeMap<ProgramPoint, RelationalObligation>,
    concrete_layout_obligations: BTreeMap<ProgramPoint, ConcreteLayoutObligation>,
    source_local_generic_capabilities: BTreeSet<String>,
    endpoint_graph: EndpointFlowGraph,
    influence_seeds: InfluenceSeedIndex,
    structural_sites: BTreeMap<String, StructuralSiteCandidate>,
    contract_descriptors: BTreeMap<String, ContractDescriptor>,
    contract_sites_by_point: BTreeMap<ProgramPoint, BTreeSet<String>>,
    bound_sites: BTreeMap<String, BoundUnsafeSite>,
    initialized_site_operands: BTreeSet<(String, unsafe_site::OperandRole)>,
    possibly_uninitialized_site_operands: BTreeSet<(String, unsafe_site::OperandRole)>,
    initialization_witnesses:
        BTreeMap<(String, unsafe_site::OperandRole), BTreeSet<InitializationWitness>>,
    proven_contract_sites: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PreoptTransmute {
    first_failure: Operation,
    sink: Operation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ValidationRoute {
    Satisfied,
    Conditional(ValidationFact),
    Hard(ContractRequirement),
}

/// Compact identity for one finite endpoint relation during the callsite
/// propagation fixed point.  The arena owns the full relation once; both the
/// stable summary map and each seminaive delta carry only this ordered ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PropagationRelationId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PropagationOwnerId(u32);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PropagationRelation {
    owner: PropagationOwnerId,
    relation: EndpointRelation,
}

#[derive(Default)]
struct PropagationRelationArena {
    owners: Vec<Arc<FunctionKey>>,
    owner_ids: BTreeMap<Arc<FunctionKey>, PropagationOwnerId>,
    values: Vec<Arc<PropagationRelation>>,
    ids: BTreeMap<Arc<PropagationRelation>, PropagationRelationId>,
}

impl PropagationRelationArena {
    fn intern_owner(&mut self, owner: FunctionKey) -> PropagationOwnerId {
        if let Some(id) = self.owner_ids.get(&owner) {
            return *id;
        }
        let id = PropagationOwnerId(
            u32::try_from(self.owners.len()).expect("endpoint owner arena exceeds u32"),
        );
        let owner = Arc::new(owner);
        self.owners.push(owner.clone());
        self.owner_ids.insert(owner, id);
        id
    }

    fn intern(&mut self, owner: FunctionKey, relation: EndpointRelation) -> PropagationRelationId {
        let relation = PropagationRelation {
            owner: self.intern_owner(owner),
            relation,
        };
        if let Some(id) = self.ids.get(&relation) {
            return *id;
        }
        let id = PropagationRelationId(
            u32::try_from(self.values.len()).expect("endpoint relation arena exceeds u32"),
        );
        let relation = Arc::new(relation);
        self.values.push(relation.clone());
        self.ids.insert(relation, id);
        id
    }

    fn get(&self, id: PropagationRelationId) -> &PropagationRelation {
        self.values[usize::try_from(id.0).expect("relation id fits usize")].as_ref()
    }
}

struct Engine<'tcx> {
    tcx: TyCtxt<'tcx>,
    source_root: PathBuf,
    bodies: BTreeMap<FunctionKey, BodyFacts>,
    programs: BTreeMap<FunctionKey, BodyProgram>,
    preopt_transmutes: BTreeMap<FunctionKey, BTreeSet<PreoptTransmute>>,
    built_unsafe_events: BTreeMap<FunctionKey, Vec<MirEventAnchor>>,
    solver_unsafe_events: BTreeMap<FunctionKey, Vec<MirEventAnchor>>,
    binding_records: BTreeMap<String, SiteBindingRecord>,
    /// High-recall compatibility graph. It retains synthetic shared
    /// call-boundary bridges solely so pre-context candidate enumeration does
    /// not silently lose a route while callers migrate to ContextProofIndex.
    coarse_endpoint_graph: EndpointFlowGraph,
    endpoint_graph: EndpointFlowGraph,
    context_proofs: ContextProofIndex,
}

pub(super) struct AnalysisOutput {
    pub structural_candidates: Vec<StructuralFindingCandidate>,
    pub suppressed_structural_candidates: Vec<SuppressedStructuralCandidate>,
    pub unestablished_structural_candidates: Vec<UnestablishedStructuralCandidate>,
    pub missing_sanitizer_warnings: Vec<MissingSanitizerWarning>,
    pub binding_records: BTreeMap<String, SiteBindingRecord>,
    pub endpoint_graph: EndpointFlowGraph,
    pub owner_spans: BTreeMap<FunctionKey, StableSpan>,
    pub owner_code_spans: BTreeMap<FunctionKey, StableSpan>,
}

pub fn analyze(tcx: TyCtxt<'_>, trace: &mut PerfTrace) -> Result<AnalysisOutput, String> {
    Engine::new(tcx)?.run(trace)
}

fn normalized_relative_span_path(path: &str) -> Option<PathBuf> {
    if path.is_empty()
        || path.contains('\\')
        || path.contains('\0')
        || path.starts_with('/')
        || path.as_bytes().get(1) == Some(&b':')
        || path
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return None;
    }
    let path = PathBuf::from(path);
    let normalized = path
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)));
    normalized.then_some(path)
}

pub(super) fn select_source_root(
    project_root: &Path,
    configured_source_root: Option<PathBuf>,
) -> Result<PathBuf, String> {
    let project_root = project_root.canonicalize().map_err(|error| {
        format!(
            "UnsoundAudit v2 project root cannot be canonicalized '{}': {error}",
            project_root.display()
        )
    })?;
    let source_root = configured_source_root.unwrap_or_else(|| project_root.clone());
    let source_root = source_root.canonicalize().map_err(|error| {
        format!(
            "UnsoundAudit v2 source root cannot be canonicalized '{}': {error}",
            source_root.display()
        )
    })?;
    if !project_root.starts_with(&source_root) {
        return Err(format!(
            "UnsoundAudit v2 source root '{}' does not contain project root '{}'",
            source_root.display(),
            project_root.display()
        ));
    }
    Ok(source_root)
}

pub(super) fn stable_span_from_locations(
    source_root: &Path,
    local_path: Option<&str>,
    remapped_path: Option<&str>,
    lo_line: usize,
    lo_column: usize,
    hi_line: usize,
    hi_column: usize,
) -> Result<StableSpan, String> {
    let local_relative = local_path.and_then(|local| {
        let local = Path::new(local);
        let relative = match local.canonicalize() {
            Ok(canonical) => canonical.strip_prefix(source_root).ok().map(PathBuf::from),
            Err(_) => local.strip_prefix(source_root).ok().map(PathBuf::from),
        };
        relative.and_then(|relative| normalized_relative_span_path(&relative.to_string_lossy()))
    });
    let relative = local_relative.or_else(|| remapped_path.and_then(normalized_relative_span_path));
    let relative = relative.ok_or_else(|| {
        format!(
            "UnsoundAudit v2 cannot form a stable repository-relative span path: local={local_path:?}, remapped={remapped_path:?}"
        )
    })?;
    if lo_line == 0
        || hi_line < lo_line
        || lo_column == 0
        || hi_column == 0
        || (hi_line == lo_line && hi_column < lo_column)
    {
        return Err(format!(
            "UnsoundAudit v2 received a zero-based or invalid source range: {lo_line}:{lo_column}..={hi_line}:{hi_column}"
        ));
    }
    let lo_line = u32::try_from(lo_line)
        .map_err(|_| "UnsoundAudit v2 source start line exceeds u32".to_string())?;
    let hi_line = u32::try_from(hi_line)
        .map_err(|_| "UnsoundAudit v2 source end line exceeds u32".to_string())?;
    let lo_column = u32::try_from(lo_column)
        .map_err(|_| "UnsoundAudit v2 source start column exceeds u32".to_string())?;
    let hi_column = u32::try_from(hi_column)
        .map_err(|_| "UnsoundAudit v2 source end column exceeds u32".to_string())?;
    Ok(StableSpan::new(
        relative.to_string_lossy(),
        StablePosition::new(lo_line, lo_column),
        StablePosition::new(hi_line, hi_column),
    ))
}

impl<'tcx> Engine<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Result<Self, String> {
        let project_root = env::var_os("UNSOUND_SCANNER_PROJECT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let source_root = select_source_root(
            &project_root,
            env::var_os("UNSOUND_SCANNER_SOURCE_ROOT").map(PathBuf::from),
        )?;
        Ok(Self {
            tcx,
            source_root,
            bodies: BTreeMap::new(),
            programs: BTreeMap::new(),
            preopt_transmutes: BTreeMap::new(),
            built_unsafe_events: BTreeMap::new(),
            solver_unsafe_events: BTreeMap::new(),
            binding_records: BTreeMap::new(),
            coarse_endpoint_graph: EndpointFlowGraph::default(),
            endpoint_graph: EndpointFlowGraph::default(),
            context_proofs: ContextProofIndex::default(),
        })
    }

    fn run(mut self, trace: &mut PerfTrace) -> Result<AnalysisOutput, String> {
        trace.event("collect_ids", "begin", [])?;
        let ids = self.local_function_ids();
        let binding_ids = self.local_binding_ids()?;
        trace.event("collect_ids", "end", [("function_count", ids.len() as u64)])?;
        trace.event("preopt", "begin", [])?;
        self.preopt_transmutes = self.collect_preopt_transmutes(&ids)?;
        self.built_unsafe_events = unsafe_site::census_snapshot()?.built_events;
        trace.event(
            "preopt",
            "end",
            [("function_count", self.preopt_transmutes.len() as u64)],
        )?;
        trace.event("collect_bodies", "begin", [])?;
        self.collect_bodies(&ids)?;
        drop(ids);
        self.bind_supplemental_owners(&binding_ids)?;
        drop(binding_ids);
        self.account_for_unprocessed_census_owners()?;
        unsafe_site::binding::maybe_write_binding_sidecar(
            &self.binding_records,
            &self.built_unsafe_events,
            &self.solver_unsafe_events,
        )?;
        let program_point_count = self
            .programs
            .values()
            .flat_map(|program| program.operations.iter())
            .map(Vec::len)
            .sum::<usize>();
        trace.event(
            "collect_bodies",
            "end",
            [
                ("body_count", self.bodies.len() as u64),
                ("program_point_count", program_point_count as u64),
            ],
        )?;
        trace.event("call_graph", "begin", [])?;
        let graph = self.call_graph();
        let call_edge_count = graph.values().map(BTreeSet::len).sum::<usize>();
        trace.event(
            "call_graph",
            "end",
            [
                ("function_count", graph.len() as u64),
                ("call_edge_count", call_edge_count as u64),
            ],
        )?;
        // The historical whole-summary solver no longer consumes this graph;
        // keep it only for the trace-compatible collection measurement, then
        // release its dense call adjacency before endpoint construction.
        drop(graph);
        trace.event("out_must", "begin", [])?;
        let must_outputs = self.solve_local_out_must();
        let definite_out_count = must_outputs.values().map(BTreeSet::len).sum::<usize>();
        trace.event(
            "out_must",
            "end",
            [("definite_out_count", definite_out_count as u64)],
        )?;
        let mut local_outputs = self.local_out_effects(&must_outputs);
        let initial_owners = local_outputs
            .keys()
            .chain(
                must_outputs
                    .iter()
                    .filter(|(_, outputs)| !outputs.is_empty())
                    .map(|(function, _)| function),
            )
            .cloned()
            .collect::<BTreeSet<_>>();
        trace.event("initial_rebuild", "begin", [])?;
        self.rebuild_body_owners(
            &initial_owners,
            &local_outputs,
            &must_outputs,
            ExtractionPhase::Outputs,
        );
        trace.event(
            "initial_rebuild",
            "end",
            [("rebuilt_owner_count", initial_owners.len() as u64)],
        )?;
        drop(initial_owners);
        let mut outer_iteration = 0_u64;
        loop {
            outer_iteration += 1;
            trace.event("derive_outputs", "begin", [("iteration", outer_iteration)])?;
            let derived = self.derive_local_outputs(&must_outputs);
            let changed_owners = join_local_outputs(&mut local_outputs, derived);
            trace.event(
                "derive_outputs",
                "end",
                [
                    ("iteration", outer_iteration),
                    ("changed_owner_count", changed_owners.len() as u64),
                ],
            )?;
            if changed_owners.is_empty() {
                break;
            }
            trace.event("output_rebuild", "begin", [("iteration", outer_iteration)])?;
            self.rebuild_body_owners(
                &changed_owners,
                &local_outputs,
                &must_outputs,
                ExtractionPhase::Outputs,
            );
            trace.event(
                "output_rebuild",
                "end",
                [
                    ("iteration", outer_iteration),
                    ("rebuilt_owner_count", changed_owners.len() as u64),
                ],
            )?;
        }
        let final_owners = self.programs.keys().cloned().collect::<BTreeSet<_>>();
        trace.event(
            "final_rebuild",
            "begin",
            [("owner_count", final_owners.len() as u64)],
        )?;
        self.rebuild_body_owners(
            &final_owners,
            &local_outputs,
            &must_outputs,
            ExtractionPhase::Final,
        );
        trace.event(
            "final_rebuild",
            "end",
            [("owner_count", final_owners.len() as u64)],
        )?;
        drop(final_owners);
        self.verify_legacy_shadow_reconciliation()?;
        // Programs and collection-only census inputs are consumed by the
        // final rebuild and reconciliation above.  Structural candidate
        // construction uses BodyFacts, endpoint graphs, and context proofs;
        // retaining these source programs through its peak is unnecessary.
        drop(std::mem::take(&mut self.programs));
        drop(std::mem::take(&mut self.preopt_transmutes));
        drop(std::mem::take(&mut self.built_unsafe_events));
        drop(std::mem::take(&mut self.solver_unsafe_events));
        drop(local_outputs);
        drop(must_outputs);
        release_allocator_pressure();
        trace.event("endpoint_flow", "begin", [])?;
        self.instantiate_callsite_endpoint_relations(trace)?;
        // The relation/delta/call-index worklists are now dead. Release their
        // allocator pressure before aggregating the endpoint graph and making
        // the compatibility coarse-graph copy.
        release_allocator_pressure();
        let mut endpoint_graph = EndpointFlowGraph::default();
        for facts in self.bodies.values_mut() {
            endpoint_graph.append(std::mem::take(&mut facts.endpoint_graph));
        }
        self.endpoint_graph = endpoint_graph;
        trace.event(
            "endpoint_context_index",
            "begin",
            [(
                "raw_predecessor_count",
                self.context_proofs.predecessor_count() as u64,
            )],
        )?;
        trace.event(
            "context_record_locals",
            "begin",
            [(
                "endpoint_edge_count",
                self.endpoint_graph.edges.len() as u64,
            )],
        )?;
        for edge in &self.endpoint_graph.edges {
            if edge.kind != EndpointEdgeKind::CallSummary {
                self.context_proofs.record_local_ref(edge);
            }
        }
        trace.event(
            "context_record_locals",
            "end",
            [(
                "raw_predecessor_count",
                self.context_proofs.predecessor_count() as u64,
            )],
        )?;
        // Recording locals temporarily clones the aggregated edge set into
        // the proof index. Its iterator and insertion temporaries are dead
        // before exact closure, so return allocator caches before that next
        // measured peak.
        release_allocator_pressure();
        trace.event(
            "context_finalize_exact",
            "begin",
            [(
                "raw_predecessor_count",
                self.context_proofs.predecessor_count() as u64,
            )],
        )?;
        self.context_proofs.finalize();
        trace.event(
            "context_finalize_exact",
            "end",
            [(
                "exact_predecessor_count",
                self.context_proofs.exact_predecessor_count() as u64,
            )],
        )?;
        trace.event(
            "endpoint_context_index",
            "end",
            [
                (
                    "raw_predecessor_count",
                    self.context_proofs.predecessor_count() as u64,
                ),
                (
                    "exact_predecessor_count",
                    self.context_proofs.exact_predecessor_count() as u64,
                ),
            ],
        )?;
        // The high-water finalize worklists are gone before the independent
        // compatibility graph is cloned.
        release_allocator_pressure();
        trace.event(
            "endpoint_coarse_graph",
            "begin",
            [
                (
                    "endpoint_node_count",
                    self.endpoint_graph.nodes.len() as u64,
                ),
                (
                    "endpoint_edge_count",
                    self.endpoint_graph.edges.len() as u64,
                ),
            ],
        )?;
        self.coarse_endpoint_graph = self.build_coarse_endpoint_graph();
        trace.event(
            "endpoint_coarse_graph",
            "end",
            [
                (
                    "coarse_node_count",
                    self.coarse_endpoint_graph.nodes.len() as u64,
                ),
                (
                    "coarse_edge_count",
                    self.coarse_endpoint_graph.edges.len() as u64,
                ),
            ],
        )?;
        trace.event(
            "endpoint_flow",
            "end",
            [
                (
                    "endpoint_node_count",
                    self.endpoint_graph.nodes.len() as u64,
                ),
                (
                    "endpoint_edge_count",
                    self.endpoint_graph.edges.len() as u64,
                ),
            ],
        )?;
        release_allocator_pressure();
        trace.event("structural_candidates", "begin", [])?;
        let pre_discharge_structural_candidates = self.structural_candidates(trace)?;
        // The coarse compatibility graph and context proof index reach their
        // final semantic use inside structural candidate construction.  MVP
        // emission needs only the precise endpoint graph for cycle witnesses;
        // retaining these two much larger indexes through discharge and JSON
        // materialization only extends their lifetime.
        drop(std::mem::take(&mut self.coarse_endpoint_graph));
        drop(std::mem::take(&mut self.context_proofs));
        release_allocator_pressure();
        trace.event(
            "structural_candidate_order",
            "begin",
            [
                (
                    "candidate_count",
                    pre_discharge_structural_candidates.active.len() as u64,
                ),
                (
                    "suppression_count",
                    pre_discharge_structural_candidates.suppressed.len() as u64,
                ),
                (
                    "unestablished_count",
                    pre_discharge_structural_candidates.unestablished.len() as u64,
                ),
            ],
        )?;
        let pre_discharge_structural_candidates =
            pre_discharge_structural_candidates.into_global_candidate_order();
        trace.event(
            "structural_candidate_order",
            "end",
            [
                (
                    "candidate_count",
                    pre_discharge_structural_candidates.active.len() as u64,
                ),
                (
                    "suppression_count",
                    pre_discharge_structural_candidates.suppressed.len() as u64,
                ),
                (
                    "unestablished_count",
                    pre_discharge_structural_candidates.unestablished.len() as u64,
                ),
            ],
        )?;
        trace.event(
            "structural_candidates",
            "end",
            [
                (
                    "candidate_count",
                    pre_discharge_structural_candidates.active.len() as u64,
                ),
                (
                    "opaque_candidate_count",
                    pre_discharge_structural_candidates
                        .active
                        .iter()
                        .filter(|candidate| {
                            candidate.contract().precision == unsafe_site::ContractPrecision::Opaque
                        })
                        .count() as u64,
                ),
            ],
        )?;
        let structural_candidates =
            self.discharge_structural_candidates(pre_discharge_structural_candidates);
        let lifecycle_enabled = env::var_os(CANDIDATE_LIFECYCLE_TRACE_ENV).is_some();
        let mut candidate_lifecycle = if lifecycle_enabled {
            self.bodies
                .iter()
                .flat_map(|(owner, facts)| {
                    facts
                        .summary_seed
                        .requirements
                        .iter()
                        .map(|requirement| {
                            let mut event =
                                candidate_lifecycle_event("body_seed", owner, requirement);
                            event.insert("body_included".into(), true.into());
                            event.insert("dangerous_call_recognized".into(), true.into());
                            event.insert("requirement_generated".into(), true.into());
                            let point = requirement.sink.point.point_location();
                            let binding = facts.local_bindings.get(&requirement.sink.point);
                            let snapshot = facts
                                .validation_snapshots
                                .get(&(point.block, point.statement_index));
                            event.insert("local_binding".into(), format!("{binding:?}").into());
                            event.insert(
                                "validation_snapshot".into(),
                                format!("{snapshot:?}").into(),
                            );
                            event.insert(
                                "snapshot_proves_binding".into(),
                                snapshot
                                    .zip(binding)
                                    .is_some_and(|(snapshot, binding)| snapshot.proves(binding))
                                    .into(),
                            );
                            event.insert(
                                "predicate_facts".into(),
                                format!("{:?}", facts.predicates).into(),
                            );
                            event
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        for (owner, facts) in self.bodies.iter().filter(|_| lifecycle_enabled) {
            for call in &facts.calls {
                candidate_lifecycle.push(BTreeMap::from([
                    ("stage".into(), "call_model".into()),
                    ("owner".into(), owner.0.clone().into()),
                    ("point".into(), call.point.span.token().into()),
                    (
                        "disposition".into(),
                        format!("{:?}", call.disposition).into(),
                    ),
                    (
                        "callee".into(),
                        call.callee
                            .as_ref()
                            .map_or(serde_json::Value::Null, |callee| callee.0.clone().into()),
                    ),
                ]));
            }
            for relation in &facts.summary_seed.relational_postconditions {
                if !matches!(relation.relation, SummaryRelationKind::SameObject(_, _)) {
                    continue;
                }
                candidate_lifecycle.push(BTreeMap::from([
                    ("stage".into(), "relation_summary".into()),
                    ("owner".into(), owner.0.clone().into()),
                    ("relation".into(), format!("{relation:?}").into()),
                ]));
            }
            for call in facts
                .calls
                .iter()
                .filter(|call| call.disposition == BoundaryDisposition::ExactFfiOut)
            {
                candidate_lifecycle.push(BTreeMap::from([
                    ("stage".into(), "ffi_boundary".into()),
                    ("owner".into(), owner.0.clone().into()),
                    ("point".into(), call.point.span.token().into()),
                    (
                        "actual_exact_roots".into(),
                        format!(
                            "{:?}",
                            call.arg_values
                                .iter()
                                .map(|value| &value.exact_roots)
                                .collect::<Vec<_>>()
                        )
                        .into(),
                    ),
                    (
                        "actual_value_flow".into(),
                        format!(
                            "{:?}",
                            call.arg_values
                                .iter()
                                .map(|value| &value.value_flow)
                                .collect::<Vec<_>>()
                        )
                        .into(),
                    ),
                ]));
            }
        }
        for (owner, facts) in self.bodies.iter().filter(|_| lifecycle_enabled) {
            for call in &facts.calls {
                if call.disposition != BoundaryDisposition::ResolvedLocal {
                    continue;
                }
                let Some(callee) = call.callee.as_ref() else {
                    continue;
                };
                let Some(callee_facts) = self.bodies.get(callee) else {
                    continue;
                };
                let actuals = call
                    .arg_values
                    .iter()
                    .map(|value| value.value.clone())
                    .collect::<Vec<_>>();
                for requirement in callee_facts
                    .summary_seed
                    .requirements
                    .iter()
                    .filter(|requirement| requirement.rule == RuleId::P3FfiOutInvalidValue)
                {
                    let mut instantiated = requirement.clone();
                    instantiated.subject = instantiated.subject.substitute(&actuals);
                    let mut event =
                        candidate_lifecycle_event("call_boundary", owner, &instantiated);
                    event.insert("callee".into(), callee.0.clone().into());
                    event.insert(
                        "destination_restricted".into(),
                        call.destination_restricted.into(),
                    );
                    event.insert(
                        "actual_ffi_output".into(),
                        call.arg_values
                            .iter()
                            .map(|value| value.ffi_output)
                            .collect::<Vec<_>>()
                            .into(),
                    );
                    event.insert(
                        "actual_origins".into(),
                        format!(
                            "{:?}",
                            call.arg_values
                                .iter()
                                .map(|value| &value.value.origins)
                                .collect::<Vec<_>>()
                        )
                        .into(),
                    );
                    candidate_lifecycle.push(event);
                }
            }
        }
        trace.event(
            "finding_materialize",
            "begin",
            [("body_count", self.bodies.len() as u64)],
        )?;
        if lifecycle_enabled {
            candidate_lifecycle.extend(structural_candidates.active.iter().map(|candidate| {
                BTreeMap::from([
                    ("stage".into(), "v3_emission".into()),
                    ("owner".into(), candidate.public_root.0.clone().into()),
                    ("site_id".into(), candidate.site().seed.site_id().into()),
                    (
                        "primary_pattern".into(),
                        format!("{:?}", candidate.primary_pattern).into(),
                    ),
                    (
                        "endpoint_route".into(),
                        format!("{:?}", candidate.endpoint_route.as_ref()).into(),
                    ),
                    ("emitted".into(), true.into()),
                ])
            }));
        }
        write_candidate_lifecycle_trace(&candidate_lifecycle)?;
        trace.event(
            "finding_materialize",
            "end",
            [("finding_count", structural_candidates.active.len() as u64)],
        )?;
        let owner_spans = self
            .bodies
            .iter()
            .map(|(owner, facts)| (owner.clone(), facts.root_span.clone()))
            .collect();
        let owner_code_spans = self
            .bodies
            .iter()
            .map(|(owner, facts)| {
                self.owner_code_span(facts)
                    .map(|span| (owner.clone(), span))
            })
            .collect::<Result<_, _>>()?;
        let missing_sanitizer_warnings =
            self.collect_missing_sanitizer_warnings(&structural_candidates.active);
        Ok(AnalysisOutput {
            structural_candidates: structural_candidates.active,
            suppressed_structural_candidates: structural_candidates.suppressed,
            unestablished_structural_candidates: structural_candidates.unestablished,
            missing_sanitizer_warnings,
            binding_records: self.binding_records,
            endpoint_graph: self.endpoint_graph,
            owner_spans,
            owner_code_spans,
        })
    }

    /// Instantiate finite callee endpoint relations at their exact resolved
    /// callsites.  The raw body graphs deliberately retain local
    /// `Formal -> ... -> Return/Out` evidence, but never connect an actual to
    /// that shared Formal.  This fixed point supplies the missing atomic
    /// bridge as `actual-source -> CallResult(callsite, output)` (and directly
    /// to a callee unsafe operand/exposure).  Consequently a result produced
    /// at call A cannot be consumed as the result of call B.
    ///
    /// The relation domain is finite: it is drawn only from materialized
    /// endpoint nodes and resolved local callsites.  We iterate to equality,
    /// rather than imposing a wrapper-depth/recursion-round cap.
    fn instantiate_callsite_endpoint_relations(
        &mut self,
        trace: &mut PerfTrace,
    ) -> Result<(), String> {
        self.context_proofs = ContextProofIndex::default();
        #[derive(Clone)]
        struct ResolvedCall {
            caller: FunctionKey,
            id: CallSiteId,
            actual_sources: Vec<BTreeSet<EndpointNode>>,
            generic_substitution: GenericCallSubstitution,
            typed_witness_templates: BTreeSet<TypedWitnessTransportTemplate>,
        }
        let calls = self
            .bodies
            .iter()
            .flat_map(|(caller, facts)| {
                facts.calls.iter().filter_map(move |call| {
                    (call.disposition == BoundaryDisposition::ResolvedLocal)
                        .then_some(call.callee.as_ref())
                        .flatten()
                        .map(|callee| ResolvedCall {
                            caller: caller.clone(),
                            id: CallSiteId::new(caller.clone(), call.point.clone(), callee.clone()),
                            actual_sources: call.actual_endpoint_sources.clone(),
                            generic_substitution: call.generic_substitution.clone(),
                            typed_witness_templates: call.typed_witness_templates(),
                        })
                })
            })
            .collect::<Vec<_>>();
        if calls.is_empty() {
            return Ok(());
        }

        let excluded = BTreeSet::from([EndpointEdgeKind::ActualToFormal]);
        trace.event(
            "endpoint_context_seed",
            "begin",
            [
                ("body_count", self.bodies.len() as u64),
                ("resolved_call_count", calls.len() as u64),
            ],
        )?;
        let mut relation_arena = PropagationRelationArena::default();
        let mut relation_deltas = BTreeMap::<FunctionKey, BTreeSet<PropagationRelationId>>::new();
        for (owner, facts) in &self.bodies {
            let seed = endpoint_relations_for_owner(owner, &facts.endpoint_graph, &excluded)
                .into_iter()
                .map(|relation| relation_arena.intern(owner.clone(), relation))
                .collect();
            relation_deltas.insert(owner.clone(), seed);
        }
        // `relations` owns only completed rounds.  Keeping the current delta
        // separate lets the seed move into this map after its one propagation
        // pass, rather than cloning every seed relation at initialization.
        let mut relations = self
            .bodies
            .keys()
            .cloned()
            .map(|owner| (owner, BTreeSet::new()))
            .collect::<BTreeMap<_, BTreeSet<_>>>();
        let calls_by_callee = calls.iter().fold(
            BTreeMap::<FunctionKey, Vec<&ResolvedCall>>::new(),
            |mut grouped, call| {
                grouped
                    .entry(call.id.callee.clone())
                    .or_default()
                    .push(call);
                grouped
            },
        );
        for call in &calls {
            if !relations.contains_key(&call.id.callee) {
                let outputs = std::iter::once(CallSiteOutput::Return)
                    .chain((1..=call.actual_sources.len() as u32).map(CallSiteOutput::Out));
                for output in outputs {
                    self.context_proofs.mark_unknown_at(
                        EndpointNode::CallResult {
                            callsite: call.id.clone(),
                            output,
                        },
                        ContextUnknownReason::MissingResolvedBody {
                            callsite: call.id.clone(),
                        },
                    );
                }
            }
        }
        // Seminaive summary propagation: a newly discovered finite callee
        // relation is instantiated at every concrete callsite exactly once.
        // Replaying the complete callee summary after every growth round was
        // extensionally idempotent, but retained an enormous duplicate set of
        // call-proof records until the fixed point completed.
        trace.event(
            "endpoint_context_seed",
            "end",
            [
                ("relation_count", relation_arena.values.len() as u64),
                (
                    "seed_relation_count",
                    relation_deltas.values().map(BTreeSet::len).sum::<usize>() as u64,
                ),
            ],
        )?;
        trace.event(
            "endpoint_context_propagation",
            "begin",
            [
                ("resolved_call_count", calls.len() as u64),
                (
                    "seed_relation_count",
                    relation_deltas.values().map(BTreeSet::len).sum::<usize>() as u64,
                ),
            ],
        )?;
        let mut delta_round_count = 0_u64;
        let mut propagated_call_predecessor_count = 0_u64;
        while !relation_deltas.is_empty() {
            delta_round_count += 1;
            let mut dirty_callers = BTreeSet::new();
            for (callee, callee_relations) in &relation_deltas {
                for call in calls_by_callee.get(callee).into_iter().flatten() {
                    for relation_id in callee_relations {
                        let relation = &relation_arena.get(*relation_id).relation;
                        if let EndpointNode::GenericCapability { evidence, .. } = &relation.input {
                            self.context_proofs.record_typed_witness_transports(
                                call.id.clone(),
                                evidence.source_identity(),
                                call.typed_witness_templates.clone(),
                            );
                        }
                        let destination = match &relation.terminal {
                            EndpointNode::Return { .. } => EndpointNode::CallResult {
                                callsite: call.id.clone(),
                                output: CallSiteOutput::Return,
                            },
                            EndpointNode::Out { index, .. } => EndpointNode::CallResult {
                                callsite: call.id.clone(),
                                output: CallSiteOutput::Out(*index),
                            },
                            terminal => terminal.clone(),
                        };
                        let (inputs, actual_binding) = match &relation.input {
                            EndpointNode::Formal { owner, index } if owner == &call.id.callee => {
                                let inputs = call
                                    .actual_sources
                                    .get((*index as usize).saturating_sub(1))
                                    .cloned()
                                    .unwrap_or_default();
                                (inputs, ContextActualBinding::Formal(*index))
                            }
                            EndpointNode::Return { .. } | EndpointNode::Out { .. } => {
                                (BTreeSet::new(), ContextActualBinding::Internal)
                            }
                            input => (
                                BTreeSet::from([input.clone()]),
                                ContextActualBinding::Internal,
                            ),
                        };
                        if inputs.is_empty()
                            && matches!(actual_binding, ContextActualBinding::Formal(_))
                        {
                            self.context_proofs.mark_unknown_at(
                                destination.clone(),
                                ContextUnknownReason::MissingActualEndpoint {
                                    callsite: call.id.clone(),
                                },
                            );
                        }
                        for source in inputs {
                            let edge = EndpointEdge {
                                from: source,
                                to: destination.clone(),
                                kind: EndpointEdgeKind::CallSummary,
                                point: Some(call.id.point.clone()),
                            };
                            // `callee_relations` is a seminaive delta, so this
                            // alternative has not been seen at this callsite in
                            // an earlier round. Record it immediately; the proof
                            // index retains all alternatives and deduplicates
                            // only equal complete facts.
                            self.context_proofs.record_call(
                                edge.clone(),
                                call.id.clone(),
                                call.generic_substitution.clone(),
                                ContextRelationKey::new(
                                    call.id.callee.clone(),
                                    relation.input.clone(),
                                    relation.terminal.clone(),
                                ),
                                actual_binding,
                            );
                            propagated_call_predecessor_count += 1;
                            let facts = self
                                .bodies
                                .get_mut(&call.caller)
                                .expect("resolved call caller has body facts");
                            if facts.endpoint_graph.insert_edge(edge) {
                                dirty_callers.insert(call.caller.clone());
                            }
                        }
                    }
                }
            }
            // The propagated delta is now complete. Move it into the stable
            // relation map before deriving fresh deltas, avoiding a full seed
            // clone and retaining only one owner of each completed relation.
            for (owner, mut delta) in relation_deltas {
                relations.entry(owner).or_default().append(&mut delta);
            }
            let mut next_deltas = BTreeMap::<FunctionKey, BTreeSet<_>>::new();
            for owner in dirty_callers {
                let Some(facts) = self.bodies.get(&owner) else {
                    continue;
                };
                let updated =
                    endpoint_relations_for_owner(&owner, &facts.endpoint_graph, &excluded)
                        .into_iter()
                        .map(|relation| relation_arena.intern(owner.clone(), relation))
                        .collect::<BTreeSet<_>>();
                let previous = relations
                    .get(&owner)
                    .expect("every body has a relation set");
                let delta = updated
                    .difference(previous)
                    .cloned()
                    .collect::<BTreeSet<_>>();
                if !delta.is_empty() {
                    next_deltas.insert(owner, delta);
                }
            }
            relation_deltas = next_deltas;
        }
        trace.event(
            "endpoint_context_propagation",
            "end",
            [
                ("delta_round_count", delta_round_count),
                (
                    "propagated_call_predecessor_count",
                    propagated_call_predecessor_count,
                ),
                (
                    "final_relation_count",
                    relations.values().map(BTreeSet::len).sum::<usize>() as u64,
                ),
            ],
        )?;
        Ok(())
    }

    /// Builds the compatibility/superset graph only after endpoint extraction
    /// has finished.  It copies the final compact graph (rather than any MIR
    /// value state) and restores the pre-context shared boundary bridges. The
    /// context proof index never reads these edges; they exist so candidate
    /// enumeration can migrate incrementally without a silent precision-loss
    /// regression.
    fn build_coarse_endpoint_graph(&self) -> EndpointFlowGraph {
        let mut coarse = self.endpoint_graph.clone();
        for (caller, facts) in &self.bodies {
            for call in facts
                .calls
                .iter()
                .filter(|call| call.disposition == BoundaryDisposition::ResolvedLocal)
            {
                let Some(callee) = &call.callee else {
                    continue;
                };
                let callsite = CallSiteId::new(caller.clone(), call.point.clone(), callee.clone());
                for (offset, sources) in call.actual_endpoint_sources.iter().enumerate() {
                    let formal = EndpointNode::Formal {
                        owner: callee.clone(),
                        index: offset as u32 + 1,
                    };
                    for source in sources {
                        coarse.insert_edge(EndpointEdge {
                            from: source.clone(),
                            to: formal.clone(),
                            kind: EndpointEdgeKind::ActualToFormal,
                            point: Some(call.point.clone()),
                        });
                    }
                }
                let return_port = EndpointNode::Return {
                    owner: callee.clone(),
                };
                coarse.insert_edge(EndpointEdge {
                    from: return_port,
                    to: EndpointNode::CallResult {
                        callsite: callsite.clone(),
                        output: CallSiteOutput::Return,
                    },
                    kind: EndpointEdgeKind::ReturnToDestination,
                    point: Some(call.point.clone()),
                });
                // Output endpoints are meaningful only on the normal return
                // edge.  No synthetic bridge is introduced for unwind.
                for index in 1..=call.actual_endpoint_sources.len() as u32 {
                    coarse.insert_edge(EndpointEdge {
                        from: EndpointNode::Out {
                            owner: callee.clone(),
                            index,
                        },
                        to: EndpointNode::CallResult {
                            callsite: callsite.clone(),
                            output: CallSiteOutput::Out(index),
                        },
                        kind: EndpointEdgeKind::OutToCaller,
                        point: Some(call.point.clone()),
                    });
                }
            }
        }
        coarse
    }

    fn owner_code_span(&self, facts: &BodyFacts) -> Result<StableSpan, String> {
        let body_span = hir_body_value_span(self.tcx, facts.def_id).ok_or_else(|| {
            format!(
                "cannot locate HIR body for causal function {}",
                facts.function.0
            )
        })?;
        merge_function_code_span(&facts.root_span, &self.stable_span(body_span)?)
    }

    fn collect_missing_sanitizer_warnings(
        &self,
        active: &[StructuralFindingCandidate],
    ) -> Vec<MissingSanitizerWarning> {
        let owners = self
            .bodies
            .iter()
            .map(|(function, facts)| {
                (
                    function.clone(),
                    CandidateOwner {
                        function: function.clone(),
                        root_span: facts.root_span.clone(),
                        public_safe: self.is_public_safe_root(facts.def_id),
                        generic_surface: self
                            .is_public_safe_root(facts.def_id)
                            .then(|| self.generic_surface(facts.def_id.to_def_id()))
                            .unwrap_or_default(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let calls = self
            .bodies
            .iter()
            .flat_map(|(caller, facts)| {
                facts.calls.iter().filter_map(move |call| {
                    (call.disposition == BoundaryDisposition::ResolvedLocal)
                        .then(|| call.callee.clone())
                        .flatten()
                        .map(|callee| CandidateCallEdge {
                            caller: caller.clone(),
                            callee,
                            point: call.point.clone(),
                            kind: CandidateCallKind::ResolvedLocal,
                            generic_substitution: call.generic_substitution.clone(),
                        })
                })
            })
            .chain(
                self.bodies
                    .values()
                    .flat_map(|facts| facts.callback_calls.iter().cloned()),
            )
            .collect::<BTreeSet<_>>();
        let call_index = CallRouteIndex::new(&calls);
        let claimed = active
            .iter()
            .map(|candidate| {
                (
                    candidate.public_root.0.clone(),
                    candidate.site().seed.site_id(),
                )
            })
            .collect::<BTreeSet<_>>();

        let mut warnings = Vec::new();
        let mut emitted = BTreeSet::<(String, String)>::new();
        for facts in self.bodies.values() {
            for (site_id, site) in &facts.bound_sites {
                let model = facts
                    .contract_descriptors
                    .get(site_id)
                    .and_then(|descriptor| descriptor.model_id.as_deref())
                    .and_then(KnownContractModel::from_model_id);
                let Some(obligation) = classify_sink(site.seed.kind, model) else {
                    continue;
                };
                let snapshot = snapshot_for_site(facts, site_id, site);
                let subject = warning_subject_place(site, obligation.predicate);
                let proven = facts.proven_contract_sites.contains(site_id);
                if proven
                    || snapshot.as_ref().is_some_and(|snapshot| {
                        sanitizer_dominates(
                            snapshot,
                            facts,
                            site,
                            site_id,
                            obligation.predicate,
                            subject.as_ref(),
                        )
                    })
                {
                    continue;
                }
                let routes = call_index.routes_to(&site.seed.owner);
                for root in reachable_public_roots(&owners, &routes).into_keys() {
                    let key = (root.0.clone(), site_id.clone());
                    if claimed.contains(&key) || !emitted.insert(key) {
                        continue;
                    }
                    warnings.push(MissingSanitizerWarning {
                        public_root: (*root).clone(),
                        site_id: site_id.clone(),
                        predicate: obligation.predicate,
                        sink_model: obligation.sink_model.to_string(),
                    });
                }
            }
        }
        warnings.sort();
        warnings
    }

    fn structural_candidates(
        &mut self,
        trace: &mut PerfTrace,
    ) -> Result<StructuralCandidateSet, String> {
        trace.event(
            "structural_input_materialize",
            "begin",
            [("body_count", self.bodies.len() as u64)],
        )?;
        let owners = self
            .bodies
            .iter()
            .map(|(function, facts)| {
                (
                    function.clone(),
                    CandidateOwner {
                        function: function.clone(),
                        root_span: facts.root_span.clone(),
                        public_safe: self.is_public_safe_root(facts.def_id),
                        generic_surface: self
                            .is_public_safe_root(facts.def_id)
                            .then(|| self.generic_surface(facts.def_id.to_def_id()))
                            .unwrap_or_default(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let calls = self
            .bodies
            .iter()
            .flat_map(|(caller, facts)| {
                facts.calls.iter().filter_map(move |call| {
                    (call.disposition == BoundaryDisposition::ResolvedLocal)
                        .then(|| call.callee.clone())
                        .flatten()
                        .map(|callee| CandidateCallEdge {
                            caller: caller.clone(),
                            callee,
                            point: call.point.clone(),
                            kind: CandidateCallKind::ResolvedLocal,
                            generic_substitution: call.generic_substitution.clone(),
                        })
                })
            })
            .chain(
                self.bodies
                    .values()
                    .flat_map(|facts| facts.callback_calls.iter().cloned()),
            )
            .collect::<BTreeSet<_>>();
        let structural_sites = self
            .bodies
            .values_mut()
            .flat_map(|facts| std::mem::take(&mut facts.structural_sites))
            .collect::<BTreeMap<_, _>>();
        let contracts = self
            .bodies
            .values()
            .flat_map(|facts| facts.contract_descriptors.clone())
            .collect::<BTreeMap<_, _>>();
        let bound_sites = self
            .bodies
            .values()
            .flat_map(|facts| facts.bound_sites.clone())
            .collect::<BTreeMap<_, _>>();
        let mut influence_seeds = InfluenceSeedIndex::default();
        for facts in self.bodies.values() {
            influence_seeds.join(&facts.influence_seeds);
        }
        trace.event(
            "structural_input_materialize",
            "end",
            [
                ("owner_count", owners.len() as u64),
                ("candidate_call_count", calls.len() as u64),
                ("structural_site_count", structural_sites.len() as u64),
                ("contract_count", contracts.len() as u64),
                ("bound_site_count", bound_sites.len() as u64),
                ("influence_seed_count", influence_seeds.len() as u64),
            ],
        )?;
        build_structural_candidate_set_with_context_traced(
            &self.coarse_endpoint_graph,
            &structural_sites,
            &contracts,
            &bound_sites,
            &owners,
            &calls,
            Some(&self.context_proofs),
            &influence_seeds,
            trace,
        )
    }

    fn discharge_structural_candidates(
        &self,
        mut candidates: StructuralCandidateSet,
    ) -> StructuralCandidateSet {
        let mut active = Vec::new();
        for mut candidate in candidates.active {
            if let Some((reason, evidence)) =
                static_contract_suppression(candidate.contract().static_implication)
            {
                candidates
                    .suppressed
                    .push(SuppressedStructuralCandidate::new(
                        candidate, reason, evidence,
                    ));
                continue;
            }

            let site_id = candidate.site().seed.site_id();
            let Some(facts) = self.bodies.get(&candidate.site().seed.owner) else {
                active.push(candidate);
                continue;
            };
            let initialization_witnesses = facts
                .initialization_witnesses
                .iter()
                .filter(|((witness_site, _), _)| witness_site == &site_id)
                .flat_map(|(_, witnesses)| {
                    witnesses.iter().map(|witness| witness.stable_identity())
                })
                .collect::<BTreeSet<_>>();
            if let Some((reason, evidence)) = dynamic_contract_suppression(
                candidate.contract().discharge,
                &site_id,
                facts.proven_contract_sites.contains(&site_id),
                &initialization_witnesses,
            ) {
                candidates
                    .suppressed
                    .push(SuppressedStructuralCandidate::new(
                        candidate, reason, evidence,
                    ));
                continue;
            }

            if candidate.primary_pattern == unsafe_site::PrimaryPattern::P5 {
                match unsafe_result_safety(&candidate, facts) {
                    UnsafeResultSafety::ViolationWitness => {
                        candidate.classification_confidence =
                            unsafe_site::ClassificationConfidence::Exact;
                        active.push(candidate);
                    }
                    UnsafeResultSafety::ProvenSafe => {
                        candidates
                            .suppressed
                            .push(SuppressedStructuralCandidate::new(
                                candidate,
                                unsafe_site::SuppressionReason::ContractDischarged,
                                BTreeMap::from([
                                    ("proof_scope".to_string(), "unsafe_result".to_string()),
                                    (
                                        "discharge_model".to_string(),
                                        "finite_result_validity".to_string(),
                                    ),
                                ]),
                            ));
                    }
                    UnsafeResultSafety::Unknown(reason) => {
                        let requirement = result_validity_requirement(candidate.contract());
                        candidates.unestablished.push(
                            UnestablishedStructuralCandidate::from_obligation(
                                &candidate,
                                unsafe_site::OperandRole::Out,
                                "internal".to_string(),
                                "validity_bits".to_string(),
                                Some(requirement),
                                BTreeSet::from([reason]),
                            ),
                        );
                    }
                }
            } else if candidate.primary_pattern == unsafe_site::PrimaryPattern::P6 {
                match internal_derivation_safety(&candidate, facts, &self.bodies) {
                    InternalDerivationSafety::ViolationWitness => {
                        candidate.classification_confidence =
                            unsafe_site::ClassificationConfidence::Exact;
                        active.push(candidate);
                    }
                    InternalDerivationSafety::ProvenSafe => {
                        candidates
                            .suppressed
                            .push(SuppressedStructuralCandidate::new(
                                candidate,
                                unsafe_site::SuppressionReason::ContractDischarged,
                                BTreeMap::from([
                                    ("proof_scope".to_string(), "finite_relation".to_string()),
                                    (
                                        "discharge_model".to_string(),
                                        "finite_derivation_obligation".to_string(),
                                    ),
                                ]),
                            ));
                    }
                    InternalDerivationSafety::Unknown(reason) => {
                        let role = candidate
                            .operand_role
                            .clone()
                            .expect("P6 derivation candidates are operand scoped");
                        let requirement = derivation_requirement(candidate.contract(), &role);
                        candidates.unestablished.push(
                            UnestablishedStructuralCandidate::from_obligation(
                                &candidate,
                                role,
                                "internal".to_string(),
                                "finite_derivation".to_string(),
                                Some(requirement),
                                BTreeSet::from([reason]),
                            ),
                        );
                    }
                }
            } else {
                active.push(candidate);
            }
        }
        StructuralCandidateSet {
            active,
            suppressed: candidates.suppressed,
            unestablished: candidates.unestablished,
        }
    }

    fn verify_legacy_shadow_reconciliation(&self) -> Result<(), String> {
        for (owner, program) in &self.programs {
            let Some(facts) = self.bodies.get(owner) else {
                continue;
            };
            for operation in program.operations.iter().flatten() {
                let expected_model = match &operation.kind {
                    ProgramOpKind::Call(call) => call
                        .descriptor
                        .raw_def
                        .and_then(|did| self.known_contract_model(did))
                        .map(KnownContractModel::as_str),
                    ProgramOpKind::Assign(_) | ProgramOpKind::Nop => None,
                };
                let Some(expected_model) = expected_model else {
                    continue;
                };
                let point = operation.semantic_point();
                let matched = facts
                    .contract_sites_by_point
                    .get(point)
                    .into_iter()
                    .flatten()
                    .filter_map(|site_id| facts.contract_descriptors.get(site_id))
                    .any(|descriptor| descriptor.model_id.as_deref() == Some(expected_model));
                if !matched {
                    return Err(format!(
                        "legacy known operation at {} has no compiler-census site bound to model {expected_model}",
                        point.span.token()
                    ));
                }
            }
        }
        Ok(())
    }

    fn solve_local_out_must(&self) -> BTreeMap<FunctionKey, BTreeSet<u32>> {
        let mut must = self
            .programs
            .keys()
            .cloned()
            .map(|function| (function, BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        loop {
            let effects = self.local_out_effects(&must);
            let mut changed = false;
            for (function, program) in &self.programs {
                let outputs = effects.get(function).cloned().unwrap_or_default();
                let solved = solve_out_must(
                    &program.operations,
                    &program.edges,
                    &program.successors,
                    &outputs,
                );
                let return_blocks = program
                    .edges
                    .iter_enumerated()
                    .filter(|(block, edge)| {
                        matches!(edge, EdgeTerm::Return) && solved[*block].is_some()
                    })
                    .map(|(block, _)| block)
                    .collect::<Vec<_>>();
                if return_blocks.is_empty() {
                    continue;
                }
                for formal_index in program.out_formals.values() {
                    if return_blocks.iter().all(|block| {
                        solved[*block]
                            .as_ref()
                            .is_some_and(|written| written.contains(formal_index))
                    }) {
                        changed |= must
                            .entry(function.clone())
                            .or_default()
                            .insert(*formal_index);
                    }
                }
            }
            if !changed {
                return must;
            }
        }
    }

    fn local_out_effects(
        &self,
        must: &BTreeMap<FunctionKey, BTreeSet<u32>>,
    ) -> BTreeMap<FunctionKey, LocalCallOutputs> {
        let mut effects = BTreeMap::<FunctionKey, LocalCallOutputs>::new();
        for (caller_key, caller) in &self.bodies {
            let Some(program) = self.programs.get(caller_key) else {
                continue;
            };
            for boundary in &caller.summary_seed.calls {
                let callee_must = must.get(&boundary.callee);
                for mapping in &boundary.mapping {
                    let CallMapping::OutToCaller {
                        formal_index,
                        caller_place,
                    } = mapping
                    else {
                        continue;
                    };
                    let output = effects
                        .entry(caller_key.clone())
                        .or_default()
                        .entry(boundary.point.clone())
                        .or_default()
                        .outs
                        .entry(caller_place.clone())
                        .or_default();
                    let may_skip =
                        !callee_must.is_some_and(|written| written.contains(formal_index));
                    output.may_skip_write |= may_skip;
                    if let Some(slot) = program.out_formals.get(caller_place).copied().map(out_slot)
                    {
                        effects
                            .entry(caller_key.clone())
                            .or_default()
                            .entry(boundary.point.clone())
                            .or_default()
                            .outs
                            .entry(slot)
                            .or_default()
                            .may_skip_write |= may_skip;
                    }
                }
            }
        }
        effects
    }

    fn solve_current_summaries(
        &self,
        graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
    ) -> BTreeMap<FunctionKey, FunctionSummary> {
        let seeds = self.summary_seeds();
        solve_summaries(graph, &seeds, |node, current| {
            let mut derived = FunctionSummary::empty(node.clone());
            let Some(caller) = self.bodies.get(node) else {
                return derived;
            };
            for call in &caller.calls {
                if call.disposition != BoundaryDisposition::ResolvedLocal {
                    continue;
                }
                assert_eq!(
                    call.point.function, *node,
                    "a local call point must be owned by its containing summary"
                );
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
                let reaches_return = self.place_reaches_return(caller, &call.destination);
                for requirement in &callee_summary.requirements {
                    if requirement.return_exposure && !reaches_return {
                        continue;
                    }
                    let callee_route = UnresolvedRouteFact::Hard(requirement.clone());
                    let mut caller_requirement = requirement.clone();
                    caller_requirement.subject = caller_requirement.subject.substitute(&actuals);
                    caller_requirement.collection = caller_requirement
                        .collection
                        .as_ref()
                        .map(|value| value.substitute(&actuals));
                    if !p3_requirement_matches_call_destination(
                        &caller_requirement,
                        call.destination_restricted,
                    ) {
                        continue;
                    }
                    propagate_unresolved_route(
                        &mut derived,
                        node,
                        call,
                        callee,
                        UnresolvedRouteFact::Hard(caller_requirement),
                        callee_route,
                    );
                }
                for validation in &callee_summary.validations {
                    if validation.requirement.return_exposure && !reaches_return {
                        continue;
                    }
                    let callee_route = UnresolvedRouteFact::Conditional(validation.clone());
                    let Some(snapshot) = caller.validation_snapshots.get(&(
                        call.point.point_location().block,
                        call.point.statement as usize,
                    )) else {
                        propagate_unresolved_route(
                            &mut derived,
                            node,
                            call,
                            callee,
                            UnresolvedRouteFact::Hard(validation.instantiate_requirement(&actuals)),
                            callee_route,
                        );
                        continue;
                    };
                    propagate_validation_route(
                        &mut derived,
                        node,
                        call,
                        callee,
                        callee_route,
                        compose_validation_route_from_snapshot(validation, call, snapshot),
                    );
                }
                derived
                    .cycle_tokens
                    .extend(callee_summary.cycle_tokens.iter().cloned());
            }
            derived
        })
    }

    fn summary_seeds(&self) -> BTreeMap<FunctionKey, FunctionSummary> {
        self.bodies
            .iter()
            .map(|(key, facts)| (key.clone(), facts.summary_seed.clone()))
            .collect()
    }

    fn derive_local_outputs(
        &self,
        definite_outs: &BTreeMap<FunctionKey, BTreeSet<u32>>,
    ) -> BTreeMap<FunctionKey, LocalCallOutputs> {
        let mut result = BTreeMap::<FunctionKey, LocalCallOutputs>::new();
        for (caller_key, caller) in &self.bodies {
            for call in &caller.calls {
                if call.disposition != BoundaryDisposition::ResolvedLocal {
                    continue;
                }
                let Some(callee) = call.callee.as_ref() else {
                    continue;
                };
                let Some(summary) = self.bodies.get(callee).map(|facts| &facts.summary_seed) else {
                    continue;
                };
                let Some(boundary) =
                    caller.summary_seed.calls.iter().find(|boundary| {
                        boundary.callee == *callee && boundary.point == call.point
                    })
                else {
                    continue;
                };
                let actuals = call
                    .arg_values
                    .iter()
                    .map(|facts| facts.value.clone())
                    .collect::<Vec<_>>();
                let Some(program) = self.programs.get(caller_key) else {
                    continue;
                };
                let output_mapping = boundary.mapping.clone();
                let mut mapped = map_structured_local_outputs(
                    &output_mapping,
                    summary,
                    &actuals,
                    &program.out_formals,
                    definite_outs.get(callee).cloned().unwrap_or_default(),
                );
                let (relations, conditional_relations) = instantiate_relational_postconditions(
                    &summary.relational_postconditions,
                    call,
                    &output_mapping,
                );
                mapped.relations.extend(relations);
                mapped.conditional_relations.extend(conditional_relations);
                let outputs = result
                    .entry(caller_key.clone())
                    .or_default()
                    .entry(call.point.clone())
                    .or_default();
                if let Some((place, value)) = mapped.returned {
                    let returned = outputs
                        .returned
                        .get_or_insert_with(|| (place.clone(), AbstractValue::default()));
                    debug_assert_eq!(returned.0, place);
                    returned.1.origins.extend(value.origins);
                }
                for (place, value) in mapped.outs {
                    let output = outputs.outs.entry(place).or_default();
                    output.value.origins.extend(value.value.origins);
                    output.may_skip_write |= value.may_skip_write;
                }
                outputs.relations.extend(mapped.relations);
                outputs
                    .conditional_relations
                    .extend(mapped.conditional_relations);
            }
        }
        result
    }

    fn rebuild_body_owners(
        &mut self,
        owners: &BTreeSet<FunctionKey>,
        local_outputs: &BTreeMap<FunctionKey, LocalCallOutputs>,
        definite_outs: &BTreeMap<FunctionKey, BTreeSet<u32>>,
        phase: ExtractionPhase,
    ) {
        let empty_outputs = LocalCallOutputs::new();
        let empty_definite = BTreeSet::new();
        for function in owners {
            let Some(program) = self.programs.get(function) else {
                continue;
            };
            let outputs = local_outputs.get(function).unwrap_or(&empty_outputs);
            let definite = definite_outs.get(function).unwrap_or(&empty_definite);
            self.bodies.remove(function);
            let facts = self.extract_program(program, outputs, definite, phase);
            self.bodies.insert(function.clone(), facts);
        }
    }

    fn local_function_ids(&self) -> Vec<LocalDefId> {
        let mut ids = self
            .tcx
            .iter_local_def_id()
            .filter(|id| {
                let did = id.to_def_id();
                let kind = self.tcx.def_kind(did);
                supports_local_optimized_mir_body(self.tcx, *id, kind)
                    && self.tcx.is_mir_available(did)
            })
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| self.tcx.def_path_str(id.to_def_id()));
        ids
    }

    fn local_binding_ids(&self) -> Result<Vec<LocalDefId>, String> {
        let census = unsafe_site::census_snapshot()?;
        let owners_with_sites = census
            .owners
            .values()
            .filter(|owner| !owner.sites.is_empty())
            .map(|owner| owner.owner.clone())
            .collect::<BTreeSet<_>>();
        let mut ids = self
            .tcx
            .iter_local_def_id()
            .filter(|id| {
                let did = id.to_def_id();
                owners_with_sites.contains(&unsafe_site::compiler_function_key(self.tcx, did))
                    && self.tcx.is_mir_available(did)
            })
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| self.tcx.def_path_str(id.to_def_id()));
        Ok(ids)
    }

    fn collect_preopt_transmutes(
        &self,
        ids: &[LocalDefId],
    ) -> Result<BTreeMap<FunctionKey, BTreeSet<PreoptTransmute>>, String> {
        let generation = unsafe_site::census_generation()?;
        let mut result = BTreeMap::new();
        for id in ids {
            if self.tcx.def_kind(*id) == DefKind::Closure {
                continue;
            }
            let function = unsafe_site::compiler_function_key(self.tcx, id.to_def_id());
            let slot = self.tcx.mir_drops_elaborated_and_const_checked(*id);
            if slot.is_stolen() {
                unsafe_site::census_store().record_failure(
                    generation,
                    CoverageFailure {
                        kind: CoverageFailureKind::CollectorFailed,
                        owner: function,
                        site_id: None,
                        candidate_count: 0,
                    },
                )?;
                continue;
            }
            let signature_output = self
                .tcx
                .fn_sig(id.to_def_id())
                .skip_binder()
                .output()
                .skip_binder();
            let returns_static_reference =
                matches!(signature_output.kind(), ty::Ref(region, ..) if region.is_static());
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
                    let point = self
                        .point(&function, location, statement.source_info.span)
                        .map_err(|error| {
                            format!(
                                "{error}; while normalizing pre-optimization {:?} bb{} statement {statement_index}",
                                function,
                                bb.index()
                            )
                        })?;
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

    fn bind_supplemental_owners(&mut self, ids: &[LocalDefId]) -> Result<(), String> {
        let census = unsafe_site::census_snapshot()?;
        for id in ids {
            let function = unsafe_site::compiler_function_key(self.tcx, id.to_def_id());
            let Some(owner) = census.owners.get(&function) else {
                continue;
            };
            if owner
                .sites
                .iter()
                .all(|site| self.binding_records.contains_key(&site.site_id()))
            {
                continue;
            }
            let did = id.to_def_id();
            let body = if uses_ctfe_mir(self.tcx.def_kind(did)) {
                self.tcx.mir_for_ctfe(did)
            } else {
                self.tcx.optimized_mir(did)
            };
            let solver_events = enumerate_mir_events(self.tcx, *id, body, MirStage::Solver)?;
            let built_events = self
                .built_unsafe_events
                .get(&function)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let records = bind_owner(&owner.sites, built_events, &solver_events)?;
            self.record_bindings(&owner.sites, records)?;
            self.solver_unsafe_events.insert(function, solver_events);
        }
        Ok(())
    }

    fn collect_bodies(&mut self, ids: &[LocalDefId]) -> Result<(), String> {
        let census = unsafe_site::census_snapshot()?;
        for id in ids {
            let did = id.to_def_id();
            if uses_ctfe_mir(self.tcx.def_kind(did)) {
                continue;
            }
            let body = self.tcx.optimized_mir(did);
            let function = unsafe_site::compiler_function_key(self.tcx, did);
            let solver_events = enumerate_mir_events(self.tcx, *id, body, MirStage::Solver)?;
            let sites = census
                .owners
                .get(&function)
                .map(|owner| owner.sites.as_slice())
                .unwrap_or_default();
            let built_events = self
                .built_unsafe_events
                .get(&function)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let records = bind_owner(sites, built_events, &solver_events)?;
            let bound_sites = self.record_bindings(sites, records)?;
            self.solver_unsafe_events
                .insert(function.clone(), solver_events);
            let program = self.normalize_body(*id, body, &function, &bound_sites)?;
            let expected_bound_sites = bound_sites.values().map(Vec::len).sum::<usize>();
            if program.attached_unsafe_site_count() != expected_bound_sites {
                return Err(format!(
                    "unsafe-site attachment drift for {}: expected {expected_bound_sites}, attached {}",
                    function.0,
                    program.attached_unsafe_site_count()
                ));
            }
            let facts = self.extract_program(
                &program,
                &LocalCallOutputs::new(),
                &BTreeSet::new(),
                ExtractionPhase::Outputs,
            );
            self.programs.insert(function, program);
            self.bodies.insert(facts.function.clone(), facts);
        }
        Ok(())
    }

    fn record_bindings(
        &mut self,
        sites: &[unsafe_site::UnsafeSiteSeed],
        records: Vec<SiteBindingRecord>,
    ) -> Result<BTreeMap<(u32, u32), Vec<BoundUnsafeSite>>, String> {
        let sites_by_id = sites
            .iter()
            .map(|site| (site.site_id(), site))
            .collect::<BTreeMap<_, _>>();
        let mut bound = BTreeMap::<(u32, u32), Vec<BoundUnsafeSite>>::new();
        let generation = unsafe_site::census_generation()?;
        for record in records {
            let site = sites_by_id
                .get(&record.site_id)
                .ok_or_else(|| "binding record refers to an unknown unsafe site".to_string())?;
            if !insert_binding_record(&mut self.binding_records, record.clone())? {
                continue;
            }
            if let Some(bound_site) = BoundUnsafeSite::from_record(site, &record) {
                let key = (bound_site.solver.block, bound_site.solver.statement);
                bound.entry(key).or_default().push(bound_site);
                continue;
            }
            if record.status == BindingStatus::OptimizedAwayExact {
                continue;
            }
            let (kind, candidate_count) = match record.status {
                BindingStatus::Unbound => (
                    CoverageFailureKind::UnboundUnsafeSite,
                    record.built_candidate_count,
                ),
                BindingStatus::Ambiguous => (
                    CoverageFailureKind::AmbiguousUnsafeSite,
                    record
                        .built_candidate_count
                        .max(record.solver_candidate_count),
                ),
                BindingStatus::Elided => (
                    CoverageFailureKind::ElidedUnsafeSite,
                    record.solver_candidate_count,
                ),
                BindingStatus::OptimizedAwayExact => {
                    unreachable!("optimized-away exact record handled above")
                }
                BindingStatus::BoundExact => unreachable!("bound record handled above"),
            };
            unsafe_site::census_store().record_failure(
                generation,
                CoverageFailure {
                    kind,
                    owner: site.owner.clone(),
                    site_id: Some(record.site_id.clone()),
                    candidate_count,
                },
            )?;
        }
        for sites in bound.values_mut() {
            sites.sort_by_key(|site| site.seed.site_id());
        }
        Ok(bound)
    }

    fn account_for_unprocessed_census_owners(&mut self) -> Result<(), String> {
        let census = unsafe_site::census_snapshot()?;
        let generation = census.generation;
        for owner in census.owners.values() {
            for site in &owner.sites {
                let site_id = site.site_id();
                if self.binding_records.contains_key(&site_id) {
                    continue;
                }
                self.binding_records.insert(
                    site_id.clone(),
                    SiteBindingRecord {
                        site_id: site_id.clone(),
                        status: BindingStatus::Unbound,
                        built_candidate_count: 0,
                        solver_candidate_count: 0,
                        built: None,
                        solver: None,
                        solver_event: None,
                    },
                );
                unsafe_site::census_store().record_failure(
                    generation,
                    CoverageFailure {
                        kind: CoverageFailureKind::UnboundUnsafeSite,
                        owner: owner.owner.clone(),
                        site_id: Some(site_id),
                        candidate_count: 0,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn extract_program(
        &self,
        program: &BodyProgram,
        local_outputs: &LocalCallOutputs,
        definite_outs: &BTreeSet<u32>,
        phase: ExtractionPhase,
    ) -> BodyFacts {
        let mut facts = BodyFacts {
            def_id: program.def_id,
            function: program.function.clone(),
            root_span: program.root_span.clone(),
            arg_count: program.arg_count,
            has_self: program.has_self,
            argument_utf8_capability: program.argument_utf8_capability.clone(),
            summary_seed: FunctionSummary::empty(program.function.clone()),
            values: BTreeMap::new(),
            calls: Vec::new(),
            callback_calls: BTreeSet::new(),
            compares: Vec::new(),
            lengths: BTreeSet::new(),
            predicates: Vec::new(),
            branches: Vec::new(),
            writes: Vec::new(),
            validation_snapshots: BTreeMap::new(),
            local_bindings: BTreeMap::new(),
            local_relational_obligations: BTreeMap::new(),
            concrete_layout_obligations: BTreeMap::new(),
            source_local_generic_capabilities: BTreeSet::new(),
            endpoint_graph: EndpointFlowGraph::default(),
            influence_seeds: InfluenceSeedIndex::default(),
            structural_sites: BTreeMap::new(),
            contract_descriptors: BTreeMap::new(),
            contract_sites_by_point: BTreeMap::new(),
            bound_sites: BTreeMap::new(),
            initialized_site_operands: BTreeSet::new(),
            possibly_uninitialized_site_operands: BTreeSet::new(),
            initialization_witnesses: BTreeMap::new(),
            proven_contract_sites: BTreeSet::new(),
        };
        for index in 1..=program.arg_count {
            let place = PlaceKey::new(format!("_{index}"));
            facts.values.insert(
                place.clone(),
                BodyValueFacts {
                    value_flow: BTreeSet::from([place.clone()]),
                },
            );
        }
        let operation_counts = program.operations.iter().map(Vec::len).collect();
        let mut entry = BlockState::empty();
        for index in 1..=program.arg_count {
            let place = PlaceKey::new(format!("_{index}"));
            let endpoint = EndpointNode::Formal {
                owner: program.function.clone(),
                index: index as u32,
            };
            let mut endpoint_sources = BTreeSet::from([endpoint.clone()]);
            let mut influence_seeds = InfluenceSeedIndex::default();
            if phase.tracks_endpoints() {
                if let Some(seed) = program.argument_influence.get(&(index as u32)) {
                    influence_seeds.insert(endpoint.clone(), seed.clone());
                    if seed.origin == InfluenceOrigin::FfiCallbackInput {
                        let ffi_endpoint = ffi_callback_input_endpoint(
                            &program.function,
                            index as u32,
                            &program.root_span,
                        );
                        endpoint_sources.insert(ffi_endpoint.clone());
                        influence_seeds.insert(ffi_endpoint, seed.clone());
                    }
                }
            }
            entry.set_value(
                place.clone(),
                ValueFacts {
                    value: AbstractValue::new([AbstractOrigin::Formal(index as u32)]),
                    dependencies: BTreeSet::from([place.clone()]),
                    value_flow: BTreeSet::from([place.clone()]),
                    exact_roots: BTreeSet::from([place]),
                    may_be_non_length: true,
                    utf8_relation_unknown: true,
                    checked_end_unknown: true,
                    range_limit_unknown: true,
                    known_length_unknown: true,
                    behavior_control: BehaviorControlProvenance::MaybeExternal,
                    storage_provenance: StorageProvenance::EntryFormal {
                        owner: program.function.clone(),
                        index: index as u32,
                    },
                    endpoint_sources: phase
                        .tracks_endpoints()
                        .then(|| Arc::new(endpoint_sources))
                        .unwrap_or_default(),
                    influence_seeds,
                    ..ValueFacts::default()
                },
            );
        }
        for field in entry_field_reads(program).into_values() {
            let Some(index) = entry_formal_index(program.arg_count, &field.base) else {
                continue;
            };
            let base = entry
                .may_values()
                .get(&field.base)
                .expect("entry formal field base must be seeded");
            entry.set_value(
                field.full_place.clone(),
                entry_field_projection_value(base, &field, index),
            );
        }
        let solved = solve_cfg(
            &program.successors,
            &operation_counts,
            entry,
            |block, index, state| {
                apply_program_op_with_context(
                    &program.operations[block][index],
                    block,
                    index,
                    state,
                    program.is_closure,
                );
                if !phase.tracks_endpoints() {
                    clear_endpoint_facts(state);
                }
            },
            |block, target, state| {
                let sites = program.operations[block]
                    .last()
                    .map(|operation| operation.unsafe_sites.as_slice())
                    .unwrap_or_default();
                apply_program_edge_with_outputs_sites_and_programs(
                    &program.edges[block],
                    target,
                    state,
                    local_outputs,
                    sites,
                    Some(&self.programs),
                );
                if !phase.tracks_endpoints() {
                    clear_endpoint_facts(state);
                }
            },
        );
        replay_final_entries(
            &solved.entry,
            &operation_counts,
            |block, index, state| {
                apply_program_op_with_context(
                    &program.operations[block][index],
                    block,
                    index,
                    state,
                    program.is_closure,
                );
                if !phase.tracks_endpoints() {
                    clear_endpoint_facts(state);
                }
            },
            |block, index, state| {
                let operation = &program.operations[block][index];
                if matches!(operation.kind, ProgramOpKind::Call(_))
                    || !operation.unsafe_sites.is_empty()
                {
                    facts.validation_snapshots.insert(
                        (block, index),
                        ValidationSnapshot::capture(state, facts.arg_count),
                    );
                }
                if phase.tracks_endpoints() {
                    self.capture_endpoint_operation(program, block, index, state, &mut facts);
                } else {
                    let operation = &program.operations[block][index];
                    record_site_contracts(
                        operation,
                        |site| self.known_contract_model_for_site(operation, site),
                        &mut facts.structural_sites,
                        &mut facts.contract_descriptors,
                        &mut facts.contract_sites_by_point,
                        &mut facts.bound_sites,
                    );
                }
                self.materialize_operation(program, &solved, block, index, state, &mut facts);
            },
        );
        self.materialize_exits(program, &solved, &mut facts, phase);
        facts
            .summary_seed
            .relational_postconditions
            .extend(reachable_relational_postconditions(
                &program.edges,
                &solved,
                facts.arg_count,
            ));
        self.add_local_requirements(program, &mut facts);
        self.partition_validation_contracts(&mut facts);
        facts
            .proven_contract_sites
            .extend(proven_contract_sites(&facts));
        let sink_local_literal_sites = sink_local_literal_index_sites(&facts);
        facts.proven_contract_sites.extend(sink_local_literal_sites);
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
            .summary_seed
            .out_dependencies
            .extend(reachable_out_dependencies(
                &program.edges,
                (1..=program.arg_count).map(|index| index as u32),
                &solved,
                definite_outs,
            ));
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
            let mut predicates = requirement.predicates.iter().copied();
            let Some(predicate) = predicates.next() else {
                continue;
            };
            if predicates.next().is_some() {
                continue;
            }
            let policy = validation_contract_policy(predicate);
            if policy == LocalValidationPolicy::Unsupported {
                continue;
            }
            let point = requirement.sink.point.point_location();
            let Some(snapshot) = facts
                .validation_snapshots
                .get(&(point.block, point.statement_index))
            else {
                continue;
            };
            if local_requirement_is_proven(
                snapshot,
                None,
                facts
                    .local_relational_obligations
                    .get(&requirement.sink.point),
            ) {
                facts.summary_seed.requirements.remove(&requirement);
                continue;
            }
            let Some(binding) = facts.local_bindings.get(&requirement.sink.point) else {
                continue;
            };
            if binding.predicate != predicate {
                continue;
            }
            if snapshot.proves(binding) {
                facts.summary_seed.requirements.remove(&requirement);
                continue;
            }
            if policy == LocalValidationPolicy::LocalOnly {
                continue;
            }
            let subject = snapshot.entry_formal(&binding.subject);
            let collection = match &binding.collection {
                Some(place) => {
                    let Some(slot) = snapshot.entry_formal(place) else {
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

    fn materialize_operation(
        &self,
        program: &BodyProgram,
        solved: &ProgramDataflowResult,
        block: BasicBlock,
        index: usize,
        state: &ProgramState,
        facts: &mut BodyFacts,
    ) {
        let operation = &program.operations[block][index];
        for (place, value) in state.may_values() {
            facts
                .values
                .entry(place.clone())
                .or_default()
                .join_value_facts(value);
            facts.influence_seeds.join(&value.influence_seeds);
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
                        let exact_formal = |operand: &OperandModel| {
                            let value = eval_operand(state, operand);
                            if value.havoced || value.semantic_unknown {
                                return None;
                            }
                            let mut origins = value.value.origins.iter();
                            match (origins.next(), origins.next()) {
                                (Some(AbstractOrigin::Formal(index)), None) => Some(*index),
                                _ => None,
                            }
                        };
                        facts.compares.push(CompareFact {
                            result: destination.clone(),
                            op: *op,
                            left: left.operand.clone(),
                            right: right.operand.clone(),
                            left_formal: exact_formal(left),
                            right_formal: exact_formal(right),
                            point: operation.semantic_point().clone(),
                        });
                    }
                    ValueModel::PointerCast {
                        operand,
                        obligation,
                    } => {
                        let mut obligation = *obligation;
                        let source = eval_operand(state, operand);
                        obligation.externally_controlled_bits =
                            has_externally_controlled_validity_bits(&source);
                        facts
                            .concrete_layout_obligations
                            .insert(operation.semantic_point().clone(), obligation);
                    }
                    ValueModel::GenericPointerCast { operand, evidence } => {
                        let source = eval_operand(state, operand);
                        if establishes_source_local_generic_capability(&source) {
                            facts
                                .source_local_generic_capabilities
                                .insert(evidence.stable_identity());
                        }
                    }
                    _ => {}
                }
                if assignment.legacy_write {
                    record_write(
                        facts,
                        destination,
                        operation.semantic_point().clone(),
                        WriteKind::Assignment,
                    );
                }
                if let Some(source) = &assignment.raw_read_source {
                    if !has_contract_model_id(
                        facts,
                        operation.semantic_point(),
                        "raw_pointer_access_v1",
                    ) {
                        return;
                    }
                    let subject = eval_operand(state, source).value;
                    facts.summary_seed.requirements.insert(ContractRequirement {
                        seed_id: format!("P1.raw:{}", operation.semantic_point().span.token()),
                        collection: None,
                        subject,
                        source_hint: None,
                        internal_derivation: false,
                        source_span: operation.semantic_point().span.clone(),
                        first_failure: Operation {
                            kind: OperationKind::RawRead,
                            point: operation.semantic_point().clone(),
                        },
                        sink: Operation {
                            kind: OperationKind::RawRead,
                            point: operation.semantic_point().clone(),
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
                match &call.value {
                    RegistryValueModel::PointerCast {
                        operand,
                        obligation,
                    } => {
                        let mut obligation = *obligation;
                        let source = eval_operand(state, operand);
                        obligation.externally_controlled_bits =
                            has_externally_controlled_validity_bits(&source);
                        facts
                            .concrete_layout_obligations
                            .insert(operation.semantic_point().clone(), obligation);
                    }
                    RegistryValueModel::GenericPointerCast { base, evidence } => {
                        let source = eval_operand(state, base);
                        if establishes_source_local_generic_capability(&source) {
                            facts
                                .source_local_generic_capabilities
                                .insert(evidence.stable_identity());
                        }
                    }
                    _ => {}
                }
                let normal_successor = match &program.edges[block] {
                    EdgeTerm::CallNormal {
                        normal_target: Some(target),
                        ..
                    } if solved.edge.contains_key(&(block, *target)) => Some(*target),
                    _ => None,
                };
                let normal_edge_state =
                    normal_successor.and_then(|target| solved.edge.get(&(block, target)));
                let mut arg_values = descriptor
                    .args
                    .iter()
                    .map(|operand| eval_operand(state, operand))
                    .collect::<Vec<_>>();
                // Keep the endpoint provenance at this exact boundary.  The
                // summary-oriented copy below deliberately drops it.
                let actual_endpoint_sources = arg_values
                    .iter()
                    .map(|value| value.endpoint_sources.as_ref().clone())
                    .collect::<Vec<_>>();
                for value in &mut arg_values {
                    value.endpoint_sources = Arc::default();
                    value.endpoint_edges = Arc::default();
                }
                let args = descriptor
                    .args
                    .iter()
                    .map(|operand| operand.operand.clone())
                    .collect::<Vec<_>>();
                let p3_assume_init = match &call.value {
                    RegistryValueModel::AssumeInit {
                        restricted_target, ..
                    } => *restricted_target,
                    _ => false,
                };
                if normal_edge_state.is_some() && p3_assume_init {
                    if let Some(subject) = descriptor
                        .args
                        .first()
                        .zip(arg_values.first())
                        .and_then(|(_, value)| ffi_invalid_value_subject(value))
                    {
                        facts
                            .summary_seed
                            .requirements
                            .insert(ffi_invalid_value_requirement(
                                &facts.function,
                                operation.semantic_point(),
                                subject,
                            ));
                    }
                }
                facts.calls.push(CallSite {
                    callee: descriptor.callee.clone(),
                    raw_def: descriptor.raw_def,
                    disposition: descriptor.disposition,
                    args: args.clone(),
                    arg_values,
                    actual_endpoint_sources,
                    destination: descriptor.destination.clone(),
                    point: operation.semantic_point().clone(),
                    destination_is_bool: call.destination_is_bool,
                    destination_restricted: call.destination_restricted,
                    access_width: call.access_width,
                    index_range_kind: descriptor
                        .args
                        .get(1)
                        .and_then(|operand| operand.range_index_kind),
                    generic_substitution: descriptor.generic_substitution.clone(),
                    #[cfg(not(test))]
                    typed_witness_templates: descriptor.typed_witness_templates.clone(),
                });
                if let Some(callee) = &descriptor.callee {
                    let mapping = local_call_boundary_mapping(call, state);
                    facts.summary_seed.calls.insert(CallBoundary {
                        caller: facts.function.clone(),
                        callee: callee.clone(),
                        point: operation.semantic_point().clone(),
                        mapping,
                    });
                }
                if call.legacy_write && normal_edge_state.is_some() {
                    record_edge_write(
                        facts,
                        descriptor.destination.clone(),
                        operation.semantic_point().clone(),
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
                            operation.semantic_point().clone(),
                            WriteKind::OpaqueMutable,
                            normal_successor,
                        );
                    }
                    for target in call
                        .ffi_out_actuals
                        .iter()
                        .flat_map(|actual| state_alias_closure(state, &actual.place).into_iter())
                    {
                        record_edge_write(
                            facts,
                            target,
                            operation.semantic_point().clone(),
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
                        let predicate = match predicate {
                            PredicateModel::Utf8Result { .. } | PredicateModel::IsErr { .. } => {
                                None
                            }
                            PredicateModel::IsEmpty { slice } => Some(PredicateFact::IsEmpty {
                                result: descriptor.destination.clone(),
                                slice: slice.clone(),
                            }),
                            PredicateModel::IsNull { pointer } => Some(PredicateFact::IsNull {
                                result: descriptor.destination.clone(),
                                pointer: pointer.clone(),
                            }),
                        };
                        facts.predicates.extend(predicate);
                    }
                }
            }
            ProgramOpKind::Nop => {}
        }
        if index + 1 == program.operations[block].len() {
            if let EdgeTerm::Switch {
                condition,
                false_target,
                ..
            } = &program.edges[block]
            {
                let point = operation.semantic_point().clone();
                facts.branches.push(BranchFact {
                    result: condition.clone(),
                    success: *false_target,
                    point,
                });
            }
        }
    }

    fn materialize_exits(
        &self,
        program: &BodyProgram,
        solved: &ProgramDataflowResult,
        facts: &mut BodyFacts,
        phase: ExtractionPhase,
    ) {
        for (block, state) in solved.exit.iter_enumerated() {
            let Some(state) = state else {
                continue;
            };
            for (place, value) in state.may_values() {
                facts
                    .values
                    .entry(place.clone())
                    .or_default()
                    .join_value_facts(value);
                facts.influence_seeds.join(&value.influence_seeds);
            }
            if !matches!(program.edges[block], EdgeTerm::Return) {
                continue;
            }
            if phase.tracks_endpoints() {
                if let Some(value) = state.may_values().get(&PlaceKey::new("_0")) {
                    facts
                        .endpoint_graph
                        .edges
                        .extend(value.endpoint_edges.iter().cloned());
                    for edge in value.endpoint_edges.iter() {
                        facts.endpoint_graph.nodes.insert(edge.from.clone());
                        facts.endpoint_graph.nodes.insert(edge.to.clone());
                    }
                    let target = EndpointNode::Return {
                        owner: facts.function.clone(),
                    };
                    facts.endpoint_graph.insert_node(target.clone());
                    for source in value.endpoint_sources.iter() {
                        facts.endpoint_graph.insert_edge(EndpointEdge {
                            from: source.clone(),
                            to: target.clone(),
                            kind: EndpointEdgeKind::ReturnToDestination,
                            point: None,
                        });
                    }
                }
            }
            if !phase.tracks_endpoints() {
                continue;
            }
            for (place, value) in state.may_values() {
                let Some(index) = out_slot_index(place) else {
                    continue;
                };
                facts
                    .endpoint_graph
                    .edges
                    .extend(value.endpoint_edges.iter().cloned());
                for edge in value.endpoint_edges.iter() {
                    facts.endpoint_graph.nodes.insert(edge.from.clone());
                    facts.endpoint_graph.nodes.insert(edge.to.clone());
                }
                let target = EndpointNode::Out {
                    owner: facts.function.clone(),
                    index,
                };
                facts.endpoint_graph.insert_node(target.clone());
                for source in value.endpoint_sources.iter() {
                    facts.endpoint_graph.insert_edge(EndpointEdge {
                        from: source.clone(),
                        to: target.clone(),
                        kind: EndpointEdgeKind::OutToCaller,
                        point: None,
                    });
                }
            }
        }
    }

    fn capture_endpoint_operation(
        &self,
        program: &BodyProgram,
        block: BasicBlock,
        index: usize,
        state: &ProgramState,
        facts: &mut BodyFacts,
    ) {
        let operation = &program.operations[block][index];
        record_site_contracts(
            operation,
            |site| self.known_contract_model_for_site(operation, site),
            &mut facts.structural_sites,
            &mut facts.contract_descriptors,
            &mut facts.contract_sites_by_point,
            &mut facts.bound_sites,
        );
        facts.proven_contract_sites.extend(
            operation
                .unsafe_sites
                .iter()
                .filter(|site| {
                    facts
                        .contract_descriptors
                        .get(&site.seed.site_id())
                        .and_then(|descriptor| descriptor.model_id.as_deref())
                        == Some(KnownContractModel::ConditionTrue.as_str())
                        && state_proves_condition_true_site(state, site)
                })
                .map(|site| site.seed.site_id()),
        );
        if let ProgramOpKind::Assign(assignment) = &operation.kind {
            if assignment.field_origin.is_none()
                && value_model_derivation_kind(&assignment.value).is_some()
            {
                let value = eval_value_model(
                    state,
                    &assignment.value,
                    StaticWriteToken::new(block.index() as u32, index as u32),
                );
                for edge in value.endpoint_edges.iter().cloned() {
                    facts.endpoint_graph.insert_edge(edge);
                }
            }
            if let ValueModel::Aggregate {
                operands,
                closure: Some(closure),
                ..
            } = &assignment.value
            {
                for (index, operand) in operands.iter().enumerate() {
                    let target = EndpointNode::Capture {
                        owner: closure.clone(),
                        index: index as u32,
                    };
                    facts.endpoint_graph.insert_node(target.clone());
                    let value = eval_operand(state, operand);
                    for edge in value.endpoint_edges.iter().cloned() {
                        facts.endpoint_graph.insert_edge(edge);
                    }
                    for source in value.endpoint_sources.iter().cloned() {
                        facts.endpoint_graph.insert_edge(EndpointEdge {
                            from: source,
                            to: target.clone(),
                            kind: EndpointEdgeKind::ActualToFormal,
                            point: operation.point.clone(),
                        });
                    }
                }
            }
            if let Some(field) = &assignment.field_origin {
                let destination = public_field_node(state, field);
                if matches!(
                    &destination,
                    EndpointNode::PublicField {
                        persistent: true,
                        ..
                    }
                ) {
                    let value = eval_value_model(
                        state,
                        &assignment.value,
                        StaticWriteToken::new(block.index() as u32, index as u32),
                    );
                    append_value_endpoint_edges(
                        &mut facts.endpoint_graph,
                        value.endpoint_edges.as_ref(),
                    );
                    let exposure = EndpointNode::Exposure {
                        owner: program.function.clone(),
                        token: format!("public-field-store:{}", field.def_path),
                        point: operation.semantic_point().clone(),
                    };
                    facts.endpoint_graph.insert_node(exposure.clone());
                    for source in value.endpoint_sources.iter().cloned() {
                        facts.endpoint_graph.insert_edge(EndpointEdge {
                            from: source,
                            to: exposure.clone(),
                            kind: EndpointEdgeKind::Exposure,
                            point: operation.point.clone(),
                        });
                    }
                }
            }
        }
        if let ProgramOpKind::Call(call) = &operation.kind {
            if call.descriptor.disposition != BoundaryDisposition::ResolvedLocal
                && registry_value_derivation_kind(&call.value).is_some()
            {
                let value = eval_registry_value(state, &call.value, operation.semantic_point());
                for edge in value.endpoint_edges.iter().cloned() {
                    facts.endpoint_graph.insert_edge(edge);
                }
            }
            if call.descriptor.disposition == BoundaryDisposition::OpaqueIndirect {
                if let RegistryValueModel::IndirectBehavior { source: callee, .. } = &call.value {
                    let callable = eval_operand(state, callee);
                    if let Some(target) = observed_internal_behavior_target(&callable, |function| {
                        self.programs
                            .get(function)
                            .map(|program| program.is_closure)
                    }) {
                        let actual_values = call
                            .descriptor
                            .args
                            .iter()
                            .map(|operand| eval_operand(state, operand))
                            .collect::<Vec<_>>();
                        let (edges, call_edge) = observed_internal_behavior_edges(
                            &program.function,
                            operation.semantic_point(),
                            &target,
                            actual_values.iter(),
                        );
                        for edge in edges {
                            facts.endpoint_graph.insert_edge(edge);
                        }
                        facts.callback_calls.insert(call_edge);
                    }
                }
            }
            if let Some(callee) = &call.descriptor.callee {
                // Do not add actual -> shared Formal edges here.  The endpoint
                // relation solver instantiates callee Formal relations at this
                // exact callsite after all bodies have been materialized.
                // Connecting these nodes directly lets call A's actual escape
                // through call B's shared callee Return.
                for operand in &call.descriptor.args {
                    let value = eval_operand(state, operand);
                    for edge in value.endpoint_edges.iter().cloned() {
                        facts.endpoint_graph.insert_edge(edge);
                    }
                }
                let _ = callee;
            }
        }
        capture_bound_endpoint_sites(
            &program.function,
            self.tcx.def_kind(facts.def_id) == DefKind::Closure,
            block,
            index,
            operation,
            state,
            &mut facts.endpoint_graph,
            &mut facts.initialized_site_operands,
            &mut facts.possibly_uninitialized_site_operands,
            &mut facts.initialization_witnesses,
        );
    }

    fn normalize_body(
        &self,
        local_id: LocalDefId,
        body: &Body<'tcx>,
        function: &FunctionKey,
        bound_sites: &BTreeMap<(u32, u32), Vec<BoundUnsafeSite>>,
    ) -> Result<BodyProgram, String> {
        let caller = local_id.to_def_id();
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
                let (point, kind) = match &statement.kind {
                    StatementKind::Assign(box (destination, rvalue)) => {
                        let point = self
                            .point(
                                function,
                                Location {
                                    block,
                                    statement_index,
                                },
                                statement.source_info.span,
                            )
                            .map_err(|error| {
                                format!(
                                    "{error}; while normalizing {:?} bb{} statement {statement_index} ({:?})",
                                    function, block.index(), statement.kind
                                )
                            })?;
                        let destination_key = place_key(*destination);
                        let legacy_write = !destination.projection.is_empty()
                            || definition_counts
                                .get(&destination_key)
                                .copied()
                                .unwrap_or(0)
                                > 1;
                        (
                            Some(point.clone()),
                            ProgramOpKind::Assign(self.normalize_assignment(
                                caller,
                                body,
                                *destination,
                                rvalue,
                                &point,
                                legacy_write,
                            )),
                        )
                    }
                    _ => (None, ProgramOpKind::Nop),
                };
                block_ops.push(ProgramOp {
                    point,
                    kind,
                    unsafe_sites: bound_sites
                        .get(&(block.index() as u32, statement_index as u32))
                        .cloned()
                        .unwrap_or_default(),
                });
            }
            let statement_index = data.statements.len();
            let terminator = data.terminator();
            let semantic_terminator_point = || {
                self.point(
                    function,
                    Location {
                        block,
                        statement_index,
                    },
                    terminator.source_info.span,
                )
                .map_err(|error| {
                    format!(
                        "{error}; while normalizing {:?} bb{} terminator ({:?})",
                        function,
                        block.index(),
                        terminator.kind
                    )
                })
            };
            let (point, kind) = match &terminator.kind {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } => {
                    let point = semantic_terminator_point()?;
                    let destination_key = place_key(*destination);
                    let legacy_write = !destination.projection.is_empty()
                        || definition_counts
                            .get(&destination_key)
                            .copied()
                            .unwrap_or(0)
                            > 1;
                    (
                        Some(point.clone()),
                        ProgramOpKind::Call(self.normalize_call(
                            caller,
                            body,
                            func,
                            args,
                            *destination,
                            &point,
                            legacy_write,
                        )),
                    )
                }
                TerminatorKind::SwitchInt { discr, .. } if discr.place().is_some() => {
                    (Some(semantic_terminator_point()?), ProgramOpKind::Nop)
                }
                TerminatorKind::Assert { cond, .. } if cond.place().is_some() => {
                    (Some(semantic_terminator_point()?), ProgramOpKind::Nop)
                }
                _ => (None, ProgramOpKind::Nop),
            };
            block_ops.push(ProgramOp {
                point: point.clone(),
                kind: kind.clone(),
                unsafe_sites: bound_sites
                    .get(&(block.index() as u32, statement_index as u32))
                    .cloned()
                    .unwrap_or_default(),
            });
            operations.push(block_ops);
            successors.push(terminator.successors().collect());
            edges.push(match &terminator.kind {
                TerminatorKind::Call { target, .. } => match kind {
                    ProgramOpKind::Call(call) => EdgeTerm::CallNormal {
                        normal_target: *target,
                        point: point.expect("normalized calls have stable source points"),
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
        Ok(BodyProgram {
            def_id: local_id,
            function: function.clone(),
            root_span: self.stable_span(self.tcx.def_span(caller))?,
            is_closure: self.tcx.def_kind(local_id) == DefKind::Closure,
            arg_count: body.arg_count,
            has_self: associated_item_has_self(self.tcx, caller),
            argument_influence: body
                .args_iter()
                .filter_map(|argument| {
                    let index = argument.index() as u32;
                    let (origin, controller) = if self.is_exported_ffi_callback(caller) {
                        (
                            InfluenceOrigin::FfiCallbackInput,
                            InfluenceController::FfiHost,
                        )
                    } else if self.is_public_safe_root(local_id) {
                        (
                            InfluenceOrigin::PublicFormal,
                            InfluenceController::PublicFormal {
                                owner: function.clone(),
                                index,
                            },
                        )
                    } else {
                        return None;
                    };
                    Some((
                        index,
                        InfluenceToken::from_type_class(
                            origin,
                            controller,
                            influence_type_class_for_ty(body.local_decls[argument].ty),
                        ),
                    ))
                })
                .collect(),
            argument_utf8_capability: body
                .args_iter()
                .map(|argument| {
                    (
                        argument.index() as u32,
                        utf8_formal_capability(body.local_decls[argument].ty),
                    )
                })
                .collect(),
            out_formals: body
                .args_iter()
                .filter(|arg| is_mutable_call_actual(body.local_decls[*arg].ty))
                .map(|arg| {
                    (
                        PlaceKey::new(format!("_{}", arg.index())),
                        arg.index() as u32,
                    )
                })
                .collect(),
            operations,
            edges,
            successors,
        })
    }

    fn pointer_transform_kind(
        &self,
        owner: DefId,
        source_ty: ty::Ty<'tcx>,
        target_ty: ty::Ty<'tcx>,
    ) -> PointerTransformKind {
        let source_pointee = match source_ty.kind() {
            ty::RawPtr(pointee, _) => Some(*pointee),
            _ => None,
        };
        let target_pointee = match target_ty.kind() {
            ty::RawPtr(pointee, _) => Some(*pointee),
            _ => None,
        };
        let same_pointee = source_pointee
            .zip(target_pointee)
            .is_some_and(|(source, target)| source == target);
        let sequence_element_view =
            source_pointee
                .zip(target_pointee)
                .is_some_and(|(source, target)| match source.kind() {
                    ty::Array(element, _) | ty::Slice(element) => *element == target,
                    _ => false,
                });
        let transparent_wrapper_view =
            source_pointee
                .zip(target_pointee)
                .is_some_and(|(source, target)| {
                    self.transparent_pointer_wrapper_inner(source) == Some(target)
                        || self.transparent_pointer_wrapper_inner(target) == Some(source)
                });
        let concrete_implication = source_pointee
            .zip(target_pointee)
            .filter(|_| !source_ty.has_non_region_param() && !target_ty.has_non_region_param())
            .map(|(source, target)| {
                (
                    pointee_layout_implies_target(self.tcx, owner, source, target),
                    pointee_validity_implies_target(source, target),
                )
            });
        classify_pointer_transform(PointerTransformEvidence {
            same_pointee,
            sequence_element_view,
            transparent_wrapper_view,
            unit_erasure: target_pointee.is_some_and(ty::Ty::is_unit),
            compiler_layout_implies_target: concrete_implication.is_some_and(|(layout, _)| layout),
            source_validity_implies_target: concrete_implication
                .is_some_and(|(_, validity)| validity),
            has_generic: source_ty.has_non_region_param() || target_ty.has_non_region_param(),
        })
    }

    fn concrete_layout_obligation(
        &self,
        owner: DefId,
        source_ty: ty::Ty<'tcx>,
        target_ty: ty::Ty<'tcx>,
    ) -> ConcreteLayoutObligation {
        let (Some(source), Some(target)) = (
            raw_pointer_pointee(source_ty),
            raw_pointer_pointee(target_ty),
        ) else {
            return ConcreteLayoutObligation::default();
        };
        if source_ty.has_non_region_param() || target_ty.has_non_region_param() {
            return ConcreteLayoutObligation::default();
        }
        let (Some(source_size), Some(target_size), Some(source_align), Some(target_align)) = (
            type_layout_size(self.tcx, owner, source),
            type_layout_size(self.tcx, owner, target),
            type_layout_align(self.tcx, owner, source),
            type_layout_align(self.tcx, owner, target),
        ) else {
            return ConcreteLayoutObligation::default();
        };
        let source_validity = pointee_validity_class(source);
        let target_validity = pointee_validity_class(target);
        let source_all_bits = matches!(
            source_validity,
            TypeValidityClass::AllBitsValidScalar
                | TypeValidityClass::AllBitsValidBytes
                | TypeValidityClass::AllBitsValidStructural
                | TypeValidityClass::AllBitsValidRawPointer
        );
        let target_restricted = matches!(
            target_validity,
            TypeValidityClass::RestrictedScalar
                | TypeValidityClass::RestrictedStructural
                | TypeValidityClass::RestrictedEnum
                | TypeValidityClass::ReferenceOrFunctionPointer
        );
        let validity_violation = source_all_bits && target_restricted;
        let validity_complete = source == target
            || pointee_validity_implies_target(source, target)
            || validity_violation;
        ConcreteLayoutObligation {
            size_violation: target_size > source_size,
            alignment_violation: target_align > source_align,
            validity_violation,
            complete: validity_complete,
            externally_controlled_bits: false,
        }
    }

    fn generic_components(
        &self,
        owner: DefId,
        types: impl IntoIterator<Item = ty::Ty<'tcx>>,
    ) -> (BTreeSet<GenericParamId>, BTreeSet<String>) {
        let generics = self.tcx.generics_of(owner);
        let mut controllers = BTreeSet::new();
        let mut associated_items = BTreeSet::new();
        for value in types {
            for argument in value.walk() {
                match generic_arg_kind(argument) {
                    ty::GenericArgKind::Type(value) => match value.kind() {
                        ty::Param(parameter) => {
                            let definition = generics.type_param(*parameter, self.tcx);
                            controllers.insert(GenericParamId {
                                def_path: self.tcx.def_path_str(definition.def_id),
                                kind: GenericParamKind::Type,
                            });
                        }
                        ty::Alias(_, alias) => {
                            associated_items.insert(self.tcx.def_path_str(alias_ty_def_id(*alias)));
                        }
                        _ => {}
                    },
                    ty::GenericArgKind::Const(value) => {
                        if let ty::ConstKind::Param(parameter) = value.kind() {
                            let definition = generics.const_param(parameter, self.tcx);
                            controllers.insert(GenericParamId {
                                def_path: self.tcx.def_path_str(definition.def_id),
                                kind: GenericParamKind::Const,
                            });
                        }
                    }
                    ty::GenericArgKind::Lifetime(_) => {}
                }
            }
        }
        (controllers, associated_items)
    }

    fn pointer_capability_evidence(
        &self,
        owner: DefId,
        source: ty::Ty<'tcx>,
        target: ty::Ty<'tcx>,
        kind: GenericCapabilityKind,
    ) -> Option<GenericCapabilityEvidence> {
        let source = raw_pointer_pointee(source)?;
        let target = raw_pointer_pointee(target)?;
        self.pointer_capability_evidence_for_pointees(owner, source, target, kind)
    }

    fn transparent_pointer_view_evidence(
        &self,
        owner: DefId,
        receiver: ty::Ty<'tcx>,
        target: ty::Ty<'tcx>,
    ) -> Option<GenericCapabilityEvidence> {
        // This path is called only after the callee has been compiler-typed as
        // MaybeUninit::{as_ptr,as_mut_ptr} or UnsafeCell::{get,raw_get}.
        // Their receiver is a reference to the transparent wrapper, not a raw
        // pointer, so extract the represented value without teaching the
        // general raw-pointer cast model that arbitrary references are views.
        let source = self.transparent_pointer_receiver_inner(receiver)?;
        let target = raw_pointer_pointee(target)?;
        self.pointer_capability_evidence_for_pointees(
            owner,
            source,
            target,
            GenericCapabilityKind::RepresentationPreservingPointerView,
        )
    }

    fn pointer_capability_evidence_for_pointees(
        &self,
        owner: DefId,
        source: ty::Ty<'tcx>,
        target: ty::Ty<'tcx>,
        kind: GenericCapabilityKind,
    ) -> Option<GenericCapabilityEvidence> {
        let mut source_term = self.capability_term(owner, source);
        let mut target_term = self.capability_term(owner, target);
        if self.normalized_types_equal(owner, source, target) {
            source_term.equality_class = Some(0);
            target_term.equality_class = Some(0);
        }
        let controllers = source_term
            .controllers
            .iter()
            .chain(&target_term.controllers)
            .cloned()
            .collect::<BTreeSet<_>>();
        if controllers.is_empty() {
            return None;
        }
        let extent = match target_term.type_facts.size {
            Some(0) => AccessExtent::Zero,
            Some(width) => AccessExtent::Positive(width),
            None if !target_term.controllers.is_empty() => AccessExtent::Symbolic {
                controllers: target_term.controllers.clone(),
            },
            None => AccessExtent::Unknown,
        };
        let (requirement, witness) = match kind {
            GenericCapabilityKind::RepresentationPreservingPointerView => (
                CapabilityRequirement::RepresentationPreservingView,
                CapabilityWitness::ImpossibleByCompilerFacts {
                    reason: super::endpoint_flow::CapabilityImpossibleReason::RepresentationPreservingView,
                },
            ),
            GenericCapabilityKind::PointerLayoutReinterpretation => (
                CapabilityRequirement::PointerLayout { extent },
                self.pointer_layout_witness(owner, source, target),
            ),
            GenericCapabilityKind::AssociatedSequenceShape { .. } => {
                return None;
            }
        };
        Some(GenericCapabilityEvidence {
            kind,
            terms: BTreeMap::from([
                (CapabilityTermRole::SourcePointee, source_term),
                (CapabilityTermRole::TargetPointee, target_term),
            ]),
            requirement,
            witness,
        })
    }

    fn associated_sequence_evidence(
        &self,
        owner: DefId,
        sequence: ty::Ty<'tcx>,
        result: ty::Ty<'tcx>,
    ) -> GenericCapabilityEvidence {
        let (_, associated_items) = self.generic_components(owner, [sequence, result]);
        let element = sequence_element_ty(result).unwrap_or(result);
        GenericCapabilityEvidence {
            kind: GenericCapabilityKind::AssociatedSequenceShape { associated_items },
            terms: BTreeMap::from([
                (
                    CapabilityTermRole::Sequence,
                    self.capability_term(owner, sequence),
                ),
                (
                    CapabilityTermRole::Element,
                    self.capability_term(owner, element),
                ),
            ]),
            requirement: CapabilityRequirement::AssociatedSequenceNonEmpty,
            witness: self.associated_sequence_witness(owner, sequence, element),
        }
    }

    fn capability_term(&self, owner: DefId, value: ty::Ty<'tcx>) -> CapabilityTerm {
        let normalized = normalize_type(self.tcx, owner, value).unwrap_or(value);
        let (controllers, residual_aliases) = self.generic_components(owner, [normalized]);
        let form = match normalized.kind() {
            ty::Param(_) => CapabilityTermForm::DirectParam,
            ty::Alias(..) => CapabilityTermForm::Alias,
            _ if controllers.is_empty() && residual_aliases.is_empty() => {
                CapabilityTermForm::Concrete
            }
            _ if !residual_aliases.is_empty() => CapabilityTermForm::Alias,
            _ if !controllers.is_empty() => CapabilityTermForm::Composite,
            _ => CapabilityTermForm::Unknown,
        };
        CapabilityTerm {
            form,
            controllers,
            residual_aliases,
            equality_class: None,
            type_facts: type_capability_facts(self.tcx, owner, normalized),
        }
    }

    fn normalized_types_equal(
        &self,
        owner: DefId,
        left: ty::Ty<'tcx>,
        right: ty::Ty<'tcx>,
    ) -> bool {
        normalize_type(self.tcx, owner, left)
            .zip(normalize_type(self.tcx, owner, right))
            .is_some_and(|(left, right)| left == right)
    }

    fn associated_sequence_witness(
        &self,
        owner: DefId,
        sequence: ty::Ty<'tcx>,
        element: ty::Ty<'tcx>,
    ) -> CapabilityWitness {
        let Some(alias) = first_type_alias(sequence) else {
            return CapabilityWitness::Unestablished {
                reason: CapabilityUnestablishedReason::MissingRoleTerm,
            };
        };
        let Some(item) = self.tcx.opt_associated_item(alias_ty_def_id(alias)) else {
            return CapabilityWitness::Unestablished {
                reason: CapabilityUnestablishedReason::MissingRoleTerm,
            };
        };
        let Some(trait_id) = item.trait_container(self.tcx) else {
            return CapabilityWitness::Unestablished {
                reason: CapabilityUnestablishedReason::ComplexBounds,
            };
        };
        if self.tcx.trait_def(trait_id).safety != Safety::Safe {
            return CapabilityWitness::Unestablished {
                reason: CapabilityUnestablishedReason::UnsafeTrait,
            };
        }
        if !self.is_downstream_nameable(trait_id)
            || self.has_unnameable_sealing_supertrait(trait_id)
        {
            return CapabilityWitness::Unestablished {
                reason: CapabilityUnestablishedReason::SealedOrPrivateTrait,
            };
        }
        if !self
            .tcx
            .generics_of(alias_ty_def_id(alias))
            .own_params
            .is_empty()
        {
            return CapabilityWitness::Unestablished {
                reason: CapabilityUnestablishedReason::GenericAssociatedType,
            };
        }
        let witness = ty::Ty::new_array(self.tcx, element, 0);
        match associated_type_witness_satisfies_bounds(self.tcx, owner, alias, witness) {
            WitnessBoundResult::Proven => CapabilityWitness::Proven {
                assignment: BTreeMap::from([(
                    CapabilityTermRole::Sequence,
                    FiniteTypeWitness::EmptyArray,
                )]),
                violations: BTreeSet::from([CapabilityViolation::InBounds]),
            },
            WitnessBoundResult::Rejected | WitnessBoundResult::Unsupported => {
                CapabilityWitness::Unestablished {
                    reason: CapabilityUnestablishedReason::ComplexBounds,
                }
            }
        }
    }

    fn pointer_layout_witness(
        &self,
        owner: DefId,
        source: ty::Ty<'tcx>,
        target: ty::Ty<'tcx>,
    ) -> CapabilityWitness {
        if self.normalized_types_equal(owner, source, target) {
            return CapabilityWitness::ImpossibleByCompilerFacts {
                reason: super::endpoint_flow::CapabilityImpossibleReason::NormalizedTermsEqual,
            };
        }
        let aliases = first_type_alias(source).zip(first_type_alias(target));
        let direct_parameters =
            matches!(source.kind(), ty::Param(_)) && matches!(target.kind(), ty::Param(_));
        if aliases.is_none() && !direct_parameters {
            return CapabilityWitness::Unestablished {
                reason: CapabilityUnestablishedReason::NoFiniteWitness,
            };
        }
        if let Some((source_alias, target_alias)) = aliases {
            for alias in [source_alias, target_alias] {
                let Some(item) = self.tcx.opt_associated_item(alias_ty_def_id(alias)) else {
                    return CapabilityWitness::Unestablished {
                        reason: CapabilityUnestablishedReason::MissingRoleTerm,
                    };
                };
                let Some(trait_id) = item.trait_container(self.tcx) else {
                    return CapabilityWitness::Unestablished {
                        reason: CapabilityUnestablishedReason::ComplexBounds,
                    };
                };
                if self.tcx.trait_def(trait_id).safety != Safety::Safe {
                    return CapabilityWitness::Unestablished {
                        reason: CapabilityUnestablishedReason::UnsafeTrait,
                    };
                }
                if !self.is_downstream_nameable(trait_id)
                    || self.has_unnameable_sealing_supertrait(trait_id)
                {
                    return CapabilityWitness::Unestablished {
                        reason: CapabilityUnestablishedReason::SealedOrPrivateTrait,
                    };
                }
                if !self
                    .tcx
                    .generics_of(alias_ty_def_id(alias))
                    .own_params
                    .is_empty()
                {
                    return CapabilityWitness::Unestablished {
                        reason: CapabilityUnestablishedReason::GenericAssociatedType,
                    };
                }
            }
        }

        let pairs = [
            (
                self.tcx.types.u8,
                self.tcx.types.u64,
                FiniteTypeWitness::U8,
                FiniteTypeWitness::U64,
            ),
            (
                ty::Ty::new_array(self.tcx, self.tcx.types.u8, 8),
                self.tcx.types.u64,
                FiniteTypeWitness::U8Array8,
                FiniteTypeWitness::U64,
            ),
            (
                self.tcx.types.unit,
                ty::Ty::new_array(self.tcx, self.tcx.types.u64, 0),
                FiniteTypeWitness::Unit,
                FiniteTypeWitness::U64Array0,
            ),
        ];
        for (source_witness, target_witness, source_name, target_name) in pairs {
            let bounds = aliases.map_or_else(
                || {
                    direct_type_parameter_witnesses_satisfy_bounds(
                        self.tcx,
                        owner,
                        source,
                        source_witness,
                        target,
                        target_witness,
                    )
                },
                |(source_alias, target_alias)| {
                    if associated_type_witness_satisfies_bounds(
                        self.tcx,
                        owner,
                        source_alias,
                        source_witness,
                    ) != WitnessBoundResult::Proven
                        || associated_type_witness_satisfies_bounds(
                            self.tcx,
                            owner,
                            target_alias,
                            target_witness,
                        ) != WitnessBoundResult::Proven
                    {
                        WitnessBoundResult::Rejected
                    } else {
                        WitnessBoundResult::Proven
                    }
                },
            );
            if bounds != WitnessBoundResult::Proven {
                continue;
            }
            let source_facts = type_capability_facts(self.tcx, owner, source_witness);
            let target_facts = type_capability_facts(self.tcx, owner, target_witness);
            let mut violations = BTreeSet::new();
            if source_facts
                .size
                .zip(target_facts.size)
                .is_some_and(|(source, target)| source < target)
            {
                violations.insert(CapabilityViolation::Size);
            }
            if source_facts
                .align
                .zip(target_facts.align)
                .is_some_and(|(source, target)| source < target)
            {
                violations.insert(CapabilityViolation::Alignment);
            }
            if !violations.is_empty() {
                return CapabilityWitness::Proven {
                    assignment: BTreeMap::from([
                        (CapabilityTermRole::SourcePointee, source_name),
                        (CapabilityTermRole::TargetPointee, target_name),
                    ]),
                    violations,
                };
            }
        }
        CapabilityWitness::Unestablished {
            reason: CapabilityUnestablishedReason::NoFiniteWitness,
        }
    }

    fn generic_surface(&self, owner: DefId) -> BTreeSet<GenericParamId> {
        let generics = self.tcx.generics_of(owner);
        let mut surface = (0..generics.count())
            .filter_map(|index| {
                let parameter = generics.param_at(index, self.tcx);
                let kind = match parameter.kind {
                    ty::GenericParamDefKind::Type { .. } => GenericParamKind::Type,
                    ty::GenericParamDefKind::Const { .. } => GenericParamKind::Const,
                    ty::GenericParamDefKind::Lifetime => return None,
                };
                Some(GenericParamId {
                    def_path: self.tcx.def_path_str(parameter.def_id),
                    kind,
                })
            })
            .collect::<BTreeSet<_>>();
        let signature = self.tcx.fn_sig(owner).instantiate_identity().skip_binder();
        surface.extend(
            self.generic_components(owner, signature.inputs_and_output.iter())
                .0,
        );
        surface
    }

    fn transparent_pointer_wrapper_inner(&self, value: ty::Ty<'tcx>) -> Option<ty::Ty<'tcx>> {
        let ty::Adt(definition, arguments) = value.kind() else {
            return None;
        };
        (self
            .tcx
            .is_lang_item(definition.did(), LangItem::UnsafeCell)
            || self
                .tcx
                .is_lang_item(definition.did(), LangItem::MaybeUninit))
        .then(|| arguments.type_at(0))
    }

    fn transparent_pointer_receiver_inner(&self, value: ty::Ty<'tcx>) -> Option<ty::Ty<'tcx>> {
        let wrapper = match value.kind() {
            ty::Ref(_, inner, _) => *inner,
            _ => value,
        };
        self.transparent_pointer_wrapper_inner(wrapper)
    }

    fn normalize_assignment(
        &self,
        caller: DefId,
        body: &Body<'tcx>,
        destination: Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        point: &ProgramPoint,
        legacy_write: bool,
    ) -> AssignmentModel {
        let destination_key = place_key(destination);
        let value = if let Some(operand) = rvalue_use_operand(rvalue) {
            ValueModel::Operand(self.normalize_operand(body, operand, point))
        } else if let Some(place) = rvalue_len_place(rvalue) {
            ValueModel::Length(self.normalize_place(body, place, point))
        } else {
            match rvalue {
                Rvalue::CopyForDeref(place) => {
                    ValueModel::Operand(self.normalize_place(body, *place, point))
                }
                Rvalue::Cast(
                    CastKind::PointerCoercion(ty::adjustment::PointerCoercion::Unsize, _),
                    operand,
                    _,
                ) => ValueModel::Unsize {
                    operand: self.normalize_operand(body, operand, point),
                    known_length: match operand.ty(&body.local_decls, self.tcx).kind() {
                        ty::Ref(_, inner, _) => match inner.kind() {
                            ty::Array(_, length) => target_usize_from_const(self.tcx, *length),
                            _ => None,
                        },
                        _ => None,
                    },
                },
                Rvalue::Cast(CastKind::PointerCoercion(coercion, _), operand, _)
                    if is_behavior_pointer_coercion(*coercion) =>
                {
                    let seed = operand
                        .const_fn_def()
                        .map(|(did, _)| {
                            behavior_pointer_controller(
                                unsafe_site::compiler_function_key(self.tcx, did),
                                self.tcx.is_foreign_item(did),
                            )
                        })
                        .or_else(|| match operand.ty(&body.local_decls, self.tcx).kind() {
                            ty::Closure(def_id, _) => Some(BehaviorController::InternalClosure {
                                function: unsafe_site::compiler_function_key(self.tcx, *def_id),
                            }),
                            _ => None,
                        });
                    ValueModel::BehaviorPointer {
                        operand: self.normalize_operand(body, operand, point),
                        seed,
                    }
                }
                Rvalue::Cast(_, operand, target)
                    if matches!(
                        operand.ty(&body.local_decls, self.tcx).kind(),
                        ty::RawPtr(..)
                    ) && matches!(target.kind(), ty::RawPtr(..)) =>
                {
                    let source_ty = operand.ty(&body.local_decls, self.tcx);
                    let operand = self.normalize_operand(body, operand, point);
                    match self.pointer_transform_kind(caller, source_ty, *target) {
                        PointerTransformKind::IdentityView => ValueModel::IdentityPointerView {
                            operand,
                            generic_view: self.pointer_capability_evidence(
                                caller,
                                source_ty,
                                *target,
                                GenericCapabilityKind::RepresentationPreservingPointerView,
                            ),
                        },
                        PointerTransformKind::AddressView => ValueModel::AddressPointerView {
                            operand,
                            generic_view: self.pointer_capability_evidence(
                                caller,
                                source_ty,
                                *target,
                                GenericCapabilityKind::RepresentationPreservingPointerView,
                            ),
                        },
                        PointerTransformKind::GenericLayoutRisk => ValueModel::GenericPointerCast {
                            operand,
                            evidence: self
                                .pointer_capability_evidence(
                                    caller,
                                    source_ty,
                                    *target,
                                    GenericCapabilityKind::PointerLayoutReinterpretation,
                                )
                                .expect("a generic layout transform has a type/const controller"),
                        },
                        PointerTransformKind::Reinterpretation => ValueModel::PointerCast {
                            operand,
                            obligation: self.concrete_layout_obligation(caller, source_ty, *target),
                        },
                    }
                }
                Rvalue::Cast(_, _, _) => ValueModel::Unknown,
                Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
                    let mut operand = self.normalize_place(body, *place, point);
                    if matches!(place.projection.as_ref(), [ProjectionElem::Deref]) {
                        operand.pure_deref_base = Some(place_key(Place::from(place.local)));
                        operand.exact_projection = None;
                        operand.projection_unknown = false;
                    }
                    let definitely_local = place.local.as_usize() > body.arg_count
                        && !place
                            .projection
                            .iter()
                            .any(|element| matches!(element, ProjectionElem::Deref));
                    if definitely_local {
                        ValueModel::LocalRefOrRaw(operand)
                    } else {
                        ValueModel::RefOrRaw(operand)
                    }
                }
                Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => operand
                    .place()
                    .map(|place| ValueModel::Length(self.normalize_place(body, place, point)))
                    .unwrap_or(ValueModel::Unknown),
                Rvalue::Discriminant(place) => {
                    ValueModel::Discriminant(self.normalize_place(body, *place, point))
                }
                Rvalue::BinaryOp(op, operands) => {
                    let left = self.normalize_operand(body, &operands.0, point);
                    let right = self.normalize_operand(body, &operands.1, point);
                    if *op == BinOp::AddWithOverflow {
                        ValueModel::CheckedBinary {
                            op: BinOp::Add,
                            left,
                            right,
                        }
                    } else {
                        ValueModel::Compare {
                            op: *op,
                            left,
                            right,
                        }
                    }
                }
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
                    variant: match &**kind {
                        AggregateKind::Adt(did, variant, ..)
                            if self.tcx.adt_def(*did).is_enum() =>
                        {
                            Some(variant.as_u32() as u128)
                        }
                        _ => None,
                    },
                    closure: match &**kind {
                        AggregateKind::Closure(def_id, ..) => {
                            Some(unsafe_site::compiler_function_key(self.tcx, *def_id))
                        }
                        _ => None,
                    },
                },
                _ => ValueModel::Unknown,
            }
        };
        let field_origin = self.field_info(body, destination, point);
        let raw_read_source = rvalue_use_operand(rvalue)
            .and_then(Operand::place)
            .filter(|place| {
                place
                    .projection
                    .iter()
                    .any(|element| matches!(element, ProjectionElem::Deref))
                    && matches!(body.local_decls[place.local].ty.kind(), ty::RawPtr(..))
            })
            .map(|place| self.normalize_place(body, Place::from(place.local), point));
        AssignmentModel {
            destination: destination_key,
            destination_projection: Some(self.normalize_place_projection(destination)),
            value,
            field_origin,
            raw_read_source,
            legacy_write,
            out_formal: out_formal_index(body, destination),
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
                projection_path: None,
                range_index_kind: None,
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
        let field_origin = self.field_info(body, place, point);
        let exact = place_key(place);
        let pure_deref_base = None;
        let projection = self.normalize_place_projection(place);
        let projection_unknown = projection.projection_unknown;
        let exact_projection =
            (!place.projection.is_empty() && !projection_unknown).then_some(exact.clone());
        let projection_path =
            (!place.projection.is_empty() && !projection_unknown).then_some(projection.path);
        OperandModel {
            operand: CallOperand::Place(exact.clone()),
            field_origin,
            pure_deref_base,
            exact_projection,
            projection_path,
            range_index_kind: self.range_index_kind(place.ty(&body.local_decls, self.tcx).ty),
            projection_unknown,
        }
    }

    fn normalize_place_projection(&self, place: Place<'tcx>) -> PlaceProjectionModel {
        let mut projection_unknown = false;
        let path = place
            .projection
            .iter()
            .filter_map(|element| match element {
                ProjectionElem::Deref => Some(ProjectionKey::Deref),
                ProjectionElem::Downcast(_, variant) => {
                    Some(ProjectionKey::Downcast(variant.as_u32()))
                }
                ProjectionElem::Field(field, _) => Some(ProjectionKey::Field(field.as_u32())),
                _ => {
                    projection_unknown = true;
                    None
                }
            })
            .collect();
        PlaceProjectionModel {
            base: place_key(Place::from(place.local)),
            path,
            projection_unknown,
        }
    }

    fn range_index_kind(&self, ty: ty::Ty<'tcx>) -> Option<RangeIndexKind> {
        let ty::Adt(definition, arguments) = ty.kind() else {
            return None;
        };
        let lang_items = self.tcx.lang_items();
        let kind = match definition.did() {
            def_id if Some(def_id) == lang_items.range_full_struct() => {
                return Some(RangeIndexKind::Full)
            }
            def_id if Some(def_id) == lang_items.range_from_struct() => RangeIndexKind::From,
            def_id if Some(def_id) == lang_items.range_to_struct() => RangeIndexKind::To,
            def_id if Some(def_id) == lang_items.range_struct() => RangeIndexKind::Between,
            _ => return None,
        };
        let element = arguments.iter().find_map(|argument| argument.as_type())?;
        matches!(element.kind(), ty::Uint(ty::UintTy::Usize)).then_some(kind)
    }

    fn normalize_call(
        &self,
        caller: DefId,
        body: &Body<'tcx>,
        func: &Operand<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        point: &ProgramPoint,
        legacy_write: bool,
    ) -> CallModel {
        let raw = func.const_fn_def();
        let raw_def = raw.map(|(did, _)| did);
        let resolved_local =
            raw.and_then(|(did, generic_args)| self.resolve_local(caller, did, generic_args));
        let callee = resolved_local.as_ref().map(|resolved| resolved.key.clone());
        let generic_substitution = resolved_local
            .as_ref()
            .map(|resolved| self.generic_call_substitution(caller, resolved.target, resolved.args))
            .unwrap_or_default();
        #[cfg(not(test))]
        let typed_witness_templates = resolved_local
            .as_ref()
            .map(|resolved| {
                self.typed_witness_transport_templates(
                    caller,
                    resolved.target,
                    resolved.args,
                    &generic_substitution,
                )
            })
            .unwrap_or_default();
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
        let selected_open_behavior = raw.and_then(|(did, generic_args)| {
            self.open_behavior_controller(caller, did, generic_args, &callee, body, args)
        });
        let modeled_registry = raw_def.is_some_and(|did| {
            self.is_registry_semantic_helper(did)
                || self.is_slice_iterator_position(did, body, args)
                || self.is_usize_range_into_iterator(did, body, args)
                || self.is_usize_range_iterator_next(did, body, args)
                || self.is_associated_slice_as_ref(did, body, args, destination)
        });
        let disposition = select_boundary(BoundaryInputs {
            resolved_local: callee.is_some(),
            exact_ffi_out,
            selected_open_behavior: selected_open_behavior.is_some(),
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
                .filter_map(|(argument, operand)| {
                    operand.operand.place().cloned().map(|place| FfiOutActual {
                        place,
                        type_class: ffi_out_value_type_class(
                            argument.node.ty(&body.local_decls, self.tcx),
                        ),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        let destination_ty = destination.ty(&body.local_decls, self.tcx).ty;
        let mut value = RegistryValueModel::Empty;
        let mut predicate = None;
        if let Some((did, generic_args)) = raw {
            if self.is_maybe_uninit_constructor(did, "uninit") {
                value = RegistryValueModel::MaybeUninit(false);
            } else if self.is_maybe_uninit_constructor(did, "new") {
                value = RegistryValueModel::MaybeUninit(true);
            } else if self.tcx.is_diagnostic_item(sym::assume_init, did) && !operands.is_empty() {
                value = RegistryValueModel::AssumeInit {
                    source: operands[0].clone(),
                    restricted_target: has_restricted_validity(destination_ty),
                };
            } else if (self.is_slice_len(did) || self.is_vec_len(did)) && !operands.is_empty() {
                value = RegistryValueModel::Length(operands[0].clone());
            } else if self.is_slice_iterator_constructor(did) && !operands.is_empty() {
                value = RegistryValueModel::SliceIterator(operands[0].clone());
            } else if self.is_slice_iterator_position(did, body, args) && !operands.is_empty() {
                value = RegistryValueModel::SlicePosition(operands[0].clone());
            } else if self.is_usize_range_into_iterator(did, body, args) && !operands.is_empty() {
                value = RegistryValueModel::RangeIntoIterator(operands[0].clone());
            } else if self.is_usize_range_iterator_next(did, body, args) && !operands.is_empty() {
                value = RegistryValueModel::RangeIteratorNext(operands[0].clone());
            } else if self.is_associated_slice_as_ref(did, body, args, destination) {
                value = RegistryValueModel::GenericCapability {
                    inputs: operands.clone(),
                    evidence: self.associated_sequence_evidence(
                        caller,
                        args[0].node.ty(&body.local_decls, self.tcx),
                        destination_ty,
                    ),
                };
            } else if disposition == BoundaryDisposition::SelectedOpenBehavior {
                value = RegistryValueModel::SelectedOpenBehavior {
                    token: format!("trait-return:{}", self.tcx.def_path_str(did)),
                    controller: selected_open_behavior
                        .clone()
                        .expect("selected open behavior must have a controller"),
                    inputs: operands.clone(),
                    type_class: influence_type_class_for_ty(destination_ty),
                };
            } else if self.is_exact_foreign_c(did) {
                value = RegistryValueModel::ExternalBehavior {
                    kind: ExternalSourceKind::FfiReturn,
                    controller: BehaviorController::FfiHost,
                    token: format!("ffi-return:{}", self.tcx.def_path_str(did)),
                    type_class: influence_type_class_for_ty(destination_ty),
                };
            }

            if self.is_saturating_sub(did) && operands.len() >= 2 {
                value = RegistryValueModel::SaturatingSub(operands.clone());
            } else if self.is_checked_add(did) && operands.len() >= 2 {
                value = RegistryValueModel::CheckedAdd {
                    offset: operands[0].clone(),
                    width: operands[1].clone(),
                };
            } else if self.is_try_trait_branch(did) && !operands.is_empty() {
                value = RegistryValueModel::CheckedTryBranch(operands[0].clone());
            } else if is_mem_size_of(self.tcx, did) {
                if let Some(size) = type_layout_size(self.tcx, caller, generic_args.type_at(0)) {
                    value = RegistryValueModel::Constant(size);
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
            } else if self.is_maybe_uninit_pointer_view(did) && !operands.is_empty() {
                value = RegistryValueModel::TypedInitializationPointer {
                    receiver: operands[0].clone(),
                    generic_view: self.transparent_pointer_view_evidence(
                        caller,
                        args[0].node.ty(&body.local_decls, self.tcx),
                        destination_ty,
                    ),
                };
            } else if self.is_transparent_pointer_view_call(did) && !operands.is_empty() {
                let source_ty = args[0].node.ty(&body.local_decls, self.tcx);
                value = RegistryValueModel::AddressPointerView {
                    operand: operands[0].clone(),
                    generic_view: self.transparent_pointer_view_evidence(
                        caller,
                        source_ty,
                        destination_ty,
                    ),
                };
            } else if self.is_pointer_cast(did) && !operands.is_empty() {
                let source_ty = args[0].node.ty(&body.local_decls, self.tcx);
                value = match self.pointer_transform_kind(caller, source_ty, destination_ty) {
                    PointerTransformKind::IdentityView => RegistryValueModel::IdentityPointerView {
                        operand: operands[0].clone(),
                        generic_view: self.pointer_capability_evidence(
                            caller,
                            source_ty,
                            destination_ty,
                            GenericCapabilityKind::RepresentationPreservingPointerView,
                        ),
                    },
                    PointerTransformKind::AddressView => RegistryValueModel::AddressPointerView {
                        operand: operands[0].clone(),
                        generic_view: self.pointer_capability_evidence(
                            caller,
                            source_ty,
                            destination_ty,
                            GenericCapabilityKind::RepresentationPreservingPointerView,
                        ),
                    },
                    PointerTransformKind::GenericLayoutRisk => {
                        RegistryValueModel::GenericPointerCast {
                            base: operands[0].clone(),
                            evidence: self
                                .pointer_capability_evidence(
                                    caller,
                                    source_ty,
                                    destination_ty,
                                    GenericCapabilityKind::PointerLayoutReinterpretation,
                                )
                                .expect("a generic layout transform has a type/const controller"),
                        }
                    }
                    PointerTransformKind::Reinterpretation => RegistryValueModel::PointerCast {
                        operand: operands[0].clone(),
                        obligation: self.concrete_layout_obligation(
                            caller,
                            source_ty,
                            destination_ty,
                        ),
                    },
                };
            }

            predicate = if (self.tcx.is_diagnostic_item(sym::str_from_utf8, did)
                || self.is_internal_utf8_validation_call(
                    caller,
                    did,
                    callee.is_some(),
                    body,
                    args,
                    destination_ty,
                ))
                && !operands.is_empty()
            {
                Some(PredicateModel::Utf8Result {
                    bytes: operands[0].clone(),
                })
            } else if self.is_result_is_err(did) && !operands.is_empty() {
                Some(PredicateModel::IsErr {
                    checked: operands[0].clone(),
                })
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
        if callee.is_some()
            && matches!(value, RegistryValueModel::Empty)
            && is_private_aggregate_type(self.tcx, destination_ty)
        {
            value = RegistryValueModel::PrivateAggregateTransform;
        }
        if disposition == BoundaryDisposition::OpaqueIndirect
            && matches!(value, RegistryValueModel::Empty)
        {
            value = RegistryValueModel::IndirectBehavior {
                source: self.normalize_operand(body, func, point),
                type_class: influence_type_class_for_ty(destination_ty),
            };
        }

        let access_width = type_layout_size(self.tcx, caller, destination_ty);
        let typed_initialization_effect = raw_def.and_then(|did| {
            if self.is_maybe_uninit_method(did, "write") {
                return operands.first().cloned().map(|destination| {
                    TypedInitializationEffect::MaybeUninitWrite { destination }
                });
            }
            if self.is_free_ptr_write(did) && operands.len() >= 2 {
                return Some(TypedInitializationEffect::PointerWrite {
                    destination: operands[0].clone(),
                    source: operands[1].clone(),
                });
            }
            if let Some(nonoverlapping) = self.free_ptr_copy_kind(did) {
                return (operands.len() >= 3).then(|| TypedInitializationEffect::PointerCopy {
                    source: operands[0].clone(),
                    destination: operands[1].clone(),
                    count: operands[2].operand.clone(),
                    nonoverlapping,
                });
            }
            let destinations = self
                .tcx
                .intrinsic(did)
                .is_some()
                .then(|| {
                    operands
                        .iter()
                        .filter(|operand| {
                            operand
                                .operand
                                .place()
                                .is_some_and(|actual| mutable_actuals.contains(actual))
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .filter(|destinations| !destinations.is_empty());
            destinations.map(
                |destinations| TypedInitializationEffect::OpaqueIntrinsicMutation { destinations },
            )
        });
        CallModel {
            descriptor: CallDescriptor {
                callee,
                raw_def,
                disposition,
                args: operands,
                destination: destination_key,
                generic_substitution,
                #[cfg(not(test))]
                typed_witness_templates,
            },
            value,
            predicate,
            mutable_actuals,
            ffi_out_actuals,
            typed_initialization_effect,
            destination_is_bool: destination_ty.is_bool(),
            destination_restricted: has_restricted_validity(destination_ty),
            access_width,
            legacy_write,
            destination_out_formal: out_formal_index(body, destination),
        }
    }

    fn add_local_requirements(&self, _program: &BodyProgram, facts: &mut BodyFacts) {
        if let Some(items) = self.preopt_transmutes.get(&facts.function) {
            for item in items {
                let origin = AbstractOrigin::InternalLocal {
                    function: facts.function.clone(),
                    place: PlaceKey::new("_0"),
                };
                let requirement = ContractRequirement {
                    seed_id: format!("P5.lifetime:{}", item.first_failure.point.span.token()),
                    collection: None,
                    subject: AbstractValue::new([origin.clone()]),
                    source_hint: Some(SourceKind::InternalUnsafeOrigin),
                    internal_derivation: false,
                    source_span: item.first_failure.point.span.clone(),
                    first_failure: item.first_failure.clone(),
                    sink: item.sink.clone(),
                    predicates: BTreeSet::from([Predicate::ReferentOutlivesReference]),
                    access_width: None,
                    rule: RuleId::P5LifetimeTransmute,
                    return_exposure: true,
                    sink_function: facts.function.clone(),
                };
                facts.summary_seed.requirements.insert(requirement);
                facts.summary_seed.return_value.origins.insert(origin);
            }
        }

        let calls = std::mem::take(&mut facts.calls);
        for call in &calls {
            if (has_contract_model(facts, &call.point, KnownContractModel::GetUnchecked)
                || has_contract_model(facts, &call.point, KnownContractModel::VecGetUnchecked))
                && call.args.len() >= 2
            {
                let collection = call.arg_values[0].value.clone();
                let index_subject = call.arg_values[1].value.clone();
                let constant_zero = matches!(call.args[1], CallOperand::Constant(0));
                let p4_subject = generic_nonempty_subject(&collection, constant_zero);
                let source_hint = if p4_subject.is_some() {
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
                let subject = p4_subject.unwrap_or(index_subject);
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
                    Some(SourceKind::OpenBehaviorOutput) => RuleId::P3OpenTraitIndex,
                    _ => RuleId::P1GetUnchecked,
                };
                let seed_id = format!("{}:{}", rule.as_str(), call.point.span.token());
                let exact_collection = local_sink_operand_place(&call.args[0], &call.arg_values[0]);
                let location = call.point.point_location();
                let relational_obligation = facts
                    .validation_snapshots
                    .get(&(location.block, location.statement_index))
                    .and_then(|snapshot| proven_range_index_obligation(snapshot, &call))
                    .or_else(|| {
                        exact_collection
                            .as_ref()
                            .and_then(|collection| range_index_obligation(&call, collection))
                    });
                if let Some(obligation) = relational_obligation {
                    facts
                        .local_relational_obligations
                        .insert(call.point.clone(), obligation);
                }
                let binding = match (&call.args[0], &call.args[1], constant_zero) {
                    (CallOperand::Place(_), CallOperand::Place(_), false)
                        if call.index_range_kind.is_none() =>
                    {
                        local_sink_operand_place(&call.args[1], &call.arg_values[1]).and_then(
                            |subject| {
                                exact_collection
                                    .clone()
                                    .map(|collection| ValidationBinding {
                                        predicate: Predicate::InBounds,
                                        access_width: None,
                                        proof_definition: None,
                                        subject,
                                        collection: Some(collection),
                                    })
                            },
                        )
                    }
                    (CallOperand::Place(_), _, true) => {
                        local_sink_operand_place(&call.args[0], &call.arg_values[0]).map(
                            |collection| ValidationBinding {
                                predicate: Predicate::NonEmpty,
                                access_width: None,
                                proof_definition: None,
                                subject: collection,
                                collection: None,
                            },
                        )
                    }
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
            } else if has_contract_model(facts, &call.point, KnownContractModel::AssumeInit)
                && !call.args.is_empty()
            {
                let initialization = &call.arg_values[0];
                // Unknown initialization is not evidence of a valid `T`.
                // Only a definite, site-current whole-value initialization
                // may suppress the assume_init requirement.
                if assume_init_requires_requirement(initialization) {
                    let legacy_bool_return = call.destination_is_bool
                        && self.place_reaches_return(facts, &call.destination);
                    let requirement =
                        assume_init_requirement(&facts.function, &call, legacy_bool_return);
                    if legacy_bool_return {
                        facts
                            .summary_seed
                            .return_value
                            .origins
                            .extend(requirement.subject.origins.iter().cloned());
                    }
                    facts.summary_seed.requirements.insert(requirement);
                }
            } else if (has_contract_model(
                facts,
                &call.point,
                KnownContractModel::FromUtf8Unchecked,
            ) || has_contract_model(
                facts,
                &call.point,
                KnownContractModel::StringFromUtf8Unchecked,
            )) && !call.args.is_empty()
            {
                if let Some(subject) = call.arg_values.first().and_then(unique_semantic_value) {
                    facts.local_bindings.insert(
                        call.point.clone(),
                        ValidationBinding {
                            predicate: Predicate::ValidUtf8,
                            access_width: None,
                            proof_definition: None,
                            subject,
                            collection: None,
                        },
                    );
                }
                facts
                    .summary_seed
                    .requirements
                    .insert(unchecked_utf8_requirement(&facts.function, &call));
                if self.place_reaches_return(facts, &call.destination) {
                    facts
                        .summary_seed
                        .return_value
                        .origins
                        .insert(AbstractOrigin::InternalLocal {
                            function: facts.function.clone(),
                            place: call.destination.clone(),
                        });
                }
            } else if has_contract_model(
                facts,
                &call.point,
                KnownContractModel::NonNullNewUnchecked,
            ) && !call.args.is_empty()
            {
                let subject = call.arg_values[0].value.clone();
                if subject
                    .origins
                    .iter()
                    .any(|origin| matches!(origin, AbstractOrigin::FfiOutput { .. }))
                {
                    let seed_id = format!("P3.ffi:{}", call.point.span.token());
                    if matches!(&call.args[0], CallOperand::Place(_)) {
                        if let Some(subject) = unique_semantic_value(&call.arg_values[0]) {
                            facts.local_bindings.insert(
                                call.point.clone(),
                                ValidationBinding {
                                    predicate: Predicate::NonNull,
                                    access_width: None,
                                    proof_definition: None,
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
                        rule: RuleId::P3FfiOutParam,
                        return_exposure: false,
                        sink_function: facts.function.clone(),
                    });
                }
            } else if has_contract_model(facts, &call.point, KnownContractModel::ReadUnaligned)
                && !call.args.is_empty()
            {
                let pointer = call.arg_values[0].value.clone();
                if let (Some(width), Some(pointer_facts)) =
                    (call.access_width, call.arg_values.first())
                {
                    if let Some(binding) = range_binding_for_pointer(pointer_facts, width) {
                        facts.local_bindings.insert(call.point.clone(), binding);
                    }
                }
                if internal_only_origins(&pointer, facts.has_self) {
                    facts.summary_seed.requirements.insert(ContractRequirement {
                        seed_id: format!("P6.offset:{}", call.point.span.token()),
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
                        rule: RuleId::P6Offset,
                        return_exposure: false,
                        sink_function: facts.function.clone(),
                    });
                }
            }
        }
        facts.calls = calls;
    }

    #[rustversion::since(1.96)]
    fn associated_impl(&self, did: DefId) -> Option<DefId> {
        self.tcx.opt_associated_item(did)?.impl_container(self.tcx)
    }

    #[rustversion::before(1.96)]
    fn associated_impl(&self, did: DefId) -> Option<DefId> {
        self.tcx.opt_associated_item(did)?.impl_container(self.tcx)
    }

    #[rustversion::since(1.96)]
    fn associated_trait(&self, did: DefId) -> Option<DefId> {
        let item = self.tcx.opt_associated_item(did)?;
        match item.container {
            ty::AssocContainer::Trait => Some(self.tcx.parent(item.def_id)),
            ty::AssocContainer::TraitImpl(Ok(trait_item)) => Some(self.tcx.parent(trait_item)),
            ty::AssocContainer::TraitImpl(Err(_)) | ty::AssocContainer::InherentImpl => None,
        }
    }

    #[rustversion::before(1.96)]
    fn associated_trait(&self, did: DefId) -> Option<DefId> {
        self.tcx.trait_of_item(did)
    }

    #[rustversion::since(1.96)]
    fn associated_trait_item(&self, did: DefId) -> Option<DefId> {
        let item = self.tcx.opt_associated_item(did)?;
        match item.container {
            ty::AssocContainer::Trait => Some(item.def_id),
            ty::AssocContainer::TraitImpl(Ok(trait_item)) => Some(trait_item),
            ty::AssocContainer::TraitImpl(Err(_)) | ty::AssocContainer::InherentImpl => None,
        }
    }

    #[rustversion::before(1.96)]
    fn associated_trait_item(&self, did: DefId) -> Option<DefId> {
        let item = self.tcx.opt_associated_item(did)?;
        item.trait_item_def_id.or_else(|| {
            matches!(item.container, ty::AssocItemContainer::TraitContainer).then_some(item.def_id)
        })
    }

    fn is_open_trait_method(&self, did: DefId) -> bool {
        self.associated_trait(did).is_some()
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
            .or_else(|| {
                let lengths = self.symbolic_array_lengths(body, operand);
                (!lengths.is_empty()).then_some(CallOperand::ArrayLengths(lengths))
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
        eval_operand_target_usize(self.tcx, body.source.def_id(), operand.constant()?.const_)
    }

    fn symbolic_array_lengths(
        &self,
        body: &Body<'tcx>,
        operand: &Operand<'tcx>,
    ) -> BTreeSet<PlaceKey> {
        if !matches!(
            operand.ty(&body.local_decls, self.tcx).kind(),
            ty::Uint(ty::UintTy::Usize)
        ) {
            return BTreeSet::new();
        }
        let Some(rustc_middle::mir::Const::Ty(_, constant)) =
            operand.constant().map(|constant| constant.const_)
        else {
            return BTreeSet::new();
        };
        body.local_decls
            .iter_enumerated()
            .filter_map(|(local, declaration)| {
                (self.array_length_const(declaration.ty) == Some(constant))
                    .then(|| PlaceKey::new(format!("{local:?}")))
            })
            .collect()
    }

    fn array_length_const(&self, ty: ty::Ty<'tcx>) -> Option<ty::Const<'tcx>> {
        match ty.kind() {
            ty::Array(_, length) => Some(*length),
            ty::Ref(_, inner, _) | ty::RawPtr(inner, _) => self.array_length_const(*inner),
            ty::Adt(definition, generic_args)
                if self
                    .tcx
                    .is_lang_item(definition.did(), LangItem::MaybeUninit) =>
            {
                self.array_length_const(generic_args.type_at(0))
            }
            _ => None,
        }
    }

    fn field_info(
        &self,
        body: &Body<'tcx>,
        place: Place<'tcx>,
        point: &ProgramPoint,
    ) -> Option<FieldAccessModel> {
        let projection = self.normalize_place_projection(place);
        let type_class = influence_type_class_for_ty(place.ty(&body.local_decls, self.tcx).ty);
        let mut place_ty = PlaceTy::from_ty(body.local_decls[place.local].ty);
        let mut result = None;
        let mut path_public = true;
        let mut crosses_enum = false;
        for element in place.projection.iter() {
            if let ProjectionElem::Field(index, _) = element {
                if let ty::Adt(adt, _) = place_ty.ty.kind() {
                    let variant = place_ty
                        .variant_index
                        .map(|variant| adt.variant(variant))
                        .unwrap_or_else(|| adt.non_enum_variant());
                    let field = &variant.fields[index];
                    // `Reachable` also contains metadata-reachable ADTs behind
                    // private modules. Their public fields are not a
                    // caller-nameable persistent state surface, so P2 needs a
                    // directly exported or re-exported ADT path.
                    let parent_public = !adt.did().is_local()
                        || self
                            .tcx
                            .effective_visibilities(())
                            .is_public_at_level(adt.did().expect_local(), Level::Reexported);
                    path_public &= field.vis.is_public() && parent_public;
                    crosses_enum |= adt.is_enum();
                    let adt_kind = if crosses_enum {
                        PublicFieldAdtKind::Enum
                    } else if adt.is_union() {
                        PublicFieldAdtKind::Union
                    } else {
                        PublicFieldAdtKind::Struct
                    };
                    result = Some(FieldAccessModel {
                        access_owner: point.function.clone(),
                        full_place: place_key(place),
                        base: projection.base.clone(),
                        projection_path: (!projection.projection_unknown)
                            .then(|| projection.path.clone()),
                        projection_unknown: projection.projection_unknown,
                        def_path: self.tcx.def_path_str(field.did),
                        adt_kind,
                        path_public,
                        type_class,
                        span: point.span.clone(),
                    });
                }
            }
            place_ty = place_ty.projection_ty(self.tcx, element);
        }
        result
    }

    fn resolve_local(
        &self,
        caller: DefId,
        raw: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Option<ResolvedLocalCall<'tcx>> {
        if let Some(instance) = resolve_instance(self.tcx, caller, raw, args) {
            if let InstanceKind::Item(target) = instance.def {
                if target.is_local() && self.tcx.is_mir_available(target) {
                    return Some(ResolvedLocalCall {
                        key: unsafe_site::compiler_function_key(self.tcx, target),
                        target,
                        // `instance.args`, not the raw trait-call args, is the
                        // rebased substitution for a resolved impl method.
                        args: instance.args,
                    });
                }
            }
        }
        None
    }

    fn generic_call_substitution(
        &self,
        caller: DefId,
        target: DefId,
        arguments: ty::GenericArgsRef<'tcx>,
    ) -> GenericCallSubstitution {
        let generics = self.tcx.generics_of(target);
        let mut status = GenericSubstitutionStatus::Complete;
        let mut bindings = BTreeMap::new();
        let mut equality_representatives = Vec::<(ty::GenericArg<'tcx>, u32)>::new();
        let mut next_equality_class = 0_u32;
        for index in 0..generics.count() {
            let parameter = generics.param_at(index, self.tcx);
            let kind = match parameter.kind {
                ty::GenericParamDefKind::Type { .. } => GenericParamKind::Type,
                ty::GenericParamDefKind::Const { .. } => GenericParamKind::Const,
                ty::GenericParamDefKind::Lifetime => continue,
            };
            let Some(argument) = arguments.get(parameter.index as usize).copied() else {
                status = GenericSubstitutionStatus::NormalizationIncomplete;
                continue;
            };
            let normalized = match normalize_generic_argument(self.tcx, caller, argument) {
                Some(normalized) => normalized,
                None => {
                    status = GenericSubstitutionStatus::NormalizationIncomplete;
                    argument
                }
            };
            let equality_class = equality_representatives
                .iter()
                .find_map(|(representative, class)| {
                    compiler_arguments_proven_equal(self.tcx, caller, *representative, normalized)
                        .then_some(*class)
                })
                .unwrap_or_else(|| {
                    let class = next_equality_class;
                    next_equality_class += 1;
                    equality_representatives.push((normalized, class));
                    class
                });
            let id = GenericParamId {
                def_path: self.tcx.def_path_str(parameter.def_id),
                kind,
            };
            bindings.insert(
                id,
                self.generic_argument_binding(caller, normalized, equality_class, kind),
            );
        }
        GenericCallSubstitution { status, bindings }
    }

    /// Produce only the narrow certificate needed to transport an already
    /// proven direct `S -> T` pointer witness through `helper::<P::Source,
    /// P::Target>`.  General projection normalization remains unknown.
    fn typed_witness_transport_templates(
        &self,
        caller: DefId,
        target: DefId,
        arguments: ty::GenericArgsRef<'tcx>,
        substitution: &GenericCallSubstitution,
    ) -> BTreeSet<TypedWitnessTransportTemplate> {
        let generics = self.tcx.generics_of(target);
        let parameters = (0..generics.count())
            .filter_map(|index| {
                let parameter = generics.param_at(index, self.tcx);
                matches!(parameter.kind, ty::GenericParamDefKind::Type { .. })
                    .then(|| {
                        arguments
                            .get(parameter.index as usize)
                            .copied()
                            .and_then(ty::GenericArg::as_type)
                            .map(|argument| {
                                (
                                    GenericParamId {
                                        def_path: self.tcx.def_path_str(parameter.def_id),
                                        kind: GenericParamKind::Type,
                                    },
                                    argument,
                                )
                            })
                    })
                    .flatten()
            })
            .collect::<Vec<_>>();
        let mut templates = BTreeSet::new();
        for (source_parameter, source_argument) in &parameters {
            for (target_parameter, target_argument) in &parameters {
                if source_parameter == target_parameter {
                    continue;
                }
                let Some(template) = self.typed_projection_witness_transport(
                    caller,
                    source_parameter,
                    *source_argument,
                    target_parameter,
                    *target_argument,
                    substitution,
                ) else {
                    continue;
                };
                templates.insert(template);
            }
        }
        templates
    }

    fn typed_projection_witness_transport(
        &self,
        caller: DefId,
        source_parameter: &GenericParamId,
        source_argument: ty::Ty<'tcx>,
        target_parameter: &GenericParamId,
        target_argument: ty::Ty<'tcx>,
        substitution: &GenericCallSubstitution,
    ) -> Option<TypedWitnessTransportTemplate> {
        let source_alias = first_type_alias(source_argument)?;
        let target_alias = first_type_alias(target_argument)?;
        let source_item = self
            .tcx
            .opt_associated_item(alias_ty_def_id(source_alias))?;
        let target_item = self
            .tcx
            .opt_associated_item(alias_ty_def_id(target_alias))?;
        let trait_id = source_item.trait_container(self.tcx)?;
        let same_trait = target_item.trait_container(self.tcx) == Some(trait_id);
        let distinct_items = source_item.def_id != target_item.def_id;
        let safe_trait = self.tcx.trait_def(trait_id).safety == Safety::Safe;
        let nameable = self.is_downstream_nameable(trait_id);
        let unsealed = !self.has_unnameable_sealing_supertrait(trait_id);
        let simple_supertraits = !self.has_complex_supertrait(trait_id);
        let no_blanket_impl = self.tcx.trait_impls_of(trait_id).blanket_impls().is_empty();
        let source_not_gat = self
            .tcx
            .generics_of(alias_ty_def_id(source_alias))
            .own_params
            .is_empty();
        let target_not_gat = self
            .tcx
            .generics_of(alias_ty_def_id(target_alias))
            .own_params
            .is_empty();
        if !(same_trait
            && distinct_items
            && safe_trait
            && nameable
            && unsealed
            && simple_supertraits
            && no_blanket_impl
            && source_not_gat
            && target_not_gat)
        {
            return None;
        }
        let source_self = source_alias.args.type_at(0);
        let target_self = target_alias.args.type_at(0);
        if source_self != target_self {
            return None;
        }
        let ty::Param(controller) = source_self.kind() else {
            return None;
        };
        let caller_parameter = self
            .tcx
            .generics_of(caller)
            .type_param(*controller, self.tcx);
        let caller_controller = GenericParamId {
            def_path: self.tcx.def_path_str(caller_parameter.def_id),
            kind: GenericParamKind::Type,
        };
        let simple_bound = self.simple_caller_projection_bound(caller, source_self, trait_id);
        let plain_hir = self.has_only_plain_hir_controller_bound(caller, *controller, trait_id);
        if !simple_bound || !plain_hir {
            return None;
        }
        let source_binding = substitution.bindings.get(source_parameter)?;
        let target_binding = substitution.bindings.get(target_parameter)?;
        if source_binding.caller_controllers != BTreeSet::from([caller_controller.clone()])
            || target_binding.caller_controllers != BTreeSet::from([caller_controller.clone()])
            || source_binding.form != CapabilityTermForm::Alias
            || target_binding.form != CapabilityTermForm::Alias
        {
            return None;
        }
        let source_witness = self.tcx.types.u8;
        let target_witness = self.tcx.types.u64;
        let source_status = associated_type_witness_satisfies_bounds(
            self.tcx,
            caller,
            source_alias,
            source_witness,
        );
        let target_status = associated_type_witness_satisfies_bounds(
            self.tcx,
            caller,
            target_alias,
            target_witness,
        );
        if source_status != WitnessBoundResult::Proven
            || target_status != WitnessBoundResult::Proven
        {
            return None;
        }
        Some(TypedWitnessTransportTemplate {
            source_parameter: source_parameter.clone(),
            target_parameter: target_parameter.clone(),
            caller_controller,
            assignment: BTreeMap::from([
                (CapabilityTermRole::SourcePointee, FiniteTypeWitness::U8),
                (CapabilityTermRole::TargetPointee, FiniteTypeWitness::U64),
            ]),
        })
    }

    fn simple_caller_projection_bound(
        &self,
        caller: DefId,
        controller: ty::Ty<'tcx>,
        trait_id: DefId,
    ) -> bool {
        let mut found = false;
        for clause in self.tcx.param_env(caller).caller_bounds().iter() {
            let Some(kind) = clause.kind().no_bound_vars() else {
                return false;
            };
            let ty::ClauseKind::Trait(predicate) = kind else {
                // Projection equality, negative predicates, HRTB and other
                // strengthened constraints are intentionally out of scope.
                return false;
            };
            if predicate.polarity != ty::PredicatePolarity::Positive {
                return false;
            }
            // Rust inserts this implicit positive bound for a normal type
            // parameter. It cannot constrain the finite Source/Target
            // assignment, unlike any user-written projection or trait bound.
            if predicate.trait_ref.self_ty() == controller
                && is_implicit_sizedness_trait(self.tcx, predicate.def_id())
                && predicate.trait_ref.args.len() == 1
            {
                continue;
            }
            // The certificate is intentionally for one plain `P: Trait`
            // clause only.  In particular, `P::Source: Extra`, another
            // bound on P, or trait arguments beyond Self all affect the
            // realizable assignment and must remain Unknown until a general
            // trait solver transport is available.
            if predicate.trait_ref.self_ty() != controller
                || predicate.def_id() != trait_id
                || predicate.trait_ref.args.len() != 1
                || found
            {
                return false;
            }
            found = true;
        }
        found
    }

    /// The parameter environment can omit source-level projection predicates
    /// from its normalized view. Inspect the local HIR declaration as the
    /// bounded certificate's final guard: one controller parameter may have
    /// exactly one plain trait bound and no additional predicates.
    /// This rejects associated-type equality/extra bounds, HRTBs, and any
    /// non-plain bound modifier before a finite witness is allowed to cross an
    /// otherwise incomplete normalization.
    fn has_only_plain_hir_controller_bound(
        &self,
        caller: DefId,
        controller: ty::ParamTy,
        trait_id: DefId,
    ) -> bool {
        let Some(caller_local) = caller.as_local() else {
            return false;
        };
        let controller_def_id = self
            .tcx
            .generics_of(caller)
            .type_param(controller, self.tcx)
            .def_id;
        has_only_plain_hir_bound(self.tcx, caller_local, controller_def_id, trait_id)
    }

    #[rustversion::before(1.96)]
    fn has_complex_supertrait(&self, trait_id: DefId) -> bool {
        self.tcx
            .explicit_super_predicates_of(trait_id)
            .skip_binder()
            .iter()
            .any(|(clause, _)| {
                let ty::ClauseKind::Trait(predicate) = clause.kind().skip_binder() else {
                    return true;
                };
                predicate.def_id() != trait_id
                    && !self.tcx.trait_is_auto(predicate.def_id())
                    && !is_implicit_sizedness_trait(self.tcx, predicate.def_id())
            })
    }

    #[rustversion::since(1.96)]
    fn has_complex_supertrait(&self, trait_id: DefId) -> bool {
        self.tcx
            .explicit_super_clauses_of(trait_id)
            .skip_binder()
            .iter()
            .any(|(clause, _)| {
                let ty::ClauseKind::Trait(predicate) = clause.kind().skip_binder() else {
                    return true;
                };
                predicate.def_id() != trait_id
                    && !self.tcx.trait_is_auto(predicate.def_id())
                    && !is_implicit_sizedness_trait(self.tcx, predicate.def_id())
            })
    }

    fn generic_argument_binding(
        &self,
        caller: DefId,
        argument: ty::GenericArg<'tcx>,
        equality_class: u32,
        kind: GenericParamKind,
    ) -> GenericArgBinding {
        let (caller_controllers, residual_aliases) = match generic_arg_kind(argument) {
            ty::GenericArgKind::Type(value) => self.generic_components(caller, [value]),
            ty::GenericArgKind::Const(value) => {
                let mut controllers = BTreeSet::new();
                if let ty::ConstKind::Param(parameter) = value.kind() {
                    let definition = self
                        .tcx
                        .generics_of(caller)
                        .const_param(parameter, self.tcx);
                    controllers.insert(GenericParamId {
                        def_path: self.tcx.def_path_str(definition.def_id),
                        kind: GenericParamKind::Const,
                    });
                }
                (controllers, BTreeSet::new())
            }
            ty::GenericArgKind::Lifetime(_) => (BTreeSet::new(), BTreeSet::new()),
        };
        let form = match generic_arg_kind(argument) {
            ty::GenericArgKind::Type(value) => match value.kind() {
                ty::Param(_) => CapabilityTermForm::DirectParam,
                ty::Alias(..) => CapabilityTermForm::Alias,
                _ if caller_controllers.is_empty() && residual_aliases.is_empty() => {
                    CapabilityTermForm::Concrete
                }
                _ => CapabilityTermForm::Composite,
            },
            ty::GenericArgKind::Const(value) => match value.kind() {
                ty::ConstKind::Param(_) => CapabilityTermForm::DirectParam,
                _ if caller_controllers.is_empty() => CapabilityTermForm::Concrete,
                _ => CapabilityTermForm::Composite,
            },
            ty::GenericArgKind::Lifetime(_) => CapabilityTermForm::Unknown,
        };
        let type_facts = argument
            .as_type()
            .map(|value| type_capability_facts(self.tcx, caller, value))
            .unwrap_or_default();
        let const_value = argument
            .as_const()
            .and_then(|value| target_usize_from_const(self.tcx, value));
        GenericArgBinding {
            kind,
            form,
            caller_controllers,
            residual_aliases,
            behavior_receiver: match generic_arg_kind(argument) {
                ty::GenericArgKind::Type(value) => self.typed_open_behavior_receiver(caller, value),
                ty::GenericArgKind::Const(_) | ty::GenericArgKind::Lifetime(_) => None,
            },
            equality_class: Some(equality_class),
            type_facts,
            const_value,
        }
    }

    fn is_public_safe_root(&self, id: LocalDefId) -> bool {
        let did = id.to_def_id();
        self.tcx.def_kind(did) != DefKind::Closure
            && self
                .tcx
                .effective_visibilities(())
                .is_public_at_level(id, Level::Reachable)
            && self.tcx.fn_sig(did).skip_binder().safety() == Safety::Safe
    }

    fn is_exported_ffi_callback(&self, did: DefId) -> bool {
        if !def_kind_has_direct_fn_sig(self.tcx.def_kind(did)) {
            return false;
        }
        let attrs = self.tcx.codegen_fn_attrs(did);
        self.is_exact_c_abi(did)
            && (attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE)
                || self.has_explicit_export_name(did))
    }

    #[rustversion::since(1.96)]
    fn has_explicit_export_name(&self, did: DefId) -> bool {
        self.tcx.codegen_fn_attrs(did).symbol_name.is_some()
    }

    #[rustversion::before(1.96)]
    fn has_explicit_export_name(&self, did: DefId) -> bool {
        self.tcx.codegen_fn_attrs(did).export_name.is_some()
    }

    #[rustversion::since(1.96)]
    fn is_exact_c_abi(&self, did: DefId) -> bool {
        matches!(
            self.tcx.fn_sig(did).skip_binder().abi(),
            ExternAbi::C { unwind: false }
        )
    }

    #[rustversion::before(1.96)]
    fn is_exact_c_abi(&self, did: DefId) -> bool {
        matches!(
            self.tcx.fn_sig(did).skip_binder().abi(),
            Abi::C { unwind: false }
        )
    }

    #[rustversion::since(1.96)]
    fn is_exact_foreign_c(&self, did: DefId) -> bool {
        self.tcx.is_foreign_item(did) && self.is_exact_c_abi(did)
    }

    #[rustversion::before(1.96)]
    fn is_exact_foreign_c(&self, did: DefId) -> bool {
        self.tcx.is_foreign_item(did) && self.is_exact_c_abi(did)
    }

    /// Calls whose return/alias semantics are modeled by the MIR transfer.
    ///
    /// This list never decides whether an unsafe site exists; that authority
    /// belongs exclusively to the compiler-aligned census and binder.
    fn is_registry_semantic_helper(&self, did: DefId) -> bool {
        self.is_get_unchecked(did)
            || self.is_nonnull_new_unchecked(did)
            || self.is_maybe_uninit_constructor(did, "uninit")
            || self.is_maybe_uninit_constructor(did, "new")
            || self.is_maybe_uninit_method(did, "write")
            || self.is_free_ptr_write(did)
            || self.free_ptr_copy_kind(did).is_some()
            || self.is_slice_len(did)
            || self.is_vec_len(did)
            || self.is_slice_iterator_constructor(did)
            || self.tcx.is_diagnostic_item(sym::assume_init, did)
            || self.tcx.is_diagnostic_item(sym::str_from_utf8, did)
            || self
                .tcx
                .is_diagnostic_item(sym::str_from_utf8_unchecked, did)
            || is_mem_size_of(self.tcx, did)
            || self.is_slice_is_empty(did)
            || self.is_pointer_is_null(did)
            || self.is_result_is_err(did)
            || self.is_saturating_sub(did)
            || self.is_checked_add(did)
            || self.is_try_trait_branch(did)
            || self.is_slice_prefix_index(did)
            || self.is_slice_as_ptr(did)
            || self.is_wrapping_add(did)
            || self.is_pointer_cast(did)
            || self.is_transparent_pointer_view_call(did)
            || self.is_ptr_read(did)
            || self.is_read_unaligned(did)
            || self.is_vec_get_unchecked(did)
            || self.is_string_from_utf8_unchecked(did)
            || self.is_slice_from_raw_parts(did)
            || self.is_cstr_from_ptr(did)
            || self.is_cstring_from_raw(did)
            || self.is_box_from_raw(did)
            || self.is_vec_from_raw_parts(did)
            || self.is_ptr_offset_op(did)
            || self.is_ptr_copy_contract(did)
    }

    fn known_contract_model(&self, did: DefId) -> Option<KnownContractModel> {
        if self.is_get_unchecked(did) {
            Some(KnownContractModel::GetUnchecked)
        } else if self.is_vec_get_unchecked(did) {
            Some(KnownContractModel::VecGetUnchecked)
        } else if self.is_read_unaligned(did) {
            Some(KnownContractModel::ReadUnaligned)
        } else if self.is_ptr_read(did) {
            Some(KnownContractModel::PtrRead)
        } else if self.is_nonnull_new_unchecked(did) {
            Some(KnownContractModel::NonNullNewUnchecked)
        } else if self.tcx.is_diagnostic_item(sym::assume_init, did) {
            Some(KnownContractModel::AssumeInit)
        } else if self
            .tcx
            .is_diagnostic_item(sym::str_from_utf8_unchecked, did)
        {
            Some(KnownContractModel::FromUtf8Unchecked)
        } else if self.is_string_from_utf8_unchecked(did) {
            Some(KnownContractModel::StringFromUtf8Unchecked)
        } else if self.is_slice_from_raw_parts(did) {
            Some(KnownContractModel::SliceFromRawParts)
        } else if self.is_cstr_from_ptr(did) {
            Some(KnownContractModel::CStrFromPtr)
        } else if self.is_cstring_from_raw(did) {
            Some(KnownContractModel::CStringFromRaw)
        } else if self.is_box_from_raw(did) {
            Some(KnownContractModel::BoxFromRaw)
        } else if self.is_vec_from_raw_parts(did) {
            Some(KnownContractModel::VecFromRawParts)
        } else if self.is_ptr_offset_op(did) {
            Some(KnownContractModel::PtrOffset)
        } else if self.is_ptr_copy_contract(did) {
            Some(KnownContractModel::PtrCopy)
        } else if self.tcx.is_intrinsic(did, sym::assume) || self.is_core_assert_unchecked(did) {
            Some(KnownContractModel::ConditionTrue)
        } else if self.tcx.is_intrinsic(did, sym::transmute) {
            Some(KnownContractModel::Transmute)
        } else {
            None
        }
    }

    fn known_contract_model_for_site(
        &self,
        operation: &ProgramOp,
        site: &BoundUnsafeSite,
    ) -> Option<KnownContractModel> {
        if site.seed.kind != unsafe_site::UnsafeSiteKind::UnsafeCall {
            return None;
        }
        if let ProgramOpKind::Call(call) = &operation.kind {
            if let Some(model) = call
                .descriptor
                .raw_def
                .and_then(|did| self.known_contract_model(did))
            {
                return Some(model);
            }
            if site
                .seed
                .callee
                .as_deref()
                .is_some_and(is_core_assert_unchecked_def_path)
            {
                return Some(KnownContractModel::ConditionTrue);
            }
        }
        if site
            .seed
            .callee
            .as_deref()
            .is_some_and(is_core_assume_intrinsic_def_path)
        {
            return Some(KnownContractModel::ConditionTrue);
        }
        let transmute = self.tcx.get_diagnostic_item(sym::transmute)?;
        (site.seed.callee.as_deref() == Some(self.tcx.def_path_str(transmute).as_str()))
            .then_some(KnownContractModel::Transmute)
    }

    fn is_core_assert_unchecked(&self, did: DefId) -> bool {
        self.tcx
            .lang_items()
            .sized_trait()
            .is_some_and(|sized| sized.krate == did.krate)
            && is_hint_assert_unchecked_def_path(&self.tcx.def_path(did))
    }

    fn is_get_unchecked(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| matches!(name.as_str(), "get_unchecked" | "get_unchecked_mut"))
            && self
                .associated_impl(did)
                .is_some_and(|impl_id| self.tcx.type_of(impl_id).skip_binder().is_slice())
    }

    fn is_vec_get_unchecked(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| matches!(name.as_str(), "get_unchecked" | "get_unchecked_mut"))
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::Vec, adt.did()))
            })
    }

    fn is_string_from_utf8_unchecked(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "from_utf8_unchecked")
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| {
                        self.tcx.crate_name(adt.did().krate).as_str() == "alloc"
                            && self.tcx.item_name(adt.did()).as_str() == "String"
                    })
            })
    }

    fn is_slice_from_raw_parts(&self, did: DefId) -> bool {
        self.tcx.is_diagnostic_item(sym::slice_from_raw_parts, did)
            || self
                .tcx
                .is_diagnostic_item(sym::slice_from_raw_parts_mut, did)
    }

    fn is_cstr_from_ptr(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "from_ptr")
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::cstr_type, adt.did()))
            })
    }

    fn is_cstring_from_raw(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "from_raw")
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| {
                        self.tcx
                            .opt_item_name(adt.did())
                            .is_some_and(|name| name.as_str() == "CString")
                    })
            })
    }

    fn is_box_from_raw(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "from_raw")
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| {
                        self.tcx.is_diagnostic_item(sym::owned_box, adt.did())
                            || self
                                .tcx
                                .opt_item_name(adt.did())
                                .is_some_and(|name| name.as_str() == "Box")
                    })
            })
    }

    fn is_vec_from_raw_parts(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "from_raw_parts")
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::Vec, adt.did()))
            })
    }

    fn is_ptr_offset_op(&self, did: DefId) -> bool {
        self.tcx.opt_item_name(did).is_some_and(|name| {
            matches!(name.as_str(), "add" | "offset" | "sub" | "wrapping_offset")
        }) && self.associated_impl(did).is_some_and(|impl_id| {
            matches!(
                self.tcx.type_of(impl_id).skip_binder().kind(),
                ty::RawPtr(..)
            )
        })
    }

    fn is_ptr_copy_contract(&self, did: DefId) -> bool {
        self.tcx.is_diagnostic_item(sym::ptr_copy, did)
            || self
                .tcx
                .is_diagnostic_item(sym::ptr_copy_nonoverlapping, did)
            || self.tcx.is_intrinsic(did, sym::copy)
            || self.tcx.is_intrinsic(did, sym::copy_nonoverlapping)
            || self
                .tcx
                .opt_item_name(did)
                .is_some_and(|name| matches!(name.as_str(), "memcpy" | "memmove"))
    }

    fn is_nonnull_new_unchecked(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "new_unchecked")
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::NonNull, adt.did()))
            })
    }

    fn is_maybe_uninit_constructor(&self, did: DefId, expected: &str) -> bool {
        self.is_maybe_uninit_method(did, expected)
    }

    fn is_maybe_uninit_method(&self, did: DefId, expected: &str) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == expected)
            && self.associated_impl(did).is_some_and(|impl_id| {
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
                .associated_impl(did)
                .is_some_and(|impl_id| self.tcx.type_of(impl_id).skip_binder().is_slice())
    }

    fn is_slice_len(&self, did: DefId) -> bool {
        self.tcx.lang_items().slice_len_fn() == Some(did)
    }

    fn is_vec_len(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "len")
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::Vec, adt.did()))
            })
    }

    fn is_slice_iterator_constructor(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| matches!(name.as_str(), "iter" | "iter_mut"))
            && self
                .associated_impl(did)
                .is_some_and(|impl_id| self.tcx.type_of(impl_id).skip_binder().is_slice())
    }

    fn is_slice_iterator_position(
        &self,
        did: DefId,
        body: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
    ) -> bool {
        if !self
            .tcx
            .opt_item_name(did)
            .is_some_and(|name| matches!(name.as_str(), "position" | "rposition"))
        {
            return false;
        }
        let Some(receiver) = args.first() else {
            return false;
        };
        let receiver = receiver.node.ty(&body.local_decls, self.tcx);
        let receiver = match receiver.kind() {
            ty::Ref(_, inner, _) => *inner,
            _ => receiver,
        };
        let Some(adt) = receiver.ty_adt_def() else {
            return false;
        };
        let path = self.tcx.def_path_str(adt.did());
        self.tcx.crate_name(adt.did().krate).as_str() == "core"
            && matches!(self.tcx.item_name(adt.did()).as_str(), "Iter" | "IterMut")
            && matches!(
                path.as_str(),
                "core::slice::iter::Iter"
                    | "core::slice::iter::IterMut"
                    | "slice::iter::Iter"
                    | "slice::iter::IterMut"
                    | "std::slice::Iter"
                    | "std::slice::IterMut"
            )
    }

    fn is_usize_range_iterator_next(
        &self,
        did: DefId,
        body: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
    ) -> bool {
        if !self
            .tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "next")
            || !self
                .associated_trait(did)
                .is_some_and(|trait_id| self.tcx.is_lang_item(trait_id, LangItem::Iterator))
        {
            return false;
        }
        let Some(receiver) = args.first() else {
            return false;
        };
        let receiver = receiver.node.ty(&body.local_decls, self.tcx);
        let receiver = match receiver.kind() {
            ty::Ref(_, inner, _) => *inner,
            _ => receiver,
        };
        let ty::Adt(definition, generic_args) = receiver.kind() else {
            return false;
        };
        Some(definition.did()) == self.tcx.lang_items().range_struct()
            && matches!(generic_args.type_at(0).kind(), ty::Uint(ty::UintTy::Usize))
    }

    fn is_usize_range_into_iterator(
        &self,
        did: DefId,
        body: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
    ) -> bool {
        if !self.tcx.is_lang_item(did, LangItem::IntoIterIntoIter) {
            return false;
        }
        let Some(receiver) = args.first() else {
            return false;
        };
        let ty::Adt(definition, generic_args) =
            receiver.node.ty(&body.local_decls, self.tcx).kind()
        else {
            return false;
        };
        Some(definition.did()) == self.tcx.lang_items().range_struct()
            && matches!(generic_args.type_at(0).kind(), ty::Uint(ty::UintTy::Usize))
    }

    fn is_pointer_is_null(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "is_null")
            && self.associated_impl(did).is_some_and(|impl_id| {
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
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_diagnostic_item(sym::Result, adt.did()))
            })
    }

    fn is_internal_utf8_validation_call(
        &self,
        caller: DefId,
        callee: DefId,
        resolved_local: bool,
        body: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        destination_ty: ty::Ty<'tcx>,
    ) -> bool {
        // `str::from_utf8` delegates its actual scan to a private helper that
        // has no diagnostic item.  Bind that helper only inside the compiler-
        // identified public validator and only when its complete typed
        // signature is `same input -> Result<(), same error>`.
        let caller_is_str_from_utf8 = self.tcx.is_diagnostic_item(sym::str_from_utf8, caller);
        if !caller_is_str_from_utf8 || !resolved_local || args.len() != 1 {
            return false;
        }
        if !def_kind_has_direct_fn_sig(self.tcx.def_kind(callee)) {
            return false;
        }
        // MIR operand regions are erased. Instantiate each late-bound
        // signature region with the same erased region before comparing the
        // compiler-identified caller, private helper, and actual operand.
        let caller_signature = erased_identity_fn_sig(self.tcx, caller);
        let callee_signature = erased_identity_fn_sig(self.tcx, callee);
        let Some(caller_input) = caller_signature.inputs().first().copied() else {
            return false;
        };
        let [callee_input] = callee_signature.inputs() else {
            return false;
        };
        let callee_output = callee_signature.output();
        let Some((validator_ok, validator_error)) = result_type_arguments(self.tcx, callee_output)
        else {
            return false;
        };
        let Some((_, caller_error)) = result_type_arguments(self.tcx, caller_signature.output())
        else {
            return false;
        };
        let actual_input = args[0].node.ty(&body.local_decls, self.tcx);
        let erase = |value| erase_type_regions(self.tcx, value);
        exact_utf8_validation_call_shape(Utf8ValidationCallShape {
            caller_is_str_from_utf8,
            resolved_local,
            one_argument: true,
            actual_matches_caller_input: erase(actual_input) == erase(caller_input),
            callee_input_matches_actual: erase(*callee_input) == erase(actual_input),
            destination_matches_callee_output: erase(destination_ty) == erase(callee_output),
            validator_ok_is_unit: validator_ok.is_unit(),
            error_matches: validator_error == caller_error,
        })
    }

    fn is_saturating_sub(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "saturating_sub")
            && self.associated_impl(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::Uint(ty::UintTy::Usize)
                )
            })
    }

    fn is_checked_add(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "checked_add")
            && self.associated_impl(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::Uint(ty::UintTy::Usize)
                )
            })
    }

    fn is_try_trait_branch(&self, did: DefId) -> bool {
        // Depending on monomorphization, MIR may name either `Try::branch`
        // itself or the concrete impl item. Match only compiler-owned trait
        // identity; the input fact decides which validation atom propagates.
        self.tcx.is_lang_item(did, LangItem::TryTraitBranch)
            || self.associated_trait_item(did).is_some_and(|trait_item| {
                self.tcx.is_lang_item(trait_item, LangItem::TryTraitBranch)
            })
    }

    fn is_slice_prefix_index(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "index")
            && self
                .associated_trait(did)
                .is_some_and(|trait_id| self.tcx.item_name(trait_id).as_str() == "Index")
    }

    fn is_slice_as_ptr(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "as_ptr")
            && self
                .associated_impl(did)
                .is_some_and(|impl_id| self.tcx.type_of(impl_id).skip_binder().is_slice())
    }

    fn is_wrapping_add(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "wrapping_add")
            && self.associated_impl(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::RawPtr(..)
                )
            })
    }

    fn is_pointer_cast(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| matches!(name.as_str(), "cast" | "cast_const" | "cast_mut"))
            && self.associated_impl(did).is_some_and(|impl_id| {
                matches!(
                    self.tcx.type_of(impl_id).skip_binder().kind(),
                    ty::RawPtr(..)
                )
            })
    }

    fn is_transparent_pointer_view_call(&self, did: DefId) -> bool {
        let Some(name) = self.tcx.opt_item_name(did) else {
            return false;
        };
        let Some(adt) = self
            .associated_impl(did)
            .and_then(|impl_id| self.tcx.type_of(impl_id).skip_binder().ty_adt_def())
        else {
            return false;
        };
        (self.tcx.is_lang_item(adt.did(), LangItem::MaybeUninit)
            && matches!(name.as_str(), "as_ptr" | "as_mut_ptr"))
            || (self.tcx.is_lang_item(adt.did(), LangItem::UnsafeCell)
                && matches!(name.as_str(), "get" | "raw_get"))
    }

    fn is_maybe_uninit_pointer_view(&self, did: DefId) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| matches!(name.as_str(), "as_ptr" | "as_mut_ptr"))
            && self.associated_impl(did).is_some_and(|impl_id| {
                self.tcx
                    .type_of(impl_id)
                    .skip_binder()
                    .ty_adt_def()
                    .is_some_and(|adt| self.tcx.is_lang_item(adt.did(), LangItem::MaybeUninit))
            })
    }

    fn is_free_ptr_write(&self, did: DefId) -> bool {
        self.tcx.is_diagnostic_item(sym::ptr_write, did)
            || self.tcx.is_intrinsic(did, sym::write_via_move)
    }

    fn free_ptr_copy_kind(&self, did: DefId) -> Option<bool> {
        if self
            .tcx
            .is_diagnostic_item(sym::ptr_copy_nonoverlapping, did)
            || self.tcx.is_intrinsic(did, sym::copy_nonoverlapping)
        {
            Some(true)
        } else if self.tcx.is_diagnostic_item(sym::ptr_copy, did)
            || self.tcx.is_intrinsic(did, sym::copy)
        {
            Some(false)
        } else {
            None
        }
    }

    fn is_read_unaligned(&self, did: DefId) -> bool {
        self.tcx.is_diagnostic_item(sym::ptr_read_unaligned, did)
            || (self
                .tcx
                .opt_item_name(did)
                .is_some_and(|name| name.as_str() == "read_unaligned")
                && self.associated_impl(did).is_some_and(|impl_id| {
                    matches!(
                        self.tcx.type_of(impl_id).skip_binder().kind(),
                        ty::RawPtr(..)
                    )
                }))
    }

    fn is_ptr_read(&self, did: DefId) -> bool {
        self.tcx.is_diagnostic_item(sym::ptr_read, did)
            || (self
                .tcx
                .opt_item_name(did)
                .is_some_and(|name| name.as_str() == "read")
                && self.associated_impl(did).is_some_and(|impl_id| {
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
        args: &[Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
    ) -> bool {
        self.tcx
            .opt_item_name(did)
            .is_some_and(|name| name.as_str() == "as_ref")
            && self
                .associated_trait(did)
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

    fn open_behavior_controller(
        &self,
        caller: DefId,
        did: DefId,
        generic_args: ty::GenericArgsRef<'tcx>,
        callee: &Option<FunctionKey>,
        body: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
    ) -> Option<BehaviorController> {
        if callee.is_some() || !self.is_open_trait_method(did) {
            return None;
        }
        let receiver_ty = args
            .first()
            .map(|receiver| receiver.node.ty(&body.local_decls, self.tcx))?;
        let receiver = self.typed_open_behavior_receiver(caller, receiver_ty)?;
        let trait_id = self.associated_trait(did)?;
        let resolved_target = resolve_instance(self.tcx, caller, did, generic_args).and_then(
            |instance| match instance.def {
                InstanceKind::Item(target) if self.associated_impl(target).is_some() => {
                    Some(unsafe_site::compiler_function_key(self.tcx, target))
                }
                _ => None,
            },
        );
        let openness = if resolved_target.is_some()
            || !self.is_downstream_nameable(trait_id)
            || self.has_unnameable_sealing_supertrait(trait_id)
        {
            BehaviorOpenness::ProvenClosed
        } else {
            match &receiver {
                OpenBehaviorReceiver::Parameter(_) | OpenBehaviorReceiver::Dynamic => {
                    BehaviorOpenness::ProvenOpen
                }
                OpenBehaviorReceiver::Alias(_) | OpenBehaviorReceiver::Unknown => {
                    BehaviorOpenness::Unknown
                }
            }
        };
        Some(BehaviorController::OpenTrait {
            trait_id: unsafe_site::compiler_function_key(self.tcx, trait_id),
            openness,
            resolved_target,
            receiver,
        })
    }

    fn typed_open_behavior_receiver(
        &self,
        owner: DefId,
        mut receiver: ty::Ty<'tcx>,
    ) -> Option<OpenBehaviorReceiver> {
        while let ty::Ref(_, inner, _) = receiver.kind() {
            receiver = *inner;
        }
        match receiver.kind() {
            ty::Param(parameter) => {
                let definition = self.tcx.generics_of(owner).type_param(*parameter, self.tcx);
                Some(OpenBehaviorReceiver::Parameter(GenericParamId {
                    def_path: self.tcx.def_path_str(definition.def_id),
                    kind: GenericParamKind::Type,
                }))
            }
            ty::Dynamic(..) => Some(OpenBehaviorReceiver::Dynamic),
            ty::Alias(..) => {
                let controllers = self.generic_components(owner, [receiver]).0;
                Some(if controllers.is_empty() {
                    OpenBehaviorReceiver::Unknown
                } else {
                    OpenBehaviorReceiver::Alias(controllers)
                })
            }
            _ => None,
        }
    }

    fn is_downstream_nameable(&self, did: DefId) -> bool {
        did.as_local().is_some_and(|local| {
            self.tcx
                .effective_visibilities(())
                .is_public_at_level(local, Level::Reachable)
        }) || (!did.is_local() && self.tcx.visible_parent_map(()).contains_key(&did))
    }

    #[rustversion::before(1.96)]
    fn has_unnameable_sealing_supertrait(&self, trait_id: DefId) -> bool {
        self.tcx
            .explicit_super_predicates_of(trait_id)
            .skip_binder()
            .iter()
            .any(|(clause, _)| {
                let ty::ClauseKind::Trait(predicate) = clause.kind().skip_binder() else {
                    return false;
                };
                let supertrait = predicate.def_id();
                supertrait != trait_id
                    && !self.tcx.trait_is_auto(supertrait)
                    && !self.is_downstream_nameable(supertrait)
                    && self
                        .tcx
                        .trait_impls_of(supertrait)
                        .blanket_impls()
                        .is_empty()
            })
    }

    #[rustversion::since(1.96)]
    fn has_unnameable_sealing_supertrait(&self, trait_id: DefId) -> bool {
        self.tcx
            .explicit_super_clauses_of(trait_id)
            .skip_binder()
            .iter()
            .any(|(clause, _)| {
                let ty::ClauseKind::Trait(predicate) = clause.kind().skip_binder() else {
                    return false;
                };
                let supertrait = predicate.def_id();
                supertrait != trait_id
                    && !self.tcx.trait_is_auto(supertrait)
                    && !self.is_downstream_nameable(supertrait)
                    && self
                        .tcx
                        .trait_impls_of(supertrait)
                        .blanket_impls()
                        .is_empty()
            })
    }

    fn stable_span(&self, span: Span) -> Result<StableSpan, String> {
        let span = span.source_callsite();
        let source_map = self.tcx.sess.source_map();
        let (file, lo_line, lo_column, hi_line, hi_column) = source_map.span_to_location_info(span);
        let local = file.as_ref().map(|file| local_file_name(&file.name));
        let remapped = file.as_ref().map(|file| remapped_file_name(&file.name));
        stable_span_from_locations(
            &self.source_root,
            local.as_deref(),
            remapped.as_deref(),
            lo_line,
            lo_column,
            hi_line,
            hi_column,
        )
    }

    fn point(
        &self,
        function: &FunctionKey,
        location: Location,
        span: Span,
    ) -> Result<ProgramPoint, String> {
        Ok(ProgramPoint {
            function: function.clone(),
            block: location.block.index() as u32,
            statement: location.statement_index as u32,
            span: self.stable_span(span)?,
        })
    }
}

fn def_kind_has_direct_fn_sig(kind: DefKind) -> bool {
    matches!(kind, DefKind::Fn | DefKind::AssocFn)
}

fn insert_binding_record(
    records: &mut BTreeMap<String, SiteBindingRecord>,
    record: SiteBindingRecord,
) -> Result<bool, String> {
    match records.get(&record.site_id) {
        Some(existing) if existing == &record => Ok(false),
        Some(_) => Err(format!(
            "conflicting unsafe-site binding record for {}",
            record.site_id
        )),
        None => {
            records.insert(record.site_id.clone(), record);
            Ok(true)
        }
    }
}

#[rustversion::before(1.96)]
fn is_const_binding_owner(kind: DefKind) -> bool {
    matches!(
        kind,
        DefKind::Const | DefKind::AssocConst | DefKind::AnonConst | DefKind::InlineConst
    )
}

#[rustversion::since(1.96)]
fn is_const_binding_owner(kind: DefKind) -> bool {
    matches!(
        kind,
        DefKind::Const { .. } | DefKind::AssocConst { .. } | DefKind::AnonConst
    )
}

fn uses_ctfe_mir(kind: DefKind) -> bool {
    is_const_binding_owner(kind) || matches!(kind, DefKind::Static { .. })
}

#[rustversion::before(1.96)]
pub(super) fn local_file_name(name: &FileName) -> String {
    name.display(FileNameDisplayPreference::Local).to_string()
}

#[rustversion::since(1.96)]
pub(super) fn local_file_name(name: &FileName) -> String {
    name.prefer_local_unconditionally().to_string()
}

#[rustversion::before(1.96)]
pub(super) fn remapped_file_name(name: &FileName) -> String {
    name.display(FileNameDisplayPreference::Remapped)
        .to_string()
}

#[rustversion::since(1.96)]
pub(super) fn remapped_file_name(name: &FileName) -> String {
    name.prefer_remapped_unconditionally().to_string()
}

#[rustversion::before(1.96)]
fn associated_item_has_self(tcx: TyCtxt<'_>, did: DefId) -> bool {
    tcx.opt_associated_item(did)
        .is_some_and(|item| item.fn_has_self_parameter)
}

#[rustversion::since(1.96)]
fn associated_item_has_self(tcx: TyCtxt<'_>, did: DefId) -> bool {
    tcx.opt_associated_item(did)
        .is_some_and(|item| item.is_method())
}

#[rustversion::before(1.96)]
fn rvalue_use_operand<'a, 'tcx>(rvalue: &'a Rvalue<'tcx>) -> Option<&'a Operand<'tcx>> {
    match rvalue {
        Rvalue::Use(operand) => Some(operand),
        _ => None,
    }
}

#[rustversion::since(1.96)]
fn rvalue_use_operand<'a, 'tcx>(rvalue: &'a Rvalue<'tcx>) -> Option<&'a Operand<'tcx>> {
    match rvalue {
        Rvalue::Use(operand, _) => Some(operand),
        _ => None,
    }
}

#[rustversion::before(1.96)]
fn rvalue_len_place<'tcx>(rvalue: &Rvalue<'tcx>) -> Option<Place<'tcx>> {
    match rvalue {
        Rvalue::Len(place) => Some(*place),
        _ => None,
    }
}

#[rustversion::since(1.96)]
fn rvalue_len_place<'tcx>(_rvalue: &Rvalue<'tcx>) -> Option<Place<'tcx>> {
    None
}

#[rustversion::before(1.96)]
fn is_mem_size_of(tcx: TyCtxt<'_>, did: DefId) -> bool {
    tcx.is_diagnostic_item(sym::mem_size_of, did)
}

#[rustversion::since(1.96)]
fn is_mem_size_of(tcx: TyCtxt<'_>, did: DefId) -> bool {
    tcx.is_diagnostic_item(Symbol::intern("mem_size_of"), did)
}

#[rustversion::before(1.96)]
fn is_implicit_sizedness_trait(tcx: TyCtxt<'_>, did: DefId) -> bool {
    tcx.lang_items().sized_trait() == Some(did)
}

#[rustversion::since(1.96)]
fn is_implicit_sizedness_trait(tcx: TyCtxt<'_>, did: DefId) -> bool {
    tcx.lang_items().sized_trait() == Some(did) || tcx.lang_items().meta_sized_trait() == Some(did)
}

#[rustversion::before(1.96)]
fn type_layout_size<'tcx>(tcx: TyCtxt<'tcx>, caller: DefId, ty: Ty<'tcx>) -> Option<u64> {
    tcx.layout_of(tcx.param_env(caller).and(ty))
        .ok()
        .map(|layout| layout.size.bytes())
}

#[rustversion::before(1.96)]
fn type_layout_align<'tcx>(tcx: TyCtxt<'tcx>, caller: DefId, ty: Ty<'tcx>) -> Option<u64> {
    tcx.layout_of(tcx.param_env(caller).and(ty))
        .ok()
        .map(|layout| layout.align.abi.bytes())
}

#[rustversion::since(1.96)]
fn type_layout_size<'tcx>(tcx: TyCtxt<'tcx>, caller: DefId, ty: Ty<'tcx>) -> Option<u64> {
    tcx.layout_of(ty::TypingEnv::post_analysis(tcx, caller).as_query_input(ty))
        .ok()
        .map(|layout| layout.size.bytes())
}

#[rustversion::since(1.96)]
fn type_layout_align<'tcx>(tcx: TyCtxt<'tcx>, caller: DefId, ty: Ty<'tcx>) -> Option<u64> {
    tcx.layout_of(ty::TypingEnv::post_analysis(tcx, caller).as_query_input(ty))
        .ok()
        .map(|layout| layout.align.abi.bytes())
}

fn raw_pointer_pointee<'tcx>(value: Ty<'tcx>) -> Option<Ty<'tcx>> {
    match value.kind() {
        ty::RawPtr(pointee, _) => Some(*pointee),
        _ => None,
    }
}

fn sequence_element_ty<'tcx>(value: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let value = match value.kind() {
        ty::Ref(_, inner, _) => *inner,
        _ => value,
    };
    match value.kind() {
        ty::Array(element, _) | ty::Slice(element) => Some(*element),
        _ => None,
    }
}

#[rustversion::before(1.96)]
fn generic_arg_kind<'tcx>(argument: ty::GenericArg<'tcx>) -> ty::GenericArgKind<'tcx> {
    argument.unpack()
}

#[rustversion::since(1.96)]
fn generic_arg_kind<'tcx>(argument: ty::GenericArg<'tcx>) -> ty::GenericArgKind<'tcx> {
    argument.kind()
}

#[rustversion::before(1.96)]
fn alias_ty_def_id(alias: ty::AliasTy<'_>) -> DefId {
    alias.def_id
}

#[rustversion::since(1.96)]
fn alias_ty_def_id(alias: ty::AliasTy<'_>) -> DefId {
    match alias.kind {
        ty::AliasTyKind::Projection { def_id }
        | ty::AliasTyKind::Inherent { def_id }
        | ty::AliasTyKind::Opaque { def_id }
        | ty::AliasTyKind::Free { def_id } => def_id,
    }
}

fn first_type_alias<'tcx>(value: Ty<'tcx>) -> Option<ty::AliasTy<'tcx>> {
    value
        .walk()
        .find_map(|argument| match generic_arg_kind(argument) {
            ty::GenericArgKind::Type(value) => match value.kind() {
                ty::Alias(_, alias) => Some(*alias),
                _ => None,
            },
            ty::GenericArgKind::Lifetime(_) | ty::GenericArgKind::Const(_) => None,
        })
}

#[rustversion::before(1.96)]
fn normalize_type<'tcx>(tcx: TyCtxt<'tcx>, owner: DefId, value: Ty<'tcx>) -> Option<Ty<'tcx>> {
    tcx.try_normalize_erasing_regions(tcx.param_env(owner), value)
        .ok()
}

#[rustversion::since(1.96)]
fn normalize_type<'tcx>(tcx: TyCtxt<'tcx>, owner: DefId, value: Ty<'tcx>) -> Option<Ty<'tcx>> {
    tcx.try_normalize_erasing_regions(
        ty::TypingEnv::post_analysis(tcx, owner),
        ty::Unnormalized::new_wip(value),
    )
    .ok()
}

#[rustversion::before(1.96)]
fn normalize_generic_argument<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    value: ty::GenericArg<'tcx>,
) -> Option<ty::GenericArg<'tcx>> {
    tcx.try_normalize_erasing_regions(tcx.param_env(owner), value)
        .ok()
}

#[rustversion::since(1.96)]
fn normalize_generic_argument<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    value: ty::GenericArg<'tcx>,
) -> Option<ty::GenericArg<'tcx>> {
    tcx.try_normalize_erasing_regions(
        ty::TypingEnv::post_analysis(tcx, owner),
        ty::Unnormalized::new_wip(value),
    )
    .ok()
}

/// Decide equality only from rustc's exact generic arguments and the
/// projection equalities in the caller's `ParamEnv`. This is deliberately a
/// finite equality closure: it proves cases such as
/// `P: Pair<Source = T, Target = T>` without attempting open-world trait
/// normalization or parsing rendered type names.
fn compiler_arguments_proven_equal<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    left: ty::GenericArg<'tcx>,
    right: ty::GenericArg<'tcx>,
) -> bool {
    let normalize = |argument| normalize_generic_argument(tcx, owner, argument).unwrap_or(argument);
    let left = normalize(left);
    let right = normalize(right);
    if left == right {
        return true;
    }

    let mut equalities = Vec::new();
    for clause in tcx.param_env(owner).caller_bounds().iter() {
        let ty::ClauseKind::Projection(predicate) = clause.kind().skip_binder() else {
            continue;
        };
        let projection = projection_term_argument(tcx, predicate.projection_term);
        let term: ty::GenericArg<'tcx> = predicate.term.into();
        if projection.has_escaping_bound_vars() || term.has_escaping_bound_vars() {
            continue;
        }
        equalities.push((normalize(projection), normalize(term)));
    }

    let mut pending = vec![left];
    let mut seen = vec![left];
    while let Some(current) = pending.pop() {
        for (first, second) in &equalities {
            let next = if current == *first {
                Some(*second)
            } else if current == *second {
                Some(*first)
            } else {
                None
            };
            let Some(next) = next else {
                continue;
            };
            if next == right {
                return true;
            }
            if !seen.contains(&next) {
                seen.push(next);
                pending.push(next);
            }
        }
    }
    false
}

#[rustversion::before(1.96)]
fn projection_term_argument<'tcx>(
    tcx: TyCtxt<'tcx>,
    projection: ty::AliasTerm<'tcx>,
) -> ty::GenericArg<'tcx> {
    projection.to_term(tcx).into()
}

#[rustversion::since(1.96)]
fn projection_term_argument<'tcx>(
    tcx: TyCtxt<'tcx>,
    projection: ty::AliasTerm<'tcx>,
) -> ty::GenericArg<'tcx> {
    projection.to_term(tcx, ty::IsRigid::No).into()
}

fn type_capability_facts<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    value: Ty<'tcx>,
) -> CapabilityTypeFacts {
    CapabilityTypeFacts {
        size: type_layout_size(tcx, owner, value),
        align: type_layout_align(tcx, owner, value),
        inhabited: None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WitnessBoundResult {
    Proven,
    Rejected,
    Unsupported,
}

#[rustversion::before(1.96)]
fn associated_type_witness_satisfies_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    _owner: DefId,
    alias: ty::AliasTy<'tcx>,
    witness: Ty<'tcx>,
) -> WitnessBoundResult {
    let infcx = tcx.infer_ctxt().build();
    for (clause, _) in tcx
        .explicit_item_bounds(alias.def_id)
        .iter_instantiated_copied(tcx, alias.args)
    {
        let Some(kind) = clause.kind().no_bound_vars() else {
            return WitnessBoundResult::Unsupported;
        };
        let ty::ClauseKind::Trait(predicate) = kind else {
            return WitnessBoundResult::Unsupported;
        };
        if predicate.polarity != ty::PredicatePolarity::Positive {
            return WitnessBoundResult::Unsupported;
        }
        let mut arguments = predicate.trait_ref.args.iter().collect::<Vec<_>>();
        let Some(self_argument) = arguments.first_mut() else {
            return WitnessBoundResult::Unsupported;
        };
        *self_argument = witness.into();
        if !infcx
            .type_implements_trait(predicate.def_id(), arguments, ty::ParamEnv::empty())
            .must_apply_modulo_regions()
        {
            return WitnessBoundResult::Rejected;
        }
    }
    WitnessBoundResult::Proven
}

#[rustversion::since(1.96)]
fn associated_type_witness_satisfies_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    _owner: DefId,
    alias: ty::AliasTy<'tcx>,
    witness: Ty<'tcx>,
) -> WitnessBoundResult {
    let infcx = tcx.infer_ctxt().build(ty::TypingMode::PostAnalysis);
    for bound in tcx
        .explicit_item_bounds(alias_ty_def_id(alias))
        .iter_instantiated_copied(tcx, alias.args)
    {
        let (clause, _) = bound.skip_normalization();
        let Some(kind) = clause.kind().no_bound_vars() else {
            return WitnessBoundResult::Unsupported;
        };
        let ty::ClauseKind::Trait(predicate) = kind else {
            return WitnessBoundResult::Unsupported;
        };
        if predicate.polarity != ty::PredicatePolarity::Positive {
            return WitnessBoundResult::Unsupported;
        }
        let mut arguments = predicate.trait_ref.args.iter().collect::<Vec<_>>();
        let Some(self_argument) = arguments.first_mut() else {
            return WitnessBoundResult::Unsupported;
        };
        *self_argument = witness.into();
        if !infcx
            .type_implements_trait(predicate.def_id(), arguments, ty::ParamEnv::empty())
            .must_apply_modulo_regions()
        {
            return WitnessBoundResult::Rejected;
        }
    }
    WitnessBoundResult::Proven
}

#[rustversion::before(1.96)]
fn direct_type_parameter_witnesses_satisfy_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    source: Ty<'tcx>,
    source_witness: Ty<'tcx>,
    target: Ty<'tcx>,
    target_witness: Ty<'tcx>,
) -> WitnessBoundResult {
    if !matches!(source.kind(), ty::Param(_)) || !matches!(target.kind(), ty::Param(_)) {
        return WitnessBoundResult::Unsupported;
    }
    let identity = ty::GenericArgs::identity_for_item(tcx, owner);
    let substitutions =
        tcx.mk_args_from_iter(identity.iter().map(|argument| match argument.unpack() {
            ty::GenericArgKind::Type(value) if value == source => source_witness.into(),
            ty::GenericArgKind::Type(value) if value == target => target_witness.into(),
            _ => argument,
        }));
    // This bounded proof handles only a fully concrete instantiation of the
    // two direct type parameters. Any third type/const controller or
    // normalization residue remains Unestablished at the caller.
    if substitutions.has_non_region_param() {
        return WitnessBoundResult::Unsupported;
    }

    let infcx = tcx.infer_ctxt().build();
    for clause in tcx
        .predicates_of(owner)
        .instantiate(tcx, substitutions)
        .predicates
    {
        let Some(kind) = clause.kind().no_bound_vars() else {
            return WitnessBoundResult::Unsupported;
        };
        let ty::ClauseKind::Trait(predicate) = kind else {
            return WitnessBoundResult::Unsupported;
        };
        if predicate.polarity != ty::PredicatePolarity::Positive
            || predicate.trait_ref.args.has_non_region_param()
        {
            return WitnessBoundResult::Unsupported;
        }
        if !infcx
            .type_implements_trait(
                predicate.def_id(),
                predicate.trait_ref.args.iter().collect::<Vec<_>>(),
                ty::ParamEnv::empty(),
            )
            .must_apply_modulo_regions()
        {
            return WitnessBoundResult::Rejected;
        }
    }
    WitnessBoundResult::Proven
}

#[rustversion::since(1.96)]
fn direct_type_parameter_witnesses_satisfy_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    source: Ty<'tcx>,
    source_witness: Ty<'tcx>,
    target: Ty<'tcx>,
    target_witness: Ty<'tcx>,
) -> WitnessBoundResult {
    if !matches!(source.kind(), ty::Param(_)) || !matches!(target.kind(), ty::Param(_)) {
        return WitnessBoundResult::Unsupported;
    }
    let identity = ty::GenericArgs::identity_for_item(tcx, owner);
    let substitutions = tcx.mk_args_from_iter(identity.iter().map(
        |argument| match generic_arg_kind(argument) {
            ty::GenericArgKind::Type(value) if value == source => source_witness.into(),
            ty::GenericArgKind::Type(value) if value == target => target_witness.into(),
            _ => argument,
        },
    ));
    if substitutions.has_non_region_param() {
        return WitnessBoundResult::Unsupported;
    }

    let infcx = tcx.infer_ctxt().build(ty::TypingMode::PostAnalysis);
    for (clause, _) in tcx.clauses_of(owner).instantiate(tcx, substitutions) {
        let clause = clause.skip_normalization();
        let Some(kind) = clause.kind().no_bound_vars() else {
            return WitnessBoundResult::Unsupported;
        };
        let ty::ClauseKind::Trait(predicate) = kind else {
            return WitnessBoundResult::Unsupported;
        };
        if predicate.polarity != ty::PredicatePolarity::Positive
            || predicate.trait_ref.args.has_non_region_param()
        {
            return WitnessBoundResult::Unsupported;
        }
        if !infcx
            .type_implements_trait(
                predicate.def_id(),
                predicate.trait_ref.args.iter().collect::<Vec<_>>(),
                ty::ParamEnv::empty(),
            )
            .must_apply_modulo_regions()
        {
            return WitnessBoundResult::Rejected;
        }
    }
    WitnessBoundResult::Proven
}

fn pointee_layout_implies_target<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner: DefId,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
) -> bool {
    let (Some(source_size), Some(target_size), Some(source_align), Some(target_align)) = (
        type_layout_size(tcx, owner, source),
        type_layout_size(tcx, owner, target),
        type_layout_align(tcx, owner, source),
        type_layout_align(tcx, owner, target),
    ) else {
        return false;
    };
    target_size <= source_size && target_align <= source_align
}

fn pointee_validity_implies_target(source: Ty<'_>, target: Ty<'_>) -> bool {
    let source = pointee_validity_class(source);
    let target = pointee_validity_class(target);
    source.guarantees_initialized_bytes()
        && matches!(
            target,
            TypeValidityClass::AllBitsValidScalar
                | TypeValidityClass::AllBitsValidBytes
                | TypeValidityClass::AllBitsValidStructural
                | TypeValidityClass::AllBitsValidRawPointer
        )
}

fn pointee_validity_class(ty: Ty<'_>) -> TypeValidityClass {
    match ty.kind() {
        ty::Int(_) | ty::Uint(_) | ty::Float(_) => TypeValidityClass::AllBitsValidScalar,
        ty::Bool | ty::Char | ty::Never | ty::Pat(_, _) => TypeValidityClass::RestrictedScalar,
        ty::Array(element, _) => match pointee_validity_class(*element) {
            TypeValidityClass::AllBitsValidScalar
                if matches!(
                    element.kind(),
                    ty::Uint(ty::UintTy::U8) | ty::Int(ty::IntTy::I8)
                ) =>
            {
                TypeValidityClass::AllBitsValidBytes
            }
            TypeValidityClass::AllBitsValidScalar
            | TypeValidityClass::AllBitsValidBytes
            | TypeValidityClass::AllBitsValidStructural
            | TypeValidityClass::AllBitsValidRawPointer => {
                TypeValidityClass::AllBitsValidStructural
            }
            TypeValidityClass::RestrictedScalar | TypeValidityClass::RestrictedStructural => {
                TypeValidityClass::RestrictedStructural
            }
            TypeValidityClass::ReferenceOrFunctionPointer => {
                TypeValidityClass::ReferenceOrFunctionPointer
            }
            TypeValidityClass::RestrictedEnum => TypeValidityClass::RestrictedEnum,
            TypeValidityClass::UninitializedOrPadding => TypeValidityClass::UninitializedOrPadding,
            TypeValidityClass::Unknown => TypeValidityClass::Unknown,
        },
        ty::Tuple(fields) if fields.is_empty() => TypeValidityClass::AllBitsValidStructural,
        ty::Tuple(_) => TypeValidityClass::UninitializedOrPadding,
        ty::RawPtr(_, _) => TypeValidityClass::AllBitsValidRawPointer,
        ty::Ref(_, _, _) | ty::FnDef(_, _) | ty::FnPtr(..) => {
            TypeValidityClass::ReferenceOrFunctionPointer
        }
        ty::Adt(definition, _) if definition.is_union() => {
            TypeValidityClass::UninitializedOrPadding
        }
        ty::Adt(definition, _) if definition.is_enum() => TypeValidityClass::RestrictedEnum,
        _ => TypeValidityClass::Unknown,
    }
}

#[rustversion::before(1.96)]
fn has_only_plain_hir_bound(
    tcx: TyCtxt<'_>,
    caller: LocalDefId,
    controller: DefId,
    trait_id: DefId,
) -> bool {
    let Some(generics) = tcx.hir().get_generics(caller) else {
        return false;
    };
    let [predicate] = generics.predicates else {
        return false;
    };
    let WherePredicate::BoundPredicate(bound_predicate) = predicate else {
        return false;
    };
    if !bound_predicate.bound_generic_params.is_empty() {
        return false;
    }
    let TyKind::Path(QPath::Resolved(None, path)) = bound_predicate.bounded_ty.kind else {
        return false;
    };
    if path.res.opt_def_id() != Some(controller) {
        return false;
    }
    let [bound] = bound_predicate.bounds else {
        return false;
    };
    let GenericBound::Trait(poly_trait, modifier) = bound else {
        return false;
    };
    *modifier == rustc_hir::TraitBoundModifier::None
        && poly_trait.bound_generic_params.is_empty()
        && poly_trait.trait_ref.path.res.opt_def_id() == Some(trait_id)
        && poly_trait
            .trait_ref
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.args.is_none())
}

#[rustversion::since(1.96)]
fn has_only_plain_hir_bound(
    tcx: TyCtxt<'_>,
    caller: LocalDefId,
    controller: DefId,
    trait_id: DefId,
) -> bool {
    let Some(generics) = tcx.hir_get_generics(caller) else {
        return false;
    };
    let [predicate] = generics.predicates else {
        return false;
    };
    let WherePredicateKind::BoundPredicate(bound_predicate) = predicate.kind else {
        return false;
    };
    if !bound_predicate.bound_generic_params.is_empty() {
        return false;
    }
    let TyKind::Path(QPath::Resolved(None, path)) = bound_predicate.bounded_ty.kind else {
        return false;
    };
    if path.res.opt_def_id() != Some(controller) {
        return false;
    }
    let [bound] = bound_predicate.bounds else {
        return false;
    };
    let GenericBound::Trait(poly_trait) = bound else {
        return false;
    };
    poly_trait.modifiers == rustc_hir::TraitBoundModifiers::NONE
        && poly_trait.bound_generic_params.is_empty()
        && poly_trait.trait_ref.path.res.opt_def_id() == Some(trait_id)
        && poly_trait
            .trait_ref
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.args.is_none())
}

#[rustversion::before(1.96)]
fn resolve_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: DefId,
    raw: DefId,
    args: ty::GenericArgsRef<'tcx>,
) -> Option<Instance<'tcx>> {
    Instance::try_resolve(tcx, tcx.param_env(caller), raw, args)
        .ok()
        .flatten()
}

#[rustversion::since(1.96)]
fn resolve_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: DefId,
    raw: DefId,
    args: ty::GenericArgsRef<'tcx>,
) -> Option<Instance<'tcx>> {
    Instance::try_resolve(tcx, ty::TypingEnv::post_analysis(tcx, caller), raw, args)
        .ok()
        .flatten()
}

#[rustversion::before(1.96)]
fn target_usize_from_const<'tcx>(tcx: TyCtxt<'tcx>, value: ty::Const<'tcx>) -> Option<u64> {
    // rustc 1.83's target-usize helper converts before checking the evaluated
    // scalar type, which ICEs on non-usize const arguments.
    let (value_ty, scalar) = value.try_eval_scalar_int(tcx, ty::ParamEnv::empty())?;
    if !matches!(value_ty.kind(), ty::Uint(ty::UintTy::Usize)) {
        return None;
    }
    Some(scalar.to_target_usize(tcx))
}

#[rustversion::since(1.96)]
fn target_usize_from_const<'tcx>(tcx: TyCtxt<'tcx>, value: ty::Const<'tcx>) -> Option<u64> {
    value.try_to_target_usize(tcx)
}

#[rustversion::before(1.96)]
fn eval_operand_target_usize<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: DefId,
    value: MirConst<'tcx>,
) -> Option<u64> {
    value.try_eval_target_usize(tcx, tcx.param_env(caller))
}

#[rustversion::since(1.96)]
fn eval_operand_target_usize<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: DefId,
    value: MirConst<'tcx>,
) -> Option<u64> {
    value.try_eval_target_usize(tcx, ty::TypingEnv::post_analysis(tcx, caller))
}

#[rustversion::since(1.96)]
fn supports_local_optimized_mir_body(tcx: TyCtxt<'_>, id: LocalDefId, kind: DefKind) -> bool {
    match kind {
        DefKind::Closure => true,
        DefKind::Fn | DefKind::AssocFn => {
            supports_optimized_mir_body(kind, tcx.hir_body_const_context(id))
        }
        _ => false,
    }
}

#[rustversion::before(1.96)]
fn supports_local_optimized_mir_body(_tcx: TyCtxt<'_>, _id: LocalDefId, kind: DefKind) -> bool {
    matches!(kind, DefKind::Fn | DefKind::AssocFn | DefKind::Closure)
}

#[rustversion::before(1.96)]
fn hir_body_value_span(tcx: TyCtxt<'_>, id: LocalDefId) -> Option<Span> {
    tcx.hir()
        .maybe_body_owned_by(id)
        .map(|body| body.value.span)
}

#[rustversion::since(1.96)]
fn hir_body_value_span(tcx: TyCtxt<'_>, id: LocalDefId) -> Option<Span> {
    tcx.hir_maybe_body_owned_by(id).map(|body| body.value.span)
}

fn merge_function_code_span(
    definition: &StableSpan,
    body: &StableSpan,
) -> Result<StableSpan, String> {
    if definition.path != body.path {
        return Err(format!(
            "causal function definition and body use different source paths: '{}' and '{}'",
            definition.path, body.path
        ));
    }
    Ok(StableSpan::new(
        definition.path.clone(),
        definition.start.clone().min(body.start.clone()),
        definition.end.clone().max(body.end.clone()),
    ))
}

#[rustversion::since(1.96)]
fn supports_optimized_mir_body(kind: DefKind, context: Option<ConstContext>) -> bool {
    matches!(kind, DefKind::Fn | DefKind::AssocFn)
        && !matches!(
            context,
            Some(ConstContext::Const { .. } | ConstContext::Static(_))
        )
}

#[rustversion::before(1.96)]
fn is_behavior_pointer_coercion(coercion: ty::adjustment::PointerCoercion) -> bool {
    matches!(
        coercion,
        ty::adjustment::PointerCoercion::ReifyFnPointer
            | ty::adjustment::PointerCoercion::UnsafeFnPointer
            | ty::adjustment::PointerCoercion::ClosureFnPointer(_)
    )
}

#[rustversion::since(1.96)]
fn is_behavior_pointer_coercion(coercion: ty::adjustment::PointerCoercion) -> bool {
    matches!(
        coercion,
        ty::adjustment::PointerCoercion::ReifyFnPointer(_)
            | ty::adjustment::PointerCoercion::UnsafeFnPointer
            | ty::adjustment::PointerCoercion::ClosureFnPointer(_)
    )
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

fn relation_storage_places(relation: &RelationFact) -> Vec<PlaceKey> {
    fn insert_term(places: &mut BTreeSet<PlaceKey>, term: &RelationTerm) {
        let place = match term {
            RelationTerm::Place(place)
            | RelationTerm::Length(place)
            | RelationTerm::Capacity(place)
            | RelationTerm::Projection { base: place, .. } => Some(place.clone()),
            RelationTerm::Formal(index) => Some(PlaceKey::new(format!("_{index}"))),
            RelationTerm::Return => Some(PlaceKey::new("_0")),
            RelationTerm::Out(index) => Some(out_slot(*index)),
            RelationTerm::Constant(_) => None,
        };
        if let Some(place) = place {
            places.insert(canonical_storage(&place));
        }
    }

    let mut places = BTreeSet::new();
    match &relation.kind {
        RelationKind::Eq(left, right)
        | RelationKind::Le(left, right)
        | RelationKind::Lt(left, right)
        | RelationKind::SameObject(left, right) => {
            insert_term(&mut places, left);
            insert_term(&mut places, right);
        }
        RelationKind::AddConst { result, base, .. } => {
            insert_term(&mut places, result);
            insert_term(&mut places, base);
        }
        RelationKind::Sub {
            result,
            upper,
            lower,
        } => {
            insert_term(&mut places, result);
            insert_term(&mut places, upper);
            insert_term(&mut places, lower);
        }
        RelationKind::SumEq { total, left, right } => {
            insert_term(&mut places, total);
            insert_term(&mut places, left);
            insert_term(&mut places, right);
        }
        RelationKind::Within { base, start, end }
        | RelationKind::InitializedWithin { base, start, end } => {
            places.insert(canonical_storage(base));
            insert_term(&mut places, start);
            insert_term(&mut places, end);
        }
        RelationKind::CapacityFor { base, end } => {
            places.insert(canonical_storage(base));
            insert_term(&mut places, end);
        }
    }
    places.into_iter().collect()
}

fn state_value(state: &ProgramState, place: &PlaceKey) -> ValueFacts {
    let mut value = state.may_values().get(place).cloned().unwrap_or_default();
    let storage = canonical_storage(place);
    if storage != *place {
        if let Some(storage_value) = state.may_values().get(&storage) {
            value.join_may(storage_value);
        }
    }
    value
}

fn exact_semantics(state: &ProgramState, place: &PlaceKey) -> (BTreeSet<PlaceKey>, bool) {
    state
        .may_values()
        .get(place)
        .map(|facts| (facts.exact_roots.clone(), facts.semantic_unknown))
        .unwrap_or_default()
}

fn semantic_value_at(state: &ProgramState, place: &PlaceKey) -> ValueFacts {
    let (exact_roots, semantic_unknown) = exact_semantics(state, place);
    ValueFacts {
        exact_roots,
        semantic_unknown,
        ..ValueFacts::default()
    }
}

fn current_storage_versions(state: &ProgramState, place: &PlaceKey) -> BTreeSet<StaticWriteToken> {
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

fn local_sink_operand_place(operand: &CallOperand, value: &ValueFacts) -> Option<PlaceKey> {
    // The raw fallback names the exact operand used by this sink. It is not a
    // semantic-identity proof; discharge still requires current relation facts.
    unique_semantic_value(value).or_else(|| match operand {
        CallOperand::Place(place) => Some(place.clone()),
        CallOperand::Constant(_) | CallOperand::ArrayLengths(_) | CallOperand::Unknown => None,
    })
}

fn assume_init_requirement(
    function: &FunctionKey,
    call: &CallSite,
    legacy_bool_return: bool,
) -> ContractRequirement {
    let origin = AbstractOrigin::InternalLocal {
        function: function.clone(),
        place: call.destination.clone(),
    };
    ContractRequirement {
        seed_id: format!("P5.assume_init:{}", call.point.span.token()),
        collection: None,
        subject: AbstractValue::new([origin]),
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
        predicates: if legacy_bool_return {
            BTreeSet::from([Predicate::Initialized, Predicate::ValidBool])
        } else {
            BTreeSet::from([Predicate::Initialized])
        },
        access_width: None,
        rule: if legacy_bool_return {
            RuleId::P5AssumeInitBool
        } else {
            RuleId::P5AssumeInitUninitialized
        },
        return_exposure: legacy_bool_return,
        sink_function: function.clone(),
    }
}

fn typed_initialization_state(value: &ValueFacts) -> TypedInitializationState {
    if value.havoced || value.ffi_output {
        return TypedInitializationState::Unknown;
    }
    match value
        .maybe_uninit_initialized
        .iter()
        .copied()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [true] => TypedInitializationState::Initialized,
        [false] => TypedInitializationState::Uninitialized,
        [] | [false, true] => TypedInitializationState::Unknown,
        _ => unreachable!("MaybeUninit facts contain only boolean lattice atoms"),
    }
}

fn definitely_uninitialized(value: &ValueFacts) -> bool {
    typed_initialization_state(value) == TypedInitializationState::Uninitialized
}

fn definitely_initialized(value: &ValueFacts) -> bool {
    typed_initialization_state(value) == TypedInitializationState::Initialized
}

/// A concrete validity mismatch is externally controllable only when the
/// exact influence domain transports invalid bit patterns to the cast.  Entry
/// storage provenance alone is insufficient: a safe reference formal carries
/// the validity invariant of its Rust type even though its address is supplied
/// by the caller.
fn has_externally_controlled_validity_bits(value: &ValueFacts) -> bool {
    value
        .influence_seeds
        .has_capability(InfluenceCapabilities::VALIDITY_BITS)
}

/// Absence of external provenance is not proof of local storage.  In
/// particular, a resolved helper whose return summary has not transported
/// storage identity is `Untracked`, not a source-local object whose pointee
/// extent can be bounded by one compiler-known source value.
fn establishes_source_local_generic_capability(value: &ValueFacts) -> bool {
    value.storage_provenance == StorageProvenance::DefinitelyLocal
}

fn assume_init_requires_requirement(value: &ValueFacts) -> bool {
    !definitely_initialized(value)
}

const FFI_UNINITIALIZED_OUTPUT_PREFIX: &str = "ffi-uninit-out:";

fn is_uninitialized_ffi_output_origin(origin: &AbstractOrigin) -> bool {
    matches!(origin, AbstractOrigin::FfiOutput { token, .. }
        if token.starts_with(FFI_UNINITIALIZED_OUTPUT_PREFIX))
}

fn ffi_invalid_value_subject(value: &ValueFacts) -> Option<AbstractValue> {
    let ffi_origins = value
        .value
        .origins
        .iter()
        .filter(|origin| is_uninitialized_ffi_output_origin(origin))
        .cloned()
        .collect::<BTreeSet<_>>();
    (!ffi_origins.is_empty()).then(|| AbstractValue::new(ffi_origins))
}

fn unchecked_utf8_requirement(function: &FunctionKey, call: &CallSite) -> ContractRequirement {
    ContractRequirement {
        seed_id: format!("P5.utf8:{}", call.point.span.token()),
        collection: None,
        subject: call
            .arg_values
            .first()
            .map(|value| value.value.clone())
            .unwrap_or_default(),
        source_hint: None,
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
        rule: RuleId::P5UncheckedUtf8,
        return_exposure: false,
        sink_function: function.clone(),
    }
}

fn ffi_invalid_value_requirement(
    function: &FunctionKey,
    point: &ProgramPoint,
    subject: AbstractValue,
) -> ContractRequirement {
    ContractRequirement {
        seed_id: format!("P3.ffi-invalid-value:{}", point.span.token()),
        collection: None,
        subject,
        source_hint: Some(SourceKind::FfiOutput),
        internal_derivation: false,
        source_span: point.span.clone(),
        first_failure: Operation {
            kind: OperationKind::InvalidValueExposure,
            point: point.clone(),
        },
        sink: Operation {
            kind: OperationKind::InvalidValueExposure,
            point: point.clone(),
        },
        predicates: BTreeSet::from([Predicate::ValidForType]),
        access_width: None,
        rule: RuleId::P3FfiOutInvalidValue,
        return_exposure: false,
        sink_function: function.clone(),
    }
}

fn p3_requirement_matches_call_destination(
    requirement: &ContractRequirement,
    destination_restricted: bool,
) -> bool {
    requirement.rule != RuleId::P3FfiOutInvalidValue
        || !requirement
            .subject
            .origins
            .iter()
            .any(is_uninitialized_ffi_output_origin)
        || destination_restricted
}

fn unique_place(places: &BTreeSet<PlaceKey>) -> Option<PlaceKey> {
    let mut places = places.iter();
    let place = places.next()?.clone();
    places.next().is_none().then_some(place)
}

fn length_relation_is_current(
    state: &ProgramState,
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

fn current_length_collection(state: &ProgramState, value: &ValueFacts) -> Option<PlaceKey> {
    let mut collections = value.len_of.iter();
    let collection = collections.next()?.clone();
    if collections.next().is_some()
        || value.may_be_non_length
        || value.length_relation_unknown
        || value.havoced
        || !length_relation_is_current(state, value, &collection)
    {
        return None;
    }
    Some(collection)
}

fn current_utf8_bytes(state: &ProgramState, value: &ValueFacts) -> Option<PlaceKey> {
    let mut atoms = value.utf8_atoms.iter();
    let atom = atoms.next()?;
    if atoms.next().is_some()
        || value.utf8_relation_unknown
        || value.havoced
        || atom.invalidated
        || atom.captured_versions != current_storage_versions(state, &atom.bytes)
    {
        return None;
    }
    Some(atom.bytes.clone())
}

fn constant_candidates(value: &ValueFacts) -> (BTreeSet<u64>, bool) {
    let constants = value
        .value
        .origins
        .iter()
        .filter_map(|origin| match origin {
            AbstractOrigin::Constant(value) => Some(*value),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let unknown = constants.is_empty()
        || value
            .value
            .origins
            .iter()
            .any(|origin| !matches!(origin, AbstractOrigin::Constant(_)));
    (constants, unknown)
}

fn range_binding_for_pointer(value: &ValueFacts, width: u64) -> Option<ValidationBinding> {
    Some(ValidationBinding::range(
        unique_place(&value.pointer_offset)?,
        Some(unique_place(&value.pointer_base)?),
        width,
        None,
    ))
}

fn range_index_obligation_for_base(
    call: &CallSite,
    collection: &PlaceKey,
    base: &PlaceKey,
) -> Option<RelationalObligation> {
    let kind = call.index_range_kind?;
    let field = |index| RelationTerm::Projection {
        base: canonical_storage(base),
        path: vec![ProjectionKey::Field(index)],
    };
    let length = RelationTerm::Length(collection.clone());
    let (start, end) = match kind {
        RangeIndexKind::Full => (RelationTerm::Constant(0), length),
        RangeIndexKind::From => (field(0), length),
        RangeIndexKind::To => (RelationTerm::Constant(0), field(0)),
        RangeIndexKind::Between => (field(0), field(1)),
    };
    Some(RelationalObligation::Within {
        base: collection.clone(),
        start,
        end,
    })
}

fn range_index_obligation(call: &CallSite, collection: &PlaceKey) -> Option<RelationalObligation> {
    let range = call.arg_values.get(1)?;
    if range.havoced {
        return None;
    }
    let base = unique_semantic_value(range)?;
    range_index_obligation_for_base(call, collection, &base)
}

fn equivalent_semantic_candidates(
    snapshot: &ValidationSnapshot,
    value: &ValueFacts,
) -> Option<BTreeSet<PlaceKey>> {
    if value.semantic_unknown || value.havoced {
        return None;
    }
    let candidates = semantic_place_candidates(value);
    let first = candidates.first()?;
    let proof = RelationProof::from_facts(snapshot.current_relations.iter().cloned());
    candidates
        .iter()
        .all(|candidate| {
            proof.proves_same_value(
                &RelationTerm::Place(first.clone()),
                &RelationTerm::Place(candidate.clone()),
            )
        })
        .then_some(candidates)
}

fn proven_range_index_obligation(
    snapshot: &ValidationSnapshot,
    call: &CallSite,
) -> Option<RelationalObligation> {
    let collections = equivalent_semantic_candidates(snapshot, call.arg_values.first()?)?;
    let ranges = equivalent_semantic_candidates(snapshot, call.arg_values.get(1)?)?;
    collections.iter().find_map(|collection| {
        ranges.iter().find_map(|range| {
            let obligation = range_index_obligation_for_base(call, collection, range)?;
            snapshot
                .proves_relational(&obligation)
                .then_some(obligation)
        })
    })
}

fn current_checked_end(state: &ProgramState, value: &ValueFacts) -> Option<CheckedEndAtom> {
    let mut atoms = value.checked_end_atoms.iter();
    let atom = atoms.next()?;
    if atoms.next().is_some()
        || value.checked_end_unknown
        || value.havoced
        || !atom.branch_ready
        || atom.invalidated
        || atom.captured_versions != current_storage_versions(state, &atom.offset)
    {
        return None;
    }
    Some(atom.clone())
}

fn current_range_limit(state: &ProgramState, value: &ValueFacts) -> Option<RangeLimitAtom> {
    let mut atoms = value.range_limit_atoms.iter();
    let atom = atoms.next()?;
    if atoms.next().is_some()
        || value.range_limit_unknown
        || value.havoced
        || atom.invalidated
        || atom.captured_versions != current_storage_versions(state, &atom.collection)
    {
        return None;
    }
    Some(atom.clone())
}

fn current_relation_proof(state: &ProgramState) -> RelationProof {
    RelationProof::from_facts(state.must_validations().iter().filter_map(|validation| {
        let relation = validation.predicate().relation()?;
        state
            .has_current_validation(
                &ProofBinding::Relation(relation.clone()),
                relation_storage_places(relation),
            )
            .then(|| relation.clone())
    }))
}

#[derive(Clone, Debug)]
enum RegistryRelationEffect {
    Unconditional(RelationFact),
    Pending {
        relation: RelationFact,
        polarity: bool,
    },
}

fn current_slice_iterator_base(state: &ProgramState, iterator: &OperandModel) -> Option<PlaceKey> {
    let iterator = unique_semantic_value(&eval_operand(state, iterator))?;
    let mut bases = current_relation_proof(state)
        .facts()
        .filter_map(|fact| match fact.kind {
            RelationKind::Eq(RelationTerm::Length(left), RelationTerm::Length(right))
                if left == iterator && right != iterator =>
            {
                Some(right)
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    (bases.len() == 1).then(|| bases.pop_first().expect("one iterator base"))
}

fn current_range_iterator_bounds(
    state: &ProgramState,
    iterator: &OperandModel,
) -> BTreeSet<PlaceKey> {
    let Some(iterator) = unique_semantic_value(&eval_operand(state, iterator)) else {
        return BTreeSet::new();
    };
    let proof = current_relation_proof(state);
    proof
        .facts()
        .filter_map(|fact| {
            let (base, collection) = match fact.kind {
                RelationKind::Eq(
                    RelationTerm::Projection { base, path },
                    RelationTerm::Length(collection),
                ) if path.as_slice() == [ProjectionKey::Field(1)] => (base, collection),
                RelationKind::Eq(
                    RelationTerm::Length(collection),
                    RelationTerm::Projection { base, path },
                ) if path.as_slice() == [ProjectionKey::Field(1)] => (base, collection),
                _ => return None,
            };
            proof
                .proves_same_value(
                    &RelationTerm::Place(iterator.clone()),
                    &RelationTerm::Place(base),
                )
                .then_some(collection)
        })
        .collect()
}

fn registry_relation_effects(
    state: &ProgramState,
    call: &CallModel,
) -> Vec<RegistryRelationEffect> {
    match &call.value {
        RegistryValueModel::TypedInitializationPointer { .. }
        | RegistryValueModel::IdentityPointerView { .. }
        | RegistryValueModel::AddressPointerView { .. }
        | RegistryValueModel::PointerCast { .. }
        | RegistryValueModel::GenericPointerCast { .. } => Vec::new(),
        RegistryValueModel::Length(source) => {
            let source_value = eval_operand(state, source);
            // A modeled `len` result is tied to this call's actual receiver even
            // when the receiver's broader semantic provenance remains unknown.
            let Some(collection) =
                unique_semantic_value(&source_value).or_else(|| source.operand.place().cloned())
            else {
                return Vec::new();
            };
            vec![RegistryRelationEffect::Unconditional(RelationFact::new(
                RelationKind::Eq(
                    RelationTerm::Place(call.descriptor.destination.clone()),
                    RelationTerm::Length(collection),
                ),
            ))]
        }
        RegistryValueModel::SlicePrefix { range, .. } => {
            let range = state_value(state, range);
            let mut ends = range.range_end.iter();
            let Some(end) = ends.next() else {
                return Vec::new();
            };
            if ends.next().is_some() || range.havoced {
                return Vec::new();
            }
            // On the normal edge of `&base[..end]`, the resulting slice has
            // exactly `end` elements. This helper postcondition is generic and
            // supplies both guard discharge and the finite `index == len`
            // counterexample when the same `end` is used unchecked.
            vec![RegistryRelationEffect::Unconditional(RelationFact::new(
                RelationKind::Eq(
                    RelationTerm::Length(call.descriptor.destination.clone()),
                    RelationTerm::Place(end.clone()),
                ),
            ))]
        }
        RegistryValueModel::SliceIterator(slice) => {
            let Some(collection) = unique_semantic_value(&eval_operand(state, slice)) else {
                return Vec::new();
            };
            vec![RegistryRelationEffect::Unconditional(RelationFact::new(
                RelationKind::Eq(
                    RelationTerm::Length(call.descriptor.destination.clone()),
                    RelationTerm::Length(collection),
                ),
            ))]
        }
        RegistryValueModel::SlicePosition(iterator) => {
            let Some(collection) = current_slice_iterator_base(state, iterator) else {
                return Vec::new();
            };
            vec![RegistryRelationEffect::Pending {
                relation: RelationFact::new(RelationKind::Lt(
                    RelationTerm::Projection {
                        base: call.descriptor.destination.clone(),
                        path: vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)],
                    },
                    RelationTerm::Length(collection),
                )),
                polarity: true,
            }]
        }
        RegistryValueModel::RangeIntoIterator(range) => current_range_iterator_bounds(state, range)
            .into_iter()
            .map(|collection| {
                RegistryRelationEffect::Unconditional(RelationFact::new(RelationKind::Eq(
                    RelationTerm::Projection {
                        base: call.descriptor.destination.clone(),
                        path: vec![ProjectionKey::Field(1)],
                    },
                    RelationTerm::Length(collection),
                )))
            })
            .collect(),
        RegistryValueModel::RangeIteratorNext(iterator) => {
            let Some(iterator_base) = unique_semantic_value(&eval_operand(state, iterator)) else {
                return Vec::new();
            };
            let bounds = current_range_iterator_bounds(state, iterator);
            bounds
                .into_iter()
                .flat_map(|collection| {
                    [
                        RegistryRelationEffect::Unconditional(RelationFact::new(RelationKind::Eq(
                            RelationTerm::Projection {
                                base: iterator_base.clone(),
                                path: vec![ProjectionKey::Field(1)],
                            },
                            RelationTerm::Length(collection.clone()),
                        ))),
                        RegistryRelationEffect::Pending {
                            relation: RelationFact::new(RelationKind::Lt(
                                RelationTerm::Projection {
                                    base: call.descriptor.destination.clone(),
                                    path: vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)],
                                },
                                RelationTerm::Length(collection),
                            )),
                            polarity: true,
                        },
                    ]
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

fn project_summary_term(
    term: &RelationTerm,
    state: &ProgramState,
    arg_count: usize,
) -> Option<SummaryRelationTerm> {
    let project_place = |place: &PlaceKey| {
        if *place == PlaceKey::new("_0") {
            return Some(SummaryRelationTerm::Return);
        }
        if let Some(formal_index) = out_slot_index(place) {
            return Some(SummaryRelationTerm::Out(formal_index));
        }
        match entry_formal_slot(arg_count, state, place) {
            Some(BoundarySlot::Formal(index)) => Some(SummaryRelationTerm::Formal(index)),
            _ => None,
        }
    };
    match term {
        RelationTerm::Place(place) => project_place(place),
        RelationTerm::Length(place) => match project_place(place)? {
            SummaryRelationTerm::Formal(index) => Some(SummaryRelationTerm::LengthFormal(index)),
            _ => None,
        },
        RelationTerm::Capacity(place) => match project_place(place)? {
            SummaryRelationTerm::Formal(index) => Some(SummaryRelationTerm::CapacityFormal(index)),
            _ => None,
        },
        RelationTerm::Constant(value) => Some(SummaryRelationTerm::Constant(*value)),
        RelationTerm::Projection { base, path } => match project_place(base)? {
            SummaryRelationTerm::Formal(formal) => Some(SummaryRelationTerm::FormalProjection {
                formal,
                path: path.clone(),
            }),
            SummaryRelationTerm::Return => {
                Some(SummaryRelationTerm::ReturnProjection(path.clone()))
            }
            SummaryRelationTerm::Out(formal) => Some(SummaryRelationTerm::OutProjection {
                formal,
                path: path.clone(),
            }),
            _ => None,
        },
        RelationTerm::Formal(index) => Some(SummaryRelationTerm::Formal(*index)),
        RelationTerm::Return => Some(SummaryRelationTerm::Return),
        RelationTerm::Out(index) => Some(SummaryRelationTerm::Out(*index)),
    }
}

fn project_summary_relation(
    relation: &RelationKind,
    state: &ProgramState,
    arg_count: usize,
) -> Option<SummaryRelationKind> {
    let term = |term| project_summary_term(term, state, arg_count);
    Some(match relation {
        RelationKind::Eq(left, right) => SummaryRelationKind::Eq(term(left)?, term(right)?),
        RelationKind::Le(left, right) => SummaryRelationKind::Le(term(left)?, term(right)?),
        RelationKind::Lt(left, right) => SummaryRelationKind::Lt(term(left)?, term(right)?),
        RelationKind::AddConst {
            result,
            base,
            addend,
        } => SummaryRelationKind::AddConst {
            result: term(result)?,
            base: term(base)?,
            addend: *addend,
        },
        RelationKind::Sub {
            result,
            upper,
            lower,
        } => SummaryRelationKind::Sub {
            result: term(result)?,
            upper: term(upper)?,
            lower: term(lower)?,
        },
        RelationKind::SumEq { total, left, right } => SummaryRelationKind::SumEq {
            total: term(total)?,
            left: term(left)?,
            right: term(right)?,
        },
        RelationKind::Within { base, start, end } => {
            let SummaryRelationTerm::Formal(base_formal) =
                project_summary_term(&RelationTerm::Place(base.clone()), state, arg_count)?
            else {
                return None;
            };
            SummaryRelationKind::Within {
                base_formal,
                start: term(start)?,
                end: term(end)?,
            }
        }
        RelationKind::InitializedWithin { base, start, end } => {
            let SummaryRelationTerm::Formal(base_formal) =
                project_summary_term(&RelationTerm::Place(base.clone()), state, arg_count)?
            else {
                return None;
            };
            SummaryRelationKind::InitializedWithin {
                base_formal,
                start: term(start)?,
                end: term(end)?,
            }
        }
        RelationKind::CapacityFor { base, end } => {
            let SummaryRelationTerm::Formal(base_formal) =
                project_summary_term(&RelationTerm::Place(base.clone()), state, arg_count)?
            else {
                return None;
            };
            SummaryRelationKind::CapacityFor {
                base_formal,
                end: term(end)?,
            }
        }
        RelationKind::SameObject(left, right) => {
            let left = term(left)?;
            let right = term(right)?;
            if left == right {
                return None;
            }
            SummaryRelationKind::SameObject(left, right)
        }
    })
}

fn project_relational_postconditions(
    state: &ProgramState,
    arg_count: usize,
) -> BTreeSet<RelationalPostcondition> {
    let condition = current_return_condition(state);
    current_relation_proof(state)
        .facts()
        .filter_map(|fact| {
            let relation = project_summary_relation(&fact.kind, state, arg_count)?;
            Some(match &condition {
                Some(condition) => {
                    RelationalPostcondition::conditional(condition.clone(), relation)
                }
                None => RelationalPostcondition::unconditional(relation),
            })
        })
        .collect()
}

fn current_return_condition(state: &ProgramState) -> Option<ReturnCondition> {
    let mut variants = state.must_validations().iter().filter_map(|validation| {
        let variant = validation.predicate().return_variant()?;
        state
            .has_current_validation(
                &ProofBinding::ReturnVariant(variant.clone()),
                vec![canonical_storage(&variant.destination)],
            )
            .then(|| ReturnCondition {
                discriminant: variant.discriminant,
                payload_path: variant.payload_path.clone(),
            })
    });
    let variant = variants.next()?;
    variants.next().is_none().then_some(variant)
}

fn reachable_relational_postconditions(
    edges: &IndexVec<BasicBlock, EdgeTerm>,
    solved: &ProgramDataflowResult,
    arg_count: usize,
) -> BTreeSet<RelationalPostcondition> {
    let returns = edges
        .iter_enumerated()
        .filter(|(block, edge)| matches!(edge, EdgeTerm::Return) && solved.exit[*block].is_some())
        .filter_map(|(block, _)| solved.exit[block].as_ref())
        .map(|state| {
            (
                current_return_condition(state),
                project_relational_postconditions(state, arg_count),
            )
        })
        .collect::<Vec<_>>();
    let mut conditions = returns
        .iter()
        .map(|(condition, _)| condition.clone())
        .collect::<BTreeSet<_>>();
    let mut result = BTreeSet::new();
    for condition in std::mem::take(&mut conditions) {
        let mut candidates = returns
            .iter()
            .filter(|(candidate, _)| candidate == &condition)
            .map(|(_, postconditions)| postconditions.clone());
        let Some(mut common) = candidates.next() else {
            continue;
        };
        for postconditions in candidates {
            common.retain(|postcondition| postconditions.contains(postcondition));
        }
        result.extend(common);
    }
    result
}

fn relation_term_for_operand(state: &ProgramState, operand: &OperandModel) -> Option<RelationTerm> {
    if let CallOperand::Constant(value) = &operand.operand {
        return Some(RelationTerm::Constant(*value));
    }
    if let CallOperand::ArrayLengths(collections) = &operand.operand {
        let mut collections = collections.iter();
        let collection = collections.next()?.clone();
        return collections
            .next()
            .is_none()
            .then_some(RelationTerm::Length(collection));
    }
    let value = eval_operand(state, operand);
    if let Some(collection) = current_length_collection(state, &value) {
        return Some(RelationTerm::Length(collection));
    }
    if let (Some(path), Some(place)) = (operand.projection_path.as_ref(), operand.operand.place()) {
        return Some(RelationTerm::Projection {
            base: canonical_storage(place),
            path: path.clone(),
        });
    }
    Some(RelationTerm::Place(unique_semantic_value(&value)?))
}

fn normalized_relation_pending(
    state: &ProgramState,
    op: BinOp,
    left: &OperandModel,
    right: &OperandModel,
) -> Vec<(ProofBinding, bool, Vec<PlaceKey>)> {
    let (Some(left), Some(right)) = (
        relation_term_for_operand(state, left),
        relation_term_for_operand(state, right),
    ) else {
        return Vec::new();
    };
    let relations = match op {
        BinOp::Lt => vec![
            (RelationKind::Lt(left.clone(), right.clone()), true),
            (RelationKind::Le(right, left), false),
        ],
        BinOp::Le => vec![
            (RelationKind::Le(left.clone(), right.clone()), true),
            (RelationKind::Lt(right, left), false),
        ],
        BinOp::Gt => vec![
            (RelationKind::Lt(right.clone(), left.clone()), true),
            (RelationKind::Le(left, right), false),
        ],
        BinOp::Ge => vec![
            (RelationKind::Le(right.clone(), left.clone()), true),
            (RelationKind::Lt(left, right), false),
        ],
        BinOp::Eq => vec![(RelationKind::Eq(left, right), true)],
        BinOp::Ne => vec![(RelationKind::Eq(left, right), false)],
        _ => Vec::new(),
    };
    relations
        .into_iter()
        .map(|(kind, polarity)| {
            let relation = RelationFact::new(kind);
            let places = relation_storage_places(&relation);
            (ProofBinding::Relation(relation), polarity, places)
        })
        .collect()
}

fn assignment_relation(state: &ProgramState, assignment: &AssignmentModel) -> Option<RelationFact> {
    let kind = match &assignment.value {
        ValueModel::Operand(operand) => {
            let operand_term = relation_term_for_operand(state, operand)?;
            let result = match &operand_term {
                RelationTerm::Projection { .. } => {
                    RelationTerm::Place(unique_semantic_value(&eval_operand(state, operand))?)
                }
                _ => RelationTerm::Place(assignment.destination.clone()),
            };
            RelationKind::Eq(result, operand_term)
        }
        ValueModel::Unsize { operand, .. } => {
            // Unsize changes metadata, not the identity of the referenced
            // object. Record that local fact without inferring provenance.
            let source = operand
                .operand
                .place()
                .cloned()
                .map(RelationTerm::Place)
                .or_else(|| relation_term_for_operand(state, operand))?;
            RelationKind::SameObject(RelationTerm::Place(assignment.destination.clone()), source)
        }
        ValueModel::IdentityPointerView { .. }
        | ValueModel::AddressPointerView { .. }
        | ValueModel::PointerCast { .. }
        | ValueModel::GenericPointerCast { .. }
        | ValueModel::BehaviorPointer { .. } => return None,
        ValueModel::RefOrRaw(operand) | ValueModel::LocalRefOrRaw(operand) => RelationKind::Eq(
            RelationTerm::Place(assignment.destination.clone()),
            relation_term_for_operand(state, operand)?,
        ),
        ValueModel::Length(collection) => {
            let collection = unique_semantic_value(&eval_operand(state, collection))?;
            RelationKind::Eq(
                RelationTerm::Place(assignment.destination.clone()),
                RelationTerm::Length(collection),
            )
        }
        ValueModel::Compare {
            op: BinOp::Add,
            left,
            right,
        } => {
            let left = relation_term_for_operand(state, left)?;
            let right = relation_term_for_operand(state, right)?;
            let (base, addend) = match (left, right) {
                (base, RelationTerm::Constant(addend)) | (RelationTerm::Constant(addend), base) => {
                    (base, addend)
                }
                _ => return None,
            };
            if addend != 0
                && !(addend == 1
                    && current_relation_proof(state).proves_increment_without_overflow(&base))
            {
                return None;
            }
            RelationKind::AddConst {
                result: RelationTerm::Place(assignment.destination.clone()),
                base,
                addend,
            }
        }
        ValueModel::Compare {
            op: BinOp::Sub,
            left,
            right,
        } => {
            let upper = relation_term_for_operand(state, left)?;
            let lower = relation_term_for_operand(state, right)?;
            if !current_relation_proof(state).proves_le(&lower, &upper) {
                return None;
            }
            RelationKind::Sub {
                result: RelationTerm::Place(assignment.destination.clone()),
                upper,
                lower,
            }
        }
        ValueModel::CheckedBinary {
            op: BinOp::Add,
            left,
            right,
        } => {
            let left = relation_term_for_operand(state, left)?;
            let right = relation_term_for_operand(state, right)?;
            let (base, addend) = match (left, right) {
                (base, RelationTerm::Constant(addend)) | (RelationTerm::Constant(addend), base) => {
                    (base, addend)
                }
                _ => return None,
            };
            if addend != 0
                && !(addend == 1
                    && current_relation_proof(state).proves_increment_without_overflow(&base))
            {
                return None;
            }
            RelationKind::AddConst {
                result: RelationTerm::Projection {
                    base: assignment.destination.clone(),
                    path: vec![ProjectionKey::Field(0)],
                },
                base,
                addend,
            }
        }
        _ => return None,
    };
    Some(RelationFact::new(kind))
}

fn aggregate_relation_terms(
    state: &ProgramState,
    operand: &OperandModel,
) -> BTreeSet<RelationTerm> {
    match &operand.operand {
        CallOperand::ArrayLengths(collections) => collections
            .iter()
            .cloned()
            .map(RelationTerm::Length)
            .collect(),
        _ => relation_term_for_operand(state, operand)
            .into_iter()
            .collect(),
    }
}

fn aggregate_proofs(state: &ProgramState, assignment: &AssignmentModel) -> Vec<ProofBinding> {
    let ValueModel::Aggregate {
        operands, variant, ..
    } = &assignment.value
    else {
        return Vec::new();
    };
    let fields = operands
        .iter()
        .enumerate()
        .flat_map(|(field, operand)| {
            let mut path = Vec::new();
            if let Some(discriminant) = variant {
                path.push(ProjectionKey::Downcast(*discriminant as u32));
            }
            path.push(ProjectionKey::Field(field as u32));
            aggregate_relation_terms(state, operand)
                .into_iter()
                .map(move |payload| {
                    (
                        path.clone(),
                        ProofBinding::Relation(RelationFact::new(RelationKind::Eq(
                            RelationTerm::Projection {
                                base: canonical_storage(&assignment.destination),
                                path: path.clone(),
                            },
                            payload,
                        ))),
                    )
                })
        })
        .collect::<Vec<_>>();
    let mut proofs = fields
        .iter()
        .map(|(_, proof)| proof.clone())
        .collect::<Vec<_>>();
    if assignment.destination == PlaceKey::new("_0") && fields.len() == 1 && variant.is_some() {
        let (payload_path, _) = &fields[0];
        proofs.push(ProofBinding::Relation(RelationFact::new(RelationKind::Eq(
            RelationTerm::Return,
            RelationTerm::Projection {
                base: PlaceKey::new("_0"),
                path: payload_path.clone(),
            },
        ))));
        proofs.push(ProofBinding::ReturnVariant(ReturnVariantFact {
            destination: assignment.destination.clone(),
            discriminant: variant.expect("variant was checked"),
            payload_path: payload_path.clone(),
        }));
    }
    proofs
}

fn normalized_compare_pending(
    state: &ProgramState,
    op: BinOp,
    left: &OperandModel,
    right: &OperandModel,
) -> Vec<(ProofBinding, bool, Vec<PlaceKey>)> {
    let left_value = eval_operand(state, left);
    let right_value = eval_operand(state, right);

    if op == BinOp::Gt {
        if let (Some(end), Some(collection)) = (
            current_checked_end(state, &left_value),
            current_length_collection(state, &right_value),
        ) {
            let overflow_proof =
                ValidationBinding::range(end.offset.clone(), None, end.width, Some(end.definition));
            if state.has_current_validation(&overflow_proof, overflow_proof.storage_places()) {
                let binding =
                    ValidationBinding::range(end.offset, Some(collection), end.width, None);
                let places = binding.storage_places();
                return vec![(binding.into(), false, places)];
            }
        }
        if let (Some(offset), Some(limit)) = (
            unique_semantic_value(&left_value),
            current_range_limit(state, &right_value),
        ) {
            if !left_value.havoced {
                let binding =
                    ValidationBinding::range(offset, Some(limit.collection), limit.width, None);
                let places = binding.storage_places();
                return vec![(binding.into(), false, places)];
            }
        }
    }

    let in_bounds_polarity = match op {
        BinOp::Ge => Some(false),
        BinOp::Lt => Some(true),
        _ => None,
    };
    if let Some(polarity) = in_bounds_polarity {
        if let (Some(subject), Some(collection)) = (
            unique_semantic_value(&left_value),
            current_length_collection(state, &right_value),
        ) {
            if !left_value.havoced {
                let binding = ValidationBinding {
                    predicate: Predicate::InBounds,
                    access_width: None,
                    proof_definition: None,
                    subject,
                    collection: Some(collection),
                };
                let places = binding.storage_places();
                return vec![(binding.into(), polarity, places)];
            }
        }
    }

    let polarity = match op {
        BinOp::Eq => false,
        BinOp::Ne => true,
        _ => return Vec::new(),
    };
    let length_value = match (&left.operand, &right.operand) {
        (_, CallOperand::Constant(0)) => &left_value,
        (CallOperand::Constant(0), _) => &right_value,
        _ => return Vec::new(),
    };
    let Some(collection) = current_length_collection(state, length_value) else {
        return Vec::new();
    };
    let binding = ValidationBinding {
        predicate: Predicate::NonEmpty,
        access_width: None,
        proof_definition: None,
        subject: collection,
        collection: None,
    };
    let places = binding.storage_places();
    vec![(binding.into(), polarity, places)]
}

fn eval_operand(state: &ProgramState, operand: &OperandModel) -> ValueFacts {
    let mut result = match &operand.operand {
        CallOperand::Place(place) => {
            let mut value = operand
                .field_origin
                .as_ref()
                .map(|field| field_read_value(state, field))
                .unwrap_or_else(|| state_value(state, place));
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
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        },
        CallOperand::ArrayLengths(_) | CallOperand::Unknown => ValueFacts {
            may_be_non_length: true,
            semantic_unknown: true,
            behavior_controller_unknown: true,
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        },
    };
    if let Some(field) = &operand.field_origin {
        let node = public_field_node(state, field);
        let persistent = matches!(
            &node,
            EndpointNode::PublicField {
                persistent: true,
                ..
            }
        );
        let field_source_is_live = result.may_hold_entry_public_field
            || (!persistent && !result.endpoint_sources.is_empty());
        if field.path_public && persistent && field_source_is_live {
            result.value.origins.insert(AbstractOrigin::PublicField {
                def_path: field.def_path.clone(),
                span: field.span.clone(),
            });
            result.influence_seeds.insert(
                node.clone(),
                InfluenceToken::from_type_class(
                    InfluenceOrigin::PersistentPublicField,
                    InfluenceController::PersistentPublicField {
                        access_owner: field.access_owner.clone(),
                        field: field.def_path.clone(),
                    },
                    field.type_class,
                ),
            );
        }
        if field.path_public && field_source_is_live {
            Arc::make_mut(&mut result.endpoint_sources).insert(node);
        }
    }
    // These descriptors identify storage slots in the dataflow map; they are
    // not value provenance and must not be copied into an assignment target.
    result.stored_public_field_slots.clear();
    result
}

fn entry_formal_index(arg_count: usize, base: &PlaceKey) -> Option<u32> {
    if canonical_storage(base) != *base {
        return None;
    }
    let index = base.0.strip_prefix('_')?.parse::<u32>().ok()?;
    (index > 0 && index as usize <= arg_count).then_some(index)
}

fn entry_field_projection_value(
    base: &ValueFacts,
    field: &FieldAccessModel,
    formal_index: u32,
) -> ValueFacts {
    if !field.path_public {
        return ValueFacts {
            exact_roots: BTreeSet::from([field.full_place.clone()]),
            semantic_unknown: true,
            may_be_non_length: true,
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            behavior_control: BehaviorControlProvenance::InternalOnly,
            storage_provenance: StorageProvenance::EntryFormal {
                owner: field.access_owner.clone(),
                index: formal_index,
            },
            ..ValueFacts::default()
        };
    }

    let mut value = base.clone();
    value.dependencies = BTreeSet::from([field.full_place.clone()]);
    value.value_flow = BTreeSet::from([field.full_place.clone()]);
    value.exact_roots = BTreeSet::from([field.full_place.clone()]);
    value.semantic_unknown = false;
    value.may_hold_entry_public_field = true;
    if let Some(path) = &field.projection_path {
        value.stored_public_field_slots = BTreeSet::from([PublicFieldStorageSlot {
            full_place: field.full_place.clone(),
            base: field.base.clone(),
            path: path.clone(),
        }]);
    }
    value
}

fn field_read_value(state: &ProgramState, field: &FieldAccessModel) -> ValueFacts {
    if let Some(value) = state.may_values().get(&field.full_place) {
        return value.clone();
    }
    if field.path_public {
        return state_value(state, &field.base);
    }
    let storage = state
        .may_values()
        .get(&field.base)
        .map(|value| value.storage_provenance.clone())
        .unwrap_or(StorageProvenance::DefinitelyLocal);
    ValueFacts {
        exact_roots: BTreeSet::from([field.full_place.clone()]),
        semantic_unknown: true,
        may_be_non_length: true,
        utf8_relation_unknown: true,
        checked_end_unknown: true,
        range_limit_unknown: true,
        known_length_unknown: true,
        behavior_control: BehaviorControlProvenance::InternalOnly,
        storage_provenance: storage,
        ..ValueFacts::default()
    }
}

fn public_field_node(state: &ProgramState, field: &FieldAccessModel) -> EndpointNode {
    let storage = state
        .may_values()
        .get(&field.base)
        .map(|value| value.storage_provenance.clone())
        .unwrap_or(StorageProvenance::MaybeExternal);
    let persistent = field.path_public
        && field.adt_kind != PublicFieldAdtKind::Enum
        && storage.may_be_externally_controlled();
    EndpointNode::PublicField {
        access_owner: field.access_owner.clone(),
        def_path: field.def_path.clone(),
        access_path: field.full_place.clone(),
        base: field.base.clone(),
        storage_versions: current_storage_versions(state, &field.base),
        storage,
        adt_kind: field.adt_kind,
        persistent,
        span: field.span.clone(),
    }
}

fn endpoint_node_owner(node: &EndpointNode) -> Option<&FunctionKey> {
    match node {
        EndpointNode::Formal { owner, .. }
        | EndpointNode::Capture { owner, .. }
        | EndpointNode::ExternalBehavior { owner, .. }
        | EndpointNode::InternalDerivation { owner, .. }
        | EndpointNode::GenericCapability { owner, .. }
        | EndpointNode::UnsafeOperand { owner, .. }
        | EndpointNode::UnsafeResult { owner, .. }
        | EndpointNode::Return { owner }
        | EndpointNode::Out { owner, .. }
        | EndpointNode::Exposure { owner, .. } => Some(owner),
        EndpointNode::CallResult { callsite, .. } => Some(&callsite.caller),
        EndpointNode::PublicField { access_owner, .. } => Some(access_owner),
    }
}

fn endpoint_relations_for_owner(
    owner: &FunctionKey,
    graph: &EndpointFlowGraph,
    excluded: &BTreeSet<EndpointEdgeKind>,
) -> BTreeSet<EndpointRelation> {
    let routes = graph.route_index_excluding(excluded);
    let terminals = graph
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node,
                EndpointNode::Return { owner: terminal_owner }
                    | EndpointNode::Out { owner: terminal_owner, .. }
                    if terminal_owner == owner
            ) || matches!(
                node,
                EndpointNode::UnsafeOperand { .. } | EndpointNode::Exposure { .. }
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut summary = BTreeSet::new();
    for terminal in terminals {
        let is_output = matches!(
            terminal,
            EndpointNode::Return { .. } | EndpointNode::Out { .. }
        );
        for input in routes.routes_to(&terminal).reaching_nodes() {
            if (is_output
                && !matches!(
                    input,
                    EndpointNode::Return { .. } | EndpointNode::Out { .. }
                ))
                || (!is_output && is_context_summary_source(input, owner))
            {
                summary.insert(EndpointRelation {
                    input: input.clone(),
                    terminal: terminal.clone(),
                });
            }
        }
    }
    summary
}

/// Sources that a function summary may carry to an unsafe operand or
/// exposure.  The structural sources deliberately do not have to be owned by
/// `owner`: a prior exact CallSummary can import a callee-local derivation
/// into this caller, and a further wrapper must be able to export it again.
/// Terminal nodes are excluded so an unrelated Return/Out/unsafe endpoint is
/// never used as a synthetic source for another terminal.
fn is_context_summary_source(input: &EndpointNode, owner: &FunctionKey) -> bool {
    match input {
        EndpointNode::Formal {
            owner: formal_owner,
            ..
        } => formal_owner == owner,
        EndpointNode::CallResult { callsite, .. } => callsite.caller == *owner,
        EndpointNode::Capture { .. }
        | EndpointNode::PublicField { .. }
        | EndpointNode::ExternalBehavior { .. }
        | EndpointNode::InternalDerivation { .. }
        | EndpointNode::GenericCapability { .. }
        | EndpointNode::UnsafeResult { .. } => true,
        EndpointNode::UnsafeOperand { .. }
        | EndpointNode::Return { .. }
        | EndpointNode::Out { .. }
        | EndpointNode::Exposure { .. } => false,
    }
}

fn join_value_facts(values: impl IntoIterator<Item = ValueFacts>) -> ValueFacts {
    let mut values = values.into_iter();
    let mut result = values.next().unwrap_or_default();
    for value in values {
        result.join_may(&value);
    }
    result
}

fn behavior_controllers(value: &ValueFacts) -> BTreeSet<BehaviorController> {
    let mut controllers = value.behavior_controllers.clone();
    for source in value.endpoint_sources.iter() {
        match source {
            EndpointNode::Formal { owner, index } => {
                controllers.insert(BehaviorController::PublicFormal {
                    owner: owner.clone(),
                    index: *index,
                });
            }
            EndpointNode::PublicField {
                access_owner,
                def_path,
                access_path,
                base,
                storage,
                storage_versions,
                adt_kind,
                persistent: true,
                span,
                ..
            } => {
                controllers.insert(BehaviorController::PublicField {
                    access_owner: access_owner.clone(),
                    def_path: def_path.clone(),
                    access_path: access_path.clone(),
                    base: base.clone(),
                    storage: storage.clone(),
                    storage_versions: storage_versions.clone(),
                    adt_kind: *adt_kind,
                    persistent: true,
                    span: span.clone(),
                });
            }
            EndpointNode::PublicField {
                persistent: false, ..
            } => {}
            EndpointNode::ExternalBehavior { controller, .. } => {
                controllers.insert(controller.clone());
            }
            EndpointNode::Capture { .. }
            | EndpointNode::InternalDerivation { .. }
            | EndpointNode::GenericCapability { .. }
            | EndpointNode::UnsafeOperand { .. }
            | EndpointNode::UnsafeResult { .. }
            | EndpointNode::Return { .. }
            | EndpointNode::Out { .. }
            | EndpointNode::CallResult { .. }
            | EndpointNode::Exposure { .. } => {}
        }
    }
    if (controllers.is_empty() && value.behavior_control != BehaviorControlProvenance::InternalOnly)
        || value.behavior_controller_unknown
        || value.havoced
    {
        controllers.insert(BehaviorController::Unknown);
    }
    controllers
}

fn behavior_pointer_controller(function: FunctionKey, foreign: bool) -> BehaviorController {
    if foreign {
        BehaviorController::FfiHost
    } else {
        BehaviorController::InternalFunction { function }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObservedInternalBehaviorTarget {
    function: FunctionKey,
    controller: BehaviorController,
    formal_offset: u32,
}

fn observed_internal_behavior_target(
    value: &ValueFacts,
    local_target_kind: impl Fn(&FunctionKey) -> Option<bool>,
) -> Option<ObservedInternalBehaviorTarget> {
    if value.behavior_controller_unknown || value.havoced {
        return None;
    }
    let mut controllers = behavior_controllers(value).into_iter();
    let controller = controllers.next()?;
    if controllers.next().is_some() {
        return None;
    }
    let (function, expected_closure, formal_offset) = match &controller {
        BehaviorController::InternalFunction { function } => (function.clone(), false, 1),
        BehaviorController::InternalClosure { function } => (function.clone(), true, 2),
        BehaviorController::OpenTrait { .. }
        | BehaviorController::PublicFormal { .. }
        | BehaviorController::PublicField { .. }
        | BehaviorController::FfiHost
        | BehaviorController::Unknown => return None,
    };
    (local_target_kind(&function) == Some(expected_closure)).then_some(
        ObservedInternalBehaviorTarget {
            function,
            controller,
            formal_offset,
        },
    )
}

fn indirect_behavior_result_node(
    point: &ProgramPoint,
    controller: &BehaviorController,
) -> EndpointNode {
    EndpointNode::ExternalBehavior {
        owner: point.function.clone(),
        kind: ExternalSourceKind::FunctionPointer,
        controller: controller.clone(),
        token: format!("fn-pointer-return:{}", point.span.token()),
        span: point.span.clone(),
    }
}

fn ffi_callback_input_endpoint(owner: &FunctionKey, index: u32, span: &StableSpan) -> EndpointNode {
    EndpointNode::ExternalBehavior {
        owner: owner.clone(),
        kind: ExternalSourceKind::Host,
        controller: BehaviorController::FfiHost,
        token: format!("ffi-callback-input:{}:{index}", owner.0),
        span: span.clone(),
    }
}

fn append_value_endpoint_edges(graph: &mut EndpointFlowGraph, edges: &BTreeSet<EndpointEdge>) {
    for edge in edges.iter().cloned() {
        graph.insert_edge(edge);
    }
}

fn observed_internal_behavior_edges<'a>(
    caller: &FunctionKey,
    point: &ProgramPoint,
    target: &ObservedInternalBehaviorTarget,
    actual_values: impl IntoIterator<Item = &'a ValueFacts>,
) -> (BTreeSet<EndpointEdge>, CandidateCallEdge) {
    let mut edges = BTreeSet::new();
    for (index, value) in actual_values.into_iter().enumerate() {
        edges.extend(value.endpoint_edges.iter().cloned());
        let formal = EndpointNode::Formal {
            owner: target.function.clone(),
            index: target.formal_offset + index as u32,
        };
        edges.extend(
            value
                .endpoint_sources
                .iter()
                .cloned()
                .map(|source| EndpointEdge {
                    from: source,
                    to: formal.clone(),
                    kind: EndpointEdgeKind::ActualToFormal,
                    point: Some(point.clone()),
                }),
        );
    }
    edges.insert(EndpointEdge {
        from: EndpointNode::Return {
            owner: target.function.clone(),
        },
        to: indirect_behavior_result_node(point, &target.controller),
        kind: EndpointEdgeKind::ReturnToDestination,
        point: Some(point.clone()),
    });
    (
        edges,
        CandidateCallEdge {
            caller: caller.clone(),
            callee: target.function.clone(),
            point: point.clone(),
            kind: CandidateCallKind::Callback,
            generic_substitution: GenericCallSubstitution::default(),
        },
    )
}

fn eval_value_model(
    state: &ProgramState,
    model: &ValueModel,
    definition: StaticWriteToken,
) -> ValueFacts {
    match model {
        ValueModel::Operand(operand) => eval_operand(state, operand),
        ValueModel::Unsize {
            operand,
            known_length,
        } => {
            let mut result = eval_operand(state, operand);
            clear_whole_initialization_target(&mut result);
            result.known_lengths.clear();
            result.known_length_unknown = known_length.is_none();
            if let Some(length) = known_length {
                result.known_lengths.insert(*length);
            }
            result
        }
        ValueModel::RefOrRaw(place) => {
            let mut result = eval_operand(state, place);
            clear_whole_initialization_target(&mut result);
            result
        }
        ValueModel::LocalRefOrRaw(place) => {
            let mut result = eval_operand(state, place);
            clear_whole_initialization_target(&mut result);
            result.storage_provenance = StorageProvenance::DefinitelyLocal;
            result
        }
        ValueModel::IdentityPointerView { operand, .. } => eval_operand(state, operand),
        ValueModel::AddressPointerView { operand, .. } => {
            let mut result = eval_operand(state, operand);
            // An address view can discard pointer metadata, but the compiler
            // evidence used to select this model proves that the address bits
            // and their external controller are preserved.  Keep influence
            // exact while independently invalidating metadata-sensitive
            // initialization facts.
            clear_pointer_metadata_identity(&mut result);
            // A safe reference does not itself grant external control over
            // the derived address/extent. Until that relation is modeled,
            // retain the former auditable Unknown. Raw-pointer/FFI seeds
            // already carry ADDRESS and remain exact through this view.
            let field_address_source = result
                .endpoint_sources
                .iter()
                .any(|source| matches!(source, EndpointNode::PublicField { .. }));
            if !field_address_source
                && !result
                    .influence_seeds
                    .has_capability(InfluenceCapabilities::ADDRESS)
            {
                result
                    .influence_seeds
                    .mark_all_incomplete(UnestablishedReason::OpaqueTransform);
            }
            result
        }
        ValueModel::PointerCast { operand, .. } => {
            let mut result = eval_operand(state, operand);
            clear_pointer_metadata_identity(&mut result);
            result
                .influence_seeds
                .mark_all_incomplete(UnestablishedReason::OpaqueTransform);
            result
        }
        ValueModel::GenericPointerCast { operand, .. } => {
            let mut result = eval_operand(state, operand);
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.generic_capability = true;
            clear_whole_initialization_target(&mut result);
            result
                .influence_seeds
                .mark_all_incomplete(UnestablishedReason::OpaqueTransform);
            result
        }
        ValueModel::Discriminant(operand) => {
            let mut result = eval_operand(state, operand);
            result.exact_roots.clear();
            result.semantic_unknown = true;
            clear_whole_initialization_target(&mut result);
            result
        }
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
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            result.may_be_non_length = false;
            result.havoced = false;
            clear_whole_initialization_target(&mut result);
            result
        }
        ValueModel::Compare { left, right, .. } | ValueModel::CheckedBinary { left, right, .. } => {
            let mut result =
                join_value_facts([eval_operand(state, left), eval_operand(state, right)]);
            result.value_flow.clear();
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.len_of.clear();
            result.length_atoms.clear();
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            result.may_be_non_length = true;
            clear_whole_initialization_target(&mut result);
            result
        }
        ValueModel::Aggregate {
            operands,
            range_end,
            closure,
            ..
        } => {
            let mut result =
                join_value_facts(operands.iter().map(|operand| eval_operand(state, operand)));
            if let Some(range_end) = range_end {
                result.range_end.insert(range_end.clone());
            }
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            clear_whole_initialization_target(&mut result);
            if let Some(function) = closure {
                // Captured data flows through the explicit Capture nodes built
                // during final replay. It is not itself a possible callable
                // controller for the freshly-created closure value.
                result.endpoint_sources = Arc::default();
                result.endpoint_edges = Arc::default();
                result.behavior_controllers.clear();
                result
                    .behavior_controllers
                    .insert(BehaviorController::InternalClosure {
                        function: function.clone(),
                    });
                result.behavior_controller_unknown = false;
                result.behavior_control = BehaviorControlProvenance::InternalOnly;
            }
            result
        }
        ValueModel::BehaviorPointer { operand, seed } => {
            let mut result = eval_operand(state, operand);
            if let Some(controller) = seed {
                result.behavior_controllers.clear();
                result.behavior_controllers.insert(controller.clone());
                result.behavior_controller_unknown = false;
                result.behavior_control = match controller {
                    BehaviorController::InternalFunction { .. }
                    | BehaviorController::InternalClosure { .. } => {
                        BehaviorControlProvenance::InternalOnly
                    }
                    BehaviorController::OpenTrait { .. }
                    | BehaviorController::PublicFormal { .. }
                    | BehaviorController::PublicField { .. }
                    | BehaviorController::FfiHost
                    | BehaviorController::Unknown => BehaviorControlProvenance::MaybeExternal,
                };
            }
            result
        }
        ValueModel::Unknown => ValueFacts {
            may_be_non_length: true,
            semantic_unknown: true,
            whole_init_target_unknown: true,
            behavior_controller_unknown: true,
            behavior_control: BehaviorControlProvenance::MaybeExternal,
            storage_provenance: StorageProvenance::MaybeExternal,
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        },
    }
}

fn clear_pointer_metadata_identity(value: &mut ValueFacts) {
    value.exact_roots.clear();
    value.semantic_unknown = true;
    value.len_of.clear();
    value.length_atoms.clear();
    value.length_relation_unknown = true;
    value.utf8_atoms.clear();
    value.utf8_relation_unknown = true;
    value.checked_end_atoms.clear();
    value.checked_end_unknown = true;
    value.range_limit_atoms.clear();
    value.range_limit_unknown = true;
    value.known_lengths.clear();
    value.known_length_unknown = true;
    value.range_end.clear();
    clear_whole_initialization_target(value);
}

fn clear_whole_initialization_target(value: &mut ValueFacts) {
    // Exact pointer identity is gone, but retaining the may-target set is what
    // lets a later inexact write conservatively havoc every known destination.
    value.whole_init_target_unknown = true;
}

fn attach_generic_capability(
    value: &mut ValueFacts,
    evidence: &GenericCapabilityEvidence,
    point: &ProgramPoint,
    legacy_origin: bool,
) {
    if legacy_origin {
        value
            .value
            .origins
            .insert(AbstractOrigin::GenericCapability {
                token: evidence.stable_identity(),
                span: point.span.clone(),
            });
        value.generic_capability = true;
    }
    Arc::make_mut(&mut value.endpoint_sources).insert(EndpointNode::GenericCapability {
        owner: point.function.clone(),
        evidence: evidence.clone(),
        span: point.span.clone(),
    });
}

fn external_influence_origin(kind: ExternalSourceKind) -> InfluenceOrigin {
    match kind {
        ExternalSourceKind::FfiReturn => InfluenceOrigin::FfiReturn,
        ExternalSourceKind::FfiOut => InfluenceOrigin::FfiOut,
        ExternalSourceKind::OpenTrait | ExternalSourceKind::Iterator => {
            InfluenceOrigin::OpenTraitReturn
        }
        ExternalSourceKind::Callback => InfluenceOrigin::PublicCallbackReturn,
        ExternalSourceKind::FunctionPointer => InfluenceOrigin::PublicFunctionPointerReturn,
        ExternalSourceKind::Host => InfluenceOrigin::FfiCallbackInput,
    }
}

fn influence_controller_from_behavior(controller: &BehaviorController) -> InfluenceController {
    match controller {
        BehaviorController::FfiHost => InfluenceController::FfiHost,
        BehaviorController::PublicFormal { owner, index } => InfluenceController::PublicFormal {
            owner: owner.clone(),
            index: *index,
        },
        BehaviorController::PublicField {
            access_owner,
            def_path,
            ..
        } => InfluenceController::PersistentPublicField {
            access_owner: access_owner.clone(),
            field: def_path.clone(),
        },
        BehaviorController::OpenTrait {
            trait_id,
            openness: BehaviorOpenness::ProvenOpen,
            resolved_target: None,
            receiver,
        } => InfluenceController::OpenTrait {
            trait_id: trait_id.clone(),
            receiver: receiver.clone(),
        },
        BehaviorController::OpenTrait {
            openness: BehaviorOpenness::ProvenClosed,
            ..
        } => InfluenceController::Internal,
        BehaviorController::OpenTrait { .. } => InfluenceController::Unknown,
        BehaviorController::InternalFunction { .. }
        | BehaviorController::InternalClosure { .. } => InfluenceController::Internal,
        BehaviorController::Unknown => InfluenceController::Unknown,
    }
}

fn insert_external_influence_seed(
    value: &mut ValueFacts,
    endpoint: EndpointNode,
    kind: ExternalSourceKind,
    controller: &BehaviorController,
    type_class: InfluenceTypeClass,
) {
    value.influence_seeds.insert(
        endpoint,
        InfluenceToken::from_type_class(
            external_influence_origin(kind),
            influence_controller_from_behavior(controller),
            type_class,
        ),
    );
}

fn eval_registry_value(
    state: &ProgramState,
    model: &RegistryValueModel,
    point: &ProgramPoint,
) -> ValueFacts {
    let definition = StaticWriteToken::new(point.block, point.statement);
    match model {
        RegistryValueModel::RangeIntoIterator(range) => eval_operand(state, range),
        RegistryValueModel::ExternalBehavior {
            kind,
            controller,
            token,
            type_class,
        } => {
            let endpoint = EndpointNode::ExternalBehavior {
                owner: point.function.clone(),
                kind: *kind,
                controller: controller.clone(),
                token: token.clone(),
                span: point.span.clone(),
            };
            let mut result = ValueFacts {
                endpoint_sources: Arc::new(BTreeSet::from([endpoint.clone()])),
                may_be_non_length: true,
                semantic_unknown: true,
                utf8_relation_unknown: true,
                checked_end_unknown: true,
                range_limit_unknown: true,
                known_length_unknown: true,
                ..ValueFacts::default()
            };
            insert_external_influence_seed(&mut result, endpoint, *kind, controller, *type_class);
            result
        }
        RegistryValueModel::PrivateAggregateTransform => ValueFacts {
            private_aggregate_transform_unknown: true,
            may_be_non_length: true,
            semantic_unknown: true,
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::Empty
        | RegistryValueModel::IndirectBehavior { .. }
        | RegistryValueModel::SliceIterator(_)
        | RegistryValueModel::SlicePosition(_)
        | RegistryValueModel::RangeIteratorNext(_) => ValueFacts {
            may_be_non_length: true,
            semantic_unknown: true,
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::MaybeUninit(initialized) => ValueFacts {
            maybe_uninit_initialized: BTreeSet::from([*initialized]),
            initialization_witnesses: initialized
                .then(|| {
                    BTreeSet::from([InitializationWitness {
                        kind: InitializationWitnessKind::MaybeUninitNew,
                        point: point.clone(),
                    }])
                })
                .unwrap_or_default(),
            may_be_non_length: true,
            semantic_unknown: true,
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::TypedInitializationPointer {
            receiver,
            generic_view,
        } => {
            let mut result = eval_operand(state, receiver);
            // `MaybeUninit::{as_ptr,as_mut_ptr}` is a compiler-identified
            // typed view into storage whose bytes may legally be invalid for
            // `T`.  Unlike an ordinary reference-preserving edge, this view
            // therefore derives address/validity influence explicitly.
            result.influence_seeds.derive_all_exact(
                super::influence::InfluenceCapabilities::ADDRESS
                    | super::influence::InfluenceCapabilities::VALIDITY_BITS,
            );
            let target = unique_semantic_value(&result);
            clear_pointer_metadata_identity(&mut result);
            result.whole_init_targets.clear();
            result.whole_init_target_tracked = true;
            result.whole_init_target_unknown = target.is_none();
            result.whole_init_targets.extend(target);
            if let Some(evidence) = generic_view {
                attach_generic_capability(&mut result, evidence, point, false);
            }
            result
        }
        RegistryValueModel::AssumeInit { source, .. } => {
            let mut result = eval_operand(state, source);
            result.maybe_uninit_initialized.clear();
            result
        }
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
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            result.may_be_non_length = false;
            result.havoced = false;
            result
        }
        RegistryValueModel::SelectedOpenBehavior {
            token,
            controller,
            inputs,
            type_class,
        } => {
            let mut result =
                join_value_facts(inputs.iter().map(|input| eval_operand(state, input)));
            result
                .value
                .origins
                .insert(AbstractOrigin::OpenBehaviorOutput {
                    token: token.clone(),
                    span: point.span.clone(),
                });
            result.open_behavior = true;
            let endpoint = EndpointNode::ExternalBehavior {
                owner: point.function.clone(),
                kind: ExternalSourceKind::OpenTrait,
                controller: controller.clone(),
                token: token.clone(),
                span: point.span.clone(),
            };
            Arc::make_mut(&mut result.endpoint_sources).insert(endpoint.clone());
            insert_external_influence_seed(
                &mut result,
                endpoint,
                ExternalSourceKind::OpenTrait,
                controller,
                *type_class,
            );
            result.may_be_non_length = true;
            result.semantic_unknown = true;
            result.utf8_relation_unknown = true;
            result.checked_end_unknown = true;
            result.range_limit_unknown = true;
            result.known_length_unknown = true;
            result
        }
        RegistryValueModel::GenericCapability { inputs, evidence } => {
            // A capability source refines the provenance of the actual value;
            // it must not replace that value.  Keeping the actual endpoint
            // sources is what lets an unproven P4 classification fall back to
            // the lower-ranked P1/P6 causal source instead of disappearing.
            let mut result =
                join_value_facts(inputs.iter().map(|input| eval_operand(state, input)));
            result.may_be_non_length = true;
            result.semantic_unknown = true;
            result.utf8_relation_unknown = true;
            result.checked_end_unknown = true;
            result.range_limit_unknown = true;
            result.known_length_unknown = true;
            attach_generic_capability(&mut result, evidence, point, true);
            result
        }
        RegistryValueModel::SaturatingSub(operands) => {
            let mut result =
                join_value_facts(operands.iter().map(|operand| eval_operand(state, operand)));
            let length = operands.first().map(|operand| eval_operand(state, operand));
            let width = operands.get(1).map(|operand| eval_operand(state, operand));
            let mut relations = BTreeSet::new();
            let mut relation_unknown = true;
            if let (Some(length), Some(width)) = (length.as_ref(), width.as_ref()) {
                let (widths, width_unknown) = constant_candidates(width);
                relation_unknown = width_unknown
                    || length.length_relation_unknown
                    || length.may_be_non_length
                    || length.havoced
                    || length.length_atoms.is_empty()
                    || length.known_lengths.is_empty()
                    || length.known_length_unknown;
                for length_atom in &length.length_atoms {
                    let current = !length_atom.invalidated
                        && length_atom.captured_versions
                            == current_storage_versions(state, &length_atom.collection);
                    relation_unknown |= !current;
                    if !current {
                        continue;
                    }
                    for width in &widths {
                        let has_fitting_length = length
                            .known_lengths
                            .iter()
                            .any(|known_length| known_length >= width);
                        relation_unknown |= length
                            .known_lengths
                            .iter()
                            .any(|known_length| known_length < width);
                        if has_fitting_length {
                            relations.insert(RangeLimitAtom {
                                captured_versions: current_storage_versions(
                                    state,
                                    &length_atom.collection,
                                ),
                                collection: length_atom.collection.clone(),
                                width: *width,
                                definition,
                                invalidated: false,
                            });
                        }
                    }
                }
            }
            result.value_flow.clear();
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.len_of.clear();
            result.length_atoms.clear();
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_atoms = relations;
            result.range_limit_unknown = relation_unknown;
            result.may_be_non_length = true;
            result
        }
        RegistryValueModel::CheckedAdd { offset, width } => {
            let offset = eval_operand(state, offset);
            let width_value = eval_operand(state, width);
            let mut result = join_value_facts([offset.clone(), width_value.clone()]);
            let (widths, width_unknown) = constant_candidates(&width_value);
            let offsets = semantic_place_candidates(&offset);
            let relation_unknown =
                offset.semantic_unknown || offset.havoced || offsets.is_empty() || width_unknown;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = relation_unknown;
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            result.len_of.clear();
            result.length_atoms.clear();
            result.length_relation_unknown = true;
            for offset in offsets {
                for width in &widths {
                    result.checked_end_atoms.insert(CheckedEndAtom {
                        captured_versions: current_storage_versions(state, &offset),
                        offset: offset.clone(),
                        width: *width,
                        definition,
                        branch_ready: false,
                        invalidated: false,
                    });
                }
            }
            result.value_flow.clear();
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.may_be_non_length = true;
            result
        }
        RegistryValueModel::CheckedTryBranch(option) => {
            let mut result = eval_operand(state, option);
            result.checked_end_atoms = result
                .checked_end_atoms
                .into_iter()
                .map(|mut atom| {
                    atom.branch_ready = true;
                    atom
                })
                .collect();
            result
        }
        RegistryValueModel::Constant(value) => ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Constant(*value)]),
            may_be_non_length: true,
            semantic_unknown: true,
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        },
        RegistryValueModel::SlicePrefix { base, range } => {
            let mut result = eval_operand(state, base);
            result.range_end.extend(state_value(state, range).range_end);
            result.range_end.insert(range.clone());
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            result
        }
        RegistryValueModel::SliceAsPtr(base) => {
            let mut result = eval_operand(state, base);
            result.pointer_base = semantic_place_candidates(&result);
            clear_whole_initialization_target(&mut result);
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            result
        }
        RegistryValueModel::WrappingAdd { base, offset } => {
            let mut result = eval_operand(state, base);
            let offset = eval_operand(state, offset);
            result.pointer_offset = semantic_place_candidates(&offset);
            result.value.origins.extend(offset.value.origins);
            result.exact_roots.clear();
            result.semantic_unknown = true;
            result.utf8_atoms.clear();
            result.utf8_relation_unknown = true;
            result.checked_end_atoms.clear();
            result.checked_end_unknown = true;
            result.range_limit_atoms.clear();
            result.range_limit_unknown = true;
            clear_whole_initialization_target(&mut result);
            result
        }
        RegistryValueModel::IdentityPointerView {
            operand,
            generic_view,
        } => {
            let mut result = eval_operand(state, operand);
            if let Some(evidence) = generic_view {
                attach_generic_capability(&mut result, evidence, point, false);
            }
            result
        }
        RegistryValueModel::AddressPointerView {
            operand,
            generic_view,
        } => {
            let mut result = eval_operand(state, operand);
            clear_pointer_metadata_identity(&mut result);
            if let Some(evidence) = generic_view {
                attach_generic_capability(&mut result, evidence, point, false);
            }
            result
        }
        RegistryValueModel::PointerCast { operand, .. } => {
            let mut result = eval_operand(state, operand);
            clear_pointer_metadata_identity(&mut result);
            result
        }
        RegistryValueModel::GenericPointerCast { base, evidence } => {
            let mut result = eval_operand(state, base);
            attach_generic_capability(&mut result, evidence, point, true);
            clear_pointer_metadata_identity(&mut result);
            result
        }
    }
}

fn state_alias_closure(state: &ProgramState, seed: &PlaceKey) -> BTreeSet<PlaceKey> {
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

fn has_relation_evidence_for_storage(value: &ValueFacts, written_storage: &PlaceKey) -> bool {
    value
        .length_atoms
        .iter()
        .any(|atom| !atom.invalidated && canonical_storage(&atom.collection) == *written_storage)
        || value
            .utf8_atoms
            .iter()
            .any(|atom| !atom.invalidated && canonical_storage(&atom.bytes) == *written_storage)
        || value
            .checked_end_atoms
            .iter()
            .any(|atom| !atom.invalidated && canonical_storage(&atom.offset) == *written_storage)
        || value.range_limit_atoms.iter().any(|atom| {
            !atom.invalidated && canonical_storage(&atom.collection) == *written_storage
        })
}

#[cfg(test)]
std::thread_local! {
    static RELATION_INVALIDATION_CLONE_COUNT: std::cell::Cell<usize> =
        std::cell::Cell::new(0);
}

#[cfg(test)]
fn record_relation_invalidation_clone() {
    RELATION_INVALIDATION_CLONE_COUNT.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn take_relation_invalidation_clone_count() -> usize {
    RELATION_INVALIDATION_CLONE_COUNT.with(|count| count.replace(0))
}

fn invalidate_relation_evidence(state: &mut ProgramState, written_storage: &PlaceKey) {
    let updates = state
        .may_values()
        .iter()
        .filter_map(|(place, value)| {
            if !has_relation_evidence_for_storage(value, written_storage) {
                return None;
            }
            #[cfg(test)]
            record_relation_invalidation_clone();
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
            let utf8_atoms = updated.utf8_atoms.iter().cloned().collect::<Vec<_>>();
            for atom in utf8_atoms {
                if canonical_storage(&atom.bytes) != *written_storage || atom.invalidated {
                    continue;
                }
                updated.utf8_atoms.remove(&atom);
                let mut invalidated = atom;
                invalidated.invalidated = true;
                updated.utf8_atoms.insert(invalidated);
                changed = true;
            }
            let checked_atoms = updated
                .checked_end_atoms
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            for atom in checked_atoms {
                if canonical_storage(&atom.offset) != *written_storage || atom.invalidated {
                    continue;
                }
                updated.checked_end_atoms.remove(&atom);
                let mut invalidated = atom;
                invalidated.invalidated = true;
                updated.checked_end_atoms.insert(invalidated);
                changed = true;
            }
            let limit_atoms = updated
                .range_limit_atoms
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            for atom in limit_atoms {
                if canonical_storage(&atom.collection) != *written_storage || atom.invalidated {
                    continue;
                }
                updated.range_limit_atoms.remove(&atom);
                let mut invalidated = atom;
                invalidated.invalidated = true;
                updated.range_limit_atoms.insert(invalidated);
                changed = true;
            }
            changed.then(|| (place.clone(), updated))
        })
        .collect::<Vec<_>>();
    for (place, value) in updates {
        state.set_value(place, value);
    }
}

fn assignment_destination_projection(assignment: &AssignmentModel) -> PlaceProjectionModel {
    if let Some(projection) = &assignment.destination_projection {
        return projection.clone();
    }
    if let Some(field) = &assignment.field_origin {
        if let Some(path) = &field.projection_path {
            return PlaceProjectionModel {
                base: field.base.clone(),
                path: path.clone(),
                projection_unknown: field.projection_unknown,
            };
        }
    }
    // Hand-built tests and legacy owned operations which predate structured
    // destinations can still prove a write to the exact unprojected place.
    PlaceProjectionModel {
        base: assignment.destination.clone(),
        path: Vec::new(),
        projection_unknown: false,
    }
}

fn strong_destination_targets(
    state: &ProgramState,
    destination: &PlaceProjectionModel,
) -> Vec<(PlaceKey, Vec<ProjectionKey>)> {
    if destination.projection_unknown {
        return Vec::new();
    }
    let mut targets = vec![(destination.base.clone(), destination.path.clone())];
    if let Some((ProjectionKey::Deref, suffix)) = destination.path.split_first() {
        let (roots, semantic_unknown) = exact_semantics(state, &destination.base);
        if !semantic_unknown && roots.len() == 1 {
            let root = roots.into_iter().next().expect("one exact root");
            targets.push((root, suffix.to_vec()));
        }
    }
    targets.sort();
    targets.dedup();
    targets
}

fn invalidate_overwritten_public_field_slots(
    state: &mut ProgramState,
    assignment: &AssignmentModel,
    replacement: &ValueFacts,
) {
    let destination = assignment_destination_projection(assignment);
    let strong_targets = strong_destination_targets(state, &destination);
    let mut remove = BTreeSet::new();
    let mut havoc = BTreeSet::new();
    for (place, value) in state.may_values() {
        for slot in &value.stored_public_field_slots {
            if slot.full_place != *place {
                continue;
            }
            if strong_targets
                .iter()
                .any(|(base, path)| base == &slot.base && slot.path.starts_with(path))
            {
                remove.insert(place.clone());
            } else if destination.projection_unknown && destination.base == slot.base {
                havoc.insert(place.clone());
            }
        }
    }
    for place in remove {
        let mut killed = replacement.clone();
        killed.stored_public_field_slots.clear();
        killed.may_hold_entry_public_field = false;
        if place != assignment.destination {
            killed.exact_roots.clear();
            killed.semantic_unknown = true;
        }
        state.set_value(place, killed);
    }
    for place in havoc {
        let Some(value) = state.may_values_mut().get_mut(&place) else {
            continue;
        };
        value.havoced = true;
        value.semantic_unknown = true;
        value.behavior_controller_unknown = true;
    }
}

fn capture_bound_endpoint_sites(
    function: &FunctionKey,
    closure_body: bool,
    block: BasicBlock,
    index: usize,
    operation: &ProgramOp,
    state: &ProgramState,
    graph: &mut EndpointFlowGraph,
    initialized_operands: &mut BTreeSet<(String, unsafe_site::OperandRole)>,
    possibly_uninitialized_operands: &mut BTreeSet<(String, unsafe_site::OperandRole)>,
    initialization_witnesses: &mut BTreeMap<
        (String, unsafe_site::OperandRole),
        BTreeSet<InitializationWitness>,
    >,
) {
    for site in &operation.unsafe_sites {
        let site_id = site.seed.site_id();
        let point = operation.point.clone().unwrap_or_else(|| ProgramPoint {
            function: function.clone(),
            block: block.index() as u32,
            statement: index as u32,
            span: site.seed.raw_span.stable.clone(),
        });
        for operand in &site.operands {
            let Some(place) = &operand.place else {
                continue;
            };
            let value = state_value(state, place);
            if definitely_initialized(&value) {
                initialized_operands.insert((site_id.clone(), operand.role.clone()));
                initialization_witnesses
                    .entry((site_id.clone(), operand.role.clone()))
                    .or_default()
                    .extend(value.initialization_witnesses.iter().cloned());
            } else if !value.ffi_output && value.maybe_uninit_initialized.contains(&false) {
                // Preserve a realizable uninitialized branch even when a
                // sibling branch or an ambiguous typed write may initialize
                // the same storage. This is a finite may-witness, not an
                // inference from the absence of a proof.
                possibly_uninitialized_operands.insert((site_id.clone(), operand.role.clone()));
            }
            for edge in value.endpoint_edges.iter().cloned() {
                graph.insert_edge(edge);
            }
            let mut sources = value.endpoint_sources.as_ref().clone();
            if closure_body {
                sources.extend(
                    value
                        .dependencies
                        .iter()
                        .chain(value.value_flow.iter())
                        .chain(std::iter::once(place))
                        .filter_map(closure_capture_index)
                        .map(|index| EndpointNode::Capture {
                            owner: function.clone(),
                            index,
                        }),
                );
            }
            let target = EndpointNode::UnsafeOperand {
                owner: function.clone(),
                site_id: site_id.clone(),
                role: operand.role.clone(),
            };
            graph.insert_node(target.clone());
            for source in sources.iter().cloned() {
                let kind = if value.influence_seeds.has_exact_derived_capabilities(
                    &source,
                    InfluenceCapabilities::ADDRESS | InfluenceCapabilities::VALIDITY_BITS,
                ) {
                    EndpointEdgeKind::TypedValidityPointer
                } else if value.private_aggregate_transform_unknown {
                    EndpointEdgeKind::PrivateAggregateTransform
                } else {
                    EndpointEdgeKind::UnsafeOperand
                };
                graph.insert_edge(EndpointEdge {
                    from: source,
                    to: target.clone(),
                    kind,
                    point: Some(point.clone()),
                });
            }
        }
        if site.result_place.is_some() {
            graph.insert_node(EndpointNode::UnsafeResult {
                owner: function.clone(),
                site_id,
            });
        }
    }
}

fn closure_capture_index(place: &PlaceKey) -> Option<u32> {
    let marker = "_1.";
    let start = place.0.find(marker)? + marker.len();
    let digits = place.0[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn closure_capture_index_from_value(value: &ValueModel) -> Option<u32> {
    let operand = match value {
        ValueModel::Operand(operand)
        | ValueModel::RefOrRaw(operand)
        | ValueModel::LocalRefOrRaw(operand)
        | ValueModel::IdentityPointerView { operand, .. }
        | ValueModel::AddressPointerView { operand, .. }
        | ValueModel::PointerCast { operand, .. }
        | ValueModel::Unsize { operand, .. }
        | ValueModel::GenericPointerCast { operand, .. }
        | ValueModel::BehaviorPointer { operand, .. } => operand,
        _ => return None,
    };
    operand.operand.place().and_then(closure_capture_index)
}



fn site_collection_place(facts: &BodyFacts, site_id: &str) -> Option<PlaceKey> {
    facts.contract_sites_by_point.iter().find_map(|(point, ids)| {
        ids.contains(site_id).then(|| facts.local_bindings.get(point)).flatten()
    }).and_then(|binding| binding.collection.clone())
}

fn snapshot_for_site<'a>(
    facts: &'a BodyFacts,
    site_id: &str,
    site: &BoundUnsafeSite,
) -> Option<&'a ValidationSnapshot> {
    for (point, ids) in &facts.contract_sites_by_point {
        if !ids.contains(site_id) {
            continue;
        }
        let location = point.point_location();
        if let Some(snapshot) = facts
            .validation_snapshots
            .get(&(location.block, location.statement_index))
        {
            return Some(snapshot);
        }
    }
    facts.validation_snapshots.get(&(
        BasicBlock::from_usize(site.solver.block as usize),
        site.solver.statement as usize,
    ))
}

fn warning_subject_place(site: &BoundUnsafeSite, predicate: WarningPredicate) -> Option<PlaceKey> {
    site.operands
        .iter()
        .filter_map(|operand| {
            let rank = match predicate {
                WarningPredicate::NonNull => match operand.role {
                    unsafe_site::OperandRole::Address => Some(0),
                    unsafe_site::OperandRole::Place => Some(1),
                    unsafe_site::OperandRole::Argument(0) => Some(2),
                    unsafe_site::OperandRole::Receiver => Some(3),
                    _ => None,
                },
                WarningPredicate::InBounds => match operand.role {
                    unsafe_site::OperandRole::Index => Some(0),
                    unsafe_site::OperandRole::Argument(1) => Some(1),
                    _ => None,
                },
            }?;
            operand.place.clone().map(|place| (rank, place))
        })
        .min_by_key(|(rank, _)| *rank)
        .map(|(_, place)| place)
}

fn places_equivalent(left: &PlaceKey, right: &PlaceKey) -> bool {
    left == right || canonical_storage(left) == canonical_storage(right)
}

fn place_carries(facts: &BodyFacts, origin: &PlaceKey, target: &PlaceKey) -> bool {
    if places_equivalent(origin, target) {
        return true;
    }
    facts.values.get(target).is_some_and(|value| {
        value
            .value_flow
            .iter()
            .any(|place| places_equivalent(place, origin))
    }) || facts.values.get(origin).is_some_and(|value| {
        value
            .value_flow
            .iter()
            .any(|place| places_equivalent(place, target))
    })
}

fn sanitizer_dominates(
    snapshot: &ValidationSnapshot,
    facts: &BodyFacts,
    site: &BoundUnsafeSite,
    site_id: &str,
    predicate: WarningPredicate,
    subject: Option<&PlaceKey>,
) -> bool {
    match predicate {
        WarningPredicate::NonNull => {
            let Some(subject) = subject else {
                return false;
            };
            snapshot.current_validations.iter().any(|binding| {
                binding.predicate == Predicate::NonNull
                    && place_carries(facts, &binding.subject, subject)
            })
        }
        WarningPredicate::InBounds => {
            let via_local = facts.contract_sites_by_point.iter().any(|(point, ids)| {
                ids.contains(site_id)
                    && facts
                        .local_bindings
                        .get(point)
                        .is_some_and(|binding| snapshot.proves(binding))
            });
            if via_local {
                return true;
            }
            let Some(subject) = subject else {
                return false;
            };
            let collection = site
                .operands
                .iter()
                .find(|operand| {
                    matches!(
                        operand.role,
                        super::unsafe_site::OperandRole::Receiver
                            | super::unsafe_site::OperandRole::Argument(0)
                    )
                })
                .and_then(|operand| operand.place.clone())
                .or_else(|| site_collection_place(facts, site_id));
            snapshot.current_validations.iter().any(|binding| {
                binding.predicate == Predicate::InBounds
                    && place_carries(facts, &binding.subject, subject)
            }) || snapshot.proves(&ValidationBinding {
                predicate: Predicate::InBounds,
                subject: subject.clone(),
                collection,
                access_width: None,
                proof_definition: None,
            })
        }
    }
}

fn record_site_contracts(
    operation: &ProgramOp,
    known_contract: impl Fn(&BoundUnsafeSite) -> Option<KnownContractModel>,
    structural_sites: &mut BTreeMap<String, StructuralSiteCandidate>,
    descriptors: &mut BTreeMap<String, ContractDescriptor>,
    sites_by_point: &mut BTreeMap<ProgramPoint, BTreeSet<String>>,
    bound_sites: &mut BTreeMap<String, BoundUnsafeSite>,
) {
    let catalog = ContractCatalog::full();
    for site in &operation.unsafe_sites {
        let site_id = site.seed.site_id();
        let descriptor = catalog.describe(site, known_contract(site));
        if let Some(candidate) = structural_candidate_with_contract(site, &descriptor) {
            structural_sites.insert(site_id.clone(), candidate);
        }
        let point = operation.point.clone().unwrap_or_else(|| ProgramPoint {
            function: site.seed.owner.clone(),
            block: site.solver.block,
            statement: site.solver.statement,
            span: site.seed.raw_span.stable.clone(),
        });
        sites_by_point
            .entry(point)
            .or_default()
            .insert(site_id.clone());
        bound_sites.insert(site_id.clone(), site.clone());
        descriptors.insert(site_id, descriptor);
    }
}

fn has_contract_model(facts: &BodyFacts, point: &ProgramPoint, model: KnownContractModel) -> bool {
    has_contract_model_id(facts, point, model.as_str())
}

fn has_contract_model_id(facts: &BodyFacts, point: &ProgramPoint, model_id: &str) -> bool {
    facts
        .contract_sites_by_point
        .get(point)
        .into_iter()
        .flatten()
        .filter_map(|site_id| facts.contract_descriptors.get(site_id))
        .any(|descriptor| {
            descriptor.precision != unsafe_site::ContractPrecision::Opaque
                && descriptor.model_id.as_deref() == Some(model_id)
        })
}

fn install_unsafe_result_sites(sites: &[BoundUnsafeSite], state: &mut ProgramState) {
    let mut sites_by_result = BTreeMap::<PlaceKey, Vec<&BoundUnsafeSite>>::new();
    for site in sites {
        let Some(result_place) = &site.result_place else {
            continue;
        };
        sites_by_result
            .entry(result_place.clone())
            .or_default()
            .push(site);
    }
    for (result_place, result_sites) in sites_by_result {
        let mut value = state
            .may_values()
            .get(&result_place)
            .cloned()
            .unwrap_or_default();
        let sources = value.endpoint_sources.as_ref().clone();
        let mut pending_edges = value.endpoint_edges.as_ref().clone();
        let mut results = BTreeSet::new();
        for site in result_sites {
            let result = EndpointNode::UnsafeResult {
                owner: site.seed.owner.clone(),
                site_id: site.seed.site_id(),
            };
            let point = ProgramPoint {
                function: site.seed.owner.clone(),
                block: site.solver.block,
                statement: site.solver.statement,
                span: site.seed.raw_span.stable.clone(),
            };
            pending_edges.extend(sources.iter().cloned().map(|source| EndpointEdge {
                from: source,
                to: result.clone(),
                kind: EndpointEdgeKind::UnsafeResult,
                point: Some(point.clone()),
            }));
            results.insert(result);
        }
        value.endpoint_edges = Arc::new(pending_edges);
        value.endpoint_sources = Arc::new(results);
        state.set_value(result_place, value);
    }
}

fn apply_program_op(
    operation: &ProgramOp,
    block: BasicBlock,
    index: usize,
    state: &mut ProgramState,
) {
    apply_program_op_with_context(operation, block, index, state, false);
}

fn clear_endpoint_facts(state: &mut ProgramState) {
    for value in state.may_values_mut().values_mut() {
        if !value.endpoint_sources.is_empty() {
            value.endpoint_sources = Arc::default();
        }
        if !value.endpoint_edges.is_empty() {
            value.endpoint_edges = Arc::default();
        }
        value.influence_seeds.clear();
        value.behavior_controllers.clear();
        value.behavior_controller_unknown = false;
    }
}

fn apply_program_op_with_context(
    operation: &ProgramOp,
    block: BasicBlock,
    index: usize,
    state: &mut ProgramState,
    closure_body: bool,
) {
    let token = StaticWriteToken::new(block.index() as u32, index as u32);
    match &operation.kind {
        ProgramOpKind::Assign(assignment) => {
            let copied_pending = match &assignment.value {
                ValueModel::Operand(operand) | ValueModel::Discriminant(operand) => {
                    operand.operand.place().map(|source| {
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
                    })
                }
                _ => None,
            };
            let mut value = eval_value_model(state, &assignment.value, token);
            if closure_body {
                if let Some(capture_index) = closure_capture_index_from_value(&assignment.value) {
                    Arc::make_mut(&mut value.endpoint_sources).insert(EndpointNode::Capture {
                        owner: operation.semantic_point().function.clone(),
                        index: capture_index,
                    });
                }
            }
            let relation = assignment_relation(state, assignment);
            let aggregate_proofs = aggregate_proofs(state, assignment);
            let pending = match &assignment.value {
                ValueModel::Length(_) => current_length_collection(state, &value)
                    .map(|collection| {
                        let binding = ValidationBinding {
                            predicate: Predicate::NonEmpty,
                            access_width: None,
                            proof_definition: None,
                            subject: collection,
                            collection: None,
                        };
                        let places = binding.storage_places();
                        vec![(binding.into(), true, places)]
                    })
                    .unwrap_or_default(),
                ValueModel::Discriminant(_) => {
                    let mut pending =
                        current_checked_end(state, &value).map_or_else(Vec::new, |end| {
                            let binding = ValidationBinding::range(
                                end.offset,
                                None,
                                end.width,
                                Some(end.definition),
                            );
                            let places = binding.storage_places();
                            vec![(binding.into(), false, places)]
                        });
                    if let Some(bytes) = current_utf8_bytes(state, &value) {
                        let binding = ValidationBinding {
                            predicate: Predicate::ValidUtf8,
                            access_width: None,
                            proof_definition: None,
                            subject: bytes,
                            collection: None,
                        };
                        let places = binding.storage_places();
                        pending.push((binding.into(), false, places));
                    }
                    pending.extend(copied_pending.clone().unwrap_or_default());
                    pending
                }
                ValueModel::Compare { op, left, right } => {
                    let pending = normalized_compare_pending(state, *op, left, right);
                    if pending.is_empty() {
                        normalized_relation_pending(state, *op, left, right)
                    } else {
                        pending
                    }
                }
                _ => copied_pending.unwrap_or_default(),
            };
            if let Some(field) = &assignment.field_origin {
                value.storage_provenance = state
                    .may_values()
                    .get(&field.base)
                    .map(|base| base.storage_provenance.clone())
                    .unwrap_or(StorageProvenance::MaybeExternal);
            }
            let generic_evidence = match &assignment.value {
                ValueModel::GenericPointerCast { evidence, .. } => Some((evidence, true)),
                ValueModel::IdentityPointerView {
                    generic_view: Some(evidence),
                    ..
                }
                | ValueModel::AddressPointerView {
                    generic_view: Some(evidence),
                    ..
                } => Some((evidence, false)),
                _ => None,
            };
            if let Some((evidence, legacy_origin)) = generic_evidence {
                attach_generic_capability(
                    &mut value,
                    evidence,
                    operation.semantic_point(),
                    legacy_origin,
                );
            }
            let defines_fresh_identity = matches!(
                &assignment.value,
                ValueModel::Operand(OperandModel {
                    operand: CallOperand::Constant(_),
                    ..
                }) | ValueModel::Length(_)
                    | ValueModel::Discriminant(_)
                    | ValueModel::Compare { .. }
                    | ValueModel::CheckedBinary { .. }
                    | ValueModel::Aggregate { .. }
            );
            if defines_fresh_identity {
                value.exact_roots.clear();
                value.exact_roots.insert(assignment.destination.clone());
                value.semantic_unknown = false;
                value.storage_provenance = StorageProvenance::DefinitelyLocal;
            }
            if let Some(kind) = value_model_derivation_kind(&assignment.value) {
                let derivation = EndpointNode::InternalDerivation {
                    owner: operation.semantic_point().function.clone(),
                    kind,
                    token: assignment.destination.0.clone(),
                    point: operation.semantic_point().clone(),
                };
                let source_edges =
                    value
                        .endpoint_sources
                        .iter()
                        .cloned()
                        .map(|source| EndpointEdge {
                            from: source,
                            to: derivation.clone(),
                            kind: EndpointEdgeKind::Derivation,
                            point: operation.point.clone(),
                        });
                value.endpoint_edges = Arc::new(source_edges.collect());
                Arc::make_mut(&mut value.endpoint_sources).clear();
                Arc::make_mut(&mut value.endpoint_sources).insert(derivation);
            }
            let out_value = assignment.out_formal.map(|formal_index| {
                let mut output = value.clone();
                output.exact_roots.clear();
                output.semantic_unknown = true;
                (formal_index, output)
            });
            let storage = canonical_storage(&assignment.destination);
            invalidate_overwritten_public_field_slots(state, assignment, &value);
            state.record_write(storage.clone(), token);
            invalidate_relation_evidence(state, &storage);
            state.set_value(assignment.destination.clone(), value);
            if let Some(relation) = relation {
                let places = relation_storage_places(&relation);
                state.establish(BoundValidation::new(
                    ProofBinding::Relation(relation),
                    places,
                ));
            }
            for binding in aggregate_proofs {
                state.establish(BoundValidation::new(
                    binding.clone(),
                    binding.storage_places(),
                ));
            }
            if let Some((formal_index, output)) = out_value {
                strong_write_out_formal(state, formal_index, &output, token);
            }

            if !pending.is_empty() {
                state.set_pending(assignment.destination.clone(), pending);
            }
        }
        // Call effects belong to the normal edge, never the unwind edge.
        ProgramOpKind::Call(_) => {}
        ProgramOpKind::Nop => {}
    }
    if !matches!(operation.kind, ProgramOpKind::Call(_)) {
        install_unsafe_result_sites(&operation.unsafe_sites, state);
    }
}

fn apply_program_edge(edge: &EdgeTerm, target: BasicBlock, state: &mut ProgramState) {
    apply_program_edge_with_outputs(edge, target, state, &LocalCallOutputs::new());
}

fn apply_program_edge_with_outputs(
    edge: &EdgeTerm,
    target: BasicBlock,
    state: &mut ProgramState,
    local_outputs: &LocalCallOutputs,
) {
    apply_program_edge_with_outputs_and_sites(edge, target, state, local_outputs, &[]);
}

fn apply_program_edge_with_outputs_and_sites(
    edge: &EdgeTerm,
    target: BasicBlock,
    state: &mut ProgramState,
    local_outputs: &LocalCallOutputs,
    unsafe_sites: &[BoundUnsafeSite],
) {
    apply_program_edge_with_outputs_sites_and_programs(
        edge,
        target,
        state,
        local_outputs,
        unsafe_sites,
        None,
    );
}

fn apply_program_edge_with_outputs_sites_and_programs(
    edge: &EdgeTerm,
    target: BasicBlock,
    state: &mut ProgramState,
    local_outputs: &LocalCallOutputs,
    unsafe_sites: &[BoundUnsafeSite],
    programs: Option<&BTreeMap<FunctionKey, BodyProgram>>,
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
        let local_boundary = (call.descriptor.disposition == BoundaryDisposition::ResolvedLocal)
            .then(|| local_call_boundary_mapping(call, state));
        let registry_relation_effects = registry_relation_effects(state, call);
        let observed_internal_target = match (&call.value, programs) {
            (RegistryValueModel::IndirectBehavior { source: callee, .. }, Some(programs))
                if call.descriptor.disposition == BoundaryDisposition::OpaqueIndirect =>
            {
                let callable = eval_operand(state, callee);
                observed_internal_behavior_target(&callable, |function| {
                    programs.get(function).map(|program| program.is_closure)
                })
            }
            _ => None,
        };
        let mut return_value = eval_registry_value(state, &call.value, point);
        if matches!(
            call.value,
            RegistryValueModel::Empty | RegistryValueModel::IndirectBehavior { .. }
        ) && matches!(
            call.descriptor.disposition,
            BoundaryDisposition::OpaqueDirect | BoundaryDisposition::OpaqueIndirect
        ) {
            if let Some(target) = &observed_internal_target {
                return_value = ValueFacts {
                    endpoint_sources: Arc::new(BTreeSet::from([indirect_behavior_result_node(
                        point,
                        &target.controller,
                    )])),
                    may_be_non_length: true,
                    semantic_unknown: true,
                    behavior_controller_unknown: true,
                    utf8_relation_unknown: true,
                    checked_end_unknown: true,
                    range_limit_unknown: true,
                    known_length_unknown: true,
                    ..ValueFacts::default()
                };
            } else {
                return_value = join_value_facts(
                    call.descriptor
                        .args
                        .iter()
                        .map(|operand| eval_operand(state, operand)),
                );
                clear_whole_initialization_target(&mut return_value);
                return_value.havoced = true;
                return_value
                    .influence_seeds
                    .mark_all_incomplete(UnestablishedReason::HavocedProvenance);
                return_value.exact_roots.clear();
                return_value.semantic_unknown = true;
                if call.descriptor.disposition == BoundaryDisposition::OpaqueIndirect {
                    let (controllers, type_class) = match &call.value {
                        RegistryValueModel::IndirectBehavior {
                            source: callee,
                            type_class,
                        } => (
                            behavior_controllers(&eval_operand(state, callee)),
                            *type_class,
                        ),
                        _ => (
                            BTreeSet::from([BehaviorController::Unknown]),
                            InfluenceTypeClass::Opaque,
                        ),
                    };
                    for controller in controllers {
                        let endpoint = indirect_behavior_result_node(point, &controller);
                        Arc::make_mut(&mut return_value.endpoint_sources).insert(endpoint.clone());
                        insert_external_influence_seed(
                            &mut return_value,
                            endpoint,
                            ExternalSourceKind::FunctionPointer,
                            &controller,
                            type_class,
                        );
                    }
                }
            }
        }
        if let Some(PredicateModel::Utf8Result { bytes }) = &call.predicate {
            let bytes_value = eval_operand(state, bytes);
            let candidates = semantic_place_candidates(&bytes_value);
            return_value.utf8_relation_unknown = bytes_value.semantic_unknown;
            for bytes in candidates {
                return_value.utf8_atoms.insert(Utf8Atom {
                    captured_versions: current_storage_versions(state, &bytes),
                    bytes,
                    definition: StaticWriteToken::new(point.block, point.statement),
                    invalidated: false,
                });
            }
        }
        // A crate-local return is defined by the callee's return summary below.
        // Reclassifying it from a caller-side helper model would fabricate a
        // derivation and discard constant/strong return semantics.
        if call.descriptor.disposition != BoundaryDisposition::ResolvedLocal
            && registry_value_derivation_kind(&call.value).is_some()
        {
            let derivation = EndpointNode::InternalDerivation {
                owner: point.function.clone(),
                kind: registry_value_derivation_kind(&call.value)
                    .expect("derivation kind was checked"),
                token: call.descriptor.destination.0.clone(),
                point: point.clone(),
            };
            let source_edges =
                return_value
                    .endpoint_sources
                    .iter()
                    .cloned()
                    .map(|source| EndpointEdge {
                        from: source,
                        to: derivation.clone(),
                        kind: EndpointEdgeKind::Derivation,
                        point: Some(point.clone()),
                    });
            return_value.endpoint_edges = Arc::new(source_edges.collect());
            Arc::make_mut(&mut return_value.endpoint_sources).clear();
            Arc::make_mut(&mut return_value.endpoint_sources).insert(derivation);
        }
        apply_call_side_effects(call, point, state);
        if let Some(outputs) = local_outputs.get(point) {
            apply_local_outputs(state, &outputs.outs);
        }
        apply_call_return(call, point, return_value, state);
        if let Some((destination, returned)) = local_outputs
            .get(point)
            .and_then(|outputs| outputs.returned.as_ref())
        {
            debug_assert_eq!(*destination, call.descriptor.destination);
            let mut current = state_value(state, destination);
            current.value = returned.clone();
            state.set_value(destination.clone(), current);
        }
        if let (Some(callee), Some(_)) = (&call.descriptor.callee, &local_boundary) {
            let destination = &call.descriptor.destination;
            let mut current = state
                .may_values()
                .get(destination)
                .cloned()
                .unwrap_or_default();
            Arc::make_mut(&mut current.endpoint_sources).clear();
            Arc::make_mut(&mut current.endpoint_sources).insert(EndpointNode::CallResult {
                callsite: CallSiteId::new(point.function.clone(), point.clone(), callee.clone()),
                output: CallSiteOutput::Return,
            });
            state.set_value(destination.clone(), current);
        }
        if let (Some(callee), Some(mapping)) = (&call.descriptor.callee, &local_boundary) {
            for boundary in mapping {
                let CallMapping::OutToCaller {
                    formal_index,
                    caller_place,
                } = boundary
                else {
                    continue;
                };
                let mut current = state
                    .may_values()
                    .get(caller_place)
                    .cloned()
                    .unwrap_or_default();
                let mapped = local_outputs
                    .get(point)
                    .and_then(|outputs| outputs.outs.get(caller_place));
                if mapped.is_some_and(|output| !output.may_skip_write) {
                    Arc::make_mut(&mut current.endpoint_sources).clear();
                }
                // Endpoint provenance is independently materialized from the
                // resolved callee relation.  Do not let an absent AbstractValue
                // local-output summary erase this exact normal-edge endpoint.
                // In that case the context layer remains Unknown rather than
                // proving the output unrealizable.
                Arc::make_mut(&mut current.endpoint_sources).insert(EndpointNode::CallResult {
                    callsite: CallSiteId::new(
                        point.function.clone(),
                        point.clone(),
                        callee.clone(),
                    ),
                    output: CallSiteOutput::Out(*formal_index),
                });
                state.set_value(caller_place.clone(), current);
            }
        }
        if let Some(formal_index) = call.destination_out_formal {
            let output = state
                .may_values()
                .get(&call.descriptor.destination)
                .cloned()
                .unwrap_or_default();
            strong_write_out_formal(
                state,
                formal_index,
                &output,
                StaticWriteToken::new(point.block, point.statement),
            );
        }
        let mut pending = Vec::new();
        for effect in registry_relation_effects {
            match effect {
                RegistryRelationEffect::Unconditional(relation) => {
                    state.establish(BoundValidation::new(
                        ProofBinding::Relation(relation.clone()),
                        relation_storage_places(&relation),
                    ));
                }
                RegistryRelationEffect::Pending { relation, polarity } => pending.push((
                    ProofBinding::Relation(relation.clone()),
                    polarity,
                    relation_storage_places(&relation),
                )),
            }
        }
        if let Some(outputs) = local_outputs.get(point) {
            for relation in &outputs.relations {
                state.establish(BoundValidation::new(
                    ProofBinding::Relation(relation.clone()),
                    relation_storage_places(relation),
                ));
            }
            pending.extend(
                outputs
                    .conditional_relations
                    .iter()
                    .filter_map(|conditional| {
                        let polarity = match conditional.condition.discriminant {
                            0 => false,
                            1 => true,
                            _ => return None,
                        };
                        Some((
                            ProofBinding::Relation(conditional.relation.clone()),
                            polarity,
                            relation_storage_places(&conditional.relation),
                        ))
                    }),
            );
        }
        if !pending.is_empty() {
            state.set_pending(call.descriptor.destination.clone(), pending);
        }
        install_unsafe_result_sites(unsafe_sites, state);
        return;
    }
    if let EdgeTerm::AssertPlumbingOnly {
        condition,
        expected,
        target: success,
    } = edge
    {
        if target != *success {
            return;
        }
        let bindings = state
            .pending_must()
            .get(condition)
            .into_iter()
            .flat_map(|facts| facts.keys())
            .filter(|(binding, polarity)| {
                (binding.relation().is_some()
                    || binding
                        .validation()
                        .is_some_and(|binding| binding.predicate == Predicate::InBounds))
                    && *polarity == *expected
                    && state.pending_is_current(condition, binding, *polarity)
            })
            .map(|(binding, _)| binding.clone())
            .collect::<Vec<_>>();
        for binding in bindings {
            establish_proof(state, binding);
        }
        return;
    }
    let EdgeTerm::Switch {
        condition,
        false_target,
        true_target,
    } = edge
    else {
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
        establish_proof(state, binding);
    }
}

fn registry_value_derivation_kind(value: &RegistryValueModel) -> Option<DerivationKind> {
    match value {
        RegistryValueModel::Length(_) => Some(DerivationKind::Length),
        RegistryValueModel::SaturatingSub(_) | RegistryValueModel::WrappingAdd { .. } => {
            Some(DerivationKind::Offset)
        }
        RegistryValueModel::CheckedAdd { .. } => Some(DerivationKind::AccessWidth),
        RegistryValueModel::SlicePrefix { .. } => Some(DerivationKind::Length),
        RegistryValueModel::PointerCast { .. } => Some(DerivationKind::Layout),
        RegistryValueModel::Empty
        | RegistryValueModel::PrivateAggregateTransform
        | RegistryValueModel::ExternalBehavior { .. }
        | RegistryValueModel::MaybeUninit(_)
        | RegistryValueModel::TypedInitializationPointer { .. }
        | RegistryValueModel::AssumeInit { .. }
        | RegistryValueModel::SliceIterator(_)
        | RegistryValueModel::SlicePosition(_)
        | RegistryValueModel::RangeIntoIterator(_)
        | RegistryValueModel::RangeIteratorNext(_)
        | RegistryValueModel::SelectedOpenBehavior { .. }
        | RegistryValueModel::IndirectBehavior { .. }
        | RegistryValueModel::GenericCapability { .. }
        | RegistryValueModel::Constant(_)
        | RegistryValueModel::SliceAsPtr(_)
        | RegistryValueModel::CheckedTryBranch(_)
        | RegistryValueModel::IdentityPointerView { .. }
        | RegistryValueModel::AddressPointerView { .. }
        | RegistryValueModel::GenericPointerCast { .. } => None,
    }
}

fn value_model_derivation_kind(value: &ValueModel) -> Option<DerivationKind> {
    match value {
        ValueModel::Length(_) => Some(DerivationKind::Length),
        ValueModel::Discriminant(_) => Some(DerivationKind::Discriminant),
        ValueModel::Compare { op, .. } => match op {
            BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                Some(DerivationKind::UncheckedBooleanRelation)
            }
            BinOp::Add | BinOp::Sub | BinOp::Offset => Some(DerivationKind::Offset),
            BinOp::Mul => Some(DerivationKind::AccessWidth),
            _ => None,
        },
        ValueModel::CheckedBinary { .. } => Some(DerivationKind::Offset),
        ValueModel::PointerCast { .. } => Some(DerivationKind::Layout),
        ValueModel::Operand(_)
        | ValueModel::Unsize { .. }
        | ValueModel::RefOrRaw(_)
        | ValueModel::LocalRefOrRaw(_)
        | ValueModel::IdentityPointerView { .. }
        | ValueModel::AddressPointerView { .. }
        | ValueModel::GenericPointerCast { .. }
        | ValueModel::Aggregate { .. }
        | ValueModel::BehaviorPointer { .. }
        | ValueModel::Unknown => None,
    }
}

fn establish_proof(state: &mut ProgramState, binding: ProofBinding) {
    state.establish(BoundValidation::new(
        binding.clone(),
        binding.storage_places(),
    ));
    let ProofBinding::Validation(validation) = binding else {
        return;
    };
    if validation.predicate != Predicate::InBounds {
        return;
    }
    let Some(collection) = validation.collection else {
        return;
    };
    let relation = RelationFact::new(RelationKind::Lt(
        RelationTerm::Place(validation.subject),
        RelationTerm::Length(collection),
    ));
    state.establish(BoundValidation::new(
        ProofBinding::Relation(relation.clone()),
        relation_storage_places(&relation),
    ));
}

fn apply_local_outputs<'a>(
    state: &mut ProgramState,
    outputs: &'a BTreeMap<PlaceKey, MappedOutValue>,
) {
    for (place, output) in outputs {
        apply_mapped_out_value(state, place, output);
        if let Some(formal_index) = out_slot_index(place) {
            // A nested local out call updates both summary bookkeeping and the
            // concrete referent read later in this function. The base formal is
            // carried as a second mapping; synchronizing the exact dereference
            // prevents an earlier `*out = ...` entry from retaining stale value
            // provenance after a later definite call write.
            let dereference = PlaceKey::new(format!("(*_{formal_index})"));
            apply_mapped_out_value(state, &dereference, output);
        }
    }
}

fn apply_mapped_out_value(state: &mut ProgramState, place: &PlaceKey, output: &MappedOutValue) {
    let mut current = state.may_values().get(place).cloned().unwrap_or_default();
    if output.may_skip_write {
        current
            .value
            .origins
            .extend(output.value.origins.iter().cloned());
    } else {
        current.value = output.value.clone();
    }
    state.set_value(place.clone(), current);
}

fn strong_write_out_formal(
    state: &mut ProgramState,
    formal_index: u32,
    written: &ValueFacts,
    token: StaticWriteToken,
) {
    // The base local carries the reference identity, while its `value` field is
    // the provenance observed by reads through the referent. Keep the former
    // and replace only the latter after an exact `*formal` write.
    let base = PlaceKey::new(format!("_{formal_index}"));
    let mut base_value = state.may_values().get(&base).cloned().unwrap_or_default();
    base_value.value = written.value.clone();
    state.set_value(base, base_value);

    let slot = out_slot(formal_index);
    let mut output = written.clone();
    output.exact_roots.clear();
    output.semantic_unknown = true;
    state.record_write(slot.clone(), token);
    state.set_value(slot, output);
}

fn state_proves_local_binding(state: &ProgramState, binding: Option<&ValidationBinding>) -> bool {
    binding.is_some_and(|binding| state.has_current_validation(binding, binding.storage_places()))
}

fn state_proves_condition_true_site(state: &ProgramState, site: &BoundUnsafeSite) -> bool {
    let Some(condition) = site
        .operands
        .iter()
        .find(|operand| operand.role == unsafe_site::OperandRole::Argument(0))
        .and_then(|operand| operand.place.as_ref())
    else {
        return false;
    };
    state
        .pending_must()
        .get(condition)
        .into_iter()
        .flat_map(|facts| facts.keys())
        .any(|(binding, polarity)| {
            *polarity
                && state.pending_is_current(condition, binding, *polarity)
                && state.has_current_validation(binding, binding.storage_places())
        })
}

fn is_hint_assert_unchecked_def_path(path: &rustc_hir::definitions::DefPath) -> bool {
    path.data.len() == 2
        && path
            .data
            .iter()
            .all(|component| component.disambiguator == 0)
        && path.data[0]
            .data
            .get_opt_name()
            .is_some_and(|name| name.as_str() == "hint")
        && path.data[1]
            .data
            .get_opt_name()
            .is_some_and(|name| name.as_str() == "assert_unchecked")
}

fn is_core_assert_unchecked_def_path(def_path: &str) -> bool {
    matches!(
        def_path,
        "core::hint::assert_unchecked" | "std::hint::assert_unchecked"
    )
}

fn is_core_assume_intrinsic_def_path(def_path: &str) -> bool {
    def_path == "core::intrinsics::assume"
}

fn local_requirement_is_proven(
    snapshot: &ValidationSnapshot,
    binding: Option<&ValidationBinding>,
    relational: Option<&RelationalObligation>,
) -> bool {
    relational.is_some_and(|obligation| snapshot.proves_relational(obligation))
        || binding.is_some_and(|binding| snapshot.proves(binding))
}

fn exact_contract_is_proven(
    descriptor: &ContractDescriptor,
    snapshot: Option<&ValidationSnapshot>,
    binding: Option<&ValidationBinding>,
    relational: Option<&RelationalObligation>,
    initialized_roles: &BTreeSet<unsafe_site::OperandRole>,
) -> bool {
    match descriptor.discharge {
        DischargeModel::None => false,
        DischargeModel::ExactValidation | DischargeModel::ExactRelation => snapshot
            .is_some_and(|snapshot| local_requirement_is_proven(snapshot, binding, relational)),
        DischargeModel::ExactInitialization => {
            !descriptor.relevant_roles.is_empty()
                && descriptor
                    .relevant_roles
                    .iter()
                    .all(|role| initialized_roles.contains(role))
        }
    }
}

fn proven_contract_sites(facts: &BodyFacts) -> BTreeSet<String> {
    facts
        .contract_sites_by_point
        .iter()
        .flat_map(|(point, sites)| {
            let location = point.point_location();
            let snapshot = facts
                .validation_snapshots
                .get(&(location.block, location.statement_index));
            let binding = facts.local_bindings.get(point);
            let relational = facts.local_relational_obligations.get(point);
            sites.iter().filter_map(move |site_id| {
                let descriptor = facts.contract_descriptors.get(site_id)?;
                let initialized_roles = facts
                    .initialized_site_operands
                    .iter()
                    .filter_map(|(initialized_site, role)| {
                        (initialized_site == site_id).then_some(role.clone())
                    })
                    .collect::<BTreeSet<_>>();
                exact_contract_is_proven(
                    descriptor,
                    snapshot,
                    binding,
                    relational,
                    &initialized_roles,
                )
                .then(|| site_id.clone())
            })
        })
        .collect()
}

fn sink_local_literal_index_sites(facts: &BodyFacts) -> BTreeSet<String> {
    facts
        .calls
        .iter()
        .filter(|call| {
            has_contract_model(facts, &call.point, KnownContractModel::GetUnchecked)
                || has_contract_model(facts, &call.point, KnownContractModel::VecGetUnchecked)
        })
        .filter_map(|call| {
            let CallOperand::Constant(index) = call.args.get(1)? else {
                return None;
            };
            let collection = call.arg_values.first()?;
            literal_index_within_known_lengths(*index, collection).then_some(&call.point)
        })
        .flat_map(|point| {
            facts
                .contract_sites_by_point
                .get(point)
                .into_iter()
                .flatten()
        })
        .filter(|site_id| {
            facts
                .contract_descriptors
                .get(*site_id)
                .and_then(|descriptor| descriptor.model_id.as_deref())
                .is_some_and(|model| {
                    model == KnownContractModel::GetUnchecked.as_str()
                        || model == KnownContractModel::VecGetUnchecked.as_str()
                })
        })
        .cloned()
        .collect()
}

fn literal_index_within_known_lengths(index: u64, collection: &ValueFacts) -> bool {
    !collection.known_length_unknown
        && !collection.known_lengths.is_empty()
        && collection
            .known_lengths
            .iter()
            .all(|length| index < *length)
}

fn discharge_model_name(model: DischargeModel) -> &'static str {
    match model {
        DischargeModel::None => "none",
        DischargeModel::ExactValidation => "exact_validation",
        DischargeModel::ExactRelation => "exact_relation",
        DischargeModel::ExactInitialization => "exact_initialization",
    }
}

fn static_contract_suppression(
    implication: StaticContractImplication,
) -> Option<(unsafe_site::SuppressionReason, BTreeMap<String, String>)> {
    let StaticContractImplication::Proven(evidence) = implication else {
        return None;
    };
    Some((
        unsafe_site::SuppressionReason::ContractImpliedByType,
        BTreeMap::from([
            (
                "proof_kind".to_string(),
                "static_type_implication".to_string(),
            ),
            (
                "static_implication".to_string(),
                evidence.as_str().to_string(),
            ),
        ]),
    ))
}

fn dynamic_contract_suppression(
    model: DischargeModel,
    site_id: &str,
    site_local_proven: bool,
    initialization_witnesses: &BTreeSet<String>,
) -> Option<(unsafe_site::SuppressionReason, BTreeMap<String, String>)> {
    let proof_scope = if site_local_proven {
        "site_local"
    } else {
        return None;
    };
    let mut evidence = BTreeMap::from([
        ("proof_scope".to_string(), proof_scope.to_string()),
        (
            "discharge_model".to_string(),
            discharge_model_name(model).to_string(),
        ),
        ("site_id".to_string(), site_id.to_string()),
    ]);
    if model == DischargeModel::ExactInitialization && !initialization_witnesses.is_empty() {
        evidence.insert(
            "initialization_witnesses".to_string(),
            initialization_witnesses
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    Some((unsafe_site::SuppressionReason::ContractDischarged, evidence))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnsafeResultSafety {
    ViolationWitness,
    ProvenSafe,
    Unknown(UnestablishedReason),
}

fn classify_utf8_input_safety(
    value: &ValueFacts,
    arg_count: usize,
    formal_capability: &BTreeMap<u32, Utf8FormalCapability>,
) -> UnsafeResultSafety {
    // An opaque call/transform may sanitize or replace the bytes. Provenance
    // alone is therefore not a counterexample and is not a safety proof.
    if value.havoced || value.semantic_unknown {
        return UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown);
    }

    let constants = value
        .value
        .origins
        .iter()
        .filter_map(|origin| match origin {
            AbstractOrigin::Constant(value) => Some(*value),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if !constants.is_empty()
        && value
            .value
            .origins
            .iter()
            .all(|origin| matches!(origin, AbstractOrigin::Constant(_)))
    {
        if constants.iter().all(|byte| *byte <= 0x7f) {
            return UnsafeResultSafety::ProvenSafe;
        }
        if constants.iter().any(|byte| matches!(*byte, 0xfe | 0xff)) {
            return UnsafeResultSafety::ViolationWitness;
        }
        return UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown);
    }

    let Some(root) = unique_semantic_value(value) else {
        return UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown);
    };
    let Some(index) = entry_formal_index(arg_count, &root) else {
        return UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown);
    };
    match formal_capability
        .get(&index)
        .copied()
        .unwrap_or(Utf8FormalCapability::Unknown)
    {
        Utf8FormalCapability::ValidUtf8 => UnsafeResultSafety::ProvenSafe,
        Utf8FormalCapability::UnrestrictedBytes => UnsafeResultSafety::ViolationWitness,
        Utf8FormalCapability::Unknown => {
            UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown)
        }
    }
}

fn from_utf8_input_safety(site_id: &str, facts: &BodyFacts) -> UnsafeResultSafety {
    let call = facts.calls.iter().find(|call| {
        facts
            .contract_sites_by_point
            .get(&call.point)
            .is_some_and(|sites| sites.contains(site_id))
    });
    let Some(input) = call.and_then(|call| call.arg_values.first()) else {
        return UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown);
    };
    classify_utf8_input_safety(input, facts.arg_count, &facts.argument_utf8_capability)
}

fn static_unsafe_result_safety(
    implication: StaticContractImplication,
    contract_model: Option<&str>,
) -> Option<UnsafeResultSafety> {
    match implication {
        StaticContractImplication::Proven(_) => Some(UnsafeResultSafety::ProvenSafe),
        // The type-level capability of `MaybeUninit<T>` is not a site-current
        // counterexample.  Its exact initialization state is evaluated after
        // typed writes/copies, helper postconditions, and FFI-out effects.
        StaticContractImplication::Unproven(
            StaticContractGap::SourceMayContainUninitializedBytes,
        ) if contract_model == Some(KnownContractModel::AssumeInit.as_str()) => None,
        StaticContractImplication::Unproven(
            StaticContractGap::SourceMayContainUninitializedBytes
            | StaticContractGap::TargetRestrictedValidity,
        ) => Some(UnsafeResultSafety::ViolationWitness),
        StaticContractImplication::Unproven(
            StaticContractGap::LifetimeOrProvenanceSensitive
            | StaticContractGap::StructuralValidityMismatch
            | StaticContractGap::UnknownTypeValidity,
        ) => Some(UnsafeResultSafety::Unknown(
            UnestablishedReason::ValidityObligationUnknown,
        )),
        StaticContractImplication::NotApplicable => None,
    }
}

fn unsafe_result_safety(
    candidate: &StructuralFindingCandidate,
    facts: &BodyFacts,
) -> UnsafeResultSafety {
    if candidate.classification_confidence != unsafe_site::ClassificationConfidence::Exact {
        return UnsafeResultSafety::Unknown(UnestablishedReason::MissingExactContext);
    }

    let contract_model = candidate.contract().model_id.as_deref();
    if let Some(safety) =
        static_unsafe_result_safety(candidate.contract().static_implication, contract_model)
    {
        return safety;
    }

    let site_id = candidate.site().seed.site_id();
    let relevant_roles = &candidate.contract().relevant_roles;
    let any_possibly_uninitialized = relevant_roles.iter().any(|role| {
        facts
            .possibly_uninitialized_site_operands
            .contains(&(site_id.clone(), role.clone()))
    });
    let all_definitely_initialized = !relevant_roles.is_empty()
        && relevant_roles.iter().all(|role| {
            facts
                .initialized_site_operands
                .contains(&(site_id.clone(), role.clone()))
        });

    match contract_model {
        Some(model) if model == KnownContractModel::AssumeInit.as_str() => {
            if any_possibly_uninitialized {
                UnsafeResultSafety::ViolationWitness
            } else if all_definitely_initialized {
                UnsafeResultSafety::ProvenSafe
            } else {
                UnsafeResultSafety::Unknown(UnestablishedReason::InitializationStateUnknown)
            }
        }
        Some(model)
            if model == KnownContractModel::FromUtf8Unchecked.as_str()
                || model == KnownContractModel::StringFromUtf8Unchecked.as_str() =>
        {
            // Exact validation was discharged before this gate. Remaining
            // sites are active only when the concrete sink operand is still
            // an unrestricted byte formal (or a concrete invalid sequence).
            // Mere absence of a proof is Unknown, not a violation witness.
            from_utf8_input_safety(&site_id, facts)
        }
        Some("language_validity_v1") => match candidate.site().type_validity.result {
            TypeValidityClass::RestrictedScalar
            | TypeValidityClass::RestrictedStructural
            | TypeValidityClass::RestrictedEnum => UnsafeResultSafety::ViolationWitness,
            TypeValidityClass::AllBitsValidScalar
            | TypeValidityClass::AllBitsValidBytes
            | TypeValidityClass::AllBitsValidStructural
            | TypeValidityClass::AllBitsValidRawPointer => UnsafeResultSafety::ProvenSafe,
            TypeValidityClass::ReferenceOrFunctionPointer
            | TypeValidityClass::UninitializedOrPadding
            | TypeValidityClass::Unknown => {
                UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown)
            }
        },
        Some(_) => UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown),
        None => UnsafeResultSafety::Unknown(UnestablishedReason::UnmodeledUnsafeCall),
    }
}

fn result_validity_requirement(contract: &ContractDescriptor) -> String {
    let model = contract.model_id.as_deref().unwrap_or("opaque");
    let predicates = contract
        .predicates
        .iter()
        .map(|predicate| predicate.as_str())
        .collect::<Vec<_>>()
        .join(",");
    format!("result_validity:model={model};predicates={predicates}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InternalDerivationSafety {
    ViolationWitness,
    ProvenSafe,
    Unknown(UnestablishedReason),
}

fn direct_condition_negation_witness(
    facts: &BodyFacts,
    sink_point: &ProgramPoint,
    derivation_token: &str,
) -> Option<bool> {
    let comparison = facts
        .compares
        .iter()
        .filter(|comparison| {
            comparison.result.0 == derivation_token && comparison.point < *sink_point
        })
        .max_by_key(|comparison| &comparison.point)?;
    let atom = |operand: &CallOperand, formal: Option<u32>| match operand {
        CallOperand::Constant(value) => Some((None, Some(*value))),
        CallOperand::Place(place) => formal
            .or_else(|| entry_formal_index(facts.arg_count, place))
            .map(|index| (Some(index), None)),
        CallOperand::ArrayLengths(_) | CallOperand::Unknown => None,
    };
    let left = atom(&comparison.left, comparison.left_formal)?;
    let right = atom(&comparison.right, comparison.right_formal)?;
    let same = left == right;

    // `Some(true)` means the negated condition has a concrete assignment;
    // `Some(false)` means equality of the exact terms makes it impossible.
    // Distinct constants need signedness/type information that this compact
    // domain intentionally does not guess.
    if left.1.is_some() && right.1.is_some() && !same {
        return None;
    }
    Some(match comparison.op {
        BinOp::Eq => !same,
        BinOp::Ne => true,
        BinOp::Lt | BinOp::Gt if same => true,
        BinOp::Le | BinOp::Ge if same => false,
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => left.0.is_some() && right.0.is_some(),
        _ => return None,
    })
}

fn finite_relation_negation_witness(
    snapshot: Option<&ValidationSnapshot>,
    binding: Option<&ValidationBinding>,
    relational: Option<&RelationalObligation>,
) -> bool {
    let Some(snapshot) = snapshot else {
        return false;
    };
    let proof = RelationProof::from_facts(snapshot.current_relations.iter().cloned());
    if let Some(binding) = binding {
        let subject = RelationTerm::Place(binding.subject.clone());
        if binding.predicate == Predicate::InBounds
            && binding.collection.as_ref().is_some_and(|collection| {
                proof.proves_le(&RelationTerm::Length(collection.clone()), &subject)
            })
        {
            return true;
        }
        if binding.predicate == Predicate::RangeInBounds {
            if let (Some(collection), Some(width)) =
                (binding.collection.as_ref(), binding.access_width)
            {
                let length = RelationTerm::Length(collection.clone());
                if proof.facts().any(|fact| {
                    matches!(fact.kind,
                        RelationKind::AddConst { ref result, ref base, addend }
                            if base == &subject && addend == width
                                && proof.proves_lt(&length, result))
                }) {
                    return true;
                }
            }
        }
    }
    relational.is_some_and(|obligation| match obligation {
        RelationalObligation::Within { base, start, end } => {
            proof.proves_lt(end, start) || proof.proves_lt(&RelationTerm::Length(base.clone()), end)
        }
    })
}

fn candidate_layout_obligations(
    candidate: &StructuralFindingCandidate,
    facts: &BodyFacts,
) -> Vec<ConcreteLayoutObligation> {
    let mut points = BTreeSet::new();
    let mut collect = |node: &EndpointNode| {
        if let EndpointNode::InternalDerivation {
            kind: DerivationKind::Layout,
            point,
            ..
        } = node
        {
            points.insert(point.clone());
        }
    };
    collect(candidate.source.as_ref());
    for edge in candidate.endpoint_route.iter() {
        collect(&edge.from);
        collect(&edge.to);
    }
    points
        .into_iter()
        .filter_map(|point| facts.concrete_layout_obligations.get(&point).copied())
        .collect()
}

fn concrete_layout_negation_witness(
    obligation: ConcreteLayoutObligation,
    predicates: &BTreeSet<super::contract_catalog::ContractPredicate>,
) -> Option<bool> {
    use super::contract_catalog::ContractPredicate;

    let size_relevant = predicates.contains(&ContractPredicate::ValidForRead)
        || predicates.contains(&ContractPredicate::RangeInBounds)
        || predicates.contains(&ContractPredicate::LayoutCompatible);
    let alignment_relevant = predicates.contains(&ContractPredicate::Aligned)
        || predicates.contains(&ContractPredicate::LayoutCompatible);
    let validity_relevant = predicates.contains(&ContractPredicate::ValidForType)
        || predicates.contains(&ContractPredicate::ValidForRead);
    if (size_relevant && obligation.size_violation)
        || (alignment_relevant && obligation.alignment_violation)
        || (validity_relevant
            && obligation.validity_violation
            && obligation.externally_controlled_bits)
    {
        Some(true)
    } else if validity_relevant
        && obligation.validity_violation
        && !obligation.externally_controlled_bits
    {
        None
    } else if obligation.complete {
        Some(false)
    } else {
        None
    }
}

fn internal_derivation_safety(
    candidate: &StructuralFindingCandidate,
    facts: &BodyFacts,
    bodies: &BTreeMap<FunctionKey, BodyFacts>,
) -> InternalDerivationSafety {
    use super::contract_catalog::ContractPredicate;

    if candidate.classification_confidence != unsafe_site::ClassificationConfidence::Exact {
        return InternalDerivationSafety::Unknown(UnestablishedReason::MissingExactContext);
    }
    // A concrete type mismatch is a sink witness only when the capability
    // source reads a source-local object of the compiler-known source type.
    // A formal/slice/iterator pointer may designate a larger valid region;
    // size/alignment mismatch alone is then not a precondition violation.
    if candidate.mechanism_id() == "p6_concrete_capability_violation_v1" {
        return match candidate.source.as_ref() {
            EndpointNode::GenericCapability {
                owner, evidence, ..
            } if bodies.get(owner).is_some_and(|source_facts| {
                source_facts
                    .source_local_generic_capabilities
                    .contains(&evidence.stable_identity())
            }) =>
            {
                InternalDerivationSafety::ViolationWitness
            }
            _ => {
                InternalDerivationSafety::Unknown(UnestablishedReason::DerivationObligationUnknown)
            }
        };
    }
    let EndpointNode::InternalDerivation {
        kind,
        token: derivation_token,
        ..
    } = candidate.source.as_ref()
    else {
        return InternalDerivationSafety::Unknown(UnestablishedReason::DerivationObligationUnknown);
    };
    let site_id = candidate.site().seed.site_id();
    let Some(point) = facts
        .contract_sites_by_point
        .iter()
        .find_map(|(point, sites)| sites.contains(&site_id).then_some(point))
    else {
        return InternalDerivationSafety::Unknown(UnestablishedReason::DerivationObligationUnknown);
    };
    let location = point.point_location();
    let snapshot = facts
        .validation_snapshots
        .get(&(location.block, location.statement_index));
    let binding = facts.local_bindings.get(point);
    let relational = facts.local_relational_obligations.get(point);

    if snapshot.is_some_and(|snapshot| local_requirement_is_proven(snapshot, binding, relational)) {
        return InternalDerivationSafety::ProvenSafe;
    }

    let predicates = &candidate.contract().predicates;
    let layout_obligations = candidate_layout_obligations(candidate, facts);
    if layout_obligations
        .iter()
        .copied()
        .any(|obligation| concrete_layout_negation_witness(obligation, predicates) == Some(true))
    {
        return InternalDerivationSafety::ViolationWitness;
    }
    if *kind == DerivationKind::Layout
        && !layout_obligations.is_empty()
        && layout_obligations.iter().copied().all(|obligation| {
            concrete_layout_negation_witness(obligation, predicates) == Some(false)
        })
    {
        return InternalDerivationSafety::ProvenSafe;
    }
    if *kind == DerivationKind::UncheckedBooleanRelation
        && predicates.contains(&ContractPredicate::ConditionTrue)
    {
        return match direct_condition_negation_witness(facts, point, derivation_token) {
            Some(true) => InternalDerivationSafety::ViolationWitness,
            Some(false) => InternalDerivationSafety::ProvenSafe,
            None => {
                InternalDerivationSafety::Unknown(UnestablishedReason::DerivationObligationUnknown)
            }
        };
    }

    let possibly_uninitialized = candidate.contract().relevant_roles.iter().any(|role| {
        facts
            .possibly_uninitialized_site_operands
            .contains(&(site_id.clone(), role.clone()))
    });
    if possibly_uninitialized && predicates.contains(&ContractPredicate::Initialized) {
        return InternalDerivationSafety::ViolationWitness;
    }

    let finite_bounds_domain = predicates.contains(&ContractPredicate::InBounds)
        || predicates.contains(&ContractPredicate::RangeInBounds);
    if finite_bounds_domain && finite_relation_negation_witness(snapshot, binding, relational) {
        return InternalDerivationSafety::ViolationWitness;
    }

    InternalDerivationSafety::Unknown(UnestablishedReason::DerivationObligationUnknown)
}

fn derivation_requirement(
    contract: &ContractDescriptor,
    role: &unsafe_site::OperandRole,
) -> String {
    let predicates = contract
        .predicates
        .iter()
        .map(|predicate| predicate.as_str())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "derivation:{};predicates={predicates}",
        match role {
            unsafe_site::OperandRole::Receiver => "receiver".to_string(),
            unsafe_site::OperandRole::Argument(index) => format!("argument:{index}"),
            unsafe_site::OperandRole::Place => "place".to_string(),
            unsafe_site::OperandRole::Address => "address".to_string(),
            unsafe_site::OperandRole::Index => "index".to_string(),
            unsafe_site::OperandRole::Length => "length".to_string(),
            unsafe_site::OperandRole::Size => "size".to_string(),
            unsafe_site::OperandRole::Alignment => "alignment".to_string(),
            unsafe_site::OperandRole::Layout => "layout".to_string(),
            unsafe_site::OperandRole::Discriminant => "discriminant".to_string(),
            unsafe_site::OperandRole::Out => "out".to_string(),
            unsafe_site::OperandRole::Callee => "callee".to_string(),
            unsafe_site::OperandRole::Other(index) => format!("other:{index}"),
        }
    )
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalValidationPolicy {
    Exportable,
    LocalOnly,
    Unsupported,
}

fn validation_contract_policy(predicate: Predicate) -> LocalValidationPolicy {
    match predicate {
        Predicate::InBounds | Predicate::NonEmpty | Predicate::NonNull => {
            LocalValidationPolicy::Exportable
        }
        Predicate::RangeInBounds | Predicate::ValidUtf8 => LocalValidationPolicy::LocalOnly,
        _ => LocalValidationPolicy::Unsupported,
    }
}

fn entry_formal_slot(
    arg_count: usize,
    state: &ProgramState,
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

fn unresolved_routes_for_requirement(
    owner: &FunctionKey,
    summary: &FunctionSummary,
    requirement: &ContractRequirement,
) -> BTreeSet<UnresolvedRouteNode> {
    let mut roots = BTreeSet::new();
    if summary.requirements.contains(requirement) {
        roots.insert(UnresolvedRouteNode {
            owner: owner.clone(),
            route: UnresolvedRouteFact::Hard(requirement.clone()),
        });
    }
    roots.extend(
        summary
            .validations
            .iter()
            .filter(|validation| validation.requirement == *requirement)
            .cloned()
            .map(|validation| UnresolvedRouteNode {
                owner: owner.clone(),
                route: UnresolvedRouteFact::Conditional(validation),
            }),
    );
    roots
}

fn build_unresolved_witness(
    root: FunctionKey,
    root_span: StableSpan,
    requirement: &ContractRequirement,
    route: &SelectedUnresolvedRoute,
    cycle: Option<&UnresolvedRouteCycle>,
) -> CanonicalWitness {
    let mut steps = vec![WitnessStep {
        kind: WitnessStepKind::Entry,
        function: root,
        span: root_span,
    }];
    let mut boundaries = BTreeSet::new();
    let mut emitted_cycle = false;
    if let Some(cycle) = cycle.filter(|cycle| cycle.nodes.contains(&route.root)) {
        steps.push(WitnessStep {
            kind: WitnessStepKind::SccCycle,
            function: cycle_step_function(&cycle.token),
            span: cycle.span.clone(),
        });
        emitted_cycle = true;
    }
    for route_step in &route.steps {
        let callee = route_step.edge.callee.clone();
        boundaries.insert(callee.clone());
        steps.push(WitnessStep {
            kind: WitnessStepKind::LocalCall,
            function: callee.clone(),
            span: route_step.edge.call_point.span.clone(),
        });
        let callee_node = UnresolvedRouteNode {
            owner: callee,
            route: route_step.edge.callee_route.clone(),
        };
        if !emitted_cycle && cycle.is_some_and(|cycle| cycle.nodes.contains(&callee_node)) {
            let cycle = cycle.expect("cycle evidence was just matched");
            steps.push(WitnessStep {
                kind: WitnessStepKind::SccCycle,
                function: cycle_step_function(&cycle.token),
                span: cycle.span.clone(),
            });
            emitted_cycle = true;
        }
    }
    assert!(
        cycle.is_none() || emitted_cycle,
        "route cycle must intersect the selected route"
    );
    steps.push(WitnessStep {
        kind: if requirement.sink.kind == OperationKind::InvalidValueExposure {
            WitnessStepKind::Exposure
        } else {
            WitnessStepKind::Sink
        },
        function: requirement.sink_function.clone(),
        span: requirement.sink.point.span.clone(),
    });
    CanonicalWitness::new(steps, boundaries.into_iter().collect())
}

fn propagate_unresolved_route(
    derived: &mut FunctionSummary,
    caller: &FunctionKey,
    call: &CallSite,
    callee: &FunctionKey,
    caller_route: UnresolvedRouteFact,
    callee_route: UnresolvedRouteFact,
) {
    if call.disposition != BoundaryDisposition::ResolvedLocal {
        return;
    }
    assert_eq!(
        call.point.function, *caller,
        "an unresolved predecessor must be owned by the caller summary"
    );
    match &caller_route {
        UnresolvedRouteFact::Hard(requirement) => {
            derived.requirements.insert(requirement.clone());
        }
        UnresolvedRouteFact::Conditional(validation) => {
            derived.validations.insert(validation.clone());
        }
    }
    derived
        .unresolved_predecessors
        .insert(UnresolvedPredecessor {
            caller_route,
            callee_route,
            callee: callee.clone(),
            call_point: call.point.clone(),
        });
}

fn propagate_validation_route(
    derived: &mut FunctionSummary,
    caller: &FunctionKey,
    call: &CallSite,
    callee: &FunctionKey,
    callee_route: UnresolvedRouteFact,
    route: ValidationRoute,
) {
    let caller_route = match route {
        ValidationRoute::Satisfied => return,
        ValidationRoute::Conditional(validation) => UnresolvedRouteFact::Conditional(validation),
        ValidationRoute::Hard(requirement) => UnresolvedRouteFact::Hard(requirement),
    };
    propagate_unresolved_route(derived, caller, call, callee, caller_route, callee_route);
}

fn compose_validation_route(
    validation: &ValidationFact,
    call: &CallSite,
    state: &ProgramState,
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

fn compose_validation_route_from_snapshot(
    validation: &ValidationFact,
    call: &CallSite,
    snapshot: &ValidationSnapshot,
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
    if snapshot.proves(&binding) {
        return ValidationRoute::Satisfied;
    }
    let subject = snapshot.entry_formal(&binding.subject);
    let collection = match &binding.collection {
        Some(place) => match snapshot.entry_formal(place) {
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
        access_width: None,
        proof_definition: None,
        subject,
        collection,
    })
}

fn local_call_boundary_mapping(call: &CallModel, state: &ProgramState) -> BTreeSet<CallMapping> {
    let mut mapping = BTreeSet::new();
    for (index, operand) in call.descriptor.args.iter().enumerate() {
        if let Some(actual) = operand.operand.place() {
            mapping.insert(CallMapping::ActualToFormal {
                actual: actual.clone(),
                formal_index: index as u32 + 1,
            });
        }
    }
    mapping.insert(CallMapping::ReturnToDestination {
        destination: call.descriptor.destination.clone(),
    });
    if call.descriptor.disposition != BoundaryDisposition::ResolvedLocal {
        return mapping;
    }
    for (index, operand) in call.descriptor.args.iter().enumerate() {
        let Some(raw_actual) = operand.operand.place() else {
            continue;
        };
        if !call.mutable_actuals.contains(raw_actual) {
            continue;
        }
        let Some(caller_place) = unique_semantic_value(&eval_operand(state, operand)) else {
            continue;
        };
        mapping.insert(CallMapping::OutToCaller {
            formal_index: index as u32 + 1,
            caller_place,
        });
    }
    mapping
}

fn apply_call_side_effects(call: &CallModel, point: &ProgramPoint, state: &mut ProgramState) {
    let token = StaticWriteToken::new(point.block, point.statement);

    // A mutable pointer derived from tracked MaybeUninit storage is itself an
    // initialization boundary. Unless a typed write/copy model below proves a
    // postcondition, the call may initialize, partially initialize, or leave
    // that storage untouched. Keeping the pre-call `uninit` fact would invent
    // a violation witness after helpers such as a typed store.
    if call.typed_initialization_effect.is_none()
        && call.descriptor.disposition == BoundaryDisposition::ResolvedLocal
    {
        for operand in &call.descriptor.args {
            let (targets, _) = initialization_pointer_targets(state, operand);
            if !targets.is_empty() {
                forget_typed_initialization_targets(&targets, token, state);
            }
        }
    }
    for actual in &call.mutable_actuals {
        for target in state_alias_closure(state, actual) {
            let mut havoced = state_value(state, &target);
            havoced.havoced = true;
            havoced
                .influence_seeds
                .mark_all_incomplete(UnestablishedReason::HavocedProvenance);
            state.record_may_write(target.clone(), token);
            state.set_value(target.clone(), havoced);
            invalidate_relation_evidence(state, &target);
        }
    }
    for actual in &call.ffi_out_actuals {
        for target in state_alias_closure(state, &actual.place) {
            let mut output = state_value(state, &target);
            let was_definitely_uninitialized =
                output.maybe_uninit_initialized == BTreeSet::from([false]) && !output.havoced;
            output.exact_roots.insert(target.clone());
            output.ffi_output = true;
            let endpoint = EndpointNode::ExternalBehavior {
                owner: point.function.clone(),
                kind: ExternalSourceKind::FfiOut,
                controller: BehaviorController::FfiHost,
                token: format!("ffi-out:{}", point.span.token()),
                span: point.span.clone(),
            };
            Arc::make_mut(&mut output.endpoint_sources).insert(endpoint.clone());
            insert_external_influence_seed(
                &mut output,
                endpoint,
                ExternalSourceKind::FfiOut,
                &BehaviorController::FfiHost,
                actual.type_class,
            );
            output.value.origins.insert(AbstractOrigin::FfiOutput {
                token: format!("ffi-out:{}", point.span.token()),
                span: point.span.clone(),
            });
            if was_definitely_uninitialized {
                output.value.origins.insert(AbstractOrigin::FfiOutput {
                    token: format!("{FFI_UNINITIALIZED_OUTPUT_PREFIX}{}", point.span.token()),
                    span: point.span.clone(),
                });
            }
            state.record_may_write(target.clone(), token);
            state.set_value(target.clone(), output);
            invalidate_relation_evidence(state, &target);
        }
    }
    if let Some(effect) = &call.typed_initialization_effect {
        apply_typed_initialization_effect(effect, point, token, state);
    }
}

fn initialization_pointer_targets(
    state: &ProgramState,
    pointer: &OperandModel,
) -> (BTreeSet<PlaceKey>, bool) {
    let value = eval_operand(state, pointer);
    let mut targets = value.whole_init_targets.clone();
    let mut target_unknown = value.whole_init_target_unknown;
    let referent_storages = targets
        .iter()
        .map(canonical_storage)
        .collect::<BTreeSet<_>>();
    if let Some(raw_pointer) = pointer.operand.place() {
        for alias in state_alias_closure(state, raw_pointer) {
            // The pointee is connected by value-flow provenance, but its own
            // ValueFacts do not describe another possible pointer value.
            if referent_storages.contains(&alias) {
                continue;
            }
            if let Some(alias_value) = state.may_values().get(&alias) {
                // `value_flow` also connects ordinary receiver/dependency
                // temporaries. Only facts carrying explicit whole-target
                // provenance describe another possible pointer value.
                if !alias_value.whole_init_target_tracked
                    && alias_value.whole_init_targets.is_empty()
                {
                    continue;
                }
                target_unknown |=
                    alias_value.whole_init_target_unknown || !alias_value.whole_init_target_tracked;
                targets.extend(alias_value.whole_init_targets.iter().cloned());
            }
        }
    }
    let exact = value.whole_init_target_tracked
        && !target_unknown
        && targets.len() == 1
        && targets
            .iter()
            .all(|target| canonical_storage(target) == *target);
    let targets = targets
        .into_iter()
        .map(|target| canonical_storage(&target))
        .collect::<BTreeSet<_>>();
    (targets, exact)
}

fn write_typed_initialization_state(
    target: &PlaceKey,
    initialization: TypedInitializationState,
    mut witnesses: BTreeSet<InitializationWitness>,
    witness: InitializationWitness,
    token: StaticWriteToken,
    state: &mut ProgramState,
) {
    let storage = canonical_storage(target);
    let mut value = state_value(state, target);
    witnesses.insert(witness);
    value.initialization_witnesses = witnesses;
    match initialization {
        TypedInitializationState::Initialized => {
            value.maybe_uninit_initialized = BTreeSet::from([true]);
            value.havoced = false;
            value.ffi_output = false;
            value
                .value
                .origins
                .retain(|origin| !is_uninitialized_ffi_output_origin(origin));
        }
        TypedInitializationState::Uninitialized => {
            value.maybe_uninit_initialized = BTreeSet::from([false]);
            value.havoced = false;
            value.ffi_output = false;
        }
        TypedInitializationState::Unknown => {
            value.maybe_uninit_initialized.clear();
            value.havoced = true;
        }
    }
    state.record_write(storage.clone(), token);
    invalidate_relation_evidence(state, &storage);
    state.set_value(target.clone(), value);
}

fn havoc_typed_initialization_targets(
    targets: &BTreeSet<PlaceKey>,
    token: StaticWriteToken,
    state: &mut ProgramState,
) {
    for target in targets {
        let storage = canonical_storage(target);
        let mut unknown = state_value(state, &target);
        unknown.havoced = true;
        unknown.initialization_witnesses.clear();
        state.record_may_write(storage.clone(), token);
        invalidate_relation_evidence(state, &storage);
        state.set_value(target.clone(), unknown);
    }
}

/// A resolved helper without an initialization postcondition invalidates both
/// definite states. Unlike an ambiguous typed write/copy, there is no modeled
/// branch on which the pre-call `uninit` state is known to survive, so keeping
/// that bit would manufacture a violation witness.
fn forget_typed_initialization_targets(
    targets: &BTreeSet<PlaceKey>,
    token: StaticWriteToken,
    state: &mut ProgramState,
) {
    havoc_typed_initialization_targets(targets, token, state);
    for target in targets {
        let mut unknown = state_value(state, target);
        unknown.maybe_uninit_initialized.clear();
        state.set_value(target.clone(), unknown);
    }
}

fn apply_typed_initialization_effect(
    effect: &TypedInitializationEffect,
    point: &ProgramPoint,
    token: StaticWriteToken,
    state: &mut ProgramState,
) {
    match effect {
        TypedInitializationEffect::MaybeUninitWrite { destination } => {
            let receiver = eval_operand(state, destination);
            if let Some(target) = unique_semantic_value(&receiver) {
                write_typed_initialization_state(
                    &target,
                    TypedInitializationState::Initialized,
                    BTreeSet::new(),
                    InitializationWitness {
                        kind: InitializationWitnessKind::MaybeUninitWrite,
                        point: point.clone(),
                    },
                    token,
                    state,
                );
                return;
            }
            let targets = destination
                .operand
                .place()
                .map(|raw| state_alias_closure(state, raw))
                .unwrap_or_default();
            havoc_typed_initialization_targets(&targets, token, state);
        }
        TypedInitializationEffect::PointerWrite {
            destination,
            source,
        } => {
            let (targets, exact) = initialization_pointer_targets(state, destination);
            if exact {
                let target = targets.iter().next().expect("one exact target");
                let source_witnesses = eval_operand(state, source).initialization_witnesses;
                write_typed_initialization_state(
                    target,
                    TypedInitializationState::Initialized,
                    source_witnesses,
                    InitializationWitness {
                        kind: InitializationWitnessKind::PointerWrite,
                        point: point.clone(),
                    },
                    token,
                    state,
                );
            } else {
                havoc_typed_initialization_targets(&targets, token, state);
            }
        }
        TypedInitializationEffect::PointerCopy {
            source,
            destination,
            count,
            nonoverlapping,
        } => {
            if matches!(count, CallOperand::Constant(0)) {
                return;
            }
            let (source_targets, source_exact) = initialization_pointer_targets(state, source);
            let (destination_targets, destination_exact) =
                initialization_pointer_targets(state, destination);
            let source_may_be_uninitialized = source_targets.iter().any(|target| {
                state_value(state, target)
                    .maybe_uninit_initialized
                    .contains(&false)
            });
            if !matches!(count, CallOperand::Constant(1)) || !destination_exact {
                havoc_typed_initialization_targets(&destination_targets, token, state);
                if source_may_be_uninitialized {
                    for target in &destination_targets {
                        let mut value = state_value(state, target);
                        value.maybe_uninit_initialized.insert(false);
                        state.set_value(target.clone(), value);
                    }
                }
                return;
            }

            // Snapshot before writing: ptr::copy explicitly permits overlap,
            // and even for copy_nonoverlapping the unsafe precondition must not
            // be used as a postcondition proof.
            let source_snapshot = source_exact.then(|| {
                let source_target = source_targets.iter().next().expect("one exact source");
                let value = state_value(state, source_target);
                (
                    typed_initialization_state(&value),
                    value.initialization_witnesses,
                )
            });
            let destination_target = destination_targets
                .iter()
                .next()
                .expect("one exact destination");
            let Some((initialization, witnesses)) = source_snapshot else {
                write_typed_initialization_state(
                    destination_target,
                    TypedInitializationState::Unknown,
                    BTreeSet::new(),
                    InitializationWitness {
                        kind: if *nonoverlapping {
                            InitializationWitnessKind::PointerCopyNonoverlapping
                        } else {
                            InitializationWitnessKind::PointerCopy
                        },
                        point: point.clone(),
                    },
                    token,
                    state,
                );
                return;
            };
            write_typed_initialization_state(
                destination_target,
                initialization,
                witnesses,
                InitializationWitness {
                    kind: if *nonoverlapping {
                        InitializationWitnessKind::PointerCopyNonoverlapping
                    } else {
                        InitializationWitnessKind::PointerCopy
                    },
                    point: point.clone(),
                },
                token,
                state,
            );
        }
        TypedInitializationEffect::OpaqueIntrinsicMutation { destinations } => {
            for destination in destinations {
                let (targets, _) = initialization_pointer_targets(state, destination);
                forget_typed_initialization_targets(&targets, token, state);
            }
        }
    }
}

fn apply_call_return(
    call: &CallModel,
    point: &ProgramPoint,
    mut value: ValueFacts,
    state: &mut ProgramState,
) {
    let token = StaticWriteToken::new(point.block, point.statement);
    let destination = call.descriptor.destination.clone();
    value.exact_roots.clear();
    value.exact_roots.insert(destination.clone());
    value.semantic_unknown = false;
    let storage = canonical_storage(&destination);
    state.record_write(storage.clone(), token);
    invalidate_relation_evidence(state, &storage);
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
                access_width: None,
                proof_definition: None,
                subject: slice,
                collection: None,
            };
            vec![(binding.clone(), false, binding.storage_places())]
        }
        Some(PredicateModel::IsNull { pointer })
            if unique_semantic_value(&semantic_value_at(state, pointer)).is_some() =>
        {
            let pointer = unique_semantic_value(&semantic_value_at(state, pointer))
                .expect("the match guard established one semantic place");
            let binding = ValidationBinding {
                predicate: Predicate::NonNull,
                access_width: None,
                proof_definition: None,
                subject: pointer,
                collection: None,
            };
            vec![(binding.clone(), false, binding.storage_places())]
        }
        Some(PredicateModel::IsErr { checked }) => {
            let checked = eval_operand(state, checked);
            current_utf8_bytes(state, &checked)
                .map(|bytes| {
                    let binding = ValidationBinding {
                        predicate: Predicate::ValidUtf8,
                        access_width: None,
                        proof_definition: None,
                        subject: bytes,
                        collection: None,
                    };
                    vec![(binding.clone(), false, binding.storage_places())]
                })
                .unwrap_or_default()
        }
        _ => Vec::new(),
    };
    if !pending.is_empty() {
        state.set_pending(destination, pending);
    }
}

fn predicate_result(predicate: &PredicateFact) -> &PlaceKey {
    match predicate {
        PredicateFact::IsEmpty { result, .. } | PredicateFact::IsNull { result, .. } => result,
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
        if matches!(inner.kind(), ty::RawPtr(_, Mutability::Mut))
            || has_restricted_validity(*inner))
}

fn ffi_out_value_type_class(ty: ty::Ty<'_>) -> InfluenceTypeClass {
    match ty.kind() {
        ty::RawPtr(inner, Mutability::Mut) => influence_type_class_for_ty(*inner),
        _ => InfluenceTypeClass::Opaque,
    }
}

fn has_restricted_validity(ty: ty::Ty<'_>) -> bool {
    match ty.kind() {
        ty::Bool | ty::Char => true,
        ty::Adt(definition, _) => definition.is_enum(),
        _ => false,
    }
}

fn is_mutable_call_actual(ty: ty::Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        ty::Ref(_, _, Mutability::Mut) | ty::RawPtr(_, Mutability::Mut)
    )
}

fn out_formal_index(body: &Body<'_>, destination: Place<'_>) -> Option<u32> {
    if !matches!(destination.projection.as_ref(), [ProjectionElem::Deref]) {
        return None;
    }
    let index = destination.local.index();
    (index > 0
        && index <= body.arg_count
        && is_mutable_call_actual(body.local_decls[destination.local].ty))
    .then_some(index as u32)
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
        RuleId::P5LifetimeTransmute
            | RuleId::P5AssumeInitBool
            | RuleId::P5AssumeInitUninitialized
            | RuleId::P5UncheckedUtf8
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

fn origin_matches_requirement_source(
    requirement: &ContractRequirement,
    origin: &AbstractOrigin,
    kind: SourceKind,
) -> bool {
    origin_matches_kind(origin, kind)
        && (requirement.rule != RuleId::P3FfiOutInvalidValue
            || is_uninitialized_ffi_output_origin(origin))
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
    use super::super::summary::scc_cycle_token;
    use super::*;

    #[test]
    fn context_summary_exports_structural_sources_but_not_terminal_chains() {
        let owner = FunctionKey::new("crate::wrapper");
        let structural = EndpointNode::UnsafeResult {
            owner: FunctionKey::new("crate::inner"),
            site_id: "site".into(),
        };
        assert!(is_context_summary_source(&structural, &owner));
        assert!(is_context_summary_source(
            &EndpointNode::CallResult {
                callsite: CallSiteId::new(
                    owner.clone(),
                    ProgramPoint {
                        function: owner.clone(),
                        block: 0,
                        statement: 0,
                        span: StableSpan::new(
                            "src/lib.rs",
                            StablePosition::new(1, 1),
                            StablePosition::new(1, 1),
                        ),
                    },
                    FunctionKey::new("crate::inner"),
                ),
                output: CallSiteOutput::Return,
            },
            &owner,
        ));
        assert!(!is_context_summary_source(
            &EndpointNode::Return {
                owner: owner.clone(),
            },
            &owner,
        ));
        assert!(!is_context_summary_source(
            &EndpointNode::Exposure {
                owner,
                token: "exposure".into(),
                point: ProgramPoint {
                    function: FunctionKey::new("crate::wrapper"),
                    block: 0,
                    statement: 0,
                    span: StableSpan::new(
                        "src/lib.rs",
                        StablePosition::new(1, 1),
                        StablePosition::new(1, 1),
                    ),
                },
            },
            &FunctionKey::new("crate::wrapper"),
        ));
    }

    #[test]
    fn static_type_implication_is_a_typed_suppression_but_gaps_stay_active() {
        use super::super::contract_catalog::{StaticContractEvidence, StaticContractGap};

        let (reason, evidence) = static_contract_suppression(StaticContractImplication::Proven(
            StaticContractEvidence::TargetAllBitsValidScalar,
        ))
        .unwrap();
        assert_eq!(
            reason,
            unsafe_site::SuppressionReason::ContractImpliedByType
        );
        assert_eq!(
            evidence.get("static_implication").map(String::as_str),
            Some("target_all_bits_valid_scalar")
        );
        assert!(!evidence.is_empty());
        assert!(
            static_contract_suppression(StaticContractImplication::Unproven(
                StaticContractGap::TargetRestrictedValidity,
            ))
            .is_none()
        );
        assert!(static_contract_suppression(StaticContractImplication::NotApplicable).is_none());
    }

    #[test]
    fn only_explicit_site_local_proofs_are_ledgered() {
        let initialization_witnesses = BTreeSet::from([
            "ptr_copy@src/lib.rs:3:1-3:8".to_string(),
            "ptr_write@src/lib.rs:2:1-2:8".to_string(),
        ]);
        let (local_reason, local_evidence) = dynamic_contract_suppression(
            DischargeModel::ExactInitialization,
            "site-local",
            true,
            &initialization_witnesses,
        )
        .unwrap();
        assert_eq!(
            local_reason,
            unsafe_site::SuppressionReason::ContractDischarged
        );
        assert_eq!(local_evidence["proof_scope"], "site_local");
        assert_eq!(local_evidence["discharge_model"], "exact_initialization");
        assert_eq!(
            local_evidence["initialization_witnesses"],
            "ptr_copy@src/lib.rs:3:1-3:8,ptr_write@src/lib.rs:2:1-2:8"
        );

        assert!(dynamic_contract_suppression(
            DischargeModel::ExactRelation,
            "site-summary",
            false,
            &BTreeSet::new(),
        )
        .is_none());
        assert!(dynamic_contract_suppression(
            DischargeModel::ExactRelation,
            "site-unresolved",
            false,
            &BTreeSet::new(),
        )
        .is_none());
        assert!(dynamic_contract_suppression(
            DischargeModel::None,
            "site-opaque",
            false,
            &BTreeSet::new(),
        )
        .is_none());
    }

    #[test]
    fn public_field_store_endpoint_edge_seam_keeps_graph_nodes_closed() {
        let owner = FunctionKey::new("crate::public_field_store");
        let source = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let target = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Layout,
            token: "stored-derived".to_string(),
            point: ProgramPoint {
                function: owner,
                block: 2,
                statement: 0,
                span: span(),
            },
        };
        let edges = BTreeSet::from([EndpointEdge {
            from: source.clone(),
            to: target.clone(),
            kind: EndpointEdgeKind::Derivation,
            point: None,
        }]);
        let mut graph = EndpointFlowGraph::default();
        append_value_endpoint_edges(&mut graph, &edges);

        assert_eq!(graph.edges, edges);
        assert!(graph.nodes.contains(&source));
        assert!(graph.nodes.contains(&target));
        assert!(graph
            .edges
            .iter()
            .all(|edge| graph.nodes.contains(&edge.from) && graph.nodes.contains(&edge.to)));
    }

    #[test]
    fn source_local_capability_requires_positive_local_storage_evidence() {
        let untracked = ValueFacts::default();
        assert!(!establishes_source_local_generic_capability(&untracked));

        let local = ValueFacts {
            storage_provenance: StorageProvenance::DefinitelyLocal,
            ..ValueFacts::default()
        };
        assert!(establishes_source_local_generic_capability(&local));

        let formal = ValueFacts {
            storage_provenance: StorageProvenance::EntryFormal {
                owner: FunctionKey::new("crate::wrapper"),
                index: 1,
            },
            ..ValueFacts::default()
        };
        assert!(!establishes_source_local_generic_capability(&formal));
    }

    #[test]
    fn safe_reference_storage_does_not_grant_external_validity_bits() {
        let owner = FunctionKey::new("crate::wrapper");
        let endpoint = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let storage_only = ValueFacts {
            storage_provenance: StorageProvenance::EntryFormal {
                owner: owner.clone(),
                index: 1,
            },
            ..ValueFacts::default()
        };
        assert!(!has_externally_controlled_validity_bits(&storage_only));

        let mut raw_pointer = storage_only.clone();
        raw_pointer.influence_seeds.insert(
            endpoint,
            InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal { owner, index: 1 },
                InfluenceTypeClass::Pointer,
            ),
        );
        assert!(has_externally_controlled_validity_bits(&raw_pointer));
    }

    #[test]
    fn assume_init_type_capability_defers_to_site_current_initialization() {
        assert_eq!(
            static_unsafe_result_safety(
                StaticContractImplication::Unproven(
                    StaticContractGap::SourceMayContainUninitializedBytes,
                ),
                Some(KnownContractModel::AssumeInit.as_str()),
            ),
            None
        );
        assert_eq!(
            static_unsafe_result_safety(
                StaticContractImplication::Unproven(
                    StaticContractGap::SourceMayContainUninitializedBytes,
                ),
                Some("other_model"),
            ),
            Some(UnsafeResultSafety::ViolationWitness)
        );
    }

    #[test]
    fn literal_index_discharge_requires_every_known_array_length_to_cover_the_index() {
        let fixed = ValueFacts {
            known_lengths: BTreeSet::from([1]),
            ..ValueFacts::default()
        };
        assert!(literal_index_within_known_lengths(0, &fixed));
        assert!(!literal_index_within_known_lengths(1, &fixed));

        let joined_with_empty = ValueFacts {
            known_lengths: BTreeSet::from([0, 1]),
            ..ValueFacts::default()
        };
        assert!(!literal_index_within_known_lengths(0, &joined_with_empty));

        let unknown = ValueFacts {
            known_lengths: BTreeSet::from([1]),
            known_length_unknown: true,
            ..ValueFacts::default()
        };
        assert!(!literal_index_within_known_lengths(0, &unknown));
    }

    #[test]
    fn binding_record_replay_is_idempotent_but_conflicts_fail_closed() {
        let record = SiteBindingRecord {
            site_id: "site-1".to_string(),
            status: BindingStatus::Unbound,
            built_candidate_count: 0,
            solver_candidate_count: 0,
            built: None,
            solver: None,
            solver_event: None,
        };
        let mut records = BTreeMap::new();
        assert!(insert_binding_record(&mut records, record.clone()).unwrap());
        assert!(!insert_binding_record(&mut records, record.clone()).unwrap());

        let mut conflict = record;
        conflict.solver_candidate_count = 1;
        assert!(insert_binding_record(&mut records, conflict)
            .unwrap_err()
            .contains("conflicting unsafe-site binding record"));
    }

    #[rustversion::since(1.96)]
    #[test]
    fn current_comptime_functions_are_not_sent_to_optimized_mir() {
        assert!(supports_optimized_mir_body(DefKind::Fn, None,));
        assert!(supports_optimized_mir_body(
            DefKind::AssocFn,
            Some(ConstContext::ConstFn),
        ));
        assert!(!supports_optimized_mir_body(
            DefKind::AssocFn,
            Some(ConstContext::Const {
                allow_const_fn_promotion: false,
            }),
        ));
        assert!(uses_ctfe_mir(DefKind::Const {
            is_type_const: false
        }));
        assert!(!uses_ctfe_mir(DefKind::Fn));
    }

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
            projection_path: None,
            range_index_kind: None,
            projection_unknown: false,
        }
    }

    fn test_generic_capability(kind: GenericCapabilityKind) -> GenericCapabilityEvidence {
        let parameter = GenericParamId {
            def_path: "crate::f::T".to_string(),
            kind: GenericParamKind::Type,
        };
        GenericCapabilityEvidence {
            kind,
            terms: BTreeMap::from([(
                CapabilityTermRole::Sequence,
                CapabilityTerm {
                    form: CapabilityTermForm::Alias,
                    controllers: BTreeSet::from([parameter]),
                    residual_aliases: BTreeSet::from(["crate::Provider::Results".to_string()]),
                    equality_class: None,
                    type_facts: CapabilityTypeFacts::default(),
                },
            )]),
            requirement: CapabilityRequirement::AssociatedSequenceNonEmpty,
            witness: CapabilityWitness::Proven {
                assignment: BTreeMap::from([(
                    CapabilityTermRole::Sequence,
                    FiniteTypeWitness::EmptyArray,
                )]),
                violations: BTreeSet::from([CapabilityViolation::InBounds]),
            },
        }
    }

    fn constant_operand(value: u64) -> OperandModel {
        OperandModel {
            operand: CallOperand::Constant(value),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: None,
            projection_path: None,
            range_index_kind: None,
            projection_unknown: false,
        }
    }

    #[test]
    fn closure_capture_projection_propagates_to_later_unsafe_operand() {
        let owner = FunctionKey::new("crate::outer::{closure#0}");
        let mut capture_point = point(0, 0);
        capture_point.function = owner.clone();
        let capture = ProgramOp {
            point: Some(capture_point),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: PlaceKey::new("_4"),
                destination_projection: None,
                value: ValueModel::Operand(place_operand("(_1.0: &*mut u8)")),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
            unsafe_sites: Vec::new(),
        };
        let mut state = ProgramState::empty();
        apply_program_op_with_context(&capture, BasicBlock::from_usize(0), 0, &mut state, true);
        let expected = EndpointNode::Capture {
            owner: owner.clone(),
            index: 0,
        };
        assert!(state_value(&state, &PlaceKey::new("_4"))
            .endpoint_sources
            .contains(&expected));

        let mut copy_point = point(0, 1);
        copy_point.function = owner;
        let copy = ProgramOp {
            point: Some(copy_point),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: PlaceKey::new("_3"),
                destination_projection: None,
                value: ValueModel::Operand(place_operand("(*_4)")),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
            unsafe_sites: Vec::new(),
        };
        apply_program_op_with_context(&copy, BasicBlock::from_usize(0), 1, &mut state, true);
        assert!(state_value(&state, &PlaceKey::new("_3"))
            .endpoint_sources
            .contains(&expected));
    }

    fn exact_value(place: &PlaceKey) -> ValueFacts {
        ValueFacts {
            exact_roots: BTreeSet::from([place.clone()]),
            utf8_relation_unknown: true,
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        }
    }

    fn seed_exact(state: &mut ProgramState, place: &PlaceKey) {
        state.set_value(place.clone(), exact_value(place));
    }

    #[test]
    fn typed_initialization_state_keeps_uncertainty_active() {
        assert_eq!(
            typed_initialization_state(&ValueFacts {
                maybe_uninit_initialized: BTreeSet::from([true]),
                ..ValueFacts::default()
            }),
            TypedInitializationState::Initialized
        );
        assert_eq!(
            typed_initialization_state(&ValueFacts {
                maybe_uninit_initialized: BTreeSet::from([false]),
                ..ValueFacts::default()
            }),
            TypedInitializationState::Uninitialized
        );
        assert_eq!(
            typed_initialization_state(&ValueFacts {
                maybe_uninit_initialized: BTreeSet::from([true]),
                havoced: true,
                ..ValueFacts::default()
            }),
            TypedInitializationState::Unknown
        );
        assert_eq!(
            typed_initialization_state(&ValueFacts::default()),
            TypedInitializationState::Unknown
        );
    }

    fn typed_initialization_call(receiver: OperandModel) -> CallModel {
        CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![receiver.clone()],
                destination: PlaceKey::new("_9"),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: Some(TypedInitializationEffect::MaybeUninitWrite {
                destination: receiver,
            }),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        }
    }

    fn typed_effect_call(args: Vec<OperandModel>, effect: TypedInitializationEffect) -> CallModel {
        CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args,
                destination: PlaceKey::new("_99"),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: Some(effect),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        }
    }

    fn typed_storage(initialization: TypedInitializationState) -> ValueFacts {
        ValueFacts {
            maybe_uninit_initialized: match initialization {
                TypedInitializationState::Initialized => BTreeSet::from([true]),
                TypedInitializationState::Uninitialized => BTreeSet::from([false]),
                TypedInitializationState::Unknown => BTreeSet::new(),
            },
            havoced: initialization == TypedInitializationState::Unknown,
            ..ValueFacts::default()
        }
    }

    fn whole_initialization_pointer(targets: impl IntoIterator<Item = PlaceKey>) -> ValueFacts {
        ValueFacts {
            whole_init_targets: targets.into_iter().collect(),
            whole_init_target_tracked: true,
            ..ValueFacts::default()
        }
    }

    fn pointer_write_call(pointer: &PlaceKey, source: &PlaceKey) -> CallModel {
        let destination = place_operand(&pointer.0);
        let source = place_operand(&source.0);
        typed_effect_call(
            vec![destination.clone(), source.clone()],
            TypedInitializationEffect::PointerWrite {
                destination,
                source,
            },
        )
    }

    fn pointer_copy_call(
        source: &PlaceKey,
        destination: &PlaceKey,
        count: CallOperand,
        nonoverlapping: bool,
    ) -> CallModel {
        let source = place_operand(&source.0);
        let destination = place_operand(&destination.0);
        typed_effect_call(
            vec![source.clone(), destination.clone()],
            TypedInitializationEffect::PointerCopy {
                source,
                destination,
                count,
                nonoverlapping,
            },
        )
    }

    fn endpoint_value(source: EndpointNode) -> ValueFacts {
        ValueFacts {
            endpoint_sources: Arc::new(BTreeSet::from([source])),
            ..ValueFacts::default()
        }
    }

    fn bound_endpoint_site(place: &str) -> BoundUnsafeSite {
        let location = MirEventLocation {
            stage: MirStage::Solver,
            block: 0,
            statement: 0,
            event: 0,
        };
        BoundUnsafeSite {
            seed: unsafe_site::UnsafeSiteSeed {
                adapter_id: "test-adapter".into(),
                owner: FunctionKey::new("crate::f"),
                owner_def_path_hash: "owner".into(),
                ordinal: 0,
                kind: unsafe_site::UnsafeSiteKind::RawPointerDereference,
                raw_span: unsafe_site::RawSpanAnchor::new(span(), 0, 1, "#0"),
                hir_context: "HirId(1)".into(),
                callee: None,
                operands: Vec::new(),
                result: None,
            },
            built: MirEventLocation {
                stage: MirStage::Built,
                ..location.clone()
            },
            solver: location,
            operands: vec![MirOperandAnchor {
                role: unsafe_site::OperandRole::Address,
                type_fingerprint: "*const u8".into(),
                place: Some(PlaceKey::new(place)),
            }],
            result_place: Some(PlaceKey::new("_2")),
            type_validity: MirTypeValidity::default(),
        }
    }

    fn bound_condition_true_site(condition: &PlaceKey) -> BoundUnsafeSite {
        let mut site = bound_endpoint_site(&condition.0);
        site.seed.kind = unsafe_site::UnsafeSiteKind::UnsafeCall;
        site.seed.callee = Some("core::hint::assert_unchecked".into());
        site.operands[0].role = unsafe_site::OperandRole::Argument(0);
        site.operands[0].type_fingerprint = "bool".into();
        site.result_place = None;
        site
    }

    #[test]
    fn v3_endpoint_capture_is_sparse_and_does_not_enter_contract_proof_state() {
        let owner = FunctionKey::new("crate::f");
        let source = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let mut state = ProgramState::empty();
        state.set_value(PlaceKey::new("_1"), endpoint_value(source.clone()));
        let operation = ProgramOp {
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Nop,
            unsafe_sites: vec![bound_endpoint_site("_1")],
        };
        let before_validations = state.must_validations().clone();
        let before_pending = state.pending_must().clone();
        let mut graph = EndpointFlowGraph::default();
        let mut initialized = BTreeSet::new();
        let mut possibly_uninitialized = BTreeSet::new();
        let mut initialization_witnesses = BTreeMap::new();
        capture_bound_endpoint_sites(
            &owner,
            false,
            BasicBlock::from_usize(0),
            0,
            &operation,
            &state,
            &mut graph,
            &mut initialized,
            &mut possibly_uninitialized,
            &mut initialization_witnesses,
        );
        let site_id = operation.unsafe_sites[0].seed.site_id();
        let target = EndpointNode::UnsafeOperand {
            owner,
            site_id: site_id.clone(),
            role: unsafe_site::OperandRole::Address,
        };
        assert_eq!(graph.canonical_route(&source, &target).unwrap().len(), 1);
        assert_eq!(state.must_validations(), &before_validations);
        assert_eq!(state.pending_must(), &before_pending);
        assert!(graph.nodes.contains(&EndpointNode::UnsafeResult {
            owner: FunctionKey::new("crate::f"),
            site_id: operation.unsafe_sites[0].seed.site_id(),
        }));
    }

    #[test]
    fn v3_catalog_records_bound_sites_but_does_not_own_structural_discovery() {
        let operation = ProgramOp {
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Nop,
            unsafe_sites: vec![bound_endpoint_site("_1")],
        };
        let site_id = operation.unsafe_sites[0].seed.site_id();
        let expected_candidate = structural_candidate(&operation.unsafe_sites[0]).unwrap();
        let mut candidates = BTreeMap::new();
        let mut descriptors = BTreeMap::new();
        let mut sites_by_point = BTreeMap::new();
        let mut bound_sites = BTreeMap::new();
        record_site_contracts(
            &operation,
            |_| None,
            &mut candidates,
            &mut descriptors,
            &mut sites_by_point,
            &mut bound_sites,
        );
        assert_eq!(
            candidates,
            BTreeMap::from([(site_id.clone(), expected_candidate)])
        );
        assert_eq!(descriptors.len(), operation.unsafe_sites.len());
        assert_eq!(bound_sites.len(), operation.unsafe_sites.len());
        assert_eq!(
            sites_by_point[&point(0, 0)],
            BTreeSet::from([site_id.clone()])
        );
        assert_eq!(
            descriptors[&site_id].precision,
            unsafe_site::ContractPrecision::Generic
        );
    }

    #[test]
    fn v3_contract_proof_discharges_only_exact_current_evidence() {
        let subject = PlaceKey::new("_1");
        let collection = PlaceKey::new("_2");
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            subject,
            collection: Some(collection),
            access_width: None,
            proof_definition: None,
        };
        let snapshot = ValidationSnapshot {
            current_validations: BTreeSet::from([binding.clone()]),
            ..ValidationSnapshot::default()
        };
        let validation = ContractDescriptor {
            precision: unsafe_site::ContractPrecision::Precise,
            model_id: Some(KnownContractModel::GetUnchecked.as_str().into()),
            relevant_roles: BTreeSet::from([unsafe_site::OperandRole::Argument(1)]),
            predicates: BTreeSet::new(),
            failure_phase: super::super::contract_catalog::FailurePhase::Precondition,
            discharge: DischargeModel::ExactValidation,
            static_implication: StaticContractImplication::NotApplicable,
        };
        assert!(exact_contract_is_proven(
            &validation,
            Some(&snapshot),
            Some(&binding),
            None,
            &BTreeSet::new(),
        ));
        assert!(!exact_contract_is_proven(
            &validation,
            Some(&ValidationSnapshot::default()),
            Some(&binding),
            None,
            &BTreeSet::new(),
        ));

        let initialization = ContractDescriptor {
            precision: unsafe_site::ContractPrecision::Precise,
            model_id: Some(KnownContractModel::AssumeInit.as_str().into()),
            relevant_roles: BTreeSet::from([unsafe_site::OperandRole::Argument(0)]),
            predicates: BTreeSet::new(),
            failure_phase: super::super::contract_catalog::FailurePhase::ImmediateValidity,
            discharge: DischargeModel::ExactInitialization,
            static_implication: StaticContractImplication::NotApplicable,
        };
        assert!(exact_contract_is_proven(
            &initialization,
            None,
            None,
            None,
            &BTreeSet::from([unsafe_site::OperandRole::Argument(0)]),
        ));
        assert!(!exact_contract_is_proven(
            &initialization,
            None,
            None,
            None,
            &BTreeSet::new(),
        ));

        let mut opaque = validation;
        opaque.precision = unsafe_site::ContractPrecision::Opaque;
        opaque.discharge = DischargeModel::None;
        assert!(!exact_contract_is_proven(
            &opaque,
            Some(&snapshot),
            Some(&binding),
            None,
            &BTreeSet::from([unsafe_site::OperandRole::Argument(1)]),
        ));
    }

    #[test]
    fn condition_true_proof_requires_a_current_must_relation_for_the_exact_boolean() {
        let left = PlaceKey::new("_1");
        let right = PlaceKey::new("_2");
        let condition = PlaceKey::new("_3");
        let relation = ProofBinding::Relation(RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(left.clone()),
            RelationTerm::Place(right.clone()),
        )));
        let storage = relation.storage_places();
        let mut pending_only = ProgramState::empty();
        pending_only.set_pending(
            condition.clone(),
            [(relation.clone(), true, storage.clone())],
        );
        let site = bound_condition_true_site(&condition);
        assert!(!state_proves_condition_true_site(&pending_only, &site));

        let mut proven = pending_only.clone();
        proven.establish(BoundValidation::new(relation.clone(), storage.clone()));
        assert!(state_proves_condition_true_site(&proven, &site));

        let mut stale = proven.clone();
        stale.record_write(left, StaticWriteToken::new(4, 0));
        assert!(!state_proves_condition_true_site(&stale, &site));

        let diamond = BlockState::join_predecessors([&proven, &pending_only]).unwrap();
        assert!(!state_proves_condition_true_site(&diamond, &site));

        let other_condition = bound_condition_true_site(&PlaceKey::new("_4"));
        assert!(!state_proves_condition_true_site(&proven, &other_condition));
    }

    #[test]
    fn assert_unchecked_identity_is_exact_to_core_hint() {
        assert!(is_core_assert_unchecked_def_path(
            "core::hint::assert_unchecked"
        ));
        assert!(is_core_assert_unchecked_def_path(
            "std::hint::assert_unchecked"
        ));
        assert!(!is_core_assert_unchecked_def_path(
            "core::hint::unreachable_unchecked"
        ));
        assert!(!is_core_assert_unchecked_def_path(
            "alloc::hint::assert_unchecked"
        ));
        assert!(!is_core_assert_unchecked_def_path("core::hint::assert"));
    }

    #[test]
    fn influence_seed_field_reads_preserve_rhs_flow_and_track_persistent_storage() {
        let owner = FunctionKey::new("crate::f");
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let input = PlaceKey::new("_1");
        let mut state = ProgramState::empty();
        let mut input_value = endpoint_value(formal.clone());
        input_value.storage_provenance = StorageProvenance::EntryFormal {
            owner: owner.clone(),
            index: 1,
        };
        // Production entry setup seeds each public field projection through
        // `entry_field_projection_value`; mirror that state in this hand-built
        // transfer test rather than weakening the overwrite-liveness gate.
        input_value.may_hold_entry_public_field = true;
        state.set_value(input.clone(), input_value);
        let public = FieldAccessModel {
            access_owner: owner.clone(),
            full_place: PlaceKey::new("_1.0"),
            base: input.clone(),
            projection_path: Some(vec![ProjectionKey::Field(0)]),
            projection_unknown: false,
            def_path: "crate::Config::index".into(),
            adt_kind: PublicFieldAdtKind::Struct,
            path_public: true,
            type_class: InfluenceTypeClass::Integer,
            span: span(),
        };
        apply_program_op(
            &ProgramOp {
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: PlaceKey::new("_2"),
                    destination_projection: None,
                    value: ValueModel::Operand(place_operand("_1")),
                    field_origin: Some(public.clone()),
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
                unsafe_sites: Vec::new(),
            },
            BasicBlock::from_usize(0),
            0,
            &mut state,
        );
        assert_eq!(
            state_value(&state, &PlaceKey::new("_2"))
                .endpoint_sources
                .as_ref(),
            &BTreeSet::from([formal.clone()])
        );
        let mut public_read = place_operand("_1");
        public_read.field_origin = Some(public);
        let public_value = eval_operand(&state, &public_read);
        assert!(public_value.endpoint_sources.contains(&formal));
        assert!(public_value.endpoint_sources.iter().any(|source| matches!(
            source,
            EndpointNode::PublicField {
                access_owner,
                persistent: true,
                ..
            } if access_owner == &owner
        )));
        assert!(public_value.influence_seeds.iter().any(|(endpoint, seed)| {
            matches!(
                endpoint,
                EndpointNode::PublicField {
                    access_owner,
                    persistent: true,
                    ..
                } if access_owner == &owner
            ) && seed.origin == InfluenceOrigin::PersistentPublicField
                && matches!(
                    &seed.controller,
                    InfluenceController::PersistentPublicField { access_owner, field }
                        if access_owner == &owner && field == "crate::Config::index"
                )
                && seed
                    .capabilities
                    .intersects(super::super::influence::InfluenceCapabilities::INDEX_OR_LENGTH)
        }));

        let mut private_read = place_operand("_1");
        private_read.field_origin = Some(FieldAccessModel {
            access_owner: owner.clone(),
            full_place: PlaceKey::new("_1.1"),
            base: input,
            projection_path: Some(vec![ProjectionKey::Field(1)]),
            projection_unknown: false,
            def_path: "crate::Config::private_index".into(),
            adt_kind: PublicFieldAdtKind::Struct,
            path_public: false,
            type_class: InfluenceTypeClass::Integer,
            span: span(),
        });
        // A private projection cannot be supplied independently by safe
        // external code. Keeping the receiver Formal here would recreate the
        // false P1/P3 source that the exact-field seed is meant to remove.
        let private_value = eval_operand(&state, &private_read);
        assert!(private_value.endpoint_sources.is_empty());
        assert_eq!(
            private_value.behavior_control,
            BehaviorControlProvenance::InternalOnly
        );
        apply_program_op(
            &ProgramOp {
                point: Some(point(0, 2)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: PlaceKey::new("_3"),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Add,
                        left: place_operand("_1"),
                        right: constant_operand(1),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
                unsafe_sites: Vec::new(),
            },
            BasicBlock::from_usize(0),
            2,
            &mut state,
        );
        let derived = state_value(&state, &PlaceKey::new("_3"));
        assert_eq!(derived.endpoint_sources.len(), 1);
        let derivation = derived
            .endpoint_sources
            .iter()
            .next()
            .expect("internal arithmetic has a derivation endpoint");
        assert!(matches!(
            derivation,
            EndpointNode::InternalDerivation {
                owner: candidate,
                kind: DerivationKind::Offset,
                ..
            } if candidate == &owner
        ));
        assert!(derived.endpoint_edges.iter().any(|edge| matches!(
            edge,
            EndpointEdge { from, to, kind: EndpointEdgeKind::Derivation, .. }
                if from == &formal && to == derivation
        )));
    }

    #[test]
    fn v3_exact_field_projection_strong_writes_kill_only_written_paths() {
        let owner = FunctionKey::new("crate::PublicRaw::read");
        let base = PlaceKey::new("_1");
        let field = FieldAccessModel {
            access_owner: owner.clone(),
            full_place: PlaceKey::new("_1.0"),
            base: base.clone(),
            projection_path: Some(vec![ProjectionKey::Field(0)]),
            projection_unknown: false,
            def_path: "crate::PublicRaw::pointer".into(),
            adt_kind: PublicFieldAdtKind::Struct,
            path_public: true,
            type_class: InfluenceTypeClass::Pointer,
            span: span(),
        };
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let mut base_value = endpoint_value(formal.clone());
        base_value.storage_provenance = StorageProvenance::EntryFormal {
            owner: owner.clone(),
            index: 1,
        };
        base_value.behavior_control = BehaviorControlProvenance::MaybeExternal;
        let mut entry = ProgramState::empty();
        entry.set_value(base.clone(), base_value.clone());
        entry.set_value(
            field.full_place.clone(),
            entry_field_projection_value(&base_value, &field, 1),
        );
        let mut read = place_operand(&field.full_place.0);
        read.field_origin = Some(field.clone());
        assert!(eval_operand(&entry, &read)
            .endpoint_sources
            .iter()
            .any(|source| matches!(
                source,
                EndpointNode::PublicField {
                    persistent: true,
                    ..
                }
            )));

        let write = |state: &mut ProgramState, block, statement, value| {
            apply_program_op(
                &ProgramOp {
                    point: Some(point(block, statement)),
                    kind: ProgramOpKind::Assign(AssignmentModel {
                        destination: field.full_place.clone(),
                        destination_projection: None,
                        value: ValueModel::Operand(constant_operand(value)),
                        field_origin: Some(field.clone()),
                        raw_read_source: None,
                        legacy_write: true,
                        out_formal: None,
                    }),
                    unsafe_sites: Vec::new(),
                },
                BasicBlock::from_usize(block as usize),
                statement as usize,
                state,
            );
        };

        let mut straight = entry.clone();
        write(&mut straight, 0, 0, 0);
        let straight_read = eval_operand(&straight, &read);
        assert!(!straight_read.endpoint_sources.contains(&formal));
        assert!(!straight_read
            .endpoint_sources
            .iter()
            .any(|source| matches!(source, EndpointNode::PublicField { .. })));

        let one_branch = ProgramState::join_predecessors([&entry, &straight]).unwrap();
        assert!(eval_operand(&one_branch, &read)
            .endpoint_sources
            .iter()
            .any(|source| matches!(
                source,
                EndpointNode::PublicField {
                    persistent: true,
                    ..
                }
            )));

        let mut other_write = entry.clone();
        write(&mut other_write, 1, 0, 1);
        let all_paths = ProgramState::join_predecessors([&straight, &other_write]).unwrap();
        assert!(!eval_operand(&all_paths, &read)
            .endpoint_sources
            .iter()
            .any(|source| matches!(source, EndpointNode::PublicField { .. })));

        let mut whole_base = ProgramState::empty();
        whole_base.set_value(base.clone(), base_value.clone());
        whole_base.set_value(
            field.full_place.clone(),
            entry_field_projection_value(&base_value, &field, 1),
        );
        apply_program_op(
            &ProgramOp {
                point: Some(point(2, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: base,
                    destination_projection: None,
                    value: ValueModel::Operand(constant_operand(0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: true,
                    out_formal: None,
                }),
                unsafe_sites: Vec::new(),
            },
            BasicBlock::from_usize(2),
            0,
            &mut whole_base,
        );
        assert!(
            !eval_operand(&whole_base, &read)
                .endpoint_sources
                .iter()
                .any(|source| matches!(source, EndpointNode::PublicField { .. })),
            "a strong write of the whole base must kill its seeded field descendants"
        );

        let one_base_branch = ProgramState::join_predecessors([&whole_base, &entry]).unwrap();
        assert!(eval_operand(&one_base_branch, &read)
            .endpoint_sources
            .iter()
            .any(|source| matches!(source, EndpointNode::PublicField { .. })));

        let mut other_base = entry;
        apply_program_op(
            &ProgramOp {
                point: Some(point(3, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: field.base.clone(),
                    destination_projection: None,
                    value: ValueModel::Operand(constant_operand(1)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: true,
                    out_formal: None,
                }),
                unsafe_sites: Vec::new(),
            },
            BasicBlock::from_usize(3),
            0,
            &mut other_base,
        );
        let all_base_paths = ProgramState::join_predecessors([&whole_base, &other_base]).unwrap();
        assert!(eval_operand(&all_base_paths, &read)
            .endpoint_sources
            .is_empty());
    }

    #[test]
    fn v3_field_behavior_requires_persistent_external_storage() {
        let owner = FunctionKey::new("crate::root");
        let base = PlaceKey::new("_1");
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let mut base_value = endpoint_value(formal.clone());
        base_value.storage_provenance = StorageProvenance::EntryFormal {
            owner: owner.clone(),
            index: 1,
        };
        base_value.behavior_control = BehaviorControlProvenance::MaybeExternal;
        let mut state = ProgramState::empty();
        state.set_value(base.clone(), base_value.clone());

        let private = FieldAccessModel {
            access_owner: owner.clone(),
            full_place: PlaceKey::new("_1.0"),
            base: base.clone(),
            projection_path: Some(vec![ProjectionKey::Field(0)]),
            projection_unknown: false,
            def_path: "crate::PrivateCallbacks::callback".into(),
            adt_kind: PublicFieldAdtKind::Struct,
            path_public: false,
            type_class: InfluenceTypeClass::Callable,
            span: span(),
        };
        state.set_value(
            private.full_place.clone(),
            entry_field_projection_value(&base_value, &private, 1),
        );
        let mut private_read = place_operand(&private.full_place.0);
        private_read.field_origin = Some(private);
        let private_value = eval_operand(&state, &private_read);
        assert!(private_value.endpoint_sources.is_empty());
        assert!(behavior_controllers(&private_value).is_empty());

        let public_enum = FieldAccessModel {
            access_owner: owner.clone(),
            full_place: PlaceKey::new("_1.1"),
            base,
            projection_path: Some(vec![ProjectionKey::Field(1)]),
            projection_unknown: false,
            def_path: "crate::CallbackChoice::Call::0".into(),
            adt_kind: PublicFieldAdtKind::Enum,
            path_public: true,
            type_class: InfluenceTypeClass::Callable,
            span: span(),
        };
        state.set_value(
            public_enum.full_place.clone(),
            entry_field_projection_value(&base_value, &public_enum, 1),
        );
        let mut enum_read = place_operand(&public_enum.full_place.0);
        enum_read.field_origin = Some(public_enum);
        let enum_value = eval_operand(&state, &enum_read);
        assert!(enum_value.endpoint_sources.contains(&formal));
        assert_eq!(
            behavior_controllers(&enum_value),
            BTreeSet::from([BehaviorController::PublicFormal { owner, index: 1 }])
        );

        let local_owner = FunctionKey::new("crate::local_callback");
        let local_base = PlaceKey::new("_2");
        let internal = BehaviorController::InternalFunction {
            function: FunctionKey::new("crate::zero_pointer"),
        };
        state.set_value(
            local_base.clone(),
            ValueFacts {
                behavior_controllers: BTreeSet::from([internal.clone()]),
                behavior_control: BehaviorControlProvenance::InternalOnly,
                storage_provenance: StorageProvenance::DefinitelyLocal,
                ..ValueFacts::default()
            },
        );
        let local_field = FieldAccessModel {
            access_owner: local_owner,
            full_place: PlaceKey::new("_2.0"),
            base: local_base,
            projection_path: Some(vec![ProjectionKey::Field(0)]),
            projection_unknown: false,
            def_path: "crate::LocalCallback::callback".into(),
            adt_kind: PublicFieldAdtKind::Struct,
            path_public: true,
            type_class: InfluenceTypeClass::Callable,
            span: span(),
        };
        let mut local_read = place_operand(&local_field.full_place.0);
        local_read.field_origin = Some(local_field);
        assert_eq!(
            behavior_controllers(&eval_operand(&state, &local_read)),
            BTreeSet::from([internal])
        );
    }

    #[test]
    fn v3_ancestor_and_must_alias_writes_kill_descendants_but_unknown_writes_do_not() {
        let owner = FunctionKey::new("crate::NestedRaw::read");
        let base = PlaceKey::new("_1");
        let field = FieldAccessModel {
            access_owner: owner.clone(),
            full_place: PlaceKey::new("_1.0.1"),
            base: base.clone(),
            projection_path: Some(vec![ProjectionKey::Field(0), ProjectionKey::Field(1)]),
            projection_unknown: false,
            def_path: "crate::Inner::pointer".into(),
            adt_kind: PublicFieldAdtKind::Struct,
            path_public: true,
            type_class: InfluenceTypeClass::Pointer,
            span: span(),
        };
        let mut base_value = endpoint_value(EndpointNode::Formal { owner, index: 1 });
        base_value.storage_provenance = StorageProvenance::EntryFormal {
            owner: FunctionKey::new("crate::NestedRaw::read"),
            index: 1,
        };
        let seeded = || {
            let mut state = ProgramState::empty();
            state.set_value(base.clone(), base_value.clone());
            state.set_value(
                field.full_place.clone(),
                entry_field_projection_value(&base_value, &field, 1),
            );
            state
        };
        let write = |state: &mut ProgramState, projection: PlaceProjectionModel| {
            apply_program_op(
                &ProgramOp {
                    point: Some(point(0, 0)),
                    kind: ProgramOpKind::Assign(AssignmentModel {
                        destination: PlaceKey::new("destination"),
                        destination_projection: Some(projection),
                        value: ValueModel::Operand(constant_operand(0)),
                        field_origin: None,
                        raw_read_source: None,
                        legacy_write: true,
                        out_formal: None,
                    }),
                    unsafe_sites: Vec::new(),
                },
                BasicBlock::from_usize(0),
                0,
                state,
            );
        };

        let mut ancestor = seeded();
        write(
            &mut ancestor,
            PlaceProjectionModel {
                base: base.clone(),
                path: vec![ProjectionKey::Field(0)],
                projection_unknown: false,
            },
        );
        let mut read = place_operand(&field.full_place.0);
        read.field_origin = Some(field.clone());
        assert!(eval_operand(&ancestor, &read).endpoint_sources.is_empty());

        let alias = PlaceKey::new("_2");
        let mut must_alias = seeded();
        must_alias.set_value(
            alias.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([base.clone()]),
                semantic_unknown: false,
                ..ValueFacts::default()
            },
        );
        write(
            &mut must_alias,
            PlaceProjectionModel {
                base: alias,
                path: vec![ProjectionKey::Deref],
                projection_unknown: false,
            },
        );
        assert!(eval_operand(&must_alias, &read).endpoint_sources.is_empty());

        let mut unknown = seeded();
        write(
            &mut unknown,
            PlaceProjectionModel {
                base,
                path: vec![ProjectionKey::Field(0)],
                projection_unknown: true,
            },
        );
        let unresolved = unknown
            .may_values()
            .get(&field.full_place)
            .expect("a may-alias write cannot prove the entry field dead");
        assert!(unresolved.may_hold_entry_public_field);
        assert!(unresolved.havoced);
    }

    #[test]
    fn v3_local_return_endpoint_is_installed_only_on_the_normal_edge() {
        let caller = FunctionKey::new("crate::f");
        let callee = FunctionKey::new("crate::callee");
        let destination = PlaceKey::new("_2");
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(callee.clone()),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand("_1")],
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        let normal = BasicBlock::from_usize(1);
        let unwind = BasicBlock::from_usize(2);
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: ProgramPoint {
                function: caller.clone(),
                ..point(0, 0)
            },
            call,
        };
        let outputs = BTreeMap::from([(
            ProgramPoint {
                function: caller,
                ..point(0, 0)
            },
            MappedCallOutputs {
                returned: Some((
                    destination.clone(),
                    AbstractValue::new([AbstractOrigin::Formal(1)]),
                )),
                ..MappedCallOutputs::default()
            },
        )]);
        let mut normal_state = ProgramState::empty();
        normal_state.set_value(
            PlaceKey::new("_1"),
            endpoint_value(EndpointNode::Formal {
                owner: FunctionKey::new("crate::f"),
                index: 1,
            }),
        );
        let mut unwind_state = normal_state.clone();
        apply_program_edge_with_outputs(&edge, unwind, &mut unwind_state, &outputs);
        apply_program_edge_with_outputs(&edge, normal, &mut normal_state, &outputs);
        assert!(state_value(&unwind_state, &destination)
            .endpoint_sources
            .is_empty());
        assert!(matches!(
            state_value(&normal_state, &destination)
                .endpoint_sources
                .iter()
                .next(),
            Some(EndpointNode::CallResult {
                callsite,
                output: CallSiteOutput::Return,
            }) if callsite.callee == callee && callsite.caller == FunctionKey::new("crate::f")
        ));
    }

    #[test]
    fn v3_unsafe_results_replace_provenance_after_assignment_and_normal_call_only() {
        let owner = FunctionKey::new("crate::f");
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let destination = PlaceKey::new("_2");
        let site = bound_endpoint_site("_1");
        let site_id = site.seed.site_id();
        let result = EndpointNode::UnsafeResult {
            owner: owner.clone(),
            site_id: site_id.clone(),
        };
        let mut assignment_state = ProgramState::empty();
        assignment_state.set_value(PlaceKey::new("_1"), endpoint_value(formal.clone()));
        apply_program_op(
            &ProgramOp {
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: destination.clone(),
                    destination_projection: None,
                    value: ValueModel::Operand(place_operand("_1")),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
                unsafe_sites: vec![site.clone()],
            },
            BasicBlock::from_usize(0),
            0,
            &mut assignment_state,
        );
        let assigned = state_value(&assignment_state, &destination);
        assert_eq!(
            assigned.endpoint_sources.as_ref(),
            &BTreeSet::from([result.clone()])
        );
        assert!(assigned.endpoint_edges.contains(&EndpointEdge {
            from: formal.clone(),
            to: result.clone(),
            kind: EndpointEdgeKind::UnsafeResult,
            point: Some(point(0, 0)),
        }));

        let normal = BasicBlock::from_usize(1);
        let unwind = BasicBlock::from_usize(2);
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::OpaqueDirect,
                args: vec![place_operand("_1")],
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: point(0, 0),
            call,
        };
        let mut normal_state = ProgramState::empty();
        normal_state.set_value(PlaceKey::new("_1"), endpoint_value(formal.clone()));
        let mut unwind_state = normal_state.clone();
        apply_program_edge_with_outputs_and_sites(
            &edge,
            unwind,
            &mut unwind_state,
            &LocalCallOutputs::new(),
            std::slice::from_ref(&site),
        );
        apply_program_edge_with_outputs_and_sites(
            &edge,
            normal,
            &mut normal_state,
            &LocalCallOutputs::new(),
            std::slice::from_ref(&site),
        );
        assert!(state_value(&unwind_state, &destination)
            .endpoint_sources
            .is_empty());
        let returned = state_value(&normal_state, &destination);
        assert_eq!(
            returned.endpoint_sources.as_ref(),
            &BTreeSet::from([result.clone()])
        );
        assert!(returned.endpoint_edges.contains(&EndpointEdge {
            from: formal,
            to: result,
            kind: EndpointEdgeKind::UnsafeResult,
            point: Some(point(0, 0)),
        }));
    }

    #[test]
    fn v3_local_out_endpoint_replaces_only_the_normal_edge_value() {
        let callee = FunctionKey::new("crate::callee");
        let actual = PlaceKey::new("_1");
        let call_point = point(0, 0);
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(callee.clone()),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand("_1")],
                destination: PlaceKey::new("_2"),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![actual.clone()],
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        let normal = BasicBlock::from_usize(1);
        let unwind = BasicBlock::from_usize(2);
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: call_point.clone(),
            call,
        };
        let outputs = BTreeMap::from([(
            call_point,
            MappedCallOutputs {
                outs: BTreeMap::from([(
                    actual.clone(),
                    MappedOutValue {
                        value: AbstractValue::new([AbstractOrigin::Constant(0)]),
                        may_skip_write: false,
                    },
                )]),
                ..MappedCallOutputs::default()
            },
        )]);
        let original = EndpointNode::Formal {
            owner: FunctionKey::new("crate::f"),
            index: 1,
        };
        let mut normal_state = ProgramState::empty();
        normal_state.set_value(
            actual.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([actual.clone()]),
                endpoint_sources: Arc::new(BTreeSet::from([original.clone()])),
                ..ValueFacts::default()
            },
        );
        let mut unwind_state = normal_state.clone();
        apply_program_edge_with_outputs(&edge, unwind, &mut unwind_state, &outputs);
        apply_program_edge_with_outputs(&edge, normal, &mut normal_state, &outputs);
        assert_eq!(
            state_value(&unwind_state, &actual)
                .endpoint_sources
                .as_ref(),
            &BTreeSet::from([original])
        );
        assert!(matches!(
            state_value(&normal_state, &actual)
                .endpoint_sources
                .iter()
                .next(),
            Some(EndpointNode::CallResult {
                callsite,
                output: CallSiteOutput::Out(1),
            }) if callsite.callee == callee && callsite.caller == FunctionKey::new("crate::f")
        ));
    }

    #[test]
    fn influence_seed_external_behavior_results_have_structural_endpoint_origins() {
        let open = eval_registry_value(
            &ProgramState::empty(),
            &RegistryValueModel::SelectedOpenBehavior {
                token: "trait-return".into(),
                controller: BehaviorController::OpenTrait {
                    trait_id: FunctionKey::new("crate::OpenTrait"),
                    openness: BehaviorOpenness::ProvenOpen,
                    resolved_target: None,
                    receiver: OpenBehaviorReceiver::Dynamic,
                },
                inputs: Vec::new(),
                type_class: InfluenceTypeClass::Aggregate,
            },
            &point(1, 0),
        );
        assert!(open.endpoint_sources.iter().any(|source| matches!(
            source,
            EndpointNode::ExternalBehavior {
                kind: ExternalSourceKind::OpenTrait,
                ..
            }
        )));
        assert!(open.influence_seeds.iter().any(|(_, seed)| {
            seed.origin == InfluenceOrigin::OpenTraitReturn
                && seed
                    .capabilities
                    .intersects(super::super::influence::InfluenceCapabilities::TYPED_VALUE)
                && !seed
                    .capabilities
                    .intersects(super::super::influence::InfluenceCapabilities::ADDRESS)
        }));

        let ffi_return = eval_registry_value(
            &ProgramState::empty(),
            &RegistryValueModel::ExternalBehavior {
                kind: ExternalSourceKind::FfiReturn,
                controller: BehaviorController::FfiHost,
                token: "ffi-return".into(),
                type_class: InfluenceTypeClass::Pointer,
            },
            &point(1, 1),
        );
        assert!(ffi_return.influence_seeds.iter().any(|(_, seed)| {
            seed.origin == InfluenceOrigin::FfiReturn
                && seed.controller == InfluenceController::FfiHost
                && seed
                    .capabilities
                    .intersects(super::super::influence::InfluenceCapabilities::ADDRESS)
        }));

        let actual = PlaceKey::new("_1");
        let mut state = ProgramState::empty();
        state.set_value(actual.clone(), ValueFacts::default());
        apply_call_side_effects(
            &CallModel {
                descriptor: CallDescriptor {
                    callee: None,
                    raw_def: None,
                    disposition: BoundaryDisposition::ExactFfiOut,
                    args: vec![place_operand("_1")],
                    destination: PlaceKey::new("_2"),
                    generic_substitution: GenericCallSubstitution::default(),
                },
                value: RegistryValueModel::Empty,
                predicate: None,
                mutable_actuals: Vec::new(),
                ffi_out_actuals: vec![FfiOutActual {
                    place: actual.clone(),
                    type_class: InfluenceTypeClass::Pointer,
                }],
                typed_initialization_effect: None,
                destination_is_bool: false,
                destination_restricted: false,
                access_width: None,
                legacy_write: false,
                destination_out_formal: None,
            },
            &point(2, 0),
            &mut state,
        );
        assert!(state_value(&state, &actual)
            .endpoint_sources
            .iter()
            .any(|source| matches!(
                source,
                EndpointNode::ExternalBehavior {
                    kind: ExternalSourceKind::FfiOut,
                    ..
                }
            )));
        assert!(state_value(&state, &actual)
            .influence_seeds
            .iter()
            .any(|(_, seed)| {
                seed.origin == InfluenceOrigin::FfiOut
                    && seed.controller == InfluenceController::FfiHost
                    && seed
                        .capabilities
                        .intersects(super::super::influence::InfluenceCapabilities::ADDRESS)
            }));
    }

    #[test]
    fn influence_transfer_preserves_address_views_and_marks_opaque_sources_incomplete() {
        let owner = FunctionKey::new("crate::lossy");
        let endpoint = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let source = PlaceKey::new("_1");
        let mut seeded = ValueFacts::default();
        seeded.influence_seeds.insert(
            endpoint.clone(),
            InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal { owner, index: 1 },
                InfluenceTypeClass::Pointer,
            ),
        );
        let mut state = ProgramState::empty();
        state.set_value(source.clone(), seeded);

        let cast = eval_value_model(
            &state,
            &ValueModel::AddressPointerView {
                operand: place_operand("_1"),
                generic_view: None,
            },
            StaticWriteToken::new(0, 0),
        );
        assert_eq!(
            cast.influence_seeds
                .get(&endpoint)
                .expect("address view keeps source auditable")
                .precision,
            super::super::influence::TokenPrecision::StrongBoundary,
        );

        let opaque_cast = eval_value_model(
            &state,
            &ValueModel::PointerCast {
                operand: place_operand("_1"),
                obligation: ConcreteLayoutObligation::default(),
            },
            StaticWriteToken::new(0, 1),
        );
        assert_eq!(
            opaque_cast
                .influence_seeds
                .get(&endpoint)
                .expect("opaque cast keeps source auditable")
                .precision,
            super::super::influence::TokenPrecision::Incomplete,
        );

        let call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::OpaqueDirect,
                args: vec![place_operand("_1")],
                destination: PlaceKey::new("_2"),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![source.clone()],
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        apply_call_side_effects(&call, &point(0, 2), &mut state);
        assert_eq!(
            state_value(&state, &source)
                .influence_seeds
                .get(&endpoint)
                .expect("havoc keeps source auditable")
                .precision,
            super::super::influence::TokenPrecision::Incomplete,
        );
    }

    #[test]
    fn pointer_helper_without_a_postcondition_invalidates_a_stale_uninitialized_witness() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let mut state = ProgramState::empty();
        state.set_value(
            storage.clone(),
            ValueFacts {
                maybe_uninit_initialized: BTreeSet::from([false]),
                exact_roots: BTreeSet::from([storage.clone()]),
                ..ValueFacts::default()
            },
        );
        state.set_value(
            pointer.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([pointer.clone()]),
                whole_init_targets: BTreeSet::from([storage.clone()]),
                whole_init_target_tracked: true,
                ..ValueFacts::default()
            },
        );
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::helper")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand(&pointer.0)],
                destination: PlaceKey::new("_3"),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        apply_call_side_effects(&call, &point(1, 0), &mut state);
        assert_eq!(
            typed_initialization_state(&state_value(&state, &storage)),
            TypedInitializationState::Unknown
        );
    }

    #[test]
    fn typed_maybe_uninit_pointer_view_derives_exact_address_and_validity_influence() {
        let owner = FunctionKey::new("crate::maybe_uninit_view");
        let endpoint = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let receiver = PlaceKey::new("_1");
        let mut seeded = ValueFacts::default();
        seeded.endpoint_sources = Arc::new(BTreeSet::from([endpoint.clone()]));
        seeded.influence_seeds.insert(
            endpoint.clone(),
            InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal {
                    owner: owner.clone(),
                    index: 1,
                },
                InfluenceTypeClass::Reference,
            ),
        );
        let mut state = ProgramState::empty();
        state.set_value(receiver.clone(), seeded);

        let derived = eval_registry_value(
            &state,
            &RegistryValueModel::TypedInitializationPointer {
                receiver: place_operand(&receiver.0),
                generic_view: None,
            },
            &point(0, 1),
        );
        let token = derived
            .influence_seeds
            .get(&endpoint)
            .expect("typed pointer view retains the public source");
        assert_eq!(
            token.precision,
            super::super::influence::TokenPrecision::ExactDerived,
        );
        assert!(token
            .capabilities
            .intersects(super::super::influence::InfluenceCapabilities::ADDRESS));
        assert!(token
            .capabilities
            .intersects(super::super::influence::InfluenceCapabilities::VALIDITY_BITS));

        let pointer = PlaceKey::new("_2");
        state.set_value(pointer.clone(), derived);
        let mut site = bound_endpoint_site(&pointer.0);
        site.seed.owner = owner.clone();
        let operation = ProgramOp {
            point: Some(point(0, 2)),
            kind: ProgramOpKind::Nop,
            unsafe_sites: vec![site],
        };
        let site_id = operation.unsafe_sites[0].seed.site_id();
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id,
            role: unsafe_site::OperandRole::Address,
        };
        let mut graph = EndpointFlowGraph::default();
        capture_bound_endpoint_sites(
            &owner,
            false,
            BasicBlock::from_usize(0),
            2,
            &operation,
            &state,
            &mut graph,
            &mut BTreeSet::new(),
            &mut BTreeSet::new(),
            &mut BTreeMap::new(),
        );
        assert!(graph.edges.contains(&EndpointEdge {
            from: endpoint,
            to: target,
            kind: EndpointEdgeKind::TypedValidityPointer,
            point: Some(point(0, 2)),
        }));
    }

    #[test]
    fn v3_behavior_controllers_preserve_internal_public_and_unknown_provenance() {
        let internal = BehaviorController::InternalFunction {
            function: FunctionKey::new("crate::identity"),
        };
        let mut internal_value = ValueFacts::default();
        internal_value.behavior_controllers.insert(internal.clone());
        assert_eq!(
            behavior_controllers(&internal_value),
            BTreeSet::from([internal.clone()])
        );

        let owner = FunctionKey::new("crate::root");
        let formal = BehaviorController::PublicFormal {
            owner: owner.clone(),
            index: 1,
        };
        let mut public_value = ValueFacts::default();
        public_value.endpoint_sources =
            Arc::new(BTreeSet::from([EndpointNode::Formal { owner, index: 1 }]));
        assert_eq!(
            behavior_controllers(&public_value),
            BTreeSet::from([formal])
        );

        internal_value.behavior_controller_unknown = true;
        assert_eq!(
            behavior_controllers(&internal_value),
            BTreeSet::from([internal, BehaviorController::Unknown])
        );
        assert_eq!(
            behavior_controllers(&ValueFacts::default()),
            BTreeSet::from([BehaviorController::Unknown])
        );
    }

    #[test]
    fn v3_closed_trait_result_keeps_lower_input_provenance_for_reclassification() {
        let owner = FunctionKey::new("crate::root");
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let input = PlaceKey::new("_1");
        let mut state = ProgramState::empty();
        state.set_value(input, endpoint_value(formal.clone()));
        let result = eval_registry_value(
            &state,
            &RegistryValueModel::SelectedOpenBehavior {
                token: "trait-return".into(),
                controller: BehaviorController::OpenTrait {
                    trait_id: FunctionKey::new("crate::sealed::Behavior"),
                    openness: BehaviorOpenness::ProvenClosed,
                    resolved_target: None,
                    receiver: OpenBehaviorReceiver::Unknown,
                },
                inputs: vec![place_operand("_1")],
                type_class: InfluenceTypeClass::Aggregate,
            },
            &point(1, 0),
        );

        assert!(result.endpoint_sources.contains(&formal));
        assert!(result.endpoint_sources.iter().any(|source| matches!(
            source,
            EndpointNode::ExternalBehavior {
                controller: BehaviorController::OpenTrait {
                    openness: BehaviorOpenness::ProvenClosed,
                    ..
                },
                ..
            }
        )));
    }

    #[test]
    fn v3_internal_function_pointer_seed_is_not_treated_as_unknown() {
        let mut state = ProgramState::empty();
        state.set_value(PlaceKey::new("_1"), ValueFacts::default());
        let controller = BehaviorController::InternalFunction {
            function: FunctionKey::new("crate::identity"),
        };
        let result = eval_value_model(
            &state,
            &ValueModel::BehaviorPointer {
                operand: place_operand("_1"),
                seed: Some(controller.clone()),
            },
            StaticWriteToken::new(0, 0),
        );
        assert_eq!(behavior_controllers(&result), BTreeSet::from([controller]));
    }

    #[test]
    fn influence_controller_public_function_pointer_return_is_typed() {
        let owner = FunctionKey::new("crate::root");
        let callable = PlaceKey::new("_1");
        let destination = PlaceKey::new("_2");
        let normal = BasicBlock::from_usize(1);
        let mut state = ProgramState::empty();
        state.set_value(
            callable.clone(),
            endpoint_value(EndpointNode::Formal {
                owner: owner.clone(),
                index: 1,
            }),
        );
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::OpaqueIndirect,
                args: Vec::new(),
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::IndirectBehavior {
                source: place_operand("_1"),
                type_class: InfluenceTypeClass::Pointer,
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call,
            },
            normal,
            &mut state,
        );

        assert!(state_value(&state, &destination)
            .influence_seeds
            .iter()
            .any(|(_, seed)| {
                seed.origin == InfluenceOrigin::PublicFunctionPointerReturn
                    && seed.controller
                        == InfluenceController::PublicFormal {
                            owner: owner.clone(),
                            index: 1,
                        }
                    && seed
                        .capabilities
                        .intersects(super::super::influence::InfluenceCapabilities::ADDRESS)
                    && seed.precision == super::super::influence::TokenPrecision::StrongBoundary
            }));
    }

    #[test]
    fn v3_foreign_function_pointer_seed_remains_external_behavior() {
        let function = FunctionKey::new("crate::foreign_pointer");
        assert_eq!(
            behavior_pointer_controller(function.clone(), true),
            BehaviorController::FfiHost,
        );
        assert_eq!(
            behavior_pointer_controller(function.clone(), false),
            BehaviorController::InternalFunction { function },
        );
        assert_eq!(BehaviorController::FfiHost.rejection_reason(), None);
    }

    #[test]
    fn exported_ffi_callback_signature_gate_excludes_closures() {
        assert!(def_kind_has_direct_fn_sig(DefKind::Fn));
        assert!(def_kind_has_direct_fn_sig(DefKind::AssocFn));
        assert!(!def_kind_has_direct_fn_sig(DefKind::Closure));
    }

    #[test]
    fn exported_ffi_callback_argument_has_a_host_boundary_endpoint() {
        let owner = FunctionKey::new("crate::callback");
        let endpoint = ffi_callback_input_endpoint(&owner, 2, &span());
        assert!(matches!(
            endpoint,
            EndpointNode::ExternalBehavior {
                owner: endpoint_owner,
                kind: ExternalSourceKind::Host,
                controller: BehaviorController::FfiHost,
                ref token,
                ..
            } if endpoint_owner == owner && token == "ffi-callback-input:crate::callback:2"
        ));
        assert_eq!(
            external_influence_origin(ExternalSourceKind::Host),
            InfluenceOrigin::FfiCallbackInput,
        );
    }

    #[test]
    fn v3_observed_internal_behavior_target_requires_unique_local_typed_controller() {
        let closure = FunctionKey::new("crate::root::{closure#0}");
        let mut value = ValueFacts::default();
        value
            .behavior_controllers
            .insert(BehaviorController::InternalClosure {
                function: closure.clone(),
            });

        let target = observed_internal_behavior_target(&value, |function| {
            (function == &closure).then_some(true)
        })
        .expect("a uniquely observed local closure must be callable");
        assert_eq!(target.function, closure);
        assert_eq!(target.formal_offset, 2);

        value.behavior_controller_unknown = true;
        assert!(observed_internal_behavior_target(&value, |_| Some(true)).is_none());

        value.behavior_controller_unknown = false;
        value
            .behavior_controllers
            .insert(BehaviorController::Unknown);
        assert!(observed_internal_behavior_target(&value, |_| Some(true)).is_none());

        let external = FunctionKey::new("dependency::callback");
        let mut external_value = ValueFacts::default();
        external_value
            .behavior_controllers
            .insert(BehaviorController::InternalFunction { function: external });
        assert!(observed_internal_behavior_target(&external_value, |_| None).is_none());
    }

    #[test]
    fn v3_observed_internal_behavior_edges_pair_actuals_and_return_with_the_target() {
        let caller = FunctionKey::new("crate::root");
        let closure = FunctionKey::new("crate::root::{closure#0}");
        let point = ProgramPoint {
            function: caller.clone(),
            block: 4,
            statement: 2,
            span: span(),
        };
        let target = ObservedInternalBehaviorTarget {
            function: closure.clone(),
            controller: BehaviorController::InternalClosure {
                function: closure.clone(),
            },
            formal_offset: 2,
        };
        let actual = EndpointNode::Formal {
            owner: caller.clone(),
            index: 1,
        };
        let value = endpoint_value(actual.clone());

        let (edges, call_edge) =
            observed_internal_behavior_edges(&caller, &point, &target, std::iter::once(&value));

        assert!(edges.contains(&EndpointEdge {
            from: actual,
            to: EndpointNode::Formal {
                owner: closure.clone(),
                index: 2,
            },
            kind: EndpointEdgeKind::ActualToFormal,
            point: Some(point.clone()),
        }));
        assert!(edges.contains(&EndpointEdge {
            from: EndpointNode::Return {
                owner: closure.clone(),
            },
            to: indirect_behavior_result_node(&point, &target.controller),
            kind: EndpointEdgeKind::ReturnToDestination,
            point: Some(point.clone()),
        }));
        assert_eq!(call_edge.caller, caller);
        assert_eq!(call_edge.callee, closure);
        assert_eq!(call_edge.kind, CandidateCallKind::Callback);
    }

    fn has_pending_validation(state: &ProgramState, condition: &PlaceKey) -> bool {
        state.pending_must().get(condition).is_some_and(|facts| {
            facts
                .keys()
                .any(|(binding, _)| binding.validation().is_some())
        })
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
            actual_endpoint_sources: Vec::new(),
            destination: PlaceKey::new("_0"),
            point: point(1, 0),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            index_range_kind: None,
            generic_substitution: GenericCallSubstitution::default(),
        }
    }

    #[test]
    fn task6_call_contract_requires_the_exact_subject_collection_pair() {
        let collection = PlaceKey::new("_1");
        let subject = PlaceKey::new("_2");
        let other = PlaceKey::new("_3");
        let call = task6_call(&collection, &subject);
        let contract = task6_contract();

        let mut guarded = ProgramState::empty();
        for place in [&collection, &subject, &other] {
            seed_exact(&mut guarded, place);
        }
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
            subject: subject.clone(),
            collection: Some(collection.clone()),
        };
        guarded.establish(BoundValidation::new(
            binding.clone().into(),
            binding.storage_places(),
        ));
        assert_eq!(
            compose_validation_route(&contract, &call, &guarded, 3),
            ValidationRoute::Satisfied
        );

        let mut wrong_collection = ProgramState::empty();
        for place in [&collection, &subject, &other] {
            seed_exact(&mut wrong_collection, place);
        }
        let wrong = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
            subject: subject.clone(),
            collection: Some(other),
        };
        wrong_collection.establish(BoundValidation::new(
            wrong.clone().into(),
            wrong.storage_places(),
        ));
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

        let mut guarded = ProgramState::empty();
        seed_exact(&mut guarded, &collection);
        seed_exact(&mut guarded, &subject);
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
            subject: subject.clone(),
            collection: Some(collection.clone()),
        };
        guarded.establish(BoundValidation::new(
            binding.clone().into(),
            binding.storage_places(),
        ));

        let mut unguarded = ProgramState::empty();
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
    fn task8_production_route_propagation_keeps_exact_variants_and_skips_satisfied() {
        let caller = FunctionKey::new("crate::f");
        let callee = FunctionKey::new("crate::sink");
        let collection = PlaceKey::new("_1");
        let subject = PlaceKey::new("_2");
        let call = task6_call(&collection, &subject);
        let validation = task6_contract();
        let callee_route = UnresolvedRouteFact::Conditional(validation.clone());
        let mut derived = FunctionSummary::empty(caller.clone());

        propagate_validation_route(
            &mut derived,
            &caller,
            &call,
            &callee,
            callee_route.clone(),
            ValidationRoute::Satisfied,
        );
        assert!(derived.requirements.is_empty());
        assert!(derived.validations.is_empty());
        assert!(derived.unresolved_predecessors.is_empty());

        propagate_validation_route(
            &mut derived,
            &caller,
            &call,
            &callee,
            callee_route.clone(),
            ValidationRoute::Conditional(validation.clone()),
        );
        propagate_validation_route(
            &mut derived,
            &caller,
            &call,
            &callee,
            callee_route.clone(),
            ValidationRoute::Hard(validation.requirement.clone()),
        );
        assert!(derived.validations.contains(&validation));
        assert!(derived.requirements.contains(&validation.requirement));
        assert_eq!(derived.unresolved_predecessors.len(), 2);
        assert!(derived
            .unresolved_predecessors
            .contains(&UnresolvedPredecessor {
                caller_route: UnresolvedRouteFact::Conditional(validation.clone()),
                callee_route: callee_route.clone(),
                callee: callee.clone(),
                call_point: call.point.clone(),
            }));
        assert!(derived
            .unresolved_predecessors
            .contains(&UnresolvedPredecessor {
                caller_route: UnresolvedRouteFact::Hard(validation.requirement.clone()),
                callee_route,
                callee: callee.clone(),
                call_point: call.point.clone(),
            }));

        let mut opaque_call = call;
        opaque_call.disposition = BoundaryDisposition::OpaqueDirect;
        let before = derived.clone();
        propagate_unresolved_route(
            &mut derived,
            &caller,
            &opaque_call,
            &callee,
            UnresolvedRouteFact::Hard(validation.requirement.clone()),
            UnresolvedRouteFact::Conditional(validation),
        );
        assert_eq!(derived, before);
    }

    #[test]
    fn task8_mutual_cycle_witness_keeps_exact_sink_anchor_and_one_real_cycle() {
        let root = FunctionKey::new("crate::read");
        let right = FunctionKey::new("crate::right");
        let left = FunctionKey::new("crate::left");
        let start = UnresolvedRouteFact::Hard(task6_contract().requirement);
        let middle = UnresolvedRouteFact::Hard({
            let mut requirement = start.requirement().clone();
            requirement.seed_id = "middle".into();
            requirement
        });
        let terminal = UnresolvedRouteFact::Hard({
            let mut requirement = start.requirement().clone();
            requirement.seed_id = "terminal".into();
            requirement.sink_function = left.clone();
            requirement
        });
        let first = UnresolvedPredecessor {
            caller_route: start.clone(),
            callee_route: middle.clone(),
            callee: right.clone(),
            call_point: ProgramPoint {
                function: root.clone(),
                block: 1,
                statement: 0,
                span: StableSpan::new(
                    "src/lib.rs",
                    StablePosition::new(10, 1),
                    StablePosition::new(10, 2),
                ),
            },
        };
        let second = UnresolvedPredecessor {
            caller_route: middle.clone(),
            callee_route: terminal.clone(),
            callee: left.clone(),
            call_point: ProgramPoint {
                function: right.clone(),
                block: 2,
                statement: 0,
                span: StableSpan::new(
                    "src/lib.rs",
                    StablePosition::new(20, 1),
                    StablePosition::new(20, 2),
                ),
            },
        };
        let route = SelectedUnresolvedRoute {
            root: UnresolvedRouteNode {
                owner: root.clone(),
                route: start,
            },
            terminal: UnresolvedRouteNode {
                owner: left.clone(),
                route: terminal.clone(),
            },
            steps: vec![
                super::super::summary::UnresolvedRouteStep {
                    caller: root.clone(),
                    edge: first,
                },
                super::super::summary::UnresolvedRouteStep {
                    caller: right.clone(),
                    edge: second,
                },
            ],
        };
        let cycle_span = StableSpan::new(
            "src/lib.rs",
            StablePosition::new(20, 1),
            StablePosition::new(20, 2),
        );
        let cycle = UnresolvedRouteCycle {
            nodes: BTreeSet::from([
                UnresolvedRouteNode {
                    owner: right,
                    route: middle,
                },
                UnresolvedRouteNode {
                    owner: left.clone(),
                    route: terminal.clone(),
                },
            ]),
            token: "scc:[crate::left,crate::right]".into(),
            span: cycle_span.clone(),
        };
        let witness =
            build_unresolved_witness(root, span(), terminal.requirement(), &route, Some(&cycle));
        assert_eq!(
            witness
                .steps
                .iter()
                .filter(|step| step.kind == WitnessStepKind::SccCycle)
                .count(),
            1
        );
        assert!(witness
            .steps
            .iter()
            .any(|step| { step.kind == WitnessStepKind::SccCycle && step.span == cycle_span }));
        let last_local = witness
            .steps
            .iter()
            .rev()
            .find(|step| step.kind == WitnessStepKind::LocalCall)
            .unwrap();
        assert_eq!(last_local.function, left);
        assert_eq!(witness.steps.last().unwrap().function, last_local.function);
        assert_eq!(
            witness.boundaries,
            vec![
                FunctionKey::new("crate::left"),
                FunctionKey::new("crate::right")
            ]
        );
        let kinds = witness
            .steps
            .iter()
            .map(|step| step.kind)
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                WitnessStepKind::Entry,
                WitnessStepKind::LocalCall,
                WitnessStepKind::SccCycle,
                WitnessStepKind::LocalCall,
                WitnessStepKind::Sink,
            ]
        );
    }

    #[test]
    fn task8_root_cycle_token_is_anchored_immediately_after_entry() {
        let root = FunctionKey::new("crate::direct");
        let route_fact = UnresolvedRouteFact::Hard(task6_contract().requirement);
        let route = SelectedUnresolvedRoute {
            root: UnresolvedRouteNode {
                owner: root.clone(),
                route: route_fact.clone(),
            },
            terminal: UnresolvedRouteNode {
                owner: root.clone(),
                route: route_fact.clone(),
            },
            steps: Vec::new(),
        };
        let cycle = UnresolvedRouteCycle {
            nodes: BTreeSet::from([route.root.clone()]),
            token: "scc:[crate::direct]".into(),
            span: StableSpan::new(
                "src/lib.rs",
                StablePosition::new(30, 1),
                StablePosition::new(30, 2),
            ),
        };
        let witness =
            build_unresolved_witness(root, span(), route_fact.requirement(), &route, Some(&cycle));
        assert_eq!(
            witness
                .steps
                .iter()
                .map(|step| step.kind)
                .collect::<Vec<_>>(),
            vec![
                WitnessStepKind::Entry,
                WitnessStepKind::SccCycle,
                WitnessStepKind::Sink,
            ]
        );
        assert_eq!(witness.steps[1].span, cycle.span);
    }

    #[test]
    fn task6_resolved_local_mutable_actual_havocs_only_on_the_normal_edge() {
        let referent = PlaceKey::new("_2");
        let reference = PlaceKey::new("_3");
        let destination = PlaceKey::new("_4");
        let mut before = ProgramState::empty();
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
            access_width: None,
            proof_definition: None,
            subject: referent.clone(),
            collection: None,
        };
        before.establish(BoundValidation::new(
            binding.clone().into(),
            binding.storage_places(),
        ));
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::mutate")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand("_3")],
                destination,
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![reference],
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };

        let mut normal = before.clone();
        apply_call_side_effects(&call, &point(1, 0), &mut normal);
        assert!(state_value(&normal, &referent).havoced);
        assert!(!normal.has_current_validation(&binding, binding.storage_places()));
        assert!(before.has_current_validation(&binding, binding.storage_places()));
    }

    fn task7_assignment(
        destination: &str,
        value: ValueModel,
        out_formal: Option<u32>,
        statement: u32,
    ) -> ProgramOp {
        ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(0, statement)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: PlaceKey::new(destination),
                destination_projection: None,
                value,
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal,
            }),
        }
    }

    #[test]
    fn task7_out_slot_strong_overwrite_exports_only_the_last_reachable_value() {
        let mut operations = IndexVec::new();
        operations.push(vec![
            task7_assignment(
                "(*_1)",
                ValueModel::Operand(place_operand("_2")),
                Some(1),
                0,
            ),
            task7_assignment(
                "(*_1)",
                ValueModel::Operand(constant_operand(0)),
                Some(1),
                1,
            ),
        ]);
        let mut edges = IndexVec::new();
        edges.push(EdgeTerm::Return);
        let mut successors = IndexVec::new();
        successors.push(BTreeSet::new());
        let mut entry = BlockState::empty();
        entry.set_value(
            PlaceKey::new("_2"),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(2)]),
                ..ValueFacts::default()
            },
        );
        let counts = operations.iter().map(Vec::len).collect();
        entry.set_value(
            PlaceKey::new("_1"),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                exact_roots: BTreeSet::from([PlaceKey::new("_1")]),
                ..ValueFacts::default()
            },
        );
        let solved = solve_cfg(
            &successors,
            &counts,
            entry,
            |block, index, state| apply_program_op(&operations[block][index], block, index, state),
            |_, _, _| {},
        );

        assert_eq!(
            reachable_out_dependencies(&edges, [1], &solved, &BTreeSet::from([1])),
            BTreeSet::from([OutDependency {
                formal_index: 1,
                value: AbstractValue::new([AbstractOrigin::Constant(0)]),
                may_skip_write: false,
            }])
        );
        let returned = solved.exit[BasicBlock::from_usize(0)]
            .as_ref()
            .expect("the return block is reachable");
        assert_eq!(
            state_value(returned, &PlaceKey::new("(*_1)")).value,
            AbstractValue::new([AbstractOrigin::Constant(0)])
        );
        assert_eq!(
            returned.may_values()[&PlaceKey::new("_1")].exact_roots,
            BTreeSet::from([PlaceKey::new("_1")])
        );
    }

    #[test]
    fn task7_nonreturn_exit_does_not_export_an_out_value() {
        let mut operations = IndexVec::new();
        operations.push(vec![task7_assignment(
            "(*_1)",
            ValueModel::Operand(place_operand("_2")),
            Some(1),
            0,
        )]);
        let mut edges = IndexVec::new();
        edges.push(EdgeTerm::None);
        let mut successors = IndexVec::new();
        successors.push(BTreeSet::new());
        let mut entry = BlockState::empty();
        entry.set_value(
            PlaceKey::new("_2"),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(2)]),
                ..ValueFacts::default()
            },
        );
        let counts = operations.iter().map(Vec::len).collect();
        let solved = solve_cfg(
            &successors,
            &counts,
            entry,
            |block, index, state| apply_program_op(&operations[block][index], block, index, state),
            |_, _, _| {},
        );

        assert!(reachable_out_dependencies(&edges, [1], &solved, &BTreeSet::new()).is_empty());
    }

    #[test]
    fn task7_partial_local_out_overlay_preserves_the_callers_old_may_origin_and_havoc() {
        let place = PlaceKey::new("_2");
        let mut state = ProgramState::empty();
        state.set_value(
            place.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Constant(0)]),
                havoced: true,
                ..ValueFacts::default()
            },
        );
        apply_local_outputs(
            &mut state,
            &BTreeMap::from([(
                place.clone(),
                MappedOutValue {
                    value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                    may_skip_write: true,
                },
            )]),
        );

        assert_eq!(
            state_value(&state, &place).value,
            AbstractValue::new([AbstractOrigin::Formal(1), AbstractOrigin::Constant(0)])
        );
        assert!(state_value(&state, &place).havoced);
    }

    #[test]
    fn task7_out_then_return_alias_preserves_normal_edge_order_and_channels() {
        let destination = PlaceKey::new("_2");
        let point = point(0, 0);
        let mut state = BlockState::empty();
        state.set_value(
            destination.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Constant(0)]),
                ..ValueFacts::default()
            },
        );
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::update")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand("_3")],
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(BasicBlock::from_usize(1)),
            point: point.clone(),
            call,
        };
        let outputs = BTreeMap::from([(
            point,
            MappedCallOutputs {
                returned: Some((
                    destination.clone(),
                    AbstractValue::new([AbstractOrigin::Constant(9)]),
                )),
                outs: BTreeMap::from([(
                    destination.clone(),
                    MappedOutValue {
                        value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                        may_skip_write: false,
                    },
                )]),
                relations: BTreeSet::new(),
                conditional_relations: BTreeSet::new(),
            },
        )]);

        apply_program_edge_with_outputs(&edge, BasicBlock::from_usize(1), &mut state, &outputs);
        assert_eq!(
            state_value(&state, &destination).value,
            AbstractValue::new([AbstractOrigin::Constant(9)])
        );
    }

    #[test]
    fn task7_return_to_mutable_formal_without_out_mapping_does_not_write_out_slot() {
        let mut callee = FunctionSummary::empty(FunctionKey::new("crate::callee"));
        callee.return_value = AbstractValue::new([AbstractOrigin::Constant(4)]);
        let outputs = map_structured_local_outputs(
            &BTreeSet::from([CallMapping::ReturnToDestination {
                destination: PlaceKey::new("_1"),
            }]),
            &callee,
            &[],
            &BTreeMap::from([(PlaceKey::new("_1"), 1)]),
            BTreeSet::new(),
        );

        assert!(outputs.outs.is_empty());
        assert_eq!(
            outputs.returned,
            Some((
                PlaceKey::new("_1"),
                AbstractValue::new([AbstractOrigin::Constant(4)])
            ))
        );
    }

    #[test]
    fn task7_definite_and_conditional_out_effects_replace_or_preserve_old_value() {
        let place = PlaceKey::new("_2");
        let old = AbstractOrigin::PublicField {
            def_path: "crate::Config::index".into(),
            span: span(),
        };
        for (may_skip_write, expected) in [
            (false, AbstractValue::new([AbstractOrigin::Constant(0)])),
            (
                true,
                AbstractValue::new([old.clone(), AbstractOrigin::Constant(0)]),
            ),
        ] {
            let mut state = BlockState::empty();
            state.set_value(
                place.clone(),
                ValueFacts {
                    value: AbstractValue::new([old.clone()]),
                    havoced: true,
                    ..ValueFacts::default()
                },
            );
            apply_local_outputs(
                &mut state,
                &BTreeMap::from([(
                    place.clone(),
                    MappedOutValue {
                        value: AbstractValue::new([AbstractOrigin::Constant(0)]),
                        may_skip_write,
                    },
                )]),
            );
            assert_eq!(state_value(&state, &place).value, expected);
            assert!(state_value(&state, &place).havoced);
        }
    }

    #[test]
    fn task7_frozen_must_output_replaces_an_earlier_direct_write_without_stale_history() {
        let base = PlaceKey::new("_1");
        let dereference = PlaceKey::new("(*_1)");
        let mut state = BlockState::empty();
        state.set_value(
            base.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                exact_roots: BTreeSet::from([base.clone()]),
                ..ValueFacts::default()
            },
        );
        apply_program_op(
            &task7_assignment(
                "(*_1)",
                ValueModel::Operand(constant_operand(0)),
                Some(1),
                0,
            ),
            BasicBlock::from_usize(0),
            0,
            &mut state,
        );

        let nested = MappedOutValue {
            value: AbstractValue::new([AbstractOrigin::Formal(2)]),
            may_skip_write: false,
        };
        apply_local_outputs(
            &mut state,
            &BTreeMap::from([(base.clone(), nested.clone()), (out_slot(1), nested)]),
        );

        assert_eq!(
            state_value(&state, &dereference).value,
            AbstractValue::new([AbstractOrigin::Formal(2)])
        );
        assert_eq!(
            state.may_values()[&out_slot(1)].value,
            AbstractValue::new([AbstractOrigin::Formal(2)])
        );
        assert_eq!(
            state.may_values()[&base].exact_roots,
            BTreeSet::from([base])
        );
    }

    #[test]
    fn task7_out_must_intersects_reachable_return_paths() {
        let entry = BasicBlock::from_usize(0);
        let writes = BasicBlock::from_usize(1);
        let skips = BasicBlock::from_usize(2);
        let returns = BasicBlock::from_usize(3);
        let mut operations = IndexVec::new();
        operations.push(Vec::new());
        operations.push(vec![task7_assignment(
            "(*_1)",
            ValueModel::Operand(constant_operand(0)),
            Some(1),
            0,
        )]);
        operations.push(Vec::new());
        operations.push(Vec::new());
        let mut edges = IndexVec::new();
        edges.push(EdgeTerm::None);
        edges.push(EdgeTerm::None);
        edges.push(EdgeTerm::None);
        edges.push(EdgeTerm::Return);
        let mut successors = IndexVec::new();
        successors.push(BTreeSet::from([writes, skips]));
        successors.push(BTreeSet::from([returns]));
        successors.push(BTreeSet::from([returns]));
        successors.push(BTreeSet::new());

        let solved = solve_out_must(&operations, &edges, &successors, &BTreeMap::new());
        assert_eq!(solved[entry], Some(BTreeSet::new()));
        assert!(solved[writes]
            .as_ref()
            .is_some_and(|facts| facts.contains(&1)));
        assert_eq!(solved[returns], Some(BTreeSet::new()));
    }

    #[test]
    fn task7_local_output_join_is_union_idempotent_and_skip_is_monotone() {
        let function = FunctionKey::new("crate::caller");
        let call_point = point(0, 0);
        let place = PlaceKey::new("_1");
        let mut current = BTreeMap::from([(
            function.clone(),
            BTreeMap::from([(
                call_point.clone(),
                MappedCallOutputs {
                    returned: None,
                    outs: BTreeMap::from([(
                        place.clone(),
                        MappedOutValue {
                            value: AbstractValue::new([AbstractOrigin::Constant(0)]),
                            may_skip_write: false,
                        },
                    )]),
                    relations: BTreeSet::new(),
                    conditional_relations: BTreeSet::new(),
                },
            )]),
        )]);
        let incoming = BTreeMap::from([(
            function.clone(),
            BTreeMap::from([(
                call_point.clone(),
                MappedCallOutputs {
                    returned: None,
                    outs: BTreeMap::from([(
                        place.clone(),
                        MappedOutValue {
                            value: AbstractValue::new([AbstractOrigin::Formal(2)]),
                            may_skip_write: true,
                        },
                    )]),
                    relations: BTreeSet::new(),
                    conditional_relations: BTreeSet::new(),
                },
            )]),
        )]);

        assert_eq!(
            join_local_outputs(&mut current, incoming.clone()),
            BTreeSet::from([function.clone()])
        );
        assert!(join_local_outputs(&mut current, incoming).is_empty());
        let joined = &current[&function][&call_point].outs[&place];
        assert_eq!(
            joined.value,
            AbstractValue::new([AbstractOrigin::Formal(2), AbstractOrigin::Constant(0)])
        );
        assert!(joined.may_skip_write);
    }

    #[test]
    fn task12_local_output_join_reports_only_changed_owners() {
        let unchanged = FunctionKey::new("crate::unchanged");
        let changed = FunctionKey::new("crate::changed");
        let point = point(0, 0);
        let place = PlaceKey::new("_1");
        let output = |origin| {
            LocalCallOutputs::from([(
                point.clone(),
                MappedCallOutputs {
                    returned: Some((
                        place.clone(),
                        AbstractValue::new([AbstractOrigin::Constant(origin)]),
                    )),
                    outs: BTreeMap::new(),
                    relations: BTreeSet::new(),
                    conditional_relations: BTreeSet::new(),
                },
            )])
        };
        let mut current =
            BTreeMap::from([(unchanged.clone(), output(1)), (changed.clone(), output(1))]);
        let incoming = BTreeMap::from([(unchanged, output(1)), (changed.clone(), output(2))]);

        assert_eq!(
            join_local_outputs(&mut current, incoming),
            BTreeSet::from([changed])
        );
    }

    #[test]
    fn task12_empty_return_mapping_does_not_dirty_or_materialize_a_caller() {
        let function = FunctionKey::new("crate::caller");
        let call_point = point(0, 0);
        let incoming = BTreeMap::from([(
            function,
            LocalCallOutputs::from([(
                call_point,
                MappedCallOutputs {
                    returned: Some((PlaceKey::new("_0"), AbstractValue::default())),
                    outs: BTreeMap::new(),
                    relations: BTreeSet::new(),
                    conditional_relations: BTreeSet::new(),
                },
            )]),
        )]);
        let mut current = BTreeMap::new();

        assert!(join_local_outputs(&mut current, incoming).is_empty());
        assert!(current.is_empty());
    }

    #[test]
    fn task12_relation_only_output_still_dirties_and_materializes_a_caller() {
        let function = FunctionKey::new("crate::caller");
        let call_point = point(0, 0);
        let relation = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(PlaceKey::new("_0")),
            RelationTerm::Length(PlaceKey::new("_1")),
        ));
        let incoming = BTreeMap::from([(
            function.clone(),
            LocalCallOutputs::from([(
                call_point.clone(),
                MappedCallOutputs {
                    returned: Some((PlaceKey::new("_0"), AbstractValue::default())),
                    outs: BTreeMap::new(),
                    relations: BTreeSet::from([relation.clone()]),
                    conditional_relations: BTreeSet::new(),
                },
            )]),
        )]);
        let mut current = BTreeMap::new();

        assert_eq!(
            join_local_outputs(&mut current, incoming),
            BTreeSet::from([function.clone()])
        );
        assert!(current[&function][&call_point].returned.is_none());
        assert_eq!(
            current[&function][&call_point].relations,
            BTreeSet::from([relation])
        );
    }

    #[test]
    fn task7_structured_out_updates_actual_and_summary_slot_without_stale_origin() {
        let mut callee = FunctionSummary::empty(FunctionKey::new("crate::set"));
        callee.out_dependencies.insert(OutDependency {
            formal_index: 1,
            value: AbstractValue::new([AbstractOrigin::Formal(2)]),
            may_skip_write: false,
        });
        let outputs = map_structured_local_outputs(
            &BTreeSet::from([CallMapping::OutToCaller {
                formal_index: 1,
                caller_place: PlaceKey::new("_1"),
            }]),
            &callee,
            &[
                AbstractValue::new([AbstractOrigin::Formal(1)]),
                AbstractValue::new([AbstractOrigin::Formal(2)]),
            ],
            &BTreeMap::from([(PlaceKey::new("_1"), 1)]),
            BTreeSet::from([1]),
        );

        assert_eq!(
            outputs.outs[&PlaceKey::new("_1")].value,
            AbstractValue::new([AbstractOrigin::Formal(2)])
        );
        assert_eq!(
            outputs.outs[&out_slot(1)].value,
            AbstractValue::new([AbstractOrigin::Formal(2)])
        );
        assert!(!outputs.outs[&PlaceKey::new("_1")].may_skip_write);
    }

    #[test]
    fn task7_call_destination_out_slot_uses_exact_return_and_is_normal_edge_only() {
        let destination = PlaceKey::new("(*_1)");
        let normal = BasicBlock::from_usize(1);
        let unwind_target = BasicBlock::from_usize(2);
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::constant")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: Vec::new(),
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: Some(1),
        };
        let call_point = point(0, 0);
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: call_point.clone(),
            call,
        };
        let outputs = BTreeMap::from([(
            call_point,
            MappedCallOutputs {
                returned: Some((
                    destination.clone(),
                    AbstractValue::new([AbstractOrigin::Constant(0)]),
                )),
                outs: BTreeMap::new(),
                relations: BTreeSet::new(),
                conditional_relations: BTreeSet::new(),
            },
        )]);
        let mut state = BlockState::empty();
        state.set_value(
            PlaceKey::new("_1"),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                exact_roots: BTreeSet::from([PlaceKey::new("_1")]),
                ..ValueFacts::default()
            },
        );
        let mut unwind = state.clone();
        apply_program_edge_with_outputs(&edge, unwind_target, &mut unwind, &outputs);
        apply_program_edge_with_outputs(&edge, normal, &mut state, &outputs);
        assert_eq!(
            state.may_values()[&out_slot(1)].value,
            AbstractValue::new([AbstractOrigin::Constant(0)])
        );
        assert_eq!(
            state_value(&state, &destination).value,
            AbstractValue::new([AbstractOrigin::Constant(0)])
        );
        assert_eq!(
            state.may_values()[&PlaceKey::new("_1")].exact_roots,
            BTreeSet::from([PlaceKey::new("_1")])
        );
        assert_eq!(unwind.may_values().get(&out_slot(1)), None);
        assert_eq!(
            state_value(&unwind, &destination).value,
            AbstractValue::new([AbstractOrigin::Formal(1)])
        );

        let mut operations = IndexVec::new();
        operations.push(vec![ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Call(match edge {
                EdgeTerm::CallNormal { call, .. } => call,
                _ => unreachable!(),
            }),
        }]);
        operations.push(Vec::new());
        operations.push(Vec::new());
        let mut edges = IndexVec::new();
        edges.push(EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: point(0, 0),
            call: match &operations[BasicBlock::from_usize(0)][0].kind {
                ProgramOpKind::Call(call) => call.clone(),
                _ => unreachable!(),
            },
        });
        edges.push(EdgeTerm::Return);
        edges.push(EdgeTerm::Return);
        let mut successors = IndexVec::new();
        successors.push(BTreeSet::from([normal, unwind_target]));
        successors.push(BTreeSet::new());
        successors.push(BTreeSet::new());
        let must = solve_out_must(&operations, &edges, &successors, &outputs);
        assert!(must[normal]
            .as_ref()
            .is_some_and(|facts| facts.contains(&1)));
        assert!(must[unwind_target]
            .as_ref()
            .is_some_and(|facts| !facts.contains(&1)));
    }

    #[test]
    fn task7_production_mapping_adds_only_exact_resolved_local_out_actuals() {
        let reference = PlaceKey::new("_3");
        let referent = PlaceKey::new("_2");
        let mut state = BlockState::empty();
        state.set_value(
            reference.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([referent.clone()]),
                value_flow: BTreeSet::from([referent.clone()]),
                ..ValueFacts::default()
            },
        );
        let mut call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::set")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand("_3")],
                destination: PlaceKey::new("_4"),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![reference.clone()],
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        assert!(
            local_call_boundary_mapping(&call, &state).contains(&CallMapping::OutToCaller {
                formal_index: 1,
                caller_place: referent,
            })
        );

        let mut ambiguous = state_value(&state, &reference);
        ambiguous.exact_roots.insert(PlaceKey::new("_5"));
        state.set_value(reference, ambiguous);
        assert!(!local_call_boundary_mapping(&call, &state)
            .iter()
            .any(|mapping| matches!(mapping, CallMapping::OutToCaller { .. })));
        call.descriptor.disposition = BoundaryDisposition::OpaqueDirect;
        assert!(!local_call_boundary_mapping(&call, &BlockState::empty())
            .iter()
            .any(|mapping| matches!(mapping, CallMapping::OutToCaller { .. })));
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
            unsafe_sites: Vec::new(),
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: condition.clone(),
                destination_projection: None,
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
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
            unsafe_sites: Vec::new(),
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: length.clone(),
                destination_projection: None,
                value: ValueModel::Length(place_operand(&collection.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
        };
        let compare = |destination: PlaceKey| ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(0, 2)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination,
                destination_projection: None,
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
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
        assert!(!has_pending_validation(&stale, &stale_condition));

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
        assert!(!has_pending_validation(&conflated, &conflated_condition));
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
        invalidate_relation_evidence(&mut written, &canonical_storage(&collection));
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
            unsafe_sites: Vec::new(),
            point: Some(point(0, 1)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: length.clone(),
                destination_projection: None,
                value: ValueModel::Length(place_operand(&collection.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
        };
        apply_program_op(&length_op, BasicBlock::from_usize(0), 1, &mut state);
        assert!(!state.may_values()[&length].havoced);
        let compare = ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(0, 2)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: condition.clone(),
                destination_projection: None,
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
        };
        apply_program_op(&compare, BasicBlock::from_usize(0), 2, &mut state);
        assert!(state.pending_must().contains_key(&condition));
        state.record_may_write(canonical_storage(&collection), StaticWriteToken::new(0, 3));
        invalidate_relation_evidence(&mut state, &canonical_storage(&collection));
        let stale_condition = PlaceKey::new("_5");
        let mut stale_compare = compare;
        if let ProgramOpKind::Assign(assignment) = &mut stale_compare.kind {
            assignment.destination = stale_condition.clone();
        }
        apply_program_op(&stale_compare, BasicBlock::from_usize(0), 4, &mut state);
        assert!(!has_pending_validation(&state, &stale_condition));
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
            access_width: None,
            proof_definition: None,
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
            unsafe_sites: Vec::new(),
            point: Some(point(0, 1)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: copied.clone(),
                destination_projection: None,
                value: ValueModel::Operand(place_operand(&source.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
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
                storage_provenance: StorageProvenance::EntryFormal {
                    owner: FunctionKey::new("crate::f"),
                    index: 1,
                },
                // This test constructs the entry state directly. Production
                // field projections receive the same marker from
                // `entry_field_projection_value`.
                may_hold_entry_public_field: true,
                ..ValueFacts::default()
            },
        );
        let field = OperandModel {
            operand: CallOperand::Place(source.clone()),
            field_origin: Some(FieldAccessModel {
                access_owner: FunctionKey::new("crate::f"),
                full_place: PlaceKey::new("_1.0"),
                base: source.clone(),
                projection_path: Some(vec![ProjectionKey::Field(0)]),
                projection_unknown: false,
                def_path: "crate::S::field".into(),
                adt_kind: PublicFieldAdtKind::Struct,
                path_public: true,
                type_class: InfluenceTypeClass::Aggregate,
                span: span(),
            }),
            pure_deref_base: None,
            exact_projection: None,
            projection_path: None,
            range_index_kind: None,
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
            access_width: None,
            proof_definition: None,
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
                        unsafe_sites: Vec::new(),
                        point: Some(point(block as u32, 0)),
                        kind: ProgramOpKind::Assign(AssignmentModel {
                            destination: length.clone(),
                            destination_projection: None,
                            value: ValueModel::Length(place_operand(&slice_temp.0)),
                            field_origin: None,
                            raw_read_source: None,
                            legacy_write: false,
                            out_formal: None,
                        }),
                    },
                    BasicBlock::from_usize(block),
                    0,
                    &mut state,
                );
                apply_program_op(
                    &ProgramOp {
                        unsafe_sites: Vec::new(),
                        point: Some(point(block as u32, 1)),
                        kind: ProgramOpKind::Assign(AssignmentModel {
                            destination: condition.clone(),
                            destination_projection: None,
                            value: ValueModel::Compare {
                                op: BinOp::Ge,
                                left: place_operand(&index_temp.0),
                                right: place_operand(&length.0),
                            },
                            field_origin: None,
                            raw_read_source: None,
                            legacy_write: false,
                            out_formal: None,
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
            access_width: None,
            proof_definition: None,
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
            projection_path: None,
            range_index_kind: None,
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
        let mut expanded_source = exact_value(&source);
        expanded_source.semantic_unknown = true;
        expanded.set_value(source.clone(), expanded_source);
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
            projection_path: Some(vec![ProjectionKey::Deref]),
            range_index_kind: None,
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
            projection_path: None,
            range_index_kind: None,
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
            projection_path: None,
            range_index_kind: None,
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
            access_width: None,
            proof_definition: None,
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
    fn task9_lt_true_edge_establishes_only_the_exact_current_bound() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_3");
        let condition = PlaceKey::new("_4");
        let mut state = BlockState::empty();
        seed_exact(&mut state, &collection);
        seed_exact(&mut state, &index);
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length.clone(),
                    destination_projection: None,
                    value: ValueModel::Length(place_operand(&collection.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            0,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Lt,
                        left: place_operand(&index.0),
                        right: place_operand(&length.0),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            1,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
            subject: index,
            collection: Some(collection),
        };
        assert!(state.pending_is_current(&condition, &binding, true));

        let true_target = BasicBlock::from_usize(1);
        let false_target = BasicBlock::from_usize(2);
        let edge = EdgeTerm::Switch {
            condition,
            false_target,
            true_target,
        };
        let mut on_true = state.clone();
        apply_program_edge(&edge, true_target, &mut on_true);
        assert!(on_true.has_current_validation(&binding, binding.storage_places()));
        let mut on_false = state;
        apply_program_edge(&edge, false_target, &mut on_false);
        assert!(!on_false.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn task9_assert_success_establishes_only_matching_in_bounds_pending() {
        let condition = PlaceKey::new("_3");
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
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
            condition.clone(),
            [(binding.clone(), true, binding.storage_places())],
        );
        apply_program_edge(&edge, target, &mut state);
        assert!(state.has_current_validation(&binding, binding.storage_places()));

        let nonempty = ValidationBinding {
            predicate: Predicate::NonEmpty,
            access_width: None,
            proof_definition: None,
            subject: PlaceKey::new("_8"),
            collection: None,
        };
        let mut wrong_predicate = BlockState::empty();
        wrong_predicate.set_pending(
            condition.clone(),
            [(nonempty.clone(), true, nonempty.storage_places())],
        );
        apply_program_edge(&edge, target, &mut wrong_predicate);
        assert!(!wrong_predicate.has_current_validation(&nonempty, nonempty.storage_places()));

        let mut wrong_polarity = BlockState::empty();
        wrong_polarity.set_pending(
            condition.clone(),
            [(binding.clone(), false, binding.storage_places())],
        );
        apply_program_edge(&edge, target, &mut wrong_polarity);
        assert!(!wrong_polarity.has_current_validation(&binding, binding.storage_places()));

        let mut false_expected = BlockState::empty();
        false_expected.set_pending(
            condition,
            [(binding.clone(), false, binding.storage_places())],
        );
        apply_program_edge(
            &EdgeTerm::AssertPlumbingOnly {
                condition: PlaceKey::new("_3"),
                expected: false,
                target,
            },
            target,
            &mut false_expected,
        );
        assert!(false_expected.has_current_validation(&binding, binding.storage_places()));

        let mut unwind = BlockState::empty();
        unwind.set_pending(
            PlaceKey::new("_3"),
            [(binding.clone(), true, binding.storage_places())],
        );
        apply_program_edge(
            &edge,
            BasicBlock::from_usize(target.index() + 1),
            &mut unwind,
        );
        assert!(!unwind.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn task9_len_zero_spellings_require_literal_zero_and_one_current_length() {
        let collection = PlaceKey::new("_1");
        let length = PlaceKey::new("_2");
        let condition = PlaceKey::new("_3");
        let mut state = BlockState::empty();
        seed_exact(&mut state, &collection);
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length.clone(),
                    destination_projection: None,
                    value: ValueModel::Length(place_operand(&collection.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            0,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::NonEmpty,
            access_width: None,
            proof_definition: None,
            subject: collection.clone(),
            collection: None,
        };
        assert!(state.pending_is_current(&length, &binding, true));

        for (op, polarity) in [(BinOp::Eq, false), (BinOp::Ne, true)] {
            let mut spelling = state.clone();
            apply_program_op(
                &ProgramOp {
                    unsafe_sites: Vec::new(),
                    point: Some(point(0, 1)),
                    kind: ProgramOpKind::Assign(AssignmentModel {
                        destination: condition.clone(),
                        destination_projection: None,
                        value: ValueModel::Compare {
                            op,
                            left: place_operand(&length.0),
                            right: constant_operand(0),
                        },
                        field_origin: None,
                        raw_read_source: None,
                        legacy_write: false,
                        out_formal: None,
                    }),
                },
                BasicBlock::from_usize(0),
                1,
                &mut spelling,
            );
            assert!(spelling.pending_is_current(&condition, &binding, polarity));
        }

        let mut nonzero = state;
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 2)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Ne,
                        left: place_operand(&length.0),
                        right: constant_operand(1),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            2,
            &mut nonzero,
        );
        assert!(!has_pending_validation(&nonzero, &condition));

        let mut stale = BlockState::empty();
        seed_exact(&mut stale, &collection);
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length.clone(),
                    destination_projection: None,
                    value: ValueModel::Length(place_operand(&collection.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            0,
            &mut stale,
        );
        stale.record_may_write(canonical_storage(&collection), StaticWriteToken::new(0, 1));
        assert!(!stale.pending_is_current(&length, &binding, true));

        let other_collection = PlaceKey::new("_9");
        let mut ambiguous = BlockState::empty();
        seed_exact(&mut ambiguous, &collection);
        seed_exact(&mut ambiguous, &other_collection);
        let mut joined_length = eval_value_model(
            &ambiguous,
            &ValueModel::Length(place_operand(&collection.0)),
            StaticWriteToken::new(0, 0),
        );
        joined_length.join_may(&eval_value_model(
            &ambiguous,
            &ValueModel::Length(place_operand(&other_collection.0)),
            StaticWriteToken::new(0, 4),
        ));
        ambiguous.set_value(length.clone(), joined_length);
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 5)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Eq,
                        left: place_operand(&length.0),
                        right: constant_operand(0),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            5,
            &mut ambiguous,
        );
        assert!(ambiguous.pending_must().get(&condition).is_none());
    }

    fn task9_checked_add_call(offset: &PlaceKey, width: u64, destination: &PlaceKey) -> CallModel {
        CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&offset.0), constant_operand(width)],
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::CheckedAdd {
                offset: place_operand(&offset.0),
                width: constant_operand(width),
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        }
    }

    #[test]
    fn task9_checked_add_requires_success_branch_and_exact_width_binding() {
        let offset = PlaceKey::new("_2");
        let collection = PlaceKey::new("_1");
        let option = PlaceKey::new("_3");
        let branch = PlaceKey::new("_4");
        let discriminant = PlaceKey::new("_5");
        let end = PlaceKey::new("_6");
        let length = PlaceKey::new("_7");
        let condition = PlaceKey::new("_8");
        let normal = BasicBlock::from_usize(1);
        let continue_block = BasicBlock::from_usize(2);
        let break_block = BasicBlock::from_usize(3);
        let mut state = BlockState::empty();
        seed_exact(&mut state, &offset);
        seed_exact(&mut state, &collection);

        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: task9_checked_add_call(&offset, 4, &option),
            },
            normal,
            &mut state,
        );
        assert!(current_checked_end(&state, &state_value(&state, &option)).is_none());
        let branch_call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&option.0)],
                destination: branch.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::CheckedTryBranch(place_operand(&option.0)),
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 0),
                call: branch_call,
            },
            normal,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: discriminant.clone(),
                    destination_projection: None,
                    value: ValueModel::Discriminant(place_operand(&branch.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            normal,
            1,
            &mut state,
        );
        let switch = EdgeTerm::Switch {
            condition: discriminant,
            false_target: continue_block,
            true_target: break_block,
        };
        let mut on_break = state.clone();
        apply_program_edge(&switch, break_block, &mut on_break);
        apply_program_edge(&switch, continue_block, &mut state);

        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(2, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: end.clone(),
                    destination_projection: None,
                    value: ValueModel::Operand(place_operand(&branch.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            continue_block,
            0,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(2, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length.clone(),
                    destination_projection: None,
                    value: ValueModel::Length(place_operand(&collection.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            continue_block,
            1,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(2, 2)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Gt,
                        left: place_operand(&end.0),
                        right: place_operand(&length.0),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            continue_block,
            2,
            &mut state,
        );
        let binding = ValidationBinding::range(offset.clone(), Some(collection.clone()), 4, None);
        assert!(state.pending_is_current(&condition, &binding, false));
        assert!(!state.pending_is_current(
            &condition,
            &ValidationBinding::range(offset.clone(), Some(collection), 8, None),
            false,
        ));

        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(3, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length,
                    destination_projection: None,
                    value: ValueModel::Length(place_operand("_1")),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            break_block,
            0,
            &mut on_break,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(3, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Gt,
                        left: place_operand(&end.0),
                        right: place_operand("_7"),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            break_block,
            1,
            &mut on_break,
        );
        assert!(on_break.pending_must().get(&condition).is_none());
    }

    #[test]
    fn task9_saturating_limit_requires_known_sufficient_extent_and_current_storage() {
        let collection = PlaceKey::new("_1");
        let offset = PlaceKey::new("_2");
        let length = PlaceKey::new("_3");
        let limit = PlaceKey::new("_4");
        let condition = PlaceKey::new("_5");
        let mut state = BlockState::empty();
        seed_exact(&mut state, &collection);
        seed_exact(&mut state, &offset);
        let mut collection_value = state.may_values()[&collection].clone();
        collection_value.known_lengths.insert(8);
        collection_value.known_length_unknown = false;
        state.set_value(collection.clone(), collection_value);
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length.clone(),
                    destination_projection: None,
                    value: ValueModel::Length(place_operand(&collection.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            0,
            &mut state,
        );
        let limit_value = eval_registry_value(
            &state,
            &RegistryValueModel::SaturatingSub(vec![place_operand(&length.0), constant_operand(4)]),
            &point(0, 1),
        );
        state.set_value(limit.clone(), limit_value);
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 2)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Gt,
                        left: place_operand(&offset.0),
                        right: place_operand(&limit.0),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            2,
            &mut state,
        );
        let binding = ValidationBinding::range(offset, Some(collection.clone()), 4, None);
        assert!(state.pending_is_current(&condition, &binding, false));

        let mut unknown = state.clone();
        let unknown_limit = eval_registry_value(
            &unknown,
            &RegistryValueModel::SaturatingSub(vec![place_operand(&length.0), constant_operand(9)]),
            &point(0, 3),
        );
        unknown.set_value(limit, unknown_limit);
        assert!(
            current_range_limit(&unknown, &state_value(&unknown, &PlaceKey::new("_4"))).is_none()
        );

        let mut mixed_length = state_value(&state, &length);
        mixed_length.known_length_unknown = true;
        let mut mixed = state.clone();
        mixed.set_value(length.clone(), mixed_length);
        let mixed_limit = eval_registry_value(
            &mixed,
            &RegistryValueModel::SaturatingSub(vec![place_operand(&length.0), constant_operand(4)]),
            &point(0, 4),
        );
        assert!(mixed_limit.range_limit_unknown);
        assert!(current_range_limit(&mixed, &mixed_limit).is_none());

        let written = canonical_storage(&collection);
        state.record_may_write(written.clone(), StaticWriteToken::new(0, 4));
        invalidate_relation_evidence(&mut state, &written);
        assert!(!state.pending_is_current(&condition, &binding, false));
    }

    #[test]
    fn task9_range_sink_binding_requires_exact_base_offset_and_width() {
        let pointer = ValueFacts {
            pointer_base: BTreeSet::from([PlaceKey::new("_1")]),
            pointer_offset: BTreeSet::from([PlaceKey::new("_2")]),
            ..ValueFacts::default()
        };
        assert_eq!(
            range_binding_for_pointer(&pointer, 4),
            Some(ValidationBinding::range(
                PlaceKey::new("_2"),
                Some(PlaceKey::new("_1")),
                4,
                None,
            ))
        );
        let mut ambiguous = pointer;
        ambiguous.pointer_base.insert(PlaceKey::new("_9"));
        assert!(range_binding_for_pointer(&ambiguous, 4).is_none());
    }

    #[test]
    fn task9_relation_joins_are_sticky_commutative_and_idempotent() {
        let offset = PlaceKey::new("_2");
        let collection = PlaceKey::new("_1");
        let checked = CheckedEndAtom {
            offset: offset.clone(),
            width: 4,
            definition: StaticWriteToken::new(1, 0),
            captured_versions: BTreeSet::new(),
            branch_ready: true,
            invalidated: false,
        };
        let limit = RangeLimitAtom {
            collection: collection.clone(),
            width: 4,
            definition: StaticWriteToken::new(2, 0),
            captured_versions: BTreeSet::new(),
            invalidated: false,
        };
        let evidence = ValueFacts {
            checked_end_atoms: BTreeSet::from([checked]),
            range_limit_atoms: BTreeSet::from([limit]),
            ..ValueFacts::default()
        };
        let unknown = ValueFacts {
            checked_end_unknown: true,
            range_limit_unknown: true,
            known_length_unknown: true,
            ..ValueFacts::default()
        };
        let mut left = evidence.clone();
        left.join_may(&unknown);
        let mut right = unknown;
        right.join_may(&evidence);
        assert_eq!(left, right);
        assert!(!left.join_may(&left.clone()));
        assert!(current_checked_end(&BlockState::empty(), &left).is_none());
        assert!(current_range_limit(&BlockState::empty(), &left).is_none());
    }

    #[test]
    fn task12_value_join_reports_every_field_growth_then_idempotence() {
        let place = PlaceKey::new("_1");
        let definition = StaticWriteToken::new(1, 0);
        let cases = vec![
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Constant(1)]),
                ..ValueFacts::default()
            },
            ValueFacts {
                dependencies: BTreeSet::from([place.clone()]),
                ..ValueFacts::default()
            },
            ValueFacts {
                value_flow: BTreeSet::from([place.clone()]),
                ..ValueFacts::default()
            },
            ValueFacts {
                exact_roots: BTreeSet::from([place.clone()]),
                ..ValueFacts::default()
            },
            ValueFacts {
                semantic_unknown: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                len_of: BTreeSet::from([place.clone()]),
                ..ValueFacts::default()
            },
            ValueFacts {
                length_atoms: BTreeSet::from([LengthAtom {
                    collection: place.clone(),
                    definition,
                    captured_versions: BTreeSet::new(),
                    invalidated: false,
                }]),
                ..ValueFacts::default()
            },
            ValueFacts {
                length_relation_unknown: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                utf8_atoms: BTreeSet::from([Utf8Atom {
                    bytes: place.clone(),
                    definition,
                    captured_versions: BTreeSet::new(),
                    invalidated: false,
                }]),
                ..ValueFacts::default()
            },
            ValueFacts {
                utf8_relation_unknown: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                checked_end_atoms: BTreeSet::from([CheckedEndAtom {
                    offset: place.clone(),
                    width: 4,
                    definition,
                    captured_versions: BTreeSet::new(),
                    branch_ready: true,
                    invalidated: false,
                }]),
                ..ValueFacts::default()
            },
            ValueFacts {
                checked_end_unknown: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                range_limit_atoms: BTreeSet::from([RangeLimitAtom {
                    collection: place.clone(),
                    width: 4,
                    definition,
                    captured_versions: BTreeSet::new(),
                    invalidated: false,
                }]),
                ..ValueFacts::default()
            },
            ValueFacts {
                range_limit_unknown: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                known_lengths: BTreeSet::from([4]),
                ..ValueFacts::default()
            },
            ValueFacts {
                known_length_unknown: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                may_be_non_length: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                range_end: BTreeSet::from([place.clone()]),
                ..ValueFacts::default()
            },
            ValueFacts {
                pointer_base: BTreeSet::from([place.clone()]),
                ..ValueFacts::default()
            },
            ValueFacts {
                pointer_offset: BTreeSet::from([place.clone()]),
                ..ValueFacts::default()
            },
            ValueFacts {
                maybe_uninit_initialized: BTreeSet::from([true]),
                ..ValueFacts::default()
            },
            ValueFacts {
                generic_capability: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                ffi_output: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                open_behavior: true,
                ..ValueFacts::default()
            },
            ValueFacts {
                havoced: true,
                ..ValueFacts::default()
            },
        ];

        for (index, incoming) in cases.into_iter().enumerate() {
            let mut current = ValueFacts::default();
            assert!(
                current.join_may(&incoming),
                "field case {index} did not grow"
            );
            assert!(
                !current.join_may(&incoming),
                "field case {index} was not idempotent"
            );
        }
    }

    #[test]
    fn task12_relation_join_reports_growth_inside_stable_atom_keys() {
        let place = PlaceKey::new("_1");
        let definition = StaticWriteToken::new(1, 0);
        let first = StaticWriteToken::new(2, 0);
        let second = StaticWriteToken::new(3, 0);
        let facts = |version, invalidated| ValueFacts {
            length_atoms: BTreeSet::from([LengthAtom {
                collection: place.clone(),
                definition,
                captured_versions: BTreeSet::from([version]),
                invalidated,
            }]),
            utf8_atoms: BTreeSet::from([Utf8Atom {
                bytes: place.clone(),
                definition,
                captured_versions: BTreeSet::from([version]),
                invalidated,
            }]),
            checked_end_atoms: BTreeSet::from([CheckedEndAtom {
                offset: place.clone(),
                width: 4,
                definition,
                captured_versions: BTreeSet::from([version]),
                branch_ready: true,
                invalidated,
            }]),
            range_limit_atoms: BTreeSet::from([RangeLimitAtom {
                collection: place.clone(),
                width: 4,
                definition,
                captured_versions: BTreeSet::from([version]),
                invalidated,
            }]),
            ..ValueFacts::default()
        };
        let mut current = facts(first, false);
        let version_growth = facts(second, false);
        let invalidation_growth = facts(first, true);

        assert!(current.join_may(&version_growth));
        assert_eq!(current.length_atoms.len(), 1);
        assert_eq!(current.utf8_atoms.len(), 1);
        assert_eq!(current.checked_end_atoms.len(), 1);
        assert_eq!(current.range_limit_atoms.len(), 1);
        for (versions, invalidated) in [
            current
                .length_atoms
                .iter()
                .next()
                .map(|atom| (&atom.captured_versions, atom.invalidated)),
            current
                .utf8_atoms
                .iter()
                .next()
                .map(|atom| (&atom.captured_versions, atom.invalidated)),
            current
                .checked_end_atoms
                .iter()
                .next()
                .map(|atom| (&atom.captured_versions, atom.invalidated)),
            current
                .range_limit_atoms
                .iter()
                .next()
                .map(|atom| (&atom.captured_versions, atom.invalidated)),
        ]
        .into_iter()
        .flatten()
        {
            assert_eq!(versions, &BTreeSet::from([first, second]));
            assert!(!invalidated);
        }

        assert!(current.join_may(&invalidation_growth));
        assert_eq!(current.length_atoms.len(), 1);
        assert_eq!(current.utf8_atoms.len(), 1);
        assert_eq!(current.checked_end_atoms.len(), 1);
        assert_eq!(current.range_limit_atoms.len(), 1);
        assert!([
            current.length_atoms.iter().all(|atom| atom.invalidated),
            current.utf8_atoms.iter().all(|atom| atom.invalidated),
            current
                .checked_end_atoms
                .iter()
                .all(|atom| atom.invalidated),
            current
                .range_limit_atoms
                .iter()
                .all(|atom| atom.invalidated),
        ]
        .into_iter()
        .all(|invalidated| invalidated));
        assert!(!current.join_may(&version_growth));
        assert!(!current.join_may(&invalidation_growth));
    }

    #[test]
    fn task12_value_join_is_commutative_associative_and_idempotent() {
        let collection = PlaceKey::new("_1");
        let definition = StaticWriteToken::new(1, 0);
        let facts = |origin, version, invalidated| ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Constant(origin)]),
            length_atoms: BTreeSet::from([LengthAtom {
                collection: collection.clone(),
                definition,
                captured_versions: BTreeSet::from([StaticWriteToken::new(version, 0)]),
                invalidated,
            }]),
            semantic_unknown: invalidated,
            ..ValueFacts::default()
        };
        let first = facts(1, 2, false);
        let second = facts(2, 3, false);
        let third = facts(3, 4, true);

        let mut first_second = first.clone();
        assert!(first_second.join_may(&second));
        let mut second_first = second.clone();
        assert!(second_first.join_may(&first));
        assert_eq!(first_second, second_first);

        let mut left_associative = first_second;
        assert!(left_associative.join_may(&third));
        let mut second_third = second;
        assert!(second_third.join_may(&third));
        let mut right_associative = first.clone();
        assert!(right_associative.join_may(&second_third));
        assert_eq!(left_associative, right_associative);

        let mut idempotent = first.clone();
        assert!(!idempotent.join_may(&first));
        assert_eq!(idempotent, first);
    }

    #[test]
    fn task12_endpoint_provenance_clones_share_storage_until_growth() {
        let mut original = endpoint_value(EndpointNode::Formal {
            owner: FunctionKey::new("crate::f"),
            index: 1,
        });
        let shared = original.clone();

        assert!(std::sync::Arc::ptr_eq(
            &original.endpoint_sources,
            &shared.endpoint_sources,
        ));
        assert!(std::sync::Arc::ptr_eq(
            &original.endpoint_edges,
            &shared.endpoint_edges,
        ));
        assert!(!original.join_may(&shared));
        assert!(std::sync::Arc::ptr_eq(
            &original.endpoint_sources,
            &shared.endpoint_sources,
        ));
        assert!(std::sync::Arc::ptr_eq(
            &original.endpoint_edges,
            &shared.endpoint_edges,
        ));

        assert!(original.join_may(&endpoint_value(EndpointNode::Formal {
            owner: FunctionKey::new("crate::f"),
            index: 2,
        })));
        assert!(!std::sync::Arc::ptr_eq(
            &original.endpoint_sources,
            &shared.endpoint_sources,
        ));
        assert!(std::sync::Arc::ptr_eq(
            &original.endpoint_edges,
            &shared.endpoint_edges,
        ));
    }

    #[test]
    fn task12_relation_invalidation_clones_only_relevant_values() {
        let written = PlaceKey::new("_1");
        let unrelated = PlaceKey::new("_9");
        let mut state = BlockState::empty();
        state.set_value(
            PlaceKey::new("_length"),
            ValueFacts {
                length_atoms: BTreeSet::from([LengthAtom {
                    collection: written.clone(),
                    definition: StaticWriteToken::new(0, 0),
                    captured_versions: BTreeSet::new(),
                    invalidated: false,
                }]),
                ..ValueFacts::default()
            },
        );
        state.set_value(
            PlaceKey::new("_utf8"),
            ValueFacts {
                utf8_atoms: BTreeSet::from([Utf8Atom {
                    bytes: unrelated,
                    definition: StaticWriteToken::new(0, 1),
                    captured_versions: BTreeSet::new(),
                    invalidated: false,
                }]),
                ..ValueFacts::default()
            },
        );

        let _ = take_relation_invalidation_clone_count();
        invalidate_relation_evidence(&mut state, &written);
        assert_eq!(take_relation_invalidation_clone_count(), 1);
        assert!(state.may_values()[&PlaceKey::new("_length")]
            .length_atoms
            .iter()
            .all(|atom| atom.invalidated));
        assert!(state.may_values()[&PlaceKey::new("_utf8")]
            .utf8_atoms
            .iter()
            .all(|atom| !atom.invalidated));

        invalidate_relation_evidence(&mut state, &PlaceKey::new("_never"));
        assert_eq!(take_relation_invalidation_clone_count(), 0);
    }

    fn task9_utf8_result_call(bytes: &PlaceKey, destination: &PlaceKey) -> CallModel {
        CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&bytes.0)],
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: Some(PredicateModel::Utf8Result {
                bytes: place_operand(&bytes.0),
            }),
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        }
    }

    #[test]
    fn internal_utf8_validation_shape_is_exact_and_fail_closed() {
        let exact = Utf8ValidationCallShape {
            caller_is_str_from_utf8: true,
            resolved_local: true,
            one_argument: true,
            actual_matches_caller_input: true,
            callee_input_matches_actual: true,
            destination_matches_callee_output: true,
            validator_ok_is_unit: true,
            error_matches: true,
        };
        assert!(exact_utf8_validation_call_shape(exact));
        for invalid in [
            Utf8ValidationCallShape {
                caller_is_str_from_utf8: false,
                ..exact
            },
            Utf8ValidationCallShape {
                resolved_local: false,
                ..exact
            },
            Utf8ValidationCallShape {
                one_argument: false,
                ..exact
            },
            Utf8ValidationCallShape {
                actual_matches_caller_input: false,
                ..exact
            },
            Utf8ValidationCallShape {
                callee_input_matches_actual: false,
                ..exact
            },
            Utf8ValidationCallShape {
                destination_matches_callee_output: false,
                ..exact
            },
            Utf8ValidationCallShape {
                validator_ok_is_unit: false,
                ..exact
            },
            Utf8ValidationCallShape {
                error_matches: false,
                ..exact
            },
        ] {
            assert!(!exact_utf8_validation_call_shape(invalid));
        }
    }

    fn historical_gitoxide_unchecked_utf8_call() -> CallSite {
        CallSite {
            callee: None,
            raw_def: None,
            disposition: BoundaryDisposition::ModeledRegistry,
            args: vec![CallOperand::Place(PlaceKey::new("_1"))],
            arg_values: vec![ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                exact_roots: BTreeSet::from([PlaceKey::new("_1")]),
                ..ValueFacts::default()
            }],
            actual_endpoint_sources: Vec::new(),
            destination: PlaceKey::new("_2"),
            point: point(1, 0),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            index_range_kind: None,
            generic_substitution: GenericCallSubstitution::default(),
        }
    }

    #[test]
    fn historical_gitoxide_unchecked_utf8_is_immediate_and_bound_to_input_bytes() {
        let function = FunctionKey::new("gix_attributes::ValueRef::from_bytes");
        let requirement =
            unchecked_utf8_requirement(&function, &historical_gitoxide_unchecked_utf8_call());
        assert_eq!(
            requirement.subject,
            AbstractValue::new([AbstractOrigin::Formal(1)])
        );
        assert_eq!(requirement.source_hint, None);
        assert!(!requirement.return_exposure);
        assert_eq!(requirement.rule, RuleId::P5UncheckedUtf8);
        assert_eq!(
            requirement.predicates,
            BTreeSet::from([Predicate::ValidUtf8])
        );
    }

    #[test]
    fn historical_lru_assume_init_is_immediate_for_arbitrary_types() {
        let function = FunctionKey::new("lru::LruCache::new");
        let mut call = historical_gitoxide_unchecked_utf8_call();
        call.destination = PlaceKey::new("_entry");
        call.destination_is_bool = false;
        let requirement = assume_init_requirement(&function, &call, false);
        assert_eq!(requirement.rule, RuleId::P5AssumeInitUninitialized);
        assert_eq!(
            requirement.predicates,
            BTreeSet::from([Predicate::Initialized])
        );
        assert_eq!(
            requirement.source_hint,
            Some(SourceKind::InternalUnsafeOrigin)
        );
        assert!(!requirement.return_exposure);
        assert_eq!(requirement.sink.point, call.point);

        let definitely_uninit = ValueFacts {
            maybe_uninit_initialized: BTreeSet::from([false]),
            ..ValueFacts::default()
        };
        assert!(definitely_uninitialized(&definitely_uninit));

        let mut branch_join = definitely_uninit.clone();
        branch_join.maybe_uninit_initialized.insert(true);
        assert!(!definitely_uninitialized(&branch_join));
        assert!(assume_init_requires_requirement(&branch_join));

        let mut opaque_write = definitely_uninit;
        opaque_write.havoced = true;
        assert!(!definitely_uninitialized(&opaque_write));
        assert!(assume_init_requires_requirement(&opaque_write));
        assert!(!assume_init_requires_requirement(&typed_storage(
            TypedInitializationState::Initialized,
        )));
    }

    #[test]
    fn typed_pointer_write_initializes_one_exact_whole_target() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let source = PlaceKey::new("_3");
        let mut state = ProgramState::empty();
        state.set_value(
            storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        state.set_value(
            pointer.clone(),
            whole_initialization_pointer([storage.clone()]),
        );
        state.set_value(source.clone(), ValueFacts::default());

        apply_call_side_effects(
            &pointer_write_call(&pointer, &source),
            &point(0, 4),
            &mut state,
        );

        let initialized = state_value(&state, &storage);
        assert_eq!(
            typed_initialization_state(&initialized),
            TypedInitializationState::Initialized
        );
        assert_eq!(
            initialized
                .initialization_witnesses
                .iter()
                .map(|witness| witness.kind)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([InitializationWitnessKind::PointerWrite])
        );
        assert_eq!(
            current_storage_versions(&state, &storage),
            BTreeSet::from([StaticWriteToken::new(0, 4)])
        );
    }

    #[test]
    fn typed_pointer_copy_count_one_copies_source_state_without_assuming_it() {
        let source_storage = PlaceKey::new("_1");
        let destination_storage = PlaceKey::new("_2");
        let source_pointer = PlaceKey::new("_3");
        let destination_pointer = PlaceKey::new("_4");
        for (source_initialization, expected) in [
            (
                TypedInitializationState::Initialized,
                TypedInitializationState::Initialized,
            ),
            (
                TypedInitializationState::Uninitialized,
                TypedInitializationState::Uninitialized,
            ),
            (
                TypedInitializationState::Unknown,
                TypedInitializationState::Unknown,
            ),
        ] {
            let mut state = ProgramState::empty();
            state.set_value(source_storage.clone(), typed_storage(source_initialization));
            state.set_value(
                destination_storage.clone(),
                typed_storage(TypedInitializationState::Initialized),
            );
            state.set_value(
                source_pointer.clone(),
                whole_initialization_pointer([source_storage.clone()]),
            );
            state.set_value(
                destination_pointer.clone(),
                whole_initialization_pointer([destination_storage.clone()]),
            );

            apply_call_side_effects(
                &pointer_copy_call(
                    &source_pointer,
                    &destination_pointer,
                    CallOperand::Constant(1),
                    true,
                ),
                &point(1, 2),
                &mut state,
            );

            assert_eq!(
                typed_initialization_state(&state_value(&state, &destination_storage)),
                expected,
                "copy must preserve the source initialization state"
            );
        }
    }

    #[test]
    fn typed_pointer_copy_count_zero_does_not_write_or_initialize() {
        let source_storage = PlaceKey::new("_1");
        let destination_storage = PlaceKey::new("_2");
        let source_pointer = PlaceKey::new("_3");
        let destination_pointer = PlaceKey::new("_4");
        let mut state = ProgramState::empty();
        state.set_value(
            source_storage.clone(),
            typed_storage(TypedInitializationState::Initialized),
        );
        state.set_value(
            destination_storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        state.set_value(
            source_pointer.clone(),
            whole_initialization_pointer([source_storage]),
        );
        state.set_value(
            destination_pointer.clone(),
            whole_initialization_pointer([destination_storage.clone()]),
        );

        apply_call_side_effects(
            &pointer_copy_call(
                &source_pointer,
                &destination_pointer,
                CallOperand::Constant(0),
                false,
            ),
            &point(1, 2),
            &mut state,
        );

        assert_eq!(
            typed_initialization_state(&state_value(&state, &destination_storage)),
            TypedInitializationState::Uninitialized
        );
        assert!(current_storage_versions(&state, &destination_storage).is_empty());
    }

    #[test]
    fn typed_pointer_copy_ambiguous_destination_is_unknown_may_write() {
        let source_storage = PlaceKey::new("_1");
        let first = PlaceKey::new("_2");
        let second = PlaceKey::new("_3");
        let source_pointer = PlaceKey::new("_4");
        let destination_pointer = PlaceKey::new("_5");
        let mut state = ProgramState::empty();
        state.set_value(
            source_storage.clone(),
            typed_storage(TypedInitializationState::Initialized),
        );
        for target in [&first, &second] {
            state.set_value(
                target.clone(),
                typed_storage(TypedInitializationState::Uninitialized),
            );
        }
        state.set_value(
            source_pointer.clone(),
            whole_initialization_pointer([source_storage]),
        );
        state.set_value(
            destination_pointer.clone(),
            whole_initialization_pointer([first.clone(), second.clone()]),
        );

        apply_call_side_effects(
            &pointer_copy_call(
                &source_pointer,
                &destination_pointer,
                CallOperand::Constant(1),
                true,
            ),
            &point(2, 3),
            &mut state,
        );

        for target in [first, second] {
            assert_eq!(
                typed_initialization_state(&state_value(&state, &target)),
                TypedInitializationState::Unknown
            );
            assert_eq!(
                current_storage_versions(&state, &target),
                BTreeSet::from([StaticWriteToken::new(2, 3)])
            );
        }
    }

    #[test]
    fn typed_pointer_copy_joined_with_untracked_pointer_is_not_exact() {
        let source_storage = PlaceKey::new("_1");
        let destination_storage = PlaceKey::new("_2");
        let source_pointer = PlaceKey::new("_3");
        let local_destination_pointer = PlaceKey::new("_4");
        let external_destination_pointer = PlaceKey::new("_5");
        let joined_destination_pointer = PlaceKey::new("_6");
        let mut entry = ProgramState::empty();
        entry.set_value(
            source_storage.clone(),
            typed_storage(TypedInitializationState::Initialized),
        );
        entry.set_value(
            destination_storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        entry.set_value(
            source_pointer.clone(),
            whole_initialization_pointer([source_storage]),
        );
        entry.set_value(
            local_destination_pointer.clone(),
            whole_initialization_pointer([destination_storage.clone()]),
        );
        entry.set_value(external_destination_pointer.clone(), ValueFacts::default());

        let assign_destination = |source: &PlaceKey| ProgramOp {
            point: Some(point(2, 1)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: joined_destination_pointer.clone(),
                destination_projection: None,
                value: ValueModel::Operand(place_operand(&source.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
            unsafe_sites: Vec::new(),
        };
        let mut local_predecessor = entry.clone();
        let mut external_predecessor = entry;
        apply_program_op(
            &assign_destination(&local_destination_pointer),
            BasicBlock::from_usize(0),
            0,
            &mut local_predecessor,
        );
        apply_program_op(
            &assign_destination(&external_destination_pointer),
            BasicBlock::from_usize(1),
            0,
            &mut external_predecessor,
        );
        let mut joined =
            ProgramState::join_predecessors([&local_predecessor, &external_predecessor])
                .expect("two reachable predecessors");

        apply_call_side_effects(
            &pointer_copy_call(
                &source_pointer,
                &joined_destination_pointer,
                CallOperand::Constant(1),
                true,
            ),
            &point(2, 2),
            &mut joined,
        );

        assert_eq!(
            typed_initialization_state(&state_value(&joined, &destination_storage)),
            TypedInitializationState::Unknown,
            "an untracked incoming pointer must prevent an exact whole-target proof"
        );
    }

    #[test]
    fn typed_pointer_receiver_temporary_does_not_pollute_explicit_target() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let receiver = PlaceKey::new("_3");
        let untracked_pointer_alias = PlaceKey::new("_4");
        let mut pointer_value = whole_initialization_pointer([storage.clone()]);
        pointer_value.value_flow.insert(storage.clone());
        pointer_value.value_flow.insert(receiver.clone());
        let mut state = ProgramState::empty();
        state.set_value(storage.clone(), ValueFacts::default());
        state.set_value(
            receiver,
            ValueFacts {
                exact_roots: BTreeSet::from([storage.clone()]),
                value_flow: BTreeSet::from([storage.clone()]),
                ..ValueFacts::default()
            },
        );
        state.set_value(pointer.clone(), pointer_value.clone());

        let (targets, exact) = initialization_pointer_targets(&state, &place_operand(&pointer.0));
        assert_eq!(targets, BTreeSet::from([storage.clone()]));
        assert!(
            exact,
            "an untracked receiver dependency is not a pointer alternative"
        );

        let mut pointer_value = state_value(&state, &pointer);
        pointer_value
            .value_flow
            .insert(untracked_pointer_alias.clone());
        state.set_value(pointer.clone(), pointer_value);
        state.set_value(
            untracked_pointer_alias,
            ValueFacts {
                whole_init_targets: BTreeSet::from([storage.clone()]),
                ..ValueFacts::default()
            },
        );
        let (targets, exact) = initialization_pointer_targets(&state, &place_operand(&pointer.0));
        assert_eq!(targets, BTreeSet::from([storage]));
        assert!(
            !exact,
            "an untracked alias with an explicit may-target must contribute Unknown"
        );
    }

    #[test]
    fn typed_pointer_registry_return_strongly_initializes_on_normal_ptr_write_edge() {
        let storage = PlaceKey::new("_1");
        let receiver = PlaceKey::new("_2");
        let pointer = PlaceKey::new("_3");
        let source = PlaceKey::new("_4");
        let pointer_normal = BasicBlock::from_usize(1);
        let write_normal = BasicBlock::from_usize(2);
        let mut state = ProgramState::empty();
        state.set_value(
            storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        state.set_value(
            receiver.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([storage.clone()]),
                value_flow: BTreeSet::from([storage.clone()]),
                ..ValueFacts::default()
            },
        );
        state.set_value(source.clone(), ValueFacts::default());

        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(pointer_normal),
                point: point(0, 1),
                call: CallModel {
                    descriptor: CallDescriptor {
                        callee: None,
                        raw_def: None,
                        disposition: BoundaryDisposition::ModeledRegistry,
                        args: vec![place_operand(&receiver.0)],
                        destination: pointer.clone(),
                        generic_substitution: GenericCallSubstitution::default(),
                    },
                    value: RegistryValueModel::TypedInitializationPointer {
                        receiver: place_operand(&receiver.0),
                        generic_view: Some(test_generic_capability(
                            GenericCapabilityKind::RepresentationPreservingPointerView,
                        )),
                    },
                    predicate: None,
                    mutable_actuals: Vec::new(),
                    ffi_out_actuals: Vec::new(),
                    typed_initialization_effect: None,
                    destination_is_bool: false,
                    destination_restricted: false,
                    access_width: None,
                    legacy_write: false,
                    destination_out_formal: None,
                },
            },
            pointer_normal,
            &mut state,
        );

        let returned = state_value(&state, &pointer);
        assert_eq!(
            returned.whole_init_targets,
            BTreeSet::from([storage.clone()])
        );
        assert!(returned.whole_init_target_tracked);
        assert!(!returned.whole_init_target_unknown);
        assert!(returned.endpoint_sources.iter().any(|source| matches!(
            source,
            EndpointNode::GenericCapability { evidence, .. }
                if evidence.kind
                    == GenericCapabilityKind::RepresentationPreservingPointerView
        )));
        let (targets, exact) = initialization_pointer_targets(&state, &place_operand(&pointer.0));
        assert_eq!(targets, BTreeSet::from([storage.clone()]));
        assert!(
            exact,
            "the normalized typed pointer return must remain exact"
        );

        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(write_normal),
                point: point(1, 1),
                call: pointer_write_call(&pointer, &source),
            },
            write_normal,
            &mut state,
        );

        let initialized = state_value(&state, &storage);
        assert_eq!(
            typed_initialization_state(&initialized),
            TypedInitializationState::Initialized
        );
        assert!(initialized
            .initialization_witnesses
            .iter()
            .any(|witness| witness.kind == InitializationWitnessKind::PointerWrite));
    }

    #[test]
    fn opaque_intrinsic_mutation_forgets_pre_call_uninitialized_state() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let mut state = ProgramState::empty();
        state.set_value(
            storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        state.set_value(
            pointer.clone(),
            whole_initialization_pointer([storage.clone()]),
        );
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::OpaqueDirect,
                args: vec![place_operand(&pointer.0)],
                destination: PlaceKey::new("_0"),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![pointer.clone()],
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: Some(TypedInitializationEffect::OpaqueIntrinsicMutation {
                destinations: vec![place_operand(&pointer.0)],
            }),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };

        apply_call_side_effects(&call, &point(0, 1), &mut state);

        assert_eq!(
            typed_initialization_state(&state_value(&state, &storage)),
            TypedInitializationState::Unknown,
            "an opaque intrinsic mutation may initialize the exact pointee"
        );
    }

    #[test]
    fn lossy_pointer_views_retain_may_targets_for_typed_copy_havoc() {
        let source_storage = PlaceKey::new("_1");
        let destination_storage = PlaceKey::new("_2");
        let source_pointer = PlaceKey::new("_3");
        let destination_pointer = PlaceKey::new("_4");
        let transformed_pointer = PlaceKey::new("_5");
        let base = place_operand(&destination_pointer.0);
        let models = [
            RegistryValueModel::WrappingAdd {
                base: base.clone(),
                offset: constant_operand(0),
            },
            RegistryValueModel::AddressPointerView {
                operand: base.clone(),
                generic_view: None,
            },
            RegistryValueModel::PointerCast {
                operand: base,
                obligation: ConcreteLayoutObligation::default(),
            },
        ];

        for model in models {
            let mut state = ProgramState::empty();
            state.set_value(
                source_storage.clone(),
                typed_storage(TypedInitializationState::Uninitialized),
            );
            state.set_value(
                destination_storage.clone(),
                typed_storage(TypedInitializationState::Initialized),
            );
            state.set_value(
                source_pointer.clone(),
                whole_initialization_pointer([source_storage.clone()]),
            );
            state.set_value(
                destination_pointer.clone(),
                whole_initialization_pointer([destination_storage.clone()]),
            );
            let transformed = eval_registry_value(&state, &model, &point(2, 3));
            assert_eq!(
                transformed.whole_init_targets,
                BTreeSet::from([destination_storage.clone()]),
                "losing exactness must retain the known may-target"
            );
            assert!(transformed.whole_init_target_unknown);
            state.set_value(transformed_pointer.clone(), transformed);

            apply_call_side_effects(
                &pointer_copy_call(
                    &source_pointer,
                    &transformed_pointer,
                    CallOperand::Constant(1),
                    true,
                ),
                &point(2, 4),
                &mut state,
            );

            assert_eq!(
                typed_initialization_state(&state_value(&state, &destination_storage)),
                TypedInitializationState::Unknown,
                "a nonexact copy must havoc every retained may-target"
            );
        }
    }

    #[test]
    fn typed_pointer_copy_projected_destination_is_partial_not_whole() {
        let source_storage = PlaceKey::new("_1");
        let destination_storage = PlaceKey::new("_2");
        let projected_destination = PlaceKey::new("_2[0]");
        let source_pointer = PlaceKey::new("_3");
        let destination_pointer = PlaceKey::new("_4");
        let mut state = ProgramState::empty();
        state.set_value(
            source_storage.clone(),
            typed_storage(TypedInitializationState::Initialized),
        );
        state.set_value(
            destination_storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        state.set_value(
            source_pointer.clone(),
            whole_initialization_pointer([source_storage]),
        );
        state.set_value(
            destination_pointer.clone(),
            whole_initialization_pointer([projected_destination]),
        );

        apply_call_side_effects(
            &pointer_copy_call(
                &source_pointer,
                &destination_pointer,
                CallOperand::Constant(1),
                true,
            ),
            &point(2, 4),
            &mut state,
        );

        assert_eq!(
            typed_initialization_state(&state_value(&state, &destination_storage)),
            TypedInitializationState::Unknown
        );
        assert_eq!(
            current_storage_versions(&state, &destination_storage),
            BTreeSet::from([StaticWriteToken::new(2, 4)])
        );
    }

    #[test]
    fn typed_pointer_copy_unknown_or_multi_count_havocs_the_whole_destination() {
        let source_storage = PlaceKey::new("_1");
        let destination_storage = PlaceKey::new("_2");
        let source_pointer = PlaceKey::new("_3");
        let destination_pointer = PlaceKey::new("_4");
        let count = PlaceKey::new("_5");
        for copy_count in [CallOperand::Place(count.clone()), CallOperand::Constant(2)] {
            let mut state = ProgramState::empty();
            state.set_value(
                source_storage.clone(),
                typed_storage(TypedInitializationState::Initialized),
            );
            state.set_value(
                destination_storage.clone(),
                typed_storage(TypedInitializationState::Uninitialized),
            );
            state.set_value(
                source_pointer.clone(),
                whole_initialization_pointer([source_storage.clone()]),
            );
            state.set_value(
                destination_pointer.clone(),
                whole_initialization_pointer([destination_storage.clone()]),
            );
            state.set_value(count.clone(), ValueFacts::default());

            apply_call_side_effects(
                &pointer_copy_call(&source_pointer, &destination_pointer, copy_count, true),
                &point(2, 4),
                &mut state,
            );

            assert_eq!(
                typed_initialization_state(&state_value(&state, &destination_storage)),
                TypedInitializationState::Unknown
            );
            assert_eq!(
                current_storage_versions(&state, &destination_storage),
                BTreeSet::from([StaticWriteToken::new(2, 4)])
            );
        }
    }

    #[test]
    fn typed_pointer_copy_snapshots_same_source_before_destination_write() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let mut state = ProgramState::empty();
        state.set_value(
            storage.clone(),
            typed_storage(TypedInitializationState::Initialized),
        );
        state.set_value(
            pointer.clone(),
            whole_initialization_pointer([storage.clone()]),
        );

        apply_call_side_effects(
            &pointer_copy_call(&pointer, &pointer, CallOperand::Constant(1), false),
            &point(3, 1),
            &mut state,
        );

        assert_eq!(
            typed_initialization_state(&state_value(&state, &storage)),
            TypedInitializationState::Initialized
        );
        assert!(state_value(&state, &storage)
            .initialization_witnesses
            .iter()
            .any(|witness| witness.kind == InitializationWitnessKind::PointerCopy));
    }

    #[test]
    fn typed_pointer_provenance_survives_an_ordinary_operand_copy() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let copied_pointer = PlaceKey::new("_3");
        let source = PlaceKey::new("_4");
        let mut state = ProgramState::empty();
        state.set_value(
            storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        state.set_value(
            pointer.clone(),
            whole_initialization_pointer([storage.clone()]),
        );
        state.set_value(source.clone(), ValueFacts::default());
        let copy = ProgramOp {
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: copied_pointer.clone(),
                destination_projection: None,
                value: ValueModel::Operand(place_operand(&pointer.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
            unsafe_sites: Vec::new(),
        };
        apply_program_op(&copy, BasicBlock::from_usize(0), 0, &mut state);
        apply_call_side_effects(
            &pointer_write_call(&copied_pointer, &source),
            &point(0, 1),
            &mut state,
        );

        assert_eq!(
            typed_initialization_state(&state_value(&state, &storage)),
            TypedInitializationState::Initialized
        );
    }

    #[test]
    fn typed_pointer_views_retain_may_target_but_only_identity_is_exact() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let mut state = ProgramState::empty();
        state.set_value(
            pointer.clone(),
            whole_initialization_pointer([storage.clone()]),
        );

        let identity = eval_value_model(
            &state,
            &ValueModel::IdentityPointerView {
                operand: place_operand(&pointer.0),
                generic_view: None,
            },
            StaticWriteToken::new(0, 0),
        );
        assert_eq!(
            identity.whole_init_targets,
            BTreeSet::from([storage.clone()])
        );
        assert!(!identity.whole_init_target_unknown);

        let address_view = eval_value_model(
            &state,
            &ValueModel::AddressPointerView {
                operand: place_operand(&pointer.0),
                generic_view: None,
            },
            StaticWriteToken::new(0, 1),
        );
        assert_eq!(address_view.whole_init_targets, BTreeSet::from([storage]));
        assert!(address_view.whole_init_target_unknown);
    }

    #[test]
    fn opaque_call_return_cannot_reuse_an_argument_whole_target() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let destination = PlaceKey::new("_3");
        let normal = BasicBlock::from_usize(1);
        let mut state = ProgramState::empty();
        state.set_value(
            pointer.clone(),
            whole_initialization_pointer([storage.clone()]),
        );
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: point(0, 2),
            call: CallModel {
                descriptor: CallDescriptor {
                    callee: None,
                    raw_def: None,
                    disposition: BoundaryDisposition::OpaqueDirect,
                    args: vec![place_operand(&pointer.0)],
                    destination: destination.clone(),
                    generic_substitution: GenericCallSubstitution::default(),
                },
                value: RegistryValueModel::Empty,
                predicate: None,
                mutable_actuals: Vec::new(),
                ffi_out_actuals: Vec::new(),
                typed_initialization_effect: None,
                destination_is_bool: false,
                destination_restricted: false,
                access_width: None,
                legacy_write: false,
                destination_out_formal: None,
            },
        };

        apply_program_edge(&edge, normal, &mut state);

        let returned = state_value(&state, &destination);
        assert_eq!(returned.whole_init_targets, BTreeSet::from([storage]));
        assert!(returned.whole_init_target_unknown);
    }

    #[test]
    fn typed_pointer_write_effect_exists_only_on_the_normal_edge() {
        let storage = PlaceKey::new("_1");
        let pointer = PlaceKey::new("_2");
        let source = PlaceKey::new("_3");
        let normal = BasicBlock::from_usize(1);
        let unwind = BasicBlock::from_usize(2);
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: point(0, 2),
            call: pointer_write_call(&pointer, &source),
        };
        let mut entry = ProgramState::empty();
        entry.set_value(
            storage.clone(),
            typed_storage(TypedInitializationState::Uninitialized),
        );
        entry.set_value(pointer, whole_initialization_pointer([storage.clone()]));
        entry.set_value(source, ValueFacts::default());
        let mut normal_state = entry.clone();
        let mut unwind_state = entry;

        apply_program_edge(&edge, normal, &mut normal_state);
        apply_program_edge(&edge, unwind, &mut unwind_state);

        assert_eq!(
            typed_initialization_state(&state_value(&normal_state, &storage)),
            TypedInitializationState::Initialized
        );
        assert_eq!(
            typed_initialization_state(&state_value(&unwind_state, &storage)),
            TypedInitializationState::Uninitialized
        );
    }

    #[test]
    fn maybe_uninit_write_strongly_initializes_only_one_exact_receiver() {
        let storage = PlaceKey::new("_1");
        let receiver = PlaceKey::new("_2");
        let stale_origin = AbstractOrigin::FfiOutput {
            token: format!("{FFI_UNINITIALIZED_OUTPUT_PREFIX}stale"),
            span: span(),
        };
        let mut state = ProgramState::empty();
        state.set_value(
            storage.clone(),
            ValueFacts {
                value: AbstractValue::new([stale_origin.clone()]),
                exact_roots: BTreeSet::from([storage.clone()]),
                maybe_uninit_initialized: BTreeSet::from([false]),
                ffi_output: true,
                havoced: true,
                ..ValueFacts::default()
            },
        );
        state.set_value(receiver.clone(), exact_value(&storage));

        apply_call_side_effects(
            &typed_initialization_call(place_operand(&receiver.0)),
            &point(0, 1),
            &mut state,
        );

        let initialized = state_value(&state, &storage);
        assert!(definitely_initialized(&initialized));
        assert!(!initialized.value.origins.contains(&stale_origin));
        assert_eq!(
            current_storage_versions(&state, &storage),
            BTreeSet::from([StaticWriteToken::new(0, 1)])
        );
    }

    #[test]
    fn maybe_uninit_write_with_ambiguous_aliases_keeps_initialization_unknown() {
        let first = PlaceKey::new("_1");
        let second = PlaceKey::new("_3");
        let receiver = PlaceKey::new("_2");
        let uninitialized = || ValueFacts {
            maybe_uninit_initialized: BTreeSet::from([false]),
            ..ValueFacts::default()
        };
        let mut state = ProgramState::empty();
        state.set_value(first.clone(), uninitialized());
        state.set_value(second.clone(), uninitialized());
        state.set_value(
            receiver.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([first.clone(), second.clone()]),
                value_flow: BTreeSet::from([first.clone(), second.clone()]),
                ..ValueFacts::default()
            },
        );

        apply_call_side_effects(
            &typed_initialization_call(place_operand(&receiver.0)),
            &point(0, 1),
            &mut state,
        );

        for target in [first, second] {
            let value = state_value(&state, &target);
            assert!(value.havoced);
            assert!(!definitely_initialized(&value));
            assert!(!definitely_uninitialized(&value));
        }
    }

    #[test]
    fn maybe_uninit_write_initializes_only_on_the_normal_call_edge() {
        let storage = PlaceKey::new("_1");
        let receiver = PlaceKey::new("_2");
        let normal = BasicBlock::from_usize(1);
        let unwind = BasicBlock::from_usize(2);
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal),
            point: point(0, 1),
            call: typed_initialization_call(place_operand(&receiver.0)),
        };
        let mut entry = ProgramState::empty();
        entry.set_value(
            storage.clone(),
            ValueFacts {
                exact_roots: BTreeSet::from([storage.clone()]),
                maybe_uninit_initialized: BTreeSet::from([false]),
                ..ValueFacts::default()
            },
        );
        entry.set_value(receiver, exact_value(&storage));
        let mut normal_state = entry.clone();
        let mut unwind_state = entry;

        apply_program_edge(&edge, normal, &mut normal_state);
        apply_program_edge(&edge, unwind, &mut unwind_state);

        assert!(definitely_initialized(&state_value(
            &normal_state,
            &storage
        )));
        assert!(definitely_uninitialized(&state_value(
            &unwind_state,
            &storage
        )));
    }

    #[test]
    fn legacy_returned_bool_assume_init_keeps_its_existing_rule_identity() {
        let function = FunctionKey::new("fixture::returned_bool");
        let mut call = historical_gitoxide_unchecked_utf8_call();
        call.destination = PlaceKey::new("_0");
        call.destination_is_bool = true;
        let requirement = assume_init_requirement(&function, &call, true);
        assert_eq!(requirement.rule, RuleId::P5AssumeInitBool);
        assert_eq!(
            requirement.predicates,
            BTreeSet::from([Predicate::Initialized, Predicate::ValidBool])
        );
        assert!(requirement.return_exposure);
    }

    #[test]
    fn historical_ffi_enum_assume_init_is_an_immediate_invalid_value_requirement() {
        let source_place = PlaceKey::new("_1");
        let ffi_origin = AbstractOrigin::FfiOutput {
            token: format!("{FFI_UNINITIALIZED_OUTPUT_PREFIX}test"),
            span: point(0, 0).span,
        };
        let source = ValueFacts {
            value: AbstractValue::new([ffi_origin.clone()]),
            exact_roots: BTreeSet::from([source_place.clone()]),
            maybe_uninit_initialized: BTreeSet::from([false]),
            ffi_output: true,
            ..ValueFacts::default()
        };
        assert!(!definitely_uninitialized(&source));
        assert_eq!(
            ffi_invalid_value_subject(&source),
            Some(AbstractValue::new([ffi_origin.clone()]))
        );

        let mut initialized_before_ffi = source.clone();
        initialized_before_ffi.maybe_uninit_initialized = BTreeSet::from([true]);
        initialized_before_ffi.value.origins = BTreeSet::from([AbstractOrigin::FfiOutput {
            token: "ffi-out:test".into(),
            span: point(0, 1).span,
        }]);
        assert_eq!(ffi_invalid_value_subject(&initialized_before_ffi), None);

        let requirement = ffi_invalid_value_requirement(
            &FunctionKey::new("nix::getsockopt"),
            &point(1, 0),
            source.value.clone(),
        );
        assert_eq!(requirement.rule, RuleId::P3FfiOutInvalidValue);
        assert_eq!(requirement.subject, AbstractValue::new([ffi_origin]));
        assert_eq!(
            requirement.predicates,
            BTreeSet::from([Predicate::ValidForType])
        );
        assert!(!requirement.return_exposure);

        let mut state = BlockState::empty();
        state.set_value(source_place.clone(), source);
        let assumed = eval_registry_value(
            &state,
            &RegistryValueModel::AssumeInit {
                source: place_operand(&source_place.0),
                restricted_target: true,
            },
            &point(1, 0),
        );
        assert!(assumed.ffi_output);
        assert!(assumed.maybe_uninit_initialized.is_empty());
    }

    #[test]
    fn historical_ffi_invalid_value_requires_a_concrete_uninitialized_ffi_origin() {
        let special = AbstractOrigin::FfiOutput {
            token: format!("{FFI_UNINITIALIZED_OUTPUT_PREFIX}test"),
            span: point(0, 0).span,
        };
        let ordinary = AbstractOrigin::FfiOutput {
            token: "ffi-out:test".into(),
            span: point(0, 1).span,
        };
        let requirement = ffi_invalid_value_requirement(
            &FunctionKey::new("wrapper"),
            &point(1, 0),
            AbstractValue::new([special.clone()]),
        );
        assert!(origin_matches_requirement_source(
            &requirement,
            &special,
            SourceKind::FfiOutput
        ));
        assert!(!origin_matches_requirement_source(
            &requirement,
            &ordinary,
            SourceKind::FfiOutput
        ));
        assert!(!p3_requirement_matches_call_destination(
            &requirement,
            false
        ));
        assert!(p3_requirement_matches_call_destination(&requirement, true));
    }

    fn task9_is_err_call(checked: &PlaceKey, destination: &PlaceKey) -> CallModel {
        CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&checked.0)],
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: Some(PredicateModel::IsErr {
                checked: place_operand(&checked.0),
            }),
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: true,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        }
    }

    #[test]
    fn p5_utf8_requires_an_exact_unrestricted_byte_witness() {
        let formal = PlaceKey::new("_1");
        let exact = ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Formal(1)]),
            exact_roots: BTreeSet::from([formal]),
            ..ValueFacts::default()
        };
        assert_eq!(
            classify_utf8_input_safety(
                &exact,
                1,
                &BTreeMap::from([(1, Utf8FormalCapability::UnrestrictedBytes)]),
            ),
            UnsafeResultSafety::ViolationWitness
        );

        let mut opaque = exact.clone();
        opaque.havoced = true;
        assert_eq!(
            classify_utf8_input_safety(
                &opaque,
                1,
                &BTreeMap::from([(1, Utf8FormalCapability::UnrestrictedBytes)]),
            ),
            UnsafeResultSafety::Unknown(UnestablishedReason::ValidityObligationUnknown)
        );
    }

    #[test]
    fn p5_utf8_distinguishes_type_and_literal_content_proofs() {
        let formal = PlaceKey::new("_1");
        let valid_str = ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Formal(1)]),
            exact_roots: BTreeSet::from([formal]),
            ..ValueFacts::default()
        };
        assert_eq!(
            classify_utf8_input_safety(
                &valid_str,
                1,
                &BTreeMap::from([(1, Utf8FormalCapability::ValidUtf8)]),
            ),
            UnsafeResultSafety::ProvenSafe
        );

        let ascii = ValueFacts {
            value: AbstractValue::new([
                AbstractOrigin::Constant(b'a' as u64),
                AbstractOrigin::Constant(b'z' as u64),
            ]),
            ..ValueFacts::default()
        };
        assert_eq!(
            classify_utf8_input_safety(&ascii, 0, &BTreeMap::new()),
            UnsafeResultSafety::ProvenSafe
        );

        let invalid = ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Constant(0xff)]),
            ..ValueFacts::default()
        };
        assert_eq!(
            classify_utf8_input_safety(&invalid, 0, &BTreeMap::new()),
            UnsafeResultSafety::ViolationWitness
        );
    }

    #[test]
    fn task9_utf8_same_bytes_false_edge_establishes_exact_local_validation() {
        let bytes = PlaceKey::new("_1");
        let checked = PlaceKey::new("_4");
        let borrowed = PlaceKey::new("_3");
        let condition = PlaceKey::new("_2");
        let normal = BasicBlock::from_usize(1);
        let mut state = BlockState::empty();
        seed_exact(&mut state, &bytes);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: task9_utf8_result_call(&bytes, &checked),
            },
            normal,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: borrowed.clone(),
                    destination_projection: None,
                    value: ValueModel::RefOrRaw(place_operand(&checked.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            normal,
            0,
            &mut state,
        );
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 1),
                call: task9_is_err_call(&borrowed, &condition),
            },
            normal,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::ValidUtf8,
            access_width: None,
            proof_definition: None,
            subject: bytes,
            collection: None,
        };
        assert!(state.pending_is_current(&condition, &binding, false));
        apply_program_edge(
            &EdgeTerm::Switch {
                condition,
                false_target: normal,
                true_target: BasicBlock::from_usize(2),
            },
            normal,
            &mut state,
        );
        assert!(state.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn task9_utf8_direct_result_discriminant_establishes_exact_local_validation() {
        let bytes = PlaceKey::new("_1");
        let checked = PlaceKey::new("_2");
        let discriminant = PlaceKey::new("_3");
        let checked_block = BasicBlock::from_usize(1);
        let ok_block = BasicBlock::from_usize(2);
        let error_block = BasicBlock::from_usize(3);
        let mut state = BlockState::empty();
        seed_exact(&mut state, &bytes);

        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(checked_block),
                point: point(0, 0),
                call: task9_utf8_result_call(&bytes, &checked),
            },
            checked_block,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: discriminant.clone(),
                    destination_projection: None,
                    value: ValueModel::Discriminant(place_operand(&checked.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            checked_block,
            0,
            &mut state,
        );

        let binding = ValidationBinding {
            predicate: Predicate::ValidUtf8,
            access_width: None,
            proof_definition: None,
            subject: bytes,
            collection: None,
        };
        assert!(state.pending_is_current(&discriminant, &binding, false));
        apply_program_edge(
            &EdgeTerm::Switch {
                condition: discriminant,
                false_target: ok_block,
                true_target: error_block,
            },
            ok_block,
            &mut state,
        );
        assert!(state.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn task9_utf8_try_branch_continue_establishes_exact_local_validation() {
        let bytes = PlaceKey::new("_1");
        let checked = PlaceKey::new("_2");
        let branch = PlaceKey::new("_3");
        let discriminant = PlaceKey::new("_4");
        let normal = BasicBlock::from_usize(1);
        let continue_block = BasicBlock::from_usize(2);
        let break_block = BasicBlock::from_usize(3);
        let mut state = BlockState::empty();
        seed_exact(&mut state, &bytes);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: task9_utf8_result_call(&bytes, &checked),
            },
            normal,
            &mut state,
        );
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 0),
                call: CallModel {
                    descriptor: CallDescriptor {
                        callee: None,
                        raw_def: None,
                        disposition: BoundaryDisposition::ModeledRegistry,
                        args: vec![place_operand(&checked.0)],
                        destination: branch.clone(),
                        generic_substitution: GenericCallSubstitution::default(),
                    },
                    value: RegistryValueModel::CheckedTryBranch(place_operand(&checked.0)),
                    predicate: None,
                    mutable_actuals: Vec::new(),
                    ffi_out_actuals: Vec::new(),
                    typed_initialization_effect: None,
                    destination_is_bool: false,
                    destination_restricted: false,
                    access_width: None,
                    legacy_write: false,
                    destination_out_formal: None,
                },
            },
            normal,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: discriminant.clone(),
                    destination_projection: None,
                    value: ValueModel::Discriminant(place_operand(&branch.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            normal,
            1,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::ValidUtf8,
            access_width: None,
            proof_definition: None,
            subject: bytes,
            collection: None,
        };
        assert!(state.pending_is_current(&discriminant, &binding, false));
        apply_program_edge(
            &EdgeTerm::Switch {
                condition: discriminant,
                false_target: continue_block,
                true_target: break_block,
            },
            continue_block,
            &mut state,
        );
        assert!(state.has_current_validation(&binding, binding.storage_places()));
    }

    #[test]
    fn task9_utf8_wrong_bytes_and_wrong_polarity_fail_closed() {
        let checked_bytes = PlaceKey::new("_1");
        let wrong_bytes = PlaceKey::new("_5");
        let checked = PlaceKey::new("_4");
        let condition = PlaceKey::new("_2");
        let normal = BasicBlock::from_usize(1);
        let mut state = BlockState::empty();
        seed_exact(&mut state, &checked_bytes);
        seed_exact(&mut state, &wrong_bytes);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: task9_utf8_result_call(&checked_bytes, &checked),
            },
            normal,
            &mut state,
        );
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 0),
                call: task9_is_err_call(&checked, &condition),
            },
            normal,
            &mut state,
        );
        let wrong_binding = ValidationBinding {
            predicate: Predicate::ValidUtf8,
            access_width: None,
            proof_definition: None,
            subject: wrong_bytes,
            collection: None,
        };
        assert!(!state.pending_is_current(&condition, &wrong_binding, false));
        let checked_binding = ValidationBinding {
            predicate: Predicate::ValidUtf8,
            access_width: None,
            proof_definition: None,
            subject: checked_bytes,
            collection: None,
        };
        assert!(!state.pending_is_current(&condition, &checked_binding, true));
    }

    #[test]
    fn task9_utf8_write_invalidates_old_evidence_but_recheck_restores_it() {
        let bytes = PlaceKey::new("_1");
        let checked = PlaceKey::new("_4");
        let condition = PlaceKey::new("_2");
        let normal = BasicBlock::from_usize(1);
        let mut state = BlockState::empty();
        seed_exact(&mut state, &bytes);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: task9_utf8_result_call(&bytes, &checked),
            },
            normal,
            &mut state,
        );
        let written = canonical_storage(&bytes);
        state.record_may_write(written.clone(), StaticWriteToken::new(1, 0));
        invalidate_relation_evidence(&mut state, &written);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 1),
                call: task9_is_err_call(&checked, &condition),
            },
            normal,
            &mut state,
        );
        assert!(state.pending_must().get(&condition).is_none());

        let rechecked = PlaceKey::new("_5");
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 2),
                call: task9_utf8_result_call(&bytes, &rechecked),
            },
            normal,
            &mut state,
        );
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 3),
                call: task9_is_err_call(&rechecked, &condition),
            },
            normal,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::ValidUtf8,
            access_width: None,
            proof_definition: None,
            subject: bytes,
            collection: None,
        };
        assert!(state.pending_is_current(&condition, &binding, false));
    }

    #[test]
    fn task9_utf8_diamond_requires_the_same_check_on_every_path() {
        let bytes = PlaceKey::new("_1");
        let checked = PlaceKey::new("_4");
        let condition = PlaceKey::new("_2");
        let normal = BasicBlock::from_usize(1);
        let mut checked_path = BlockState::empty();
        seed_exact(&mut checked_path, &bytes);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: task9_utf8_result_call(&bytes, &checked),
            },
            normal,
            &mut checked_path,
        );
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(1, 0),
                call: task9_is_err_call(&checked, &condition),
            },
            normal,
            &mut checked_path,
        );
        let unchecked_path = {
            let mut state = BlockState::empty();
            seed_exact(&mut state, &bytes);
            state
        };
        let joined = BlockState::join_predecessors([&checked_path, &unchecked_path]).unwrap();
        assert!(joined.pending_must().get(&condition).is_none());

        let all_checked = BlockState::join_predecessors([&checked_path, &checked_path]).unwrap();
        let binding = ValidationBinding {
            predicate: Predicate::ValidUtf8,
            access_width: None,
            proof_definition: None,
            subject: bytes,
            collection: None,
        };
        assert!(all_checked.pending_is_current(&condition, &binding, false));
    }

    #[test]
    fn task9_utf8_local_proof_is_consumed_but_never_exported() {
        assert_eq!(
            validation_contract_policy(Predicate::ValidUtf8),
            LocalValidationPolicy::LocalOnly
        );
        assert_eq!(
            validation_contract_policy(Predicate::InBounds),
            LocalValidationPolicy::Exportable
        );
        assert_eq!(
            validation_contract_policy(Predicate::RangeInBounds),
            LocalValidationPolicy::LocalOnly
        );
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Constant(7),
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::SelectedOpenBehavior {
                token: "trait-return".into(),
                controller: BehaviorController::Unknown,
                inputs: Vec::new(),
                type_class: InfluenceTypeClass::Aggregate,
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![actual.clone()],
            ffi_out_actuals: vec![FfiOutActual {
                place: actual,
                type_class: InfluenceTypeClass::Pointer,
            }],
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: Some(PredicateModel::IsNull {
                pointer: output.clone(),
            }),
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: true,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
            access_width: None,
            proof_definition: None,
            subject: output,
            collection: None,
        };
        assert!(state.pending_is_current(&condition, &binding, false));
    }

    #[test]
    fn copied_ffi_output_is_null_guard_proves_the_original_pointer_binding() {
        let output = PlaceKey::new("_1");
        let actual = PlaceKey::new("_2");
        let checked = PlaceKey::new("_6");
        let condition = PlaceKey::new("_5");
        let normal = BasicBlock::from_usize(1);
        let false_target = BasicBlock::from_usize(2);
        let true_target = BasicBlock::from_usize(3);
        let mut state = BlockState::empty();
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![actual.clone()],
            ffi_out_actuals: vec![FfiOutActual {
                place: actual,
                type_class: InfluenceTypeClass::Pointer,
            }],
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        apply_call_side_effects(&ffi, &point(0, 0), &mut state);
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: checked.clone(),
                    destination_projection: None,
                    value: ValueModel::Operand(place_operand(&output.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            1,
            &mut state,
        );
        let is_null = CallModel {
            descriptor: CallDescriptor {
                callee: None,
                raw_def: None,
                disposition: BoundaryDisposition::ModeledRegistry,
                args: vec![place_operand(&checked.0)],
                destination: condition.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: Some(PredicateModel::IsNull { pointer: checked }),
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: true,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
        apply_program_edge(
            &EdgeTerm::Switch {
                condition,
                false_target,
                true_target,
            },
            false_target,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::NonNull,
            access_width: None,
            proof_definition: None,
            subject: output,
            collection: None,
        };
        assert!(ValidationSnapshot::capture(&state, 0).proves(&binding));
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::GenericCapability {
                inputs: Vec::new(),
                evidence: test_generic_capability(GenericCapabilityKind::AssociatedSequenceShape {
                    associated_items: BTreeSet::from(["crate::Provider::Results".to_string()]),
                }),
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: Some(PredicateModel::IsEmpty {
                slice: slice.clone(),
            }),
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: true,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
            access_width: None,
            proof_definition: None,
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::SelectedOpenBehavior {
                token: "trait-index".into(),
                controller: BehaviorController::Unknown,
                inputs: Vec::new(),
                type_class: InfluenceTypeClass::Integer,
            },
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
                unsafe_sites: Vec::new(),
                point: Some(point(1, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: length.clone(),
                    destination_projection: None,
                    value: ValueModel::Length(place_operand(&data.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            normal,
            0,
            &mut state,
        );
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Ge,
                        left: place_operand(&index.0),
                        right: place_operand(&length.0),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            normal,
            1,
            &mut state,
        );
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Length(place_operand(&collection.0)),
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        let compare = |destination: PlaceKey| ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(1, 1)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination,
                destination_projection: None,
                value: ValueModel::Compare {
                    op: BinOp::Ge,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
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
        let mut unknown_collection = exact_value(&collection);
        unknown_collection.semantic_unknown = true;
        unknown.set_value(collection.clone(), unknown_collection);
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
        assert!(!has_pending_validation(&unknown, &unknown_condition));
    }

    #[test]
    fn modeled_length_exports_an_exact_versioned_relation_for_its_receiver() {
        let collection = PlaceKey::new("((*_1).0: [u8; 8])");
        let receiver = PlaceKey::new("_8");
        let sink_view = PlaceKey::new("_13");
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_7");
        let condition = PlaceKey::new("_5");
        let normal = BasicBlock::from_usize(1);
        let mut state = BlockState::empty();
        state.set_value(receiver.clone(), exact_value(&collection));
        state.set_value(sink_view.clone(), exact_value(&collection));
        seed_exact(&mut state, &index);

        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal),
                point: point(0, 0),
                call: CallModel {
                    descriptor: CallDescriptor {
                        callee: None,
                        raw_def: None,
                        disposition: BoundaryDisposition::ModeledRegistry,
                        args: vec![place_operand(&receiver.0)],
                        destination: length.clone(),
                        generic_substitution: GenericCallSubstitution::default(),
                    },
                    value: RegistryValueModel::Length(place_operand(&receiver.0)),
                    predicate: None,
                    mutable_actuals: Vec::new(),
                    ffi_out_actuals: Vec::new(),
                    typed_initialization_effect: None,
                    destination_is_bool: false,
                    destination_restricted: false,
                    access_width: None,
                    legacy_write: false,
                    destination_out_formal: None,
                },
            },
            normal,
            &mut state,
        );
        assert!(current_relation_proof(&state).proves_same_value(
            &RelationTerm::Place(length.clone()),
            &RelationTerm::Length(collection.clone()),
        ));

        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: condition.clone(),
                    destination_projection: None,
                    value: ValueModel::Compare {
                        op: BinOp::Ge,
                        left: place_operand(&index.0),
                        right: place_operand(&length.0),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            normal,
            1,
            &mut state,
        );
        let success = BasicBlock::from_usize(2);
        apply_program_edge(
            &EdgeTerm::Switch {
                condition,
                false_target: success,
                true_target: BasicBlock::from_usize(3),
            },
            success,
            &mut state,
        );
        let sink_collection = unique_semantic_value(&semantic_value_at(&state, &sink_view))
            .expect("the second view has one semantic receiver");
        assert_eq!(sink_collection, collection);
        assert!(current_relation_proof(&state)
            .proves_in_bounds(&RelationTerm::Place(index.clone()), &sink_collection));
        assert!(!current_relation_proof(&state).proves_in_bounds(
            &RelationTerm::Place(index.clone()),
            &PlaceKey::new("((*_1).1: [u8; 8])"),
        ));

        state.record_write(canonical_storage(&collection), StaticWriteToken::new(2, 0));
        assert!(!current_relation_proof(&state)
            .proves_in_bounds(&RelationTerm::Place(index), &sink_collection));
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: vec![FfiOutActual {
                place: actual.clone(),
                type_class: InfluenceTypeClass::Pointer,
            }],
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
        let mut expanded_first = exact_value(&first);
        expanded_first.exact_roots.insert(PlaceKey::new("_9"));
        expanded_first.semantic_unknown = true;
        expanded.set_value(first.clone(), expanded_first);
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
            access_width: None,
            proof_definition: None,
            subject: PlaceKey::new("_2"),
            collection: Some(PlaceKey::new("_1")),
        };
        let mut state = BlockState::empty();
        state.establish(BoundValidation::new(
            binding.clone().into(),
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
    fn task12_validation_snapshot_matches_full_state_queries_and_fails_closed_after_write() {
        let collection = PlaceKey::new("_1");
        let subject = PlaceKey::new("_2");
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
            subject: subject.clone(),
            collection: Some(collection.clone()),
        };
        let mut state = ProgramState::empty();
        seed_exact(&mut state, &collection);
        seed_exact(&mut state, &subject);
        state.establish(BoundValidation::new(
            binding.clone().into(),
            binding.storage_places(),
        ));

        let snapshot = ValidationSnapshot::capture(&state, 2);
        assert_eq!(
            snapshot.proves(&binding),
            state_proves_local_binding(&state, Some(&binding))
        );
        for place in [&collection, &subject] {
            assert_eq!(
                snapshot.entry_formal(place),
                entry_formal_slot(2, &state, place)
            );
        }

        state.record_write(subject.clone(), StaticWriteToken::new(0, 1));
        let stale = ValidationSnapshot::capture(&state, 2);
        assert_eq!(
            stale.proves(&binding),
            state_proves_local_binding(&state, Some(&binding))
        );
        assert_eq!(stale.entry_formal(&subject), None);
        assert_eq!(entry_formal_slot(2, &state, &subject), None);
    }

    #[test]
    fn relational_snapshot_proves_only_the_exact_in_bounds_sink() {
        let collection = PlaceKey::new("_1");
        let subject = PlaceKey::new("_2");
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
            subject: subject.clone(),
            collection: Some(collection.clone()),
        };
        let relation = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(subject.clone()),
            RelationTerm::Length(collection.clone()),
        ));
        let mut state = ProgramState::empty();
        seed_exact(&mut state, &collection);
        seed_exact(&mut state, &subject);
        state.establish(BoundValidation::new(
            ProofBinding::Relation(relation.clone()),
            relation.storage_places(),
        ));

        let snapshot = ValidationSnapshot::capture(&state, 2);
        assert!(snapshot.proves(&binding));
        assert!(!snapshot.proves(&ValidationBinding {
            subject: PlaceKey::new("_9"),
            ..binding.clone()
        }));
        assert!(!snapshot.proves(&ValidationBinding {
            collection: Some(PlaceKey::new("_8")),
            ..binding.clone()
        }));

        state.record_write(subject, StaticWriteToken::new(0, 1));
        assert!(!ValidationSnapshot::capture(&state, 2).proves(&binding));
    }

    #[test]
    fn relational_range_obligation_requires_exact_bounds_collection_and_versions() {
        let collection = PlaceKey::new("_1");
        let range = PlaceKey::new("_2");
        let start = PlaceKey::new("_3");
        let end = PlaceKey::new("_4");
        let mut state = ProgramState::empty();
        for place in [&collection, &range, &start, &end] {
            seed_exact(&mut state, place);
        }
        for relation in [
            RelationFact::new(RelationKind::Eq(
                RelationTerm::Projection {
                    base: range.clone(),
                    path: vec![ProjectionKey::Field(0)],
                },
                RelationTerm::Place(start.clone()),
            )),
            RelationFact::new(RelationKind::Eq(
                RelationTerm::Projection {
                    base: range.clone(),
                    path: vec![ProjectionKey::Field(1)],
                },
                RelationTerm::Place(end.clone()),
            )),
            RelationFact::new(RelationKind::Le(
                RelationTerm::Place(start),
                RelationTerm::Place(end.clone()),
            )),
            RelationFact::new(RelationKind::Le(
                RelationTerm::Place(end),
                RelationTerm::Length(collection.clone()),
            )),
        ] {
            state.establish(BoundValidation::new(
                ProofBinding::Relation(relation.clone()),
                relation_storage_places(&relation),
            ));
        }
        let call = CallSite {
            callee: None,
            raw_def: None,
            disposition: BoundaryDisposition::ModeledRegistry,
            args: vec![
                CallOperand::Place(collection.clone()),
                CallOperand::Place(range.clone()),
            ],
            arg_values: vec![exact_value(&collection), exact_value(&range)],
            actual_endpoint_sources: Vec::new(),
            destination: PlaceKey::new("_0"),
            point: point(0, 0),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            index_range_kind: Some(RangeIndexKind::Between),
            generic_substitution: GenericCallSubstitution::default(),
        };
        let obligation = range_index_obligation(&call, &collection).expect("range obligation");
        let snapshot = ValidationSnapshot::capture(&state, 4);
        assert!(snapshot.proves_relational(&obligation));
        assert!(local_requirement_is_proven(
            &snapshot,
            None,
            Some(&obligation)
        ));
        let mut range_to = call.clone();
        range_to.index_range_kind = Some(RangeIndexKind::To);
        let range_to = range_index_obligation(&range_to, &collection).expect("range-to obligation");
        assert!(ValidationSnapshot::capture(&state, 4).proves_relational(&range_to));

        let collection_alias = PlaceKey::new("_5");
        let range_alias = PlaceKey::new("_6");
        seed_exact(&mut state, &collection_alias);
        seed_exact(&mut state, &range_alias);
        for relation in [
            RelationFact::new(RelationKind::Eq(
                RelationTerm::Place(collection.clone()),
                RelationTerm::Place(collection_alias.clone()),
            )),
            RelationFact::new(RelationKind::Eq(
                RelationTerm::Place(range.clone()),
                RelationTerm::Place(range_alias.clone()),
            )),
        ] {
            state.establish(BoundValidation::new(
                ProofBinding::Relation(relation.clone()),
                relation_storage_places(&relation),
            ));
        }
        let alias_snapshot = ValidationSnapshot::capture(&state, 6);
        let mut aliased_call = call.clone();
        aliased_call.arg_values[0]
            .exact_roots
            .insert(collection_alias.clone());
        aliased_call.arg_values[1]
            .exact_roots
            .insert(range_alias.clone());
        assert!(proven_range_index_obligation(&alias_snapshot, &aliased_call).is_some());

        let unrelated = PlaceKey::new("_7");
        seed_exact(&mut state, &unrelated);
        aliased_call.arg_values[0].exact_roots.insert(unrelated);
        assert!(proven_range_index_obligation(
            &ValidationSnapshot::capture(&state, 7),
            &aliased_call,
        )
        .is_none());

        let wrong_collection = PlaceKey::new("_8");
        seed_exact(&mut state, &wrong_collection);
        let wrong = range_index_obligation(&call, &wrong_collection).expect("wrong obligation");
        assert!(!ValidationSnapshot::capture(&state, 8).proves_relational(&wrong));

        state.record_write(range, StaticWriteToken::new(1, 0));
        let stale = ValidationSnapshot::capture(&state, 4);
        assert!(!stale.proves_relational(&obligation));
        assert!(!local_requirement_is_proven(
            &stale,
            None,
            Some(&obligation)
        ));
    }

    #[test]
    fn relational_struct_projection_survives_conditional_return_and_reborrow_mapping() {
        let formal = PlaceKey::new("_1");
        let start_place = PlaceKey::new("((*_1).0: usize)");
        let returned = PlaceKey::new("_0");
        let start_path = vec![ProjectionKey::Deref, ProjectionKey::Field(0)];
        let end_path = vec![ProjectionKey::Deref, ProjectionKey::Field(1)];
        let mut callee = ProgramState::empty();
        for place in [&formal, &start_place, &returned] {
            seed_exact(&mut callee, place);
        }
        let ordered = RelationFact::new(RelationKind::Lt(
            RelationTerm::Projection {
                base: formal.clone(),
                path: start_path.clone(),
            },
            RelationTerm::Projection {
                base: formal.clone(),
                path: end_path.clone(),
            },
        ));
        callee.establish(BoundValidation::new(
            ProofBinding::Relation(ordered.clone()),
            relation_storage_places(&ordered),
        ));
        let start_operand = OperandModel {
            operand: CallOperand::Place(start_place),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: None,
            projection_path: Some(start_path),
            range_index_kind: None,
            projection_unknown: false,
        };
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: returned,
                    destination_projection: None,
                    value: ValueModel::Aggregate {
                        operands: vec![start_operand],
                        range_end: None,
                        variant: Some(1),
                        closure: None,
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            0,
            &mut callee,
        );
        let condition = ReturnCondition {
            discriminant: 1,
            payload_path: vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)],
        };
        let expected = RelationalPostcondition::conditional(
            condition.clone(),
            SummaryRelationKind::Lt(
                SummaryRelationTerm::Return,
                SummaryRelationTerm::FormalProjection {
                    formal: 1,
                    path: end_path,
                },
            ),
        );
        assert!(project_relational_postconditions(&callee, 1).contains(&expected));

        let raw_reborrow = PlaceKey::new("_9");
        let range_value = PlaceKey::new("_8");
        let destination = PlaceKey::new("_5");
        let call = CallSite {
            callee: Some(FunctionKey::new("crate::next")),
            raw_def: None,
            disposition: BoundaryDisposition::ResolvedLocal,
            args: vec![CallOperand::Place(raw_reborrow)],
            arg_values: vec![exact_value(&range_value)],
            actual_endpoint_sources: Vec::new(),
            destination: destination.clone(),
            point: point(1, 0),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            index_range_kind: None,
            generic_substitution: GenericCallSubstitution::default(),
        };
        let (_, mapped) = instantiate_relational_postconditions(
            &BTreeSet::from([expected]),
            &call,
            &BTreeSet::new(),
        );
        assert_eq!(
            mapped.iter().next().expect("mapped relation").relation,
            RelationFact::new(RelationKind::Lt(
                RelationTerm::Projection {
                    base: destination,
                    path: condition.payload_path,
                },
                RelationTerm::Projection {
                    base: range_value,
                    path: vec![ProjectionKey::Field(1)],
                },
            )),
        );
    }

    #[test]
    fn relational_normal_return_projects_and_instantiates_exact_actuals() {
        let formal = PlaceKey::new("_1");
        let local = PlaceKey::new("_2");
        let returned = PlaceKey::new("_0");
        let mut state = ProgramState::empty();
        for place in [&formal, &local, &returned] {
            seed_exact(&mut state, place);
        }
        for relation in [
            RelationFact::new(RelationKind::Lt(
                RelationTerm::Place(local.clone()),
                RelationTerm::Length(formal.clone()),
            )),
            RelationFact::new(RelationKind::Eq(
                RelationTerm::Place(returned),
                RelationTerm::Place(local),
            )),
        ] {
            state.establish(BoundValidation::new(
                ProofBinding::Relation(relation.clone()),
                relation.storage_places(),
            ));
        }
        let expected = RelationalPostcondition::unconditional(SummaryRelationKind::Lt(
            SummaryRelationTerm::Return,
            SummaryRelationTerm::LengthFormal(1),
        ));
        let postconditions = project_relational_postconditions(&state, 1);
        assert!(postconditions.contains(&expected));

        let actual = PlaceKey::new("_9");
        let destination = PlaceKey::new("_10");
        let call = CallSite {
            callee: Some(FunctionKey::new("crate::callee")),
            raw_def: None,
            disposition: BoundaryDisposition::ResolvedLocal,
            args: vec![CallOperand::Place(actual.clone())],
            arg_values: vec![exact_value(&actual)],
            actual_endpoint_sources: Vec::new(),
            destination: destination.clone(),
            point: point(2, 0),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            index_range_kind: None,
            generic_substitution: GenericCallSubstitution::default(),
        };
        let (instantiated, conditional) = instantiate_relational_postconditions(
            &BTreeSet::from([expected.clone()]),
            &call,
            &BTreeSet::new(),
        );
        assert!(conditional.is_empty());
        assert_eq!(
            instantiated,
            BTreeSet::from([RelationFact::new(RelationKind::Lt(
                RelationTerm::Place(destination),
                RelationTerm::Length(actual),
            ))]),
        );

        let mut ambiguous = call;
        ambiguous.arg_values[0].semantic_unknown = true;
        let (instantiated, conditional) = instantiate_relational_postconditions(
            &BTreeSet::from([expected]),
            &ambiguous,
            &BTreeSet::new(),
        );
        assert!(instantiated.is_empty());
        assert!(conditional.is_empty());
    }

    #[test]
    fn relational_option_payload_is_proved_only_on_the_matching_variant_edge() {
        let formal = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let returned = PlaceKey::new("_0");
        let mut callee = ProgramState::empty();
        for place in [&formal, &index, &returned] {
            seed_exact(&mut callee, place);
        }
        let bound = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(index.clone()),
            RelationTerm::Length(formal.clone()),
        ));
        callee.establish(BoundValidation::new(
            ProofBinding::Relation(bound.clone()),
            relation_storage_places(&bound),
        ));
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: returned.clone(),
                    destination_projection: None,
                    value: ValueModel::Aggregate {
                        operands: vec![place_operand(&index.0)],
                        range_end: None,
                        variant: Some(1),
                        closure: None,
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            0,
            &mut callee,
        );

        let condition = ReturnCondition {
            discriminant: 1,
            payload_path: vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)],
        };
        let postcondition = RelationalPostcondition::conditional(
            condition.clone(),
            SummaryRelationKind::Lt(
                SummaryRelationTerm::Return,
                SummaryRelationTerm::LengthFormal(1),
            ),
        );
        assert!(project_relational_postconditions(&callee, 1).contains(&postcondition));

        let actual = PlaceKey::new("_9");
        let destination = PlaceKey::new("_5");
        let call_site = CallSite {
            callee: Some(FunctionKey::new("crate::callee")),
            raw_def: None,
            disposition: BoundaryDisposition::ResolvedLocal,
            args: vec![CallOperand::Place(actual.clone())],
            arg_values: vec![exact_value(&actual)],
            actual_endpoint_sources: Vec::new(),
            destination: destination.clone(),
            point: point(1, 0),
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            index_range_kind: None,
            generic_substitution: GenericCallSubstitution::default(),
        };
        let (unconditional, conditional) = instantiate_relational_postconditions(
            &BTreeSet::from([postcondition]),
            &call_site,
            &BTreeSet::new(),
        );
        assert!(unconditional.is_empty());
        let mapped = conditional.iter().next().expect("conditional relation");
        assert_eq!(mapped.condition, condition);
        assert_eq!(
            mapped.relation,
            RelationFact::new(RelationKind::Lt(
                RelationTerm::Projection {
                    base: destination.clone(),
                    path: vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)],
                },
                RelationTerm::Length(actual.clone()),
            ))
        );

        let call_point = point(1, 0);
        let normal_target = BasicBlock::from_usize(1);
        let call = CallModel {
            descriptor: CallDescriptor {
                callee: Some(FunctionKey::new("crate::callee")),
                raw_def: None,
                disposition: BoundaryDisposition::ResolvedLocal,
                args: vec![place_operand(&actual.0)],
                destination: destination.clone(),
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: Vec::new(),
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
        };
        let outputs = LocalCallOutputs::from([(
            call_point.clone(),
            MappedCallOutputs {
                conditional_relations: conditional,
                ..MappedCallOutputs::default()
            },
        )]);
        let mut caller = ProgramState::empty();
        seed_exact(&mut caller, &actual);
        apply_program_edge_with_outputs(
            &EdgeTerm::CallNormal {
                normal_target: Some(normal_target),
                point: call_point,
                call,
            },
            normal_target,
            &mut caller,
            &outputs,
        );

        let discriminant = PlaceKey::new("_6");
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: discriminant.clone(),
                    destination_projection: None,
                    value: ValueModel::Discriminant(place_operand(&destination.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(1),
            1,
            &mut caller,
        );
        let before_branch = caller.clone();
        let switch = EdgeTerm::Switch {
            condition: discriminant,
            false_target: BasicBlock::from_usize(2),
            true_target: BasicBlock::from_usize(3),
        };
        let mut on_some = before_branch.clone();
        apply_program_edge(&switch, BasicBlock::from_usize(3), &mut on_some);
        let mut on_none = before_branch;
        apply_program_edge(&switch, BasicBlock::from_usize(2), &mut on_none);

        let payload = PlaceKey::new("((_5 as Some).0: usize)");
        let payload_operand = OperandModel {
            operand: CallOperand::Place(payload.clone()),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: Some(payload.clone()),
            projection_path: Some(vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)]),
            range_index_kind: None,
            projection_unknown: false,
        };
        let loaded = PlaceKey::new("_7");
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(3, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: loaded,
                    destination_projection: None,
                    value: ValueModel::Operand(payload_operand),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(3),
            0,
            &mut on_some,
        );
        let binding = ValidationBinding {
            predicate: Predicate::InBounds,
            access_width: None,
            proof_definition: None,
            subject: payload.clone(),
            collection: Some(actual.clone()),
        };
        assert!(ValidationSnapshot::capture(&on_some, 9).proves(&binding));
        assert!(!ValidationSnapshot::capture(&on_none, 9).proves(&binding));
        assert!(
            !ValidationSnapshot::capture(&on_some, 9).proves(&ValidationBinding {
                subject: PlaceKey::new("((_5 as Some).1: usize)"),
                ..binding
            })
        );

        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(4, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: returned,
                    destination_projection: None,
                    value: ValueModel::Unknown,
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(4),
            0,
            &mut callee,
        );
        assert!(!project_relational_postconditions(&callee, 1).contains(
            &RelationalPostcondition::conditional(
                condition,
                SummaryRelationKind::Lt(
                    SummaryRelationTerm::Return,
                    SummaryRelationTerm::LengthFormal(1),
                ),
            ),
        ));
    }

    #[test]
    fn slice_position_establishes_an_exact_same_slice_bound_only_on_some() {
        let slice = PlaceKey::new("_1");
        let iterator = PlaceKey::new("_2");
        let position = PlaceKey::new("_3");
        let mut state = ProgramState::empty();
        seed_exact(&mut state, &slice);

        let iterator_target = BasicBlock::from_usize(1);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(iterator_target),
                point: point(0, 0),
                call: CallModel {
                    descriptor: CallDescriptor {
                        callee: None,
                        raw_def: None,
                        disposition: BoundaryDisposition::ModeledRegistry,
                        args: vec![place_operand(&slice.0)],
                        destination: iterator.clone(),
                        generic_substitution: GenericCallSubstitution::default(),
                    },
                    value: RegistryValueModel::SliceIterator(place_operand(&slice.0)),
                    predicate: None,
                    mutable_actuals: Vec::new(),
                    ffi_out_actuals: Vec::new(),
                    typed_initialization_effect: None,
                    destination_is_bool: false,
                    destination_restricted: false,
                    access_width: None,
                    legacy_write: false,
                    destination_out_formal: None,
                },
            },
            iterator_target,
            &mut state,
        );
        assert!(current_relation_proof(&state).proves_le(
            &RelationTerm::Length(iterator.clone()),
            &RelationTerm::Length(slice.clone()),
        ));

        let iterator_receiver = PlaceKey::new("_20");
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(1, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: iterator_receiver.clone(),
                    destination_projection: None,
                    value: ValueModel::RefOrRaw(place_operand(&iterator.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            iterator_target,
            0,
            &mut state,
        );

        let position_target = BasicBlock::from_usize(2);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(position_target),
                point: point(1, 0),
                call: CallModel {
                    descriptor: CallDescriptor {
                        callee: None,
                        raw_def: None,
                        disposition: BoundaryDisposition::ModeledRegistry,
                        args: vec![place_operand(&iterator_receiver.0)],
                        destination: position.clone(),
                        generic_substitution: GenericCallSubstitution::default(),
                    },
                    value: RegistryValueModel::SlicePosition(place_operand(&iterator_receiver.0)),
                    predicate: None,
                    mutable_actuals: Vec::new(),
                    ffi_out_actuals: Vec::new(),
                    typed_initialization_effect: None,
                    destination_is_bool: false,
                    destination_restricted: false,
                    access_width: None,
                    legacy_write: false,
                    destination_out_formal: None,
                },
            },
            position_target,
            &mut state,
        );

        let discriminant = PlaceKey::new("_4");
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(2, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: discriminant.clone(),
                    destination_projection: None,
                    value: ValueModel::Discriminant(place_operand(&position.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            position_target,
            0,
            &mut state,
        );
        let switch = EdgeTerm::Switch {
            condition: discriminant,
            false_target: BasicBlock::from_usize(3),
            true_target: BasicBlock::from_usize(4),
        };
        let mut on_some = state.clone();
        apply_program_edge(&switch, BasicBlock::from_usize(4), &mut on_some);
        let mut on_none = state;
        apply_program_edge(&switch, BasicBlock::from_usize(3), &mut on_none);

        let payload = RelationTerm::Projection {
            base: position,
            path: vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)],
        };
        let bound = RelationTerm::Length(slice);
        assert!(current_relation_proof(&on_some).proves_lt(&payload, &bound));
        assert!(!current_relation_proof(&on_none).proves_lt(&payload, &bound));

        let mut ambiguous = ProgramState::empty();
        seed_exact(&mut ambiguous, &iterator);
        for base in [PlaceKey::new("_7"), PlaceKey::new("_8")] {
            seed_exact(&mut ambiguous, &base);
            let relation = RelationFact::new(RelationKind::Eq(
                RelationTerm::Length(iterator.clone()),
                RelationTerm::Length(base),
            ));
            ambiguous.establish(BoundValidation::new(
                ProofBinding::Relation(relation.clone()),
                relation_storage_places(&relation),
            ));
        }
        assert!(current_slice_iterator_base(&ambiguous, &place_operand(&iterator.0)).is_none());
    }

    #[test]
    fn usize_range_next_uses_exact_symbolic_array_extents_only_on_some() {
        let indices = PlaceKey::new("_2");
        let output = PlaceKey::new("_5");
        let range_value = PlaceKey::new("_9");
        let iterator = PlaceKey::new("_8");
        let returned = PlaceKey::new("_13");
        let mut state = ProgramState::empty();
        for place in [&indices, &output, &range_value, &iterator] {
            seed_exact(&mut state, place);
        }

        let symbolic_length = OperandModel {
            operand: CallOperand::ArrayLengths(BTreeSet::from([indices.clone(), output.clone()])),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: None,
            projection_path: None,
            range_index_kind: None,
            projection_unknown: false,
        };
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: range_value.clone(),
                    destination_projection: None,
                    value: ValueModel::Aggregate {
                        operands: vec![constant_operand(0), symbolic_length],
                        range_end: None,
                        variant: None,
                        closure: None,
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            0,
            &mut state,
        );
        let into_iter_target = BasicBlock::from_usize(1);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(into_iter_target),
                point: point(0, 1),
                call: CallModel {
                    descriptor: CallDescriptor {
                        callee: None,
                        raw_def: None,
                        disposition: BoundaryDisposition::ModeledRegistry,
                        args: vec![place_operand(&range_value.0)],
                        destination: iterator.clone(),
                        generic_substitution: GenericCallSubstitution::default(),
                    },
                    value: RegistryValueModel::RangeIntoIterator(place_operand(&range_value.0)),
                    predicate: None,
                    mutable_actuals: Vec::new(),
                    ffi_out_actuals: Vec::new(),
                    typed_initialization_effect: None,
                    destination_is_bool: false,
                    destination_restricted: false,
                    access_width: None,
                    legacy_write: false,
                    destination_out_formal: None,
                },
            },
            into_iter_target,
            &mut state,
        );
        assert_eq!(
            current_range_iterator_bounds(&state, &place_operand(&iterator.0)),
            BTreeSet::from([indices.clone(), output.clone()]),
            "Range::into_iter preserves its exact symbolic end bound"
        );

        let call_target = BasicBlock::from_usize(2);
        apply_program_edge(
            &EdgeTerm::CallNormal {
                normal_target: Some(call_target),
                point: point(1, 0),
                call: CallModel {
                    descriptor: CallDescriptor {
                        callee: Some(FunctionKey::new("crate::range_next")),
                        raw_def: None,
                        disposition: BoundaryDisposition::ResolvedLocal,
                        args: vec![place_operand(&iterator.0)],
                        destination: returned.clone(),
                        generic_substitution: GenericCallSubstitution::default(),
                    },
                    value: RegistryValueModel::RangeIteratorNext(place_operand(&iterator.0)),
                    predicate: None,
                    mutable_actuals: vec![iterator.clone()],
                    ffi_out_actuals: Vec::new(),
                    typed_initialization_effect: None,
                    destination_is_bool: false,
                    destination_restricted: false,
                    access_width: None,
                    legacy_write: false,
                    destination_out_formal: None,
                },
            },
            call_target,
            &mut state,
        );
        assert_eq!(
            current_range_iterator_bounds(&state, &place_operand(&iterator.0)),
            BTreeSet::from([indices.clone(), output.clone()]),
            "Range::next mutates the cursor but preserves its exact end bound"
        );

        let discriminant = PlaceKey::new("_14");
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(2, 0)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: discriminant.clone(),
                    destination_projection: None,
                    value: ValueModel::Discriminant(place_operand(&returned.0)),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            call_target,
            0,
            &mut state,
        );
        let switch = EdgeTerm::Switch {
            condition: discriminant,
            false_target: BasicBlock::from_usize(2),
            true_target: BasicBlock::from_usize(3),
        };
        let mut on_some = state.clone();
        apply_program_edge(&switch, BasicBlock::from_usize(3), &mut on_some);
        let mut on_none = state;
        apply_program_edge(&switch, BasicBlock::from_usize(2), &mut on_none);

        let payload = RelationTerm::Projection {
            base: returned,
            path: vec![ProjectionKey::Downcast(1), ProjectionKey::Field(0)],
        };
        for collection in [indices, output] {
            let bound = RelationTerm::Length(collection);
            assert!(current_relation_proof(&on_some).proves_lt(&payload, &bound));
            assert!(!current_relation_proof(&on_none).proves_lt(&payload, &bound));
        }

        assert!(current_range_iterator_bounds(&on_some, &place_operand("_unrelated"),).is_empty());
    }

    #[test]
    fn relational_postcondition_requires_every_reachable_normal_return() {
        let formal = PlaceKey::new("_1");
        let returned = PlaceKey::new("_0");
        let relation = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(returned.clone()),
            RelationTerm::Length(formal.clone()),
        ));
        let mut proved = ProgramState::empty();
        seed_exact(&mut proved, &formal);
        seed_exact(&mut proved, &returned);
        proved.establish(BoundValidation::new(
            ProofBinding::Relation(relation.clone()),
            relation.storage_places(),
        ));
        let mut unproved = ProgramState::empty();
        seed_exact(&mut unproved, &formal);
        seed_exact(&mut unproved, &returned);
        let mut edges = IndexVec::new();
        edges.push(EdgeTerm::Return);
        edges.push(EdgeTerm::Return);
        edges.push(EdgeTerm::None);
        let solved = ProgramDataflowResult {
            entry: IndexVec::from_elem_n(None, 3),
            exit: IndexVec::from_raw(vec![Some(proved.clone()), Some(unproved), Some(proved)]),
            edge: BTreeMap::new(),
        };
        assert!(reachable_relational_postconditions(&edges, &solved, 1).is_empty());
    }

    #[test]
    fn relational_call_postcondition_is_installed_only_on_the_normal_edge() {
        let actual = PlaceKey::new("_1");
        let destination = PlaceKey::new("_2");
        let condition = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(destination.clone()),
            RelationTerm::Length(actual.clone()),
        ));
        let call_point = point(0, 0);
        let normal_target = BasicBlock::from_usize(1);
        let unwind_target = BasicBlock::from_usize(2);
        let edge = EdgeTerm::CallNormal {
            normal_target: Some(normal_target),
            point: call_point.clone(),
            call: CallModel {
                descriptor: CallDescriptor {
                    callee: Some(FunctionKey::new("crate::callee")),
                    raw_def: None,
                    disposition: BoundaryDisposition::ResolvedLocal,
                    args: vec![place_operand(&actual.0)],
                    destination: destination.clone(),
                    generic_substitution: GenericCallSubstitution::default(),
                },
                value: RegistryValueModel::Empty,
                predicate: None,
                mutable_actuals: Vec::new(),
                ffi_out_actuals: Vec::new(),
                typed_initialization_effect: None,
                destination_is_bool: false,
                destination_restricted: false,
                access_width: None,
                legacy_write: false,
                destination_out_formal: None,
            },
        };
        let outputs = LocalCallOutputs::from([(
            call_point,
            MappedCallOutputs {
                relations: BTreeSet::from([condition.clone()]),
                ..MappedCallOutputs::default()
            },
        )]);
        let mut entry = ProgramState::empty();
        seed_exact(&mut entry, &actual);
        seed_exact(&mut entry, &destination);

        let mut normal = entry.clone();
        apply_program_edge_with_outputs(&edge, normal_target, &mut normal, &outputs);
        assert!(current_relation_proof(&normal)
            .proves_in_bounds(&RelationTerm::Place(destination), &actual,));

        let mut unwind = entry;
        apply_program_edge_with_outputs(&edge, unwind_target, &mut unwind, &outputs);
        assert!(!current_relation_proof(&unwind).proves_lt(
            &RelationTerm::Place(PlaceKey::new("_2")),
            &RelationTerm::Length(PlaceKey::new("_1")),
        ));
    }

    #[test]
    fn task12_body_value_projection_persists_only_value_flow() {
        let flowed = PlaceKey::new("_7");
        let mut projected = BodyValueFacts::default();
        projected.join_value_facts(&ValueFacts {
            value: AbstractValue::new([AbstractOrigin::Formal(2)]),
            dependencies: BTreeSet::from([PlaceKey::new("_dependency")]),
            value_flow: BTreeSet::from([flowed.clone()]),
            length_atoms: BTreeSet::from([LengthAtom {
                collection: PlaceKey::new("_1"),
                definition: StaticWriteToken::new(0, 0),
                captured_versions: BTreeSet::new(),
                invalidated: false,
            }]),
            utf8_relation_unknown: true,
            havoced: true,
            ..ValueFacts::default()
        });

        assert_eq!(projected.value_flow, BTreeSet::from([flowed]));
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
                generic_substitution: GenericCallSubstitution::default(),
            },
            value: RegistryValueModel::Empty,
            predicate: None,
            mutable_actuals: vec![actual.clone()],
            ffi_out_actuals: Vec::new(),
            typed_initialization_effect: None,
            destination_is_bool: false,
            destination_restricted: false,
            access_width: None,
            legacy_write: false,
            destination_out_formal: None,
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
            Pattern::P4
        );
        assert_eq!(
            classify_primary_with_secondary(
                FailureClass::SinkPrecondition,
                SourceKind::OpenBehaviorOutput,
                [SourceKind::OpenBehaviorOutput],
            )
            .0,
            Pattern::P3
        );
    }

    #[test]
    fn validation_value_binding_rejects_a_different_origin() {
        let checked = AbstractValue::new([AbstractOrigin::Formal(1)]);
        let sink = AbstractValue::new([AbstractOrigin::Formal(2)]);
        assert!(checked.origins.is_disjoint(&sink.origins));
    }

    #[test]
    fn p4_subject_and_source_bind_to_collection_capability_not_constant_zero() {
        let capability = AbstractOrigin::GenericCapability {
            token: "associated-slice:crate::Provider::Results".into(),
            span: span(),
        };
        let collection = AbstractValue::new([capability.clone(), AbstractOrigin::Formal(1)]);
        let constant_index = AbstractValue::new([AbstractOrigin::Constant(0)]);

        let subject = generic_nonempty_subject(&collection, true).expect("exact P4.1 subject");
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
    fn representation_preserving_pointer_views_do_not_seed_p4_or_p6() {
        let cases = [
            (
                PointerTransformEvidence {
                    same_pointee: true,
                    has_generic: true,
                    ..PointerTransformEvidence::default()
                },
                PointerTransformKind::IdentityView,
            ),
            (
                PointerTransformEvidence {
                    sequence_element_view: true,
                    has_generic: true,
                    ..PointerTransformEvidence::default()
                },
                PointerTransformKind::AddressView,
            ),
            (
                PointerTransformEvidence {
                    transparent_wrapper_view: true,
                    has_generic: true,
                    ..PointerTransformEvidence::default()
                },
                PointerTransformKind::AddressView,
            ),
            (
                PointerTransformEvidence {
                    unit_erasure: true,
                    has_generic: true,
                    ..PointerTransformEvidence::default()
                },
                PointerTransformKind::AddressView,
            ),
            (
                PointerTransformEvidence {
                    compiler_layout_implies_target: true,
                    source_validity_implies_target: true,
                    ..PointerTransformEvidence::default()
                },
                PointerTransformKind::AddressView,
            ),
            (
                PointerTransformEvidence {
                    compiler_layout_implies_target: true,
                    source_validity_implies_target: false,
                    ..PointerTransformEvidence::default()
                },
                PointerTransformKind::Reinterpretation,
            ),
            (
                PointerTransformEvidence {
                    compiler_layout_implies_target: false,
                    source_validity_implies_target: true,
                    ..PointerTransformEvidence::default()
                },
                PointerTransformKind::Reinterpretation,
            ),
        ];
        for (evidence, expected) in cases {
            assert_eq!(classify_pointer_transform(evidence), expected);
        }

        let source = PlaceKey::new("_1");
        let endpoint = EndpointNode::Formal {
            owner: FunctionKey::new("crate::view"),
            index: 1,
        };
        let mut state = ProgramState::empty();
        state.set_value(
            source.clone(),
            ValueFacts {
                value: AbstractValue::new([AbstractOrigin::Formal(1)]),
                maybe_uninit_initialized: BTreeSet::from([false]),
                exact_roots: BTreeSet::from([source.clone()]),
                known_lengths: BTreeSet::from([4]),
                endpoint_sources: Arc::new(BTreeSet::from([endpoint.clone()])),
                ..ValueFacts::default()
            },
        );
        let model = ValueModel::IdentityPointerView {
            operand: place_operand(&source.0),
            generic_view: None,
        };
        let value = eval_value_model(&state, &model, StaticWriteToken::new(0, 0));
        assert_eq!(
            value.value.origins,
            BTreeSet::from([AbstractOrigin::Formal(1)])
        );
        assert_eq!(value.endpoint_sources.as_ref(), &BTreeSet::from([endpoint]));
        assert_eq!(value.maybe_uninit_initialized, BTreeSet::from([false]));
        assert_eq!(value.known_lengths, BTreeSet::from([4]));
        assert!(!value.generic_capability);
        assert_eq!(value_model_derivation_kind(&model), None);

        let registry_model = RegistryValueModel::AddressPointerView {
            operand: place_operand(&source.0),
            generic_view: None,
        };
        let registry_value = eval_registry_value(&state, &registry_model, &point(0, 1));
        assert_eq!(
            registry_value.value.origins,
            BTreeSet::from([AbstractOrigin::Formal(1)])
        );
        assert_eq!(
            registry_value.maybe_uninit_initialized,
            BTreeSet::from([false])
        );
        assert!(registry_value.known_lengths.is_empty());
        assert!(!registry_value.generic_capability);
        assert_eq!(registry_value_derivation_kind(&registry_model), None);
    }

    #[test]
    fn genuinely_generic_pointer_reinterpretation_remains_conservative() {
        assert_eq!(
            classify_pointer_transform(PointerTransformEvidence {
                has_generic: true,
                ..PointerTransformEvidence::default()
            }),
            PointerTransformKind::GenericLayoutRisk
        );
        assert_eq!(
            classify_pointer_transform(PointerTransformEvidence::default()),
            PointerTransformKind::Reinterpretation
        );
        assert_eq!(
            value_model_derivation_kind(&ValueModel::GenericPointerCast {
                operand: place_operand("_1"),
                evidence: test_generic_capability(
                    GenericCapabilityKind::PointerLayoutReinterpretation,
                ),
            }),
            None
        );
        assert_eq!(
            value_model_derivation_kind(&ValueModel::Aggregate {
                operands: vec![place_operand("_1")],
                range_end: None,
                variant: None,
                closure: None,
            }),
            None
        );
        assert_eq!(
            value_model_derivation_kind(&ValueModel::PointerCast {
                operand: place_operand("_1"),
                obligation: ConcreteLayoutObligation::default(),
            }),
            Some(DerivationKind::Layout)
        );
    }

    #[test]
    fn in_bounds_guard_accepts_ge_but_rejects_gt() {
        assert!(is_in_bounds_guard(BinOp::Ge));
        assert!(!is_in_bounds_guard(BinOp::Gt));
    }

    #[test]
    fn boundary_precedence_is_closed_and_opaque_calls_do_not_carry_p3_origins() {
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
                RuleId::P5AssumeInitBool,
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

    #[test]
    fn task10_stable_span_uses_normalized_paths_and_preserves_one_based_columns() {
        let project = Path::new("/workspace/project");
        let local = stable_span_from_locations(
            project,
            Some("/workspace/project/src/lib.rs"),
            None,
            7,
            1,
            7,
            5,
        )
        .expect("a local project path is stable");
        assert_eq!(local.path, "src/lib.rs");
        assert_eq!(local.start, StablePosition::new(7, 1));
        assert_eq!(local.end, StablePosition::new(7, 5));

        let remapped = stable_span_from_locations(
            project,
            Some("/outside/generated.rs"),
            Some("src/generated.rs"),
            3,
            3,
            3,
            3,
        )
        .expect("a normalized repository-relative remap is stable");
        assert_eq!(remapped.path, "src/generated.rs");
        assert_eq!(remapped.start.column, 3);
    }

    #[test]
    fn task10_unstable_span_identity_fails_closed_without_a_lib_rs_fallback() {
        let project = Path::new("/workspace/project");
        for (local, remapped) in [
            (Some("/outside/generated.rs"), None),
            (Some("/outside/generated.rs"), Some("/also/outside.rs")),
            (None, Some("../src/lib.rs")),
            (None, Some("./src/lib.rs")),
            (None, Some("src//lib.rs")),
            (None, Some("C:/src/lib.rs")),
            (None, Some("src\\lib.rs")),
        ] {
            assert!(
                stable_span_from_locations(project, local, remapped, 1, 1, 1, 2).is_err(),
                "unstable identity unexpectedly passed: local={local:?}, remapped={remapped:?}"
            );
        }
        assert!(stable_span_from_locations(project, None, Some("src/lib.rs"), 1, 0, 1, 1).is_err());
        assert!(stable_span_from_locations(project, None, Some("src/lib.rs"), 1, 2, 1, 1).is_err());
    }

    #[test]
    fn stdlib_source_identity_root_can_cover_a_routed_crate_and_sibling_includes() {
        let temporary = std::env::temp_dir().join(format!(
            "rapx-unsoundaudit-source-root-{}",
            std::process::id()
        ));
        let source_root = temporary.join("rust-src");
        let project_root = source_root.join("library/core");
        let project_source = project_root.join("src/lib.rs");
        let sibling = source_root.join("library/stdarch/crates/core_arch/src/aarch64/neon/mod.rs");
        std::fs::create_dir_all(project_source.parent().expect("project source parent"))
            .expect("create routed core root");
        std::fs::write(&project_source, "// routed crate source\n").expect("write project source");
        std::fs::create_dir_all(sibling.parent().expect("sibling parent"))
            .expect("create sibling root");
        std::fs::write(&sibling, "// stable sibling include\n").expect("write sibling source");

        let selected = select_source_root(&project_root, Some(source_root.clone()))
            .expect("the rust-src root contains the routed crate");
        let span = stable_span_from_locations(
            &selected,
            Some(&sibling.to_string_lossy()),
            None,
            1,
            1,
            1,
            2,
        )
        .expect("a sibling include has a stable rust-src-relative identity");
        assert_eq!(
            span.path,
            "library/stdarch/crates/core_arch/src/aarch64/neon/mod.rs"
        );
        assert!(select_source_root(&project_root, Some(temporary.join("outside"))).is_err());
        assert!(select_source_root(&temporary.join("missing-project"), None).is_err());
        let default = select_source_root(&project_root, None)
            .expect("fixtures default stable identities to the routed project root");
        assert_eq!(
            default,
            project_root.canonicalize().expect("canonical project root")
        );
        assert_eq!(
            stable_span_from_locations(
                &default,
                Some(&project_source.to_string_lossy()),
                None,
                1,
                1,
                1,
                2,
            )
            .expect("default fixture identity")
            .path,
            "src/lib.rs"
        );
        assert!(stable_span_from_locations(
            &selected,
            Some("/outside/generated.rs"),
            None,
            1,
            1,
            1,
            2,
        )
        .is_err());

        std::fs::remove_dir_all(&temporary).expect("remove source-root fixture");
    }

    #[test]
    fn relational_must_binding_survives_only_when_all_predecessors_prove_it() {
        use super::super::relations::{RelationFact, RelationKind, RelationTerm};

        let index = PlaceKey::new("_1");
        let slice = PlaceKey::new("_2");
        let relation = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(index.clone()),
            RelationTerm::Length(slice.clone()),
        ));
        let binding = ProofBinding::Relation(relation.clone());
        let places = relation.storage_places();

        let mut left = ProgramState::empty();
        left.record_write(index.clone(), StaticWriteToken::new(1, 0));
        left.establish(BoundValidation::new(binding.clone(), places.clone()));

        let mut right = ProgramState::empty();
        right.record_write(index.clone(), StaticWriteToken::new(2, 0));
        right.establish(BoundValidation::new(binding.clone(), places.clone()));

        let joined = ProgramState::join_predecessors([&left, &right])
            .expect("both predecessors are reachable");
        assert!(joined.has_current_validation(&binding, places.clone()));

        let missing = ProgramState::empty();
        let joined = ProgramState::join_predecessors([&left, &missing])
            .expect("both predecessors are reachable");
        assert!(!joined.has_current_validation(&binding, places));
    }

    #[test]
    fn relational_must_binding_is_killed_by_an_exact_or_may_write() {
        use super::super::relations::{RelationFact, RelationKind, RelationTerm};

        let index = PlaceKey::new("_1");
        let slice = PlaceKey::new("_2");
        let relation = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(index.clone()),
            RelationTerm::Length(slice.clone()),
        ));
        let binding = ProofBinding::Relation(relation.clone());
        let places = relation.storage_places();

        for may_write in [false, true] {
            let mut state = ProgramState::empty();
            state.establish(BoundValidation::new(binding.clone(), places.clone()));
            if may_write {
                state.record_may_write(slice.clone(), StaticWriteToken::new(3, 0));
            } else {
                state.record_write(slice.clone(), StaticWriteToken::new(3, 0));
            }
            assert!(!state.has_current_validation(&binding, places.clone()));
        }
    }

    #[test]
    fn object_equivalence_chain_requires_current_all_path_evidence() {
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_7");
        let guard_slice = PlaceKey::new("_8");
        let guard_borrow = PlaceKey::new("_9");
        let sink_slice = PlaceKey::new("_13");
        let sink_borrow = PlaceKey::new("_14");
        let base = PlaceKey::new("_1");
        let projection = RelationTerm::Projection {
            base: base.clone(),
            path: vec![ProjectionKey::Deref, ProjectionKey::Field(0)],
        };
        let relations = [
            RelationKind::Eq(
                RelationTerm::Place(length.clone()),
                RelationTerm::Length(guard_slice.clone()),
            ),
            RelationKind::Lt(
                RelationTerm::Place(index.clone()),
                RelationTerm::Place(length),
            ),
            RelationKind::SameObject(
                RelationTerm::Place(guard_slice),
                RelationTerm::Place(guard_borrow.clone()),
            ),
            RelationKind::Eq(RelationTerm::Place(guard_borrow), projection.clone()),
            RelationKind::Eq(RelationTerm::Place(sink_borrow.clone()), projection),
            RelationKind::SameObject(
                RelationTerm::Place(sink_slice.clone()),
                RelationTerm::Place(sink_borrow),
            ),
        ];
        let establish = |state: &mut ProgramState, relation: RelationKind| {
            let relation = RelationFact::new(relation);
            state.establish(BoundValidation::new(
                ProofBinding::Relation(relation.clone()),
                relation_storage_places(&relation),
            ));
        };
        let mut full = ProgramState::empty();
        for relation in relations.iter().cloned() {
            establish(&mut full, relation);
        }
        assert!(current_relation_proof(&full)
            .proves_in_bounds(&RelationTerm::Place(index.clone()), &sink_slice));

        let mut written = full.clone();
        written.record_write(base, StaticWriteToken::new(4, 0));
        assert!(!current_relation_proof(&written)
            .proves_in_bounds(&RelationTerm::Place(index.clone()), &sink_slice));

        let mut missing_on_one_path = ProgramState::empty();
        for relation in relations.iter().take(relations.len() - 1).cloned() {
            establish(&mut missing_on_one_path, relation);
        }
        let joined = ProgramState::join_predecessors([&full, &missing_on_one_path])
            .expect("both CFG predecessors are reachable");
        assert!(!current_relation_proof(&joined)
            .proves_in_bounds(&RelationTerm::Place(index), &sink_slice));
    }

    #[test]
    fn relational_switch_establishes_only_the_matching_exact_ordering() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let length = PlaceKey::new("_3");
        let condition = PlaceKey::new("_4");
        let mut before = ProgramState::empty();
        seed_exact(&mut before, &collection);
        seed_exact(&mut before, &index);

        let length_op = ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: length.clone(),
                destination_projection: None,
                value: ValueModel::Length(place_operand(&collection.0)),
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
        };
        apply_program_op(&length_op, BasicBlock::from_usize(0), 0, &mut before);
        let compare = ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(0, 1)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: condition.clone(),
                destination_projection: None,
                value: ValueModel::Compare {
                    op: BinOp::Lt,
                    left: place_operand(&index.0),
                    right: place_operand(&length.0),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
        };
        apply_program_op(&compare, BasicBlock::from_usize(0), 1, &mut before);
        let false_target = BasicBlock::from_usize(1);
        let true_target = BasicBlock::from_usize(2);
        let edge = EdgeTerm::Switch {
            condition,
            false_target,
            true_target,
        };

        let mut on_true = before.clone();
        apply_program_edge(&edge, true_target, &mut on_true);
        assert!(current_relation_proof(&on_true)
            .proves_in_bounds(&RelationTerm::Place(index.clone()), &collection,));

        let mut on_false = before;
        apply_program_edge(&edge, false_target, &mut on_false);
        assert!(!current_relation_proof(&on_false)
            .proves_in_bounds(&RelationTerm::Place(index), &collection));
    }

    #[test]
    fn branch_proved_increment_creates_a_nonoverflowing_add_relation() {
        let collection = PlaceKey::new("_1");
        let index = PlaceKey::new("_2");
        let next = PlaceKey::new("_3");
        let relation = RelationFact::new(RelationKind::Lt(
            RelationTerm::Place(index.clone()),
            RelationTerm::Length(collection.clone()),
        ));
        let mut state = ProgramState::empty();
        seed_exact(&mut state, &collection);
        seed_exact(&mut state, &index);
        state.establish(BoundValidation::new(
            ProofBinding::Relation(relation.clone()),
            relation.storage_places(),
        ));
        let add = ProgramOp {
            unsafe_sites: Vec::new(),
            point: Some(point(0, 0)),
            kind: ProgramOpKind::Assign(AssignmentModel {
                destination: next.clone(),
                destination_projection: None,
                value: ValueModel::Compare {
                    op: BinOp::Add,
                    left: place_operand(&index.0),
                    right: constant_operand(1),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            }),
        };
        apply_program_op(&add, BasicBlock::from_usize(0), 0, &mut state);
        assert!(current_relation_proof(&state).proves_le(
            &RelationTerm::Place(next),
            &RelationTerm::Length(collection.clone()),
        ));

        let checked = PlaceKey::new("_4");
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 1)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: checked.clone(),
                    destination_projection: None,
                    value: ValueModel::CheckedBinary {
                        op: BinOp::Add,
                        left: place_operand(&index.0),
                        right: constant_operand(1),
                    },
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            1,
            &mut state,
        );
        let checked_value = PlaceKey::new("(_4.0: usize)");
        let checked_operand = OperandModel {
            operand: CallOperand::Place(checked_value.clone()),
            field_origin: None,
            pure_deref_base: None,
            exact_projection: Some(checked_value.clone()),
            projection_path: Some(vec![ProjectionKey::Field(0)]),
            range_index_kind: None,
            projection_unknown: false,
        };
        let checked_result = PlaceKey::new("_5");
        apply_program_op(
            &ProgramOp {
                unsafe_sites: Vec::new(),
                point: Some(point(0, 2)),
                kind: ProgramOpKind::Assign(AssignmentModel {
                    destination: checked_result.clone(),
                    destination_projection: None,
                    value: ValueModel::Operand(checked_operand),
                    field_origin: None,
                    raw_read_source: None,
                    legacy_write: false,
                    out_formal: None,
                }),
            },
            BasicBlock::from_usize(0),
            2,
            &mut state,
        );
        assert!(current_relation_proof(&state).proves_le(
            &RelationTerm::Place(checked_value),
            &RelationTerm::Length(collection.clone()),
        ));

        let mut unproved = ProgramState::empty();
        seed_exact(&mut unproved, &collection);
        seed_exact(&mut unproved, &index);
        assert!(assignment_relation(
            &unproved,
            &AssignmentModel {
                destination: PlaceKey::new("_6"),
                destination_projection: None,
                value: ValueModel::CheckedBinary {
                    op: BinOp::Add,
                    left: place_operand(&index.0),
                    right: constant_operand(1),
                },
                field_origin: None,
                raw_read_source: None,
                legacy_write: false,
                out_formal: None,
            },
        )
        .is_none());
    }
}
