use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use petgraph::algo::dominators::simple_fast;
use petgraph::graph::DiGraph;

use super::dataflow::StaticWriteToken;
use super::summary::{FunctionKey, ProgramPoint, StableSpan};
use super::unsafe_site::{OperandRole, SuppressionReason};

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum StorageProvenance {
    #[default]
    Untracked,
    DefinitelyLocal,
    EntryFormal {
        owner: FunctionKey,
        index: u32,
    },
    MaybeExternal,
}

impl StorageProvenance {
    pub(super) fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Untracked, value) | (value, Self::Untracked) => value.clone(),
            (left, right) if left == right => left.clone(),
            _ => Self::MaybeExternal,
        }
    }

    pub(super) fn may_be_externally_controlled(&self) -> bool {
        matches!(self, Self::EntryFormal { .. } | Self::MaybeExternal)
    }

    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::Untracked => "untracked",
            Self::DefinitelyLocal => "definitely_local",
            Self::EntryFormal { .. } => "entry_formal",
            Self::MaybeExternal => "maybe_external",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum PublicFieldAdtKind {
    Struct,
    Union,
    Enum,
}

impl PublicFieldAdtKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Struct => "struct",
            Self::Union => "union",
            Self::Enum => "enum",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum DerivationKind {
    Index,
    Length,
    Offset,
    AccessWidth,
    Alignment,
    Layout,
    Discriminant,
    UncheckedBooleanRelation,
}

impl DerivationKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Length => "length",
            Self::Offset => "offset",
            Self::AccessWidth => "access_width",
            Self::Alignment => "alignment",
            Self::Layout => "layout",
            Self::Discriminant => "discriminant",
            Self::UncheckedBooleanRelation => "unchecked_boolean_relation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum GenericParamKind {
    Type,
    Const,
}

impl GenericParamKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Type => "type",
            Self::Const => "const",
        }
    }
}

/// Stable identity for a public type/const parameter that controls an unsafe
/// contract. Lifetimes are deliberately excluded: the P4 capability model is
/// about representation, size, alignment, initialization and validity facts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct GenericParamId {
    pub def_path: String,
    pub kind: GenericParamKind,
}

impl GenericParamId {
    pub(super) fn stable_identity(&self) -> String {
        format!("{}:{}", self.kind.as_str(), self.def_path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum GenericCapabilityKind {
    AssociatedSequenceShape { associated_items: BTreeSet<String> },
    PointerLayoutReinterpretation,
    RepresentationPreservingPointerView,
}

impl GenericCapabilityKind {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::AssociatedSequenceShape { .. } => "associated_sequence_shape",
            Self::PointerLayoutReinterpretation => "pointer_layout_reinterpretation",
            Self::RepresentationPreservingPointerView => "representation_preserving_pointer_view",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CapabilityTermRole {
    Sequence,
    Element,
    SourcePointee,
    TargetPointee,
    LengthConst,
}

impl CapabilityTermRole {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Sequence => "sequence",
            Self::Element => "element",
            Self::SourcePointee => "source_pointee",
            Self::TargetPointee => "target_pointee",
            Self::LengthConst => "length_const",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CapabilityTermForm {
    Concrete,
    DirectParam,
    Alias,
    Composite,
    Unknown,
}

impl CapabilityTermForm {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Concrete => "concrete",
            Self::DirectParam => "direct_param",
            Self::Alias => "alias",
            Self::Composite => "composite",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CapabilityTypeFacts {
    pub size: Option<u64>,
    pub align: Option<u64>,
    pub inhabited: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CapabilityTerm {
    pub form: CapabilityTermForm,
    pub controllers: BTreeSet<GenericParamId>,
    pub residual_aliases: BTreeSet<String>,
    pub equality_class: Option<u32>,
    pub type_facts: CapabilityTypeFacts,
}

impl CapabilityTerm {
    pub(super) fn stable_identity(&self) -> String {
        let controllers = self
            .controllers
            .iter()
            .map(GenericParamId::stable_identity)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "form={};controllers={controllers};aliases={};eq={};size={};align={};inhabited={}",
            self.form.as_str(),
            self.residual_aliases
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(","),
            self.equality_class
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.type_facts
                .size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.type_facts
                .align
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.type_facts
                .inhabited
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum AccessExtent {
    NotApplicable,
    Zero,
    Positive(u64),
    Symbolic {
        controllers: BTreeSet<GenericParamId>,
    },
    Unknown,
}

impl AccessExtent {
    pub(super) fn stable_identity(&self) -> String {
        match self {
            Self::NotApplicable => "not_applicable".to_string(),
            Self::Zero => "zero".to_string(),
            Self::Positive(width) => format!("positive:{width}"),
            Self::Symbolic { controllers } => format!(
                "symbolic:{}",
                controllers
                    .iter()
                    .map(GenericParamId::stable_identity)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Unknown => "unknown".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CapabilityRequirement {
    AssociatedSequenceNonEmpty,
    PointerLayout { extent: AccessExtent },
    RepresentationPreservingView,
}

impl CapabilityRequirement {
    pub(super) fn stable_identity(&self) -> String {
        match self {
            Self::AssociatedSequenceNonEmpty => "associated_sequence_nonempty".to_string(),
            Self::PointerLayout { extent } => {
                format!("pointer_layout:{}", extent.stable_identity())
            }
            Self::RepresentationPreservingView => "representation_preserving_view".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CapabilityViolation {
    InBounds,
    Size,
    Alignment,
}

impl CapabilityViolation {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::InBounds => "in_bounds",
            Self::Size => "size",
            Self::Alignment => "alignment",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum FiniteTypeWitness {
    EmptyArray,
    Unit,
    U8,
    U64,
    U8Array8,
    U64Array0,
}

impl FiniteTypeWitness {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::EmptyArray => "empty_array",
            Self::Unit => "unit",
            Self::U8 => "u8",
            Self::U64 => "u64",
            Self::U8Array8 => "u8_array_8",
            Self::U64Array0 => "u64_array_0",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CapabilityUnestablishedReason {
    MissingRoleTerm,
    ComplexBounds,
    GenericAssociatedType,
    UnsafeTrait,
    SealedOrPrivateTrait,
    NormalizationIncomplete,
    NoFiniteWitness,
}

impl CapabilityUnestablishedReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::MissingRoleTerm => "missing_role_term",
            Self::ComplexBounds => "complex_bounds",
            Self::GenericAssociatedType => "generic_associated_type",
            Self::UnsafeTrait => "unsafe_trait",
            Self::SealedOrPrivateTrait => "sealed_or_private_trait",
            Self::NormalizationIncomplete => "normalization_incomplete",
            Self::NoFiniteWitness => "no_finite_witness",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CapabilityImpossibleReason {
    NormalizedTermsEqual,
    ConcreteRootConfiguration,
    ZeroByteExtent,
    RepresentationPreservingView,
}

impl CapabilityImpossibleReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::NormalizedTermsEqual => "normalized_terms_equal",
            Self::ConcreteRootConfiguration => "concrete_root_configuration",
            Self::ZeroByteExtent => "zero_byte_extent",
            Self::RepresentationPreservingView => "representation_preserving_view",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CapabilityWitness {
    Proven {
        assignment: BTreeMap<CapabilityTermRole, FiniteTypeWitness>,
        violations: BTreeSet<CapabilityViolation>,
    },
    ImpossibleByCompilerFacts {
        reason: CapabilityImpossibleReason,
    },
    Unestablished {
        reason: CapabilityUnestablishedReason,
    },
}

impl CapabilityWitness {
    pub(super) fn stable_identity(&self) -> String {
        match self {
            Self::Proven {
                assignment,
                violations,
            } => format!(
                "proven:assignment={};violations={}",
                assignment
                    .iter()
                    .map(|(role, value)| format!("{}={}", role.as_str(), value.as_str()))
                    .collect::<Vec<_>>()
                    .join(","),
                violations
                    .iter()
                    .map(|violation| violation.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::ImpossibleByCompilerFacts { reason } => {
                format!("impossible:{}", reason.as_str())
            }
            Self::Unestablished { reason } => format!("unestablished:{}", reason.as_str()),
        }
    }

    pub(super) fn is_proven(&self) -> bool {
        matches!(self, Self::Proven { .. })
    }
}

/// Owned, compiler-derived evidence for a generic capability source. This is
/// intentionally not a debug string: classifiers must never recover semantics
/// by parsing rustc pretty-print output.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct GenericCapabilityEvidence {
    pub kind: GenericCapabilityKind,
    pub terms: BTreeMap<CapabilityTermRole, CapabilityTerm>,
    pub requirement: CapabilityRequirement,
    pub witness: CapabilityWitness,
}

impl GenericCapabilityEvidence {
    pub(super) fn controllers(&self) -> BTreeSet<GenericParamId> {
        self.terms
            .values()
            .flat_map(|term| term.controllers.iter().cloned())
            .collect()
    }

    /// Identity of the capability obligation itself. The selected finite
    /// witness is evidence for that obligation, not part of its identity.
    pub(super) fn source_identity(&self) -> String {
        let controllers = self
            .controllers()
            .iter()
            .map(GenericParamId::stable_identity)
            .collect::<Vec<_>>()
            .join(",");
        let terms = self
            .terms
            .iter()
            .map(|(role, term)| format!("{}={}", role.as_str(), term.stable_identity()))
            .collect::<Vec<_>>()
            .join(";");
        let associated_items = match &self.kind {
            GenericCapabilityKind::AssociatedSequenceShape { associated_items } => associated_items
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(","),
            GenericCapabilityKind::PointerLayoutReinterpretation
            | GenericCapabilityKind::RepresentationPreservingPointerView => String::new(),
        };
        format!(
            "kind={};controllers={controllers};associated_items={associated_items};terms={terms};requirement={}",
            self.kind.as_str(),
            self.requirement.stable_identity(),
        )
    }

    pub(super) fn stable_identity(&self) -> String {
        format!(
            "{};witness={}",
            self.source_identity(),
            self.witness.stable_identity()
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum GenericSubstitutionStatus {
    Complete,
    NormalizationIncomplete,
}

impl GenericSubstitutionStatus {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::NormalizationIncomplete => "normalization_incomplete",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct GenericArgBinding {
    pub kind: GenericParamKind,
    pub form: CapabilityTermForm,
    pub caller_controllers: BTreeSet<GenericParamId>,
    pub residual_aliases: BTreeSet<String>,
    pub behavior_receiver: Option<OpenBehaviorReceiver>,
    pub equality_class: Option<u32>,
    pub type_facts: CapabilityTypeFacts,
    pub const_value: Option<u64>,
}

impl GenericArgBinding {
    pub(super) fn stable_identity(&self) -> String {
        format!(
            "kind={};form={};controllers={};aliases={};behavior={};eq={};size={};align={};inhabited={};const={}",
            self.kind.as_str(),
            self.form.as_str(),
            self.caller_controllers
                .iter()
                .map(GenericParamId::stable_identity)
                .collect::<Vec<_>>()
                .join(","),
            self.residual_aliases.iter().cloned().collect::<Vec<_>>().join(","),
            self.behavior_receiver
                .as_ref()
                .map(OpenBehaviorReceiver::stable_identity)
                .unwrap_or_else(|| "none".to_string()),
            self.equality_class
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.type_facts
                .size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.type_facts
                .align
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.type_facts
                .inhabited
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            self.const_value
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct GenericCallSubstitution {
    pub status: GenericSubstitutionStatus,
    pub bindings: BTreeMap<GenericParamId, GenericArgBinding>,
}

impl Default for GenericCallSubstitution {
    fn default() -> Self {
        Self {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::new(),
        }
    }
}

impl GenericCallSubstitution {
    pub(super) fn stable_identity(&self) -> String {
        format!(
            "status={};bindings={}",
            self.status.as_str(),
            self.bindings
                .iter()
                .map(|(parameter, binding)| format!(
                    "{}=>{}",
                    parameter.stable_identity(),
                    binding.stable_identity()
                ))
                .collect::<Vec<_>>()
                .join(";")
        )
    }
}

/// A compiler-checked, finite assignment that may carry a direct callee
/// source/target parameter pair through one otherwise-un-normalizable
/// Internal call.  This is deliberately a template: the context index binds
/// it to the exact callsite and GenericCapability source identity later.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TypedWitnessTransportTemplate {
    pub source_parameter: GenericParamId,
    pub target_parameter: GenericParamId,
    pub caller_controller: GenericParamId,
    pub assignment: BTreeMap<CapabilityTermRole, FiniteTypeWitness>,
}

/// A template after it has been bound to the exact callsite predecessor and
/// capability source from which the finite witness originated.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TypedWitnessTransport {
    pub source_identity: String,
    pub template: TypedWitnessTransportTemplate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum ExternalSourceKind {
    Iterator,
    OpenTrait,
    Callback,
    FunctionPointer,
    FfiReturn,
    FfiOut,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum BehaviorOpenness {
    ProvenOpen,
    ProvenClosed,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum OpenBehaviorReceiver {
    Parameter(GenericParamId),
    Dynamic,
    Alias(BTreeSet<GenericParamId>),
    Unknown,
}

impl OpenBehaviorReceiver {
    pub(super) fn stable_identity(&self) -> String {
        match self {
            Self::Parameter(parameter) => {
                format!("parameter:{}", parameter.stable_identity())
            }
            Self::Dynamic => "dynamic".to_string(),
            Self::Alias(parameters) => format!(
                "alias:{}",
                parameters
                    .iter()
                    .map(GenericParamId::stable_identity)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Unknown => "unknown".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum BehaviorController {
    OpenTrait {
        trait_id: FunctionKey,
        openness: BehaviorOpenness,
        resolved_target: Option<FunctionKey>,
        receiver: OpenBehaviorReceiver,
    },
    PublicFormal {
        owner: FunctionKey,
        index: u32,
    },
    PublicField {
        access_owner: FunctionKey,
        def_path: String,
        access_path: super::summary::PlaceKey,
        base: super::summary::PlaceKey,
        storage: StorageProvenance,
        storage_versions: BTreeSet<StaticWriteToken>,
        adt_kind: PublicFieldAdtKind,
        persistent: bool,
        span: StableSpan,
    },
    InternalFunction {
        function: FunctionKey,
    },
    InternalClosure {
        function: FunctionKey,
    },
    FfiHost,
    Unknown,
}

/// A resolved local call is a semantic boundary, not merely an edge label.
///
/// In particular, two invocations of the same callee must never share a
/// return/out endpoint: doing so would permit the actual of call A to reach
/// the result of call B through the callee's summary nodes.  This is owned
/// data so it can participate in the finite, ordered endpoint relation set.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CallSiteId {
    pub caller: FunctionKey,
    pub point: ProgramPoint,
    pub callee: FunctionKey,
}

impl CallSiteId {
    pub(super) fn new(caller: FunctionKey, point: ProgramPoint, callee: FunctionKey) -> Self {
        assert_eq!(
            point.function, caller,
            "a callsite point must belong to its caller"
        );
        Self {
            caller,
            point,
            callee,
        }
    }

    pub(super) fn stable_identity(&self) -> String {
        format!(
            "{}@{}:{}:{}=>{}",
            self.caller.0,
            self.point.span.token(),
            self.point.block,
            self.point.statement,
            self.callee.0,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CallSiteOutput {
    Return,
    Out(u32),
}

impl BehaviorController {
    pub(super) fn rejection_reason(&self) -> Option<SuppressionReason> {
        match self {
            Self::OpenTrait {
                openness: BehaviorOpenness::ProvenClosed,
                ..
            } => Some(SuppressionReason::TraitNotDownstreamImplementable),
            Self::InternalFunction { .. } | Self::InternalClosure { .. } => {
                Some(SuppressionReason::InternalBehaviorController)
            }
            Self::OpenTrait {
                openness: BehaviorOpenness::ProvenOpen | BehaviorOpenness::Unknown,
                ..
            }
            | Self::PublicFormal { .. }
            | Self::PublicField { .. }
            | Self::FfiHost
            | Self::Unknown => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum EndpointNode {
    Formal {
        owner: FunctionKey,
        index: u32,
    },
    Capture {
        owner: FunctionKey,
        index: u32,
    },
    PublicField {
        access_owner: FunctionKey,
        def_path: String,
        access_path: super::summary::PlaceKey,
        base: super::summary::PlaceKey,
        storage: StorageProvenance,
        storage_versions: BTreeSet<StaticWriteToken>,
        adt_kind: PublicFieldAdtKind,
        persistent: bool,
        span: StableSpan,
    },
    ExternalBehavior {
        owner: FunctionKey,
        kind: ExternalSourceKind,
        controller: BehaviorController,
        token: String,
        span: StableSpan,
    },
    InternalDerivation {
        owner: FunctionKey,
        kind: DerivationKind,
        token: String,
        point: ProgramPoint,
    },
    GenericCapability {
        owner: FunctionKey,
        evidence: GenericCapabilityEvidence,
        span: StableSpan,
    },
    UnsafeOperand {
        owner: FunctionKey,
        site_id: String,
        role: OperandRole,
    },
    UnsafeResult {
        owner: FunctionKey,
        site_id: String,
    },
    Return {
        owner: FunctionKey,
    },
    Out {
        owner: FunctionKey,
        index: u32,
    },
    /// A caller-local result of one exact resolved call.  It deliberately
    /// cannot be connected to another callsite's callee `Return`/`Out` node.
    CallResult {
        callsite: CallSiteId,
        output: CallSiteOutput,
    },
    Exposure {
        owner: FunctionKey,
        token: String,
        point: ProgramPoint,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum EndpointEdgeKind {
    Transparent,
    Derivation,
    /// A compiler-identified `MaybeUninit::{as_ptr, as_mut_ptr}` view.  The
    /// edge preserves endpoint identity but explicitly derives address and
    /// validity-bit influence for the selected operand corridor.
    TypedValidityPointer,
    /// A resolved local call returned a non-public aggregate through a typed
    /// transformation for which no capability transfer is modeled. Public
    /// input influence remains auditable, but it is not promoted to an active
    /// finding solely from the aggregate's later use as an unsafe operand.
    PrivateAggregateTransform,
    ActualToFormal,
    /// One finite callee endpoint relation instantiated at one real callsite.
    /// The edge never enters a shared callee Formal/Return node.
    CallSummary,
    ReturnToDestination,
    OutToCaller,
    UnsafeOperand,
    UnsafeResult,
    Exposure,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct EndpointEdge {
    pub from: EndpointNode,
    pub to: EndpointNode,
    pub kind: EndpointEdgeKind,
    pub point: Option<ProgramPoint>,
}

/// A finite endpoint relation.  This is deliberately only a pair of already
/// materialized endpoint atoms: caller stacks and expanded paths are witness
/// data, never fixed-point keys.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ContextRelationKey {
    /// The function whose finite summary exports this relation.  Identical
    /// endpoint atoms in g/f/h must not collapse while their callsite
    /// substitutions are still distinct.
    pub context_owner: FunctionKey,
    pub source: EndpointNode,
    pub target: EndpointNode,
}

impl ContextRelationKey {
    pub(super) fn new(
        context_owner: FunctionKey,
        source: EndpointNode,
        target: EndpointNode,
    ) -> Self {
        Self {
            context_owner,
            source,
            target,
        }
    }
}

/// An endpoint atom scoped to the function relation that currently exports
/// it.  The same callee-owned endpoint can occur under many caller summaries;
/// keeping the owner here prevents those instances from being spliced.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ContextAtom {
    pub owner: FunctionKey,
    pub endpoint: EndpointNode,
}

impl ContextAtom {
    pub(super) fn scoped(owner: FunctionKey, endpoint: EndpointNode) -> Self {
        Self { owner, endpoint }
    }

    pub(super) fn from_source(relation: &ContextRelationKey) -> Self {
        Self {
            owner: relation.context_owner.clone(),
            endpoint: relation.source.clone(),
        }
    }

    pub(super) fn from_target(relation: &ContextRelationKey) -> Self {
        Self {
            owner: relation.context_owner.clone(),
            endpoint: relation.target.clone(),
        }
    }
}

/// The exact callee input represented by an instantiated call-summary edge.
/// `Internal` means that the callee relation starts at a callee-local source,
/// rather than one of the caller's actuals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum ContextActualBinding {
    Formal(u32),
    Internal,
}

/// One ordered, finite proof alternative for a context relation.  The store
/// retains every alternative; consumers may select a canonical route later,
/// but that selection can never erase another realizable call context.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum ContextPredecessor {
    Local {
        /// One interned local edge is shared by the raw, exact, and
        /// query-cropped proof stores.  Local edges occur in all three, so
        /// copying their endpoint payload here was needlessly quadratic.
        fragment: Arc<EndpointEdge>,
    },
    Call {
        /// These three facts are intrinsic to the call alternative and are
        /// intentionally interned.  A callee summary may be instantiated at
        /// many sources, and the exact index retains the alternative again;
        /// Arc keeps those exact copies as references rather than duplicating
        /// function paths, generic bindings, and endpoint atoms.
        callsite: Arc<CallSiteId>,
        substitution: Arc<GenericCallSubstitution>,
        callee_relation: Arc<ContextRelationKey>,
        actual_binding: ContextActualBinding,
    },
}

/// Compact internal predecessor payload. The owning relation already carries
/// the local edge endpoints, so retaining them once more for every local
/// predecessor would duplicate the aggregated endpoint graph at finalize.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum StoredContextPredecessor {
    Local {
        kind: EndpointEdgeKind,
        point: Option<Arc<ProgramPoint>>,
    },
    Call {
        callsite: Arc<CallSiteId>,
        substitution: Arc<GenericCallSubstitution>,
        callee_relation: Arc<ContextRelationKey>,
        actual_binding: ContextActualBinding,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum ContextUnknownReason {
    /// A closure/callback bridge still uses the old shared formal/return lane.
    UncontextualizedCallback,
    /// The call was resolved but the endpoint frontier for one actual was not
    /// available at this replay point.
    MissingActualEndpoint { callsite: CallSiteId },
    /// A local body or endpoint relation was unavailable for exact expansion.
    MissingResolvedBody { callsite: CallSiteId },
    /// Generic normalization at this exact callsite did not finish.
    NormalizationIncomplete { callsite: CallSiteId },
    /// The caller did not establish a closed, exact resolved-call universe.
    CompletenessUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ContextCompletenessEvidence {
    /// The only form that may justify a hard negative.  The current MIR
    /// exporter does not manufacture this evidence: it is reserved for a
    /// future closed-corridor audit that accounts for every local successor.
    ClosedResolvedCorridor {
        source: EndpointNode,
        target: EndpointNode,
        relation_count: usize,
    },
    /// We deliberately expose why a negative answer was withheld.  Absence
    /// from the finite summary alone is not a proof that no execution exists.
    Unavailable {
        root: EndpointNode,
        source: EndpointNode,
        target: EndpointNode,
    },
}

/// A finite proof DAG for one realizable context query.  `predecessors`
/// contains every endpoint relation that lies on some source-to-target route,
/// including all local and call hyperedge alternatives.  Recursive components
/// are represented by finite SCC member sets rather than expanded call stacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ContextProof {
    /// The caller summary that owns the first realizable execution step.
    /// A future candidate can prepend a public-root route to this owner
    /// without guessing from the intrinsic owner of an internal source.
    pub execution_anchor_owner: FunctionKey,
    pub root_context: FunctionKey,
    pub root: EndpointNode,
    pub source_context: FunctionKey,
    pub source: EndpointNode,
    pub target_context: FunctionKey,
    pub target: EndpointNode,
    pub predecessors: BTreeMap<ContextRelationKey, BTreeSet<ContextPredecessor>>,
    /// For every recursive relation-dependency SCC retained in this proof,
    /// the exact local/escaping relations that ground it. A bare SCC member
    /// set is never treated as a successful execution route.
    pub recursive_sccs: BTreeSet<BTreeSet<ContextRelationKey>>,
    pub recursive_scc_grounding:
        BTreeMap<BTreeSet<ContextRelationKey>, BTreeSet<ContextRelationKey>>,
}

/// One concrete, compatible choice through a [`ContextProof`].  In
/// particular, every relation in `selections` has exactly one predecessor;
/// a caller cannot take the prefix of one `Call` predecessor and the suffix
/// of another merely because both happened to export the same endpoint pair.
///
/// Recursive call dependencies are represented on [`ContextProof`] by their
/// finite SCC member set. They are a proof-DAG token, not an unbounded
/// reconstructed call stack; this selected route itself must still bottom out
/// through a grounded predecessor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ContextProofAlternative {
    pub selections: BTreeMap<ContextRelationKey, ContextPredecessor>,
    pub steps: Vec<ContextPredecessor>,
}

impl ContextProofAlternative {
    pub(super) fn callsites(&self) -> Vec<(CallSiteId, GenericCallSubstitution)> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                ContextPredecessor::Call {
                    callsite,
                    substitution,
                    ..
                } => Some(((**callsite).clone(), (**substitution).clone())),
                ContextPredecessor::Local { .. } => None,
            })
            .collect()
    }

    /// This status is deliberately attached to one selected proof
    /// alternative.  A normalization-incomplete sibling must not taint an
    /// independently complete call context.
    pub(super) fn has_incomplete_generic_substitution(&self) -> bool {
        self.steps.iter().any(|step| {
            matches!(
                step,
                ContextPredecessor::Call {
                    substitution,
                    ..
                } if substitution.status == GenericSubstitutionStatus::NormalizationIncomplete
            )
        })
    }
}

#[derive(Clone, Debug)]
enum AlternativePending {
    Reach {
        at: ContextAtom,
        target: ContextAtom,
        visited: BTreeSet<ContextAtom>,
    },
    Select(ContextRelationKey),
    ExitCall(ContextRelationKey),
}

#[derive(Clone, Debug)]
struct AlternativeState {
    pending: Vec<AlternativePending>,
    selections: BTreeMap<ContextRelationKey, ContextPredecessor>,
    steps: Vec<ContextPredecessor>,
    expanding_calls: BTreeSet<ContextRelationKey>,
}

/// Lazy, finite enumeration of compatible proof alternatives for
/// display/diagnostics only. It holds only the DFS frontier; it never first
/// materializes the Cartesian product of all predecessor choices. It uses a
/// simple path and skips recursive predecessor choices, so requirement
/// eligibility (notably P4) must instead use the exact relation APIs below in
/// its own product-state least fixed point.
pub(super) struct ContextProofAlternatives<'proof> {
    proof: &'proof ContextProof,
    outgoing: BTreeMap<ContextAtom, Vec<ContextRelationKey>>,
    backward: BTreeMap<ContextAtom, BTreeSet<ContextAtom>>,
    frontier: Vec<AlternativeState>,
}

impl<'proof> ContextProofAlternatives<'proof> {
    fn new(proof: &'proof ContextProof) -> Self {
        let mut outgoing = BTreeMap::<ContextAtom, Vec<ContextRelationKey>>::new();
        let mut reverse = BTreeMap::<ContextAtom, BTreeSet<ContextAtom>>::new();
        for relation in proof.predecessors.keys() {
            let source = ContextAtom::from_source(relation);
            let target = ContextAtom::from_target(relation);
            outgoing
                .entry(source.clone())
                .or_default()
                .push(relation.clone());
            reverse.entry(target).or_default().insert(source);
        }
        for relations in outgoing.values_mut() {
            relations.sort();
        }
        let root = ContextAtom {
            owner: proof.root_context.clone(),
            endpoint: proof.root.clone(),
        };
        let target = ContextAtom {
            owner: proof.target_context.clone(),
            endpoint: proof.target.clone(),
        };
        Self {
            proof,
            outgoing,
            backward: reverse,
            frontier: vec![AlternativeState {
                pending: vec![AlternativePending::Reach {
                    at: root.clone(),
                    target,
                    visited: BTreeSet::from([root]),
                }],
                selections: BTreeMap::new(),
                steps: Vec::new(),
                expanding_calls: BTreeSet::new(),
            }],
        }
    }

    fn can_reach_target(&self, start: &ContextAtom, target: &ContextAtom) -> bool {
        if start == target {
            return true;
        }
        let mut visited = BTreeSet::from([target.clone()]);
        let mut pending = VecDeque::from([target.clone()]);
        while let Some(node) = pending.pop_front() {
            for predecessor in self.backward.get(&node).into_iter().flatten() {
                if predecessor == start {
                    return true;
                }
                if visited.insert(predecessor.clone()) {
                    pending.push_back(predecessor.clone());
                }
            }
        }
        false
    }

    fn push_relation_choices(&mut self, state: AlternativeState, relation: ContextRelationKey) {
        let Some(choices) = self.proof.predecessors.get(&relation) else {
            return;
        };
        // Stack is LIFO; reverse preserves BTree's stable lexical order for
        // the observable canonical (first complete) alternative.
        for choice in choices.iter().rev() {
            if let Some(existing) = state.selections.get(&relation) {
                if existing != choice {
                    continue;
                }
            }
            let mut next = state.clone();
            next.selections
                .entry(relation.clone())
                .or_insert_with(|| choice.clone());
            next.steps.push(choice.clone());
            if let ContextPredecessor::Call {
                callee_relation, ..
            } = choice
            {
                if next.expanding_calls.contains(callee_relation) {
                    // Do not turn a recursive dependency into a successful
                    // leaf. `finalize` proves that this relation has *some*
                    // grounded alternative, not that this selected Call is
                    // grounded. A canonical route therefore skips the cycle
                    // and can select the SCC's local/escaping base instead.
                    // The finite SCC token remains on ContextProof for the
                    // witness/DAG consumer.
                    continue;
                } else {
                    next.expanding_calls.insert((**callee_relation).clone());
                    next.pending
                        .push(AlternativePending::ExitCall((**callee_relation).clone()));
                    next.pending.push(AlternativePending::Reach {
                        at: ContextAtom::from_source(callee_relation),
                        target: ContextAtom::from_target(callee_relation),
                        visited: BTreeSet::from([ContextAtom::from_source(callee_relation)]),
                    });
                }
            }
            self.frontier.push(next);
        }
    }
}

impl Iterator for ContextProofAlternatives<'_> {
    type Item = ContextProofAlternative;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut state) = self.frontier.pop() {
            let Some(pending) = state.pending.pop() else {
                return Some(ContextProofAlternative {
                    selections: state.selections,
                    steps: state.steps,
                });
            };
            match pending {
                AlternativePending::Reach {
                    at,
                    target,
                    visited,
                } => {
                    if at == target {
                        self.frontier.push(state);
                        continue;
                    }
                    let candidates = self.outgoing.get(&at).cloned().unwrap_or_default();
                    for relation in candidates.into_iter().rev() {
                        let next_atom = ContextAtom::from_target(&relation);
                        if visited.contains(&next_atom)
                            || !self.can_reach_target(&next_atom, &target)
                        {
                            continue;
                        }
                        let mut next = state.clone();
                        let mut next_visited = visited.clone();
                        next_visited.insert(next_atom.clone());
                        // Select must run before the route continuation so a
                        // call's callee proof is nested at the exact boundary.
                        next.pending.push(AlternativePending::Reach {
                            at: next_atom,
                            target: target.clone(),
                            visited: next_visited,
                        });
                        next.pending.push(AlternativePending::Select(relation));
                        self.frontier.push(next);
                    }
                }
                AlternativePending::Select(relation) => self.push_relation_choices(state, relation),
                AlternativePending::ExitCall(relation) => {
                    state.expanding_calls.remove(&relation);
                    self.frontier.push(state);
                }
            }
        }
        None
    }
}

impl ContextProof {
    /// Lazily enumerate compatible predecessor choices for a witness or
    /// diagnostic. This is not a requirement-eligibility API: consumers with
    /// generic/capability state must use `ContextProofIndex`'s exact relation
    /// alternatives in a product-state fixed point.
    pub(super) fn alternatives(&self) -> ContextProofAlternatives<'_> {
        ContextProofAlternatives::new(self)
    }

    /// Select one stable, *complete* alternative for display.  This no
    /// longer chooses `alternatives.first()` independently per relation.
    pub(super) fn canonical_alternative(&self) -> Option<ContextProofAlternative> {
        self.alternatives().next()
    }

    /// Selects one stable route only for display.
    pub(super) fn canonical_steps(&self) -> Vec<ContextPredecessor> {
        self.canonical_alternative()
            .map(|alternative| alternative.steps)
            .unwrap_or_default()
    }

    pub(super) fn canonical_callsites(&self) -> Vec<(CallSiteId, GenericCallSubstitution)> {
        self.canonical_alternative()
            .map(|alternative| alternative.callsites())
            .unwrap_or_default()
    }

    /// Status of the canonical *display* route only. It must not be used to
    /// accept/reject a requirement when a sibling alternative exists.
    pub(super) fn canonical_has_incomplete_generic_substitution(&self) -> bool {
        self.canonical_alternative()
            .is_some_and(|alternative| alternative.has_incomplete_generic_substitution())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ContextQuery {
    Realizable(ContextProof),
    ProvenUnrealizable(ContextCompletenessEvidence),
    Unknown {
        reasons: BTreeSet<ContextUnknownReason>,
        completeness: ContextCompletenessEvidence,
    },
}

/// Ordered predecessor/hyperedge index for callsite-realizable endpoint
/// relations.  It is intentionally separate from the coarse endpoint graph:
/// the latter remains a high-recall candidate source, while this index is the
/// only authority for a later hard `unrealizable_call_context` suppression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ContextRelationId(u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ContextAtomId(u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ContextPredecessorId(u32);

/// Opaque compact handles for requirement-specific product solvers.  The
/// insertion IDs never define observable order; every facade method restores
/// the decoded semantic order before returning handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ContextAtomHandle(ContextAtomId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ContextRelationHandle(ContextRelationId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ContextPredecessorHandle(ContextPredecessorId);

/// Compact reverse corridor for one endpoint across every exact caller scope.
/// Atom IDs are dense arena indices, so membership is one indexed load rather
/// than a tree lookup.  The mask is reusable by every source considered for
/// the same unsafe operand.
pub(super) struct ExactReverseReachability {
    reached_atoms: Vec<bool>,
    target_by_owner: BTreeMap<FunctionKey, Option<ContextAtomHandle>>,
}

impl ExactReverseReachability {
    pub(super) fn contains(&self, atom: ContextAtomHandle) -> bool {
        self.reached_atoms
            .get(atom.0 .0 as usize)
            .copied()
            .unwrap_or(false)
    }

    pub(super) fn target_in_owner(&self, owner: &FunctionKey) -> Option<ContextAtomHandle> {
        self.target_by_owner.get(owner).copied().flatten()
    }
}

/// Borrowed decoding of one exact predecessor alternative.  Local endpoints
/// are intentionally absent: they are already the owning relation's source
/// and target and can be reconstructed only for a selected display witness.
pub(super) enum ExactPredecessorRef<'a> {
    Local {
        kind: EndpointEdgeKind,
        point: Option<&'a ProgramPoint>,
    },
    Call {
        callsite: &'a CallSiteId,
        substitution: &'a GenericCallSubstitution,
        callee_relation: ContextRelationHandle,
        actual_binding: ContextActualBinding,
    },
}

pub(super) struct ContextPredecessorRefs<'a> {
    index: &'a ContextProofIndex,
    relation: ContextRelationId,
    ids: Vec<ContextPredecessorId>,
    next: usize,
}

impl<'a> Iterator for ContextPredecessorRefs<'a> {
    type Item = ContextPredecessor;

    fn next(&mut self) -> Option<Self::Item> {
        let id = *self.ids.get(self.next)?;
        self.next += 1;
        Some(self.index.materialize_predecessor(self.relation, id))
    }
}

impl ContextPredecessorRefs<'_> {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.ids.len().saturating_sub(self.next)
    }
}

pub(super) struct ContextExactRelations<'a> {
    index: &'a ContextProofIndex,
    ids: Vec<ContextRelationId>,
    next: usize,
}

impl<'a> Iterator for ContextExactRelations<'a> {
    type Item = (&'a ContextRelationKey, ContextPredecessorRefs<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let id = *self.ids.get(self.next)?;
        self.next += 1;
        let predecessors = self.index.exact_predecessor_refs(id);
        Some((self.index.relation(id), predecessors))
    }
}

/// Interned payload backing the compact proof index. The `BTreeMap` keys and
/// arena values share the same `Arc`; IDs, rather than heavyweight endpoint
/// and function keys, are retained by raw/exact adjacency and worklists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ContextProofArena {
    relation_ids: BTreeMap<Arc<ContextRelationKey>, ContextRelationId>,
    relations: Vec<Arc<ContextRelationKey>>,
    atom_ids: BTreeMap<Arc<ContextAtom>, ContextAtomId>,
    atoms: Vec<Arc<ContextAtom>>,
    predecessor_ids: BTreeMap<Arc<StoredContextPredecessor>, ContextPredecessorId>,
    predecessors: Vec<Arc<StoredContextPredecessor>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ContextProofIndex {
    /// Raw callsite alternatives are populated during the propagation hot
    /// phase. Relation keys are interned once in `relations`; postponing raw
    /// atom adjacency construction until `finalize` avoids retaining three
    /// heavyweight endpoint/function copies per raw relation at that peak.
    arena: ContextProofArena,
    predecessors: BTreeMap<ContextRelationId, BTreeSet<ContextPredecessorId>>,
    unknown_at: BTreeMap<EndpointNode, BTreeSet<ContextUnknownReason>>,
    unknown_by_callsite: BTreeMap<CallSiteId, BTreeSet<ContextUnknownReason>>,
    typed_witness_by_callsite: BTreeMap<CallSiteId, BTreeSet<TypedWitnessTransport>>,
    outgoing: BTreeMap<ContextAtomId, BTreeSet<ContextAtomId>>,
    incoming: BTreeMap<ContextAtomId, BTreeSet<ContextAtomId>>,
    /// Exactness is a dependency fixed point over individual predecessor
    /// alternatives.  This intentionally does *not* mean that a similarly
    /// shaped relation happened to be declared somewhere in the callee.
    exact_predecessors: BTreeMap<ContextRelationId, BTreeSet<ContextPredecessorId>>,
    /// Relation-key adjacency retained alongside endpoint adjacency so a
    /// requirement product state expands only the selected atom's outdegree,
    /// never scans every exact relation.
    exact_relation_outgoing: BTreeMap<ContextAtomId, BTreeSet<ContextRelationId>>,
    /// Semantic-order snapshots built once by `finalize`. Requirement
    /// solvers can execute hundreds of thousands of target-scoped queries;
    /// sorting decoded relation/predecessor payloads inside every product
    /// state made that finite analysis needlessly superlinear in practice.
    exact_relation_outgoing_ordered: Vec<Vec<ContextRelationId>>,
    exact_predecessors_ordered: Vec<Vec<ContextPredecessorId>>,
    /// Reverse relation-key adjacency for target-scoped source discovery.
    /// Keeping keys (rather than just atoms) preserves a Call predecessor's
    /// callee dependency when walking backwards through an exact corridor.
    exact_relation_incoming: BTreeMap<ContextAtomId, BTreeSet<ContextRelationId>>,
    exact_outgoing: BTreeMap<ContextAtomId, BTreeSet<ContextAtomId>>,
    exact_incoming: BTreeMap<ContextAtomId, BTreeSet<ContextAtomId>>,
    /// Endpoint-to-scoped-atom index. A target endpoint can occur in more
    /// than one caller scope, so discovery always starts multi-source.
    exact_atoms_by_endpoint: BTreeMap<EndpointNode, BTreeSet<ContextAtomId>>,
    points: BTreeSet<Arc<ProgramPoint>>,
    callsites: BTreeSet<Arc<CallSiteId>>,
    substitutions: BTreeSet<Arc<GenericCallSubstitution>>,
}

impl ContextProofIndex {
    pub(super) fn predecessor_count(&self) -> usize {
        self.predecessors.values().map(BTreeSet::len).sum()
    }

    pub(super) fn exact_predecessor_count(&self) -> usize {
        self.exact_predecessors.values().map(BTreeSet::len).sum()
    }

    pub(super) fn record_typed_witness_transports(
        &mut self,
        callsite: CallSiteId,
        source_identity: String,
        templates: impl IntoIterator<Item = TypedWitnessTransportTemplate>,
    ) {
        let transports = self.typed_witness_by_callsite.entry(callsite).or_default();
        transports.extend(templates.into_iter().map(|template| TypedWitnessTransport {
            source_identity: source_identity.clone(),
            template,
        }));
    }

    pub(super) fn typed_witness_transports_for_callsite(
        &self,
        callsite: &CallSiteId,
    ) -> BTreeSet<TypedWitnessTransport> {
        self.typed_witness_by_callsite
            .get(callsite)
            .cloned()
            .unwrap_or_default()
    }

    pub(super) fn record_local(&mut self, fragment: EndpointEdge) {
        self.record_local_ref(&fragment);
    }

    /// Production aggregation borrows the body graph directly so the local
    /// record pass does not first clone every retained endpoint edge.
    pub(super) fn record_local_ref(&mut self, fragment: &EndpointEdge) {
        if fragment.kind == EndpointEdgeKind::ActualToFormal {
            self.mark_unknown_at(
                fragment.to.clone(),
                ContextUnknownReason::UncontextualizedCallback,
            );
        }
        let Some(owner) = fragment
            .point
            .as_ref()
            .map(|point| point.function.clone())
            .or_else(|| context_endpoint_owner(&fragment.from))
            .or_else(|| context_endpoint_owner(&fragment.to))
        else {
            return;
        };
        let key = ContextRelationKey::new(owner, fragment.from.clone(), fragment.to.clone());
        let point = fragment.point.clone().map(|point| self.intern_point(point));
        self.insert_predecessor(
            key,
            StoredContextPredecessor::Local {
                kind: fragment.kind,
                point,
            },
        );
    }

    fn intern_point(&mut self, point: ProgramPoint) -> Arc<ProgramPoint> {
        if let Some(shared) = self.points.get(&point) {
            return shared.clone();
        }
        let shared = Arc::new(point);
        self.points.insert(shared.clone());
        shared
    }

    fn intern_callsite(&mut self, callsite: CallSiteId) -> Arc<CallSiteId> {
        if let Some(shared) = self.callsites.get(&callsite) {
            return shared.clone();
        }
        let shared = Arc::new(callsite);
        self.callsites.insert(shared.clone());
        shared
    }

    fn intern_substitution(
        &mut self,
        substitution: GenericCallSubstitution,
    ) -> Arc<GenericCallSubstitution> {
        if let Some(shared) = self.substitutions.get(&substitution) {
            return shared.clone();
        }
        let shared = Arc::new(substitution);
        self.substitutions.insert(shared.clone());
        shared
    }

    fn relation_id(&mut self, relation: ContextRelationKey) -> ContextRelationId {
        if let Some(id) = self.arena.relation_ids.get(&relation) {
            return *id;
        }
        let id = ContextRelationId(
            u32::try_from(self.arena.relations.len())
                .expect("context relation arena exceeds u32 IDs"),
        );
        let shared = Arc::new(relation);
        self.arena.relation_ids.insert(shared.clone(), id);
        self.arena.relations.push(shared);
        id
    }

    fn relation(&self, id: ContextRelationId) -> &ContextRelationKey {
        &self.arena.relations[id.0 as usize]
    }

    fn atom_id(&mut self, atom: ContextAtom) -> ContextAtomId {
        if let Some(id) = self.arena.atom_ids.get(&atom) {
            return *id;
        }
        let id = ContextAtomId(
            u32::try_from(self.arena.atoms.len()).expect("context atom arena exceeds u32 IDs"),
        );
        let shared = Arc::new(atom);
        self.arena.atom_ids.insert(shared.clone(), id);
        self.arena.atoms.push(shared);
        id
    }

    fn atom(&self, id: ContextAtomId) -> &ContextAtom {
        &self.arena.atoms[id.0 as usize]
    }

    fn predecessor_id(&mut self, predecessor: StoredContextPredecessor) -> ContextPredecessorId {
        if let Some(id) = self.arena.predecessor_ids.get(&predecessor) {
            return *id;
        }
        let id = ContextPredecessorId(
            u32::try_from(self.arena.predecessors.len())
                .expect("context predecessor arena exceeds u32 IDs"),
        );
        let shared = Arc::new(predecessor);
        self.arena.predecessor_ids.insert(shared.clone(), id);
        self.arena.predecessors.push(shared);
        id
    }

    fn predecessor(&self, id: ContextPredecessorId) -> &StoredContextPredecessor {
        &self.arena.predecessors[id.0 as usize]
    }

    fn materialize_predecessor(
        &self,
        relation: ContextRelationId,
        predecessor: ContextPredecessorId,
    ) -> ContextPredecessor {
        match self.predecessor(predecessor) {
            StoredContextPredecessor::Local { kind, point } => ContextPredecessor::Local {
                fragment: Arc::new(EndpointEdge {
                    from: self.relation(relation).source.clone(),
                    to: self.relation(relation).target.clone(),
                    kind: *kind,
                    point: point.as_deref().cloned(),
                }),
            },
            StoredContextPredecessor::Call {
                callsite,
                substitution,
                callee_relation,
                actual_binding,
            } => ContextPredecessor::Call {
                callsite: callsite.clone(),
                substitution: substitution.clone(),
                callee_relation: callee_relation.clone(),
                actual_binding: *actual_binding,
            },
        }
    }

    fn insert_predecessor(
        &mut self,
        key: ContextRelationKey,
        predecessor: StoredContextPredecessor,
    ) {
        let key = self.relation_id(key);
        let predecessor = self.predecessor_id(predecessor);
        self.predecessors
            .entry(key)
            .or_default()
            .insert(predecessor);
    }

    pub(super) fn record_call(
        &mut self,
        instantiated_edge: EndpointEdge,
        callsite: CallSiteId,
        substitution: GenericCallSubstitution,
        callee_relation: ContextRelationKey,
        actual_binding: ContextActualBinding,
    ) {
        let key = ContextRelationKey::new(
            callsite.caller.clone(),
            instantiated_edge.from.clone(),
            instantiated_edge.to.clone(),
        );
        let callsite = self.intern_callsite(callsite);
        let substitution = self.intern_substitution(substitution);
        let callee_relation = self.relation_id(callee_relation);
        // A requested callee summary can be absent from raw local
        // predecessors. Its scoped endpoints must nevertheless have arena
        // IDs so exact membership can correctly discover an empty corridor
        // (rather than assuming a raw relation exists or panicking).
        let (callee_source, callee_target) = {
            let relation = self.relation(callee_relation);
            (
                ContextAtom::from_source(relation),
                ContextAtom::from_target(relation),
            )
        };
        self.atom_id(callee_source);
        self.atom_id(callee_target);
        let callee_relation = self.arena.relations[callee_relation.0 as usize].clone();
        if substitution.status == GenericSubstitutionStatus::NormalizationIncomplete {
            self.unknown_by_callsite
                .entry((*callsite).clone())
                .or_default()
                .insert(ContextUnknownReason::NormalizationIncomplete {
                    callsite: (*callsite).clone(),
                });
        }
        self.insert_predecessor(
            key,
            StoredContextPredecessor::Call {
                callsite,
                substitution,
                callee_relation,
                actual_binding,
            },
        );
    }

    /// Reconstruct the exact finite caller edge represented by a `Call`
    /// predecessor. Its endpoints are the owning outer relation and its
    /// program point is the interned callsite, so retaining a second copied
    /// `EndpointEdge` in every predecessor cannot add information.
    pub(super) fn instantiated_call_edge(
        relation: &ContextRelationKey,
        callsite: &CallSiteId,
    ) -> EndpointEdge {
        EndpointEdge {
            from: relation.source.clone(),
            to: relation.target.clone(),
            kind: EndpointEdgeKind::CallSummary,
            point: Some(callsite.point.clone()),
        }
    }

    /// Freeze exact adjacency after all local summaries and call hyperedges
    /// have been recorded. A call predecessor becomes exact only when its
    /// referenced callee relation is derivable from exact predecessor
    /// alternatives. The worklist is owner-local: an exact edge wakes only
    /// the summaries for its owner, and a newly exact summary wakes only its
    /// direct instantiated-call dependents. This avoids a graph BFS for every
    /// call predecessor on every fixed-point round.
    pub(super) fn finalize(&mut self) {
        // Raw adjacency is only queried after exact finalization. Building it
        // here releases the propagation phase from its duplicate atom maps.
        self.outgoing.clear();
        self.incoming.clear();
        let raw_relations = self.predecessors.keys().copied().collect::<Vec<_>>();
        for relation in raw_relations {
            let source = ContextAtom::from_source(self.relation(relation));
            let target = ContextAtom::from_target(self.relation(relation));
            let source = self.atom_id(source);
            let target = self.atom_id(target);
            self.outgoing.entry(source).or_default().insert(target);
            self.incoming.entry(target).or_default().insert(source);
        }
        self.exact_predecessors.clear();
        self.exact_relation_outgoing.clear();
        self.exact_relation_outgoing_ordered.clear();
        self.exact_predecessors_ordered.clear();
        self.exact_relation_incoming.clear();
        self.exact_outgoing.clear();
        self.exact_incoming.clear();
        self.exact_atoms_by_endpoint.clear();
        let mut required_by_owner = BTreeMap::<FunctionKey, BTreeSet<ContextRelationId>>::new();
        let mut dependents = BTreeMap::<
            ContextRelationId,
            BTreeSet<(ContextRelationId, ContextPredecessorId)>,
        >::new();
        for (outer, alternatives) in &self.predecessors {
            for alternative in alternatives {
                let StoredContextPredecessor::Call {
                    callee_relation, ..
                } = self.predecessor(*alternative)
                else {
                    continue;
                };
                let callee_relation = *self
                    .arena
                    .relation_ids
                    .get(callee_relation)
                    .expect("call predecessor callee relation is interned");
                required_by_owner
                    .entry(self.relation(callee_relation).context_owner.clone())
                    .or_default()
                    .insert(callee_relation);
                dependents
                    .entry(callee_relation)
                    .or_default()
                    .insert((*outer, *alternative));
            }
        }

        let mut dirty_owners = BTreeSet::new();
        let local_exact = self
            .predecessors
            .iter()
            .flat_map(|(relation, alternatives)| {
                alternatives.iter().filter_map(|alternative| {
                    matches!(self.predecessor(*alternative), StoredContextPredecessor::Local { kind, .. }
                        if *kind != EndpointEdgeKind::ActualToFormal)
                    .then_some((*relation, *alternative))
                })
            })
            .collect::<Vec<_>>();
        for (relation, predecessor) in local_exact {
            if self.insert_exact_predecessor(relation, predecessor) {
                dirty_owners.insert(self.relation(relation).context_owner.clone());
            }
        }

        let mut exact_summaries = BTreeSet::<ContextRelationId>::new();
        while let Some(owner) = take_first(&mut dirty_owners) {
            let Some(required) = required_by_owner.get(&owner) else {
                continue;
            };
            let newly_exact = self
                .exact_summary_membership_ids(required)
                .into_iter()
                .filter(|relation| exact_summaries.insert(*relation))
                .collect::<Vec<_>>();
            for callee_relation in newly_exact {
                for (outer, predecessor) in dependents.get(&callee_relation).into_iter().flatten() {
                    if self.insert_exact_predecessor(*outer, *predecessor) {
                        dirty_owners.insert(self.relation(*outer).context_owner.clone());
                    }
                }
            }
        }

        self.exact_relation_outgoing_ordered = vec![Vec::new(); self.arena.atoms.len()];
        for (atom, relations) in &self.exact_relation_outgoing {
            let mut ordered = relations.iter().copied().collect::<Vec<_>>();
            ordered.sort_by(|left, right| self.relation(*left).cmp(self.relation(*right)));
            self.exact_relation_outgoing_ordered[atom.0 as usize] = ordered;
        }
        self.exact_predecessors_ordered = vec![Vec::new(); self.arena.relations.len()];
        for (relation, predecessors) in &self.exact_predecessors {
            let mut ordered = predecessors.iter().copied().collect::<Vec<_>>();
            ordered.sort_by(|left, right| self.predecessor(*left).cmp(self.predecessor(*right)));
            self.exact_predecessors_ordered[relation.0 as usize] = ordered;
        }
    }

    fn insert_exact_predecessor(
        &mut self,
        relation: ContextRelationId,
        predecessor: ContextPredecessorId,
    ) -> bool {
        if !self
            .exact_predecessors
            .entry(relation)
            .or_default()
            .insert(predecessor)
        {
            return false;
        }
        let source = ContextAtom::from_source(self.relation(relation));
        let target = ContextAtom::from_target(self.relation(relation));
        let source = self.atom_id(source);
        let target = self.atom_id(target);
        self.exact_relation_outgoing
            .entry(source)
            .or_default()
            .insert(relation);
        self.exact_relation_incoming
            .entry(target)
            .or_default()
            .insert(relation);
        self.exact_outgoing
            .entry(source)
            .or_default()
            .insert(target);
        self.exact_incoming
            .entry(target)
            .or_default()
            .insert(source);
        self.exact_atoms_by_endpoint
            .entry(self.atom(source).endpoint.clone())
            .or_default()
            .insert(source);
        self.exact_atoms_by_endpoint
            .entry(self.atom(target).endpoint.clone())
            .or_default()
            .insert(target);
        true
    }

    fn exact_summary_membership_ids(
        &self,
        required: &BTreeSet<ContextRelationId>,
    ) -> BTreeSet<ContextRelationId> {
        let mut reachable_from = BTreeMap::<ContextAtomId, BTreeSet<ContextAtomId>>::new();
        for relation in required {
            let source = self.atom_id_for_relation_source(*relation);
            if !reachable_from.contains_key(&source) {
                reachable_from.insert(source, context_reachable(&source, &self.exact_outgoing));
            }
        }
        required
            .iter()
            .copied()
            .filter(|relation| {
                let source = self.atom_id_for_relation_source(*relation);
                let target = self.atom_id_for_relation_target(*relation);
                reachable_from
                    .get(&source)
                    .is_some_and(|reached| reached.contains(&target))
            })
            .collect()
    }

    fn atom_id_for_relation_source(&self, relation: ContextRelationId) -> ContextAtomId {
        let atom = ContextAtom::from_source(self.relation(relation));
        *self
            .arena
            .atom_ids
            .get(&atom)
            .expect("exact relation source atom was interned")
    }

    fn atom_id_for_relation_target(&self, relation: ContextRelationId) -> ContextAtomId {
        let atom = ContextAtom::from_target(self.relation(relation));
        *self
            .arena
            .atom_ids
            .get(&atom)
            .expect("exact relation target atom was interned")
    }

    pub(super) fn mark_unknown_at(&mut self, node: EndpointNode, reason: ContextUnknownReason) {
        if let Some(callsite) = context_unknown_callsite(&reason) {
            self.unknown_by_callsite
                .entry(callsite.clone())
                .or_default()
                .insert(reason.clone());
        }
        self.unknown_at.entry(node).or_default().insert(reason);
    }

    /// Answer a conjunctive root -> source -> target query.  This is the
    /// operation candidate construction must use: separately proving a root
    /// can reach a source and that the source can reach a sink is not enough
    /// when sibling callsites can otherwise be spliced together.
    pub(super) fn query_via(
        &self,
        root: &EndpointNode,
        source: &EndpointNode,
        target: &EndpointNode,
    ) -> ContextQuery {
        let Some(root_owner) = context_endpoint_owner(root) else {
            return Self::unknown_query(
                root,
                source,
                target,
                BTreeSet::from([ContextUnknownReason::CompletenessUnavailable]),
            );
        };
        let Some(source_owner) = context_endpoint_owner(source) else {
            return Self::unknown_query(
                root,
                source,
                target,
                BTreeSet::from([ContextUnknownReason::CompletenessUnavailable]),
            );
        };
        let Some(target_owner) = context_endpoint_owner(target) else {
            return Self::unknown_query(
                root,
                source,
                target,
                BTreeSet::from([ContextUnknownReason::CompletenessUnavailable]),
            );
        };
        self.query_via_in_context(
            &ContextAtom {
                owner: root_owner,
                endpoint: root.clone(),
            },
            &ContextAtom {
                owner: source_owner,
                endpoint: source.clone(),
            },
            &ContextAtom {
                owner: target_owner,
                endpoint: target.clone(),
            },
        )
    }

    pub(super) fn query_via_in_context(
        &self,
        root: &ContextAtom,
        source: &ContextAtom,
        target: &ContextAtom,
    ) -> ContextQuery {
        let Some(root_id) = self.existing_atom_id(root) else {
            return self.unknown_context_query(root, source, target);
        };
        let Some(source_id) = self.existing_atom_id(source) else {
            return self.unknown_context_query(root, source, target);
        };
        let Some(target_id) = self.existing_atom_id(target) else {
            return self.unknown_context_query(root, source, target);
        };
        let root_forward = context_reachable(&root_id, &self.exact_outgoing);
        let source_to_root = context_reachable(&source_id, &self.exact_incoming);
        let source_forward = context_reachable(&source_id, &self.exact_outgoing);
        let target_backward = context_reachable(&target_id, &self.exact_incoming);
        if root_forward.contains(&source_id) && source_forward.contains(&target_id) {
            let mut relevant = self.relations_through_ids(
                &root_forward,
                &source_to_root,
                &source_forward,
                &target_backward,
            );
            self.expand_callee_relation_closure(&mut relevant);
            // SCC metadata is display-only and intentionally computed only
            // for this already-cropped proof closure. The product-state
            // requirement solver consumes exact relation adjacency directly.
            let recursive_sccs = context_recursive_sccs(&relevant);
            let recursive_scc_grounding =
                context_recursive_scc_grounding(&relevant, &recursive_sccs);
            return ContextQuery::Realizable(ContextProof {
                execution_anchor_owner: root.owner.clone(),
                root_context: root.owner.clone(),
                root: root.endpoint.clone(),
                source_context: source.owner.clone(),
                source: source.endpoint.clone(),
                target_context: target.owner.clone(),
                target: target.endpoint.clone(),
                recursive_sccs,
                recursive_scc_grounding,
                predecessors: relevant,
            });
        }

        let all_root_forward = context_reachable(&root_id, &self.outgoing);
        let all_source_forward = context_reachable(&source_id, &self.outgoing);
        let mut unknown = all_root_forward
            .iter()
            .chain(all_source_forward.iter())
            .flat_map(|node| {
                self.unknown_at
                    .get(&self.atom(*node).endpoint)
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect::<BTreeSet<_>>();
        // This index is an exact positive proof engine, not a closed-world
        // model of Rust execution.  Until a caller supplies an audited,
        // successor-complete corridor, no missing route is eligible for hard
        // suppression.  In particular, opaque callbacks and incomplete
        // generic normalization remain active rather than becoming negatives.
        if unknown.is_empty() {
            unknown.insert(ContextUnknownReason::CompletenessUnavailable);
        }
        Self::unknown_query(&root.endpoint, &source.endpoint, &target.endpoint, unknown)
    }

    fn existing_atom_id(&self, atom: &ContextAtom) -> Option<ContextAtomId> {
        self.arena.atom_ids.get(atom).copied()
    }

    fn unknown_context_query(
        &self,
        root: &ContextAtom,
        source: &ContextAtom,
        target: &ContextAtom,
    ) -> ContextQuery {
        Self::unknown_query(
            &root.endpoint,
            &source.endpoint,
            &target.endpoint,
            BTreeSet::from([ContextUnknownReason::CompletenessUnavailable]),
        )
    }

    /// Context-explicit form for consumers which already know the public-root
    /// caller scope. It avoids reconstructing that scope from an intrinsic
    /// callee endpoint owner.
    pub(super) fn query_via_scoped(
        &self,
        root_context: FunctionKey,
        root: EndpointNode,
        source_context: FunctionKey,
        source: EndpointNode,
        target_context: FunctionKey,
        target: EndpointNode,
    ) -> ContextQuery {
        self.query_via_in_context(
            &ContextAtom {
                owner: root_context,
                endpoint: root,
            },
            &ContextAtom {
                owner: source_context,
                endpoint: source,
            },
            &ContextAtom {
                owner: target_context,
                endpoint: target,
            },
        )
    }

    /// A source-to-target convenience wrapper.  It cannot establish that a
    /// distinct public root shares the same callsite chain; use `query_via`
    /// for candidate suppression or classification.
    pub(super) fn query(&self, source: &EndpointNode, target: &EndpointNode) -> ContextQuery {
        self.query_via(source, source, target)
    }

    /// Enumerate scoped proofs without first coalescing atoms that merely
    /// share an intrinsic endpoint owner.  A source imported into caller A
    /// and the same source imported into caller B therefore produce distinct
    /// alternatives; consumers must choose one complete alternative rather
    /// than splice the prefix of A onto the suffix of B.
    pub(super) fn scoped_alternatives_via(
        &self,
        root: &EndpointNode,
        source: &EndpointNode,
        target: &EndpointNode,
    ) -> Vec<ContextProof> {
        let roots = self.scoped_atoms_for(root);
        let sources = self.scoped_atoms_for(source);
        let targets = self.scoped_atoms_for(target);
        let mut alternatives = Vec::new();
        for root_atom in &roots {
            for source_atom in &sources {
                for target_atom in &targets {
                    let ContextQuery::Realizable(proof) =
                        self.query_via_in_context(root_atom, source_atom, target_atom)
                    else {
                        continue;
                    };
                    if !alternatives.contains(&proof) {
                        alternatives.push(proof);
                    }
                }
            }
        }
        alternatives
    }

    pub(super) fn scoped_alternatives(
        &self,
        source: &EndpointNode,
        target: &EndpointNode,
    ) -> Vec<ContextProof> {
        self.scoped_alternatives_via(source, source, target)
    }

    /// Compact exact atoms for one endpoint.  The returned handles are sorted
    /// by decoded `(scope, endpoint)` value, not by arena insertion order.
    pub(super) fn exact_atom_handles_for_endpoint(
        &self,
        endpoint: &EndpointNode,
    ) -> Vec<ContextAtomHandle> {
        let mut atoms = self
            .exact_atoms_by_endpoint
            .get(endpoint)
            .into_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        atoms.sort_by(|left, right| self.atom(*left).cmp(self.atom(*right)));
        atoms.into_iter().map(ContextAtomHandle).collect()
    }

    /// Return the unique exact atom for one endpoint in one caller scope.
    /// Product solvers require uniqueness rather than an ordered alternative
    /// list, so this avoids allocating and sorting that list for every source
    /// considered against the same unsafe operand.
    pub(super) fn unique_exact_atom_handle(
        &self,
        endpoint: &EndpointNode,
        owner: &FunctionKey,
    ) -> Option<ContextAtomHandle> {
        let mut found = None;
        for atom in self
            .exact_atoms_by_endpoint
            .get(endpoint)
            .into_iter()
            .flatten()
            .copied()
        {
            if &self.atom(atom).owner != owner {
                continue;
            }
            if found.replace(atom).is_some() {
                return None;
            }
        }
        found.map(ContextAtomHandle)
    }

    pub(super) fn exact_atom(&self, handle: ContextAtomHandle) -> &ContextAtom {
        self.atom(handle.0)
    }

    /// Positive-only exact reachability without constructing an owned proof
    /// closure. Consumers which need only an admission/confidence bit should
    /// not pay for relation alternatives, SCC display metadata, or canonical
    /// witness materialization. Returning `false` is deliberately not a
    /// closed-world safety proof.
    pub(super) fn has_exact_route_in_context(
        &self,
        source_owner: &FunctionKey,
        source: &EndpointNode,
        target_owner: &FunctionKey,
        target: &EndpointNode,
    ) -> bool {
        let Some(source) = self.existing_atom_id(&ContextAtom {
            owner: source_owner.clone(),
            endpoint: source.clone(),
        }) else {
            return false;
        };
        let Some(target) = self.existing_atom_id(&ContextAtom {
            owner: target_owner.clone(),
            endpoint: target.clone(),
        }) else {
            return false;
        };
        if source == target {
            return true;
        }
        let mut reached = vec![false; self.arena.atoms.len()];
        reached[source.0 as usize] = true;
        let mut pending = VecDeque::from([source]);
        while let Some(atom) = pending.pop_front() {
            for successor in self.exact_outgoing.get(&atom).into_iter().flatten() {
                if *successor == target {
                    return true;
                }
                if !reached[successor.0 as usize] {
                    reached[successor.0 as usize] = true;
                    pending.push_back(*successor);
                }
            }
        }
        false
    }

    /// Exact outgoing relations for one compact atom.  This is O(outdegree)
    /// and preserves the semantic relation ordering used by the owned facade.
    pub(super) fn exact_relation_handles_from(
        &self,
        atom: ContextAtomHandle,
    ) -> impl ExactSizeIterator<Item = ContextRelationHandle> + '_ {
        let relations = self
            .exact_relation_outgoing_ordered
            .get(atom.0 .0 as usize)
            .map(Vec::as_slice)
            .unwrap_or_default();
        relations.iter().copied().map(ContextRelationHandle)
    }

    pub(super) fn exact_relation(&self, handle: ContextRelationHandle) -> &ContextRelationKey {
        self.relation(handle.0)
    }

    pub(super) fn exact_relation_source(&self, handle: ContextRelationHandle) -> ContextAtomHandle {
        ContextAtomHandle(self.atom_id_for_relation_source(handle.0))
    }

    pub(super) fn exact_relation_target(&self, handle: ContextRelationHandle) -> ContextAtomHandle {
        ContextAtomHandle(self.atom_id_for_relation_target(handle.0))
    }

    /// Every exact predecessor alternative for a relation, in decoded
    /// semantic order.  Keeping alternatives distinct is what prevents two
    /// sibling callsites or substitutions from being spliced together.
    pub(super) fn exact_predecessor_handles(
        &self,
        relation: ContextRelationHandle,
    ) -> impl ExactSizeIterator<Item = ContextPredecessorHandle> + '_ {
        let predecessors = self
            .exact_predecessors_ordered
            .get(relation.0 .0 as usize)
            .map(Vec::as_slice)
            .unwrap_or_default();
        predecessors.iter().copied().map(ContextPredecessorHandle)
    }

    pub(super) fn exact_predecessor(
        &self,
        handle: ContextPredecessorHandle,
    ) -> ExactPredecessorRef<'_> {
        match self.predecessor(handle.0) {
            StoredContextPredecessor::Local { kind, point } => ExactPredecessorRef::Local {
                kind: *kind,
                point: point.as_deref(),
            },
            StoredContextPredecessor::Call {
                callsite,
                substitution,
                callee_relation,
                actual_binding,
            } => {
                let callee_relation = *self
                    .arena
                    .relation_ids
                    .get(callee_relation)
                    .expect("exact call callee relation is interned");
                ExactPredecessorRef::Call {
                    callsite,
                    substitution,
                    callee_relation: ContextRelationHandle(callee_relation),
                    actual_binding: *actual_binding,
                }
            }
        }
    }

    /// Materialize one selected product-proof step only after eligibility has
    /// been established.  The compact index deliberately does not retain a
    /// second owned edge per predecessor alternative.
    pub(super) fn materialize_exact_step(
        &self,
        relation: ContextRelationHandle,
        predecessor: ContextPredecessorHandle,
    ) -> EndpointEdge {
        let relation = self.exact_relation(relation);
        match self.exact_predecessor(predecessor) {
            ExactPredecessorRef::Local { kind, point } => EndpointEdge {
                from: relation.source.clone(),
                to: relation.target.clone(),
                kind,
                point: point.cloned(),
            },
            ExactPredecessorRef::Call { callsite, .. } => {
                Self::instantiated_call_edge(relation, callsite)
            }
        }
    }

    /// Finite relation alternatives for a requirement-specific product-state
    /// fixed point. This is intentionally not the canonical witness API.
    pub(super) fn exact_alternatives(
        &self,
        relation: &ContextRelationKey,
    ) -> Option<ContextPredecessorRefs<'_>> {
        self.arena
            .relation_ids
            .get(relation)
            .copied()
            .filter(|relation| self.exact_predecessors.contains_key(relation))
            .map(|relation| self.exact_predecessor_refs(relation))
    }

    pub(super) fn exact_successors(&self, atom: &ContextAtom) -> BTreeSet<ContextAtom> {
        self.existing_atom_id(atom)
            .into_iter()
            .flat_map(|atom| self.exact_outgoing.get(&atom).into_iter().flatten())
            .map(|atom| self.atom(*atom).clone())
            .collect()
    }

    /// Return every exact scoped atom that can contribute to `target`.
    ///
    /// The multi-source worklist begins at all caller scopes that materialize
    /// the endpoint, then follows exact incoming adjacency.  At an exact Call
    /// predecessor it also enters that predecessor's callee summary target,
    /// so a callee-local structural source is discovered without pretending
    /// that two callsites sharing an endpoint are interchangeable.  Each
    /// scoped atom and relation is visited at most once for this target.
    pub(super) fn exact_reaching_atoms_for_endpoint(
        &self,
        target: &EndpointNode,
    ) -> BTreeSet<ContextAtom> {
        self.exact_reaching_atom_ids_for_endpoint(target)
            .into_iter()
            .map(|atom| self.atom(atom).clone())
            .collect()
    }

    /// Return only the exact GenericCapability atoms relevant to structural
    /// P4 discovery. The reverse walk stays entirely in the compact ID
    /// index; unlike `exact_reaching_atoms_for_endpoint`, it never
    /// materializes unrelated scoped atoms merely to discard them at the
    /// caller.
    pub(super) fn exact_generic_capability_scopes_reaching_endpoint(
        &self,
        target: &EndpointNode,
    ) -> BTreeMap<EndpointNode, BTreeSet<FunctionKey>> {
        let reverse = self.exact_reverse_reachability(target);
        self.exact_generic_capability_scopes_in(&reverse)
    }

    pub(super) fn exact_generic_capability_scopes_in(
        &self,
        reverse: &ExactReverseReachability,
    ) -> BTreeMap<EndpointNode, BTreeSet<FunctionKey>> {
        reverse
            .reached_atoms
            .iter()
            .enumerate()
            .filter_map(|(index, reached)| {
                reached.then(|| {
                    ContextAtomId(
                        u32::try_from(index).expect("context atom arena index fits its u32 ID"),
                    )
                })
            })
            .into_iter()
            .filter_map(|atom| match self.atom(atom) {
                ContextAtom {
                    owner,
                    endpoint: capability @ EndpointNode::GenericCapability { .. },
                } => Some((capability.clone(), owner.clone())),
                _ => None,
            })
            .fold(
                BTreeMap::<EndpointNode, BTreeSet<FunctionKey>>::new(),
                |mut scopes, (endpoint, owner)| {
                    scopes.entry(endpoint).or_default().insert(owner);
                    scopes
                },
            )
    }

    pub(super) fn exact_reverse_reachability(
        &self,
        target: &EndpointNode,
    ) -> ExactReverseReachability {
        // Atom and relation IDs are dense arena indices.  A reverse query used
        // to retain two BTreeSet nodes per visited ID, which multiplied the
        // largest per-role structural temporary.  Bit vectors preserve the
        // same membership fixed point without materializing ordered trees;
        // both public callers restore semantic order in their BTree output.
        let mut reached = vec![false; self.arena.atoms.len()];
        let mut target_by_owner = BTreeMap::new();
        let mut pending = VecDeque::new();
        for atom in self
            .exact_atoms_by_endpoint
            .get(target)
            .into_iter()
            .flatten()
            .copied()
        {
            let owner = self.atom(atom).owner.clone();
            target_by_owner
                .entry(owner)
                .and_modify(|target| *target = None)
                .or_insert(Some(ContextAtomHandle(atom)));
            if !reached[atom.0 as usize] {
                reached[atom.0 as usize] = true;
                pending.push_back(atom);
            }
        }
        let mut visited_relations = vec![false; self.arena.relations.len()];
        while let Some(atom) = pending.pop_front() {
            for relation in self
                .exact_relation_incoming
                .get(&atom)
                .into_iter()
                .flatten()
            {
                if visited_relations[relation.0 as usize] {
                    continue;
                }
                visited_relations[relation.0 as usize] = true;
                let source = self.atom_id_for_relation_source(*relation);
                if !reached[source.0 as usize] {
                    reached[source.0 as usize] = true;
                    pending.push_back(source);
                }
                for predecessor in self.exact_predecessors.get(relation).into_iter().flatten() {
                    let predecessor = self.predecessor(*predecessor);
                    let StoredContextPredecessor::Call {
                        callee_relation, ..
                    } = predecessor
                    else {
                        continue;
                    };
                    let callee_relation = *self
                        .arena
                        .relation_ids
                        .get(callee_relation)
                        .expect("exact call callee relation is interned");
                    let callee_target = self.atom_id_for_relation_target(callee_relation);
                    if !reached[callee_target.0 as usize] {
                        reached[callee_target.0 as usize] = true;
                        pending.push_back(callee_target);
                    }
                }
            }
        }
        ExactReverseReachability {
            reached_atoms: reached,
            target_by_owner,
        }
    }

    fn exact_reaching_atom_ids_for_endpoint(&self, target: &EndpointNode) -> Vec<ContextAtomId> {
        self.exact_reverse_reachability(target)
            .reached_atoms
            .into_iter()
            .enumerate()
            .filter_map(|(index, reached)| {
                reached.then(|| {
                    ContextAtomId(
                        u32::try_from(index).expect("context atom arena index fits its u32 ID"),
                    )
                })
            })
            .collect()
    }

    /// Exact outgoing relation keys, retaining their full predecessor sets.
    /// This prevents a requirement-specific solver from reconstructing a key
    /// from endpoint atoms and accidentally coalescing callsite alternatives.
    pub(super) fn exact_relations_from<'a>(
        &'a self,
        atom: &'a ContextAtom,
    ) -> ContextExactRelations<'a> {
        let mut ids = self
            .existing_atom_id(atom)
            .into_iter()
            .flat_map(|atom| {
                self.exact_relation_outgoing
                    .get(&atom)
                    .into_iter()
                    .flatten()
            })
            .copied()
            .collect::<Vec<_>>();
        ids.sort_by(|left, right| self.relation(*left).cmp(self.relation(*right)));
        ContextExactRelations {
            index: self,
            ids,
            next: 0,
        }
    }

    fn exact_predecessor_refs(&self, relation: ContextRelationId) -> ContextPredecessorRefs<'_> {
        let mut ids = self
            .exact_predecessors
            .get(&relation)
            .into_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        ids.sort_by(|left, right| self.predecessor(*left).cmp(self.predecessor(*right)));
        ContextPredecessorRefs {
            index: self,
            relation,
            ids,
            next: 0,
        }
    }

    pub(super) fn unknown_reasons_for(
        &self,
        node: &EndpointNode,
    ) -> BTreeSet<ContextUnknownReason> {
        self.unknown_at.get(node).cloned().unwrap_or_default()
    }

    /// Requirement consumers must attach normalization uncertainty to the
    /// selected callsite, never to every sibling predecessor of the same
    /// endpoint relation.
    pub(super) fn unknown_reasons_for_callsite(
        &self,
        callsite: &CallSiteId,
    ) -> BTreeSet<ContextUnknownReason> {
        self.unknown_by_callsite
            .get(callsite)
            .cloned()
            .unwrap_or_default()
    }

    fn scoped_atoms_for(&self, endpoint: &EndpointNode) -> BTreeSet<ContextAtom> {
        let mut atoms = self
            .exact_atoms_by_endpoint
            .get(endpoint)
            .into_iter()
            .flatten()
            .map(|atom| self.atom(*atom).clone())
            .collect::<BTreeSet<_>>();
        if atoms.is_empty() {
            if let Some(owner) = context_endpoint_owner(endpoint) {
                atoms.insert(ContextAtom {
                    owner,
                    endpoint: endpoint.clone(),
                });
            }
        }
        atoms
    }

    fn relations_through_ids(
        &self,
        root_forward: &BTreeSet<ContextAtomId>,
        source_backward: &BTreeSet<ContextAtomId>,
        source_forward: &BTreeSet<ContextAtomId>,
        target_backward: &BTreeSet<ContextAtomId>,
    ) -> BTreeMap<ContextRelationKey, BTreeSet<ContextPredecessor>> {
        self.predecessors
            .iter()
            .filter_map(|(key, _alternatives)| {
                let source = self.atom_id_for_relation_source(*key);
                let target = self.atom_id_for_relation_target(*key);
                let prefix = root_forward.contains(&source) && source_backward.contains(&target);
                let suffix = source_forward.contains(&source) && target_backward.contains(&target);
                (prefix || suffix).then(|| {
                    (
                        self.relation(*key).clone(),
                        self.exact_predecessors
                            .get(key)
                            .into_iter()
                            .flatten()
                            .map(|predecessor| self.materialize_predecessor(*key, *predecessor))
                            .collect(),
                    )
                })
            })
            .collect()
    }

    fn expand_callee_relation_closure(
        &self,
        relevant: &mut BTreeMap<ContextRelationKey, BTreeSet<ContextPredecessor>>,
    ) {
        let mut pending = relevant
            .values()
            .flat_map(|alternatives| alternatives.iter())
            .filter_map(|predecessor| match predecessor {
                ContextPredecessor::Call {
                    callee_relation, ..
                } => Some(callee_relation.clone()),
                ContextPredecessor::Local { .. } => None,
            })
            .collect::<VecDeque<_>>();
        let mut expanded = BTreeSet::new();
        while let Some(relation) = pending.pop_front() {
            if !expanded.insert(relation.clone()) {
                continue;
            }
            let relation_id = *self
                .arena
                .relation_ids
                .get(relation.as_ref())
                .expect("proof call relation is interned");
            let forward = context_reachable(
                &self.atom_id_for_relation_source(relation_id),
                &self.exact_outgoing,
            );
            let backward = context_reachable(
                &self.atom_id_for_relation_target(relation_id),
                &self.exact_incoming,
            );
            for (key, _alternatives) in &self.predecessors {
                if !forward.contains(&self.atom_id_for_relation_source(*key))
                    || !backward.contains(&self.atom_id_for_relation_target(*key))
                {
                    continue;
                }
                let exact = self
                    .exact_predecessors
                    .get(key)
                    .into_iter()
                    .flatten()
                    .map(|predecessor| self.materialize_predecessor(*key, *predecessor))
                    .collect::<BTreeSet<_>>();
                if exact.is_empty() {
                    continue;
                }
                let inserted = relevant.entry(self.relation(*key).clone()).or_default();
                if !exact.is_subset(inserted) {
                    inserted.extend(exact.iter().cloned());
                    pending.extend(
                        exact
                            .into_iter()
                            .filter_map(|predecessor| match predecessor {
                                ContextPredecessor::Call {
                                    callee_relation, ..
                                } => Some(callee_relation),
                                ContextPredecessor::Local { .. } => None,
                            }),
                    );
                }
            }
        }
    }

    fn unknown_query(
        root: &EndpointNode,
        source: &EndpointNode,
        target: &EndpointNode,
        reasons: BTreeSet<ContextUnknownReason>,
    ) -> ContextQuery {
        ContextQuery::Unknown {
            reasons,
            completeness: ContextCompletenessEvidence::Unavailable {
                root: root.clone(),
                source: source.clone(),
                target: target.clone(),
            },
        }
    }
}

fn context_endpoint_owner(node: &EndpointNode) -> Option<FunctionKey> {
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
        | EndpointNode::Exposure { owner, .. } => Some(owner.clone()),
        EndpointNode::CallResult { callsite, .. } => Some(callsite.caller.clone()),
        EndpointNode::PublicField { access_owner, .. } => Some(access_owner.clone()),
    }
}

fn context_unknown_callsite(reason: &ContextUnknownReason) -> Option<&CallSiteId> {
    match reason {
        ContextUnknownReason::MissingActualEndpoint { callsite }
        | ContextUnknownReason::MissingResolvedBody { callsite }
        | ContextUnknownReason::NormalizationIncomplete { callsite } => Some(callsite),
        ContextUnknownReason::UncontextualizedCallback
        | ContextUnknownReason::CompletenessUnavailable => None,
    }
}

fn context_reachable<T: Clone + Ord>(
    start: &T,
    adjacency: &BTreeMap<T, BTreeSet<T>>,
) -> BTreeSet<T> {
    let mut reached = BTreeSet::from([start.clone()]);
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(node) = queue.pop_front() {
        for successor in adjacency.get(&node).into_iter().flatten() {
            if reached.insert(successor.clone()) {
                queue.push_back(successor.clone());
            }
        }
    }
    reached
}

fn take_first<T: Clone + Ord>(values: &mut BTreeSet<T>) -> Option<T> {
    let first = values.iter().next()?.clone();
    values.remove(&first);
    Some(first)
}

fn context_recursive_sccs(
    relations: &BTreeMap<ContextRelationKey, BTreeSet<ContextPredecessor>>,
) -> BTreeSet<BTreeSet<ContextRelationKey>> {
    let (forward, reverse, _) = context_summary_request_graph(relations);
    let mut visited = BTreeSet::new();
    let mut finish = Vec::new();
    for root in forward.keys() {
        if !visited.insert(root.clone()) {
            continue;
        }
        let mut stack = vec![(root.clone(), false)];
        while let Some((node, exiting)) = stack.pop() {
            if exiting {
                finish.push(node);
                continue;
            }
            stack.push((node.clone(), true));
            for successor in forward.get(&node).into_iter().flatten().rev() {
                if visited.insert(successor.clone()) {
                    stack.push((successor.clone(), false));
                }
            }
        }
    }
    visited.clear();
    let mut result = BTreeSet::new();
    while let Some(root) = finish.pop() {
        if !visited.insert(root.clone()) {
            continue;
        }
        let mut component = BTreeSet::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            component.insert(node.clone());
            for predecessor in reverse.get(&node).into_iter().flatten().rev() {
                if visited.insert(predecessor.clone()) {
                    stack.push(predecessor.clone());
                }
            }
        }
        let self_cycle = component.first().is_some_and(|node| {
            forward
                .get(node)
                .is_some_and(|successors| successors.contains(node))
        });
        if component.len() <= 1 && !self_cycle {
            continue;
        }
        result.insert(component);
    }
    result
}

/// Records the finite base certificate for each recursive call-dependency
/// component. A `Local` predecessor is immediately grounded; a `Call` that
/// leaves the component is grounded by the already-exact callee relation.
/// Calls that remain inside the component are intentionally not certificates.
fn context_recursive_scc_grounding(
    relations: &BTreeMap<ContextRelationKey, BTreeSet<ContextPredecessor>>,
    components: &BTreeSet<BTreeSet<ContextRelationKey>>,
) -> BTreeMap<BTreeSet<ContextRelationKey>, BTreeSet<ContextRelationKey>> {
    let (_, _, corridors) = context_summary_request_graph(relations);
    let mut result = BTreeMap::new();
    for component in components {
        let grounded =
            component
                .iter()
                .filter(|relation| {
                    let mut allowed = BTreeMap::<ContextAtom, BTreeSet<ContextAtom>>::new();
                    for direct_relation in corridors.get(*relation).into_iter().flatten() {
                        let usable = relations.get(direct_relation).into_iter().flatten().any(
                            |predecessor| match predecessor {
                                ContextPredecessor::Local { .. } => true,
                                ContextPredecessor::Call {
                                    callee_relation, ..
                                } => !component.contains(callee_relation),
                            },
                        );
                        if usable {
                            allowed
                                .entry(ContextAtom::from_source(direct_relation))
                                .or_default()
                                .insert(ContextAtom::from_target(direct_relation));
                        }
                    }
                    context_reachable(&ContextAtom::from_source(relation), &allowed)
                        .contains(&ContextAtom::from_target(relation))
                })
                .cloned()
                .collect::<BTreeSet<_>>();
        result.insert(component.clone(), grounded);
    }
    result
}

/// Builds the dependency graph over summary *requests*, not over individual
/// direct endpoint edges. A request such as `Formal -> Return` commonly has a
/// multi-edge local corridor and is therefore absent from `relations.keys()`;
/// it must still become a graph node when a Call predecessor requests it.
///
/// The third result maps each request to the finite direct-edge corridor used
/// to discover its call dependencies and grounding evidence.
fn context_summary_request_graph(
    relations: &BTreeMap<ContextRelationKey, BTreeSet<ContextPredecessor>>,
) -> (
    BTreeMap<ContextRelationKey, BTreeSet<ContextRelationKey>>,
    BTreeMap<ContextRelationKey, BTreeSet<ContextRelationKey>>,
    BTreeMap<ContextRelationKey, BTreeSet<ContextRelationKey>>,
) {
    let mut endpoint_forward = BTreeMap::<ContextAtom, BTreeSet<ContextAtom>>::new();
    let mut endpoint_reverse = BTreeMap::<ContextAtom, BTreeSet<ContextAtom>>::new();
    // SCC metadata is only for interprocedural summary requests. Do not make
    // every direct endpoint edge a request: that would turn display metadata
    // into an O(E * graph-search) hot path for a large local body.
    let mut requests = BTreeSet::new();
    for (relation, predecessors) in relations {
        let source = ContextAtom::from_source(relation);
        let target = ContextAtom::from_target(relation);
        endpoint_forward
            .entry(source.clone())
            .or_default()
            .insert(target.clone());
        endpoint_reverse.entry(target).or_default().insert(source);
        for predecessor in predecessors {
            if let ContextPredecessor::Call {
                callee_relation, ..
            } = predecessor
            {
                requests.insert((**callee_relation).clone());
            }
        }
    }
    let mut forward = requests
        .iter()
        .cloned()
        .map(|request| (request, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    let mut reverse = forward.clone();
    let mut corridors = BTreeMap::new();
    let mut forward_cache = BTreeMap::<ContextAtom, BTreeSet<ContextAtom>>::new();
    let mut reverse_cache = BTreeMap::<ContextAtom, BTreeSet<ContextAtom>>::new();
    for request in &requests {
        let request_source = ContextAtom::from_source(request);
        let request_target = ContextAtom::from_target(request);
        let reachable_from_source = forward_cache
            .entry(request_source.clone())
            .or_insert_with(|| context_reachable(&request_source, &endpoint_forward));
        let reaching_target = reverse_cache
            .entry(request_target.clone())
            .or_insert_with(|| context_reachable(&request_target, &endpoint_reverse));
        let corridor = relations
            .keys()
            .filter(|direct_relation| {
                reachable_from_source.contains(&ContextAtom::from_source(direct_relation))
                    && reaching_target.contains(&ContextAtom::from_target(direct_relation))
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        for direct_relation in &corridor {
            for predecessor in relations.get(direct_relation).into_iter().flatten() {
                let ContextPredecessor::Call {
                    callee_relation, ..
                } = predecessor
                else {
                    continue;
                };
                forward
                    .entry(request.clone())
                    .or_default()
                    .insert((**callee_relation).clone());
                reverse
                    .entry((**callee_relation).clone())
                    .or_default()
                    .insert(request.clone());
            }
        }
        corridors.insert(request.clone(), corridor);
    }
    (forward, reverse, corridors)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct EndpointFlowGraph {
    pub nodes: BTreeSet<EndpointNode>,
    pub edges: BTreeSet<EndpointEdge>,
}

impl EndpointFlowGraph {
    pub fn insert_node(&mut self, node: EndpointNode) -> bool {
        self.nodes.insert(node)
    }

    pub fn insert_edge(&mut self, edge: EndpointEdge) -> bool {
        self.nodes.insert(edge.from.clone());
        self.nodes.insert(edge.to.clone());
        self.edges.insert(edge)
    }

    pub fn join(&mut self, other: &Self) -> bool {
        let before = (self.nodes.len(), self.edges.len());
        self.nodes.extend(other.nodes.iter().cloned());
        self.edges.extend(other.edges.iter().cloned());
        before != (self.nodes.len(), self.edges.len())
    }

    pub fn append(&mut self, other: Self) {
        self.nodes.extend(other.nodes);
        self.edges.extend(other.edges);
    }

    pub fn canonical_route(
        &self,
        source: &EndpointNode,
        target: &EndpointNode,
    ) -> Option<Vec<EndpointEdge>> {
        self.route_index().routes_to(target).route_from(source)
    }

    pub fn route_index(&self) -> EndpointRouteIndex<'_> {
        self.route_index_excluding(&BTreeSet::new())
    }

    /// Build the same deterministic route index while treating selected edge
    /// kinds as absent.  Endpoint summaries use this to ensure an old
    /// caller-to-callee bridge cannot participate while deriving a callee
    /// local relation.
    pub fn route_index_excluding(
        &self,
        excluded: &BTreeSet<EndpointEdgeKind>,
    ) -> EndpointRouteIndex<'_> {
        // BTreeSet iteration defines the public route order. Preserve that
        // order once in these tables, then keep hot adjacency as compact IDs.
        let nodes = self.nodes.iter().collect::<Vec<_>>();
        let mut edges = Vec::new();
        let mut outgoing_counts = vec![0_usize; nodes.len()];
        let mut incoming_counts = vec![0_usize; nodes.len()];
        for edge in &self.edges {
            if excluded.contains(&edge.kind) {
                continue;
            }
            let from = node_id_in_sorted_nodes(&nodes, &edge.from)
                .expect("endpoint edge source must be present in graph nodes");
            let to = node_id_in_sorted_nodes(&nodes, &edge.to)
                .expect("endpoint edge target must be present in graph nodes");
            let _ = EdgeId::from_index(edges.len());
            edges.push(IndexedEndpointEdge { edge, from, to });
            outgoing_counts[from.index()] = outgoing_counts[from.index()]
                .checked_add(1)
                .expect("endpoint route index outgoing row length overflows usize");
            incoming_counts[to.index()] = incoming_counts[to.index()]
                .checked_add(1)
                .expect("endpoint route index incoming row length overflows usize");
        }

        let mut outgoing = EndpointEdgeRows::from_counts(outgoing_counts, edges.len());
        let mut incoming = EndpointEdgeRows::from_counts(incoming_counts, edges.len());
        let mut outgoing_cursors = outgoing.row_starts();
        let mut incoming_cursors = incoming.row_starts();
        for (index, edge) in edges.iter().enumerate() {
            let id = EdgeId::from_index(index);
            outgoing.push_in_row(&mut outgoing_cursors, edge.from, id);
            incoming.push_in_row(&mut incoming_cursors, edge.to, id);
        }
        outgoing.assert_filled(&outgoing_cursors);
        incoming.assert_filled(&incoming_cursors);

        EndpointRouteIndex {
            nodes,
            edges,
            outgoing,
            incoming,
        }
    }

    pub fn cyclic_components(&self) -> Vec<BTreeSet<EndpointNode>> {
        let mut outgoing = BTreeMap::<EndpointNode, BTreeSet<EndpointNode>>::new();
        let mut incoming = BTreeMap::<EndpointNode, BTreeSet<EndpointNode>>::new();
        for node in &self.nodes {
            outgoing.entry(node.clone()).or_default();
            incoming.entry(node.clone()).or_default();
        }
        for edge in &self.edges {
            outgoing
                .entry(edge.from.clone())
                .or_default()
                .insert(edge.to.clone());
            incoming
                .entry(edge.to.clone())
                .or_default()
                .insert(edge.from.clone());
        }

        let mut visited = BTreeSet::new();
        let mut finish_order = Vec::new();
        for root in &self.nodes {
            if !visited.insert(root.clone()) {
                continue;
            }
            let mut stack = vec![(root.clone(), false)];
            while let Some((node, exiting)) = stack.pop() {
                if exiting {
                    finish_order.push(node);
                    continue;
                }
                stack.push((node.clone(), true));
                if let Some(next) = outgoing.get(&node) {
                    for successor in next.iter().rev() {
                        if visited.insert(successor.clone()) {
                            stack.push((successor.clone(), false));
                        }
                    }
                }
            }
        }

        let self_edges = self
            .edges
            .iter()
            .filter(|edge| edge.from == edge.to)
            .map(|edge| edge.from.clone())
            .collect::<BTreeSet<_>>();
        visited.clear();
        let mut components = Vec::new();
        while let Some(root) = finish_order.pop() {
            if !visited.insert(root.clone()) {
                continue;
            }
            let mut component = BTreeSet::new();
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                component.insert(node.clone());
                if let Some(previous) = incoming.get(&node) {
                    for predecessor in previous.iter().rev() {
                        if visited.insert(predecessor.clone()) {
                            stack.push(predecessor.clone());
                        }
                    }
                }
            }
            if component.len() > 1
                || component
                    .first()
                    .is_some_and(|node| self_edges.contains(node))
            {
                components.push(component);
            }
        }
        components.sort();
        components
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeId(u32);

impl NodeId {
    fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("endpoint route index node count exceeds u32"))
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeId(u32);

impl EdgeId {
    fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("endpoint route index edge count exceeds u32"))
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

struct IndexedEndpointEdge<'graph> {
    edge: &'graph EndpointEdge,
    from: NodeId,
    to: NodeId,
}

/// Compact sparse rows of edge IDs.  `offsets[node]..offsets[node + 1]` is
/// one adjacency row; values in each row retain global EdgeId order.
struct EndpointEdgeRows {
    offsets: Vec<usize>,
    edge_ids: Vec<EdgeId>,
}

impl EndpointEdgeRows {
    fn from_counts(counts: Vec<usize>, expected_edge_count: usize) -> Self {
        let mut offsets = Vec::<usize>::with_capacity(
            counts
                .len()
                .checked_add(1)
                .expect("endpoint route index node count overflows usize"),
        );
        offsets.push(0);
        for count in counts {
            let next = offsets
                .last()
                .copied()
                .expect("endpoint route offsets always start at zero")
                .checked_add(count)
                .expect("endpoint route index edge count overflows usize");
            offsets.push(next);
        }
        let edge_count = offsets
            .last()
            .copied()
            .expect("endpoint route offsets always contain a sentinel");
        assert_eq!(
            edge_count, expected_edge_count,
            "endpoint route row counts must cover every included edge"
        );
        Self {
            offsets,
            edge_ids: vec![EdgeId(0); edge_count],
        }
    }

    fn row(&self, node: NodeId) -> &[EdgeId] {
        let start = self.offsets[node.index()];
        let end = self.offsets[node.index() + 1];
        &self.edge_ids[start..end]
    }

    fn row_starts(&self) -> Vec<usize> {
        self.offsets[..self.offsets.len() - 1].to_vec()
    }

    fn push_in_row(&mut self, cursors: &mut [usize], node: NodeId, edge: EdgeId) {
        let node = node.index();
        let cursor = &mut cursors[node];
        assert!(
            *cursor < self.offsets[node + 1],
            "endpoint route row fill must not exceed its CSR offset"
        );
        self.edge_ids[*cursor] = edge;
        *cursor = cursor
            .checked_add(1)
            .expect("endpoint route index edge count overflows usize");
    }

    fn assert_filled(&self, cursors: &[usize]) {
        assert_eq!(
            cursors,
            &self.offsets[1..],
            "endpoint route row fill must end at every CSR offset"
        );
    }
}

fn node_id_in_sorted_nodes(nodes: &[&EndpointNode], node: &EndpointNode) -> Option<NodeId> {
    nodes
        .binary_search_by(|candidate| (*candidate).cmp(node))
        .ok()
        .map(NodeId::from_index)
}

pub(super) struct EndpointRouteIndex<'graph> {
    // IDs are assigned from BTreeSet semantic order, never hash or insertion
    // order. Adjacency preserves the original BTreeSet edge order per row.
    nodes: Vec<&'graph EndpointNode>,
    edges: Vec<IndexedEndpointEdge<'graph>>,
    outgoing: EndpointEdgeRows,
    incoming: EndpointEdgeRows,
}

const UNREACHABLE_ENDPOINT_DISTANCE: u32 = u32::MAX;

impl<'graph> EndpointRouteIndex<'graph> {
    fn node_id(&self, node: &EndpointNode) -> Option<NodeId> {
        node_id_in_sorted_nodes(&self.nodes, node)
    }

    fn node(&self, id: NodeId) -> &'graph EndpointNode {
        self.nodes[id.index()]
    }

    fn edge(&self, id: EdgeId) -> &IndexedEndpointEdge<'graph> {
        &self.edges[id.index()]
    }

    pub fn routes_to(&self, target: &EndpointNode) -> EndpointRoutesTo<'_, 'graph> {
        let target = self.node_id(target);
        let mut distance_to_target = vec![UNREACHABLE_ENDPOINT_DISTANCE; self.nodes.len()];
        if let Some(target) = target {
            distance_to_target[target.index()] = 0;
            let mut pending = VecDeque::from([target]);
            while let Some(node) = pending.pop_front() {
                let distance = distance_to_target[node.index()];
                assert_ne!(
                    distance, UNREACHABLE_ENDPOINT_DISTANCE,
                    "queued endpoint route node must have a distance"
                );
                let predecessor_distance = distance
                    .checked_add(1)
                    .filter(|distance| *distance != UNREACHABLE_ENDPOINT_DISTANCE)
                    .expect("endpoint route distance exceeds the compact u32 domain");
                for edge in self.incoming.row(node) {
                    let predecessor = self.edge(*edge).from;
                    if distance_to_target[predecessor.index()] == UNREACHABLE_ENDPOINT_DISTANCE {
                        distance_to_target[predecessor.index()] = predecessor_distance;
                        pending.push_back(predecessor);
                    }
                }
            }
        }
        EndpointRoutesTo {
            index: self,
            target,
            distance_to_target,
        }
    }

    pub fn routes_from(&self, source: &EndpointNode) -> EndpointRoutesFrom<'_, 'graph> {
        let source = self.node_id(source);
        let mut predecessors = vec![None; self.nodes.len()];
        if let Some(source) = source {
            let mut visited = vec![false; self.nodes.len()];
            visited[source.index()] = true;
            let mut queue = VecDeque::from([source]);
            while let Some(node) = queue.pop_front() {
                for edge_id in self.outgoing.row(node) {
                    let edge = self.edge(*edge_id);
                    if visited[edge.to.index()] {
                        continue;
                    }
                    visited[edge.to.index()] = true;
                    predecessors[edge.to.index()] = Some(*edge_id);
                    queue.push_back(edge.to);
                }
            }
        }
        EndpointRoutesFrom {
            index: self,
            source,
            predecessors,
        }
    }

    #[cfg(test)]
    pub fn any_route_avoiding<'node>(
        &self,
        starts: impl IntoIterator<Item = &'node EndpointNode>,
        target: &EndpointNode,
        avoided: &EndpointNode,
    ) -> bool {
        if target == avoided {
            return false;
        }
        let Some(target) = self.node_id(target) else {
            return false;
        };
        let avoided = self.node_id(avoided);
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();
        for start in starts {
            let Some(start) = self.node_id(start) else {
                continue;
            };
            if Some(start) == avoided || visited[start.index()] {
                continue;
            }
            visited[start.index()] = true;
            queue.push_back(start);
        }
        while let Some(node) = queue.pop_front() {
            for edge in self.outgoing.row(node) {
                let edge = self.edge(*edge);
                if Some(edge.to) == avoided || visited[edge.to.index()] {
                    continue;
                }
                if edge.to == target {
                    return true;
                }
                visited[edge.to.index()] = true;
                queue.push_back(edge.to);
            }
        }
        false
    }

    pub fn condensation(&self) -> EndpointCondensation<'_, 'graph> {
        let mut visited = vec![false; self.nodes.len()];
        let mut finish_order = Vec::with_capacity(self.nodes.len());
        for root in (0..self.nodes.len()).map(NodeId::from_index) {
            if visited[root.index()] {
                continue;
            }
            visited[root.index()] = true;
            let mut stack = vec![(root, false)];
            while let Some((node, exiting)) = stack.pop() {
                if exiting {
                    finish_order.push(node);
                    continue;
                }
                stack.push((node, true));
                for edge in self.outgoing.row(node).iter().rev() {
                    let successor = self.edge(*edge).to;
                    if !visited[successor.index()] {
                        visited[successor.index()] = true;
                        stack.push((successor, false));
                    }
                }
            }
        }

        visited.fill(false);
        let mut component_by_node = vec![usize::MAX; self.nodes.len()];
        let mut component_count = 0_usize;
        while let Some(root) = finish_order.pop() {
            if visited[root.index()] {
                continue;
            }
            visited[root.index()] = true;
            let component = component_count;
            component_count += 1;
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                component_by_node[node.index()] = component;
                for edge in self.incoming.row(node).iter().rev() {
                    let predecessor = self.edge(*edge).from;
                    if !visited[predecessor.index()] {
                        visited[predecessor.index()] = true;
                        stack.push(predecessor);
                    }
                }
            }
        }

        let mut successors = vec![Vec::new(); component_count];
        for node in (0..self.nodes.len()).map(NodeId::from_index) {
            for edge in self.outgoing.row(node) {
                let edge = self.edge(*edge);
                let from = component_by_node[edge.from.index()];
                let to = component_by_node[edge.to.index()];
                if from != to {
                    successors[from].push(to);
                }
            }
        }
        for successors in &mut successors {
            successors.sort_unstable();
            successors.dedup();
        }

        EndpointCondensation {
            index: self,
            component_by_node,
            successors,
        }
    }
}

pub(super) struct EndpointCondensation<'index, 'graph> {
    index: &'index EndpointRouteIndex<'graph>,
    component_by_node: Vec<usize>,
    successors: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedOrigin {
    None,
    One(usize),
    Many,
}

impl SelectedOrigin {
    fn join(&mut self, incoming: Self) -> bool {
        let joined = match (*self, incoming) {
            (Self::Many, _) | (_, Self::Many) => Self::Many,
            (Self::None, other) => other,
            (current, Self::None) => current,
            (Self::One(left), Self::One(right)) if left == right => Self::One(left),
            (Self::One(_), Self::One(_)) => Self::Many,
        };
        if joined == *self {
            false
        } else {
            *self = joined;
            true
        }
    }
}

impl EndpointCondensation<'_, '_> {
    fn component_for(&self, node: &EndpointNode) -> Option<usize> {
        self.index
            .node_id(node)
            .map(|node| self.component_by_node[node.index()])
    }

    pub fn nodes_reached_from_another_in_set(
        &self,
        sources: &BTreeSet<EndpointNode>,
    ) -> BTreeSet<EndpointNode> {
        if sources.len() < 2 {
            return BTreeSet::new();
        }

        let mut sources_by_component = BTreeMap::<usize, Vec<&EndpointNode>>::new();
        for source in sources {
            let component = self
                .component_for(source)
                .expect("selected endpoint source must exist in the endpoint graph");
            sources_by_component
                .entry(component)
                .or_default()
                .push(source);
        }

        let mut origins = vec![SelectedOrigin::None; self.successors.len()];
        let mut queue = VecDeque::new();
        for component in sources_by_component.keys().copied() {
            origins[component] = SelectedOrigin::One(component);
            queue.push_back(component);
        }
        while let Some(component) = queue.pop_front() {
            let incoming = origins[component];
            for successor in &self.successors[component] {
                if origins[*successor].join(incoming) {
                    queue.push_back(*successor);
                }
            }
        }

        let mut downstream = BTreeSet::new();
        for (component, selected) in sources_by_component {
            if selected.len() > 1 || origins[component] == SelectedOrigin::Many {
                downstream.extend(selected.into_iter().cloned());
            }
        }
        downstream
    }
}

pub(super) struct EndpointRoutesTo<'index, 'graph> {
    index: &'index EndpointRouteIndex<'graph>,
    target: Option<NodeId>,
    distance_to_target: Vec<u32>,
}

impl EndpointRoutesTo<'_, '_> {
    pub fn reaching_nodes(&self) -> impl Iterator<Item = &EndpointNode> {
        self.index
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                (self.distance_to_target[index] != UNREACHABLE_ENDPOINT_DISTANCE).then_some(*node)
            })
    }

    pub fn route_from(&self, source: &EndpointNode) -> Option<Vec<EndpointEdge>> {
        let mut cursor = self.index.node_id(source)?;
        let distance = self.distance_to_target[cursor.index()];
        if distance == UNREACHABLE_ENDPOINT_DISTANCE {
            return None;
        }
        let mut route = Vec::with_capacity(distance as usize);
        for remaining in (1..=distance).rev() {
            let edge = self
                .index
                .outgoing
                .row(cursor)
                .iter()
                .copied()
                .find(|edge| {
                    self.distance_to_target[self.index.edge(*edge).to.index()] == remaining - 1
                })?;
            let edge = self.index.edge(edge);
            route.push(edge.edge.clone());
            cursor = edge.to;
        }
        Some(route)
    }

    pub fn unavoidable_from<'node>(
        &self,
        starts: impl IntoIterator<Item = &'node EndpointNode>,
    ) -> BTreeSet<EndpointNode> {
        let starts = starts
            .into_iter()
            .filter_map(|start| self.index.node_id(start))
            .filter(|start| self.distance_to_target[start.index()] != UNREACHABLE_ENDPOINT_DISTANCE)
            .collect::<BTreeSet<_>>();
        if starts.is_empty() {
            return BTreeSet::new();
        }
        let target = self
            .target
            .expect("a nonempty reverse-reachable set must have a graph-owned target");

        let mut graph = DiGraph::<(), ()>::new();
        let root = graph.add_node(());
        let mut node_ids = vec![None; self.index.nodes.len()];
        let mut nodes_by_id = vec![None];
        for node in (0..self.index.nodes.len()).map(NodeId::from_index) {
            if self.distance_to_target[node.index()] == UNREACHABLE_ENDPOINT_DISTANCE {
                continue;
            }
            let id = graph.add_node(());
            node_ids[node.index()] = Some(id);
            nodes_by_id.push(Some(node));
        }
        for node in (0..self.index.nodes.len()).map(NodeId::from_index) {
            let Some(from) = node_ids[node.index()] else {
                continue;
            };
            for edge in self.index.outgoing.row(node) {
                let to = self.index.edge(*edge).to;
                if let Some(to) = node_ids[to.index()] {
                    graph.add_edge(from, to, ());
                }
            }
        }
        for start in starts {
            graph.add_edge(
                root,
                node_ids[start.index()].expect("reachable start must have a graph node"),
                (),
            );
        }

        let target = node_ids[target.index()].expect("reachable target must have a graph node");
        simple_fast(&graph, root)
            .dominators(target)
            .into_iter()
            .flatten()
            .filter_map(|id| nodes_by_id[id.index()])
            .map(|id| self.index.node(id).clone())
            .collect()
    }
}

pub(super) struct EndpointRoutesFrom<'index, 'graph> {
    index: &'index EndpointRouteIndex<'graph>,
    source: Option<NodeId>,
    predecessors: Vec<Option<EdgeId>>,
}

impl EndpointRoutesFrom<'_, '_> {
    pub fn reachable_nodes(&self) -> impl Iterator<Item = &EndpointNode> {
        self.index
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| self.predecessors[index].is_some().then_some(*node))
    }

    pub fn route_to(&self, target: &EndpointNode) -> Option<Vec<EndpointEdge>> {
        let mut cursor = self.index.node_id(target)?;
        if Some(cursor) == self.source {
            return Some(Vec::new());
        }
        let mut route = Vec::new();
        while Some(cursor) != self.source {
            let edge = self.predecessors[cursor.index()]?;
            let edge = self.index.edge(edge);
            route.push(edge.edge.clone());
            cursor = edge.from;
        }
        route.reverse();
        Some(route)
    }
}

#[cfg(test)]
fn graph_distances(
    start: &EndpointNode,
    adjacency: &BTreeMap<&EndpointNode, Vec<&EndpointEdge>>,
    reverse: bool,
) -> BTreeMap<EndpointNode, usize> {
    let mut distances = BTreeMap::from([(start.clone(), 0)]);
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(node) = queue.pop_front() {
        let distance = distances[&node];
        for edge in adjacency.get(&node).into_iter().flatten() {
            let neighbor = if reverse { &edge.from } else { &edge.to };
            if distances.contains_key(neighbor) {
                continue;
            }
            distances.insert(neighbor.clone(), distance + 1);
            queue.push_back(neighbor.clone());
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::unsoundaudit::summary::{StablePosition, StableSpan};

    fn point(owner: &FunctionKey, block: u32) -> ProgramPoint {
        ProgramPoint {
            function: owner.clone(),
            block,
            statement: 0,
            span: StableSpan::new(
                "src/lib.rs",
                StablePosition::new(block + 1, 1),
                StablePosition::new(block + 1, 2),
            ),
        }
    }

    fn formal(owner: &FunctionKey, index: u32) -> EndpointNode {
        EndpointNode::Formal {
            owner: owner.clone(),
            index,
        }
    }

    fn edge(from: EndpointNode, to: EndpointNode, block: u32) -> EndpointEdge {
        let owner = match &from {
            EndpointNode::Formal { owner, .. }
            | EndpointNode::Capture { owner, .. }
            | EndpointNode::Return { owner }
            | EndpointNode::Out { owner, .. }
            | EndpointNode::UnsafeOperand { owner, .. }
            | EndpointNode::UnsafeResult { owner, .. }
            | EndpointNode::InternalDerivation { owner, .. }
            | EndpointNode::GenericCapability { owner, .. }
            | EndpointNode::ExternalBehavior { owner, .. }
            | EndpointNode::Exposure { owner, .. } => owner.clone(),
            EndpointNode::CallResult { callsite, .. } => callsite.caller.clone(),
            EndpointNode::PublicField { access_owner, .. } => access_owner.clone(),
        };
        EndpointEdge {
            from,
            to,
            kind: EndpointEdgeKind::ActualToFormal,
            point: Some(point(&owner, block)),
        }
    }

    #[test]
    fn canonical_route_crosses_sixty_four_hops_without_a_cap() {
        let owner = FunctionKey::new("crate::chain");
        let nodes = (0..=64)
            .map(|index| formal(&owner, index + 1))
            .collect::<Vec<_>>();
        let mut graph = EndpointFlowGraph::default();
        for index in 0..64 {
            graph.insert_edge(edge(
                nodes[index].clone(),
                nodes[index + 1].clone(),
                index as u32,
            ));
        }
        let route = graph.canonical_route(&nodes[0], &nodes[64]).unwrap();
        assert_eq!(route.len(), 64);
        assert_eq!(route.first().unwrap().from, nodes[0]);
        assert_eq!(route.last().unwrap().to, nodes[64]);
    }

    #[test]
    fn distinct_callsites_cannot_share_a_return_port() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let input_a = formal(&caller, 1);
        let input_b = formal(&caller, 2);
        let call_a = EndpointNode::CallResult {
            callsite: CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone()),
            output: CallSiteOutput::Return,
        };
        let call_b = EndpointNode::CallResult {
            callsite: CallSiteId::new(caller.clone(), point(&caller, 20), callee),
            output: CallSiteOutput::Return,
        };
        let sink = EndpointNode::UnsafeOperand {
            owner: caller.clone(),
            site_id: "site-b".to_string(),
            role: OperandRole::Argument(0),
        };
        let mut graph = EndpointFlowGraph::default();
        graph.insert_edge(EndpointEdge {
            from: input_a.clone(),
            to: call_a.clone(),
            kind: EndpointEdgeKind::CallSummary,
            point: Some(point(&caller, 10)),
        });
        graph.insert_edge(EndpointEdge {
            from: input_b,
            to: call_b.clone(),
            kind: EndpointEdgeKind::CallSummary,
            point: Some(point(&caller, 20)),
        });
        graph.insert_edge(EndpointEdge {
            from: call_b,
            to: sink.clone(),
            kind: EndpointEdgeKind::UnsafeOperand,
            point: Some(point(&caller, 21)),
        });

        assert!(graph.canonical_route(&input_a, &call_a).is_some());
        assert!(graph.canonical_route(&input_a, &sink).is_none());
    }

    #[test]
    fn reversed_edge_insertion_keeps_the_same_lexical_shortest_route() {
        let owner = FunctionKey::new("crate::diamond");
        let source = formal(&owner, 1);
        let left = formal(&owner, 2);
        let right = formal(&owner, 3);
        let target = formal(&owner, 4);
        let mut edges = vec![
            edge(source.clone(), right, 3),
            edge(source.clone(), left.clone(), 1),
            edge(left, target.clone(), 2),
            edge(formal(&owner, 3), target.clone(), 4),
        ];
        let mut first = EndpointFlowGraph::default();
        for edge in &edges {
            first.insert_edge(edge.clone());
        }
        edges.reverse();
        let mut second = EndpointFlowGraph::default();
        for edge in edges {
            second.insert_edge(edge);
        }
        assert_eq!(
            first.canonical_route(&source, &target),
            second.canonical_route(&source, &target)
        );
        assert_eq!(
            first.canonical_route(&source, &target).unwrap()[0].point,
            Some(point(&owner, 1))
        );

        let first_index = first.route_index();
        let second_index = second.route_index();
        assert_eq!(
            first_index.routes_to(&target).route_from(&source),
            second_index.routes_to(&target).route_from(&source)
        );
    }

    #[test]
    fn absent_target_has_no_routes_or_reaching_nodes() {
        let owner = FunctionKey::new("crate::absent");
        let source = formal(&owner, 1);
        let present = formal(&owner, 2);
        let absent = formal(&owner, 3);
        let mut graph = EndpointFlowGraph::default();
        graph.insert_edge(edge(source.clone(), present, 1));

        let index = graph.route_index();
        let routes = index.routes_to(&absent);
        assert_eq!(
            std::mem::size_of_val(routes.distance_to_target.as_slice()),
            graph.nodes.len() * std::mem::size_of::<u32>(),
        );
        assert_eq!(routes.reaching_nodes().count(), 0);
        assert_eq!(routes.route_from(&source), None);
        assert_eq!(routes.route_from(&absent), None);
        assert!(routes.unavoidable_from([&source, &absent]).is_empty());
    }

    #[test]
    fn unavoidable_nodes_remain_stable_across_a_loop_and_bypass() {
        let owner = FunctionKey::new("crate::unavoidable_loop");
        let nodes = (0..6)
            .map(|index| formal(&owner, index))
            .collect::<Vec<_>>();
        let mut graph = EndpointFlowGraph::default();
        for (from, to, block) in [
            (0, 1, 1),
            (1, 2, 2),
            (2, 1, 3),
            (2, 4, 4),
            (0, 3, 5),
            (3, 4, 6),
            (4, 5, 7),
        ] {
            graph.insert_edge(edge(nodes[from].clone(), nodes[to].clone(), block));
        }

        let index = graph.route_index();
        assert_eq!(
            index.routes_to(&nodes[5]).unavoidable_from([&nodes[0]]),
            BTreeSet::from([nodes[0].clone(), nodes[4].clone(), nodes[5].clone()])
        );
    }

    #[test]
    fn target_route_index_matches_the_reference_for_all_sources() {
        fn reference_route(
            graph: &EndpointFlowGraph,
            source: &EndpointNode,
            target: &EndpointNode,
        ) -> Option<Vec<EndpointEdge>> {
            if source == target {
                return graph.nodes.contains(source).then(Vec::new);
            }
            let mut outgoing = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
            let mut incoming = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
            for edge in &graph.edges {
                outgoing.entry(&edge.from).or_default().push(edge);
                incoming.entry(&edge.to).or_default().push(edge);
            }
            let from_source = graph_distances(source, &outgoing, false);
            let to_target = graph_distances(target, &incoming, true);
            let distance = *from_source.get(target)?;
            let mut route = Vec::with_capacity(distance);
            let mut cursor = source;
            for step in 0..distance {
                let edge = outgoing.get(cursor)?.iter().copied().find(|edge| {
                    from_source.get(&edge.to) == Some(&(step + 1))
                        && to_target.get(&edge.to) == Some(&(distance - step - 1))
                })?;
                route.push(edge.clone());
                cursor = &edge.to;
            }
            Some(route)
        }

        let owner = FunctionKey::new("crate::indexed");
        let nodes = (0..7)
            .map(|index| formal(&owner, index))
            .collect::<Vec<_>>();
        let mut graph = EndpointFlowGraph::default();
        for (from, to, block) in [
            (0, 1, 1),
            (0, 2, 2),
            (1, 3, 3),
            (2, 3, 4),
            (3, 1, 5),
            (3, 4, 6),
            (2, 5, 7),
            (5, 4, 8),
        ] {
            graph.insert_edge(edge(nodes[from].clone(), nodes[to].clone(), block));
        }
        graph.insert_node(nodes[6].clone());

        let index = graph.route_index();
        let routes_to_target = index.routes_to(&nodes[4]);
        let mut incoming = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
        for edge in &graph.edges {
            incoming.entry(&edge.to).or_default().push(edge);
        }
        assert_eq!(
            routes_to_target
                .reaching_nodes()
                .cloned()
                .collect::<Vec<_>>(),
            graph_distances(&nodes[4], &incoming, true)
                .into_keys()
                .collect::<Vec<_>>()
        );
        for source in &nodes {
            assert_eq!(
                routes_to_target.route_from(source),
                reference_route(&graph, source, &nodes[4]),
                "source {source:?}"
            );
        }

        let routes_from_source = index.routes_from(&nodes[0]);
        for target in &nodes {
            assert_eq!(
                routes_from_source.route_to(target),
                reference_route(&graph, &nodes[0], target),
                "target {target:?}"
            );
        }

        fn reference_exists_avoiding(
            graph: &EndpointFlowGraph,
            start: &EndpointNode,
            target: &EndpointNode,
            avoided: &EndpointNode,
        ) -> bool {
            if start == avoided || target == avoided {
                return false;
            }
            let mut visited = BTreeSet::from([start.clone()]);
            let mut queue = VecDeque::from([start.clone()]);
            while let Some(node) = queue.pop_front() {
                for edge in graph.edges.iter().filter(|edge| edge.from == node) {
                    if &edge.to == avoided || !visited.insert(edge.to.clone()) {
                        continue;
                    }
                    if &edge.to == target {
                        return true;
                    }
                    queue.push_back(edge.to.clone());
                }
            }
            false
        }

        let starts = [&nodes[0], &nodes[2]];
        let unavoidable = routes_to_target.unavoidable_from(starts);
        for avoided in &nodes {
            let reference_avoidable = starts
                .iter()
                .any(|start| reference_exists_avoiding(&graph, start, &nodes[4], avoided));
            assert_eq!(
                index.any_route_avoiding(starts, &nodes[4], avoided),
                reference_avoidable,
                "avoided {avoided:?}"
            );
            assert_eq!(
                unavoidable.contains(avoided),
                !reference_avoidable,
                "dominator {avoided:?}"
            );
        }
    }

    #[test]
    fn compact_route_index_matches_reference_for_parallel_ties_cycles_and_p5_routes() {
        fn reference_route(
            graph: &EndpointFlowGraph,
            source: &EndpointNode,
            target: &EndpointNode,
        ) -> Option<Vec<EndpointEdge>> {
            if source == target {
                return graph.nodes.contains(source).then(Vec::new);
            }
            let mut outgoing = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
            let mut incoming = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
            for edge in &graph.edges {
                outgoing.entry(&edge.from).or_default().push(edge);
                incoming.entry(&edge.to).or_default().push(edge);
            }
            let from_source = graph_distances(source, &outgoing, false);
            let to_target = graph_distances(target, &incoming, true);
            let distance = *from_source.get(target)?;
            let mut route = Vec::with_capacity(distance);
            let mut cursor = source;
            for step in 0..distance {
                let edge = outgoing.get(cursor)?.iter().copied().find(|edge| {
                    from_source.get(&edge.to) == Some(&(step + 1))
                        && to_target.get(&edge.to) == Some(&(distance - step - 1))
                })?;
                route.push(edge.clone());
                cursor = &edge.to;
            }
            Some(route)
        }

        let owner = FunctionKey::new("crate::compact_routes");
        let nodes = (0..7)
            .map(|index| formal(&owner, index))
            .collect::<Vec<_>>();
        let mut graph = EndpointFlowGraph::default();
        for (from, to, block) in [
            (0, 1, 20),
            (0, 1, 10), // parallel edges: the lower semantic edge wins.
            (0, 2, 30),
            (1, 3, 40),
            (2, 3, 50), // equal shortest alternative through node 2.
            (3, 1, 60), // cycle.
            (3, 4, 70),
            (2, 5, 80),
            (5, 4, 90),
        ] {
            graph.insert_edge(edge(nodes[from].clone(), nodes[to].clone(), block));
        }
        graph.insert_node(nodes[6].clone());

        let index = graph.route_index();
        let routes_to = index.routes_to(&nodes[4]);
        let mut incoming = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
        for edge in &graph.edges {
            incoming.entry(&edge.to).or_default().push(edge);
        }
        assert_eq!(
            routes_to.reaching_nodes().cloned().collect::<Vec<_>>(),
            graph_distances(&nodes[4], &incoming, true)
                .into_keys()
                .collect::<Vec<_>>()
        );
        for source in &nodes {
            assert_eq!(
                routes_to.route_from(source),
                reference_route(&graph, source, &nodes[4]),
                "reverse route from {source:?}"
            );
        }

        let routes_from = index.routes_from(&nodes[0]);
        for target in &nodes {
            assert_eq!(
                routes_from.route_to(target),
                reference_route(&graph, &nodes[0], target),
                "P5 forward route to {target:?}"
            );
        }

        let condensation = index.condensation();
        for selected_mask in 0..(1_u32 << nodes.len()) {
            let selected = nodes
                .iter()
                .enumerate()
                .filter(|(index, _)| selected_mask & (1 << index) != 0)
                .map(|(_, node)| node.clone())
                .collect::<BTreeSet<_>>();
            let mut expected = BTreeSet::new();
            for source in &selected {
                for target in &selected {
                    if source != target && reference_route(&graph, source, target).is_some() {
                        expected.insert(target.clone());
                    }
                }
            }
            assert_eq!(
                condensation.nodes_reached_from_another_in_set(&selected),
                expected,
                "selected mask {selected_mask:#09b}"
            );
        }
    }

    #[test]
    fn route_index_csr_rows_preserve_semantic_edge_order_empty_rows_and_exclusions() {
        let owner = FunctionKey::new("crate::csr_rows");
        let nodes = (0..4)
            .map(|index| formal(&owner, index))
            .collect::<Vec<_>>();
        let mut graph = EndpointFlowGraph::default();
        let mut transparent_first = edge(nodes[0].clone(), nodes[1].clone(), 20);
        transparent_first.kind = EndpointEdgeKind::Transparent;
        let mut transparent_second = edge(nodes[0].clone(), nodes[1].clone(), 10);
        transparent_second.kind = EndpointEdgeKind::Transparent;
        let mut transparent_other = edge(nodes[0].clone(), nodes[2].clone(), 30);
        transparent_other.kind = EndpointEdgeKind::Transparent;
        graph.insert_edge(transparent_first);
        graph.insert_edge(transparent_second);
        graph.insert_edge(transparent_other);
        graph.insert_edge(edge(nodes[2].clone(), nodes[0].clone(), 40));
        graph.insert_node(nodes[3].clone());

        fn expected_row(
            graph: &EndpointFlowGraph,
            node: &EndpointNode,
            excluded: &BTreeSet<EndpointEdgeKind>,
        ) -> Vec<EndpointEdge> {
            graph
                .edges
                .iter()
                .filter(|edge| &edge.from == node && !excluded.contains(&edge.kind))
                .cloned()
                .collect()
        }

        fn assert_rows_match(
            graph: &EndpointFlowGraph,
            index: &EndpointRouteIndex<'_>,
            excluded: &BTreeSet<EndpointEdgeKind>,
        ) {
            assert_eq!(index.outgoing.offsets.len(), graph.nodes.len() + 1);
            assert_eq!(index.incoming.offsets.len(), graph.nodes.len() + 1);
            assert_eq!(index.outgoing.edge_ids.len(), index.edges.len());
            assert_eq!(index.incoming.edge_ids.len(), index.edges.len());
            for node in &graph.nodes {
                let node_id = index.node_id(node).unwrap();
                let expected_outgoing = expected_row(graph, node, excluded);
                assert_eq!(
                    index
                        .outgoing
                        .row(node_id)
                        .iter()
                        .map(|id| index.edge(*id).edge.clone())
                        .collect::<Vec<_>>(),
                    expected_outgoing,
                    "outgoing CSR row for {node:?}"
                );
                assert!(
                    index
                        .outgoing
                        .row(node_id)
                        .windows(2)
                        .all(|ids| ids[0] < ids[1]),
                    "outgoing CSR row must retain global EdgeId order"
                );

                let expected_incoming = graph
                    .edges
                    .iter()
                    .filter(|edge| &edge.to == node && !excluded.contains(&edge.kind))
                    .cloned()
                    .collect::<Vec<_>>();
                assert_eq!(
                    index
                        .incoming
                        .row(node_id)
                        .iter()
                        .map(|id| index.edge(*id).edge.clone())
                        .collect::<Vec<_>>(),
                    expected_incoming,
                    "incoming CSR row for {node:?}"
                );
                assert!(
                    index
                        .incoming
                        .row(node_id)
                        .windows(2)
                        .all(|ids| ids[0] < ids[1]),
                    "incoming CSR row must retain global EdgeId order"
                );
            }
        }

        let included = graph.route_index();
        assert_rows_match(&graph, &included, &BTreeSet::new());
        let empty = included.node_id(&nodes[3]).unwrap();
        assert!(included.outgoing.row(empty).is_empty());
        assert!(included.incoming.row(empty).is_empty());

        let excluded = BTreeSet::from([EndpointEdgeKind::ActualToFormal]);
        let without_actual_to_formal = graph.route_index_excluding(&excluded);
        assert_rows_match(&graph, &without_actual_to_formal, &excluded);
        let only_excluded = without_actual_to_formal.node_id(&nodes[2]).unwrap();
        assert!(without_actual_to_formal
            .outgoing
            .row(only_excluded)
            .is_empty());
    }

    #[test]
    fn condensation_marks_only_sources_reached_from_a_distinct_selected_source() {
        let owner = FunctionKey::new("crate::condensation");
        let nodes = (0..6)
            .map(|index| formal(&owner, index))
            .collect::<Vec<_>>();
        let mut graph = EndpointFlowGraph::default();
        for (from, to, block) in [(0, 1, 1), (1, 2, 2), (2, 1, 3), (2, 3, 4), (4, 4, 5)] {
            graph.insert_edge(edge(nodes[from].clone(), nodes[to].clone(), block));
        }
        graph.insert_node(nodes[5].clone());
        let route_index = graph.route_index();
        let condensation = route_index.condensation();

        for selected_mask in 0..(1_u32 << nodes.len()) {
            let selected = nodes
                .iter()
                .enumerate()
                .filter(|(index, _)| selected_mask & (1 << index) != 0)
                .map(|(_, node)| node.clone())
                .collect::<BTreeSet<_>>();
            let mut reference = BTreeSet::new();
            for source in &selected {
                reference.extend(
                    route_index
                        .routes_from(source)
                        .reachable_nodes()
                        .filter(|reachable| selected.contains(*reachable))
                        .cloned(),
                );
            }
            assert_eq!(
                condensation.nodes_reached_from_another_in_set(&selected),
                reference,
                "selected mask {selected_mask:#08b}"
            );
        }
    }

    #[test]
    fn condensation_compacts_parallel_component_edges_without_changing_selection() {
        let owner = FunctionKey::new("crate::condensation_parallel");
        let left = formal(&owner, 0);
        let right = formal(&owner, 1);
        let sink = formal(&owner, 2);
        let mut graph = EndpointFlowGraph::default();
        graph.insert_edge(edge(left.clone(), right.clone(), 1));
        graph.insert_edge(edge(right.clone(), left.clone(), 2));
        graph.insert_edge(edge(left.clone(), sink.clone(), 3));
        graph.insert_edge(edge(right.clone(), sink.clone(), 4));

        let route_index = graph.route_index();
        let condensation = route_index.condensation();
        let left_component = condensation.component_for(&left).unwrap();
        let sink_component = condensation.component_for(&sink).unwrap();
        assert_eq!(
            condensation.successors[left_component],
            vec![sink_component]
        );
        assert_eq!(
            condensation
                .nodes_reached_from_another_in_set(&BTreeSet::from([left.clone(), sink.clone(),])),
            BTreeSet::from([sink])
        );
    }

    #[test]
    fn direct_and_mutual_recursion_form_finite_stable_components() {
        let owner = FunctionKey::new("crate::recursive");
        let direct = formal(&owner, 1);
        let left = formal(&owner, 2);
        let right = formal(&owner, 3);
        let mut graph = EndpointFlowGraph::default();
        graph.insert_edge(edge(direct.clone(), direct.clone(), 1));
        graph.insert_edge(edge(left.clone(), right.clone(), 2));
        graph.insert_edge(edge(right.clone(), left.clone(), 3));
        assert_eq!(
            graph.cyclic_components(),
            vec![BTreeSet::from([direct]), BTreeSet::from([left, right])]
        );
    }

    #[test]
    fn graph_join_is_monotone_idempotent_and_preserves_one_hop_edges() {
        let owner = FunctionKey::new("crate::join");
        let first_edge = edge(formal(&owner, 1), formal(&owner, 2), 1);
        let second_edge = edge(formal(&owner, 2), formal(&owner, 3), 2);
        let mut graph = EndpointFlowGraph::default();
        graph.insert_edge(first_edge.clone());
        let mut other = EndpointFlowGraph::default();
        other.insert_edge(second_edge.clone());
        assert!(graph.join(&other));
        assert!(!graph.join(&other));
        assert_eq!(graph.edges, BTreeSet::from([first_edge, second_edge]));
        assert!(graph
            .edges
            .iter()
            .all(|edge| graph.nodes.contains(&edge.from) && graph.nodes.contains(&edge.to)));
    }

    fn context_edge(from: EndpointNode, to: EndpointNode, block: u32) -> EndpointEdge {
        let owner = context_endpoint_owner(&from).expect("test endpoint has an owner");
        EndpointEdge {
            from,
            to,
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(&owner, block)),
        }
    }

    #[test]
    fn context_proof_keeps_distinct_callsites_and_orders_two_hops() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let leaf = FunctionKey::new("crate::leaf");
        let root = formal(&caller, 1);
        let first = EndpointNode::CallResult {
            callsite: CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone()),
            output: CallSiteOutput::Return,
        };
        let second = EndpointNode::CallResult {
            callsite: CallSiteId::new(caller.clone(), point(&caller, 20), leaf.clone()),
            output: CallSiteOutput::Return,
        };
        let sink = EndpointNode::UnsafeOperand {
            owner: caller.clone(),
            site_id: "sink".to_string(),
            role: OperandRole::Argument(0),
        };
        let mut index = ContextProofIndex::default();
        let call_a = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let call_b = CallSiteId::new(caller.clone(), point(&caller, 20), leaf.clone());
        let callee_relation = ContextRelationKey::new(
            callee.clone(),
            formal(&callee, 1),
            EndpointNode::Return {
                owner: callee.clone(),
            },
        );
        let leaf_relation = ContextRelationKey::new(
            leaf.clone(),
            formal(&leaf, 1),
            EndpointNode::Return {
                owner: leaf.clone(),
            },
        );
        index.record_local(context_edge(
            callee_relation.source.clone(),
            callee_relation.target.clone(),
            1,
        ));
        index.record_local(context_edge(
            leaf_relation.source.clone(),
            leaf_relation.target.clone(),
            2,
        ));
        index.record_call(
            EndpointEdge {
                from: root.clone(),
                to: first.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(call_a.point.clone()),
            },
            call_a,
            GenericCallSubstitution::default(),
            callee_relation,
            ContextActualBinding::Formal(1),
        );
        index.record_call(
            EndpointEdge {
                from: first,
                to: second.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(call_b.point.clone()),
            },
            call_b,
            GenericCallSubstitution::default(),
            leaf_relation,
            ContextActualBinding::Formal(1),
        );
        index.record_local(context_edge(second, sink.clone(), 21));
        index.finalize();

        let reaching = index.exact_reaching_atoms_for_endpoint(&sink);
        assert!(reaching.contains(&ContextAtom::scoped(caller.clone(), root.clone())));
        assert!(reaching.contains(&ContextAtom::scoped(callee.clone(), formal(&callee, 1))));
        assert!(reaching.contains(&ContextAtom::scoped(leaf.clone(), formal(&leaf, 1))));

        let ContextQuery::Realizable(proof) = index.query_via(&root, &root, &sink) else {
            panic!("two exact call contexts must be realizable");
        };
        assert_eq!(proof.canonical_callsites().len(), 2);
        assert_eq!(proof.canonical_callsites()[0].0.point.block, 10);
        assert_eq!(proof.canonical_callsites()[1].0.point.block, 20);
    }

    #[test]
    fn compact_generic_capability_reverse_walk_matches_atom_facade() {
        let owner = FunctionKey::new("crate::generic_context");
        let parameter = GenericParamId {
            def_path: "crate::generic_context::T".to_string(),
            kind: GenericParamKind::Type,
        };
        let capability = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence: GenericCapabilityEvidence {
                kind: GenericCapabilityKind::AssociatedSequenceShape {
                    associated_items: BTreeSet::from(["crate::Provider::Results".to_string()]),
                },
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
            },
            span: point(&owner, 1).span,
        };
        let unrelated = formal(&owner, 7);
        let sink = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: "sink".to_string(),
            role: OperandRole::Argument(0),
        };
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(capability.clone(), sink.clone(), 1));
        index.record_local(context_edge(unrelated, sink.clone(), 2));
        index.finalize();

        let expected = index
            .exact_reaching_atoms_for_endpoint(&sink)
            .into_iter()
            .filter_map(|atom| match atom.endpoint {
                endpoint @ EndpointNode::GenericCapability { .. } => Some((endpoint, atom.owner)),
                _ => None,
            })
            .fold(
                BTreeMap::<EndpointNode, BTreeSet<FunctionKey>>::new(),
                |mut scopes, (endpoint, owner)| {
                    scopes.entry(endpoint).or_default().insert(owner);
                    scopes
                },
            );
        assert_eq!(
            index.exact_generic_capability_scopes_reaching_endpoint(&sink),
            expected
        );
        assert_eq!(
            expected,
            BTreeMap::from([(capability, BTreeSet::from([owner]))])
        );
    }

    #[test]
    fn context_unknown_callback_never_becomes_unrealizable() {
        let owner = FunctionKey::new("crate::callback");
        let root = formal(&owner, 1);
        let sink = formal(&owner, 2);
        let mut index = ContextProofIndex::default();
        index.record_local(EndpointEdge {
            from: root.clone(),
            to: sink.clone(),
            kind: EndpointEdgeKind::ActualToFormal,
            point: Some(point(&owner, 1)),
        });
        index.finalize();
        assert!(matches!(
            index.query_via(&root, &root, &sink),
            ContextQuery::Unknown { reasons, .. }
                if reasons.contains(&ContextUnknownReason::UncontextualizedCallback)
        ));
    }

    #[test]
    fn context_proof_is_finite_across_64_hops_and_recursive_scc() {
        let owner = FunctionKey::new("crate::long_context");
        let nodes = (0..=64)
            .map(|index| formal(&owner, index))
            .collect::<Vec<_>>();
        let mut index = ContextProofIndex::default();
        for index_at in 0..64 {
            index.record_local(context_edge(
                nodes[index_at].clone(),
                nodes[index_at + 1].clone(),
                index_at as u32,
            ));
        }
        index.record_local(context_edge(nodes[32].clone(), nodes[31].clone(), 80));
        index.finalize();
        let ContextQuery::Realizable(proof) = index.query_via(&nodes[0], &nodes[0], &nodes[64])
        else {
            panic!("the 64-hop exact local proof must remain finite");
        };
        assert_eq!(proof.canonical_steps().len(), 64);
        // A local endpoint loop is not an interprocedural recursive summary.
        assert!(proof.recursive_sccs.is_empty());
    }

    #[test]
    fn missing_callee_dependency_is_unknown_not_realizable() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::missing");
        let root = formal(&caller, 1);
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let result = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let sink = EndpointNode::UnsafeOperand {
            owner: caller.clone(),
            site_id: "sink".into(),
            role: OperandRole::Argument(0),
        };
        let mut index = ContextProofIndex::default();
        index.record_call(
            EndpointEdge {
                from: root.clone(),
                to: result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite,
            GenericCallSubstitution::default(),
            ContextRelationKey::new(
                callee.clone(),
                formal(&callee, 1),
                EndpointNode::Return { owner: callee },
            ),
            ContextActualBinding::Formal(1),
        );
        index.record_local(context_edge(result, sink.clone(), 11));
        index.finalize();
        assert!(matches!(
            index.query_via(&root, &root, &sink),
            ContextQuery::Unknown {
                reasons,
                completeness: ContextCompletenessEvidence::Unavailable { .. },
            } if reasons.contains(&ContextUnknownReason::CompletenessUnavailable)
        ));
    }

    #[test]
    fn incomplete_substitution_keeps_non_p4_execution_route_realizable() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let root = formal(&caller, 1);
        let callee_input = formal(&callee, 1);
        let callee_return = EndpointNode::Return {
            owner: callee.clone(),
        };
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let result = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let sink = EndpointNode::UnsafeOperand {
            owner: caller.clone(),
            site_id: "sink".into(),
            role: OperandRole::Argument(0),
        };
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(callee_input.clone(), callee_return.clone(), 1));
        index.record_call(
            EndpointEdge {
                from: root.clone(),
                to: result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite,
            GenericCallSubstitution {
                status: GenericSubstitutionStatus::NormalizationIncomplete,
                bindings: BTreeMap::new(),
            },
            ContextRelationKey::new(callee, callee_input, callee_return),
            ContextActualBinding::Formal(1),
        );
        index.record_local(context_edge(result, sink.clone(), 11));
        index.finalize();
        let ContextQuery::Realizable(proof) = index.query_via(&root, &root, &sink) else {
            panic!("incomplete generic normalization must not erase real data flow");
        };
        assert!(proof.canonical_has_incomplete_generic_substitution());
    }

    #[test]
    fn proof_alternatives_keep_callsite_substitutions_separate() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let root = formal(&caller, 1);
        let sink = EndpointNode::UnsafeOperand {
            owner: caller.clone(),
            site_id: "sink".into(),
            role: OperandRole::Argument(0),
        };
        let callee_relation = ContextRelationKey::new(
            callee.clone(),
            formal(&callee, 1),
            EndpointNode::Return {
                owner: callee.clone(),
            },
        );
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(
            callee_relation.source.clone(),
            callee_relation.target.clone(),
            1,
        ));
        // Insert lexical B before A: internal arena IDs follow insertion,
        // while exact iteration and the canonical display must remain sorted
        // by the decoded callsite value.
        for (block, status) in [
            (20, GenericSubstitutionStatus::NormalizationIncomplete),
            (10, GenericSubstitutionStatus::Complete),
        ] {
            let callsite = CallSiteId::new(caller.clone(), point(&caller, block), callee.clone());
            index.record_call(
                EndpointEdge {
                    from: root.clone(),
                    to: sink.clone(),
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(callsite.point.clone()),
                },
                callsite,
                GenericCallSubstitution {
                    status,
                    bindings: BTreeMap::new(),
                },
                callee_relation.clone(),
                ContextActualBinding::Formal(1),
            );
        }
        index.finalize();
        let root_atom = ContextAtom {
            owner: caller.clone(),
            endpoint: root.clone(),
        };
        let exact_relations = index.exact_relations_from(&root_atom).collect::<Vec<_>>();
        assert_eq!(exact_relations.len(), 1);
        assert_eq!(exact_relations[0].1.len(), 2);
        let ContextQuery::Realizable(proof) = index.query_via(&root, &root, &sink) else {
            panic!("both exact call alternatives must remain realizable");
        };
        let alternatives = proof.alternatives().collect::<Vec<_>>();
        assert_eq!(alternatives.len(), 2);
        assert_eq!(alternatives[0].callsites()[0].0.point.block, 10);
        assert!(!alternatives[0].has_incomplete_generic_substitution());
        assert_eq!(alternatives[1].callsites()[0].0.point.block, 20);
        assert!(alternatives[1].has_incomplete_generic_substitution());
        // Canonical display is one whole route, not a per-relation mash-up.
        assert_eq!(proof.canonical_callsites()[0].0.point.block, 10);
        assert!(index
            .unknown_reasons_for_callsite(&CallSiteId::new(
                caller,
                point(&FunctionKey::new("crate::caller"), 20),
                callee,
            ))
            .contains(&ContextUnknownReason::NormalizationIncomplete {
                callsite: CallSiteId::new(
                    FunctionKey::new("crate::caller"),
                    point(&FunctionKey::new("crate::caller"), 20),
                    FunctionKey::new("crate::callee"),
                ),
            }));
    }

    #[test]
    fn compact_exact_facade_preserves_decoded_alternative_order() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let source = formal(&caller, 1);
        let sink = formal(&caller, 2);
        let callee_relation = ContextRelationKey::new(
            callee.clone(),
            formal(&callee, 1),
            EndpointNode::Return {
                owner: callee.clone(),
            },
        );
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(
            callee_relation.source.clone(),
            callee_relation.target.clone(),
            1,
        ));
        for (block, status) in [
            (20, GenericSubstitutionStatus::NormalizationIncomplete),
            (10, GenericSubstitutionStatus::Complete),
        ] {
            let callsite = CallSiteId::new(caller.clone(), point(&caller, block), callee.clone());
            index.record_call(
                EndpointEdge {
                    from: source.clone(),
                    to: sink.clone(),
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(callsite.point.clone()),
                },
                callsite,
                GenericCallSubstitution {
                    status,
                    bindings: BTreeMap::new(),
                },
                callee_relation.clone(),
                ContextActualBinding::Formal(1),
            );
        }
        index.finalize();

        let source_handle = index.exact_atom_handles_for_endpoint(&source)[0];
        assert_eq!(index.exact_atom(source_handle).owner, caller);
        let relations = index
            .exact_relation_handles_from(source_handle)
            .collect::<Vec<_>>();
        assert_eq!(relations.len(), 1);
        assert_eq!(index.exact_relation(relations[0]).target, sink);
        assert_eq!(
            index.exact_atom(index.exact_relation_source(relations[0])),
            index.exact_atom(source_handle)
        );
        let predecessors = index.exact_predecessor_handles(relations[0]);
        assert_eq!(predecessors.len(), 2);
        let decoded = predecessors
            .into_iter()
            .map(|predecessor| match index.exact_predecessor(predecessor) {
                ExactPredecessorRef::Call {
                    callsite,
                    substitution,
                    callee_relation: relation,
                    actual_binding,
                } => (
                    callsite.point.block,
                    substitution.status,
                    index.exact_relation(relation).clone(),
                    actual_binding,
                ),
                ExactPredecessorRef::Local { .. } => panic!("expected call alternative"),
            })
            .collect::<Vec<_>>();
        assert_eq!(decoded[0].0, 10);
        assert_eq!(decoded[0].1, GenericSubstitutionStatus::Complete);
        assert_eq!(decoded[0].2, callee_relation);
        assert_eq!(decoded[0].3, ContextActualBinding::Formal(1));
        assert_eq!(decoded[1].0, 20);
        assert_eq!(
            decoded[1].1,
            GenericSubstitutionStatus::NormalizationIncomplete
        );
    }

    #[test]
    fn dense_shared_call_payload_keeps_each_source_alternative_exact() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let sources = (1..=64)
            .map(|index| formal(&caller, index))
            .collect::<Vec<_>>();
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let result = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let relation = ContextRelationKey::new(
            callee.clone(),
            formal(&callee, 1),
            EndpointNode::Return {
                owner: callee.clone(),
            },
        );
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(
            relation.source.clone(),
            relation.target.clone(),
            1,
        ));
        for source in &sources {
            index.record_call(
                EndpointEdge {
                    from: source.clone(),
                    to: result.clone(),
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(callsite.point.clone()),
                },
                callsite.clone(),
                GenericCallSubstitution::default(),
                relation.clone(),
                ContextActualBinding::Formal(1),
            );
        }
        index.finalize();

        // The payload is physically shared, but all outer relations remain
        // independently exact: no source is coalesced merely because it uses
        // the same real callsite and substitution.
        assert_eq!(index.callsites.len(), 1);
        assert_eq!(index.substitutions.len(), 1);
        // One callee relation plus one distinct outer relation per source;
        // all 65 are interned once and referenced by the raw predecessor map.
        assert_eq!(index.arena.relations.len(), 65);
        assert_eq!(index.predecessor_count(), 65);
        assert_eq!(index.exact_predecessor_count(), 65);
        // One compact local `{kind, point}` and one compact Call payload back
        // every raw/exact predecessor entry; no local EndpointEdge survives
        // in the persistent index.
        assert_eq!(index.arena.predecessors.len(), 2);
        assert_eq!(index.points.len(), 1);
        for source in &sources {
            assert!(matches!(
                index.query_via(source, source, &result),
                ContextQuery::Realizable(_)
            ));
        }
    }

    #[test]
    fn recursive_sccs_follow_call_relation_dependencies_not_endpoint_loops() {
        let owner = FunctionKey::new("crate::recursive_calls");
        let first = ContextRelationKey::new(owner.clone(), formal(&owner, 1), formal(&owner, 2));
        let second = ContextRelationKey::new(owner.clone(), formal(&owner, 3), formal(&owner, 4));
        let mut relations = BTreeMap::new();
        for (outer, callee, block) in [
            (first.clone(), second.clone(), 10),
            (second.clone(), first.clone(), 20),
        ] {
            let callsite = CallSiteId::new(owner.clone(), point(&owner, block), owner.clone());
            relations.insert(
                outer.clone(),
                BTreeSet::from([ContextPredecessor::Call {
                    callsite: Arc::new(callsite.clone()),
                    substitution: Arc::new(GenericCallSubstitution::default()),
                    callee_relation: Arc::new(callee),
                    actual_binding: ContextActualBinding::Internal,
                }]),
            );
        }
        let components = context_recursive_sccs(&relations);
        let component = BTreeSet::from([first.clone(), second.clone()]);
        assert_eq!(components, BTreeSet::from([component.clone()]));
        assert_eq!(
            context_recursive_scc_grounding(&relations, &components)
                .get(&component)
                .cloned()
                .unwrap_or_default(),
            BTreeSet::new()
        );

        // Adding an exact local base makes the SCC groundable, while the
        // recursive call edges remain represented by the same finite token.
        relations
            .get_mut(&first)
            .expect("first relation exists")
            .insert(ContextPredecessor::Local {
                fragment: Arc::new(context_edge(first.source.clone(), first.target.clone(), 30)),
            });
        assert_eq!(
            context_recursive_scc_grounding(&relations, &components)
                .get(&component)
                .cloned()
                .unwrap_or_default(),
            BTreeSet::from([first])
        );
    }

    #[test]
    fn multi_edge_summary_requests_form_direct_and_mutual_call_sccs() {
        let left = FunctionKey::new("crate::left");
        let right = FunctionKey::new("crate::right");
        let left_request = ContextRelationKey::new(
            left.clone(),
            formal(&left, 1),
            EndpointNode::Return {
                owner: left.clone(),
            },
        );
        let right_request = ContextRelationKey::new(
            right.clone(),
            formal(&right, 1),
            EndpointNode::Return {
                owner: right.clone(),
            },
        );
        let left_prefix =
            ContextRelationKey::new(left.clone(), left_request.source.clone(), formal(&left, 99));
        let left_call = ContextRelationKey::new(
            left.clone(),
            left_prefix.target.clone(),
            left_request.target.clone(),
        );
        let right_prefix = ContextRelationKey::new(
            right.clone(),
            right_request.source.clone(),
            formal(&right, 99),
        );
        let right_call = ContextRelationKey::new(
            right.clone(),
            right_prefix.target.clone(),
            right_request.target.clone(),
        );
        let call_predecessor = |_outer: &ContextRelationKey,
                                callee: ContextRelationKey,
                                caller: &FunctionKey,
                                block| {
            let callsite = CallSiteId::new(
                caller.clone(),
                point(caller, block),
                callee.context_owner.clone(),
            );
            ContextPredecessor::Call {
                callsite: Arc::new(callsite.clone()),
                substitution: Arc::new(GenericCallSubstitution::default()),
                callee_relation: Arc::new(callee),
                actual_binding: ContextActualBinding::Internal,
            }
        };
        let mut relations = BTreeMap::from([
            (
                left_prefix.clone(),
                BTreeSet::from([ContextPredecessor::Local {
                    fragment: Arc::new(context_edge(
                        left_prefix.source.clone(),
                        left_prefix.target.clone(),
                        1,
                    )),
                }]),
            ),
            (
                left_call.clone(),
                BTreeSet::from([call_predecessor(
                    &left_call,
                    right_request.clone(),
                    &left,
                    10,
                )]),
            ),
            (
                right_prefix.clone(),
                BTreeSet::from([ContextPredecessor::Local {
                    fragment: Arc::new(context_edge(
                        right_prefix.source.clone(),
                        right_prefix.target.clone(),
                        2,
                    )),
                }]),
            ),
            (
                right_call.clone(),
                BTreeSet::from([call_predecessor(
                    &right_call,
                    left_request.clone(),
                    &right,
                    20,
                )]),
            ),
        ]);
        let component = BTreeSet::from([left_request.clone(), right_request.clone()]);
        let components = context_recursive_sccs(&relations);
        assert!(components.contains(&component));
        // The local prefixes alone are not a certificate: both complete
        // summary requests still depend on the cycle.
        assert!(context_recursive_scc_grounding(&relations, &components)
            .get(&component)
            .is_some_and(BTreeSet::is_empty));

        relations
            .get_mut(&left_call)
            .expect("left call relation exists")
            .insert(ContextPredecessor::Local {
                fragment: Arc::new(context_edge(
                    left_call.source.clone(),
                    left_call.target.clone(),
                    30,
                )),
            });
        assert_eq!(
            context_recursive_scc_grounding(&relations, &components)
                .get(&component)
                .cloned()
                .unwrap_or_default(),
            BTreeSet::from([left_request])
        );
    }

    #[test]
    fn direct_recursive_alternative_uses_one_finite_scc_token() {
        let owner = FunctionKey::new("crate::direct_recursive");
        let root = formal(&owner, 1);
        let target = formal(&owner, 2);
        let relation = ContextRelationKey::new(owner.clone(), root.clone(), target.clone());
        let callsite = CallSiteId::new(owner.clone(), point(&owner, 10), owner.clone());
        let predecessor = ContextPredecessor::Call {
            callsite: Arc::new(callsite.clone()),
            substitution: Arc::new(GenericCallSubstitution::default()),
            callee_relation: Arc::new(relation.clone()),
            actual_binding: ContextActualBinding::Internal,
        };
        let proof = ContextProof {
            execution_anchor_owner: owner.clone(),
            root_context: owner.clone(),
            root,
            source_context: owner.clone(),
            source: formal(&owner, 1),
            target_context: owner,
            target,
            predecessors: BTreeMap::from([(relation.clone(), BTreeSet::from([predecessor]))]),
            recursive_sccs: BTreeSet::from([BTreeSet::from([relation.clone()])]),
            recursive_scc_grounding: BTreeMap::from([(
                BTreeSet::from([relation.clone()]),
                BTreeSet::new(),
            )]),
        };
        // A bare recursive edge is not a successful proof leaf. The finite
        // SCC token stays on the proof DAG, but canonical selection requires
        // a grounded local or escaping predecessor.
        assert!(proof.alternatives().next().is_none());
        assert_eq!(
            proof.recursive_sccs,
            BTreeSet::from([BTreeSet::from([relation])])
        );
    }

    #[test]
    fn endpoint_only_out_result_has_a_normal_context_proof_but_no_unwind_bridge() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let source = EndpointNode::UnsafeResult {
            owner: callee.clone(),
            site_id: "unsafe".into(),
        };
        let out = EndpointNode::Out {
            owner: callee.clone(),
            index: 1,
        };
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let normal = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: CallSiteOutput::Out(1),
        };
        let unwind = EndpointNode::CallResult {
            callsite: CallSiteId::new(caller.clone(), point(&caller, 11), callee.clone()),
            output: CallSiteOutput::Out(1),
        };
        let sink = EndpointNode::UnsafeOperand {
            owner: caller.clone(),
            site_id: "sink".into(),
            role: OperandRole::Argument(0),
        };
        let relation = ContextRelationKey::new(callee.clone(), source.clone(), out.clone());
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(source.clone(), out, 1));
        index.record_call(
            EndpointEdge {
                from: source.clone(),
                to: normal.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite,
            GenericCallSubstitution::default(),
            relation,
            ContextActualBinding::Internal,
        );
        index.record_local(context_edge(normal.clone(), sink.clone(), 12));
        index.finalize();
        assert!(matches!(
            index.query_via_scoped(
                caller.clone(),
                source.clone(),
                caller.clone(),
                source.clone(),
                caller.clone(),
                sink,
            ),
            ContextQuery::Realizable(_)
        ));
        assert!(matches!(
            index.query_via_scoped(
                caller.clone(),
                source.clone(),
                caller.clone(),
                source,
                caller,
                unwind,
            ),
            ContextQuery::Unknown { .. }
        ));
    }

    #[test]
    fn callsite_unknown_reason_is_indexed_even_when_marked_on_unsafe_terminal() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 10), callee);
        let terminal = EndpointNode::UnsafeOperand {
            owner: caller,
            site_id: "sink".into(),
            role: OperandRole::Argument(0),
        };
        let mut index = ContextProofIndex::default();
        index.mark_unknown_at(
            terminal,
            ContextUnknownReason::MissingActualEndpoint {
                callsite: callsite.clone(),
            },
        );
        assert_eq!(
            index.unknown_reasons_for_callsite(&callsite),
            BTreeSet::from([ContextUnknownReason::MissingActualEndpoint { callsite }])
        );
    }

    #[test]
    fn exact_alternative_survives_unknown_sibling() {
        let owner = FunctionKey::new("crate::alternatives");
        let root = formal(&owner, 1);
        let sink = formal(&owner, 2);
        let callback_formal = formal(&owner, 3);
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(root.clone(), sink.clone(), 1));
        index.record_local(EndpointEdge {
            from: root.clone(),
            to: callback_formal,
            kind: EndpointEdgeKind::ActualToFormal,
            point: Some(point(&owner, 2)),
        });
        index.finalize();
        assert!(matches!(
            index.query_via(&root, &root, &sink),
            ContextQuery::Realizable(_)
        ));
    }

    #[test]
    fn scoped_internal_source_alternatives_do_not_splice_sibling_calls() {
        let callee = FunctionKey::new("crate::callee");
        let caller_a = FunctionKey::new("crate::caller_a");
        let caller_b = FunctionKey::new("crate::caller_b");
        let source = EndpointNode::InternalDerivation {
            owner: callee.clone(),
            kind: DerivationKind::Index,
            token: "source".into(),
            point: point(&callee, 1),
        };
        let callee_return = EndpointNode::Return {
            owner: callee.clone(),
        };
        let call_a = CallSiteId::new(caller_a.clone(), point(&caller_a, 10), callee.clone());
        let call_b = CallSiteId::new(caller_b.clone(), point(&caller_b, 20), callee.clone());
        let result_a = EndpointNode::CallResult {
            callsite: call_a.clone(),
            output: CallSiteOutput::Return,
        };
        let result_b = EndpointNode::CallResult {
            callsite: call_b.clone(),
            output: CallSiteOutput::Return,
        };
        let sink = EndpointNode::UnsafeOperand {
            owner: caller_b.clone(),
            site_id: "sink_b".into(),
            role: OperandRole::Argument(0),
        };
        let callee_relation =
            ContextRelationKey::new(callee.clone(), source.clone(), callee_return.clone());
        let mut index = ContextProofIndex::default();
        index.record_local(context_edge(source.clone(), callee_return, 1));
        for (callsite, result) in [(call_a, result_a), (call_b.clone(), result_b.clone())] {
            index.record_call(
                EndpointEdge {
                    from: source.clone(),
                    to: result,
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(callsite.point.clone()),
                },
                callsite,
                GenericCallSubstitution::default(),
                callee_relation.clone(),
                ContextActualBinding::Internal,
            );
        }
        index.record_local(context_edge(result_b, sink.clone(), 21));
        index.finalize();
        let alternatives = index.scoped_alternatives(&source, &sink);
        assert_eq!(alternatives.len(), 1);
        assert_eq!(alternatives[0].execution_anchor_owner, caller_b);
        assert_eq!(alternatives[0].canonical_callsites()[0].0, call_b);
    }

    #[test]
    fn coarse_fabricated_boundary_route_is_not_a_context_proof() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let actual = formal(&caller, 1);
        let shared_formal = formal(&callee, 1);
        let sink = EndpointNode::UnsafeOperand {
            owner: callee.clone(),
            site_id: "sink".into(),
            role: OperandRole::Argument(0),
        };
        let fabricated = EndpointEdge {
            from: actual.clone(),
            to: shared_formal.clone(),
            kind: EndpointEdgeKind::ActualToFormal,
            point: Some(point(&caller, 10)),
        };
        let local = context_edge(shared_formal, sink.clone(), 1);
        let mut coarse = EndpointFlowGraph::default();
        coarse.insert_edge(fabricated.clone());
        coarse.insert_edge(local.clone());
        assert!(coarse.canonical_route(&actual, &sink).is_some());

        let mut context = ContextProofIndex::default();
        context.record_local(fabricated);
        context.record_local(local);
        context.finalize();
        assert!(matches!(
            context.query_via(&actual, &actual, &sink),
            ContextQuery::Unknown {
                completeness: ContextCompletenessEvidence::Unavailable { .. },
                ..
            }
        ));
    }

    #[test]
    fn call_dependency_fixed_point_keeps_sixty_four_nested_predecessors() {
        let owners = (0..=64)
            .map(|index| FunctionKey::new(format!("crate::f{index}")))
            .collect::<Vec<_>>();
        let mut index = ContextProofIndex::default();
        let final_input = formal(&owners[64], 1);
        let final_return = EndpointNode::Return {
            owner: owners[64].clone(),
        };
        index.record_local(context_edge(final_input, final_return, 64));
        for position in 0..64 {
            let caller = &owners[position];
            let callee = &owners[position + 1];
            let input = formal(caller, 1);
            let result = EndpointNode::CallResult {
                callsite: CallSiteId::new(
                    caller.clone(),
                    point(caller, position as u32),
                    callee.clone(),
                ),
                output: CallSiteOutput::Return,
            };
            let caller_return = EndpointNode::Return {
                owner: caller.clone(),
            };
            index.record_local(context_edge(
                result.clone(),
                caller_return.clone(),
                position as u32,
            ));
            index.record_call(
                EndpointEdge {
                    from: input,
                    to: result,
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(point(caller, position as u32)),
                },
                CallSiteId::new(
                    caller.clone(),
                    point(caller, position as u32),
                    callee.clone(),
                ),
                GenericCallSubstitution::default(),
                ContextRelationKey::new(
                    callee.clone(),
                    formal(callee, 1),
                    EndpointNode::Return {
                        owner: callee.clone(),
                    },
                ),
                ContextActualBinding::Formal(1),
            );
        }
        index.finalize();
        let root = formal(&owners[0], 1);
        let target = EndpointNode::Return {
            owner: owners[0].clone(),
        };
        let ContextQuery::Realizable(proof) = index.query_via(&root, &root, &target) else {
            panic!("the call relation fixed point must retain the 64-hop route");
        };
        let callsites = proof.canonical_callsites();
        assert_eq!(callsites.len(), 64);
        assert_eq!(callsites.first().unwrap().0.caller, owners[0]);
        assert_eq!(callsites.last().unwrap().0.caller, owners[63]);
    }
}
