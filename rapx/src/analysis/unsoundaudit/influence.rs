use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::ops::{BitAnd, BitOr, BitOrAssign};
use std::sync::Arc;

use super::contract_catalog::{ContractDescriptor, ContractPredicate};
use super::endpoint_flow::{
    ContextActualBinding, ContextAtomHandle, ContextPredecessorHandle, ContextProofIndex,
    ContextRelationHandle, ContextUnknownReason, EndpointNode, ExactPredecessorRef,
    ExactReverseReachability, GenericParamId, GenericSubstitutionStatus, OpenBehaviorReceiver,
};
use super::summary::FunctionKey;
use super::unsafe_site::{ContractPrecision, OperandRole};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct InfluenceCapabilities(u8);

impl InfluenceCapabilities {
    pub(super) const ADDRESS: Self = Self(1 << 0);
    pub(super) const INDEX_OR_LENGTH: Self = Self(1 << 1);
    pub(super) const TYPED_VALUE: Self = Self(1 << 2);
    pub(super) const VALIDITY_BITS: Self = Self(1 << 3);
    pub(super) const BOOLEAN: Self = Self(1 << 4);
    pub(super) const CALLABLE_RESULT: Self = Self(1 << 5);

    pub(super) fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub(super) fn stable_identity(self) -> String {
        [
            (Self::ADDRESS, "address"),
            (Self::INDEX_OR_LENGTH, "index_or_length"),
            (Self::TYPED_VALUE, "typed_value"),
            (Self::VALIDITY_BITS, "validity_bits"),
            (Self::BOOLEAN, "boolean"),
            (Self::CALLABLE_RESULT, "callable_result"),
        ]
        .into_iter()
        .filter_map(|(capability, name)| self.intersects(capability).then_some(name))
        .collect::<Vec<_>>()
        .join(",")
    }
}

impl BitOr for InfluenceCapabilities {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for InfluenceCapabilities {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for InfluenceCapabilities {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum InfluenceOrigin {
    PublicFormal,
    PersistentPublicField,
    FfiReturn,
    FfiOut,
    FfiCallbackInput,
    OpenTraitReturn,
    PublicCallbackReturn,
    PublicFunctionPointerReturn,
}

impl InfluenceOrigin {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::PublicFormal => "public_formal",
            Self::PersistentPublicField => "persistent_public_field",
            Self::FfiReturn => "ffi_return",
            Self::FfiOut => "ffi_out",
            Self::FfiCallbackInput => "ffi_callback_input",
            Self::OpenTraitReturn => "open_trait_return",
            Self::PublicCallbackReturn => "public_callback_return",
            Self::PublicFunctionPointerReturn => "public_function_pointer_return",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum InfluenceController {
    FfiHost,
    PublicFormal {
        owner: FunctionKey,
        index: u32,
    },
    PersistentPublicField {
        access_owner: FunctionKey,
        field: String,
    },
    OpenTrait {
        trait_id: FunctionKey,
        receiver: OpenBehaviorReceiver,
    },
    PublicCallback,
    Internal,
    Unknown,
}

impl InfluenceController {
    pub(super) fn stable_identity(&self) -> String {
        match self {
            Self::FfiHost => "ffi_host".to_string(),
            Self::PublicFormal { owner, index } => {
                format!("public_formal:{}:{index}", owner.0)
            }
            Self::PersistentPublicField {
                access_owner,
                field,
            } => format!("persistent_public_field:{}:{field}", access_owner.0),
            Self::OpenTrait { trait_id, receiver } => {
                format!("open_trait:{}:{}", trait_id.0, receiver.stable_identity())
            }
            Self::PublicCallback => "public_callback".to_string(),
            Self::Internal => "internal".to_string(),
            Self::Unknown => "unknown".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum TokenPrecision {
    StrongBoundary,
    ExactDerived,
    Incomplete,
}

/// Compiler-derived shape used to seed the finite capability domain.  MIR
/// maps rustc `TyKind` directly to this enum; no debug string or type-name
/// parsing participates in eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum InfluenceTypeClass {
    Boolean,
    Integer,
    /// A raw pointer can carry caller- or boundary-controlled address and
    /// validity bits into an unsafe requirement.
    Pointer,
    /// A reference produced as a safe external-behavior result is already
    /// valid for its referent. Origin-specific construction below keeps
    /// unresolved aggregate projections auditable without calling them an
    /// address violation.
    Reference,
    Callable,
    Sequence,
    Aggregate,
    Opaque,
}

impl InfluenceTypeClass {
    pub(super) fn capabilities(self) -> InfluenceCapabilities {
        let typed = InfluenceCapabilities::TYPED_VALUE | InfluenceCapabilities::VALIDITY_BITS;
        match self {
            Self::Boolean => typed | InfluenceCapabilities::BOOLEAN,
            Self::Integer => typed | InfluenceCapabilities::INDEX_OR_LENGTH,
            Self::Pointer => typed | InfluenceCapabilities::ADDRESS,
            Self::Reference => InfluenceCapabilities::TYPED_VALUE,
            Self::Callable => {
                typed | InfluenceCapabilities::ADDRESS | InfluenceCapabilities::CALLABLE_RESULT
            }
            Self::Sequence => typed | InfluenceCapabilities::INDEX_OR_LENGTH,
            Self::Aggregate => typed,
            Self::Opaque => InfluenceCapabilities::TYPED_VALUE,
        }
    }

    fn precision(self) -> TokenPrecision {
        if self == Self::Opaque {
            TokenPrecision::Incomplete
        } else {
            TokenPrecision::StrongBoundary
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct InfluenceToken {
    pub origin: InfluenceOrigin,
    pub controller: InfluenceController,
    pub capabilities: InfluenceCapabilities,
    pub precision: TokenPrecision,
    pub incomplete_reason: Option<UnestablishedReason>,
}

impl InfluenceToken {
    fn derive_exact(&mut self, capabilities: InfluenceCapabilities) {
        self.capabilities |= capabilities;
        if self.precision != TokenPrecision::Incomplete {
            self.precision = TokenPrecision::ExactDerived;
            self.incomplete_reason = None;
        }
    }

    pub(super) fn ffi(origin: InfluenceOrigin, capabilities: InfluenceCapabilities) -> Self {
        debug_assert!(matches!(
            origin,
            InfluenceOrigin::FfiReturn
                | InfluenceOrigin::FfiOut
                | InfluenceOrigin::FfiCallbackInput
        ));
        Self {
            origin,
            controller: InfluenceController::FfiHost,
            capabilities,
            precision: TokenPrecision::StrongBoundary,
            incomplete_reason: None,
        }
    }

    pub(super) fn from_type_class(
        origin: InfluenceOrigin,
        controller: InfluenceController,
        type_class: InfluenceTypeClass,
    ) -> Self {
        let reference_from_ffi = type_class == InfluenceTypeClass::Reference
            && matches!(
                origin,
                InfluenceOrigin::FfiReturn
                    | InfluenceOrigin::FfiOut
                    | InfluenceOrigin::FfiCallbackInput
            );
        let reference_from_safe_external_behavior = type_class == InfluenceTypeClass::Reference
            && matches!(
                origin,
                InfluenceOrigin::OpenTraitReturn
                    | InfluenceOrigin::PublicCallbackReturn
                    | InfluenceOrigin::PublicFunctionPointerReturn
            );
        let mut capabilities = type_class.capabilities();
        // FFI remains a strong boundary even when the declared Rust type is a
        // reference: foreign code can violate the validity promise before the
        // value enters Rust.  Safe Rust formals and safe trait/callback
        // returns do not receive that exception.
        if reference_from_ffi {
            capabilities = capabilities
                | InfluenceCapabilities::ADDRESS
                | InfluenceCapabilities::VALIDITY_BITS;
        }
        Self {
            origin,
            controller,
            capabilities,
            precision: if reference_from_safe_external_behavior {
                TokenPrecision::Incomplete
            } else {
                type_class.precision()
            },
            incomplete_reason: (type_class == InfluenceTypeClass::Opaque
                || reference_from_safe_external_behavior)
                .then_some(UnestablishedReason::UnknownCapabilityMatch),
        }
    }

    fn join_same_endpoint(&mut self, incoming: &Self) {
        // A source endpoint has one compiler type and one controller.  If a
        // future exporter violates that invariant, intersection plus Unknown
        // remains fail closed instead of manufacturing a capability present
        // in only one branch.
        self.capabilities = self.capabilities & incoming.capabilities;
        self.precision = self.precision.max(incoming.precision);
        self.incomplete_reason = self
            .incomplete_reason
            .into_iter()
            .chain(incoming.incomplete_reason)
            .min();
        if self.origin != incoming.origin || self.controller != incoming.controller {
            self.controller = InfluenceController::Unknown;
            self.precision = TokenPrecision::Incomplete;
            self.incomplete_reason = Some(UnestablishedReason::UnresolvedController);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct InfluenceSeedIndex {
    seeds: BTreeMap<EndpointNode, InfluenceToken>,
}

impl InfluenceSeedIndex {
    pub(super) fn insert(&mut self, endpoint: EndpointNode, seed: InfluenceToken) {
        self.seeds
            .entry(endpoint)
            .and_modify(|current| current.join_same_endpoint(&seed))
            .or_insert(seed);
    }

    pub(super) fn join(&mut self, incoming: &Self) {
        for (endpoint, seed) in &incoming.seeds {
            self.insert(endpoint.clone(), seed.clone());
        }
    }

    pub(super) fn mark_incomplete(&mut self, endpoint: &EndpointNode, reason: UnestablishedReason) {
        if let Some(seed) = self.seeds.get_mut(endpoint) {
            seed.precision = TokenPrecision::Incomplete;
            seed.incomplete_reason = Some(
                seed.incomplete_reason
                    .into_iter()
                    .chain([reason])
                    .min()
                    .expect("an explicit incomplete reason exists"),
            );
        }
    }

    pub(super) fn mark_all_incomplete(&mut self, reason: UnestablishedReason) {
        let endpoints = self.seeds.keys().cloned().collect::<Vec<_>>();
        for endpoint in endpoints {
            self.mark_incomplete(&endpoint, reason);
        }
    }

    pub(super) fn has_capability(&self, capability: InfluenceCapabilities) -> bool {
        self.seeds
            .values()
            .any(|seed| seed.capabilities.contains(capability))
    }

    pub(super) fn derive_all_exact(&mut self, capabilities: InfluenceCapabilities) {
        for seed in self.seeds.values_mut() {
            seed.derive_exact(capabilities);
        }
    }

    pub(super) fn has_exact_derived_capabilities(
        &self,
        endpoint: &EndpointNode,
        capabilities: InfluenceCapabilities,
    ) -> bool {
        self.seeds.get(endpoint).is_some_and(|seed| {
            seed.precision == TokenPrecision::ExactDerived
                && seed.capabilities.contains(capabilities)
        })
    }

    pub(super) fn get(&self, endpoint: &EndpointNode) -> Option<&InfluenceToken> {
        self.seeds.get(endpoint)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&EndpointNode, &InfluenceToken)> {
        self.seeds.iter()
    }

    pub(super) fn len(&self) -> usize {
        self.seeds.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.seeds.is_empty()
    }

    pub(super) fn clear(&mut self) {
        self.seeds.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TokenId(u32);

#[derive(Default)]
pub(super) struct InfluenceArena {
    ids: BTreeMap<Arc<InfluenceToken>, TokenId>,
    values: Vec<Arc<InfluenceToken>>,
}

impl InfluenceArena {
    pub(super) fn intern(&mut self, token: InfluenceToken) -> TokenId {
        if let Some(id) = self.ids.get(&token) {
            return *id;
        }
        let id =
            TokenId(u32::try_from(self.values.len()).expect("influence token arena exceeds u32"));
        let token = Arc::new(token);
        self.values.push(token.clone());
        self.ids.insert(token, id);
        id
    }

    pub(super) fn get(&self, id: TokenId) -> &InfluenceToken {
        &self.values[id.0 as usize]
    }

    pub(super) fn len(&self) -> usize {
        self.values.len()
    }
}

const NEED_ADDRESS: u8 = 1 << 0;
const NEED_INDEX_OR_ADDRESS: u8 = 1 << 1;
const NEED_VALIDITY_BITS: u8 = 1 << 2;
const NEED_TYPED_CONTENT: u8 = 1 << 3;
const NEED_BOOLEAN: u8 = 1 << 4;
const NEED_CALLABLE_RESULT: u8 = 1 << 5;

/// A role-scoped conjunction of finite capability requirements.  The legacy
/// associated constants keep focused solver fixtures concise; production
/// requirements are always built from a compiler-derived contract and carry
/// the exact operand role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SinkRequirement {
    role: Option<OperandRole>,
    needs: u8,
}

fn extent_argument_role(contract: &ContractDescriptor, role: &OperandRole) -> bool {
    let OperandRole::Argument(index) = role else {
        return false;
    };
    match contract.model_id.as_deref() {
        Some("slice_from_raw_parts_v1" | "ptr_offset_v1") => *index == 1,
        Some("vec_from_raw_parts_v1") => *index == 1 || *index == 2,
        Some("ptr_copy_v1") => *index == 2,
        Some("slice_get_unchecked_v1" | "vec_get_unchecked_v1") => *index == 1,
        _ => false,
    }
}

impl SinkRequirement {
    #[allow(non_upper_case_globals)]
    pub(super) const Address: Self = Self {
        role: None,
        needs: NEED_ADDRESS,
    };

    #[allow(non_upper_case_globals)]
    pub(super) const IndexOrAddress: Self = Self {
        role: None,
        needs: NEED_INDEX_OR_ADDRESS,
    };

    #[allow(non_upper_case_globals)]
    pub(super) const ValidityBits: Self = Self {
        role: None,
        needs: NEED_VALIDITY_BITS,
    };

    #[allow(non_upper_case_globals)]
    pub(super) const TypedContent: Self = Self {
        role: None,
        needs: NEED_TYPED_CONTENT,
    };

    #[allow(non_upper_case_globals)]
    pub(super) const Boolean: Self = Self {
        role: None,
        needs: NEED_BOOLEAN,
    };

    #[allow(non_upper_case_globals)]
    pub(super) const CallableResult: Self = Self {
        role: None,
        needs: NEED_CALLABLE_RESULT,
    };

    pub(super) fn from_contract(
        contract: &ContractDescriptor,
        role: &OperandRole,
    ) -> Result<Self, UnestablishedReason> {
        if contract.precision == ContractPrecision::Opaque
            || !contract.relevant_roles.contains(role)
            || contract.predicates.is_empty()
        {
            return Err(UnestablishedReason::UnmodeledUnsafeCall);
        }
        let extent_role = matches!(
            role,
            OperandRole::Index | OperandRole::Length | OperandRole::Size
        ) || extent_argument_role(contract, role);
        let mut needs = 0;
        for predicate in &contract.predicates {
            needs |= match predicate {
                ContractPredicate::NonNull
                | ContractPredicate::Aligned
                | ContractPredicate::LayoutCompatible
                | ContractPredicate::ValidForRead => {
                    if extent_role {
                        0
                    } else {
                        NEED_ADDRESS
                    }
                }
                ContractPredicate::InBounds | ContractPredicate::RangeInBounds => {
                    NEED_INDEX_OR_ADDRESS
                }
                ContractPredicate::Initialized | ContractPredicate::ValidForType => {
                    if extent_role {
                        0
                    } else {
                        NEED_VALIDITY_BITS
                    }
                }
                ContractPredicate::ValidUtf8 => NEED_TYPED_CONTENT,
                ContractPredicate::ConditionTrue => NEED_BOOLEAN,
            };
        }
        Ok(Self {
            role: Some(role.clone()),
            needs,
        })
    }

    pub(super) fn matches(&self, capabilities: InfluenceCapabilities) -> RequirementMatch {
        let satisfies = [
            (NEED_ADDRESS, InfluenceCapabilities::ADDRESS),
            (
                NEED_INDEX_OR_ADDRESS,
                InfluenceCapabilities::INDEX_OR_LENGTH | InfluenceCapabilities::ADDRESS,
            ),
            (NEED_VALIDITY_BITS, InfluenceCapabilities::VALIDITY_BITS),
            (NEED_TYPED_CONTENT, InfluenceCapabilities::TYPED_VALUE),
            (NEED_BOOLEAN, InfluenceCapabilities::BOOLEAN),
            (NEED_CALLABLE_RESULT, InfluenceCapabilities::CALLABLE_RESULT),
        ]
        .into_iter()
        .all(|(need, capability)| self.needs & need == 0 || capabilities.intersects(capability));
        if satisfies {
            RequirementMatch::Compatible
        } else {
            RequirementMatch::ProvenIncompatible
        }
    }

    pub(super) fn stable_identity(&self) -> String {
        let role = self
            .role
            .as_ref()
            .map(operand_role_identity)
            .unwrap_or_else(|| "test:any".to_string());
        let needs = [
            (NEED_ADDRESS, "address"),
            (NEED_INDEX_OR_ADDRESS, "index_or_address"),
            (NEED_VALIDITY_BITS, "validity_bits"),
            (NEED_TYPED_CONTENT, "typed_content"),
            (NEED_BOOLEAN, "boolean"),
            (NEED_CALLABLE_RESULT, "callable_result"),
        ]
        .into_iter()
        .filter_map(|(need, name)| (self.needs & need != 0).then_some(name))
        .collect::<Vec<_>>()
        .join(",");
        format!("{role}:{needs}")
    }
}

fn operand_role_identity(role: &OperandRole) -> String {
    match role {
        OperandRole::Receiver => "receiver".to_string(),
        OperandRole::Argument(index) => format!("argument:{index}"),
        OperandRole::Place => "place".to_string(),
        OperandRole::Address => "address".to_string(),
        OperandRole::Index => "index".to_string(),
        OperandRole::Length => "length".to_string(),
        OperandRole::Size => "size".to_string(),
        OperandRole::Alignment => "alignment".to_string(),
        OperandRole::Layout => "layout".to_string(),
        OperandRole::Discriminant => "discriminant".to_string(),
        OperandRole::Out => "out".to_string(),
        OperandRole::Callee => "callee".to_string(),
        OperandRole::Other(index) => format!("other:{index}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RequirementMatch {
    Compatible,
    ProvenIncompatible,
    Unknown,
}

/// Declaration order is the stable primary-reason precedence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum UnestablishedReason {
    MissingExactContext,
    IncompleteNormalization,
    UnresolvedController,
    UnmodeledUnsafeCall,
    UnknownCapabilityMatch,
    OpaqueTransform,
    HavocedProvenance,
    ValidationStateUnknown,
    ValidityObligationUnknown,
    InitializationStateUnknown,
    DerivationObligationUnknown,
    GenericWitnessUnknown,
}

impl UnestablishedReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::MissingExactContext => "missing_exact_context",
            Self::IncompleteNormalization => "incomplete_normalization",
            Self::UnresolvedController => "unresolved_controller",
            Self::UnmodeledUnsafeCall => "unmodeled_unsafe_call",
            Self::UnknownCapabilityMatch => "unknown_capability_match",
            Self::OpaqueTransform => "opaque_transform",
            Self::HavocedProvenance => "havoced_provenance",
            Self::ValidationStateUnknown => "validation_state_unknown",
            Self::ValidityObligationUnknown => "validity_obligation_unknown",
            Self::InitializationStateUnknown => "initialization_state_unknown",
            Self::DerivationObligationUnknown => "derivation_obligation_unknown",
            Self::GenericWitnessUnknown => "generic_witness_unknown",
        }
    }
}

pub(super) fn primary_unestablished_reason(
    reasons: &BTreeSet<UnestablishedReason>,
) -> Option<UnestablishedReason> {
    reasons.iter().next().copied()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InfluenceProofStepKind {
    Local,
    Call,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InfluenceProofStep {
    pub relation: ContextRelationHandle,
    pub predecessor: ContextPredecessorHandle,
    pub kind: InfluenceProofStepKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct InfluenceWitness {
    pub token: InfluenceToken,
    pub steps: Vec<InfluenceProofStep>,
}

impl InfluenceWitness {
    pub(super) fn call_boundary_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| step.kind == InfluenceProofStepKind::Call)
            .count()
    }

    pub(super) fn endpoint_route(
        &self,
        context: &ContextProofIndex,
    ) -> Vec<super::endpoint_flow::EndpointEdge> {
        self.steps
            .iter()
            .map(|step| context.materialize_exact_step(step.relation, step.predecessor))
            .collect()
    }

    pub(super) fn call_route(
        &self,
        context: &ContextProofIndex,
    ) -> Vec<(
        super::endpoint_flow::CallSiteId,
        super::endpoint_flow::GenericCallSubstitution,
    )> {
        self.steps
            .iter()
            .filter_map(|step| match context.exact_predecessor(step.predecessor) {
                ExactPredecessorRef::Call {
                    callsite,
                    substitution,
                    ..
                } => Some((callsite.clone(), substitution.clone())),
                ExactPredecessorRef::Local { .. } => None,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum InfluenceOutcome {
    Active(InfluenceWitness),
    ProvenIncompatible(InfluenceWitness),
    Unestablished {
        reasons: BTreeSet<UnestablishedReason>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct InfluenceRequest {
    start: ContextAtomHandle,
    target: ContextAtomHandle,
    seed: TokenId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InfluenceState {
    request: InfluenceRequest,
    at: ContextAtomHandle,
    token: TokenId,
    proof: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct InfluenceWaiter {
    parent: InfluenceRequest,
    next: ContextAtomHandle,
    parent_proof: usize,
    relation: ContextRelationHandle,
    predecessor: ContextPredecessorHandle,
}

#[derive(Clone, Copy, Debug)]
enum InfluenceProof {
    Root,
    Local {
        parent: usize,
        relation: ContextRelationHandle,
        predecessor: ContextPredecessorHandle,
    },
    Call {
        parent: usize,
        child: usize,
        relation: ContextRelationHandle,
        predecessor: ContextPredecessorHandle,
    },
}

/// Finite exact product fixed point over `(scoped atom, influence token)`.
/// The context index owns all heavyweight endpoint and call payloads; this
/// solver retains only compact handles and materializes one selected witness.
pub(super) struct InfluenceSolver<'context> {
    context: &'context ContextProofIndex,
    arena: InfluenceArena,
    proofs: Vec<InfluenceProof>,
    started: HashSet<InfluenceRequest>,
    pending: VecDeque<InfluenceState>,
    seen: HashSet<(InfluenceRequest, ContextAtomHandle, TokenId)>,
    facts: HashMap<InfluenceRequest, BTreeMap<TokenId, usize>>,
    waiters: HashMap<InfluenceRequest, BTreeSet<InfluenceWaiter>>,
    reasons: BTreeSet<UnestablishedReason>,
}

impl<'context> InfluenceSolver<'context> {
    pub(super) fn new(context: &'context ContextProofIndex) -> Self {
        Self {
            context,
            arena: InfluenceArena::default(),
            proofs: vec![InfluenceProof::Root],
            started: HashSet::new(),
            pending: VecDeque::new(),
            seen: HashSet::new(),
            facts: HashMap::new(),
            waiters: HashMap::new(),
            reasons: BTreeSet::new(),
        }
    }

    pub(super) fn solve(
        self,
        source_context: FunctionKey,
        source: &EndpointNode,
        target_context: FunctionKey,
        target: &EndpointNode,
        seed: InfluenceToken,
        requirement: &SinkRequirement,
    ) -> InfluenceOutcome {
        let reverse = self.context.exact_reverse_reachability(target);
        self.solve_in_reverse(
            source_context,
            source,
            target_context,
            target,
            seed,
            requirement,
            &reverse,
        )
    }

    pub(super) fn solve_in_reverse(
        self,
        source_context: FunctionKey,
        source: &EndpointNode,
        target_context: FunctionKey,
        target: &EndpointNode,
        seed: InfluenceToken,
        requirement: &SinkRequirement,
        reverse: &ExactReverseReachability,
    ) -> InfluenceOutcome {
        let Some(start) = self
            .context
            .unique_exact_atom_handle(source, &source_context)
        else {
            return InfluenceOutcome::Unestablished {
                reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
            };
        };
        let Some(target) = reverse.target_in_owner(&target_context) else {
            return InfluenceOutcome::Unestablished {
                reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
            };
        };
        if !reverse.contains(start) {
            return InfluenceOutcome::Unestablished {
                reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
            };
        }
        self.solve_handles(start, target, seed, requirement, reverse)
    }

    pub(super) fn solve_handles(
        mut self,
        start: ContextAtomHandle,
        target: ContextAtomHandle,
        seed: InfluenceToken,
        requirement: &SinkRequirement,
        reverse: &ExactReverseReachability,
    ) -> InfluenceOutcome {
        let seed = self.arena.intern(seed);
        let request = InfluenceRequest {
            start,
            target,
            seed,
        };
        self.start(request);
        while let Some(current) = self.pending.pop_front() {
            if current.at == current.request.target {
                self.finish_fact(current.request, current.token, current.proof);
                continue;
            }
            let relations = self.context.exact_relation_handles_from(current.at);
            for relation in relations {
                let next = self.context.exact_relation_target(relation);
                if !reverse.contains(next) {
                    continue;
                }
                for predecessor in self.context.exact_predecessor_handles(relation) {
                    match self.context.exact_predecessor(predecessor) {
                        ExactPredecessorRef::Local { kind, .. } => {
                            self.advance_local(&current, next, relation, predecessor, kind);
                        }
                        ExactPredecessorRef::Call {
                            callsite,
                            callee_relation,
                            ..
                        } => {
                            for reason in self.context.unknown_reasons_for_callsite(callsite) {
                                match reason {
                                    ContextUnknownReason::NormalizationIncomplete { .. } => {
                                        self.reasons
                                            .insert(UnestablishedReason::IncompleteNormalization);
                                    }
                                    ContextUnknownReason::MissingActualEndpoint { .. }
                                    | ContextUnknownReason::MissingResolvedBody { .. }
                                    | ContextUnknownReason::UncontextualizedCallback
                                    | ContextUnknownReason::CompletenessUnavailable => {
                                        self.reasons
                                            .insert(UnestablishedReason::MissingExactContext);
                                    }
                                }
                            }
                            let child = InfluenceRequest {
                                start: self.context.exact_relation_source(callee_relation),
                                target: self.context.exact_relation_target(callee_relation),
                                seed: current.token,
                            };
                            if !reverse.contains(child.target) {
                                continue;
                            }
                            let waiter = InfluenceWaiter {
                                parent: current.request,
                                next,
                                parent_proof: current.proof,
                                relation,
                                predecessor,
                            };
                            self.waiters.entry(child).or_default().insert(waiter);
                            let existing = self
                                .facts
                                .get(&child)
                                .into_iter()
                                .flat_map(|facts| facts.iter())
                                .map(|(token, proof)| (*token, *proof))
                                .collect::<Vec<_>>();
                            for (token, proof) in existing {
                                self.resume_waiter(waiter, token, proof);
                            }
                            self.start(child);
                        }
                    }
                }
            }
        }

        let terminal = self.facts.remove(&request).unwrap_or_default();
        let mut compatible = Vec::new();
        let mut incompatible = Vec::new();
        for (token, proof) in terminal {
            let value = self.arena.get(token).clone();
            if value.precision == TokenPrecision::Incomplete {
                self.reasons.insert(
                    value
                        .incomplete_reason
                        .unwrap_or(UnestablishedReason::UnknownCapabilityMatch),
                );
                continue;
            }
            if matches!(value.controller, InfluenceController::Unknown)
                || matches!(
                    value.controller,
                    InfluenceController::OpenTrait {
                        receiver: OpenBehaviorReceiver::Unknown,
                        ..
                    }
                )
            {
                self.reasons
                    .insert(UnestablishedReason::UnresolvedController);
                continue;
            }
            let witness = InfluenceWitness {
                token: value,
                steps: self.render_proof(proof),
            };
            if matches!(witness.token.controller, InfluenceController::Internal) {
                incompatible.push(witness);
                continue;
            }
            match requirement.matches(witness.token.capabilities) {
                RequirementMatch::Compatible => compatible.push(witness),
                RequirementMatch::ProvenIncompatible => incompatible.push(witness),
                RequirementMatch::Unknown => {
                    self.reasons
                        .insert(UnestablishedReason::UnknownCapabilityMatch);
                }
            }
        }
        if let Some(witness) = compatible.into_iter().next() {
            return InfluenceOutcome::Active(witness);
        }
        if !self.reasons.is_empty() {
            return InfluenceOutcome::Unestablished {
                reasons: self.reasons,
            };
        }
        if let Some(witness) = incompatible.into_iter().next() {
            return InfluenceOutcome::ProvenIncompatible(witness);
        }
        InfluenceOutcome::Unestablished {
            reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
        }
    }

    fn start(&mut self, request: InfluenceRequest) {
        if !self.started.insert(request)
            || !self.seen.insert((request, request.start, request.seed))
        {
            return;
        }
        self.pending.push_back(InfluenceState {
            request,
            at: request.start,
            token: request.seed,
            proof: 0,
        });
    }

    fn new_proof(&mut self, proof: InfluenceProof) -> usize {
        let id = self.proofs.len();
        self.proofs.push(proof);
        id
    }

    fn advance_local(
        &mut self,
        current: &InfluenceState,
        next: ContextAtomHandle,
        relation: ContextRelationHandle,
        predecessor: ContextPredecessorHandle,
        kind: super::endpoint_flow::EndpointEdgeKind,
    ) {
        let token = match kind {
            super::endpoint_flow::EndpointEdgeKind::TypedValidityPointer => {
                let mut token = self.arena.get(current.token).clone();
                token.derive_exact(
                    InfluenceCapabilities::ADDRESS | InfluenceCapabilities::VALIDITY_BITS,
                );
                self.arena.intern(token)
            }
            super::endpoint_flow::EndpointEdgeKind::PrivateAggregateTransform => {
                let mut token = self.arena.get(current.token).clone();
                if matches!(
                    token.origin,
                    InfluenceOrigin::PublicFormal | InfluenceOrigin::PersistentPublicField
                ) {
                    token.precision = TokenPrecision::Incomplete;
                    token.incomplete_reason = Some(UnestablishedReason::ValidationStateUnknown);
                    self.reasons
                        .insert(UnestablishedReason::ValidationStateUnknown);
                    self.arena.intern(token)
                } else {
                    current.token
                }
            }
            _ => current.token,
        };
        if !self.seen.insert((current.request, next, token)) {
            return;
        }
        let proof = self.new_proof(InfluenceProof::Local {
            parent: current.proof,
            relation,
            predecessor,
        });
        self.pending.push_back(InfluenceState {
            request: current.request,
            at: next,
            token,
            proof,
        });
    }

    fn finish_fact(&mut self, request: InfluenceRequest, token: TokenId, proof: usize) {
        let facts = self.facts.entry(request).or_default();
        if facts.contains_key(&token) {
            return;
        }
        facts.insert(token, proof);
        let waiters = self
            .waiters
            .get(&request)
            .into_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        for waiter in waiters {
            self.resume_waiter(waiter, token, proof);
        }
    }

    fn resume_waiter(&mut self, waiter: InfluenceWaiter, child: TokenId, child_proof: usize) {
        let token = self.project_through_boundary(child, waiter.predecessor);
        if !self.seen.insert((waiter.parent, waiter.next, token)) {
            return;
        }
        let proof = self.new_proof(InfluenceProof::Call {
            parent: waiter.parent_proof,
            child: child_proof,
            relation: waiter.relation,
            predecessor: waiter.predecessor,
        });
        self.pending.push_back(InfluenceState {
            request: waiter.parent,
            at: waiter.next,
            token,
            proof,
        });
    }

    fn project_through_boundary(
        &mut self,
        child: TokenId,
        predecessor: ContextPredecessorHandle,
    ) -> TokenId {
        let ExactPredecessorRef::Call {
            substitution,
            actual_binding,
            ..
        } = self.context.exact_predecessor(predecessor)
        else {
            unreachable!("only call predecessors resume influence waiters");
        };
        let mut token = self.arena.get(child).clone();
        if substitution.status == GenericSubstitutionStatus::NormalizationIncomplete
            && actual_binding == ContextActualBinding::Internal
            && matches!(token.controller, InfluenceController::OpenTrait { .. })
        {
            token.precision = TokenPrecision::Incomplete;
            token.incomplete_reason = Some(UnestablishedReason::IncompleteNormalization);
            self.reasons
                .insert(UnestablishedReason::IncompleteNormalization);
        }
        match actual_binding {
            ContextActualBinding::Formal(_) => {}
            ContextActualBinding::Internal => {
                token.controller =
                    project_controller_through_substitution(&token.controller, substitution);
                if matches!(token.controller, InfluenceController::Unknown) {
                    self.reasons
                        .insert(UnestablishedReason::UnresolvedController);
                }
            }
        }
        self.arena.intern(token)
    }

    fn render_proof(&self, proof: usize) -> Vec<InfluenceProofStep> {
        match self.proofs[proof] {
            InfluenceProof::Root => Vec::new(),
            InfluenceProof::Local {
                parent,
                relation,
                predecessor,
            } => {
                let mut steps = self.render_proof(parent);
                steps.push(InfluenceProofStep {
                    relation,
                    predecessor,
                    kind: InfluenceProofStepKind::Local,
                });
                steps
            }
            InfluenceProof::Call {
                parent,
                child,
                relation,
                predecessor,
            } => {
                let mut steps = self.render_proof(parent);
                steps.extend(self.render_proof(child));
                steps.push(InfluenceProofStep {
                    relation,
                    predecessor,
                    kind: InfluenceProofStepKind::Call,
                });
                steps
            }
        }
    }
}

fn project_controller_through_substitution(
    controller: &InfluenceController,
    substitution: &super::endpoint_flow::GenericCallSubstitution,
) -> InfluenceController {
    let InfluenceController::OpenTrait { trait_id, receiver } = controller else {
        return controller.clone();
    };
    let OpenBehaviorReceiver::Parameter(parameter) = receiver else {
        return controller.clone();
    };
    if substitution.status == GenericSubstitutionStatus::NormalizationIncomplete {
        return InfluenceController::Unknown;
    }
    let Some(binding) = substitution.bindings.get(parameter) else {
        return InfluenceController::Unknown;
    };
    if !binding.residual_aliases.is_empty() {
        return InfluenceController::Unknown;
    }
    match &binding.behavior_receiver {
        Some(OpenBehaviorReceiver::Parameter(parameter)) => InfluenceController::OpenTrait {
            trait_id: trait_id.clone(),
            receiver: OpenBehaviorReceiver::Parameter(parameter.clone()),
        },
        Some(OpenBehaviorReceiver::Dynamic) => InfluenceController::OpenTrait {
            trait_id: trait_id.clone(),
            receiver: OpenBehaviorReceiver::Dynamic,
        },
        Some(OpenBehaviorReceiver::Alias(parameters)) if !parameters.is_empty() => {
            InfluenceController::OpenTrait {
                trait_id: trait_id.clone(),
                receiver: OpenBehaviorReceiver::Alias(parameters.clone()),
            }
        }
        Some(OpenBehaviorReceiver::Alias(_)) | Some(OpenBehaviorReceiver::Unknown) => {
            InfluenceController::Unknown
        }
        None if binding.form == super::endpoint_flow::CapabilityTermForm::Concrete => {
            InfluenceController::Internal
        }
        None => InfluenceController::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::analysis::unsoundaudit::endpoint_flow::{
        CallSiteId, CallSiteOutput, CapabilityTermForm, CapabilityTypeFacts, ContextActualBinding,
        ContextProofIndex, ContextRelationKey, EndpointEdge, EndpointEdgeKind, EndpointNode,
        GenericArgBinding, GenericCallSubstitution, GenericParamKind, GenericSubstitutionStatus,
    };
    use crate::analysis::unsoundaudit::summary::{
        FunctionKey, ProgramPoint, StablePosition, StableSpan,
    };

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

    fn local_edge(from: EndpointNode, to: EndpointNode, block: u32) -> EndpointEdge {
        let owner = match &from {
            EndpointNode::Formal { owner, .. }
            | EndpointNode::ExternalBehavior { owner, .. }
            | EndpointNode::Return { owner }
            | EndpointNode::UnsafeOperand { owner, .. } => owner.clone(),
            EndpointNode::CallResult { callsite, .. } => callsite.caller.clone(),
            other => panic!("unexpected test endpoint {other:?}"),
        };
        EndpointEdge {
            from,
            to,
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(&owner, block)),
        }
    }

    fn ffi_address_token() -> InfluenceToken {
        InfluenceToken::ffi(
            InfluenceOrigin::FfiReturn,
            InfluenceCapabilities::ADDRESS | InfluenceCapabilities::VALIDITY_BITS,
        )
    }

    fn type_binding(
        form: CapabilityTermForm,
        receiver: Option<OpenBehaviorReceiver>,
    ) -> GenericArgBinding {
        GenericArgBinding {
            kind: GenericParamKind::Type,
            form,
            caller_controllers: BTreeSet::new(),
            residual_aliases: BTreeSet::new(),
            behavior_receiver: receiver,
            equality_class: Some(0),
            type_facts: CapabilityTypeFacts::default(),
            const_value: None,
        }
    }

    #[test]
    fn influence_controller_projects_direct_dynamic_and_concrete_bindings() {
        let child = GenericParamId {
            def_path: "crate::helper::P".into(),
            kind: GenericParamKind::Type,
        };
        let caller = GenericParamId {
            def_path: "crate::root::T".into(),
            kind: GenericParamKind::Type,
        };
        let controller = InfluenceController::OpenTrait {
            trait_id: FunctionKey::new("crate::OpenTrait"),
            receiver: OpenBehaviorReceiver::Parameter(child.clone()),
        };

        let direct = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([(
                child.clone(),
                type_binding(
                    CapabilityTermForm::DirectParam,
                    Some(OpenBehaviorReceiver::Parameter(caller.clone())),
                ),
            )]),
        };
        assert_eq!(
            project_controller_through_substitution(&controller, &direct),
            InfluenceController::OpenTrait {
                trait_id: FunctionKey::new("crate::OpenTrait"),
                receiver: OpenBehaviorReceiver::Parameter(caller),
            }
        );

        let dynamic = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([(
                child.clone(),
                type_binding(
                    CapabilityTermForm::Concrete,
                    Some(OpenBehaviorReceiver::Dynamic),
                ),
            )]),
        };
        assert!(matches!(
            project_controller_through_substitution(&controller, &dynamic),
            InfluenceController::OpenTrait {
                receiver: OpenBehaviorReceiver::Dynamic,
                ..
            }
        ));

        let concrete = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([(child, type_binding(CapabilityTermForm::Concrete, None))]),
        };
        assert_eq!(
            project_controller_through_substitution(&controller, &concrete),
            InfluenceController::Internal,
        );
    }

    #[test]
    fn influence_controller_fails_closed_for_alias_and_incomplete_transport() {
        let child = GenericParamId {
            def_path: "crate::helper::P".into(),
            kind: GenericParamKind::Type,
        };
        let controller = InfluenceController::OpenTrait {
            trait_id: FunctionKey::new("crate::OpenTrait"),
            receiver: OpenBehaviorReceiver::Parameter(child.clone()),
        };
        let mut alias = type_binding(
            CapabilityTermForm::Alias,
            Some(OpenBehaviorReceiver::Alias(BTreeSet::from([child.clone()]))),
        );
        alias.residual_aliases.insert("crate::Trait::Assoc".into());
        assert_eq!(
            project_controller_through_substitution(
                &controller,
                &GenericCallSubstitution {
                    status: GenericSubstitutionStatus::Complete,
                    bindings: BTreeMap::from([(child.clone(), alias)]),
                },
            ),
            InfluenceController::Unknown,
        );
        assert_eq!(
            project_controller_through_substitution(
                &controller,
                &GenericCallSubstitution {
                    status: GenericSubstitutionStatus::NormalizationIncomplete,
                    bindings: BTreeMap::new(),
                },
            ),
            InfluenceController::Unknown,
        );
        assert_eq!(
            project_controller_through_substitution(
                &InfluenceController::FfiHost,
                &GenericCallSubstitution {
                    status: GenericSubstitutionStatus::NormalizationIncomplete,
                    bindings: BTreeMap::new(),
                },
            ),
            InfluenceController::FfiHost,
        );
    }

    fn solve_open_controller_boundary(
        parameter: GenericParamId,
        substitution: GenericCallSubstitution,
    ) -> InfluenceOutcome {
        let caller = FunctionKey::new("crate::root");
        let callee = FunctionKey::new("crate::helper");
        let source = EndpointNode::ExternalBehavior {
            owner: callee.clone(),
            kind: super::super::endpoint_flow::ExternalSourceKind::OpenTrait,
            controller: super::super::endpoint_flow::BehaviorController::Unknown,
            token: "trait-return".into(),
            span: point(&callee, 1).span,
        };
        let callee_return = EndpointNode::Return {
            owner: callee.clone(),
        };
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let result = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let callee_relation =
            ContextRelationKey::new(callee.clone(), source.clone(), callee_return.clone());
        let mut context = ContextProofIndex::default();
        context.record_local(local_edge(source.clone(), callee_return, 1));
        context.record_call(
            EndpointEdge {
                from: source.clone(),
                to: result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite,
            substitution,
            callee_relation,
            ContextActualBinding::Internal,
        );
        context.finalize();
        InfluenceSolver::new(&context).solve(
            caller.clone(),
            &source,
            caller,
            &result,
            InfluenceToken::from_type_class(
                InfluenceOrigin::OpenTraitReturn,
                InfluenceController::OpenTrait {
                    trait_id: FunctionKey::new("crate::OpenTrait"),
                    receiver: OpenBehaviorReceiver::Parameter(parameter),
                },
                InfluenceTypeClass::Pointer,
            ),
            &SinkRequirement::Address,
        )
    }

    #[test]
    fn influence_controller_exact_boundary_distinguishes_generic_concrete_and_unknown() {
        let child = GenericParamId {
            def_path: "crate::helper::P".into(),
            kind: GenericParamKind::Type,
        };
        let caller = GenericParamId {
            def_path: "crate::root::T".into(),
            kind: GenericParamKind::Type,
        };
        let direct = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([(
                child.clone(),
                type_binding(
                    CapabilityTermForm::DirectParam,
                    Some(OpenBehaviorReceiver::Parameter(caller)),
                ),
            )]),
        };
        assert!(matches!(
            solve_open_controller_boundary(child.clone(), direct),
            InfluenceOutcome::Active(_)
        ));

        let concrete = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([(
                child.clone(),
                type_binding(CapabilityTermForm::Concrete, None),
            )]),
        };
        assert!(matches!(
            solve_open_controller_boundary(child.clone(), concrete),
            InfluenceOutcome::ProvenIncompatible(_)
        ));

        assert!(matches!(
            solve_open_controller_boundary(
                child,
                GenericCallSubstitution {
                    status: GenericSubstitutionStatus::NormalizationIncomplete,
                    bindings: BTreeMap::new(),
                },
            ),
            InfluenceOutcome::Unestablished { reasons }
                if reasons.contains(&UnestablishedReason::IncompleteNormalization)
                    && reasons.contains(&UnestablishedReason::UnresolvedController)
        ));
    }

    #[test]
    fn boolean_does_not_satisfy_address_requirement() {
        let token = InfluenceCapabilities::BOOLEAN | InfluenceCapabilities::TYPED_VALUE;
        assert_eq!(
            SinkRequirement::Address.matches(token),
            RequirementMatch::ProvenIncompatible,
        );
    }

    #[test]
    fn influence_requirement_opaque_contract_without_typed_requirement_is_unestablished() {
        let role = OperandRole::Argument(0);
        let contract = ContractDescriptor {
            precision: ContractPrecision::Opaque,
            model_id: None,
            relevant_roles: BTreeSet::from([role.clone()]),
            predicates: BTreeSet::new(),
            failure_phase: super::super::contract_catalog::FailurePhase::Precondition,
            discharge: super::super::contract_catalog::DischargeModel::None,
            static_implication:
                super::super::contract_catalog::StaticContractImplication::NotApplicable,
        };
        assert_eq!(
            SinkRequirement::from_contract(&contract, &role),
            Err(UnestablishedReason::UnmodeledUnsafeCall),
        );
    }

    #[test]
    fn influence_requirement_known_predicates_form_a_role_scoped_conjunction() {
        let role = OperandRole::Argument(0);
        let contract = ContractDescriptor {
            precision: ContractPrecision::Precise,
            model_id: Some("ptr_read_v1".to_string()),
            relevant_roles: BTreeSet::from([role.clone()]),
            predicates: BTreeSet::from([
                ContractPredicate::ValidForRead,
                ContractPredicate::RangeInBounds,
                ContractPredicate::Initialized,
                ContractPredicate::ValidForType,
            ]),
            failure_phase: super::super::contract_catalog::FailurePhase::Precondition,
            discharge: super::super::contract_catalog::DischargeModel::None,
            static_implication:
                super::super::contract_catalog::StaticContractImplication::NotApplicable,
        };
        let requirement = SinkRequirement::from_contract(&contract, &role)
            .expect("compiler-identified contract has a finite requirement");
        assert_eq!(
            requirement.stable_identity(),
            "argument:0:address,index_or_address,validity_bits",
        );
        assert_eq!(
            requirement
                .matches(InfluenceCapabilities::ADDRESS | InfluenceCapabilities::VALIDITY_BITS),
            RequirementMatch::Compatible,
        );
        assert_eq!(
            requirement
                .matches(InfluenceCapabilities::BOOLEAN | InfluenceCapabilities::VALIDITY_BITS),
            RequirementMatch::ProvenIncompatible,
        );
    }

    #[test]
    fn influence_requirement_rejects_a_predicate_from_another_operand_role() {
        let contract = ContractDescriptor {
            precision: ContractPrecision::Generic,
            model_id: Some("raw_pointer_access_v1".to_string()),
            relevant_roles: BTreeSet::from([OperandRole::Address]),
            predicates: BTreeSet::from([ContractPredicate::ValidForRead]),
            failure_phase: super::super::contract_catalog::FailurePhase::Precondition,
            discharge: super::super::contract_catalog::DischargeModel::None,
            static_implication:
                super::super::contract_catalog::StaticContractImplication::NotApplicable,
        };
        assert_eq!(
            SinkRequirement::from_contract(&contract, &OperandRole::Argument(0)),
            Err(UnestablishedReason::UnmodeledUnsafeCall),
        );
    }

    #[test]
    fn token_interner_deduplicates_complete_semantic_values() {
        let mut arena = InfluenceArena::default();
        let token = InfluenceToken::ffi(
            InfluenceOrigin::FfiReturn,
            InfluenceCapabilities::ADDRESS | InfluenceCapabilities::VALIDITY_BITS,
        );
        assert_eq!(arena.intern(token.clone()), arena.intern(token));
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn unestablished_reason_precedence_is_stable() {
        let reasons = BTreeSet::from([
            UnestablishedReason::OpaqueTransform,
            UnestablishedReason::UnresolvedController,
            UnestablishedReason::MissingExactContext,
        ]);
        assert_eq!(
            primary_unestablished_reason(&reasons),
            Some(UnestablishedReason::MissingExactContext),
        );
    }

    #[test]
    fn ffi_token_survives_two_exact_call_boundaries() {
        let caller = FunctionKey::new("crate::caller");
        let first_callee = FunctionKey::new("crate::first");
        let second_callee = FunctionKey::new("crate::second");
        let source = formal(&caller, 1);
        let first_callsite =
            CallSiteId::new(caller.clone(), point(&caller, 10), first_callee.clone());
        let first_result = EndpointNode::CallResult {
            callsite: first_callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let second_callsite =
            CallSiteId::new(caller.clone(), point(&caller, 20), second_callee.clone());
        let second_result = EndpointNode::CallResult {
            callsite: second_callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let sink = formal(&caller, 9);
        let first_relation = ContextRelationKey::new(
            first_callee.clone(),
            formal(&first_callee, 1),
            EndpointNode::Return {
                owner: first_callee.clone(),
            },
        );
        let second_relation = ContextRelationKey::new(
            second_callee.clone(),
            formal(&second_callee, 1),
            EndpointNode::Return {
                owner: second_callee.clone(),
            },
        );
        let mut context = ContextProofIndex::default();
        context.record_local(local_edge(
            first_relation.source.clone(),
            first_relation.target.clone(),
            1,
        ));
        context.record_local(local_edge(
            second_relation.source.clone(),
            second_relation.target.clone(),
            2,
        ));
        context.record_call(
            EndpointEdge {
                from: source.clone(),
                to: first_result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(first_callsite.point.clone()),
            },
            first_callsite,
            GenericCallSubstitution::default(),
            first_relation,
            ContextActualBinding::Formal(1),
        );
        context.record_call(
            EndpointEdge {
                from: first_result,
                to: second_result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(second_callsite.point.clone()),
            },
            second_callsite,
            GenericCallSubstitution::default(),
            second_relation,
            ContextActualBinding::Formal(1),
        );
        context.record_local(local_edge(second_result, sink.clone(), 21));
        context.finalize();

        let outcome = InfluenceSolver::new(&context).solve(
            caller.clone(),
            &source,
            caller,
            &sink,
            ffi_address_token(),
            &SinkRequirement::Address,
        );
        let InfluenceOutcome::Active(witness) = outcome else {
            panic!("two exact call boundaries must preserve an FFI address token");
        };
        assert_eq!(witness.token, ffi_address_token());
        assert_eq!(witness.call_boundary_count(), 2);
    }

    #[test]
    fn exact_token_solver_does_not_splice_sibling_callsites() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let source_a = formal(&caller, 1);
        let source_b = formal(&caller, 2);
        let call_a = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let call_b = CallSiteId::new(caller.clone(), point(&caller, 20), callee.clone());
        let result_a = EndpointNode::CallResult {
            callsite: call_a.clone(),
            output: CallSiteOutput::Return,
        };
        let result_b = EndpointNode::CallResult {
            callsite: call_b.clone(),
            output: CallSiteOutput::Return,
        };
        let sink_b = formal(&caller, 9);
        let callee_relation = ContextRelationKey::new(
            callee.clone(),
            formal(&callee, 1),
            EndpointNode::Return {
                owner: callee.clone(),
            },
        );
        let mut context = ContextProofIndex::default();
        context.record_local(local_edge(
            callee_relation.source.clone(),
            callee_relation.target.clone(),
            1,
        ));
        for (source, result, callsite) in [
            (source_a.clone(), result_a, call_a),
            (source_b, result_b.clone(), call_b),
        ] {
            context.record_call(
                EndpointEdge {
                    from: source,
                    to: result,
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(callsite.point.clone()),
                },
                callsite,
                GenericCallSubstitution::default(),
                callee_relation.clone(),
                ContextActualBinding::Formal(1),
            );
        }
        context.record_local(local_edge(result_b, sink_b.clone(), 21));
        context.finalize();

        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                caller.clone(),
                &source_a,
                caller,
                &sink_b,
                ffi_address_token(),
                &SinkRequirement::Address,
            ),
            InfluenceOutcome::Unestablished { reasons }
                if reasons.contains(&UnestablishedReason::MissingExactContext)
        ));
    }

    #[test]
    fn typed_validity_pointer_derivation_is_local_to_the_selected_corridor() {
        let owner = FunctionKey::new("crate::maybe_uninit_view");
        let source = formal(&owner, 1);
        let derived_target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: "derived".into(),
            role: OperandRole::Address,
        };
        let sibling_target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: "sibling".into(),
            role: OperandRole::Address,
        };
        let mut context = ContextProofIndex::default();
        context.record_local(EndpointEdge {
            from: source.clone(),
            to: derived_target.clone(),
            kind: EndpointEdgeKind::TypedValidityPointer,
            point: Some(point(&owner, 1)),
        });
        context.record_local(EndpointEdge {
            from: source.clone(),
            to: sibling_target.clone(),
            kind: EndpointEdgeKind::UnsafeOperand,
            point: Some(point(&owner, 2)),
        });
        context.finalize();
        let seed = InfluenceToken::from_type_class(
            InfluenceOrigin::PublicFormal,
            InfluenceController::PublicFormal {
                owner: owner.clone(),
                index: 1,
            },
            InfluenceTypeClass::Reference,
        );

        let derived = InfluenceSolver::new(&context).solve(
            owner.clone(),
            &source,
            owner.clone(),
            &derived_target,
            seed.clone(),
            &SinkRequirement::Address,
        );
        assert!(matches!(
            derived,
            InfluenceOutcome::Active(InfluenceWitness { token, .. })
                if token.precision == TokenPrecision::ExactDerived
                    && token.capabilities.intersects(InfluenceCapabilities::ADDRESS)
                    && token.capabilities.intersects(InfluenceCapabilities::VALIDITY_BITS)
        ));

        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                owner.clone(),
                &source,
                owner,
                &sibling_target,
                seed,
                &SinkRequirement::Address,
            ),
            InfluenceOutcome::ProvenIncompatible(_)
        ));
    }

    #[test]
    fn private_aggregate_transform_is_unestablished_for_p1_but_preserves_ffi() {
        let owner = FunctionKey::new("crate::private_range");
        let source = formal(&owner, 1);
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: "range_sink".into(),
            role: OperandRole::Argument(1),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(EndpointEdge {
            from: source.clone(),
            to: target.clone(),
            kind: EndpointEdgeKind::PrivateAggregateTransform,
            point: Some(point(&owner, 1)),
        });
        context.finalize();
        let public = InfluenceToken::from_type_class(
            InfluenceOrigin::PublicFormal,
            InfluenceController::PublicFormal {
                owner: owner.clone(),
                index: 1,
            },
            InfluenceTypeClass::Integer,
        );
        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                owner.clone(),
                &source,
                owner.clone(),
                &target,
                public,
                &SinkRequirement::IndexOrAddress,
            ),
            InfluenceOutcome::Unestablished { reasons }
                if reasons == BTreeSet::from([UnestablishedReason::ValidationStateUnknown])
        ));
        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                owner.clone(),
                &source,
                owner,
                &target,
                InfluenceToken::ffi(
                    InfluenceOrigin::FfiReturn,
                    InfluenceCapabilities::INDEX_OR_LENGTH,
                ),
                &SinkRequirement::IndexOrAddress,
            ),
            InfluenceOutcome::Active(_)
        ));
    }

    #[test]
    fn complete_call_alternative_wins_over_incomplete_sibling() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::callee");
        let source = formal(&caller, 1);
        let sink = formal(&caller, 9);
        let callee_relation = ContextRelationKey::new(
            callee.clone(),
            formal(&callee, 1),
            EndpointNode::Return {
                owner: callee.clone(),
            },
        );
        let mut context = ContextProofIndex::default();
        context.record_local(local_edge(
            callee_relation.source.clone(),
            callee_relation.target.clone(),
            1,
        ));
        for (block, status) in [
            (20, GenericSubstitutionStatus::NormalizationIncomplete),
            (10, GenericSubstitutionStatus::Complete),
        ] {
            let callsite = CallSiteId::new(caller.clone(), point(&caller, block), callee.clone());
            context.record_call(
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
        context.finalize();

        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                caller.clone(),
                &source,
                caller,
                &sink,
                ffi_address_token(),
                &SinkRequirement::Address,
            ),
            InfluenceOutcome::Active(_)
        ));
    }

    #[test]
    fn reverse_corridor_ignores_an_incomplete_branch_that_cannot_reach_the_sink() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::dead_branch");
        let source = formal(&caller, 1);
        let sink = formal(&caller, 9);
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 20), callee.clone());
        let dead_result = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let callee_relation = ContextRelationKey::new(
            callee.clone(),
            formal(&callee, 1),
            EndpointNode::Return {
                owner: callee.clone(),
            },
        );
        let mut context = ContextProofIndex::default();
        context.record_local(local_edge(source.clone(), sink.clone(), 1));
        context.record_local(local_edge(
            callee_relation.source.clone(),
            callee_relation.target.clone(),
            2,
        ));
        context.record_call(
            EndpointEdge {
                from: source.clone(),
                to: dead_result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite,
            GenericCallSubstitution {
                status: GenericSubstitutionStatus::NormalizationIncomplete,
                bindings: BTreeMap::new(),
            },
            callee_relation,
            ContextActualBinding::Internal,
        );
        context.finalize();

        let reverse = context.exact_reverse_reachability(&sink);
        let source_handle = context
            .exact_atom_handles_for_endpoint(&source)
            .into_iter()
            .find(|atom| context.exact_atom(*atom).owner == caller)
            .expect("source is scoped in the caller");
        let dead_handle = context
            .exact_atom_handles_for_endpoint(&dead_result)
            .into_iter()
            .find(|atom| context.exact_atom(*atom).owner == caller)
            .expect("dead call result is scoped in the caller");
        assert!(reverse.contains(source_handle));
        assert!(!reverse.contains(dead_handle));

        let boolean = InfluenceToken::from_type_class(
            InfluenceOrigin::FfiReturn,
            InfluenceController::FfiHost,
            InfluenceTypeClass::Boolean,
        );
        assert!(matches!(
            InfluenceSolver::new(&context).solve_in_reverse(
                caller.clone(),
                &source,
                caller,
                &sink,
                boolean,
                &SinkRequirement::Address,
                &reverse,
            ),
            InfluenceOutcome::ProvenIncompatible(_)
        ));
    }

    #[test]
    fn missing_callee_context_is_explicitly_unestablished() {
        let caller = FunctionKey::new("crate::caller");
        let callee = FunctionKey::new("crate::missing");
        let source = formal(&caller, 1);
        let callsite = CallSiteId::new(caller.clone(), point(&caller, 10), callee.clone());
        let result = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: CallSiteOutput::Return,
        };
        let mut context = ContextProofIndex::default();
        context.record_call(
            EndpointEdge {
                from: source.clone(),
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
        context.finalize();

        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                caller.clone(),
                &source,
                caller,
                &result,
                ffi_address_token(),
                &SinkRequirement::Address,
            ),
            InfluenceOutcome::Unestablished { reasons }
                if reasons == BTreeSet::from([UnestablishedReason::MissingExactContext])
        ));
    }

    #[test]
    fn recursive_call_alternative_with_local_base_is_finite() {
        let owner = FunctionKey::new("crate::recursive");
        let source = formal(&owner, 1);
        let sink = formal(&owner, 2);
        let relation = ContextRelationKey::new(owner.clone(), source.clone(), sink.clone());
        let callsite = CallSiteId::new(owner.clone(), point(&owner, 10), owner.clone());
        let mut context = ContextProofIndex::default();
        context.record_local(local_edge(source.clone(), sink.clone(), 1));
        context.record_call(
            EndpointEdge {
                from: source.clone(),
                to: sink.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite,
            GenericCallSubstitution::default(),
            relation,
            ContextActualBinding::Internal,
        );
        context.finalize();

        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                owner.clone(),
                &source,
                owner,
                &sink,
                ffi_address_token(),
                &SinkRequirement::Address,
            ),
            InfluenceOutcome::Active(_)
        ));
    }

    #[test]
    fn mutually_recursive_calls_with_one_local_base_are_finite() {
        let left = FunctionKey::new("crate::left");
        let right = FunctionKey::new("crate::right");
        let left_source = formal(&left, 1);
        let left_target = EndpointNode::Return {
            owner: left.clone(),
        };
        let right_source = formal(&right, 1);
        let right_target = EndpointNode::Return {
            owner: right.clone(),
        };
        let left_relation =
            ContextRelationKey::new(left.clone(), left_source.clone(), left_target.clone());
        let right_relation =
            ContextRelationKey::new(right.clone(), right_source.clone(), right_target.clone());
        let left_callsite = CallSiteId::new(left.clone(), point(&left, 10), right.clone());
        let right_callsite = CallSiteId::new(right.clone(), point(&right, 20), left.clone());
        let mut context = ContextProofIndex::default();
        context.record_local(local_edge(right_source.clone(), right_target.clone(), 1));
        context.record_call(
            EndpointEdge {
                from: left_source.clone(),
                to: left_target.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(left_callsite.point.clone()),
            },
            left_callsite,
            GenericCallSubstitution::default(),
            right_relation.clone(),
            ContextActualBinding::Internal,
        );
        context.record_call(
            EndpointEdge {
                from: right_source,
                to: right_target,
                kind: EndpointEdgeKind::CallSummary,
                point: Some(right_callsite.point.clone()),
            },
            right_callsite,
            GenericCallSubstitution::default(),
            left_relation,
            ContextActualBinding::Internal,
        );
        context.finalize();

        assert!(matches!(
            InfluenceSolver::new(&context).solve(
                left.clone(),
                &left_source,
                left,
                &left_target,
                ffi_address_token(),
                &SinkRequirement::Address,
            ),
            InfluenceOutcome::Active(_)
        ));
    }

    #[test]
    fn influence_seed_type_classes_are_capability_specific() {
        let boolean = InfluenceToken::from_type_class(
            InfluenceOrigin::OpenTraitReturn,
            InfluenceController::Unknown,
            InfluenceTypeClass::Boolean,
        );
        assert!(boolean
            .capabilities
            .intersects(InfluenceCapabilities::BOOLEAN));
        assert!(!boolean
            .capabilities
            .intersects(InfluenceCapabilities::ADDRESS));

        let integer = InfluenceToken::from_type_class(
            InfluenceOrigin::PublicFormal,
            InfluenceController::Unknown,
            InfluenceTypeClass::Integer,
        );
        assert!(integer
            .capabilities
            .intersects(InfluenceCapabilities::INDEX_OR_LENGTH));
        assert!(!integer
            .capabilities
            .intersects(InfluenceCapabilities::ADDRESS));

        let pointer = InfluenceToken::from_type_class(
            InfluenceOrigin::FfiReturn,
            InfluenceController::FfiHost,
            InfluenceTypeClass::Pointer,
        );
        assert!(pointer
            .capabilities
            .intersects(InfluenceCapabilities::ADDRESS));

        let callable = InfluenceToken::from_type_class(
            InfluenceOrigin::PublicFunctionPointerReturn,
            InfluenceController::PublicCallback,
            InfluenceTypeClass::Callable,
        );
        assert!(callable
            .capabilities
            .intersects(InfluenceCapabilities::CALLABLE_RESULT));

        let aggregate = InfluenceToken::from_type_class(
            InfluenceOrigin::OpenTraitReturn,
            InfluenceController::Unknown,
            InfluenceTypeClass::Aggregate,
        );
        assert!(aggregate
            .capabilities
            .intersects(InfluenceCapabilities::TYPED_VALUE));
        assert!(!aggregate
            .capabilities
            .intersects(InfluenceCapabilities::ADDRESS));
    }

    #[test]
    fn safe_external_reference_seed_is_unestablished_for_address_or_validity() {
        let reference = InfluenceToken::from_type_class(
            InfluenceOrigin::OpenTraitReturn,
            InfluenceController::OpenTrait {
                trait_id: FunctionKey::new("crate::Trait"),
                receiver: OpenBehaviorReceiver::Dynamic,
            },
            InfluenceTypeClass::Reference,
        );
        assert_eq!(
            SinkRequirement::Address.matches(reference.capabilities),
            RequirementMatch::ProvenIncompatible,
        );
        assert_eq!(
            SinkRequirement::ValidityBits.matches(reference.capabilities),
            RequirementMatch::ProvenIncompatible,
        );
        assert_eq!(
            SinkRequirement::TypedContent.matches(reference.capabilities),
            RequirementMatch::Compatible,
        );
        assert_eq!(reference.precision, TokenPrecision::Incomplete);
        assert_eq!(
            reference.incomplete_reason,
            Some(UnestablishedReason::UnknownCapabilityMatch),
        );
    }

    #[test]
    fn safe_public_reference_seed_does_not_claim_invalid_address_or_validity() {
        let reference = InfluenceToken::from_type_class(
            InfluenceOrigin::PublicFormal,
            InfluenceController::PublicFormal {
                owner: FunctionKey::new("crate::reference"),
                index: 1,
            },
            InfluenceTypeClass::Reference,
        );
        assert_eq!(
            SinkRequirement::Address.matches(reference.capabilities),
            RequirementMatch::ProvenIncompatible,
        );
        assert_eq!(
            SinkRequirement::ValidityBits.matches(reference.capabilities),
            RequirementMatch::ProvenIncompatible,
        );
        assert_eq!(
            SinkRequirement::TypedContent.matches(reference.capabilities),
            RequirementMatch::Compatible,
        );
    }

    #[test]
    fn ffi_reference_seed_keeps_strong_boundary_address_capability() {
        let reference = InfluenceToken::from_type_class(
            InfluenceOrigin::FfiReturn,
            InfluenceController::FfiHost,
            InfluenceTypeClass::Reference,
        );
        assert_eq!(
            SinkRequirement::Address.matches(reference.capabilities),
            RequirementMatch::Compatible,
        );
        assert_eq!(
            SinkRequirement::ValidityBits.matches(reference.capabilities),
            RequirementMatch::Compatible,
        );
    }

    #[test]
    fn seed_index_keeps_endpoint_identity_and_marks_opaque_join_incomplete() {
        let owner = FunctionKey::new("crate::seeds");
        let endpoint = formal(&owner, 1);
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            endpoint.clone(),
            InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal {
                    owner: owner.clone(),
                    index: 1,
                },
                InfluenceTypeClass::Pointer,
            ),
        );
        seeds.mark_incomplete(&endpoint, UnestablishedReason::OpaqueTransform);
        let seed = seeds.get(&endpoint).expect("seed remains auditable");
        assert_eq!(seed.precision, TokenPrecision::Incomplete);
        assert!(seed.capabilities.intersects(InfluenceCapabilities::ADDRESS));
        assert_eq!(seeds.len(), 1);
    }
}
