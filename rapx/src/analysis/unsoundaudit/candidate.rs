use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::sync::Arc;

use super::contract_catalog::{
    ContractDescriptor, FailurePhase, StaticContractGap, StaticContractImplication,
    StructuralSiteCandidate,
};
use super::endpoint_flow::{
    BehaviorController, BehaviorOpenness, CapabilityImpossibleReason, CapabilityRequirement,
    CapabilityTerm, CapabilityTermForm, CapabilityTermRole, CapabilityTypeFacts,
    CapabilityUnestablishedReason, CapabilityViolation, CapabilityWitness, ContextActualBinding,
    ContextAtom, ContextPredecessor, ContextProofIndex, ContextQuery, ContextRelationKey,
    ContextUnknownReason, DerivationKind, EndpointCondensation, EndpointEdge, EndpointFlowGraph,
    EndpointNode, EndpointRouteIndex, ExactReverseReachability, GenericArgBinding,
    GenericCallSubstitution, GenericCapabilityEvidence, GenericCapabilityKind, GenericParamId,
    GenericSubstitutionStatus, OpenBehaviorReceiver, PublicFieldAdtKind, StorageProvenance,
    TypedWitnessTransport,
};
use super::influence::{
    primary_unestablished_reason, InfluenceController, InfluenceOrigin, InfluenceOutcome,
    InfluenceSeedIndex, InfluenceSolver, SinkRequirement, UnestablishedReason,
};
use super::perf_trace::PerfTrace;
use super::summary::{sha256_hex, FunctionKey, ProgramPoint, StableSpan};
use super::unsafe_site::{
    binding::{BoundUnsafeSite, MirTypeValidity},
    CausalSourceKind, ClassificationConfidence, OperandRole, PrimaryPattern, SuppressionReason,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CandidateOwner {
    pub function: FunctionKey,
    pub root_span: StableSpan,
    pub public_safe: bool,
    pub generic_surface: BTreeSet<GenericParamId>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CandidateCallKind {
    ResolvedLocal,
    Callback,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CandidateCallEdge {
    pub caller: FunctionKey,
    pub callee: FunctionKey,
    pub point: ProgramPoint,
    pub kind: CandidateCallKind,
    pub generic_substitution: GenericCallSubstitution,
}

pub(super) struct CallRouteIndex<'graph> {
    outgoing: BTreeMap<&'graph FunctionKey, Vec<&'graph CandidateCallEdge>>,
    incoming: BTreeMap<&'graph FunctionKey, Vec<&'graph CandidateCallEdge>>,
}

impl<'graph> CallRouteIndex<'graph> {
    pub(super) fn new(calls: &'graph BTreeSet<CandidateCallEdge>) -> Self {
        let mut outgoing = BTreeMap::<&FunctionKey, Vec<&CandidateCallEdge>>::new();
        let mut incoming = BTreeMap::<&FunctionKey, Vec<&CandidateCallEdge>>::new();
        for edge in calls {
            outgoing.entry(&edge.caller).or_default().push(edge);
            incoming.entry(&edge.callee).or_default().push(edge);
        }
        Self { outgoing, incoming }
    }

    pub(super) fn routes_to(&self, target: &FunctionKey) -> CallRoutesTo<'_, 'graph> {
        let mut distances = BTreeMap::from([(target.clone(), 0)]);
        let mut queue = VecDeque::from([target.clone()]);
        while let Some(owner) = queue.pop_front() {
            let distance = distances[&owner];
            for edge in self.incoming.get(&owner).into_iter().flatten() {
                if distances.contains_key(&edge.caller) {
                    continue;
                }
                distances.insert(edge.caller.clone(), distance + 1);
                queue.push_back(edge.caller.clone());
            }
        }
        CallRoutesTo {
            index: self,
            target: target.clone(),
            distances,
        }
    }
}

pub(super) struct CallRoutesTo<'index, 'graph> {
    index: &'index CallRouteIndex<'graph>,
    target: FunctionKey,
    distances: BTreeMap<FunctionKey, usize>,
}

impl CallRoutesTo<'_, '_> {
    fn reaches(&self, root: &FunctionKey) -> bool {
        root == &self.target || self.distances.contains_key(root)
    }

    fn route_from(&self, root: &FunctionKey) -> Option<Vec<CandidateCallEdge>> {
        if root == &self.target {
            return Some(Vec::new());
        }
        let distance = *self.distances.get(root)?;
        let mut route = Vec::with_capacity(distance);
        let mut cursor = root;
        for remaining in (1..=distance).rev() {
            let edge = self
                .index
                .outgoing
                .get(cursor)?
                .iter()
                .copied()
                .find(|edge| self.distances.get(&edge.callee) == Some(&(remaining - 1)))?;
            route.push(edge.clone());
            cursor = &edge.callee;
        }
        Some(route)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ProjectedCapabilityTerm {
    form: CapabilityTermForm,
    controllers: BTreeSet<GenericParamId>,
    residual_aliases: BTreeSet<String>,
    equality_class: Option<u32>,
    type_facts: CapabilityTypeFacts,
    const_value: Option<u64>,
}

impl From<&CapabilityTerm> for ProjectedCapabilityTerm {
    fn from(term: &CapabilityTerm) -> Self {
        Self {
            form: term.form,
            controllers: term.controllers.clone(),
            residual_aliases: term.residual_aliases.clone(),
            equality_class: term.equality_class,
            type_facts: term.type_facts.clone(),
            const_value: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CapabilityRouteState {
    owner: FunctionKey,
    terms: BTreeMap<CapabilityTermRole, ProjectedCapabilityTerm>,
    forced_equal: bool,
    normalization_incomplete: bool,
}

enum CapabilityRouteResult {
    Parametric(Vec<CandidateCallEdge>),
    ConcreteSatisfied,
    ConcreteViolation {
        route: Vec<CandidateCallEdge>,
        violations: BTreeSet<CapabilityViolation>,
    },
    Unestablished,
    Unreachable,
}

enum P5RouteOutcome {
    Parametric {
        call_route: Vec<CandidateCallEdge>,
        endpoint_route: Option<Vec<EndpointEdge>>,
    },
    ConcreteSatisfied,
    ConcreteViolation {
        call_route: Vec<CandidateCallEdge>,
        endpoint_route: Option<Vec<EndpointEdge>>,
        violations: BTreeSet<CapabilityViolation>,
    },
    Unestablished,
    Unreachable,
}

fn capability_route_result_from_legacy(
    call_route_index: &CallRouteIndex<'_>,
    root: &CandidateOwner,
    source_owner: &FunctionKey,
    evidence: &GenericCapabilityEvidence,
) -> P5RouteOutcome {
    match call_route_index.capability_route(root, source_owner, evidence) {
        CapabilityRouteResult::Parametric(call_route) => P5RouteOutcome::Parametric {
            call_route,
            endpoint_route: None,
        },
        CapabilityRouteResult::ConcreteSatisfied => P5RouteOutcome::ConcreteSatisfied,
        CapabilityRouteResult::ConcreteViolation { route, violations } => {
            P5RouteOutcome::ConcreteViolation {
                call_route: route,
                endpoint_route: None,
                violations,
            }
        }
        CapabilityRouteResult::Unestablished => P5RouteOutcome::Unestablished,
        CapabilityRouteResult::Unreachable => P5RouteOutcome::Unreachable,
    }
}

impl CallRouteIndex<'_> {
    /// Find any real root -> source route on which the sink-local capability
    /// remains controlled by a public generic parameter. The canonical
    /// shortest route may be concrete while a sibling route is parametric, so
    /// this explores every distinct finite substitution state.
    fn capability_route(
        &self,
        root: &CandidateOwner,
        source_owner: &FunctionKey,
        evidence: &GenericCapabilityEvidence,
    ) -> CapabilityRouteResult {
        let initial = CapabilityRouteState {
            owner: source_owner.clone(),
            terms: evidence
                .terms
                .iter()
                .map(|(role, term)| (*role, ProjectedCapabilityTerm::from(term)))
                .collect(),
            forced_equal: false,
            normalization_incomplete: false,
        };
        let mut queue = VecDeque::from([(initial, Vec::<CandidateCallEdge>::new())]);
        let mut visited = BTreeSet::new();
        let mut saw_satisfied_concrete = false;
        let mut saw_unestablished = false;
        let mut reached_root = false;
        let mut concrete_violation = None;

        while let Some((state, reverse_route)) = queue.pop_front() {
            if !visited.insert(state.clone()) {
                continue;
            }
            if state.owner == root.function {
                reached_root = true;
                if state.normalization_incomplete {
                    saw_unestablished = true;
                    continue;
                }
                match evaluate_concrete_capability(&state, &evidence.requirement) {
                    ConcreteCapabilityResult::Satisfied => saw_satisfied_concrete = true,
                    ConcreteCapabilityResult::Violated(violations) => {
                        let mut route = reverse_route;
                        route.reverse();
                        concrete_violation.get_or_insert((route, violations));
                    }
                    ConcreteCapabilityResult::Unestablished => {
                        if parametric_witness_transport_established(&state, evidence, root) {
                            let mut route = reverse_route;
                            route.reverse();
                            return CapabilityRouteResult::Parametric(route);
                        }
                        saw_unestablished = true;
                    }
                }
                continue;
            }

            let Some(incoming) = self.incoming.get(&state.owner) else {
                continue;
            };
            for edge in incoming.iter().copied() {
                if edge.kind != CandidateCallKind::ResolvedLocal {
                    saw_unestablished = true;
                    continue;
                }
                let next = project_capability_through_call(&state, edge);
                let mut route = reverse_route.clone();
                route.push(edge.clone());
                queue.push_back((next, route));
            }
        }

        if let Some((route, violations)) = concrete_violation {
            CapabilityRouteResult::ConcreteViolation { route, violations }
        } else if saw_unestablished {
            CapabilityRouteResult::Unestablished
        } else if reached_root && saw_satisfied_concrete {
            CapabilityRouteResult::ConcreteSatisfied
        } else {
            CapabilityRouteResult::Unreachable
        }
    }
}

/// A source-local finite witness remains a witness at the public root only
/// when every assigned role is still a free direct parameter or an associated
/// projection whose transport was proven by the caller-bound certificate. A
/// capability that was itself an associated projection (for example
/// `P::Results`) may remain composite after substituting `P`; retaining that
/// same residual item identity is also a valid parametric transport.
///
/// Composite substitutions are deliberately excluded here. A controller
/// nested inside `*mut T`, `[T; N]`, `Simd<T, N>`, or another constructor does
/// not make the source-local `S = u8, T = u64` assignment realizable. Concrete
/// layout facts are evaluated before this predicate, so fixed-layout
/// composites such as `*mut T -> usize` can still be proven satisfied rather
/// than becoming Unknown.
fn parametric_witness_transport_established(
    state: &CapabilityRouteState,
    evidence: &GenericCapabilityEvidence,
    root: &CandidateOwner,
) -> bool {
    if state.forced_equal {
        return false;
    }
    let CapabilityWitness::Proven { assignment, .. } = &evidence.witness else {
        return false;
    };
    let mut controllers = BTreeSet::new();
    for role in assignment.keys() {
        let Some(term) = state.terms.get(role) else {
            return false;
        };
        let associated_projection_preserved = evidence.terms.get(role).is_some_and(|source| {
            source.form == CapabilityTermForm::Alias
                && !source.residual_aliases.is_empty()
                && source
                    .residual_aliases
                    .iter()
                    .all(|alias| term.residual_aliases.contains(alias))
        });
        if !matches!(
            term.form,
            CapabilityTermForm::DirectParam | CapabilityTermForm::Alias
        ) && !(term.form == CapabilityTermForm::Composite && associated_projection_preserved)
        {
            return false;
        }
        controllers.extend(term.controllers.iter());
    }
    controllers
        .iter()
        .any(|parameter| root.generic_surface.contains(*parameter))
}

fn project_capability_through_call(
    state: &CapabilityRouteState,
    edge: &CandidateCallEdge,
) -> CapabilityRouteState {
    let mut normalization_incomplete = state.normalization_incomplete
        || edge.generic_substitution.status == GenericSubstitutionStatus::NormalizationIncomplete;
    let mut projected = BTreeMap::new();
    let mut equality_classes = BTreeMap::<CapabilityTermRole, Option<u32>>::new();

    for (role, term) in &state.terms {
        let mut next = ProjectedCapabilityTerm {
            form: term.form,
            controllers: BTreeSet::new(),
            residual_aliases: term.residual_aliases.clone(),
            equality_class: term.equality_class,
            type_facts: term.type_facts.clone(),
            const_value: term.const_value,
        };
        let mut role_equality_class = None;
        for parameter in &term.controllers {
            let Some(binding) = edge.generic_substitution.bindings.get(parameter) else {
                normalization_incomplete = true;
                continue;
            };
            if binding.kind != parameter.kind {
                normalization_incomplete = true;
                continue;
            }
            next.controllers
                .extend(binding.caller_controllers.iter().cloned());
            next.residual_aliases
                .extend(binding.residual_aliases.iter().cloned());
            if term.controllers.len() == 1 && term.residual_aliases.is_empty() {
                next.form = binding.form;
                role_equality_class = binding.equality_class;
                next.equality_class = binding.equality_class;
                next.type_facts = binding.type_facts.clone();
                next.const_value = binding.const_value;
            } else if !binding.caller_controllers.is_empty() {
                next.form = CapabilityTermForm::Composite;
                next.equality_class = None;
                next.type_facts = CapabilityTypeFacts::default();
                next.const_value = None;
            }
            if binding.form != CapabilityTermForm::Concrete && binding.caller_controllers.is_empty()
            {
                normalization_incomplete = true;
            }
        }
        equality_classes.insert(*role, role_equality_class);
        projected.insert(*role, next);
    }

    let direct_pointer_terms = state
        .terms
        .get(&CapabilityTermRole::SourcePointee)
        .zip(state.terms.get(&CapabilityTermRole::TargetPointee))
        .is_some_and(|(source, target)| {
            source.form == CapabilityTermForm::DirectParam
                && target.form == CapabilityTermForm::DirectParam
                && source.residual_aliases.is_empty()
                && target.residual_aliases.is_empty()
        });
    let newly_equal = direct_pointer_terms
        && equality_classes
            .get(&CapabilityTermRole::SourcePointee)
            .copied()
            .flatten()
            .zip(
                equality_classes
                    .get(&CapabilityTermRole::TargetPointee)
                    .copied()
                    .flatten(),
            )
            .is_some_and(|(source, target)| source == target);

    CapabilityRouteState {
        owner: edge.caller.clone(),
        terms: projected,
        forced_equal: state.forced_equal || newly_equal,
        normalization_incomplete,
    }
}

enum ConcreteCapabilityResult {
    Satisfied,
    Violated(BTreeSet<CapabilityViolation>),
    Unestablished,
}

fn evaluate_concrete_capability(
    state: &CapabilityRouteState,
    requirement: &CapabilityRequirement,
) -> ConcreteCapabilityResult {
    match requirement {
        CapabilityRequirement::RepresentationPreservingView => ConcreteCapabilityResult::Satisfied,
        CapabilityRequirement::AssociatedSequenceNonEmpty => {
            let length = state
                .terms
                .get(&CapabilityTermRole::LengthConst)
                .and_then(|term| term.const_value);
            match length {
                Some(0) => ConcreteCapabilityResult::Violated(BTreeSet::from([
                    CapabilityViolation::InBounds,
                ])),
                Some(_) => ConcreteCapabilityResult::Satisfied,
                None => ConcreteCapabilityResult::Unestablished,
            }
        }
        CapabilityRequirement::PointerLayout { extent } => {
            if matches!(extent, super::endpoint_flow::AccessExtent::Zero) {
                return ConcreteCapabilityResult::Satisfied;
            }
            let Some(source) = state.terms.get(&CapabilityTermRole::SourcePointee) else {
                return ConcreteCapabilityResult::Unestablished;
            };
            let Some(target) = state.terms.get(&CapabilityTermRole::TargetPointee) else {
                return ConcreteCapabilityResult::Unestablished;
            };
            if state.forced_equal
                || source
                    .equality_class
                    .zip(target.equality_class)
                    .is_some_and(|(source, target)| source == target)
            {
                return ConcreteCapabilityResult::Satisfied;
            }
            let Some(source_size) = source.type_facts.size else {
                return ConcreteCapabilityResult::Unestablished;
            };
            let Some(target_size) = target.type_facts.size else {
                return ConcreteCapabilityResult::Unestablished;
            };
            let Some(source_align) = source.type_facts.align else {
                return ConcreteCapabilityResult::Unestablished;
            };
            let Some(target_align) = target.type_facts.align else {
                return ConcreteCapabilityResult::Unestablished;
            };
            let mut violations = BTreeSet::new();
            if target_size > source_size {
                violations.insert(CapabilityViolation::Size);
            }
            if target_align > source_align {
                violations.insert(CapabilityViolation::Alignment);
            }
            if violations.is_empty() {
                ConcreteCapabilityResult::Satisfied
            } else {
                ConcreteCapabilityResult::Violated(violations)
            }
        }
    }
}

/// Capability state is deliberately independent of an endpoint owner. The
/// owner/scope lives in `P5Request`; the same state may be carried through a
/// Formal boundary unchanged or projected through one exact Internal call.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ContextCapabilityState {
    terms: BTreeMap<CapabilityTermRole, ProjectedCapabilityTerm>,
    forced_equal: bool,
    normalization_incomplete: bool,
}

impl ContextCapabilityState {
    fn from_evidence(evidence: &GenericCapabilityEvidence) -> Self {
        Self {
            terms: evidence
                .terms
                .iter()
                .map(|(role, term)| (*role, ProjectedCapabilityTerm::from(term)))
                .collect(),
            forced_equal: false,
            normalization_incomplete: false,
        }
    }

    fn concrete_view(&self) -> CapabilityRouteState {
        CapabilityRouteState {
            owner: FunctionKey::new("context-capability-state"),
            terms: self.terms.clone(),
            forced_equal: self.forced_equal,
            normalization_incomplete: self.normalization_incomplete,
        }
    }
}

fn project_context_capability_through_substitution(
    state: &ContextCapabilityState,
    substitution: &GenericCallSubstitution,
    typed_transport_required: bool,
    typed_transport_proven: bool,
) -> ContextCapabilityState {
    let mut mapping_incomplete = false;
    let mut projected = BTreeMap::new();
    let mut equality_classes = BTreeMap::<CapabilityTermRole, Option<u32>>::new();
    for (role, term) in &state.terms {
        let mut next = ProjectedCapabilityTerm {
            form: term.form,
            controllers: BTreeSet::new(),
            residual_aliases: term.residual_aliases.clone(),
            equality_class: term.equality_class,
            type_facts: term.type_facts.clone(),
            const_value: term.const_value,
        };
        let mut role_equality_class = None;
        for parameter in &term.controllers {
            let Some(binding) = substitution.bindings.get(parameter) else {
                mapping_incomplete = true;
                continue;
            };
            if binding.kind != parameter.kind {
                mapping_incomplete = true;
                continue;
            }
            next.controllers
                .extend(binding.caller_controllers.iter().cloned());
            next.residual_aliases
                .extend(binding.residual_aliases.iter().cloned());
            if term.controllers.len() == 1 && term.residual_aliases.is_empty() {
                next.form = binding.form;
                role_equality_class = binding.equality_class;
                next.equality_class = binding.equality_class;
                next.type_facts = binding.type_facts.clone();
                next.const_value = binding.const_value;
            } else if !binding.caller_controllers.is_empty() {
                next.form = CapabilityTermForm::Composite;
                next.equality_class = None;
                next.type_facts = CapabilityTypeFacts::default();
                next.const_value = None;
            }
            if binding.form != CapabilityTermForm::Concrete && binding.caller_controllers.is_empty()
            {
                mapping_incomplete = true;
            }
        }
        equality_classes.insert(*role, role_equality_class);
        projected.insert(*role, next);
    }
    let direct_pointer_terms = state
        .terms
        .get(&CapabilityTermRole::SourcePointee)
        .zip(state.terms.get(&CapabilityTermRole::TargetPointee))
        .is_some_and(|(source, target)| {
            source.form == CapabilityTermForm::DirectParam
                && target.form == CapabilityTermForm::DirectParam
                && source.residual_aliases.is_empty()
                && target.residual_aliases.is_empty()
        });
    let newly_equal = direct_pointer_terms
        && equality_classes
            .get(&CapabilityTermRole::SourcePointee)
            .copied()
            .flatten()
            .zip(
                equality_classes
                    .get(&CapabilityTermRole::TargetPointee)
                    .copied()
                    .flatten(),
            )
            .is_some_and(|(source, target)| source == target);
    let normalization_incomplete = state.normalization_incomplete
        || (substitution.status == GenericSubstitutionStatus::NormalizationIncomplete
            && !typed_transport_proven)
        || mapping_incomplete
        // A compiler-derived equality class is itself sufficient to prove
        // that the two pointer-layout terms cannot violate size/alignment.
        // It does not need the stronger representation-transport certificate
        // used to preserve a positive source-local mismatch witness.
        || (typed_transport_required && !typed_transport_proven && !newly_equal);
    ContextCapabilityState {
        terms: projected,
        forced_equal: state.forced_equal || newly_equal,
        normalization_incomplete,
    }
}

/// A direct callee parameter remains governed by the normal substitution
/// facts when it becomes a caller direct parameter or a concrete type. An
/// associated projection is different: layout/validity of the projection is
/// not established merely because rustc retained it as an alias, so it needs
/// the bounded joint Source/Target transport certificate.
fn internal_transport_requires_typed_certificate(
    state: &ContextCapabilityState,
    substitution: &GenericCallSubstitution,
) -> bool {
    state.terms.values().any(|term| {
        term.controllers.iter().any(|parameter| {
            substitution.bindings.get(parameter).is_some_and(|binding| {
                binding.form == CapabilityTermForm::Alias || !binding.residual_aliases.is_empty()
            })
        })
    })
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct P5Request {
    scope: FunctionKey,
    start: ContextAtom,
    target: ContextAtom,
    seed: ContextCapabilityState,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct P5SearchState {
    request: P5Request,
    at: ContextAtom,
    state: ContextCapabilityState,
    proof: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct P5Fact {
    state: ContextCapabilityState,
    proof: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct P5Waiter {
    parent: P5Request,
    next: ContextAtom,
    parent_proof: usize,
    edge: EndpointEdge,
    call: CandidateCallEdge,
    callsite: super::endpoint_flow::CallSiteId,
    actual_binding: ContextActualBinding,
    callee_start: ContextAtom,
    callee_target: ContextAtom,
}

#[derive(Clone, Debug)]
enum P5Proof {
    Root,
    Local {
        parent: usize,
        edge: EndpointEdge,
    },
    Call {
        parent: usize,
        child: usize,
        edge: EndpointEdge,
        call: CandidateCallEdge,
        actual_binding: ContextActualBinding,
        callee_start: ContextAtom,
        callee_target: ContextAtom,
    },
}

struct ContextP5Solver<'context> {
    context: &'context ContextProofIndex,
    evidence: Option<&'context GenericCapabilityEvidence>,
    proofs: Vec<P5Proof>,
    started: BTreeSet<P5Request>,
    pending: VecDeque<P5SearchState>,
    seen: BTreeSet<(P5Request, ContextAtom, ContextCapabilityState)>,
    facts: BTreeMap<P5Request, BTreeSet<P5Fact>>,
    waiters: BTreeMap<P5Request, BTreeSet<P5Waiter>>,
    saw_unknown: bool,
}

impl<'context> ContextP5Solver<'context> {
    fn new(
        context: &'context ContextProofIndex,
        evidence: Option<&'context GenericCapabilityEvidence>,
    ) -> Self {
        Self {
            context,
            evidence,
            proofs: vec![P5Proof::Root],
            started: BTreeSet::new(),
            pending: VecDeque::new(),
            seen: BTreeSet::new(),
            facts: BTreeMap::new(),
            waiters: BTreeMap::new(),
            saw_unknown: false,
        }
    }

    fn new_proof(&mut self, proof: P5Proof) -> usize {
        let id = self.proofs.len();
        self.proofs.push(proof);
        id
    }

    fn start(&mut self, request: P5Request) {
        if self.started.insert(request.clone()) {
            if !self
                .seen
                .insert((request.clone(), request.start.clone(), request.seed.clone()))
            {
                return;
            }
            self.pending.push_back(P5SearchState {
                at: request.start.clone(),
                state: request.seed.clone(),
                request,
                proof: 0,
            });
        }
    }

    fn solve(
        mut self,
        request: P5Request,
        root: &CandidateOwner,
        requirement: &CapabilityRequirement,
    ) -> P5RouteOutcome {
        self.start(request.clone());
        while let Some(current) = self.pending.pop_front() {
            if current.at == current.request.target {
                self.finish_fact(
                    current.request.clone(),
                    P5Fact {
                        state: current.state,
                        proof: current.proof,
                    },
                );
                continue;
            }
            for (relation, alternatives) in self.context.exact_relations_from(&current.at) {
                let next = ContextAtom::from_target(relation);
                for alternative in alternatives {
                    match alternative {
                        ContextPredecessor::Local { fragment } => {
                            if !self.seen.insert((
                                current.request.clone(),
                                next.clone(),
                                current.state.clone(),
                            )) {
                                continue;
                            }
                            let proof = self.new_proof(P5Proof::Local {
                                parent: current.proof,
                                edge: (*fragment).clone(),
                            });
                            self.pending.push_back(P5SearchState {
                                request: current.request.clone(),
                                at: next.clone(),
                                state: current.state.clone(),
                                proof,
                            });
                        }
                        ContextPredecessor::Call {
                            callsite,
                            substitution,
                            callee_relation,
                            actual_binding,
                        } => {
                            let unknown = self.context.unknown_reasons_for_callsite(&callsite);
                            let missing_body = unknown.iter().any(|reason| {
                                matches!(
                                    reason,
                                    super::endpoint_flow::ContextUnknownReason::MissingResolvedBody { .. }
                                )
                            });
                            let missing_formal_actual = matches!(
                                actual_binding,
                                ContextActualBinding::Formal(_)
                            ) && unknown.iter().any(|reason| {
                                matches!(
                                    reason,
                                    super::endpoint_flow::ContextUnknownReason::MissingActualEndpoint { .. }
                                )
                            });
                            let unresolved_boundary = unknown.iter().any(|reason| match reason {
                                super::endpoint_flow::ContextUnknownReason::MissingActualEndpoint { .. } => {
                                    matches!(actual_binding, ContextActualBinding::Formal(_))
                                }
                                super::endpoint_flow::ContextUnknownReason::MissingResolvedBody { .. }
                                | super::endpoint_flow::ContextUnknownReason::UncontextualizedCallback
                                | super::endpoint_flow::ContextUnknownReason::CompletenessUnavailable => true,
                                // An internal generic boundary transports its
                                // exact substitution below. Equality classes
                                // or a typed certificate may close this gap;
                                // the projected terminal state records whether
                                // normalization actually remains incomplete.
                                super::endpoint_flow::ContextUnknownReason::NormalizationIncomplete { .. } => {
                                    matches!(actual_binding, ContextActualBinding::Formal(_))
                                }
                            });
                            self.saw_unknown |= unresolved_boundary;
                            if missing_body || missing_formal_actual {
                                // An unavailable boundary cannot support a
                                // P5 proof. Keep coarse provenance for the
                                // lower-ranked patterns, but do not turn an
                                // absent exact call context into a witness.
                                continue;
                            }
                            let child = P5Request {
                                scope: callee_relation.context_owner.clone(),
                                start: ContextAtom::from_source(&callee_relation),
                                target: ContextAtom::from_target(&callee_relation),
                                seed: current.state.clone(),
                            };
                            let waiter = P5Waiter {
                                parent: current.request.clone(),
                                next: next.clone(),
                                parent_proof: current.proof,
                                edge: ContextProofIndex::instantiated_call_edge(
                                    relation, &callsite,
                                ),
                                call: candidate_call_from_context(&callsite, &substitution),
                                callsite: (*callsite).clone(),
                                actual_binding,
                                callee_start: child.start.clone(),
                                callee_target: child.target.clone(),
                            };
                            self.waiters
                                .entry(child.clone())
                                .or_default()
                                .insert(waiter.clone());
                            let existing = self
                                .facts
                                .get(&child)
                                .into_iter()
                                .flatten()
                                .cloned()
                                .collect::<Vec<_>>();
                            for fact in existing {
                                self.resume_waiter(&waiter, fact);
                            }
                            self.start(child);
                        }
                    }
                }
            }
        }
        let terminal = self.facts.remove(&request).unwrap_or_default();
        self.started.clear();
        self.pending.clear();
        self.seen.clear();
        self.facts.clear();
        self.waiters.clear();
        let terminal_nonempty = !terminal.is_empty();
        let mut concrete_violation = None;
        let mut satisfied = false;
        let mut unresolved = self.saw_unknown;
        for fact in terminal {
            if fact.state.normalization_incomplete {
                unresolved = true;
                continue;
            }
            let (endpoint_route, call_route) = self.render_proof(fact.proof);
            match evaluate_concrete_capability(&fact.state.concrete_view(), requirement) {
                ConcreteCapabilityResult::Satisfied => satisfied = true,
                ConcreteCapabilityResult::Violated(violations) => {
                    concrete_violation.get_or_insert((endpoint_route, call_route, violations));
                }
                ConcreteCapabilityResult::Unestablished => {
                    if self.evidence.is_some_and(|evidence| {
                        parametric_witness_transport_established(
                            &fact.state.concrete_view(),
                            evidence,
                            root,
                        )
                    }) {
                        return P5RouteOutcome::Parametric {
                            call_route,
                            endpoint_route: Some(endpoint_route),
                        };
                    }
                    unresolved = true;
                }
            }
        }
        if let Some((endpoint_route, call_route, violations)) = concrete_violation {
            P5RouteOutcome::ConcreteViolation {
                call_route,
                endpoint_route: Some(endpoint_route),
                violations,
            }
        } else if unresolved {
            P5RouteOutcome::Unestablished
        } else if terminal_nonempty && satisfied {
            P5RouteOutcome::ConcreteSatisfied
        } else {
            P5RouteOutcome::Unreachable
        }
    }

    fn finish_fact(&mut self, request: P5Request, fact: P5Fact) {
        if !self
            .facts
            .entry(request.clone())
            .or_default()
            .insert(fact.clone())
        {
            return;
        }
        let waiters = self
            .waiters
            .get(&request)
            .into_iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        for waiter in waiters {
            self.resume_waiter(&waiter, fact.clone());
        }
    }

    fn resume_waiter(&mut self, waiter: &P5Waiter, child: P5Fact) {
        let state = match waiter.actual_binding {
            ContextActualBinding::Internal => {
                // A nested callee may refine the capability terms before
                // returning them to this boundary.  Bind the finite witness
                // to that returned fact, rather than to the caller's
                // pre-call state, so an outer complete alias cannot inherit
                // an inner unproven projection route.
                let typed_transport_required = internal_transport_requires_typed_certificate(
                    &child.state,
                    &waiter.call.generic_substitution,
                );
                let typed_transport_proven = self.evidence.is_some_and(|evidence| {
                    typed_witness_transport_matches(
                        &self
                            .context
                            .typed_witness_transports_for_callsite(&waiter.callsite),
                        evidence,
                        &child.state,
                    )
                });
                project_context_capability_through_substitution(
                    &child.state,
                    &waiter.call.generic_substitution,
                    typed_transport_required,
                    typed_transport_proven,
                )
            }
            ContextActualBinding::Formal(_) => child.state,
        };
        if !self
            .seen
            .insert((waiter.parent.clone(), waiter.next.clone(), state.clone()))
        {
            return;
        }
        let proof = self.new_proof(P5Proof::Call {
            parent: waiter.parent_proof,
            child: child.proof,
            edge: waiter.edge.clone(),
            call: waiter.call.clone(),
            actual_binding: waiter.actual_binding,
            callee_start: waiter.callee_start.clone(),
            callee_target: waiter.callee_target.clone(),
        });
        self.pending.push_back(P5SearchState {
            request: waiter.parent.clone(),
            at: waiter.next.clone(),
            state,
            proof,
        });
    }

    fn render_proof(&self, id: usize) -> (Vec<EndpointEdge>, Vec<CandidateCallEdge>) {
        let (endpoints, mut calls) = self.render_proof_forward(id);
        calls.reverse();
        (endpoints, calls)
    }

    fn render_proof_forward(&self, id: usize) -> (Vec<EndpointEdge>, Vec<CandidateCallEdge>) {
        match &self.proofs[id] {
            P5Proof::Root => (Vec::new(), Vec::new()),
            P5Proof::Local { parent, edge } => {
                let (mut endpoints, calls) = self.render_proof_forward(*parent);
                endpoints.push(edge.clone());
                (endpoints, calls)
            }
            P5Proof::Call {
                parent,
                child,
                edge,
                call,
                actual_binding,
                callee_start,
                callee_target,
            } => {
                let (mut endpoints, mut calls) = self.render_proof_forward(*parent);
                if matches!(actual_binding, ContextActualBinding::Formal(_))
                    && edge.from != callee_start.endpoint
                {
                    endpoints.push(EndpointEdge {
                        from: edge.from.clone(),
                        to: callee_start.endpoint.clone(),
                        kind: super::endpoint_flow::EndpointEdgeKind::ActualToFormal,
                        point: edge.point.clone(),
                    });
                }
                let (child_endpoints, child_calls) = self.render_proof_forward(*child);
                endpoints.extend(child_endpoints);
                if callee_target.endpoint != edge.to {
                    endpoints.push(EndpointEdge {
                        from: callee_target.endpoint.clone(),
                        to: edge.to.clone(),
                        kind: match (&callee_target.endpoint, actual_binding) {
                            (EndpointNode::Return { .. }, _) => {
                                super::endpoint_flow::EndpointEdgeKind::ReturnToDestination
                            }
                            (EndpointNode::Out { .. }, _) => {
                                super::endpoint_flow::EndpointEdgeKind::OutToCaller
                            }
                            _ => super::endpoint_flow::EndpointEdgeKind::CallSummary,
                        },
                        point: edge.point.clone(),
                    });
                }
                calls.extend(child_calls);
                calls.push(call.clone());
                (endpoints, calls)
            }
        }
    }
}

fn candidate_call_from_context(
    callsite: &super::endpoint_flow::CallSiteId,
    substitution: &GenericCallSubstitution,
) -> CandidateCallEdge {
    CandidateCallEdge {
        caller: callsite.caller.clone(),
        callee: callsite.callee.clone(),
        point: callsite.point.clone(),
        kind: CandidateCallKind::ResolvedLocal,
        generic_substitution: substitution.clone(),
    }
}

fn capability_route_result_from_context(
    context: &ContextProofIndex,
    root: &CandidateOwner,
    source: &EndpointNode,
    target: &EndpointNode,
    evidence: &GenericCapabilityEvidence,
) -> P5RouteOutcome {
    let scope = root.function.clone();
    let request = P5Request {
        scope: scope.clone(),
        start: ContextAtom::scoped(scope.clone(), source.clone()),
        target: ContextAtom::scoped(scope, target.clone()),
        seed: ContextCapabilityState::from_evidence(evidence),
    };
    ContextP5Solver::new(context, Some(evidence)).solve(request, root, &evidence.requirement)
}

fn typed_witness_transport_matches(
    transports: &BTreeSet<TypedWitnessTransport>,
    evidence: &GenericCapabilityEvidence,
    state: &ContextCapabilityState,
) -> bool {
    let CapabilityWitness::Proven { assignment, .. } = &evidence.witness else {
        return false;
    };
    let Some(source) = state.terms.get(&CapabilityTermRole::SourcePointee) else {
        return false;
    };
    let Some(target) = state.terms.get(&CapabilityTermRole::TargetPointee) else {
        return false;
    };
    transports.iter().any(|transport| {
        transport.source_identity == evidence.source_identity()
            && transport.template.assignment == *assignment
            && source.controllers == BTreeSet::from([transport.template.source_parameter.clone()])
            && target.controllers == BTreeSet::from([transport.template.target_parameter.clone()])
            && !transport.template.caller_controller.def_path.is_empty()
    })
}

struct PublicRootIdentityCache<'owners, 'index, 'graph> {
    owners: &'owners BTreeMap<FunctionKey, CandidateOwner>,
    call_route_index: &'index CallRouteIndex<'graph>,
    by_owner: BTreeMap<FunctionKey, Vec<&'owners FunctionKey>>,
    #[cfg(test)]
    build_count: usize,
}

impl<'owners, 'index, 'graph> PublicRootIdentityCache<'owners, 'index, 'graph> {
    fn new(
        owners: &'owners BTreeMap<FunctionKey, CandidateOwner>,
        call_route_index: &'index CallRouteIndex<'graph>,
    ) -> Self {
        Self {
            owners,
            call_route_index,
            by_owner: BTreeMap::new(),
            #[cfg(test)]
            build_count: 0,
        }
    }

    fn prefill(&mut self, owner: &FunctionKey, roots: impl IntoIterator<Item = FunctionKey>) {
        if self.by_owner.contains_key(owner) {
            return;
        }
        let mut roots = roots
            .into_iter()
            .filter_map(|root| self.owners.get_key_value(&root).map(|(root, _)| root))
            .collect::<Vec<_>>();
        roots.sort_unstable();
        roots.dedup();
        self.by_owner.insert(owner.clone(), roots);
    }

    fn roots_reaching(&mut self, owner: &FunctionKey) -> &[&'owners FunctionKey] {
        if !self.by_owner.contains_key(owner) {
            let routes_to_owner = self.call_route_index.routes_to(owner);
            let roots = reachable_public_root_identities(self.owners, &routes_to_owner);
            self.by_owner.insert(owner.clone(), roots);
            #[cfg(test)]
            {
                self.build_count += 1;
            }
        }
        self.by_owner
            .get(owner)
            .expect("root identity cache entry was just inserted")
    }

    #[cfg(test)]
    fn test_build_count(&self) -> usize {
        self.build_count
    }
}

enum CandidateRootIdentities<'cache, 'owners> {
    None,
    One(FunctionKey),
    Cached(&'cache [&'owners FunctionKey]),
}

impl CandidateRootIdentities<'_, '_> {
    fn contains(&self, root: &FunctionKey) -> bool {
        match self {
            Self::None => false,
            Self::One(only) => only == root,
            Self::Cached(roots) => roots
                .binary_search_by(|candidate| (*candidate).cmp(root))
                .is_ok(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CandidateExposure {
    pub node: EndpointNode,
    pub span: Arc<StableSpan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateSiteContext {
    first_failure: StableSpan,
    site: BoundUnsafeSite,
    contract: ContractDescriptor,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MechanismId {
    P1PublicParameterFlow,
    P2PublicStateFlow,
    P3InternalUnsafeExposure,
    P4ConcreteCapabilityViolation,
    P4InternalDerivation,
    P5GenericCapabilityGap,
    P6ExternalBehaviorFlow,
}

impl MechanismId {
    fn as_str(self) -> &'static str {
        match self {
            Self::P1PublicParameterFlow => "p1_public_parameter_flow_v1",
            Self::P2PublicStateFlow => "p2_public_state_flow_v1",
            Self::P3InternalUnsafeExposure => "p3_internal_unsafe_exposure_v1",
            Self::P4ConcreteCapabilityViolation => "p4_concrete_capability_violation_v1",
            Self::P4InternalDerivation => "p4_internal_derivation_v1",
            Self::P5GenericCapabilityGap => "p5_generic_capability_gap_v1",
            Self::P6ExternalBehaviorFlow => "p6_external_behavior_flow_v1",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StructuralFindingCandidate {
    pub primary_pattern: PrimaryPattern,
    mechanism_id: MechanismId,
    pub classification_confidence: ClassificationConfidence,
    pub public_root: Arc<FunctionKey>,
    pub public_root_span: Arc<StableSpan>,
    pub call_route: Arc<[CandidateCallEdge]>,
    pub source: Arc<EndpointNode>,
    pub source_kind: CausalSourceKind,
    pub source_span: Arc<StableSpan>,
    site_context: Arc<CandidateSiteContext>,
    pub operand_role: Option<OperandRole>,
    pub exposure: Option<Arc<CandidateExposure>>,
    pub endpoint_route: Arc<[EndpointEdge]>,
}

impl StructuralFindingCandidate {
    pub(super) fn mechanism_id(&self) -> &'static str {
        self.mechanism_id.as_str()
    }

    pub(super) fn first_failure(&self) -> &StableSpan {
        &self.site_context.first_failure
    }

    pub(super) fn site(&self) -> &BoundUnsafeSite {
        &self.site_context.site
    }

    pub(super) fn contract(&self) -> &ContractDescriptor {
        &self.site_context.contract
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SuppressedStructuralCandidate {
    pub candidate: StructuralFindingCandidate,
    pub reason: SuppressionReason,
    pub evidence: Vec<(String, String)>,
}

impl SuppressedStructuralCandidate {
    pub(super) fn new(
        candidate: StructuralFindingCandidate,
        reason: SuppressionReason,
        evidence: BTreeMap<String, String>,
    ) -> Self {
        assert!(
            !evidence.is_empty(),
            "structural suppression evidence must not be empty"
        );
        Self {
            candidate,
            reason,
            evidence: evidence.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct UnestablishedStructuralCandidate {
    pub decision_id: String,
    pub attempted_pattern: PrimaryPattern,
    pub public_root: Arc<FunctionKey>,
    pub site_id: String,
    pub operand_role: OperandRole,
    pub source_identity: String,
    pub source_kind: CausalSourceKind,
    pub controller: String,
    pub capabilities: String,
    pub requirement: Option<String>,
    pub primary_reason: UnestablishedReason,
    pub secondary_reasons: Vec<UnestablishedReason>,
}

impl UnestablishedStructuralCandidate {
    fn new(
        candidate: &StructuralFindingCandidate,
        token: Option<&super::influence::InfluenceToken>,
        requirement: Option<&SinkRequirement>,
        reasons: BTreeSet<UnestablishedReason>,
    ) -> Self {
        let operand_role = candidate
            .operand_role
            .clone()
            .expect("operand-scoped unestablished decisions require a role");
        Self::from_obligation(
            candidate,
            operand_role,
            token
                .map(|token| token.controller.stable_identity())
                .unwrap_or_else(|| "unknown".to_string()),
            token
                .map(|token| token.capabilities.stable_identity())
                .unwrap_or_else(|| "unknown".to_string()),
            requirement.map(SinkRequirement::stable_identity),
            reasons,
        )
    }

    pub(super) fn from_obligation(
        candidate: &StructuralFindingCandidate,
        operand_role: OperandRole,
        controller: String,
        capabilities: String,
        requirement: Option<String>,
        reasons: BTreeSet<UnestablishedReason>,
    ) -> Self {
        let primary_reason = primary_unestablished_reason(&reasons)
            .expect("an unestablished decision has at least one reason");
        let secondary_reasons = reasons
            .into_iter()
            .filter(|reason| *reason != primary_reason)
            .collect::<Vec<_>>();
        let attempted_pattern = candidate.primary_pattern;
        let public_root = candidate.public_root.clone();
        let site_id = candidate.site().seed.site_id();
        let source_identity = influence_source_identity(candidate.source.as_ref());
        let source_kind = candidate.source_kind;
        let decision_source_identity = candidate.exposure.as_ref().map_or_else(
            || source_identity.clone(),
            |exposure| {
                format!(
                    "{source_identity}|exposure={}",
                    influence_source_identity(&exposure.node)
                )
            },
        );
        let decision_id = influence_decision_id(
            attempted_pattern,
            public_root.as_ref(),
            &site_id,
            &operand_role,
            &decision_source_identity,
            source_kind,
        );
        Self {
            decision_id,
            attempted_pattern,
            public_root,
            site_id,
            operand_role,
            source_identity,
            source_kind,
            controller,
            capabilities,
            requirement,
            primary_reason,
            secondary_reasons,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct StructuralCandidateSet {
    pub active: Vec<StructuralFindingCandidate>,
    pub suppressed: Vec<SuppressedStructuralCandidate>,
    pub unestablished: Vec<UnestablishedStructuralCandidate>,
}

impl StructuralCandidateSet {
    pub(super) fn into_global_candidate_order(self) -> Self {
        Self {
            active: globally_ordered_candidates(self.active),
            suppressed: globally_ordered_suppressions(self.suppressed),
            unestablished: globally_ordered_unestablished(self.unestablished),
        }
    }
}

pub(super) fn build_structural_candidates(
    graph: &EndpointFlowGraph,
    structural_sites: &BTreeMap<String, StructuralSiteCandidate>,
    contracts: &BTreeMap<String, ContractDescriptor>,
    bound_sites: &BTreeMap<String, BoundUnsafeSite>,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
) -> Vec<StructuralFindingCandidate> {
    build_structural_candidate_set(
        graph,
        structural_sites,
        contracts,
        bound_sites,
        owners,
        calls,
    )
    .active
}

pub(super) fn build_structural_candidate_set(
    graph: &EndpointFlowGraph,
    structural_sites: &BTreeMap<String, StructuralSiteCandidate>,
    contracts: &BTreeMap<String, ContractDescriptor>,
    bound_sites: &BTreeMap<String, BoundUnsafeSite>,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
) -> StructuralCandidateSet {
    build_structural_candidate_set_with_context(
        graph,
        structural_sites,
        contracts,
        bound_sites,
        owners,
        calls,
        None,
    )
}

pub(super) fn build_structural_candidate_set_with_context(
    graph: &EndpointFlowGraph,
    structural_sites: &BTreeMap<String, StructuralSiteCandidate>,
    contracts: &BTreeMap<String, ContractDescriptor>,
    bound_sites: &BTreeMap<String, BoundUnsafeSite>,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
    context_proofs: Option<&ContextProofIndex>,
) -> StructuralCandidateSet {
    build_structural_candidate_set_inner(
        graph,
        structural_sites,
        contracts,
        bound_sites,
        owners,
        calls,
        context_proofs,
        None,
        None,
    )
    .expect("disabled structural trace cannot fail")
    .into_global_candidate_order()
}

pub(super) fn build_structural_candidate_set_with_context_traced(
    graph: &EndpointFlowGraph,
    structural_sites: &BTreeMap<String, StructuralSiteCandidate>,
    contracts: &BTreeMap<String, ContractDescriptor>,
    bound_sites: &BTreeMap<String, BoundUnsafeSite>,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
    context_proofs: Option<&ContextProofIndex>,
    influence_seeds: &InfluenceSeedIndex,
    trace: &mut PerfTrace,
) -> Result<StructuralCandidateSet, String> {
    build_structural_candidate_set_inner(
        graph,
        structural_sites,
        contracts,
        bound_sites,
        owners,
        calls,
        context_proofs,
        Some(influence_seeds),
        Some(trace),
    )
}

struct StructuralTrace<'trace> {
    trace: Option<&'trace mut PerfTrace>,
    eligibility_attempt_count: u64,
    eligibility_completed_count: u64,
    p5_eligibility_attempt_count: u64,
    p5_eligibility_completed_count: u64,
    role_started_count: u64,
    role_completed_count: u64,
    discovered_source_count: u64,
    site_started_count: u64,
    site_batch_active: bool,
}

impl<'trace> StructuralTrace<'trace> {
    fn new(trace: Option<&'trace mut PerfTrace>) -> Self {
        let trace = match trace {
            Some(trace) if trace.is_enabled() => Some(trace),
            _ => None,
        };
        Self {
            trace,
            eligibility_attempt_count: 0,
            eligibility_completed_count: 0,
            p5_eligibility_attempt_count: 0,
            p5_eligibility_completed_count: 0,
            role_started_count: 0,
            role_completed_count: 0,
            discovered_source_count: 0,
            site_started_count: 0,
            site_batch_active: false,
        }
    }

    fn event<const N: usize>(
        &mut self,
        phase: &str,
        event: &str,
        counts: [(&str, u64); N],
    ) -> Result<(), String> {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.event(phase, event, counts)?;
        }
        Ok(())
    }

    fn checkpoint<const N: usize>(
        &mut self,
        phase: &str,
        counts: [(&str, u64); N],
    ) -> Result<(), String> {
        self.event(phase, "checkpoint", counts)
    }

    fn site_started(
        &mut self,
        candidate_count: usize,
        suppression_count: usize,
    ) -> Result<(), String> {
        // A begin/end pair every 256 semantic BTree-ordered sites makes the
        // last flushed record useful after SIGTERM without turning a large
        // structural scan into a trace-file or I/O workload.
        if self.trace.is_none() {
            return Ok(());
        }
        if self.site_batch_active && self.site_started_count % 256 == 0 {
            self.site_batch_active = false;
            self.site_batch_event("end", candidate_count, suppression_count)?;
        }
        self.site_started_count += 1;
        if !self.site_batch_active {
            self.site_batch_active = true;
            self.site_batch_event("begin", candidate_count, suppression_count)?;
        }
        Ok(())
    }

    fn finish_sites(
        &mut self,
        candidate_count: usize,
        suppression_count: usize,
    ) -> Result<(), String> {
        if self.site_batch_active {
            self.site_batch_active = false;
            self.site_batch_event("end", candidate_count, suppression_count)?;
        }
        Ok(())
    }

    fn site_batch_event(
        &mut self,
        event: &str,
        candidate_count: usize,
        suppression_count: usize,
    ) -> Result<(), String> {
        let counts = [
            ("site_started_count", self.site_started_count),
            (
                "site_completed_count",
                if event == "begin" {
                    self.site_started_count - 1
                } else {
                    self.site_started_count
                },
            ),
            ("role_started_count", self.role_started_count),
            ("role_completed_count", self.role_completed_count),
            ("discovered_source_count", self.discovered_source_count),
            ("eligibility_attempt_count", self.eligibility_attempt_count),
            (
                "eligibility_completed_count",
                self.eligibility_completed_count,
            ),
            (
                "p5_eligibility_attempt_count",
                self.p5_eligibility_attempt_count,
            ),
            (
                "p5_eligibility_completed_count",
                self.p5_eligibility_completed_count,
            ),
            ("candidate_count", candidate_count as u64),
            ("suppression_count", suppression_count as u64),
        ];
        self.event("structural_site_batch", event, counts)
    }

    fn eligibility_attempt(&mut self, p5: bool) -> Result<(), String> {
        if self.trace.is_none() {
            return Ok(());
        }
        // Checkpoints grow only at powers of two, so each counter produces at
        // most 64 flushed records even if a production scan has millions of
        // eligibility or P5 attempts.
        self.eligibility_attempt_count += 1;
        if p5 {
            self.p5_eligibility_attempt_count += 1;
            if self.p5_eligibility_attempt_count.is_power_of_two() {
                self.checkpoint(
                    "structural_p5_progress",
                    [
                        ("p5_started_count", self.p5_eligibility_attempt_count),
                        ("p5_completed_count", self.p5_eligibility_completed_count),
                        ("eligibility_attempt_count", self.eligibility_attempt_count),
                        (
                            "eligibility_completed_count",
                            self.eligibility_completed_count,
                        ),
                    ],
                )?;
            }
        }
        if self.eligibility_attempt_count.is_power_of_two() {
            self.checkpoint(
                "structural_eligibility_progress",
                [
                    ("eligibility_attempt_count", self.eligibility_attempt_count),
                    (
                        "eligibility_completed_count",
                        self.eligibility_completed_count,
                    ),
                    (
                        "p5_eligibility_attempt_count",
                        self.p5_eligibility_attempt_count,
                    ),
                    (
                        "p5_eligibility_completed_count",
                        self.p5_eligibility_completed_count,
                    ),
                ],
            )?;
        }
        Ok(())
    }

    fn finish_eligibility_attempt(&mut self, p5: bool) -> Result<(), String> {
        if self.trace.is_none() {
            return Ok(());
        }
        self.eligibility_completed_count += 1;
        if p5 {
            self.p5_eligibility_completed_count += 1;
        }
        if self.eligibility_completed_count.is_power_of_two() {
            self.checkpoint(
                "structural_eligibility_progress",
                [
                    ("eligibility_attempt_count", self.eligibility_attempt_count),
                    (
                        "eligibility_completed_count",
                        self.eligibility_completed_count,
                    ),
                    (
                        "p5_eligibility_attempt_count",
                        self.p5_eligibility_attempt_count,
                    ),
                    (
                        "p5_eligibility_completed_count",
                        self.p5_eligibility_completed_count,
                    ),
                ],
            )?;
        }
        if p5 && self.p5_eligibility_completed_count.is_power_of_two() {
            self.checkpoint(
                "structural_p5_progress",
                [
                    ("p5_started_count", self.p5_eligibility_attempt_count),
                    ("p5_completed_count", self.p5_eligibility_completed_count),
                    ("eligibility_attempt_count", self.eligibility_attempt_count),
                    (
                        "eligibility_completed_count",
                        self.eligibility_completed_count,
                    ),
                ],
            )?;
        }
        Ok(())
    }

    fn role_started(&mut self) -> Result<(), String> {
        if self.trace.is_none() {
            return Ok(());
        }
        self.role_started_count += 1;
        if self.role_started_count.is_power_of_two() {
            self.checkpoint(
                "structural_role_progress",
                [
                    ("role_started_count", self.role_started_count),
                    ("role_completed_count", self.role_completed_count),
                    ("discovered_source_count", self.discovered_source_count),
                    ("eligibility_attempt_count", self.eligibility_attempt_count),
                    (
                        "eligibility_completed_count",
                        self.eligibility_completed_count,
                    ),
                ],
            )?;
        }
        Ok(())
    }

    fn finish_role(&mut self) -> Result<(), String> {
        if self.trace.is_none() {
            return Ok(());
        }
        self.role_completed_count += 1;
        if self.role_completed_count.is_power_of_two() {
            self.checkpoint(
                "structural_role_progress",
                [
                    ("role_started_count", self.role_started_count),
                    ("role_completed_count", self.role_completed_count),
                    ("discovered_source_count", self.discovered_source_count),
                    ("eligibility_attempt_count", self.eligibility_attempt_count),
                    (
                        "eligibility_completed_count",
                        self.eligibility_completed_count,
                    ),
                ],
            )?;
        }
        Ok(())
    }

    fn add_discovered_sources(&mut self, count: usize) {
        if self.trace.is_some() {
            self.discovered_source_count += count as u64;
        }
    }
}

fn build_structural_candidate_set_inner(
    graph: &EndpointFlowGraph,
    structural_sites: &BTreeMap<String, StructuralSiteCandidate>,
    contracts: &BTreeMap<String, ContractDescriptor>,
    bound_sites: &BTreeMap<String, BoundUnsafeSite>,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
    context_proofs: Option<&ContextProofIndex>,
    influence_seeds: Option<&InfluenceSeedIndex>,
    trace: Option<&mut PerfTrace>,
) -> Result<StructuralCandidateSet, String> {
    let mut trace = StructuralTrace::new(trace);
    let mut candidates = Vec::<StructuralFindingCandidate>::new();
    let mut suppressions = Vec::<SuppressedStructuralCandidate>::new();
    let mut unestablished = Vec::<UnestablishedStructuralCandidate>::new();
    trace.event(
        "structural_route_index",
        "begin",
        [
            ("endpoint_node_count", graph.nodes.len() as u64),
            ("endpoint_edge_count", graph.edges.len() as u64),
        ],
    )?;
    let route_index = graph.route_index();
    trace.event(
        "structural_route_index",
        "end",
        [
            ("endpoint_node_count", graph.nodes.len() as u64),
            ("endpoint_edge_count", graph.edges.len() as u64),
        ],
    )?;
    trace.event("structural_condensation", "begin", [])?;
    let condensation = route_index.condensation();
    trace.event("structural_condensation", "end", [])?;
    trace.event(
        "structural_call_route_index",
        "begin",
        [
            ("candidate_call_count", calls.len() as u64),
            ("owner_count", owners.len() as u64),
        ],
    )?;
    let call_route_index = CallRouteIndex::new(calls);
    let mut root_identity_cache = PublicRootIdentityCache::new(owners, &call_route_index);
    trace.event(
        "structural_call_route_index",
        "end",
        [
            ("candidate_call_count", calls.len() as u64),
            ("owner_count", owners.len() as u64),
        ],
    )?;
    trace.event(
        "structural_site_scan",
        "begin",
        [
            ("structural_site_count", structural_sites.len() as u64),
            ("bound_site_count", bound_sites.len() as u64),
            ("contract_count", contracts.len() as u64),
        ],
    )?;
    let mut source_interner = HashSet::<Arc<EndpointNode>>::new();
    for (site_id, structural) in structural_sites {
        trace.site_started(candidates.len(), suppressions.len())?;
        let Some(site) = bound_sites.get(site_id) else {
            continue;
        };
        let Some(contract) = contracts.get(site_id) else {
            continue;
        };
        // `candidate_key` includes the stable unsafe-site ID.  Deduplicate at
        // the smallest natural scope so the exact keys and BTree nodes die at
        // the end of this site instead of growing across the whole crate.
        // A final stable key sort below restores the former global BTreeMap
        // order and handles the defensive case of duplicate site IDs.
        let mut site_candidates_by_key = BTreeMap::new();
        let mut site_suppressions_by_key = BTreeMap::new();
        let mut site_unestablished_by_id = BTreeMap::new();
        let site_context = Arc::new(CandidateSiteContext {
            first_failure: site.seed.raw_span.stable.clone(),
            site: site.clone(),
            contract: contract.clone(),
        });
        let site = &site_context.site;
        let contract = &site_context.contract;
        let call_routes_to_site = call_route_index.routes_to(&site.seed.owner);
        let public_root_routes = reachable_public_roots(owners, &call_routes_to_site);

        if let Some(exposures) = p3_exposures(
            graph,
            &route_index,
            &call_route_index,
            &site_context,
            structural,
            owners,
            &public_root_routes,
            context_proofs,
            influence_seeds,
        ) {
            for candidate in exposures.active {
                insert_candidate_first_wins(&mut site_candidates_by_key, candidate);
            }
            for suppression in exposures.suppressed {
                site_suppressions_by_key
                    .entry(candidate_key(&suppression.candidate))
                    .or_insert(suppression);
            }
            for decision in exposures.unestablished {
                site_unestablished_by_id
                    .entry(decision.decision_id.clone())
                    .or_insert(decision);
            }
            candidates.extend(site_candidates_by_key.into_values());
            suppressions.extend(site_suppressions_by_key.into_values());
            unestablished.extend(site_unestablished_by_id.into_values());
            continue;
        }
        if public_root_routes.is_empty() {
            continue;
        }

        let roles = if contract.precision == super::unsafe_site::ContractPrecision::Precise {
            contract.relevant_roles.clone()
        } else {
            let mut roles = structural.relevant_roles.clone();
            roles.extend(contract.relevant_roles.iter().cloned());
            roles
        };
        // Process one operand role completely before retaining the next
        // reverse-distance map.  `routes_to` is intentionally per-role, but
        // keeping all of those maps through selection multiplied the largest
        // temporary in production bodies.  The role iteration and every
        // subsequent source/root iteration remain ordered exactly as before.
        let mut selected = Vec::new();
        for role in roles {
            trace.role_started()?;
            let target = EndpointNode::UnsafeOperand {
                owner: site.seed.owner.clone(),
                site_id: site_id.clone(),
                role: role.clone(),
            };
            if !graph.nodes.contains(&target) {
                trace.finish_role()?;
                continue;
            }
            let routes_to_target = route_index.routes_to(&target);
            let mut reaching_nodes = routes_to_target.reaching_nodes().collect::<Vec<_>>();
            // Coarse return bridges intentionally do not model every
            // callsite-instantiated callee capability.  Discover only the
            // exact GenericCapability atoms which can reach this operand;
            // the atom owner is the public-root scope that later P5 solving
            // must retain.  This is one reverse exact walk per target, not a
            // source×target or root×source expansion.
            let exact_reverse =
                context_proofs.map(|context| context.exact_reverse_reachability(&target));
            let contextual_p5_roots = context_proofs
                .zip(exact_reverse.as_ref())
                .map(|(context, reverse)| context.exact_generic_capability_scopes_in(reverse))
                .unwrap_or_default();
            reaching_nodes = merge_ordered_endpoint_nodes(
                reaching_nodes.into_iter(),
                contextual_p5_roots.keys(),
            );
            let public_formal_sources = reaching_nodes
                .iter()
                .filter(|node| {
                    matches!(node, EndpointNode::Formal { owner, .. }
                        if owners.get(owner).is_some_and(|info| info.public_safe)
                            && call_routes_to_site.reaches(owner))
                })
                .copied()
                .collect::<Vec<_>>();
            let internal_derivations = reaching_nodes
                .iter()
                .filter(|node| matches!(node, EndpointNode::InternalDerivation { .. }))
                .copied()
                .collect::<Vec<_>>();
            let bypassed_derivations =
                if public_formal_sources.is_empty() || internal_derivations.is_empty() {
                    BTreeSet::new()
                } else {
                    let unavoidable_from_public_formals =
                        routes_to_target.unavoidable_from(public_formal_sources.iter().copied());
                    internal_derivations
                        .into_iter()
                        .filter(|source| !unavoidable_from_public_formals.contains(*source))
                        .map(|source| (*source).clone())
                        .collect()
                };
            let mut pending_sources = Vec::new();
            for source in reaching_nodes {
                if bypassed_derivations.contains(&source) {
                    continue;
                }
                if let EndpointNode::InternalDerivation { kind, .. } = source {
                    if !derivation_is_relevant(*kind, &role, contract) {
                        continue;
                    }
                }
                let Some((pattern, source_kind, mechanism, rank)) = source_classification(&source)
                else {
                    continue;
                };
                let Some(source_span) = node_span(&source, owners) else {
                    continue;
                };
                pending_sources.push((rank, pattern, source_kind, mechanism, source, source_span));
            }
            trace.add_discovered_sources(pending_sources.len());
            let mut eligible_by_root =
                BTreeMap::<Arc<FunctionKey>, (u8, Vec<StructuralFindingCandidate>)>::new();
            for (rank, pattern, source_kind, mechanism, source, source_span) in &pending_sources {
                if matches!(
                    *source,
                    EndpointNode::ExternalBehavior { owner, .. }
                        | EndpointNode::GenericCapability { owner, .. }
                        | EndpointNode::InternalDerivation { owner, .. }
                        | EndpointNode::PublicField {
                            access_owner: owner,
                            ..
                        }
                        if owner == &site.seed.owner
                ) {
                    root_identity_cache.prefill(
                        &site.seed.owner,
                        public_root_routes.keys().map(|root| root.as_ref().clone()),
                    );
                }
                let coarse_endpoint_route = routes_to_target.route_from(source);
                let matched_roots = {
                    let root_identities =
                        candidate_root_identities(source, owners, &mut root_identity_cache);
                    public_root_routes
                        .keys()
                        .filter(|root| {
                            root_identities.contains(root.as_ref())
                                && (coarse_endpoint_route.is_some()
                                    || contextual_p5_roots
                                        .get(*source)
                                        .is_some_and(|scopes| scopes.contains(root.as_ref())))
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                };
                if matched_roots.is_empty() {
                    continue;
                }
                // A callee-local P5 source may lack a coarse interprocedural
                // route even though it has an exact callsite realization.
                // Do not grant the same exception to P1/P2/P4/P6.
                if coarse_endpoint_route.is_none()
                    && !(*pattern == PrimaryPattern::P5 && context_proofs.is_some())
                {
                    continue;
                }
                let endpoint_route =
                    Arc::<[EndpointEdge]>::from(coarse_endpoint_route.unwrap_or_default());
                if *pattern == PrimaryPattern::P1
                    && endpoint_route.iter().any(|edge| {
                        bypassed_derivations.contains(&edge.from)
                            || bypassed_derivations.contains(&edge.to)
                    })
                {
                    continue;
                }
                let source = intern_candidate_source(&mut source_interner, source);
                for root in matched_roots {
                    let (root_span, call_route) = &public_root_routes[&root];
                    let call_route = call_route.clone();
                    let mut candidate = StructuralFindingCandidate {
                        primary_pattern: *pattern,
                        mechanism_id: *mechanism,
                        classification_confidence: ClassificationConfidence::Structural,
                        public_root: root.clone(),
                        public_root_span: root_span.clone(),
                        call_route,
                        source: source.clone(),
                        source_kind: *source_kind,
                        source_span: source_span.clone(),
                        site_context: site_context.clone(),
                        operand_role: Some(role.clone()),
                        exposure: None,
                        endpoint_route: endpoint_route.clone(),
                    };
                    let target = EndpointNode::UnsafeOperand {
                        owner: site.seed.owner.clone(),
                        site_id: site_id.clone(),
                        role: role.clone(),
                    };
                    let eligibility = if let Some(root_owner) = owners.get(root.as_ref()) {
                        let p5 = *pattern == PrimaryPattern::P5;
                        trace.eligibility_attempt(p5)?;
                        let eligibility = if *pattern == PrimaryPattern::P2
                            && matches!(
                                source.as_ref(),
                                EndpointNode::PublicField {
                                    persistent: false,
                                    ..
                                }
                            ) {
                            source_eligibility_with_context(
                                source.as_ref(),
                                root_owner,
                                &role,
                                contract,
                                &call_route_index,
                                context_proofs,
                                &target,
                            )
                        } else if matches!(
                            pattern,
                            PrimaryPattern::P1 | PrimaryPattern::P2 | PrimaryPattern::P6
                        ) && influence_seeds.is_some()
                        {
                            exact_influence_eligibility(
                                *pattern,
                                source.as_ref(),
                                root_owner,
                                &role,
                                contract,
                                context_proofs,
                                &target,
                                influence_seeds
                                    .expect("production P1/P6 eligibility has typed seeds"),
                                exact_reverse.as_ref(),
                            )
                        } else if *pattern == PrimaryPattern::P4
                            && context_proofs.is_some()
                            && exact_reverse.is_some()
                        {
                            exact_derivation_eligibility_in_reverse(
                                source.as_ref(),
                                root_owner,
                                &role,
                                contract,
                                context_proofs
                                    .expect("production P4 eligibility has exact context"),
                                exact_reverse
                                    .as_ref()
                                    .expect("production P4 eligibility has reverse corridor"),
                            )
                        } else {
                            source_eligibility_with_context(
                                source.as_ref(),
                                root_owner,
                                &role,
                                contract,
                                &call_route_index,
                                context_proofs,
                                &target,
                            )
                        };
                        trace.finish_eligibility_attempt(p5)?;
                        eligibility
                    } else {
                        SourceEligibility::Unsupported
                    };
                    match eligibility {
                        SourceEligibility::Reclassified {
                            supporting_call_route,
                            supporting_endpoint_route,
                            reason,
                            evidence,
                            pattern,
                            source_kind,
                            mechanism_id,
                            rank,
                        } => {
                            site_suppressions_by_key
                                .entry(candidate_key(&candidate))
                                .or_insert_with(|| {
                                    SuppressedStructuralCandidate::new(
                                        candidate.clone(),
                                        reason,
                                        evidence,
                                    )
                                });
                            candidate.primary_pattern = pattern;
                            candidate.source_kind = source_kind;
                            candidate.mechanism_id = mechanism_id;
                            candidate.classification_confidence = ClassificationConfidence::Exact;
                            candidate.call_route = supporting_call_route.into();
                            candidate.endpoint_route = supporting_endpoint_route.into();
                            insert_best_rank_candidate(
                                &mut eligible_by_root,
                                root,
                                rank,
                                candidate,
                            );
                        }
                        SourceEligibility::Suppressed(reason, evidence) => {
                            site_suppressions_by_key
                                .entry(candidate_key(&candidate))
                                .or_insert_with(|| {
                                    SuppressedStructuralCandidate::new(candidate, reason, evidence)
                                });
                        }
                        SourceEligibility::Active {
                            supporting_call_route,
                            supporting_endpoint_route,
                        } => {
                            if (matches!(
                                pattern,
                                PrimaryPattern::P1 | PrimaryPattern::P2 | PrimaryPattern::P6
                            ) && influence_seeds.is_some())
                                || (matches!(pattern, PrimaryPattern::P4 | PrimaryPattern::P5)
                                    && context_proofs.is_some())
                            {
                                candidate.classification_confidence =
                                    ClassificationConfidence::Exact;
                            }
                            if let Some(supporting_call_route) = supporting_call_route {
                                candidate.call_route = supporting_call_route.into();
                            }
                            if let Some(supporting_endpoint_route) = supporting_endpoint_route {
                                candidate.endpoint_route = supporting_endpoint_route.into();
                            }
                            insert_best_rank_candidate(
                                &mut eligible_by_root,
                                root,
                                *rank,
                                candidate,
                            );
                        }
                        SourceEligibility::Unestablished {
                            token,
                            requirement,
                            reasons,
                        } => {
                            if heuristic_unmodeled_public_source(*pattern, *source_kind, &reasons) {
                                candidate.classification_confidence =
                                    ClassificationConfidence::Heuristic;
                                insert_best_rank_candidate(
                                    &mut eligible_by_root,
                                    root,
                                    *rank,
                                    candidate,
                                );
                            } else {
                                let decision = UnestablishedStructuralCandidate::new(
                                    &candidate,
                                    token.as_ref(),
                                    requirement.as_ref(),
                                    reasons,
                                );
                                site_unestablished_by_id
                                    .entry(decision.decision_id.clone())
                                    .or_insert(decision);
                            }
                        }
                        // An unresolved exact P5 route is neither a proven
                        // capability violation nor a contract discharge. It
                        // must not be reclassified as P4: generic pointer
                        // layout evidence alone has no internal derivation
                        // fact. Keep its unknown status out of selection;
                        // the context/index diagnostic remains available to
                        // the caller rather than recording a hard suppression.
                        SourceEligibility::Unsupported => {}
                    }
                }
            }
            for (_, eligible) in eligible_by_root.into_values() {
                selected.extend(eligible);
            }
            trace.finish_role()?;
        }
        prefer_relational_subject(contract, &mut selected);
        let downstream_sources = downstream_selected_sources(&condensation, &selected);
        selected.retain(|candidate| {
            !downstream_sources.contains(&(
                candidate.public_root.clone(),
                candidate.operand_role.clone(),
                candidate.source.as_ref().clone(),
            ))
        });
        for candidate in selected {
            insert_candidate_first_wins(&mut site_candidates_by_key, candidate);
        }
        candidates.extend(site_candidates_by_key.into_values());
        suppressions.extend(site_suppressions_by_key.into_values());
        unestablished.extend(site_unestablished_by_id.into_values());
    }

    // None of these indexes participate in candidate identity or ordering
    // after discovery.  The production Engine additionally releases its
    // coarse/context owners before calling `into_global_candidate_order`.
    drop(source_interner);
    drop(root_identity_cache);
    drop(call_route_index);
    drop(condensation);
    drop(route_index);
    trace.finish_sites(candidates.len(), suppressions.len())?;
    trace.event(
        "structural_site_scan",
        "end",
        [
            ("site_started_count", trace.site_started_count),
            ("site_completed_count", trace.site_started_count),
            ("role_started_count", trace.role_started_count),
            ("role_completed_count", trace.role_completed_count),
            ("discovered_source_count", trace.discovered_source_count),
            ("eligibility_attempt_count", trace.eligibility_attempt_count),
            (
                "eligibility_completed_count",
                trace.eligibility_completed_count,
            ),
            (
                "p5_eligibility_attempt_count",
                trace.p5_eligibility_attempt_count,
            ),
            (
                "p5_eligibility_completed_count",
                trace.p5_eligibility_completed_count,
            ),
            ("candidate_count", candidates.len() as u64),
            ("suppression_count", suppressions.len() as u64),
            ("unestablished_count", unestablished.len() as u64),
        ],
    )?;
    Ok(StructuralCandidateSet {
        active: candidates,
        suppressed: suppressions,
        unestablished,
    })
}

fn intern_candidate_source(
    interner: &mut HashSet<Arc<EndpointNode>>,
    source: &EndpointNode,
) -> Arc<EndpointNode> {
    if let Some(existing) = interner.get(source) {
        return existing.clone();
    }
    let source = Arc::new(source.clone());
    interner.insert(source.clone());
    source
}

fn globally_ordered_candidates(
    candidates: Vec<StructuralFindingCandidate>,
) -> Vec<StructuralFindingCandidate> {
    let keyed = candidates
        .into_iter()
        .enumerate()
        .map(|(ordinal, candidate)| (candidate_key(&candidate), ordinal, candidate))
        .collect::<Vec<_>>();
    globally_ordered_first_wins(keyed)
}

fn globally_ordered_suppressions(
    suppressions: Vec<SuppressedStructuralCandidate>,
) -> Vec<SuppressedStructuralCandidate> {
    let keyed = suppressions
        .into_iter()
        .enumerate()
        .map(|(ordinal, suppression)| (candidate_key(&suppression.candidate), ordinal, suppression))
        .collect::<Vec<_>>();
    globally_ordered_first_wins(keyed)
}

fn globally_ordered_unestablished(
    mut decisions: Vec<UnestablishedStructuralCandidate>,
) -> Vec<UnestablishedStructuralCandidate> {
    decisions.sort_unstable_by(|left, right| left.decision_id.cmp(&right.decision_id));
    decisions.dedup_by(|left, right| left.decision_id == right.decision_id);
    decisions
}

fn globally_ordered_first_wins<T>(mut keyed: Vec<(String, usize, T)>) -> Vec<T> {
    // The explicit discovery ordinal preserves first-wins even with an
    // in-place unstable sort, avoiding stable sort's O(N) scratch allocation
    // when production has hundreds of thousands of retained records.
    keyed.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut ordered = Vec::with_capacity(keyed.len());
    let mut previous_key = None::<String>;
    for (key, _, value) in keyed {
        if previous_key
            .as_ref()
            .is_some_and(|previous| previous == &key)
        {
            continue;
        }
        previous_key = Some(key);
        ordered.push(value);
    }
    ordered
}

fn insert_candidate_first_wins(
    candidates_by_key: &mut BTreeMap<String, StructuralFindingCandidate>,
    candidate: StructuralFindingCandidate,
) {
    assert!(
        candidate.primary_pattern == PrimaryPattern::P3 || candidate.exposure.is_none(),
        "only P3 candidates may retain an exposure"
    );
    candidates_by_key
        .entry(candidate_key(&candidate))
        .or_insert(candidate);
}

fn heuristic_unmodeled_public_source(
    pattern: PrimaryPattern,
    source_kind: CausalSourceKind,
    reasons: &BTreeSet<UnestablishedReason>,
) -> bool {
    matches!(pattern, PrimaryPattern::P1 | PrimaryPattern::P2)
        && matches!(
            source_kind,
            CausalSourceKind::PublicParameter | CausalSourceKind::PublicField
        )
        && reasons.len() == 1
        && reasons.contains(&UnestablishedReason::UnmodeledUnsafeCall)
}

fn insert_best_rank_candidate(
    eligible_by_root: &mut BTreeMap<Arc<FunctionKey>, (u8, Vec<StructuralFindingCandidate>)>,
    root: Arc<FunctionKey>,
    rank: u8,
    candidate: StructuralFindingCandidate,
) {
    match eligible_by_root.entry(root) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert((rank, vec![candidate]));
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            let (best_rank, candidates) = entry.get_mut();
            if rank < *best_rank {
                *best_rank = rank;
                candidates.clear();
                candidates.push(candidate);
            } else if rank == *best_rank {
                candidates.push(candidate);
            }
        }
    }
}

fn prefer_relational_subject(
    contract: &ContractDescriptor,
    selected: &mut Vec<StructuralFindingCandidate>,
) {
    if !matches!(
        contract.model_id.as_deref(),
        Some("slice_get_unchecked_v1" | "vec_get_unchecked_v1")
    ) {
        return;
    }
    let variable_index = selected.first().is_some_and(|candidate| {
        candidate
            .site()
            .operands
            .iter()
            .any(|operand| operand.role == OperandRole::Argument(1) && operand.place.is_some())
    });
    if variable_index {
        selected.retain(|candidate| {
            candidate.operand_role != Some(OperandRole::Argument(0))
                || candidate.primary_pattern != PrimaryPattern::P4
        });
    }
    let roots_with_index = selected
        .iter()
        .filter(|candidate| candidate.operand_role == Some(OperandRole::Argument(1)))
        .map(|candidate| (candidate.public_root.clone(), candidate.primary_pattern))
        .collect::<BTreeSet<_>>();
    selected.retain(|candidate| {
        candidate.operand_role != Some(OperandRole::Argument(0))
            || !roots_with_index
                .contains(&(candidate.public_root.clone(), candidate.primary_pattern))
    });
}

fn downstream_selected_sources(
    condensation: &EndpointCondensation<'_, '_>,
    selected: &[StructuralFindingCandidate],
) -> BTreeSet<(Arc<FunctionKey>, Option<OperandRole>, EndpointNode)> {
    let mut groups =
        BTreeMap::<(Arc<FunctionKey>, Option<OperandRole>), BTreeSet<EndpointNode>>::new();
    for candidate in selected {
        groups
            .entry((
                candidate.public_root.clone(),
                candidate.operand_role.clone(),
            ))
            .or_default()
            .insert(candidate.source.as_ref().clone());
    }

    let mut downstream = BTreeSet::new();
    for ((root, role), sources) in groups {
        for source in condensation.nodes_reached_from_another_in_set(&sources) {
            downstream.insert((root.clone(), role.clone(), source));
        }
    }
    downstream
}

struct P3ExposureCandidates {
    active: Vec<StructuralFindingCandidate>,
    suppressed: Vec<SuppressedStructuralCandidate>,
    unestablished: Vec<UnestablishedStructuralCandidate>,
}

fn p3_exposures(
    graph: &EndpointFlowGraph,
    route_index: &EndpointRouteIndex<'_>,
    call_route_index: &CallRouteIndex<'_>,
    site_context: &Arc<CandidateSiteContext>,
    structural: &StructuralSiteCandidate,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    public_root_routes: &BTreeMap<Arc<FunctionKey>, (Arc<StableSpan>, Arc<[CandidateCallEdge]>)>,
    context_proofs: Option<&ContextProofIndex>,
    influence_seeds: Option<&InfluenceSeedIndex>,
) -> Option<P3ExposureCandidates> {
    let site = &site_context.site;
    let contract = &site_context.contract;
    let validity_sensitive_result = structural.validity_sensitive_result
        || contract
            .predicates
            .contains(&super::contract_catalog::ContractPredicate::ValidForType);
    if site.result_place.is_none()
        || !validity_sensitive_result
        || contract.failure_phase != FailurePhase::ImmediateValidity
    {
        return None;
    }
    let source = Arc::new(EndpointNode::UnsafeResult {
        owner: site.seed.owner.clone(),
        site_id: site.seed.site_id(),
    });
    let first_failure = Arc::new(site_context.first_failure.clone());
    if !graph.nodes.contains(source.as_ref()) {
        return None;
    }
    let routes_from_source = route_index.routes_from(source.as_ref());
    let exposure_nodes = routes_from_source
        .reachable_nodes()
        .filter(|node| match node {
            EndpointNode::Return { owner } | EndpointNode::Out { owner, .. } => {
                owners.get(owner).is_some_and(|info| info.public_safe)
            }
            EndpointNode::Exposure { .. } => true,
            _ => false,
        })
        .cloned()
        .collect::<Vec<_>>();
    if exposure_nodes.is_empty() {
        return None;
    }
    let public_formal_inputs = contract
        .relevant_roles
        .iter()
        .filter_map(|role| {
            let target = EndpointNode::UnsafeOperand {
                owner: site.seed.owner.clone(),
                site_id: site.seed.site_id(),
                role: role.clone(),
            };
            graph.nodes.contains(&target).then_some((role, target))
        })
        .flat_map(|(role, target)| {
            let routes_to_target = route_index.routes_to(&target);
            routes_to_target
                .reaching_nodes()
                .filter_map(|input| {
                    let EndpointNode::Formal { owner, .. } = input else {
                        return None;
                    };
                    if !owners.get(owner).is_some_and(|info| info.public_safe) {
                        return None;
                    }
                    let endpoint_route = routes_to_target.route_from(input)?;
                    if endpoint_route.iter().any(|edge| {
                        matches!(&edge.from, EndpointNode::InternalDerivation { .. })
                            || matches!(&edge.to, EndpointNode::InternalDerivation { .. })
                    }) {
                        return None;
                    }
                    let (root_span, call_route) = public_root_routes.get(owner)?;
                    Some((
                        public_root_routes
                            .get_key_value(owner)
                            .expect("public formal root route was found above")
                            .0
                            .clone(),
                        root_span.clone(),
                        call_route.clone(),
                        Arc::new(input.clone()),
                        role.clone(),
                        Arc::<[EndpointEdge]>::from(endpoint_route),
                    ))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let exact_reverse_by_role = context_proofs
        .map(|context| {
            contract
                .relevant_roles
                .iter()
                .cloned()
                .map(|role| {
                    let target = EndpointNode::UnsafeOperand {
                        owner: site.seed.owner.clone(),
                        site_id: site.seed.site_id(),
                        role: role.clone(),
                    };
                    (role, context.exact_reverse_reachability(&target))
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let mut influence_eligibility_by_input =
        BTreeMap::<(Arc<FunctionKey>, Arc<EndpointNode>, OperandRole), SourceEligibility>::new();
    let mut result = P3ExposureCandidates {
        active: Vec::new(),
        suppressed: Vec::new(),
        unestablished: Vec::new(),
    };
    for exposure_node in &exposure_nodes {
        let exposure_owner = match exposure_node {
            EndpointNode::Return { owner } | EndpointNode::Out { owner, .. } => owner,
            EndpointNode::Exposure { owner, .. } => owner,
            _ => unreachable!("exposure candidates are filtered above"),
        };
        let Some(endpoint_route) = routes_from_source.route_to(exposure_node) else {
            continue;
        };
        let endpoint_route = Arc::<[EndpointEdge]>::from(endpoint_route);
        let exact_exposure_context = context_proofs.is_some_and(|context| {
            context.has_exact_route_in_context(
                &site.seed.owner,
                source.as_ref(),
                exposure_owner,
                exposure_node,
            )
        });
        let Some(exposure_span) = node_span(exposure_node, owners) else {
            continue;
        };
        let exposure = Arc::new(CandidateExposure {
            node: exposure_node.clone(),
            span: exposure_span.clone(),
        });
        let roots = match exposure_node {
            EndpointNode::Return { owner } | EndpointNode::Out { owner, .. }
                if owners.get(owner).is_some_and(|info| info.public_safe) =>
            {
                public_root_routes
                    .get(owner)
                    .cloned()
                    .and_then(|route| {
                        public_root_routes
                            .get_key_value(owner)
                            .map(|(root, _)| BTreeMap::from([(root.clone(), route)]))
                    })
                    .unwrap_or_default()
            }
            _ if exposure_owner == &site.seed.owner => public_root_routes.clone(),
            _ => reachable_public_roots(owners, &call_route_index.routes_to(exposure_owner))
                .into_keys()
                .filter_map(|root| {
                    public_root_routes
                        .get(&root)
                        .cloned()
                        .map(|route| (root, route))
                })
                .collect(),
        };
        for (root, (root_span, call_route)) in roots {
            let mut emitted_public_formal = false;
            for (input_root, input_root_span, input_call_route, input, input_role, input_route) in
                public_formal_inputs
                    .iter()
                    .filter(|(input_root, ..)| input_root == &root)
            {
                let Some(input_span) = node_span(input, owners) else {
                    continue;
                };
                let mut candidate = StructuralFindingCandidate {
                    primary_pattern: PrimaryPattern::P1,
                    mechanism_id: MechanismId::P1PublicParameterFlow,
                    classification_confidence: ClassificationConfidence::Structural,
                    public_root: input_root.clone(),
                    public_root_span: input_root_span.clone(),
                    call_route: input_call_route.clone(),
                    source: input.clone(),
                    source_kind: CausalSourceKind::PublicParameter,
                    source_span: input_span,
                    site_context: site_context.clone(),
                    operand_role: Some(input_role.clone()),
                    exposure: None,
                    endpoint_route: input_route.clone(),
                };
                let eligibility = if let (Some(root_owner), Some(influence_seeds)) =
                    (owners.get(root.as_ref()), influence_seeds)
                {
                    let target = EndpointNode::UnsafeOperand {
                        owner: site.seed.owner.clone(),
                        site_id: site.seed.site_id(),
                        role: input_role.clone(),
                    };
                    let cache_key = (input_root.clone(), input.clone(), input_role.clone());
                    if let Some(eligibility) = influence_eligibility_by_input.get(&cache_key) {
                        eligibility.clone()
                    } else {
                        let eligibility = exact_influence_eligibility(
                            PrimaryPattern::P1,
                            input.as_ref(),
                            root_owner,
                            input_role,
                            contract,
                            context_proofs,
                            &target,
                            influence_seeds,
                            exact_reverse_by_role.get(input_role),
                        );
                        influence_eligibility_by_input.insert(cache_key, eligibility.clone());
                        eligibility
                    }
                } else {
                    SourceEligibility::Active {
                        supporting_call_route: None,
                        supporting_endpoint_route: None,
                    }
                };
                match eligibility {
                    SourceEligibility::Active {
                        supporting_call_route,
                        supporting_endpoint_route,
                    } => {
                        if influence_seeds.is_some() {
                            candidate.classification_confidence = ClassificationConfidence::Exact;
                        }
                        if let Some(call_route) = supporting_call_route {
                            candidate.call_route = call_route.into();
                        }
                        if let Some(endpoint_route) = supporting_endpoint_route {
                            candidate.endpoint_route = endpoint_route.into();
                        }
                        emitted_public_formal = true;
                        result.active.push(candidate);
                    }
                    SourceEligibility::Suppressed(reason, evidence) => result.suppressed.push(
                        SuppressedStructuralCandidate::new(candidate, reason, evidence),
                    ),
                    SourceEligibility::Unestablished {
                        token,
                        requirement,
                        reasons,
                    } => {
                        if heuristic_unmodeled_public_source(
                            candidate.primary_pattern,
                            candidate.source_kind,
                            &reasons,
                        ) {
                            candidate.classification_confidence =
                                ClassificationConfidence::Heuristic;
                            emitted_public_formal = true;
                            result.active.push(candidate);
                        } else {
                            result
                                .unestablished
                                .push(UnestablishedStructuralCandidate::new(
                                    &candidate,
                                    token.as_ref(),
                                    requirement.as_ref(),
                                    reasons,
                                ));
                        }
                    }
                    SourceEligibility::Reclassified { .. } | SourceEligibility::Unsupported => {}
                }
            }
            if emitted_public_formal {
                continue;
            }
            result.active.push(StructuralFindingCandidate {
                primary_pattern: PrimaryPattern::P3,
                mechanism_id: MechanismId::P3InternalUnsafeExposure,
                classification_confidence: if exact_exposure_context {
                    ClassificationConfidence::Exact
                } else {
                    ClassificationConfidence::Structural
                },
                public_root: root,
                public_root_span: root_span,
                call_route: call_route.into(),
                source: source.clone(),
                source_kind: CausalSourceKind::InternalUnsafeOrigin,
                source_span: first_failure.clone(),
                site_context: site_context.clone(),
                operand_role: None,
                exposure: Some(exposure.clone()),
                endpoint_route: endpoint_route.clone(),
            });
        }
    }
    (!result.active.is_empty() || !result.suppressed.is_empty() || !result.unestablished.is_empty())
        .then_some(result)
}

fn source_classification(
    source: &EndpointNode,
) -> Option<(PrimaryPattern, CausalSourceKind, MechanismId, u8)> {
    match source {
        EndpointNode::ExternalBehavior { .. } => Some((
            PrimaryPattern::P6,
            CausalSourceKind::ExternalBehaviorOutput,
            MechanismId::P6ExternalBehaviorFlow,
            0,
        )),
        EndpointNode::GenericCapability { .. } => Some((
            PrimaryPattern::P5,
            CausalSourceKind::GenericCapabilityGap,
            MechanismId::P5GenericCapabilityGap,
            1,
        )),
        EndpointNode::PublicField { .. } => Some((
            PrimaryPattern::P2,
            CausalSourceKind::PublicField,
            MechanismId::P2PublicStateFlow,
            2,
        )),
        EndpointNode::InternalDerivation { .. } => Some((
            PrimaryPattern::P4,
            CausalSourceKind::InternalDerivation,
            MechanismId::P4InternalDerivation,
            3,
        )),
        EndpointNode::Formal { .. } => Some((
            PrimaryPattern::P1,
            CausalSourceKind::PublicParameter,
            MechanismId::P1PublicParameterFlow,
            4,
        )),
        EndpointNode::Capture { .. }
        | EndpointNode::UnsafeOperand { .. }
        | EndpointNode::UnsafeResult { .. }
        | EndpointNode::Return { .. }
        | EndpointNode::Out { .. }
        | EndpointNode::CallResult { .. }
        | EndpointNode::Exposure { .. } => None,
    }
}

fn derivation_is_relevant(
    kind: DerivationKind,
    role: &OperandRole,
    contract: &ContractDescriptor,
) -> bool {
    use super::contract_catalog::ContractPredicate;

    let has = |predicate| contract.predicates.contains(&predicate);
    let bounds = has(ContractPredicate::InBounds) || has(ContractPredicate::RangeInBounds);
    let memory_access = has(ContractPredicate::ValidForRead) || bounds;
    let layout = has(ContractPredicate::Aligned) || has(ContractPredicate::LayoutCompatible);

    match (kind, role) {
        (DerivationKind::Index, OperandRole::Index)
        | (DerivationKind::Offset, OperandRole::Index)
        | (DerivationKind::Offset, OperandRole::Address)
        | (DerivationKind::Length, OperandRole::Length)
        | (DerivationKind::Length, OperandRole::Size)
        | (DerivationKind::AccessWidth, OperandRole::Size)
        | (DerivationKind::Alignment, OperandRole::Alignment)
        | (DerivationKind::Layout, OperandRole::Layout)
        | (DerivationKind::Discriminant, OperandRole::Discriminant) => true,
        (DerivationKind::Index | DerivationKind::Offset, OperandRole::Argument(1)) => bounds,
        (DerivationKind::Length, OperandRole::Argument(0)) => bounds,
        (
            DerivationKind::Offset | DerivationKind::AccessWidth,
            OperandRole::Argument(0) | OperandRole::Place,
        ) => memory_access,
        (
            DerivationKind::Alignment | DerivationKind::Layout,
            OperandRole::Argument(0) | OperandRole::Address | OperandRole::Place,
        ) => layout,
        (DerivationKind::UncheckedBooleanRelation, OperandRole::Index) => bounds,
        (DerivationKind::UncheckedBooleanRelation, OperandRole::Discriminant) => {
            has(ContractPredicate::ValidForType)
        }
        (DerivationKind::UncheckedBooleanRelation, OperandRole::Argument(0)) => {
            has(ContractPredicate::ConditionTrue)
        }
        _ => false,
    }
}

#[derive(Clone)]
enum SourceEligibility {
    Active {
        supporting_call_route: Option<Vec<CandidateCallEdge>>,
        supporting_endpoint_route: Option<Vec<EndpointEdge>>,
    },
    Reclassified {
        supporting_call_route: Vec<CandidateCallEdge>,
        supporting_endpoint_route: Vec<EndpointEdge>,
        reason: SuppressionReason,
        evidence: BTreeMap<String, String>,
        pattern: PrimaryPattern,
        source_kind: CausalSourceKind,
        mechanism_id: MechanismId,
        rank: u8,
    },
    Suppressed(SuppressionReason, BTreeMap<String, String>),
    Unestablished {
        token: Option<super::influence::InfluenceToken>,
        requirement: Option<SinkRequirement>,
        reasons: BTreeSet<UnestablishedReason>,
    },
    Unsupported,
}

fn source_eligibility(
    source: &EndpointNode,
    root: &CandidateOwner,
    role: &OperandRole,
    contract: &ContractDescriptor,
    call_route_index: &CallRouteIndex<'_>,
) -> SourceEligibility {
    source_eligibility_with_context(source, root, role, contract, call_route_index, None, source)
}

fn source_eligibility_with_context(
    source: &EndpointNode,
    root: &CandidateOwner,
    role: &OperandRole,
    contract: &ContractDescriptor,
    call_route_index: &CallRouteIndex<'_>,
    context_proofs: Option<&ContextProofIndex>,
    target: &EndpointNode,
) -> SourceEligibility {
    match source {
        EndpointNode::ExternalBehavior {
            kind,
            controller,
            token,
            ..
        } => external_behavior_suppression(*kind, controller, token).map_or(
            SourceEligibility::Active {
                supporting_call_route: None,
                supporting_endpoint_route: None,
            },
            |(reason, evidence)| SourceEligibility::Suppressed(reason, evidence),
        ),
        EndpointNode::GenericCapability {
            owner, evidence, ..
        } => generic_capability_eligibility(
            evidence,
            owner,
            root,
            role,
            contract,
            call_route_index,
            context_proofs,
            source,
            target,
        ),
        EndpointNode::PublicField {
            access_owner,
            def_path,
            access_path,
            base,
            storage,
            storage_versions,
            adt_kind,
            persistent: false,
            ..
        } => SourceEligibility::Suppressed(
            SuppressionReason::NonpersistentPublicField,
            BTreeMap::from([
                ("access_owner".to_string(), access_owner.0.clone()),
                ("field".to_string(), def_path.clone()),
                ("access_path".to_string(), access_path.0.clone()),
                ("base".to_string(), base.0.clone()),
                ("storage".to_string(), storage.as_str().to_string()),
                ("adt_kind".to_string(), adt_kind.as_str().to_string()),
                (
                    "storage_versions".to_string(),
                    storage_versions
                        .iter()
                        .map(|version| version.stable_identity())
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ]),
        ),
        EndpointNode::PublicField {
            persistent: true, ..
        } => SourceEligibility::Active {
            supporting_call_route: None,
            supporting_endpoint_route: None,
        },
        EndpointNode::InternalDerivation { .. } => {
            exact_derivation_eligibility(source, root, role, contract, context_proofs, target)
        }
        EndpointNode::Formal { .. }
        | EndpointNode::Capture { .. }
        | EndpointNode::UnsafeOperand { .. }
        | EndpointNode::UnsafeResult { .. }
        | EndpointNode::Return { .. }
        | EndpointNode::Out { .. }
        | EndpointNode::CallResult { .. }
        | EndpointNode::Exposure { .. } => SourceEligibility::Active {
            supporting_call_route: None,
            supporting_endpoint_route: None,
        },
    }
}

fn exact_derivation_eligibility(
    source: &EndpointNode,
    root: &CandidateOwner,
    role: &OperandRole,
    contract: &ContractDescriptor,
    context_proofs: Option<&ContextProofIndex>,
    target: &EndpointNode,
) -> SourceEligibility {
    let requirement = SinkRequirement::from_contract(contract, role).ok();
    let Some(context) = context_proofs else {
        // The context-free constructor is retained for focused structural
        // unit fixtures. Production always supplies the exact context index
        // and therefore never admits this structural-only branch.
        return SourceEligibility::Active {
            supporting_call_route: None,
            supporting_endpoint_route: None,
        };
    };
    match context.query_via_scoped(
        root.function.clone(),
        source.clone(),
        root.function.clone(),
        source.clone(),
        root.function.clone(),
        target.clone(),
    ) {
        ContextQuery::Realizable(proof) => SourceEligibility::Active {
            supporting_call_route: Some(
                proof
                    .canonical_callsites()
                    .into_iter()
                    .map(|(callsite, substitution)| {
                        candidate_call_from_context(&callsite, &substitution)
                    })
                    .collect(),
            ),
            // The compact exact proof is the admission authority. Keep the
            // already selected semantic route for display so recursive proof
            // DAGs are not expanded into an arbitrary call stack.
            supporting_endpoint_route: None,
        },
        ContextQuery::ProvenUnrealizable(_) => SourceEligibility::Suppressed(
            SuppressionReason::UnrealizableCallContext,
            BTreeMap::from([
                (
                    "proof_scope".to_string(),
                    "closed_exact_corridor".to_string(),
                ),
                (
                    "requirement".to_string(),
                    requirement
                        .as_ref()
                        .map_or_else(|| "unknown".to_string(), SinkRequirement::stable_identity),
                ),
            ]),
        ),
        ContextQuery::Unknown { reasons, .. } => {
            let reason = if reasons.iter().any(|reason| {
                matches!(reason, ContextUnknownReason::NormalizationIncomplete { .. })
            }) {
                UnestablishedReason::IncompleteNormalization
            } else {
                UnestablishedReason::MissingExactContext
            };
            SourceEligibility::Unestablished {
                token: None,
                requirement,
                reasons: BTreeSet::from([reason]),
            }
        }
    }
}

/// Compact production admission for a source-local P4 derivation.
///
/// `exact_derivation_eligibility` historically constructed a complete owned
/// `ContextProof`, including every relation alternative and recursive-SCC
/// display record, for every source candidate. P4 only asks whether the
/// already scoped derivation atom lies on the exact corridor to this operand:
/// the query's execution root and source are deliberately the same atom.
/// The reusable reverse mask is therefore the same positive witness without
/// materializing the full proof closure. Missing/ambiguous atoms stay
/// `unestablished`; this fast path cannot turn absence into a safety proof or
/// an active result.
fn exact_derivation_eligibility_in_reverse(
    source: &EndpointNode,
    root: &CandidateOwner,
    role: &OperandRole,
    contract: &ContractDescriptor,
    context: &ContextProofIndex,
    reverse: &ExactReverseReachability,
) -> SourceEligibility {
    let requirement = SinkRequirement::from_contract(contract, role).ok();
    let Some(start) = context.unique_exact_atom_handle(source, &root.function) else {
        return SourceEligibility::Unestablished {
            token: None,
            requirement,
            reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
        };
    };
    if !reverse.contains(start) || reverse.target_in_owner(&root.function).is_none() {
        return SourceEligibility::Unestablished {
            token: None,
            requirement,
            reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
        };
    }
    SourceEligibility::Active {
        supporting_call_route: None,
        supporting_endpoint_route: None,
    }
}

fn exact_influence_eligibility(
    pattern: PrimaryPattern,
    source: &EndpointNode,
    root: &CandidateOwner,
    role: &OperandRole,
    contract: &ContractDescriptor,
    context_proofs: Option<&ContextProofIndex>,
    target: &EndpointNode,
    influence_seeds: &InfluenceSeedIndex,
    exact_reverse: Option<&ExactReverseReachability>,
) -> SourceEligibility {
    if pattern == PrimaryPattern::P6 {
        if let EndpointNode::ExternalBehavior {
            kind,
            controller,
            token,
            ..
        } = source
        {
            if let Some((reason, evidence)) =
                external_behavior_suppression(*kind, controller, token)
            {
                return SourceEligibility::Suppressed(reason, evidence);
            }
        }
    }
    let seed = influence_seeds.get(source).cloned();
    let Some(seed) = seed else {
        return SourceEligibility::Unestablished {
            token: None,
            requirement: SinkRequirement::from_contract(contract, role).ok(),
            reasons: BTreeSet::from([UnestablishedReason::UnknownCapabilityMatch]),
        };
    };
    let origin_matches = match pattern {
        PrimaryPattern::P1 => matches!(
            seed.origin,
            InfluenceOrigin::PublicFormal | InfluenceOrigin::PersistentPublicField
        ),
        PrimaryPattern::P2 => seed.origin == InfluenceOrigin::PersistentPublicField,
        PrimaryPattern::P6 => matches!(
            seed.origin,
            InfluenceOrigin::FfiReturn
                | InfluenceOrigin::FfiOut
                | InfluenceOrigin::FfiCallbackInput
                | InfluenceOrigin::OpenTraitReturn
                | InfluenceOrigin::PublicCallbackReturn
                | InfluenceOrigin::PublicFunctionPointerReturn
        ),
        PrimaryPattern::P3 | PrimaryPattern::P4 | PrimaryPattern::P5 => false,
    };
    if !origin_matches {
        return SourceEligibility::Unsupported;
    }
    let requirement = match SinkRequirement::from_contract(contract, role) {
        Ok(requirement) => requirement,
        Err(reason) => {
            return SourceEligibility::Unestablished {
                token: Some(seed),
                requirement: None,
                reasons: BTreeSet::from([reason]),
            };
        }
    };
    // Unknown type validity and lifetime/provenance-sensitive transmutes do
    // not establish that the public input can violate the target. Exact
    // dataflow alone proves influence, not a typed incompatibility witness,
    // so retain the decision as auditable uncertainty instead of promoting
    // it to a high-confidence P1 finding.
    if pattern == PrimaryPattern::P1
        && matches!(
            contract.static_implication,
            StaticContractImplication::Unproven(
                StaticContractGap::UnknownTypeValidity
                    | StaticContractGap::LifetimeOrProvenanceSensitive
            )
        )
    {
        return SourceEligibility::Unestablished {
            token: Some(seed),
            requirement: Some(requirement),
            reasons: BTreeSet::from([UnestablishedReason::UnknownCapabilityMatch]),
        };
    }
    // A target-restricted conversion reached through a helper boundary needs
    // a transported must-validation certificate. Exact value flow proves
    // that the caller input reaches the conversion, but it does not prove
    // that a wrapper omitted the target-range check. Keep the direct same-body
    // lane active-capable and report the unsupported cross-body case as
    // explicit uncertainty.
    if pattern == PrimaryPattern::P1
        && matches!(
            contract.static_implication,
            StaticContractImplication::Unproven(StaticContractGap::TargetRestrictedValidity)
        )
        && matches!(target, EndpointNode::UnsafeOperand { owner, .. } if owner != &root.function)
    {
        return SourceEligibility::Unestablished {
            token: Some(seed),
            requirement: Some(requirement),
            reasons: BTreeSet::from([UnestablishedReason::ValidationStateUnknown]),
        };
    }
    let Some(context) = context_proofs else {
        return SourceEligibility::Unestablished {
            token: Some(seed),
            requirement: Some(requirement),
            reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
        };
    };
    let outcome = if let Some(reverse) = exact_reverse {
        let Some(start) = context.unique_exact_atom_handle(source, &root.function) else {
            return SourceEligibility::Unestablished {
                token: Some(seed),
                requirement: Some(requirement),
                reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
            };
        };
        let Some(target) = reverse.target_in_owner(&root.function) else {
            return SourceEligibility::Unestablished {
                token: Some(seed),
                requirement: Some(requirement),
                reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
            };
        };
        if !reverse.contains(start) {
            return SourceEligibility::Unestablished {
                token: Some(seed),
                requirement: Some(requirement),
                reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
            };
        }
        InfluenceSolver::new(context).solve_handles(
            start,
            target,
            seed.clone(),
            &requirement,
            reverse,
        )
    } else {
        InfluenceSolver::new(context).solve(
            root.function.clone(),
            source,
            root.function.clone(),
            target,
            seed.clone(),
            &requirement,
        )
    };
    match outcome {
        InfluenceOutcome::Active(witness) => SourceEligibility::Active {
            supporting_call_route: Some(
                witness
                    .call_route(context)
                    .into_iter()
                    .map(|(callsite, substitution)| {
                        candidate_call_from_context(&callsite, &substitution)
                    })
                    .collect(),
            ),
            supporting_endpoint_route: Some(witness.endpoint_route(context)),
        },
        InfluenceOutcome::ProvenIncompatible(witness) => {
            let internal = matches!(witness.token.controller, InfluenceController::Internal);
            SourceEligibility::Suppressed(
                if internal {
                    SuppressionReason::InternalBehaviorController
                } else {
                    SuppressionReason::SourceCannotSatisfyRequirement
                },
                BTreeMap::from([
                    ("proof_scope".to_string(), "exact_token_route".to_string()),
                    (
                        "source_origin".to_string(),
                        witness.token.origin.as_str().to_string(),
                    ),
                    (
                        "controller".to_string(),
                        witness.token.controller.stable_identity(),
                    ),
                    (
                        "capabilities".to_string(),
                        witness.token.capabilities.stable_identity(),
                    ),
                    ("requirement".to_string(), requirement.stable_identity()),
                ]),
            )
        }
        InfluenceOutcome::Unestablished { reasons } => SourceEligibility::Unestablished {
            token: Some(seed),
            requirement: Some(requirement),
            reasons,
        },
    }
}

fn external_behavior_suppression(
    kind: super::endpoint_flow::ExternalSourceKind,
    controller: &BehaviorController,
    token: &str,
) -> Option<(SuppressionReason, BTreeMap<String, String>)> {
    let reason = controller.rejection_reason()?;
    let mut evidence = BTreeMap::from([
        (
            "source_kind".to_string(),
            external_source_kind_name(kind).to_string(),
        ),
        ("source_token".to_string(), token.to_string()),
    ]);
    match controller {
        BehaviorController::OpenTrait {
            trait_id,
            openness: BehaviorOpenness::ProvenClosed,
            resolved_target,
            ..
        } => {
            evidence.insert("controller_kind".to_string(), "open_trait".to_string());
            evidence.insert("openness".to_string(), "proven_closed".to_string());
            evidence.insert("trait_id".to_string(), trait_id.0.clone());
            evidence.insert(
                "resolved_target".to_string(),
                resolved_target
                    .as_ref()
                    .map(|target| target.0.clone())
                    .unwrap_or_else(|| "none".to_string()),
            );
        }
        BehaviorController::InternalFunction { function } => {
            evidence.insert(
                "controller_kind".to_string(),
                "internal_function".to_string(),
            );
            evidence.insert("controller".to_string(), function.0.clone());
        }
        BehaviorController::InternalClosure { function } => {
            evidence.insert(
                "controller_kind".to_string(),
                "internal_closure".to_string(),
            );
            evidence.insert("controller".to_string(), function.0.clone());
        }
        BehaviorController::OpenTrait { .. }
        | BehaviorController::PublicFormal { .. }
        | BehaviorController::PublicField { .. }
        | BehaviorController::FfiHost
        | BehaviorController::Unknown => {
            unreachable!("only structurally rejected controllers reach suppression materialization")
        }
    }
    Some((reason, evidence))
}

fn generic_capability_eligibility(
    evidence: &GenericCapabilityEvidence,
    source_owner: &FunctionKey,
    root: &CandidateOwner,
    role: &OperandRole,
    contract: &ContractDescriptor,
    call_route_index: &CallRouteIndex<'_>,
    context_proofs: Option<&ContextProofIndex>,
    source: &EndpointNode,
    target: &EndpointNode,
) -> SourceEligibility {
    let controllers = evidence.controllers();

    if matches!(
        evidence.kind,
        GenericCapabilityKind::RepresentationPreservingPointerView
    ) {
        return SourceEligibility::Suppressed(
            SuppressionReason::PointerViewHasNoCapabilityGap,
            generic_capability_evidence_map(
                evidence,
                root,
                role,
                contract,
                &controllers,
                "representation_preserving_view",
            ),
        );
    }

    match &evidence.witness {
        CapabilityWitness::Unestablished { reason } => {
            if context_proofs.is_none() {
                return SourceEligibility::Unsupported;
            }
            let reason = if *reason == CapabilityUnestablishedReason::NormalizationIncomplete {
                UnestablishedReason::IncompleteNormalization
            } else {
                UnestablishedReason::GenericWitnessUnknown
            };
            return SourceEligibility::Unestablished {
                token: None,
                requirement: SinkRequirement::from_contract(contract, role).ok(),
                reasons: BTreeSet::from([reason]),
            };
        }
        CapabilityWitness::ImpossibleByCompilerFacts { reason } => {
            return SourceEligibility::Suppressed(
                if *reason == CapabilityImpossibleReason::RepresentationPreservingView {
                    SuppressionReason::PointerViewHasNoCapabilityGap
                } else {
                    SuppressionReason::RootHasNoRelevantCapabilitySurface
                },
                generic_capability_evidence_map(
                    evidence,
                    root,
                    role,
                    contract,
                    &controllers,
                    &format!("compiler_impossible:{}", reason.as_str()),
                ),
            );
        }
        CapabilityWitness::Proven { .. } => {}
    }

    // A contract/role mismatch is not a safety proof. It only means this P5
    // refinement is unsupported; the retained value provenance remains
    // eligible for the lower-ranked P1/P4 classification.
    if contract.precision == super::unsafe_site::ContractPrecision::Opaque
        || !contract.relevant_roles.contains(role)
        || !capability_witness_matches_contract(evidence, contract)
    {
        if context_proofs.is_none() {
            return SourceEligibility::Unsupported;
        }
        return SourceEligibility::Unestablished {
            token: None,
            requirement: SinkRequirement::from_contract(contract, role).ok(),
            reasons: BTreeSet::from([
                if contract.precision == super::unsafe_site::ContractPrecision::Opaque {
                    UnestablishedReason::UnmodeledUnsafeCall
                } else {
                    UnestablishedReason::UnknownCapabilityMatch
                },
            ]),
        };
    }

    let route_result = context_proofs.map_or_else(
        || capability_route_result_from_legacy(call_route_index, root, source_owner, evidence),
        |context| capability_route_result_from_context(context, root, source, target, evidence),
    );
    match route_result {
        P5RouteOutcome::Parametric {
            call_route,
            endpoint_route,
        } => SourceEligibility::Active {
            supporting_call_route: Some(call_route),
            supporting_endpoint_route: endpoint_route,
        },
        P5RouteOutcome::ConcreteSatisfied => SourceEligibility::Suppressed(
            SuppressionReason::RootHasNoRelevantCapabilitySurface,
            generic_capability_evidence_map(
                evidence,
                root,
                role,
                contract,
                &controllers,
                "all_realizable_concrete_routes_proven_satisfied",
            ),
        ),
        P5RouteOutcome::ConcreteViolation {
            call_route,
            endpoint_route,
            violations,
        } => {
            let mut evidence = generic_capability_evidence_map(
                evidence,
                root,
                role,
                contract,
                &controllers,
                "concrete_route_proven_violation",
            );
            evidence.insert(
                "concrete_violations".to_string(),
                violations
                    .iter()
                    .map(|violation| violation.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            SourceEligibility::Reclassified {
                supporting_call_route: call_route,
                supporting_endpoint_route: endpoint_route.unwrap_or_default(),
                reason: SuppressionReason::RootHasNoRelevantCapabilitySurface,
                evidence,
                pattern: PrimaryPattern::P4,
                source_kind: CausalSourceKind::InternalDerivation,
                mechanism_id: MechanismId::P4ConcreteCapabilityViolation,
                rank: 3,
            }
        }
        P5RouteOutcome::Unestablished => SourceEligibility::Unestablished {
            token: None,
            requirement: SinkRequirement::from_contract(contract, role).ok(),
            reasons: BTreeSet::from([UnestablishedReason::GenericWitnessUnknown]),
        },
        P5RouteOutcome::Unreachable => SourceEligibility::Unestablished {
            token: None,
            requirement: SinkRequirement::from_contract(contract, role).ok(),
            reasons: BTreeSet::from([UnestablishedReason::MissingExactContext]),
        },
    }
}

fn capability_witness_matches_contract(
    evidence: &GenericCapabilityEvidence,
    contract: &ContractDescriptor,
) -> bool {
    let CapabilityWitness::Proven { violations, .. } = &evidence.witness else {
        return false;
    };
    violations.iter().any(|violation| match violation {
        super::endpoint_flow::CapabilityViolation::InBounds => {
            contract.predicates.iter().any(|predicate| {
                matches!(
                    predicate,
                    super::contract_catalog::ContractPredicate::InBounds
                        | super::contract_catalog::ContractPredicate::RangeInBounds
                )
            })
        }
        super::endpoint_flow::CapabilityViolation::Size => {
            contract.predicates.iter().any(|predicate| {
                matches!(
                    predicate,
                    super::contract_catalog::ContractPredicate::RangeInBounds
                        | super::contract_catalog::ContractPredicate::LayoutCompatible
                )
            })
        }
        super::endpoint_flow::CapabilityViolation::Alignment => {
            contract.predicates.iter().any(|predicate| {
                matches!(
                    predicate,
                    super::contract_catalog::ContractPredicate::Aligned
                        | super::contract_catalog::ContractPredicate::LayoutCompatible
                )
            })
        }
    })
}

fn generic_capability_evidence_map(
    evidence: &GenericCapabilityEvidence,
    root: &CandidateOwner,
    role: &OperandRole,
    contract: &ContractDescriptor,
    controllers: &BTreeSet<GenericParamId>,
    gate: &str,
) -> BTreeMap<String, String> {
    let associated_items = match &evidence.kind {
        GenericCapabilityKind::AssociatedSequenceShape { associated_items } => associated_items
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(","),
        GenericCapabilityKind::PointerLayoutReinterpretation
        | GenericCapabilityKind::RepresentationPreservingPointerView => String::new(),
    };
    BTreeMap::from([
        (
            "capability_kind".to_string(),
            evidence.kind.as_str().to_string(),
        ),
        (
            "controllers".to_string(),
            generic_param_set_identity(controllers),
        ),
        (
            "root_surface".to_string(),
            generic_param_set_identity(&root.generic_surface),
        ),
        ("associated_items".to_string(), associated_items),
        (
            "contract_precision".to_string(),
            match contract.precision {
                super::unsafe_site::ContractPrecision::Precise => "precise",
                super::unsafe_site::ContractPrecision::Generic => "generic",
                super::unsafe_site::ContractPrecision::Opaque => "opaque",
            }
            .to_string(),
        ),
        (
            "contract_model".to_string(),
            contract.model_id.clone().unwrap_or_default(),
        ),
        (
            "contract_predicates".to_string(),
            contract
                .predicates
                .iter()
                .map(|predicate| predicate.as_str())
                .collect::<Vec<_>>()
                .join(","),
        ),
        ("operand_role".to_string(), operand_role_identity(role)),
        ("gate".to_string(), gate.to_string()),
        (
            "requirement".to_string(),
            evidence.requirement.stable_identity(),
        ),
        ("witness".to_string(), evidence.witness.stable_identity()),
    ])
}

fn generic_param_set_identity(parameters: &BTreeSet<GenericParamId>) -> String {
    parameters
        .iter()
        .map(GenericParamId::stable_identity)
        .collect::<Vec<_>>()
        .join(",")
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

fn external_source_kind_name(kind: super::endpoint_flow::ExternalSourceKind) -> &'static str {
    use super::endpoint_flow::ExternalSourceKind;
    match kind {
        ExternalSourceKind::Iterator => "iterator",
        ExternalSourceKind::OpenTrait => "open_trait",
        ExternalSourceKind::Callback => "callback",
        ExternalSourceKind::FunctionPointer => "function_pointer",
        ExternalSourceKind::FfiReturn => "ffi_return",
        ExternalSourceKind::FfiOut => "ffi_out",
        ExternalSourceKind::Host => "host",
    }
}

fn candidate_root_identities<'cache, 'owners>(
    source: &EndpointNode,
    owners: &'owners BTreeMap<FunctionKey, CandidateOwner>,
    root_identity_cache: &'cache mut PublicRootIdentityCache<'owners, '_, '_>,
) -> CandidateRootIdentities<'cache, 'owners> {
    if let EndpointNode::Formal { owner, .. } = source {
        if owners.get(owner).is_some_and(|info| info.public_safe) {
            return CandidateRootIdentities::One(owner.clone());
        }
        return CandidateRootIdentities::None;
    }
    let source_owner = match source {
        EndpointNode::ExternalBehavior { owner, .. }
        | EndpointNode::GenericCapability { owner, .. }
        | EndpointNode::InternalDerivation { owner, .. }
        | EndpointNode::PublicField {
            access_owner: owner,
            ..
        } => Some(owner),
        EndpointNode::Capture { .. }
        | EndpointNode::UnsafeOperand { .. }
        | EndpointNode::UnsafeResult { .. }
        | EndpointNode::Return { .. }
        | EndpointNode::Out { .. }
        | EndpointNode::CallResult { .. }
        | EndpointNode::Exposure { .. }
        | EndpointNode::Formal { .. } => return CandidateRootIdentities::None,
    };
    match source_owner {
        Some(owner) => CandidateRootIdentities::Cached(root_identity_cache.roots_reaching(owner)),
        None => CandidateRootIdentities::None,
    }
}

fn reachable_public_root_identities<'owners>(
    owners: &'owners BTreeMap<FunctionKey, CandidateOwner>,
    routes_to_target: &CallRoutesTo<'_, '_>,
) -> Vec<&'owners FunctionKey> {
    owners
        .iter()
        .filter(|(_, owner)| owner.public_safe)
        .map(|(function, _)| function)
        .filter(|function| routes_to_target.reaches(function))
        .collect()
}

fn merge_ordered_endpoint_nodes<'node>(
    left: impl IntoIterator<Item = &'node EndpointNode>,
    right: impl IntoIterator<Item = &'node EndpointNode>,
) -> Vec<&'node EndpointNode> {
    let mut left = left.into_iter().peekable();
    let mut right = right.into_iter().peekable();
    let mut merged = Vec::new();
    loop {
        match (left.peek(), right.peek()) {
            (Some(left_node), Some(right_node)) => match left_node.cmp(right_node) {
                std::cmp::Ordering::Less => merged.push(
                    left.next()
                        .expect("peeked endpoint iterator must have its next item"),
                ),
                std::cmp::Ordering::Equal => {
                    merged.push(
                        left.next()
                            .expect("peeked endpoint iterator must have its next item"),
                    );
                    right.next();
                }
                std::cmp::Ordering::Greater => merged.push(
                    right
                        .next()
                        .expect("peeked endpoint iterator must have its next item"),
                ),
            },
            (Some(_), None) => {
                merged.extend(left);
                break;
            }
            (None, Some(_)) => {
                merged.extend(right);
                break;
            }
            (None, None) => break,
        }
    }
    merged
}

pub(super) fn reachable_public_roots(
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    routes_to_target: &CallRoutesTo<'_, '_>,
) -> BTreeMap<Arc<FunctionKey>, (Arc<StableSpan>, Arc<[CandidateCallEdge]>)> {
    owners
        .values()
        .filter(|owner| owner.public_safe)
        .filter_map(|owner| {
            routes_to_target.route_from(&owner.function).map(|route| {
                (
                    Arc::new(owner.function.clone()),
                    (
                        Arc::new(owner.root_span.clone()),
                        Arc::<[CandidateCallEdge]>::from(route),
                    ),
                )
            })
        })
        .collect()
}

fn node_span(
    node: &EndpointNode,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
) -> Option<Arc<StableSpan>> {
    match node {
        EndpointNode::Formal { owner, .. }
        | EndpointNode::Capture { owner, .. }
        | EndpointNode::Return { owner }
        | EndpointNode::Out { owner, .. }
        | EndpointNode::UnsafeOperand { owner, .. }
        | EndpointNode::UnsafeResult { owner, .. } => owners
            .get(owner)
            .map(|owner| Arc::new(owner.root_span.clone())),
        EndpointNode::CallResult { callsite, .. } => Some(Arc::new(callsite.point.span.clone())),
        EndpointNode::PublicField { span, .. }
        | EndpointNode::ExternalBehavior { span, .. }
        | EndpointNode::GenericCapability { span, .. } => Some(Arc::new(span.clone())),
        EndpointNode::InternalDerivation { point, .. } | EndpointNode::Exposure { point, .. } => {
            Some(Arc::new(point.span.clone()))
        }
    }
}

fn candidate_key(candidate: &StructuralFindingCandidate) -> String {
    let role = candidate
        .operand_role
        .as_ref()
        .map(|role| format!("{role:?}"))
        .unwrap_or_default();
    let exposure = candidate
        .exposure
        .as_ref()
        .map(|exposure| format!("{:?}", exposure.node))
        .unwrap_or_default();
    format!(
        "{:?}|{}|{}|{:?}|{}|{}",
        candidate.primary_pattern,
        candidate.public_root.0,
        candidate.site().seed.site_id(),
        candidate.source.as_ref(),
        role,
        exposure
    )
}

fn influence_decision_id(
    pattern: PrimaryPattern,
    public_root: &FunctionKey,
    site_id: &str,
    role: &OperandRole,
    source_identity: &str,
    source_kind: CausalSourceKind,
) -> String {
    let fields = [
        primary_pattern_identity(pattern).to_string(),
        public_root.0.clone(),
        site_id.to_string(),
        operand_role_identity(role),
        source_identity.to_string(),
        causal_source_kind_identity(source_kind).to_string(),
    ];
    let mut canonical = Vec::new();
    for field in fields {
        canonical.extend_from_slice(field.len().to_string().as_bytes());
        canonical.push(b':');
        canonical.extend_from_slice(field.as_bytes());
    }
    sha256_hex(&canonical)
}

fn primary_pattern_identity(pattern: PrimaryPattern) -> &'static str {
    match pattern {
        PrimaryPattern::P1 => "P1",
        PrimaryPattern::P2 => "P2",
        PrimaryPattern::P3 => "P3",
        PrimaryPattern::P4 => "P4",
        PrimaryPattern::P5 => "P5",
        PrimaryPattern::P6 => "P6",
    }
}

fn causal_source_kind_identity(kind: CausalSourceKind) -> &'static str {
    match kind {
        CausalSourceKind::PublicParameter => "public_parameter",
        CausalSourceKind::PublicField => "public_field",
        CausalSourceKind::InternalUnsafeOrigin => "internal_unsafe_origin",
        CausalSourceKind::InternalDerivation => "internal_derivation",
        CausalSourceKind::GenericCapabilityGap => "generic_capability_gap",
        CausalSourceKind::ExternalBehaviorOutput => "external_behavior_output",
    }
}

fn influence_source_identity(source: &EndpointNode) -> String {
    match source {
        EndpointNode::Formal { owner, index } => format!("{}:formal:{index}", owner.0),
        EndpointNode::PublicField {
            access_owner,
            def_path,
            access_path,
            base,
            storage,
            storage_versions,
            adt_kind,
            persistent,
            span,
        } => format!(
            "public_field:{}:{def_path}:{}:{}:{}:{}:{}:{persistent}@{}",
            access_owner.0,
            access_path.0,
            base.0,
            storage.as_str(),
            storage_versions
                .iter()
                .map(|version| version.stable_identity())
                .collect::<Vec<_>>()
                .join(","),
            adt_kind.as_str(),
            span.token(),
        ),
        EndpointNode::ExternalBehavior {
            owner,
            kind,
            token,
            span,
            ..
        } => format!(
            "{}:external:{}:{token}@{}",
            owner.0,
            external_source_kind_name(*kind),
            span.token(),
        ),
        EndpointNode::UnsafeResult { owner, site_id } => {
            format!("{}:unsafe_result:{site_id}", owner.0)
        }
        EndpointNode::InternalDerivation {
            owner,
            kind,
            token,
            point,
        } => format!(
            "{}:derivation:{}:{token}@{}",
            owner.0,
            kind.as_str(),
            point.span.token(),
        ),
        EndpointNode::GenericCapability {
            owner,
            evidence,
            span,
        } => format!(
            "{}:generic_capability:{}@{}",
            owner.0,
            evidence.source_identity(),
            span.token(),
        ),
        EndpointNode::Capture { owner, index } => format!("{}:capture:{index}", owner.0),
        EndpointNode::UnsafeOperand {
            owner,
            site_id,
            role,
        } => format!(
            "{}:unsafe_operand:{site_id}:{}",
            owner.0,
            operand_role_identity(role),
        ),
        EndpointNode::Return { owner } => format!("{}:return", owner.0),
        EndpointNode::Out { owner, index } => format!("{}:out:{index}", owner.0),
        EndpointNode::CallResult { callsite, output } => {
            format!("call_result:{}:{output:?}", callsite.stable_identity())
        }
        EndpointNode::Exposure {
            owner,
            token,
            point,
        } => {
            format!("{}:exposure:{token}@{}", owner.0, point.span.token())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::unsoundaudit::contract_catalog::{ContractPredicate, DischargeModel};
    use crate::analysis::unsoundaudit::endpoint_flow::{EndpointEdgeKind, ExternalSourceKind};
    use crate::analysis::unsoundaudit::summary::{PlaceKey, StablePosition};
    use crate::analysis::unsoundaudit::unsafe_site::{
        binding::{MirEventLocation, MirOperandAnchor, MirStage},
        ContractPrecision, RawSpanAnchor, ResultAnchor, UnsafeSiteKind, UnsafeSiteSeed,
    };

    fn span(line: u32) -> StableSpan {
        StableSpan::new(
            "src/lib.rs",
            StablePosition::new(line, 1),
            StablePosition::new(line, 2),
        )
    }

    fn point(owner: &FunctionKey, line: u32) -> ProgramPoint {
        ProgramPoint {
            function: owner.clone(),
            block: line,
            statement: 0,
            span: span(line + 1),
        }
    }

    fn bound_site(owner: &FunctionKey, result: Option<&str>) -> BoundUnsafeSite {
        let location = MirEventLocation {
            stage: MirStage::Solver,
            block: 9,
            statement: 0,
            event: 0,
        };
        BoundUnsafeSite {
            seed: UnsafeSiteSeed {
                adapter_id: "test".into(),
                owner: owner.clone(),
                owner_def_path_hash: "owner".into(),
                ordinal: 1,
                kind: UnsafeSiteKind::UnsafeCall,
                raw_span: RawSpanAnchor::new(span(10), 0, 1, "#0"),
                hir_context: "hir".into(),
                callee: Some("core::unsafe_call".into()),
                operands: Vec::new(),
                result: result.map(|ty| ResultAnchor {
                    span: RawSpanAnchor::new(span(10), 0, 1, "#0"),
                    type_fingerprint: ty.into(),
                    shape: "call".into(),
                }),
            },
            built: MirEventLocation {
                stage: MirStage::Built,
                ..location.clone()
            },
            solver: location,
            operands: vec![MirOperandAnchor {
                role: OperandRole::Argument(0),
                type_fingerprint: "*const u8".into(),
                place: Some(PlaceKey::new("_1")),
            }],
            result_place: result.map(|_| PlaceKey::new("_0")),
            type_validity: MirTypeValidity::default(),
        }
    }

    fn descriptor(phase: FailurePhase) -> ContractDescriptor {
        ContractDescriptor {
            precision: ContractPrecision::Opaque,
            model_id: None,
            relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
            predicates: BTreeSet::<ContractPredicate>::new(),
            failure_phase: phase,
            discharge: DischargeModel::None,
            static_implication: StaticContractImplication::NotApplicable,
        }
    }

    fn range_descriptor() -> ContractDescriptor {
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Generic;
        contract.model_id = Some("test_range_v1".to_string());
        contract.predicates.insert(ContractPredicate::RangeInBounds);
        contract
    }

    fn owner_info(owner: &FunctionKey, public_safe: bool) -> CandidateOwner {
        CandidateOwner {
            function: owner.clone(),
            root_span: span(1),
            public_safe,
            generic_surface: BTreeSet::new(),
        }
    }

    fn type_parameter(name: &str) -> GenericParamId {
        GenericParamId {
            def_path: name.to_string(),
            kind: super::super::endpoint_flow::GenericParamKind::Type,
        }
    }

    fn sequence_capability(parameter: GenericParamId) -> GenericCapabilityEvidence {
        GenericCapabilityEvidence {
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
                    type_facts: Default::default(),
                },
            )]),
            requirement: CapabilityRequirement::AssociatedSequenceNonEmpty,
            witness: CapabilityWitness::Proven {
                assignment: BTreeMap::from([(
                    CapabilityTermRole::Sequence,
                    super::super::endpoint_flow::FiniteTypeWitness::EmptyArray,
                )]),
                violations: BTreeSet::from([
                    super::super::endpoint_flow::CapabilityViolation::InBounds,
                ]),
            },
        }
    }

    fn pointer_capability(
        parameter: GenericParamId,
        kind: GenericCapabilityKind,
    ) -> GenericCapabilityEvidence {
        let requirement = match kind {
            GenericCapabilityKind::RepresentationPreservingPointerView => {
                CapabilityRequirement::RepresentationPreservingView
            }
            GenericCapabilityKind::PointerLayoutReinterpretation => {
                CapabilityRequirement::PointerLayout {
                    extent: super::super::endpoint_flow::AccessExtent::Positive(8),
                }
            }
            GenericCapabilityKind::AssociatedSequenceShape { .. } => {
                CapabilityRequirement::AssociatedSequenceNonEmpty
            }
        };
        let witness = if matches!(
            kind,
            GenericCapabilityKind::RepresentationPreservingPointerView
        ) {
            CapabilityWitness::ImpossibleByCompilerFacts {
                reason: CapabilityImpossibleReason::RepresentationPreservingView,
            }
        } else {
            CapabilityWitness::Proven {
                assignment: BTreeMap::from([
                    (
                        CapabilityTermRole::SourcePointee,
                        super::super::endpoint_flow::FiniteTypeWitness::U8,
                    ),
                    (
                        CapabilityTermRole::TargetPointee,
                        super::super::endpoint_flow::FiniteTypeWitness::U64,
                    ),
                ]),
                violations: BTreeSet::from([
                    super::super::endpoint_flow::CapabilityViolation::Size,
                    super::super::endpoint_flow::CapabilityViolation::Alignment,
                ]),
            }
        };
        GenericCapabilityEvidence {
            kind,
            terms: BTreeMap::from([
                (
                    CapabilityTermRole::SourcePointee,
                    CapabilityTerm {
                        form: CapabilityTermForm::Alias,
                        controllers: BTreeSet::from([parameter.clone()]),
                        residual_aliases: BTreeSet::from(["crate::Pair::Source".to_string()]),
                        equality_class: None,
                        type_facts: Default::default(),
                    },
                ),
                (
                    CapabilityTermRole::TargetPointee,
                    CapabilityTerm {
                        form: CapabilityTermForm::Alias,
                        controllers: BTreeSet::from([parameter]),
                        residual_aliases: BTreeSet::from(["crate::Pair::Target".to_string()]),
                        equality_class: None,
                        type_facts: Default::default(),
                    },
                ),
            ]),
            requirement,
            witness,
        }
    }

    fn direct_pointer_capability(
        source: GenericParamId,
        target: GenericParamId,
    ) -> GenericCapabilityEvidence {
        GenericCapabilityEvidence {
            kind: GenericCapabilityKind::PointerLayoutReinterpretation,
            terms: BTreeMap::from([
                (
                    CapabilityTermRole::SourcePointee,
                    CapabilityTerm {
                        form: CapabilityTermForm::DirectParam,
                        controllers: BTreeSet::from([source]),
                        residual_aliases: BTreeSet::new(),
                        equality_class: None,
                        type_facts: Default::default(),
                    },
                ),
                (
                    CapabilityTermRole::TargetPointee,
                    CapabilityTerm {
                        form: CapabilityTermForm::DirectParam,
                        controllers: BTreeSet::from([target.clone()]),
                        residual_aliases: BTreeSet::new(),
                        equality_class: None,
                        type_facts: Default::default(),
                    },
                ),
            ]),
            requirement: CapabilityRequirement::PointerLayout {
                extent: super::super::endpoint_flow::AccessExtent::Symbolic {
                    controllers: BTreeSet::from([target]),
                },
            },
            witness: CapabilityWitness::Proven {
                assignment: BTreeMap::from([
                    (
                        CapabilityTermRole::SourcePointee,
                        super::super::endpoint_flow::FiniteTypeWitness::U8,
                    ),
                    (
                        CapabilityTermRole::TargetPointee,
                        super::super::endpoint_flow::FiniteTypeWitness::U64,
                    ),
                ]),
                violations: BTreeSet::from([
                    CapabilityViolation::Size,
                    CapabilityViolation::Alignment,
                ]),
            },
        }
    }

    fn concrete_type_binding(
        size: Option<u64>,
        align: Option<u64>,
        equality_class: Option<u32>,
    ) -> GenericArgBinding {
        GenericArgBinding {
            kind: super::super::endpoint_flow::GenericParamKind::Type,
            form: CapabilityTermForm::Concrete,
            caller_controllers: BTreeSet::new(),
            residual_aliases: BTreeSet::new(),
            behavior_receiver: None,
            equality_class,
            type_facts: CapabilityTypeFacts {
                size,
                align,
                inhabited: Some(true),
            },
            const_value: None,
        }
    }

    fn generic_binding(
        callee_parameter: GenericParamId,
        form: CapabilityTermForm,
        caller_controllers: BTreeSet<GenericParamId>,
        status: GenericSubstitutionStatus,
        equality_class: Option<u32>,
    ) -> GenericCallSubstitution {
        GenericCallSubstitution {
            status,
            bindings: BTreeMap::from([(
                callee_parameter,
                GenericArgBinding {
                    kind: super::super::endpoint_flow::GenericParamKind::Type,
                    form,
                    caller_controllers,
                    residual_aliases: BTreeSet::new(),
                    behavior_receiver: None,
                    equality_class,
                    type_facts: Default::default(),
                    const_value: None,
                },
            )]),
        }
    }

    fn edge(from: EndpointNode, to: EndpointNode, kind: EndpointEdgeKind) -> EndpointEdge {
        EndpointEdge {
            from,
            to,
            kind,
            point: None,
        }
    }

    fn build_for_source(source: EndpointNode) -> StructuralFindingCandidate {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let generic_surface = match &source {
            EndpointNode::GenericCapability { evidence, .. } => evidence.controllers(),
            _ => BTreeSet::new(),
        };
        let mut contract = descriptor(FailurePhase::Precondition);
        if !generic_surface.is_empty() {
            contract.precision = ContractPrecision::Generic;
            contract.model_id = Some("test_bounds_v1".to_string());
            contract.predicates.insert(ContractPredicate::InBounds);
        }
        if matches!(&source, EndpointNode::InternalDerivation { .. }) {
            contract.precision = ContractPrecision::Generic;
            contract.model_id = Some("test_range_v1".to_string());
            contract.predicates.insert(ContractPredicate::RangeInBounds);
        }
        let mut owner_info = owner_info(&owner, true);
        owner_info.generic_surface = generic_surface;
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(source, target, EndpointEdgeKind::UnsafeOperand)]),
        };
        build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), contract)]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info)]),
            &BTreeSet::new(),
        )
        .pop()
        .unwrap()
    }

    fn build_with_exact_influence(
        graph: &EndpointFlowGraph,
        site: BoundUnsafeSite,
        contract: ContractDescriptor,
        owner: CandidateOwner,
        context: &ContextProofIndex,
        seeds: &InfluenceSeedIndex,
    ) -> StructuralCandidateSet {
        let site_id = site.seed.site_id();
        build_structural_candidate_set_inner(
            graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), contract)]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.function.clone(), owner)]),
            &BTreeSet::new(),
            Some(context),
            Some(seeds),
            None,
        )
        .expect("disabled structural trace cannot fail")
        .into_global_candidate_order()
    }

    #[test]
    fn influence_candidate_unestablished_p6_does_not_hide_exact_p1() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let ffi = EndpointNode::ExternalBehavior {
            owner: owner.clone(),
            kind: ExternalSourceKind::FfiReturn,
            controller: BehaviorController::FfiHost,
            token: "ffi-return".to_string(),
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id,
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([formal.clone(), ffi.clone(), target.clone()]),
            edges: BTreeSet::from([
                edge(
                    formal.clone(),
                    target.clone(),
                    EndpointEdgeKind::UnsafeOperand,
                ),
                edge(ffi.clone(), target.clone(), EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            formal.clone(),
            target,
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            formal,
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal {
                    owner: owner.clone(),
                    index: 1,
                },
                super::super::influence::InfluenceTypeClass::Pointer,
            ),
        );
        seeds.insert(
            ffi,
            super::super::influence::InfluenceToken::ffi(
                InfluenceOrigin::FfiReturn,
                super::super::influence::InfluenceCapabilities::ADDRESS
                    | super::super::influence::InfluenceCapabilities::VALIDITY_BITS,
            ),
        );
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Precise;
        contract.model_id = Some("ptr_read_v1".to_string());
        contract.predicates = BTreeSet::from([
            ContractPredicate::ValidForRead,
            ContractPredicate::Initialized,
        ]);
        let result = build_with_exact_influence(
            &graph,
            site,
            contract,
            owner_info(&owner, true),
            &context,
            &seeds,
        );
        assert_eq!(result.active.len(), 1);
        assert_eq!(result.active[0].primary_pattern, PrimaryPattern::P1);
        assert_eq!(
            result.active[0].classification_confidence,
            ClassificationConfidence::Exact,
        );
        assert!(result.active[0].exposure.is_none());
        assert_eq!(result.unestablished.len(), 1);
        assert_eq!(
            result.unestablished[0].attempted_pattern,
            PrimaryPattern::P6,
        );
        assert_eq!(
            result.unestablished[0].primary_reason,
            UnestablishedReason::MissingExactContext,
        );
    }

    #[test]
    fn influence_candidate_opaque_ffi_sink_is_unestablished_not_active() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let ffi = EndpointNode::ExternalBehavior {
            owner: owner.clone(),
            kind: ExternalSourceKind::FfiReturn,
            controller: BehaviorController::FfiHost,
            token: "ffi-return".to_string(),
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id,
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([ffi.clone(), target.clone()]),
            edges: BTreeSet::from([edge(
                ffi.clone(),
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            )]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(ffi.clone(), target, EndpointEdgeKind::UnsafeOperand));
        context.finalize();
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            ffi,
            super::super::influence::InfluenceToken::ffi(
                InfluenceOrigin::FfiReturn,
                super::super::influence::InfluenceCapabilities::ADDRESS,
            ),
        );
        let result = build_with_exact_influence(
            &graph,
            site,
            descriptor(FailurePhase::Precondition),
            owner_info(&owner, true),
            &context,
            &seeds,
        );
        assert!(result.active.is_empty());
        assert_eq!(result.unestablished.len(), 1);
        assert_eq!(
            result.unestablished[0].primary_reason,
            UnestablishedReason::UnmodeledUnsafeCall,
        );
    }

    #[test]
    fn influence_candidate_direct_ffi_to_compatible_sink_is_exact_p6() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let ffi = EndpointNode::ExternalBehavior {
            owner: owner.clone(),
            kind: ExternalSourceKind::FfiReturn,
            controller: BehaviorController::FfiHost,
            token: "ffi-return".to_string(),
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id,
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([ffi.clone(), target.clone()]),
            edges: BTreeSet::from([edge(
                ffi.clone(),
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            )]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(ffi.clone(), target, EndpointEdgeKind::UnsafeOperand));
        context.finalize();
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            ffi,
            super::super::influence::InfluenceToken::ffi(
                InfluenceOrigin::FfiReturn,
                super::super::influence::InfluenceCapabilities::ADDRESS
                    | super::super::influence::InfluenceCapabilities::VALIDITY_BITS,
            ),
        );
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Precise;
        contract.model_id = Some("ptr_read_v1".to_string());
        contract.predicates = BTreeSet::from([
            ContractPredicate::ValidForRead,
            ContractPredicate::Initialized,
        ]);
        let result = build_with_exact_influence(
            &graph,
            site,
            contract,
            owner_info(&owner, true),
            &context,
            &seeds,
        );
        assert_eq!(result.active.len(), 1);
        assert_eq!(result.active[0].primary_pattern, PrimaryPattern::P6);
        assert_eq!(
            result.active[0].classification_confidence,
            ClassificationConfidence::Exact,
        );
        assert!(result.active[0].exposure.is_none());
        assert!(result.suppressed.is_empty());
        assert!(result.unestablished.is_empty());
    }

    fn persistent_pointer_field(owner: &FunctionKey) -> EndpointNode {
        EndpointNode::PublicField {
            access_owner: owner.clone(),
            def_path: "crate::PublicRaw::pointer".into(),
            access_path: PlaceKey::new("_1.0"),
            base: PlaceKey::new("_1"),
            storage: StorageProvenance::EntryFormal {
                owner: owner.clone(),
                index: 1,
            },
            storage_versions: BTreeSet::new(),
            adt_kind: PublicFieldAdtKind::Struct,
            persistent: true,
            span: span(2),
        }
    }

    #[test]
    fn influence_candidate_persistent_field_requires_an_exact_compatible_route() {
        let owner = FunctionKey::new("crate::PublicRaw::read_public");
        let site = bound_site(&owner, None);
        let source = persistent_pointer_field(&owner);
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site.seed.site_id(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(
                source.clone(),
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            )]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            source.clone(),
            target,
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source,
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::PersistentPublicField,
                InfluenceController::PersistentPublicField {
                    access_owner: owner.clone(),
                    field: "crate::PublicRaw::pointer".into(),
                },
                super::super::influence::InfluenceTypeClass::Pointer,
            ),
        );
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Generic;
        contract.model_id = Some("raw_pointer_access_v1".to_string());
        contract.predicates = BTreeSet::from([
            ContractPredicate::ValidForRead,
            ContractPredicate::Initialized,
        ]);
        let result = build_with_exact_influence(
            &graph,
            site,
            contract,
            owner_info(&owner, true),
            &context,
            &seeds,
        );
        assert_eq!(result.active.len(), 1);
        assert_eq!(result.active[0].primary_pattern, PrimaryPattern::P2);
        assert_eq!(
            result.active[0].classification_confidence,
            ClassificationConfidence::Exact,
        );
        assert!(result.suppressed.is_empty());
        assert!(result.unestablished.is_empty());
    }

    #[test]
    fn influence_candidate_persistent_field_at_opaque_sink_is_heuristic_active() {
        let owner = FunctionKey::new("crate::PublicRaw::opaque");
        let site = bound_site(&owner, None);
        let source = persistent_pointer_field(&owner);
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site.seed.site_id(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(
                source.clone(),
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            )]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            source.clone(),
            target,
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source,
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::PersistentPublicField,
                InfluenceController::PersistentPublicField {
                    access_owner: owner.clone(),
                    field: "crate::PublicRaw::pointer".into(),
                },
                super::super::influence::InfluenceTypeClass::Pointer,
            ),
        );
        let result = build_with_exact_influence(
            &graph,
            site,
            descriptor(FailurePhase::Precondition),
            owner_info(&owner, true),
            &context,
            &seeds,
        );
        assert_eq!(result.active.len(), 1);
        assert_eq!(result.active[0].primary_pattern, PrimaryPattern::P2);
        assert_eq!(
            result.active[0].classification_confidence,
            ClassificationConfidence::Heuristic,
        );
        assert!(result.suppressed.is_empty());
        assert!(result.unestablished.is_empty());
    }

    #[test]
    fn influence_candidate_public_formal_at_opaque_sink_is_heuristic_active() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let source = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id,
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(
                source.clone(),
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            )]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            source.clone(),
            target,
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source,
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal {
                    owner: owner.clone(),
                    index: 1,
                },
                super::super::influence::InfluenceTypeClass::Pointer,
            ),
        );
        let result = build_with_exact_influence(
            &graph,
            site,
            descriptor(FailurePhase::Precondition),
            owner_info(&owner, true),
            &context,
            &seeds,
        );
        assert_eq!(result.active.len(), 1);
        assert_eq!(result.active[0].primary_pattern, PrimaryPattern::P1);
        assert_eq!(
            result.active[0].classification_confidence,
            ClassificationConfidence::Heuristic,
        );
        assert!(result.unestablished.is_empty());
    }

    #[test]
    fn influence_candidate_unknown_transmute_validity_is_unestablished() {
        let owner = FunctionKey::new("crate::root");
        let source = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: "site".to_string(),
            role: OperandRole::Argument(0),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            source.clone(),
            target.clone(),
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let reverse = context.exact_reverse_reachability(&target);
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source.clone(),
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal {
                    owner: owner.clone(),
                    index: 1,
                },
                super::super::influence::InfluenceTypeClass::Aggregate,
            ),
        );
        let mut contract = descriptor(FailurePhase::ImmediateValidity);
        contract.precision = ContractPrecision::Precise;
        contract.model_id = Some("transmute_validity_v1".to_string());
        contract.relevant_roles = BTreeSet::from([OperandRole::Argument(0)]);
        contract.predicates = BTreeSet::from([ContractPredicate::ValidForType]);
        contract.static_implication =
            StaticContractImplication::Unproven(StaticContractGap::UnknownTypeValidity);

        assert!(matches!(
            exact_influence_eligibility(
                PrimaryPattern::P1,
                &source,
                &owner_info(&owner, true),
                &OperandRole::Argument(0),
                &contract,
                Some(&context),
                &target,
                &seeds,
                Some(&reverse),
            ),
            SourceEligibility::Unestablished { reasons, .. }
                if reasons == BTreeSet::from([UnestablishedReason::UnknownCapabilityMatch])
        ));
    }

    #[test]
    fn influence_candidate_safe_public_reference_cannot_supply_an_invalid_address() {
        let owner = FunctionKey::new("crate::root");
        let source = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: "site".to_string(),
            role: OperandRole::Address,
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            source.clone(),
            target.clone(),
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let reverse = context.exact_reverse_reachability(&target);
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source.clone(),
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal {
                    owner: owner.clone(),
                    index: 1,
                },
                super::super::influence::InfluenceTypeClass::Reference,
            ),
        );
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Generic;
        contract.model_id = Some("raw_pointer_access_v1".to_string());
        contract.relevant_roles = BTreeSet::from([OperandRole::Address]);
        contract.predicates = BTreeSet::from([ContractPredicate::ValidForRead]);

        assert!(matches!(
            exact_influence_eligibility(
                PrimaryPattern::P1,
                &source,
                &owner_info(&owner, true),
                &OperandRole::Address,
                &contract,
                Some(&context),
                &target,
                &seeds,
                Some(&reverse),
            ),
            SourceEligibility::Suppressed(SuppressionReason::SourceCannotSatisfyRequirement, _)
        ));
    }

    #[test]
    fn influence_candidate_cross_body_restricted_validity_needs_validation_transport() {
        let root = FunctionKey::new("crate::root");
        let helper = FunctionKey::new("crate::checked_helper");
        let source = EndpointNode::Formal {
            owner: root.clone(),
            index: 1,
        };
        let target = EndpointNode::UnsafeOperand {
            owner: helper,
            site_id: "site".to_string(),
            role: OperandRole::Argument(0),
        };
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source.clone(),
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::PublicFormal,
                InfluenceController::PublicFormal {
                    owner: root.clone(),
                    index: 1,
                },
                super::super::influence::InfluenceTypeClass::Integer,
            ),
        );
        let mut contract = descriptor(FailurePhase::ImmediateValidity);
        contract.precision = ContractPrecision::Precise;
        contract.model_id = Some("transmute_validity_v1".to_string());
        contract.relevant_roles = BTreeSet::from([OperandRole::Argument(0)]);
        contract.predicates = BTreeSet::from([ContractPredicate::ValidForType]);
        contract.static_implication =
            StaticContractImplication::Unproven(StaticContractGap::TargetRestrictedValidity);

        assert!(matches!(
            exact_influence_eligibility(
                PrimaryPattern::P1,
                &source,
                &owner_info(&root, true),
                &OperandRole::Argument(0),
                &contract,
                None,
                &target,
                &seeds,
                None,
            ),
            SourceEligibility::Unestablished { reasons, .. }
                if reasons == BTreeSet::from([UnestablishedReason::ValidationStateUnknown])
        ));
    }

    #[test]
    fn influence_candidate_safe_open_trait_reference_is_unestablished_for_address() {
        let owner = FunctionKey::new("crate::root");
        let source = EndpointNode::ExternalBehavior {
            owner: owner.clone(),
            kind: ExternalSourceKind::OpenTrait,
            controller: BehaviorController::OpenTrait {
                trait_id: FunctionKey::new("crate::Trait"),
                openness: BehaviorOpenness::ProvenOpen,
                resolved_target: None,
                receiver: OpenBehaviorReceiver::Dynamic,
            },
            token: "safe-reference-return".to_string(),
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: "site".to_string(),
            role: OperandRole::Address,
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            source.clone(),
            target.clone(),
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let reverse = context.exact_reverse_reachability(&target);
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source.clone(),
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::OpenTraitReturn,
                InfluenceController::OpenTrait {
                    trait_id: FunctionKey::new("crate::Trait"),
                    receiver: OpenBehaviorReceiver::Dynamic,
                },
                super::super::influence::InfluenceTypeClass::Reference,
            ),
        );
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Generic;
        contract.model_id = Some("raw_pointer_access_v1".to_string());
        contract.relevant_roles = BTreeSet::from([OperandRole::Address]);
        contract.predicates = BTreeSet::from([ContractPredicate::ValidForRead]);

        assert!(matches!(
            exact_influence_eligibility(
                PrimaryPattern::P6,
                &source,
                &owner_info(&owner, true),
                &OperandRole::Address,
                &contract,
                Some(&context),
                &target,
                &seeds,
                Some(&reverse),
            ),
            SourceEligibility::Unestablished { reasons, .. }
                if reasons == BTreeSet::from([UnestablishedReason::UnknownCapabilityMatch])
        ));
    }

    #[test]
    fn influence_candidate_concrete_dispatch_is_evidence_suppressed() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let source = EndpointNode::ExternalBehavior {
            owner: owner.clone(),
            kind: ExternalSourceKind::OpenTrait,
            controller: BehaviorController::OpenTrait {
                trait_id: FunctionKey::new("crate::Trait"),
                openness: BehaviorOpenness::ProvenClosed,
                resolved_target: Some(FunctionKey::new("crate::Concrete::method")),
                receiver: OpenBehaviorReceiver::Dynamic,
            },
            token: "trait-return".to_string(),
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id,
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(
                source.clone(),
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            )]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(edge(
            source.clone(),
            target,
            EndpointEdgeKind::UnsafeOperand,
        ));
        context.finalize();
        let mut seeds = InfluenceSeedIndex::default();
        seeds.insert(
            source,
            super::super::influence::InfluenceToken::from_type_class(
                InfluenceOrigin::OpenTraitReturn,
                InfluenceController::Internal,
                super::super::influence::InfluenceTypeClass::Pointer,
            ),
        );
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Precise;
        contract.model_id = Some("ptr_read_v1".to_string());
        contract.predicates = BTreeSet::from([ContractPredicate::ValidForRead]);
        let result = build_with_exact_influence(
            &graph,
            site,
            contract,
            owner_info(&owner, true),
            &context,
            &seeds,
        );
        assert!(result.active.is_empty());
        assert!(result.unestablished.is_empty());
        assert_eq!(result.suppressed.len(), 1);
        assert_eq!(
            result.suppressed[0].reason,
            SuppressionReason::TraitNotDownstreamImplementable,
        );
        assert!(result.suppressed[0]
            .evidence
            .iter()
            .any(|(key, value)| key == "openness" && value == "proven_closed"));
    }

    #[test]
    fn influence_candidate_proven_closed_controller_suppresses_without_an_exact_route() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let source = EndpointNode::ExternalBehavior {
            owner: owner.clone(),
            kind: ExternalSourceKind::OpenTrait,
            controller: BehaviorController::OpenTrait {
                trait_id: FunctionKey::new("crate::Trait"),
                openness: BehaviorOpenness::ProvenClosed,
                resolved_target: Some(FunctionKey::new("crate::Concrete::method")),
                receiver: OpenBehaviorReceiver::Dynamic,
            },
            token: "trait-return".to_string(),
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id,
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(source, target, EndpointEdgeKind::UnsafeOperand)]),
        };
        let mut context = ContextProofIndex::default();
        context.finalize();
        let result = build_with_exact_influence(
            &graph,
            site,
            descriptor(FailurePhase::Precondition),
            owner_info(&owner, true),
            &context,
            &InfluenceSeedIndex::default(),
        );
        assert!(result.active.is_empty());
        assert!(result.unestablished.is_empty());
        assert_eq!(result.suppressed.len(), 1);
        assert_eq!(
            result.suppressed[0].reason,
            SuppressionReason::TraitNotDownstreamImplementable,
        );
        assert_eq!(
            result.suppressed[0]
                .evidence
                .iter()
                .find(|(key, _)| key == "openness")
                .map(|(_, value)| value.as_str()),
            Some("proven_closed"),
        );
    }

    #[test]
    fn online_rank_reduction_keeps_all_first_ordered_minimum_ties() {
        let owner = FunctionKey::new("crate::root");
        let base = build_for_source(EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        });
        let mut ranked = BTreeMap::new();
        for (rank, mechanism) in [
            (3, MechanismId::P6ExternalBehaviorFlow),
            (1, MechanismId::P1PublicParameterFlow),
            (1, MechanismId::P2PublicStateFlow),
            (2, MechanismId::P3InternalUnsafeExposure),
        ] {
            let mut candidate = base.clone();
            candidate.mechanism_id = mechanism;
            insert_best_rank_candidate(&mut ranked, Arc::new(owner.clone()), rank, candidate);
        }
        let (rank, selected) = ranked.remove(&owner).unwrap();
        assert_eq!(rank, 1);
        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.mechanism_id)
                .collect::<Vec<_>>(),
            [
                MechanismId::P1PublicParameterFlow,
                MechanismId::P2PublicStateFlow,
            ],
        );
    }

    #[test]
    fn classifies_each_non_exposure_source_as_the_expected_pattern() {
        let owner = FunctionKey::new("crate::root");
        let cases = [
            (
                EndpointNode::Formal {
                    owner: owner.clone(),
                    index: 1,
                },
                PrimaryPattern::P1,
            ),
            (
                EndpointNode::PublicField {
                    access_owner: owner.clone(),
                    def_path: "crate::State::field".into(),
                    access_path: PlaceKey::new("_1.0"),
                    base: PlaceKey::new("_1"),
                    storage: StorageProvenance::EntryFormal {
                        owner: owner.clone(),
                        index: 1,
                    },
                    storage_versions: BTreeSet::new(),
                    adt_kind: PublicFieldAdtKind::Struct,
                    persistent: true,
                    span: span(2),
                },
                PrimaryPattern::P2,
            ),
            (
                EndpointNode::InternalDerivation {
                    owner: owner.clone(),
                    kind: DerivationKind::Offset,
                    token: "offset".into(),
                    point: point(&owner, 2),
                },
                PrimaryPattern::P4,
            ),
            (
                EndpointNode::GenericCapability {
                    owner: owner.clone(),
                    evidence: sequence_capability(type_parameter("crate::root::P")),
                    span: span(2),
                },
                PrimaryPattern::P5,
            ),
            (
                EndpointNode::ExternalBehavior {
                    owner: owner.clone(),
                    kind: ExternalSourceKind::Callback,
                    controller: BehaviorController::Unknown,
                    token: "callback-return".into(),
                    span: span(2),
                },
                PrimaryPattern::P6,
            ),
        ];
        for (source, expected) in cases {
            assert_eq!(build_for_source(source).primary_pattern, expected);
        }
    }

    #[test]
    fn p5_gate_requires_a_modeled_matching_contract_and_root_surface() {
        let owner = FunctionKey::new("crate::root");
        let parameter = type_parameter("crate::root::T");
        let capability = pointer_capability(
            parameter.clone(),
            GenericCapabilityKind::PointerLayoutReinterpretation,
        );
        let source = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence: capability,
            span: span(2),
        };
        let mut root = owner_info(&owner, true);
        root.generic_surface.insert(parameter);
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Precise;
        contract.model_id = Some("ptr_read_v1".to_string());
        contract.predicates = BTreeSet::from([
            ContractPredicate::ValidForRead,
            ContractPredicate::Aligned,
            ContractPredicate::Initialized,
        ]);
        let calls = BTreeSet::new();
        let call_route_index = CallRouteIndex::new(&calls);
        assert!(matches!(
            source_eligibility(
                &source,
                &root,
                &OperandRole::Argument(0),
                &contract,
                &call_route_index,
            ),
            SourceEligibility::Active {
                supporting_call_route: Some(route),
                ..
            } if route.is_empty()
        ));

        contract.precision = ContractPrecision::Opaque;
        contract.model_id = None;
        contract.predicates.clear();
        assert!(matches!(
            source_eligibility(
                &source,
                &root,
                &OperandRole::Argument(0),
                &contract,
                &call_route_index,
            ),
            SourceEligibility::Unsupported
        ));
    }

    #[test]
    fn valid_for_read_alone_does_not_invent_a_pointer_layout_capability_gap() {
        let owner = FunctionKey::new("crate::root");
        let parameter = type_parameter("crate::root::T");
        let source = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence: pointer_capability(
                parameter.clone(),
                GenericCapabilityKind::PointerLayoutReinterpretation,
            ),
            span: span(2),
        };
        let mut root = owner_info(&owner, true);
        root.generic_surface.insert(parameter);
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Generic;
        contract.model_id = Some("read_only_v1".to_string());
        contract.predicates = BTreeSet::from([ContractPredicate::ValidForRead]);
        let calls = BTreeSet::new();
        let call_route_index = CallRouteIndex::new(&calls);
        assert!(matches!(
            source_eligibility(
                &source,
                &root,
                &OperandRole::Argument(0),
                &contract,
                &call_route_index,
            ),
            SourceEligibility::Unsupported
        ));
    }

    #[test]
    fn unestablished_finite_witness_falls_back_without_a_safety_suppression() {
        let owner = FunctionKey::new("crate::root");
        let parameter = type_parameter("crate::root::P");
        let mut evidence = sequence_capability(parameter.clone());
        evidence.witness = CapabilityWitness::Unestablished {
            reason: CapabilityUnestablishedReason::ComplexBounds,
        };
        let source = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence,
            span: span(2),
        };
        let mut root = owner_info(&owner, true);
        root.generic_surface.insert(parameter);
        let calls = BTreeSet::new();
        let call_route_index = CallRouteIndex::new(&calls);

        assert!(matches!(
            source_eligibility(
                &source,
                &root,
                &OperandRole::Argument(0),
                &range_descriptor(),
                &call_route_index,
            ),
            SourceEligibility::Unsupported
        ));
    }

    #[test]
    fn representation_preserving_generic_view_is_ledgered_and_formal_falls_back_to_p1() {
        let owner = FunctionKey::new("crate::root");
        let parameter = type_parameter("crate::root::T");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let view = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence: pointer_capability(
                parameter.clone(),
                GenericCapabilityKind::RepresentationPreservingPointerView,
            ),
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([formal.clone(), view.clone(), target.clone()]),
            edges: BTreeSet::from([
                edge(formal, target.clone(), EndpointEdgeKind::UnsafeOperand),
                edge(view, target, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let mut root = owner_info(&owner, true);
        root.generic_surface.insert(parameter);
        let mut contract = descriptor(FailurePhase::Precondition);
        contract.precision = ContractPrecision::Precise;
        contract.model_id = Some("ptr_read_v1".to_string());
        contract.predicates = BTreeSet::from([
            ContractPredicate::ValidForRead,
            ContractPredicate::Aligned,
            ContractPredicate::Initialized,
        ]);
        let candidates = build_structural_candidate_set(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), contract)]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), root)]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.active.len(), 1);
        assert_eq!(candidates.active[0].primary_pattern, PrimaryPattern::P1);
        assert_eq!(candidates.suppressed.len(), 1);
        assert_eq!(
            candidates.suppressed[0].reason,
            SuppressionReason::PointerViewHasNoCapabilityGap
        );
        assert_eq!(
            candidates.suppressed[0]
                .evidence
                .iter()
                .find(|(key, _)| key == "gate")
                .map(|(_, value)| value.as_str()),
            Some("representation_preserving_view")
        );
    }

    #[test]
    fn nonpersistent_public_field_is_ledgered_and_formal_falls_back_to_p1() {
        let owner = FunctionKey::new("crate::PublicChoice::read_payload");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let field = EndpointNode::PublicField {
            access_owner: owner.clone(),
            def_path: "crate::PublicChoice::Pointer::0".into(),
            access_path: PlaceKey::new("_1.0"),
            base: PlaceKey::new("_1"),
            storage: StorageProvenance::EntryFormal {
                owner: owner.clone(),
                index: 1,
            },
            storage_versions: BTreeSet::new(),
            adt_kind: PublicFieldAdtKind::Enum,
            persistent: false,
            span: span(2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([formal.clone(), field.clone(), target.clone()]),
            edges: BTreeSet::from([
                edge(
                    formal.clone(),
                    target.clone(),
                    EndpointEdgeKind::UnsafeOperand,
                ),
                edge(field, target, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let candidates = build_structural_candidate_set(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.active.len(), 1);
        assert_eq!(candidates.active[0].primary_pattern, PrimaryPattern::P1);
        assert_eq!(candidates.active[0].source.as_ref(), &formal);
        assert_eq!(candidates.suppressed.len(), 1);
        assert_eq!(
            candidates.suppressed[0].reason,
            SuppressionReason::NonpersistentPublicField
        );
        assert_eq!(
            candidates.suppressed[0]
                .evidence
                .iter()
                .find(|(key, _)| key == "adt_kind")
                .map(|(_, value)| value.as_str()),
            Some("enum")
        );
    }

    #[test]
    fn non_formal_sources_only_pair_with_public_roots_that_reach_the_source_owner() {
        let source_owner = FunctionKey::new("crate::source_helper");
        let site_owner = FunctionKey::new("crate::sink_helper");
        let causal_root = FunctionKey::new("crate::causal_root");
        let unrelated_root = FunctionKey::new("crate::unrelated_root");
        let source_only_root = FunctionKey::new("crate::source_only_root");
        let site = bound_site(&site_owner, None);
        let site_id = site.seed.site_id();
        let sources = [
            (ExternalSourceKind::OpenTrait, "open-trait-return"),
            (ExternalSourceKind::Callback, "callback-return"),
            (ExternalSourceKind::FunctionPointer, "fn-pointer-return"),
        ]
        .into_iter()
        .map(|(kind, token)| EndpointNode::ExternalBehavior {
            owner: source_owner.clone(),
            kind,
            controller: BehaviorController::Unknown,
            token: token.into(),
            span: span(2),
        })
        .collect::<BTreeSet<_>>();
        let target = EndpointNode::UnsafeOperand {
            owner: site_owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let mut graph = EndpointFlowGraph::default();
        graph.insert_node(target.clone());
        for source in sources {
            graph.insert_edge(edge(
                source,
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            ));
        }
        let calls = BTreeSet::from([
            CandidateCallEdge {
                caller: causal_root.clone(),
                callee: source_owner.clone(),
                point: point(&causal_root, 2),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: causal_root.clone(),
                callee: site_owner.clone(),
                point: point(&causal_root, 3),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: unrelated_root.clone(),
                callee: site_owner.clone(),
                point: point(&unrelated_root, 2),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: source_only_root.clone(),
                callee: source_owner.clone(),
                point: point(&source_only_root, 2),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
        ]);
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([
                (source_owner.clone(), owner_info(&source_owner, false)),
                (site_owner.clone(), owner_info(&site_owner, false)),
                (causal_root.clone(), owner_info(&causal_root, true)),
                (unrelated_root.clone(), owner_info(&unrelated_root, true)),
                (
                    source_only_root.clone(),
                    owner_info(&source_only_root, true),
                ),
            ]),
            &calls,
        );
        assert_eq!(candidates.len(), 3);
        assert!(candidates.iter().all(|candidate| {
            candidate.primary_pattern == PrimaryPattern::P6
                && candidate.public_root.as_ref() == &causal_root
                && candidate.call_route.len() == 1
                && candidate.call_route[0].callee == site_owner
        }));
    }

    #[test]
    fn function_pointer_source_stays_conservative_without_controller_provenance() {
        let owner = FunctionKey::new("crate::root");
        let candidate = build_for_source(EndpointNode::ExternalBehavior {
            owner,
            kind: ExternalSourceKind::FunctionPointer,
            controller: BehaviorController::Unknown,
            token: "fn-pointer-return".into(),
            span: span(2),
        });

        // EndpointNode currently records where the indirect call happened, but
        // not whether its callee came from a public formal/field or from an
        // internal fn item/closure.  Dropping it here would therefore also drop
        // the production-shaped public callback positive.
        assert_eq!(candidate.primary_pattern, PrimaryPattern::P6);
    }

    #[test]
    fn proven_closed_and_internal_controllers_do_not_win_over_a_formal_source() {
        let owner = FunctionKey::new("crate::root");
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let rejected = [
            BehaviorController::OpenTrait {
                trait_id: FunctionKey::new("crate::sealed::Behavior"),
                openness: BehaviorOpenness::ProvenClosed,
                resolved_target: None,
                receiver: OpenBehaviorReceiver::Unknown,
            },
            BehaviorController::InternalFunction {
                function: FunctionKey::new("crate::identity"),
            },
            BehaviorController::InternalClosure {
                function: FunctionKey::new("crate::root::{closure#0}"),
            },
        ];

        for controller in rejected {
            let expected_reason = controller.rejection_reason().unwrap();
            let site = bound_site(&owner, None);
            let site_id = site.seed.site_id();
            let external = EndpointNode::ExternalBehavior {
                owner: owner.clone(),
                kind: ExternalSourceKind::FunctionPointer,
                controller,
                token: "controlled-return".into(),
                span: span(2),
            };
            let target = EndpointNode::UnsafeOperand {
                owner: owner.clone(),
                site_id: site_id.clone(),
                role: OperandRole::Argument(0),
            };
            let graph = EndpointFlowGraph {
                nodes: BTreeSet::from([formal.clone(), external.clone(), target.clone()]),
                edges: BTreeSet::from([
                    edge(
                        formal.clone(),
                        target.clone(),
                        EndpointEdgeKind::UnsafeOperand,
                    ),
                    edge(external, target, EndpointEdgeKind::UnsafeOperand),
                ]),
            };
            let candidates = build_structural_candidate_set(
                &graph,
                &BTreeMap::from([(
                    site_id.clone(),
                    StructuralSiteCandidate {
                        site_id: site_id.clone(),
                        relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                        validity_sensitive_result: false,
                    },
                )]),
                &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
                &BTreeMap::from([(site_id, site)]),
                &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
                &BTreeSet::new(),
            );
            assert_eq!(candidates.active.len(), 1);
            assert_eq!(candidates.active[0].primary_pattern, PrimaryPattern::P1);
            assert_eq!(candidates.active[0].source.as_ref(), &formal);
            assert_eq!(candidates.suppressed.len(), 1);
            assert!(!candidates.suppressed[0].evidence.is_empty());
            assert_eq!(
                candidates.suppressed[0].candidate.primary_pattern,
                PrimaryPattern::P6
            );
            assert_eq!(candidates.suppressed[0].reason, expected_reason);
        }
    }

    #[test]
    fn unknown_and_proven_open_controllers_remain_p6() {
        let owner = FunctionKey::new("crate::root");
        let controllers = [
            BehaviorController::Unknown,
            BehaviorController::OpenTrait {
                trait_id: FunctionKey::new("crate::OpenBehavior"),
                openness: BehaviorOpenness::ProvenOpen,
                resolved_target: None,
                receiver: OpenBehaviorReceiver::Dynamic,
            },
            BehaviorController::PublicFormal {
                owner: owner.clone(),
                index: 1,
            },
        ];

        for controller in controllers {
            let candidate = build_for_source(EndpointNode::ExternalBehavior {
                owner: owner.clone(),
                kind: ExternalSourceKind::Callback,
                controller,
                token: "external-return".into(),
                span: span(2),
            });
            assert_eq!(candidate.primary_pattern, PrimaryPattern::P6);
        }
    }

    #[test]
    fn pattern_precedence_is_selected_per_public_root_and_subject() {
        let root_a = FunctionKey::new("crate::root_a");
        let root_b = FunctionKey::new("crate::root_b");
        let source_helper = FunctionKey::new("crate::source_helper");
        let site_owner = FunctionKey::new("crate::sink_helper");
        let site = bound_site(&site_owner, None);
        let site_id = site.seed.site_id();
        let external = EndpointNode::ExternalBehavior {
            owner: source_helper.clone(),
            kind: ExternalSourceKind::Callback,
            controller: BehaviorController::Unknown,
            token: "callback-return".into(),
            span: span(3),
        };
        let formal = EndpointNode::Formal {
            owner: root_b.clone(),
            index: 1,
        };
        let target = EndpointNode::UnsafeOperand {
            owner: site_owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([external.clone(), formal.clone(), target.clone()]),
            edges: BTreeSet::from([
                edge(external, target.clone(), EndpointEdgeKind::UnsafeOperand),
                edge(formal, target, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let calls = BTreeSet::from([
            CandidateCallEdge {
                caller: root_a.clone(),
                callee: source_helper.clone(),
                point: point(&root_a, 2),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: source_helper.clone(),
                callee: site_owner.clone(),
                point: point(&source_helper, 3),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: root_b.clone(),
                callee: site_owner.clone(),
                point: point(&root_b, 4),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
        ]);
        let candidates = build_structural_candidate_set(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([
                (root_a.clone(), owner_info(&root_a, true)),
                (root_b.clone(), owner_info(&root_b, true)),
                (source_helper.clone(), owner_info(&source_helper, false)),
                (site_owner.clone(), owner_info(&site_owner, false)),
            ]),
            &calls,
        );
        assert_eq!(candidates.active.len(), 2);
        assert!(candidates.active.iter().any(|candidate| {
            candidate.public_root.as_ref() == &root_a
                && candidate.primary_pattern == PrimaryPattern::P6
        }));
        assert!(candidates.active.iter().any(|candidate| {
            candidate.public_root.as_ref() == &root_b
                && candidate.primary_pattern == PrimaryPattern::P1
        }));
        assert!(Arc::ptr_eq(
            &candidates.active[0].site_context,
            &candidates.active[1].site_context,
        ));
    }

    #[test]
    fn source_owner_root_identity_cache_is_built_once_for_multiple_sources() {
        let root = FunctionKey::new("crate::root");
        let source_owner = FunctionKey::new("crate::source_helper");
        let owners = BTreeMap::from([
            (root.clone(), owner_info(&root, true)),
            (source_owner.clone(), owner_info(&source_owner, false)),
        ]);
        let calls = BTreeSet::from([CandidateCallEdge {
            caller: root.clone(),
            callee: source_owner.clone(),
            point: point(&root, 2),
            kind: CandidateCallKind::ResolvedLocal,
            generic_substitution: GenericCallSubstitution::default(),
        }]);
        let call_route_index = CallRouteIndex::new(&calls);
        let mut cache = PublicRootIdentityCache::new(&owners, &call_route_index);
        let first = EndpointNode::ExternalBehavior {
            owner: source_owner.clone(),
            kind: ExternalSourceKind::Callback,
            controller: BehaviorController::Unknown,
            token: "callback-return".into(),
            span: span(2),
        };
        let second = EndpointNode::GenericCapability {
            owner: source_owner,
            evidence: sequence_capability(type_parameter("crate::source_helper::P")),
            span: span(3),
        };

        {
            let roots = candidate_root_identities(&first, &owners, &mut cache);
            assert!(roots.contains(&root));
        }
        {
            let roots = candidate_root_identities(&second, &owners, &mut cache);
            assert!(roots.contains(&root));
        }
        let canonical_root = owners
            .get_key_value(&root)
            .map(|(root, _)| root)
            .expect("test root must be present");
        assert!(std::ptr::eq(
            cache.roots_reaching(&FunctionKey::new("crate::source_helper"))[0],
            canonical_root,
        ));
        assert_eq!(cache.test_build_count(), 1);
    }

    #[test]
    fn ordered_endpoint_merge_matches_btree_union_without_retaining_tree_nodes() {
        let owner = FunctionKey::new("crate::merge");
        let nodes = (0..5)
            .map(|index| EndpointNode::Formal {
                owner: owner.clone(),
                index,
            })
            .collect::<Vec<_>>();
        let merged = merge_ordered_endpoint_nodes(
            [&nodes[0], &nodes[2], &nodes[4]],
            [&nodes[1], &nodes[2], &nodes[3]],
        );
        assert_eq!(
            merged,
            nodes.iter().collect::<Vec<_>>(),
            "the compact merge must preserve semantic BTree order and deduplicate equality",
        );
    }

    #[test]
    fn optional_candidate_exposure_is_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<Option<Arc<CandidateExposure>>>(),
            std::mem::size_of::<usize>(),
            "non-P3 candidates must not reserve an inline EndpointNode payload",
        );
    }

    #[test]
    fn candidate_public_root_is_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<Arc<FunctionKey>>(),
            std::mem::size_of::<usize>(),
            "candidate rows must share their repeated public-root identity",
        );
    }

    #[test]
    fn candidate_site_context_is_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<Arc<CandidateSiteContext>>(),
            std::mem::size_of::<usize>(),
            "candidate rows must share their repeated site, contract, and failure context",
        );
    }

    #[test]
    fn candidate_mechanism_id_is_one_byte() {
        assert_eq!(
            std::mem::size_of::<MechanismId>(),
            1,
            "closed-set mechanism identities must not retain a string fat pointer per row",
        );
    }

    #[test]
    fn candidate_sources_are_value_interned_without_order_dependence() {
        let first_value = EndpointNode::Formal {
            owner: FunctionKey::new("crate::root"),
            index: 1,
        };
        let equal_value = first_value.clone();
        let distinct_value = EndpointNode::Formal {
            owner: FunctionKey::new("crate::root"),
            index: 2,
        };
        let mut interner = HashSet::new();
        let first = intern_candidate_source(&mut interner, &first_value);
        let equal = intern_candidate_source(&mut interner, &equal_value);
        let distinct = intern_candidate_source(&mut interner, &distinct_value);
        assert!(Arc::ptr_eq(&first, &equal));
        assert!(!Arc::ptr_eq(&first, &distinct));
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn suppression_evidence_uses_sorted_compact_entries() {
        let suppressed = SuppressedStructuralCandidate::new(
            build_for_source(EndpointNode::Formal {
                owner: FunctionKey::new("crate::root"),
                index: 1,
            }),
            SuppressionReason::NonpersistentPublicField,
            BTreeMap::from([
                ("zeta".to_string(), "last".to_string()),
                ("alpha".to_string(), "first".to_string()),
            ]),
        );
        assert_eq!(
            suppressed.evidence,
            [
                ("alpha".to_string(), "first".to_string()),
                ("zeta".to_string(), "last".to_string()),
            ]
        );
    }

    #[test]
    fn streaming_candidate_dedup_keeps_the_first_value() {
        let owner = FunctionKey::new("crate::root");
        let source = EndpointNode::Formal { owner, index: 1 };
        let first = build_for_source(source);
        let mut duplicate = first.clone();
        duplicate.mechanism_id = MechanismId::P2PublicStateFlow;
        let mut candidates = BTreeMap::new();

        insert_candidate_first_wins(&mut candidates, first.clone());
        insert_candidate_first_wins(&mut candidates, duplicate);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates.into_values().next(), Some(first));
    }

    #[test]
    fn deferred_global_order_matches_candidate_key_btree_first_wins() {
        let owner = FunctionKey::new("crate::root");
        let source = EndpointNode::Formal { owner, index: 1 };
        let first = build_for_source(source);
        let mut duplicate = first.clone();
        duplicate.mechanism_id = MechanismId::P2PublicStateFlow;
        let mut other = first.clone();
        other.primary_pattern = PrimaryPattern::P6;
        other.mechanism_id = MechanismId::P6ExternalBehaviorFlow;

        let mut reference = BTreeMap::new();
        for candidate in [other.clone(), first.clone(), duplicate.clone()] {
            insert_candidate_first_wins(&mut reference, candidate);
        }
        let expected = reference.into_values().collect::<Vec<_>>();

        assert_eq!(
            globally_ordered_candidates(vec![other, first, duplicate]),
            expected,
        );
    }

    #[test]
    fn deferred_global_order_is_stable_for_equal_keys() {
        assert_eq!(
            globally_ordered_first_wins(vec![
                ("b".to_string(), 0, 1_u8),
                ("a".to_string(), 1, 2_u8),
                ("a".to_string(), 2, 3_u8),
            ]),
            vec![2, 1],
        );
    }

    #[test]
    fn bounds_relation_prefers_the_index_when_both_public_formals_reach_the_site() {
        let owner = FunctionKey::new("crate::root");
        let mut site = bound_site(&owner, None);
        site.operands.push(MirOperandAnchor {
            role: OperandRole::Argument(1),
            type_fingerprint: "usize".into(),
            place: Some(PlaceKey::new("_2")),
        });
        let site_id = site.seed.site_id();
        let collection = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let index = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(1),
        };
        let collection_source = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let index_source = EndpointNode::Formal {
            owner: owner.clone(),
            index: 2,
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([
                collection_source.clone(),
                index_source.clone(),
                collection.clone(),
                index.clone(),
            ]),
            edges: BTreeSet::from([
                edge(
                    collection_source,
                    collection,
                    EndpointEdgeKind::UnsafeOperand,
                ),
                edge(index_source.clone(), index, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let contract = ContractDescriptor {
            precision: ContractPrecision::Precise,
            model_id: Some("slice_get_unchecked_v1".into()),
            relevant_roles: BTreeSet::from([OperandRole::Argument(0), OperandRole::Argument(1)]),
            predicates: BTreeSet::from([ContractPredicate::InBounds]),
            failure_phase: FailurePhase::Precondition,
            discharge: DischargeModel::ExactValidation,
            static_implication: StaticContractImplication::NotApplicable,
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: contract.relevant_roles.clone(),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), contract)]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].source.as_ref(), &index_source);
        assert_eq!(candidates[0].operand_role, Some(OperandRole::Argument(1)));
    }

    #[test]
    fn precedence_selects_external_then_capability_field_derivation_and_formal() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let sources = BTreeSet::from([
            EndpointNode::Formal {
                owner: owner.clone(),
                index: 1,
            },
            EndpointNode::InternalDerivation {
                owner: owner.clone(),
                kind: DerivationKind::Offset,
                token: "derived".into(),
                point: point(&owner, 2),
            },
            EndpointNode::PublicField {
                access_owner: owner.clone(),
                def_path: "crate::State::field".into(),
                access_path: PlaceKey::new("_1.0"),
                base: PlaceKey::new("_1"),
                storage: StorageProvenance::EntryFormal {
                    owner: owner.clone(),
                    index: 1,
                },
                storage_versions: BTreeSet::new(),
                adt_kind: PublicFieldAdtKind::Struct,
                persistent: true,
                span: span(3),
            },
            EndpointNode::GenericCapability {
                owner: owner.clone(),
                evidence: sequence_capability(type_parameter("crate::root::P")),
                span: span(4),
            },
            EndpointNode::ExternalBehavior {
                owner: owner.clone(),
                kind: ExternalSourceKind::Iterator,
                controller: BehaviorController::Unknown,
                token: "next".into(),
                span: span(5),
            },
        ]);
        let mut graph = EndpointFlowGraph::default();
        for source in sources {
            graph.insert_edge(edge(
                source,
                target.clone(),
                EndpointEdgeKind::UnsafeOperand,
            ));
        }
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .all(|candidate| candidate.primary_pattern == PrimaryPattern::P6));
    }

    #[test]
    fn internal_derivation_supersedes_a_public_formal_on_the_same_flow() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let derivation = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Offset,
            token: "offset".into(),
            point: point(&owner, 2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([formal.clone(), derivation.clone(), target.clone()]),
            edges: BTreeSet::from([
                edge(formal, derivation.clone(), EndpointEdgeKind::Derivation),
                edge(derivation, target, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), range_descriptor())]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P4);
        assert_eq!(candidates[0].endpoint_route.len(), 1);
    }

    #[test]
    fn derivation_irrelevant_to_the_sink_contract_falls_back_to_p1() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let derivation = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Layout,
            token: "transparent-layout".into(),
            point: point(&owner, 2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([formal.clone(), derivation.clone(), target.clone()]),
            edges: BTreeSet::from([
                edge(
                    formal.clone(),
                    derivation.clone(),
                    EndpointEdgeKind::Derivation,
                ),
                edge(derivation, target, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), range_descriptor())]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P1);
        assert_eq!(candidates[0].source.as_ref(), &formal);
    }

    #[test]
    fn condition_true_accepts_only_a_boolean_relation_for_argument_zero() {
        let contract = ContractDescriptor {
            precision: ContractPrecision::Precise,
            model_id: Some("condition_true_v1".into()),
            relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
            predicates: BTreeSet::from([ContractPredicate::ConditionTrue]),
            failure_phase: FailurePhase::Precondition,
            discharge: DischargeModel::ExactRelation,
            static_implication: StaticContractImplication::NotApplicable,
        };
        assert!(derivation_is_relevant(
            DerivationKind::UncheckedBooleanRelation,
            &OperandRole::Argument(0),
            &contract,
        ));
        assert!(!derivation_is_relevant(
            DerivationKind::UncheckedBooleanRelation,
            &OperandRole::Argument(1),
            &contract,
        ));
        assert!(!derivation_is_relevant(
            DerivationKind::Offset,
            &OperandRole::Argument(0),
            &contract,
        ));
    }

    #[test]
    fn a_public_formal_path_that_bypasses_the_derivation_stays_p1() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let derivation = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Offset,
            token: "conditional-rewrite".into(),
            point: point(&owner, 2),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([formal.clone(), derivation.clone(), target.clone()]),
            edges: BTreeSet::from([
                edge(
                    formal.clone(),
                    derivation.clone(),
                    EndpointEdgeKind::Derivation,
                ),
                edge(
                    formal.clone(),
                    target.clone(),
                    EndpointEdgeKind::UnsafeOperand,
                ),
                edge(derivation, target, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), range_descriptor())]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P1);
        assert_eq!(candidates[0].source.as_ref(), &formal);
    }

    #[test]
    fn a_chain_of_internal_derivations_reports_the_first_rebreak_once() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let first = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Offset,
            token: "offset".into(),
            point: point(&owner, 2),
        };
        let second = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Layout,
            token: "pointer-cast".into(),
            point: point(&owner, 3),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([
                formal.clone(),
                first.clone(),
                second.clone(),
                target.clone(),
            ]),
            edges: BTreeSet::from([
                edge(formal, first.clone(), EndpointEdgeKind::Derivation),
                edge(first.clone(), second.clone(), EndpointEdgeKind::Derivation),
                edge(second.clone(), target, EndpointEdgeKind::UnsafeOperand),
            ]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), range_descriptor())]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].source.as_ref(), &first);
    }

    #[test]
    fn internal_unsafe_result_exposed_by_a_public_return_is_p3() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, Some("&'static str"));
        let site_id = site.seed.site_id();
        let source = EndpointNode::UnsafeResult {
            owner: owner.clone(),
            site_id: site_id.clone(),
        };
        let exposure = EndpointNode::Return {
            owner: owner.clone(),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), exposure.clone()]),
            edges: BTreeSet::from([edge(source, exposure, EndpointEdgeKind::Exposure)]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::new(),
                    validity_sensitive_result: true,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::ImmediateValidity))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P3);
        assert!(candidates[0].exposure.is_some());
    }

    #[test]
    fn p3_exposure_only_pairs_with_a_public_root_that_reaches_the_unsafe_owner() {
        let site_owner = FunctionKey::new("crate::unsafe_helper");
        let causal_root = FunctionKey::new("crate::causal_root");
        let unrelated_root = FunctionKey::new("crate::unrelated_root");
        let site = bound_site(&site_owner, Some("&'static str"));
        let site_id = site.seed.site_id();
        let source = EndpointNode::UnsafeResult {
            owner: site_owner.clone(),
            site_id: site_id.clone(),
        };
        let causal_exposure = EndpointNode::Return {
            owner: causal_root.clone(),
        };
        let unrelated_exposure = EndpointNode::Return {
            owner: unrelated_root.clone(),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([
                source.clone(),
                causal_exposure.clone(),
                unrelated_exposure.clone(),
            ]),
            edges: BTreeSet::from([
                edge(source.clone(), causal_exposure, EndpointEdgeKind::Exposure),
                edge(source, unrelated_exposure, EndpointEdgeKind::Exposure),
            ]),
        };
        let calls = BTreeSet::from([CandidateCallEdge {
            caller: causal_root.clone(),
            callee: site_owner.clone(),
            point: point(&causal_root, 2),
            kind: CandidateCallKind::ResolvedLocal,
            generic_substitution: GenericCallSubstitution::default(),
        }]);
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::new(),
                    validity_sensitive_result: true,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::ImmediateValidity))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([
                (site_owner.clone(), owner_info(&site_owner, false)),
                (causal_root.clone(), owner_info(&causal_root, true)),
                (unrelated_root.clone(), owner_info(&unrelated_root, true)),
            ]),
            &calls,
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].public_root.as_ref(), &causal_root);
        assert_eq!(candidates[0].call_route.len(), 1);
        assert_eq!(candidates[0].call_route[0].callee, site_owner);
    }

    #[test]
    fn public_formal_causing_immediate_invalidity_remains_p1_when_result_is_exposed() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, Some("&'static str"));
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let operand = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let unsafe_result = EndpointNode::UnsafeResult {
            owner: owner.clone(),
            site_id: site_id.clone(),
        };
        let exposure = EndpointNode::Return {
            owner: owner.clone(),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([
                formal.clone(),
                operand.clone(),
                unsafe_result.clone(),
                exposure.clone(),
            ]),
            edges: BTreeSet::from([
                edge(formal.clone(), operand, EndpointEdgeKind::UnsafeOperand),
                edge(unsafe_result, exposure, EndpointEdgeKind::Exposure),
            ]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: true,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::ImmediateValidity))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P1);
        assert_eq!(candidates[0].source_kind, CausalSourceKind::PublicParameter);
        assert_eq!(candidates[0].source.as_ref(), &formal);
        assert_eq!(candidates[0].operand_role, Some(OperandRole::Argument(0)));
        assert!(candidates[0].exposure.is_none());
    }

    #[test]
    fn a_valid_for_type_contract_marks_a_user_enum_result_as_p3_sensitive() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, Some("crate::Tiny"));
        let site_id = site.seed.site_id();
        let source = EndpointNode::UnsafeResult {
            owner: owner.clone(),
            site_id: site_id.clone(),
        };
        let exposure = EndpointNode::Return {
            owner: owner.clone(),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), exposure.clone()]),
            edges: BTreeSet::from([edge(source, exposure, EndpointEdgeKind::Exposure)]),
        };
        let mut contract = descriptor(FailurePhase::ImmediateValidity);
        contract.predicates.insert(ContractPredicate::ValidForType);
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::new(),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), contract)]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P3);
    }

    #[test]
    fn precondition_failure_is_not_reclassified_as_p3_only_because_the_result_is_a_reference() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, Some("&'static u8"));
        let site_id = site.seed.site_id();
        let source = EndpointNode::UnsafeResult {
            owner: owner.clone(),
            site_id: site_id.clone(),
        };
        let exposure = EndpointNode::Return {
            owner: owner.clone(),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), exposure.clone()]),
            edges: BTreeSet::from([edge(source, exposure, EndpointEdgeKind::Exposure)]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::new(),
                    validity_sensitive_result: true,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert!(candidates.is_empty());
    }

    #[test]
    fn private_sink_uses_each_real_public_call_root_without_a_depth_cap() {
        let functions = (0..=64)
            .map(|index| FunctionKey::new(format!("crate::f{index}")))
            .collect::<Vec<_>>();
        let site = bound_site(&functions[64], None);
        let site_id = site.seed.site_id();
        let source = EndpointNode::InternalDerivation {
            owner: functions[64].clone(),
            kind: DerivationKind::Offset,
            token: "offset".into(),
            point: point(&functions[64], 70),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: functions[64].clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(source, target, EndpointEdgeKind::UnsafeOperand)]),
        };
        let calls = (0..64)
            .map(|index| CandidateCallEdge {
                caller: functions[index].clone(),
                callee: functions[index + 1].clone(),
                point: point(&functions[index], index as u32 + 1),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            })
            .collect::<BTreeSet<_>>();
        let owners = functions
            .iter()
            .enumerate()
            .map(|(index, owner)| (owner.clone(), owner_info(owner, index == 0)))
            .collect::<BTreeMap<_, _>>();
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), range_descriptor())]),
            &BTreeMap::from([(site_id, site)]),
            &owners,
            &calls,
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].call_route.len(), 64);
    }

    #[test]
    fn call_route_index_keeps_the_lexical_shortest_route() {
        let root = FunctionKey::new("crate::root");
        let left = FunctionKey::new("crate::a_left");
        let right = FunctionKey::new("crate::z_right");
        let target = FunctionKey::new("crate::target");
        let calls = BTreeSet::from([
            CandidateCallEdge {
                caller: root.clone(),
                callee: right.clone(),
                point: point(&root, 2),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: right,
                callee: target.clone(),
                point: point(&root, 4),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: root.clone(),
                callee: left.clone(),
                point: point(&root, 1),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: left.clone(),
                callee: target.clone(),
                point: point(&root, 3),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
            CandidateCallEdge {
                caller: left,
                callee: root.clone(),
                point: point(&root, 5),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            },
        ]);
        let index = CallRouteIndex::new(&calls);
        let route = index.routes_to(&target).route_from(&root).unwrap();
        assert_eq!(route.len(), 2);
        assert_eq!(route[0].callee, FunctionKey::new("crate::a_left"));
    }

    #[test]
    fn p5_route_gate_keeps_a_parametric_sibling_when_the_canonical_route_is_concrete() {
        let root_function = FunctionKey::new("crate::root");
        let helper = FunctionKey::new("crate::helper");
        let root_parameter = type_parameter("crate::root::T");
        let helper_parameter = type_parameter("crate::helper::U");
        let mut root = owner_info(&root_function, true);
        root.generic_surface.insert(root_parameter.clone());
        let evidence = sequence_capability(helper_parameter.clone());
        let concrete = generic_binding(
            helper_parameter.clone(),
            CapabilityTermForm::Concrete,
            BTreeSet::new(),
            GenericSubstitutionStatus::Complete,
            Some(0),
        );
        let parametric = generic_binding(
            helper_parameter,
            CapabilityTermForm::DirectParam,
            BTreeSet::from([root_parameter]),
            GenericSubstitutionStatus::Complete,
            Some(0),
        );
        let calls = BTreeSet::from([
            CandidateCallEdge {
                caller: root_function.clone(),
                callee: helper.clone(),
                point: point(&root_function, 1),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: concrete,
            },
            CandidateCallEdge {
                caller: root_function.clone(),
                callee: helper.clone(),
                point: point(&root_function, 2),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: parametric.clone(),
            },
        ]);
        let projected_parametric = project_capability_through_call(
            &CapabilityRouteState {
                owner: helper.clone(),
                terms: evidence
                    .terms
                    .iter()
                    .map(|(role, term)| (*role, ProjectedCapabilityTerm::from(term)))
                    .collect(),
                forced_equal: false,
                normalization_incomplete: false,
            },
            calls
                .iter()
                .find(|call| call.point == point(&root_function, 2))
                .expect("parametric sibling exists"),
        );
        assert!(
            parametric_witness_transport_established(&projected_parametric, &evidence, &root,),
            "projected={projected_parametric:?} evidence={evidence:?} root={root:?}"
        );
        let index = CallRouteIndex::new(&calls);
        let canonical = index.routes_to(&helper).route_from(&root_function).unwrap();
        assert!(canonical[0]
            .generic_substitution
            .bindings
            .values()
            .all(|binding| binding.caller_controllers.is_empty()));

        let CapabilityRouteResult::Parametric(route) =
            index.capability_route(&root, &helper, &evidence)
        else {
            panic!("the parametric sibling must keep the proven P5 witness active")
        };
        assert_eq!(route.len(), 1);
        assert_eq!(route[0].point, point(&root_function, 2));

        let site = bound_site(&helper, None);
        let site_id = site.seed.site_id();
        let source = EndpointNode::GenericCapability {
            owner: helper.clone(),
            evidence,
            span: span(20),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: helper.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(0),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([source.clone(), target.clone()]),
            edges: BTreeSet::from([edge(source, target, EndpointEdgeKind::UnsafeOperand)]),
        };
        let candidates = build_structural_candidates(
            &graph,
            &BTreeMap::from([(
                site_id.clone(),
                StructuralSiteCandidate {
                    site_id: site_id.clone(),
                    relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                    validity_sensitive_result: false,
                },
            )]),
            &BTreeMap::from([(site_id.clone(), range_descriptor())]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([
                (root_function.clone(), root),
                (helper.clone(), owner_info(&helper, false)),
            ]),
            &calls,
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P5);
        assert_eq!(candidates[0].call_route.len(), 1);
        assert_eq!(candidates[0].call_route[0].point, point(&root_function, 2));
        assert_eq!(
            candidates[0].call_route[0].generic_substitution, parametric,
            "the emitted witness must retain the parametric callsite identity"
        );
    }

    #[test]
    fn context_p5_keeps_parametric_callsite_and_a_contiguous_same_proof_witness() {
        let root_function = FunctionKey::new("crate::root");
        let helper = FunctionKey::new("crate::helper");
        let root_source = type_parameter("crate::root::S");
        let root_target = type_parameter("crate::root::T");
        let helper_source = type_parameter("crate::helper::S");
        let helper_target = type_parameter("crate::helper::T");
        let evidence = direct_pointer_capability(helper_source.clone(), helper_target.clone());
        let source = EndpointNode::GenericCapability {
            owner: helper.clone(),
            evidence: evidence.clone(),
            span: span(30),
        };
        let helper_return = EndpointNode::Return {
            owner: helper.clone(),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: root_function.clone(),
            site_id: "sink".into(),
            role: OperandRole::Argument(0),
        };
        let mut root = owner_info(&root_function, true);
        root.generic_surface = BTreeSet::from([root_source.clone(), root_target.clone()]);
        let mut context = ContextProofIndex::default();
        let callee_relation =
            ContextRelationKey::new(helper.clone(), source.clone(), helper_return.clone());
        context.record_local(EndpointEdge {
            from: source.clone(),
            to: helper_return.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(&helper, 1)),
        });
        let substitutions = [
            (
                10,
                GenericCallSubstitution {
                    status: GenericSubstitutionStatus::Complete,
                    bindings: BTreeMap::from([
                        (
                            helper_source.clone(),
                            concrete_type_binding(Some(8), Some(8), Some(1)),
                        ),
                        (
                            helper_target.clone(),
                            concrete_type_binding(Some(8), Some(8), Some(1)),
                        ),
                    ]),
                },
            ),
            (
                20,
                GenericCallSubstitution {
                    status: GenericSubstitutionStatus::Complete,
                    bindings: BTreeMap::from([
                        (
                            helper_source,
                            GenericArgBinding {
                                kind: super::super::endpoint_flow::GenericParamKind::Type,
                                form: CapabilityTermForm::DirectParam,
                                caller_controllers: BTreeSet::from([root_source]),
                                residual_aliases: BTreeSet::new(),
                                behavior_receiver: None,
                                equality_class: None,
                                type_facts: Default::default(),
                                const_value: None,
                            },
                        ),
                        (
                            helper_target,
                            GenericArgBinding {
                                kind: super::super::endpoint_flow::GenericParamKind::Type,
                                form: CapabilityTermForm::DirectParam,
                                caller_controllers: BTreeSet::from([root_target]),
                                residual_aliases: BTreeSet::new(),
                                behavior_receiver: None,
                                equality_class: None,
                                type_facts: Default::default(),
                                const_value: None,
                            },
                        ),
                    ]),
                },
            ),
        ];
        for (line, substitution) in substitutions {
            let callsite = super::super::endpoint_flow::CallSiteId::new(
                root_function.clone(),
                point(&root_function, line),
                helper.clone(),
            );
            let result = EndpointNode::CallResult {
                callsite: callsite.clone(),
                output: super::super::endpoint_flow::CallSiteOutput::Return,
            };
            context.record_call(
                EndpointEdge {
                    from: source.clone(),
                    to: result.clone(),
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(callsite.point.clone()),
                },
                callsite,
                substitution,
                callee_relation.clone(),
                ContextActualBinding::Internal,
            );
            context.record_local(EndpointEdge {
                from: result,
                to: target.clone(),
                kind: EndpointEdgeKind::UnsafeOperand,
                point: Some(point(&root_function, line + 1)),
            });
        }
        context.finalize();
        let P5RouteOutcome::Parametric {
            call_route,
            endpoint_route: Some(endpoint_route),
        } = capability_route_result_from_context(&context, &root, &source, &target, &evidence)
        else {
            panic!("the complete parametric sibling must remain P5-active");
        };
        assert_eq!(call_route.len(), 1);
        assert_eq!(call_route[0].point, point(&root_function, 20));
        assert!(endpoint_route
            .windows(2)
            .all(|pair| pair[0].to == pair[1].from));
        assert_eq!(endpoint_route.first().map(|edge| &edge.from), Some(&source));
        assert_eq!(endpoint_route.last().map(|edge| &edge.to), Some(&target));
    }

    #[test]
    fn context_p5_requires_a_joint_typed_transport_for_incomplete_internal_projection() {
        let root_function = FunctionKey::new("crate::root");
        let helper = FunctionKey::new("crate::helper");
        let root_parameter = type_parameter("crate::root::P");
        let helper_source = type_parameter("crate::helper::S");
        let helper_target = type_parameter("crate::helper::T");
        let evidence = direct_pointer_capability(helper_source.clone(), helper_target.clone());
        let source = EndpointNode::GenericCapability {
            owner: helper.clone(),
            evidence: evidence.clone(),
            span: span(40),
        };
        let helper_return = EndpointNode::Return {
            owner: helper.clone(),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: root_function.clone(),
            site_id: "typed-transport-sink".into(),
            role: OperandRole::Argument(0),
        };
        let mut root = owner_info(&root_function, true);
        root.generic_surface.insert(root_parameter.clone());
        let callee_relation =
            ContextRelationKey::new(helper.clone(), source.clone(), helper_return.clone());
        let callsite = super::super::endpoint_flow::CallSiteId::new(
            root_function.clone(),
            point(&root_function, 41),
            helper.clone(),
        );
        let result = EndpointNode::CallResult {
            callsite: callsite.clone(),
            output: super::super::endpoint_flow::CallSiteOutput::Return,
        };
        let alias_binding = |alias: &str| GenericArgBinding {
            kind: super::super::endpoint_flow::GenericParamKind::Type,
            form: CapabilityTermForm::Alias,
            caller_controllers: BTreeSet::from([root_parameter.clone()]),
            residual_aliases: BTreeSet::from([alias.to_string()]),
            behavior_receiver: None,
            equality_class: None,
            type_facts: Default::default(),
            const_value: None,
        };
        let substitution = GenericCallSubstitution {
            status: GenericSubstitutionStatus::NormalizationIncomplete,
            bindings: BTreeMap::from([
                (
                    helper_source.clone(),
                    alias_binding("crate::ReprPair::Source"),
                ),
                (
                    helper_target.clone(),
                    alias_binding("crate::ReprPair::Target"),
                ),
            ]),
        };
        let mut context = ContextProofIndex::default();
        context.record_local(EndpointEdge {
            from: source.clone(),
            to: helper_return.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(&helper, 40)),
        });
        context.record_call(
            EndpointEdge {
                from: source.clone(),
                to: result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite.clone(),
            substitution,
            callee_relation,
            ContextActualBinding::Internal,
        );
        // A compiler-normalized sibling cannot bypass the certificate gate:
        // it carries the same projected aliases and therefore still needs a
        // joint source/target assignment at this exact callsite.
        context.record_call(
            EndpointEdge {
                from: source.clone(),
                to: result.clone(),
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            callsite.clone(),
            GenericCallSubstitution {
                status: GenericSubstitutionStatus::Complete,
                bindings: BTreeMap::from([
                    (
                        helper_source.clone(),
                        alias_binding("crate::ReprPair::Source"),
                    ),
                    (
                        helper_target.clone(),
                        alias_binding("crate::ReprPair::Target"),
                    ),
                ]),
            },
            ContextRelationKey::new(helper.clone(), source.clone(), helper_return.clone()),
            ContextActualBinding::Internal,
        );
        context.record_local(EndpointEdge {
            from: result,
            to: target.clone(),
            kind: EndpointEdgeKind::UnsafeOperand,
            point: Some(point(&root_function, 42)),
        });
        context.finalize();

        assert!(matches!(
            capability_route_result_from_context(&context, &root, &source, &target, &evidence),
            P5RouteOutcome::Unestablished
        ));

        let CapabilityWitness::Proven { assignment, .. } = &evidence.witness else {
            panic!("direct pointer fixture must have a finite witness");
        };
        let mut unequal_assignment = assignment.clone();
        unequal_assignment.insert(
            CapabilityTermRole::TargetPointee,
            super::super::endpoint_flow::FiniteTypeWitness::U8,
        );
        let template = |assignment| super::super::endpoint_flow::TypedWitnessTransportTemplate {
            source_parameter: helper_source.clone(),
            target_parameter: helper_target.clone(),
            caller_controller: root_parameter.clone(),
            assignment,
        };
        context.record_typed_witness_transports(
            callsite.clone(),
            evidence.source_identity(),
            [template(unequal_assignment)],
        );
        assert!(matches!(
            capability_route_result_from_context(&context, &root, &source, &target, &evidence),
            P5RouteOutcome::Unestablished
        ));

        context.record_typed_witness_transports(
            callsite,
            evidence.source_identity(),
            [template(assignment.clone())],
        );
        let P5RouteOutcome::Parametric { call_route, .. } =
            capability_route_result_from_context(&context, &root, &source, &target, &evidence)
        else {
            panic!("the exact joint source/target certificate must transport the witness");
        };
        assert_eq!(call_route.len(), 1);
        assert_eq!(call_route[0].point, point(&root_function, 41));
    }

    #[test]
    fn context_p5_projection_equality_class_proves_layout_satisfaction() {
        let helper_source = type_parameter("crate::helper::S");
        let helper_target = type_parameter("crate::helper::T");
        let caller_parameter = type_parameter("crate::root::P");
        let evidence = direct_pointer_capability(helper_source.clone(), helper_target.clone());
        let equal_alias = |alias: &str| GenericArgBinding {
            kind: super::super::endpoint_flow::GenericParamKind::Type,
            form: CapabilityTermForm::Alias,
            caller_controllers: BTreeSet::from([caller_parameter.clone()]),
            residual_aliases: BTreeSet::from([alias.to_string()]),
            behavior_receiver: None,
            equality_class: Some(7),
            type_facts: Default::default(),
            const_value: None,
        };
        let substitution = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([
                (helper_source, equal_alias("crate::ReprPair::Source")),
                (helper_target, equal_alias("crate::ReprPair::Target")),
            ]),
        };

        let projected = project_context_capability_through_substitution(
            &ContextCapabilityState::from_evidence(&evidence),
            &substitution,
            true,
            false,
        );

        assert!(projected.forced_equal);
        assert!(!projected.normalization_incomplete);
        assert!(matches!(
            evaluate_concrete_capability(&projected.concrete_view(), &evidence.requirement),
            ConcreteCapabilityResult::Satisfied
        ));
    }

    #[test]
    fn p5_fixed_layout_composite_is_proved_before_generic_surface() {
        let helper_source = type_parameter("crate::helper::S");
        let helper_target = type_parameter("crate::helper::T");
        let root_parameter = type_parameter("crate::root::P");
        let evidence = direct_pointer_capability(helper_source.clone(), helper_target.clone());
        let substitution = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([
                (
                    helper_source,
                    GenericArgBinding {
                        kind: super::super::endpoint_flow::GenericParamKind::Type,
                        form: CapabilityTermForm::Composite,
                        caller_controllers: BTreeSet::from([root_parameter.clone()]),
                        residual_aliases: BTreeSet::new(),
                        behavior_receiver: None,
                        equality_class: Some(1),
                        type_facts: CapabilityTypeFacts {
                            size: Some(8),
                            align: Some(8),
                            inhabited: Some(true),
                        },
                        const_value: None,
                    },
                ),
                (
                    helper_target,
                    concrete_type_binding(Some(8), Some(8), Some(2)),
                ),
            ]),
        };
        let projected = project_context_capability_through_substitution(
            &ContextCapabilityState::from_evidence(&evidence),
            &substitution,
            false,
            false,
        );
        let mut root = owner_info(&FunctionKey::new("crate::root"), true);
        root.generic_surface.insert(root_parameter);

        assert!(matches!(
            evaluate_concrete_capability(&projected.concrete_view(), &evidence.requirement),
            ConcreteCapabilityResult::Satisfied
        ));
        assert!(!parametric_witness_transport_established(
            &projected.concrete_view(),
            &evidence,
            &root,
        ));
    }

    #[test]
    fn p5_unknown_composite_does_not_reuse_a_source_local_witness() {
        let helper_source = type_parameter("crate::helper::S");
        let helper_target = type_parameter("crate::helper::T");
        let root_source = type_parameter("crate::root::S");
        let root_target = type_parameter("crate::root::T");
        let evidence = direct_pointer_capability(helper_source.clone(), helper_target.clone());
        let composite = |controller: GenericParamId, equality_class| GenericArgBinding {
            kind: super::super::endpoint_flow::GenericParamKind::Type,
            form: CapabilityTermForm::Composite,
            caller_controllers: BTreeSet::from([controller]),
            residual_aliases: BTreeSet::new(),
            behavior_receiver: None,
            equality_class: Some(equality_class),
            type_facts: CapabilityTypeFacts::default(),
            const_value: None,
        };
        let substitution = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([
                (helper_source, composite(root_source.clone(), 1)),
                (helper_target, composite(root_target.clone(), 2)),
            ]),
        };
        let projected = project_context_capability_through_substitution(
            &ContextCapabilityState::from_evidence(&evidence),
            &substitution,
            false,
            false,
        );
        let mut root = owner_info(&FunctionKey::new("crate::root"), true);
        root.generic_surface = BTreeSet::from([root_source, root_target]);

        assert!(matches!(
            evaluate_concrete_capability(&projected.concrete_view(), &evidence.requirement),
            ConcreteCapabilityResult::Unestablished
        ));
        assert!(!parametric_witness_transport_established(
            &projected.concrete_view(),
            &evidence,
            &root,
        ));
    }

    #[test]
    fn context_p5_nested_internal_transport_checks_the_returned_child_state() {
        let root_function = FunctionKey::new("crate::root");
        let middle = FunctionKey::new("crate::middle");
        let leaf = FunctionKey::new("crate::leaf");
        let root_parameter = type_parameter("crate::root::P");
        let middle_source = type_parameter("crate::middle::S");
        let middle_target = type_parameter("crate::middle::T");
        let leaf_source = type_parameter("crate::leaf::S");
        let leaf_target = type_parameter("crate::leaf::T");
        let evidence = direct_pointer_capability(leaf_source.clone(), leaf_target.clone());
        let source = EndpointNode::GenericCapability {
            owner: leaf.clone(),
            evidence: evidence.clone(),
            span: span(50),
        };
        let target = EndpointNode::UnsafeOperand {
            owner: root_function.clone(),
            site_id: "nested-typed-transport-sink".into(),
            role: OperandRole::Argument(0),
        };
        let parent = P5Request {
            scope: root_function.clone(),
            start: ContextAtom::scoped(root_function.clone(), source.clone()),
            target: ContextAtom::scoped(root_function.clone(), target.clone()),
            seed: ContextCapabilityState::from_evidence(&evidence),
        };
        let direct_binding = |parameter: GenericParamId| GenericArgBinding {
            kind: super::super::endpoint_flow::GenericParamKind::Type,
            form: CapabilityTermForm::DirectParam,
            caller_controllers: BTreeSet::from([parameter]),
            residual_aliases: BTreeSet::new(),
            behavior_receiver: None,
            equality_class: None,
            type_facts: Default::default(),
            const_value: None,
        };
        let inner_substitution = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([
                (leaf_source.clone(), direct_binding(middle_source.clone())),
                (leaf_target.clone(), direct_binding(middle_target.clone())),
            ]),
        };
        // Model the exact return fact of a nested leaf -> middle call.  The
        // outer waiter must inspect these middle parameters, not the original
        // leaf parameters present before the nested call began.
        let returned_child_state = project_context_capability_through_substitution(
            &ContextCapabilityState::from_evidence(&evidence),
            &inner_substitution,
            false,
            false,
        );
        let alias_binding = |alias: &str| GenericArgBinding {
            kind: super::super::endpoint_flow::GenericParamKind::Type,
            form: CapabilityTermForm::Alias,
            caller_controllers: BTreeSet::from([root_parameter.clone()]),
            residual_aliases: BTreeSet::from([alias.to_string()]),
            behavior_receiver: None,
            equality_class: None,
            type_facts: Default::default(),
            const_value: None,
        };
        let outer_substitution = GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([
                (
                    middle_source.clone(),
                    alias_binding("crate::ReprPair::Source"),
                ),
                (
                    middle_target.clone(),
                    alias_binding("crate::ReprPair::Target"),
                ),
            ]),
        };
        let callsite = super::super::endpoint_flow::CallSiteId::new(
            root_function.clone(),
            point(&root_function, 51),
            middle.clone(),
        );
        let waiter = P5Waiter {
            parent: parent.clone(),
            next: parent.target.clone(),
            parent_proof: 0,
            edge: EndpointEdge {
                from: source.clone(),
                to: target,
                kind: EndpointEdgeKind::CallSummary,
                point: Some(callsite.point.clone()),
            },
            call: candidate_call_from_context(&callsite, &outer_substitution),
            callsite: callsite.clone(),
            actual_binding: ContextActualBinding::Internal,
            callee_start: ContextAtom::scoped(middle.clone(), source.clone()),
            callee_target: ContextAtom::scoped(middle, EndpointNode::Return { owner: leaf }),
        };
        let child = P5Fact {
            state: returned_child_state,
            proof: 0,
        };
        let context = ContextProofIndex::default();
        let mut solver = ContextP5Solver::new(&context, Some(&evidence));
        solver.resume_waiter(&waiter, child.clone());
        assert!(solver
            .pending
            .pop_front()
            .is_some_and(|pending| pending.state.normalization_incomplete));

        let CapabilityWitness::Proven { assignment, .. } = &evidence.witness else {
            panic!("direct pointer fixture must have a finite witness");
        };
        let mut certified_context = ContextProofIndex::default();
        certified_context.record_typed_witness_transports(
            callsite,
            evidence.source_identity(),
            [super::super::endpoint_flow::TypedWitnessTransportTemplate {
                source_parameter: middle_source,
                target_parameter: middle_target,
                caller_controller: root_parameter,
                assignment: assignment.clone(),
            }],
        );
        let mut solver = ContextP5Solver::new(&certified_context, Some(&evidence));
        solver.resume_waiter(&waiter, child);
        assert!(solver
            .pending
            .pop_front()
            .is_some_and(|pending| !pending.state.normalization_incomplete));
    }

    #[test]
    fn p5_witness_reverses_a_nested_exact_call_chain_only_once() {
        let owner = FunctionKey::new("crate::root");
        let atom = |index| {
            ContextAtom::scoped(
                owner.clone(),
                EndpointNode::Formal {
                    owner: owner.clone(),
                    index,
                },
            )
        };
        let context = ContextProofIndex::default();
        let mut solver = ContextP5Solver::new(&context, None);
        let mut proof = 0;
        for line in (1..=64).rev() {
            let call = CandidateCallEdge {
                caller: owner.clone(),
                callee: FunctionKey::new(format!("crate::callee_{line}")),
                point: point(&owner, line),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: GenericCallSubstitution::default(),
            };
            proof = solver.new_proof(P5Proof::Call {
                parent: proof,
                child: 0,
                edge: EndpointEdge {
                    from: atom(1).endpoint,
                    to: atom(2).endpoint,
                    kind: EndpointEdgeKind::CallSummary,
                    point: Some(point(&owner, line)),
                },
                call,
                actual_binding: ContextActualBinding::Internal,
                callee_start: atom(1),
                callee_target: atom(1),
            });
        }
        let (_, calls) = solver.render_proof(proof);
        assert_eq!(calls.len(), 64);
        assert_eq!(
            calls
                .iter()
                .map(|call| call.point.block)
                .collect::<Vec<_>>(),
            (1..=64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn p5_route_gate_does_not_treat_bare_concrete_as_a_safety_proof() {
        let root_function = FunctionKey::new("crate::root");
        let helper = FunctionKey::new("crate::helper");
        let helper_parameter = type_parameter("crate::helper::U");
        let evidence = sequence_capability(helper_parameter.clone());
        let calls = BTreeSet::from([CandidateCallEdge {
            caller: root_function.clone(),
            callee: helper.clone(),
            point: point(&root_function, 1),
            kind: CandidateCallKind::ResolvedLocal,
            generic_substitution: generic_binding(
                helper_parameter,
                CapabilityTermForm::Concrete,
                BTreeSet::new(),
                GenericSubstitutionStatus::Complete,
                Some(0),
            ),
        }]);
        let index = CallRouteIndex::new(&calls);
        assert!(matches!(
            index.capability_route(&owner_info(&root_function, true), &helper, &evidence),
            CapabilityRouteResult::Unestablished
        ));
    }

    #[test]
    fn p5_route_gate_preserves_concrete_facts_for_satisfaction_and_violation() {
        let root_function = FunctionKey::new("crate::root");
        let helper = FunctionKey::new("crate::helper");
        let source = type_parameter("crate::helper::S");
        let target = type_parameter("crate::helper::T");
        let evidence = direct_pointer_capability(source.clone(), target.clone());
        let substitution = |source_binding, target_binding| GenericCallSubstitution {
            status: GenericSubstitutionStatus::Complete,
            bindings: BTreeMap::from([
                (source.clone(), source_binding),
                (target.clone(), target_binding),
            ]),
        };
        let result = |substitution| {
            let calls = BTreeSet::from([CandidateCallEdge {
                caller: root_function.clone(),
                callee: helper.clone(),
                point: point(&root_function, 7),
                kind: CandidateCallKind::ResolvedLocal,
                generic_substitution: substitution,
            }]);
            CallRouteIndex::new(&calls).capability_route(
                &owner_info(&root_function, true),
                &helper,
                &evidence,
            )
        };

        assert!(matches!(
            result(substitution(
                concrete_type_binding(Some(8), Some(8), Some(1)),
                concrete_type_binding(Some(8), Some(8), Some(1)),
            )),
            CapabilityRouteResult::ConcreteSatisfied
        ));
        match result(substitution(
            concrete_type_binding(Some(1), Some(1), Some(1)),
            concrete_type_binding(Some(8), Some(8), Some(2)),
        )) {
            CapabilityRouteResult::ConcreteViolation { route, violations } => {
                assert_eq!(route.len(), 1);
                assert_eq!(route[0].point, point(&root_function, 7));
                assert_eq!(
                    violations,
                    BTreeSet::from([CapabilityViolation::Size, CapabilityViolation::Alignment])
                );
            }
            _ => panic!("u8 -> u64 must remain a concrete capability violation"),
        }
        assert!(matches!(
            result(substitution(
                concrete_type_binding(None, None, None),
                concrete_type_binding(Some(8), Some(8), Some(2)),
            )),
            CapabilityRouteResult::Unestablished
        ));
    }

    #[test]
    fn p5_route_gate_treats_incomplete_normalization_as_unestablished() {
        let root_function = FunctionKey::new("crate::root");
        let helper = FunctionKey::new("crate::helper");
        let root_parameter = type_parameter("crate::root::T");
        let helper_parameter = type_parameter("crate::helper::U");
        let mut root = owner_info(&root_function, true);
        root.generic_surface.insert(root_parameter.clone());
        let evidence = sequence_capability(helper_parameter.clone());
        let calls = BTreeSet::from([CandidateCallEdge {
            caller: root_function.clone(),
            callee: helper.clone(),
            point: point(&root_function, 1),
            kind: CandidateCallKind::ResolvedLocal,
            generic_substitution: generic_binding(
                helper_parameter,
                CapabilityTermForm::DirectParam,
                BTreeSet::from([root_parameter]),
                GenericSubstitutionStatus::NormalizationIncomplete,
                Some(0),
            ),
        }]);
        let index = CallRouteIndex::new(&calls);
        assert!(matches!(
            index.capability_route(&root, &helper, &evidence),
            CapabilityRouteResult::Unestablished
        ));
    }

    #[test]
    fn wrong_operand_role_and_insertion_order_do_not_create_or_reorder_candidates() {
        let owner = FunctionKey::new("crate::root");
        let site = bound_site(&owner, None);
        let site_id = site.seed.site_id();
        let formal = EndpointNode::Formal {
            owner: owner.clone(),
            index: 1,
        };
        let wrong = EndpointNode::UnsafeOperand {
            owner: owner.clone(),
            site_id: site_id.clone(),
            role: OperandRole::Argument(1),
        };
        let graph = EndpointFlowGraph {
            nodes: BTreeSet::from([formal.clone(), wrong.clone()]),
            edges: BTreeSet::from([edge(formal, wrong, EndpointEdgeKind::UnsafeOperand)]),
        };
        let build = |graph: &EndpointFlowGraph| {
            build_structural_candidates(
                graph,
                &BTreeMap::from([(
                    site_id.clone(),
                    StructuralSiteCandidate {
                        site_id: site_id.clone(),
                        relevant_roles: BTreeSet::from([OperandRole::Argument(0)]),
                        validity_sensitive_result: false,
                    },
                )]),
                &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
                &BTreeMap::from([(site_id.clone(), site.clone())]),
                &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
                &BTreeSet::new(),
            )
        };
        assert!(build(&graph).is_empty());
        let mut reversed = EndpointFlowGraph::default();
        for edge in graph.edges.iter().rev() {
            reversed.insert_edge(edge.clone());
        }
        assert_eq!(build(&graph), build(&reversed));
    }
}
