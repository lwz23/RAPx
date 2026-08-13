use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionKey(pub String);

impl FunctionKey {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OriginKey(pub String);

impl OriginKey {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlaceKey(pub String);

impl PlaceKey {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StablePosition {
    pub line: u32,
    pub column: u32,
}

impl StablePosition {
    pub fn new(line: u32, column: u32) -> Self {
        assert!(line > 0 && column > 0, "source positions are one-based");
        Self { line, column }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableSpan {
    pub path: String,
    pub start: StablePosition,
    pub end: StablePosition,
}

impl StableSpan {
    pub fn new(path: impl Into<String>, start: StablePosition, end: StablePosition) -> Self {
        let path = path.into().replace('\\', "/");
        assert!(!path.is_empty(), "span path must not be empty");
        assert!(
            !path.starts_with('/'),
            "span path must be repository-relative"
        );
        assert!(
            !path
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == ".."),
            "span path must be normalized"
        );
        assert!(end >= start, "span end must not precede its start");
        Self { path, start, end }
    }

    pub fn token(&self) -> String {
        format!(
            "{}:{}:{}-{}:{}",
            self.path, self.start.line, self.start.column, self.end.line, self.end.column
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramPoint {
    pub function: FunctionKey,
    pub block: u32,
    pub statement: u32,
    pub span: StableSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceKind {
    PublicParameter,
    LiteralPublicField,
    InternalUnsafeOrigin,
    InternalDerived,
    GenericNonEmptyCapability,
    FfiOutput,
    OpenBehaviorOutput,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PublicParameter => "public_parameter",
            Self::LiteralPublicField => "literal_public_field",
            Self::InternalUnsafeOrigin => "internal_unsafe_origin",
            Self::InternalDerived => "internal_derived",
            Self::GenericNonEmptyCapability => "generic_nonempty_capability",
            Self::FfiOutput => "ffi_output",
            Self::OpenBehaviorOutput => "open_behavior_output",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Source {
    pub kind: SourceKind,
    pub origin: OriginKey,
    pub span: StableSpan,
}

impl Source {
    pub fn canonical_token(&self) -> String {
        format!(
            "{}:{}@{}",
            self.kind.as_str(),
            self.origin.0,
            self.span.token()
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperationKind {
    RawRead,
    GetUnchecked,
    ReadUnaligned,
    NonNullNewUnchecked,
    LifetimeTransmute,
    AssumeInitBool,
    FromUtf8Unchecked,
    InvalidValueExposure,
}

impl OperationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawRead => "raw_read",
            Self::GetUnchecked => "get_unchecked",
            Self::ReadUnaligned => "read_unaligned",
            Self::NonNullNewUnchecked => "nonnull_new_unchecked",
            Self::LifetimeTransmute => "lifetime_transmute",
            Self::AssumeInitBool => "assume_init",
            Self::FromUtf8Unchecked => "from_utf8_unchecked",
            Self::InvalidValueExposure => "invalid_value_exposure",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Operation {
    pub kind: OperationKind,
    pub point: ProgramPoint,
}

impl Operation {
    pub fn canonical_token(&self) -> String {
        format!("{}@{}", self.kind.as_str(), self.point.span.token())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Predicate {
    ValidForRead,
    InBounds,
    NonEmpty,
    RangeInBounds,
    NonNull,
    ReferentOutlivesReference,
    Initialized,
    ValidBool,
    ValidUtf8,
}

impl Predicate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ValidForRead => "valid_for_read",
            Self::InBounds => "in_bounds",
            Self::NonEmpty => "non_empty",
            Self::RangeInBounds => "range_in_bounds",
            Self::NonNull => "nonnull",
            Self::ReferentOutlivesReference => "referent_outlives_reference",
            Self::Initialized => "initialized",
            Self::ValidBool => "valid_bool",
            Self::ValidUtf8 => "valid_utf8",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Obligation {
    pub predicates: BTreeSet<Predicate>,
    pub subject: OriginKey,
}

impl Obligation {
    pub fn new(predicates: impl IntoIterator<Item = Predicate>, subject: OriginKey) -> Self {
        let predicates = predicates.into_iter().collect::<BTreeSet<_>>();
        assert!(
            !predicates.is_empty(),
            "obligations need at least one predicate"
        );
        Self {
            predicates,
            subject,
        }
    }

    pub fn canonical_token(&self) -> String {
        let predicates = self
            .predicates
            .iter()
            .map(|predicate| predicate.as_str())
            .collect::<Vec<_>>()
            .join("+");
        format!("{}:{}", predicates, self.subject.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundarySlot {
    Formal(u32),
    Return,
    Out(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FlowKind {
    Copy,
    Move,
    Cast,
    Projection,
    FieldLoad,
    FieldStore,
    ActualToFormal,
    ReturnToDestination,
    OutToCaller,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlowFact {
    pub from: PlaceKey,
    pub to: PlaceKey,
    pub kind: FlowKind,
    pub point: ProgramPoint,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallMapping {
    ActualToFormal {
        actual: PlaceKey,
        formal_index: u32,
    },
    ReturnToDestination {
        destination: PlaceKey,
    },
    OutToCaller {
        formal_index: u32,
        caller_place: PlaceKey,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallBoundary {
    pub caller: FunctionKey,
    pub callee: FunctionKey,
    pub point: ProgramPoint,
    pub mapping: BTreeSet<CallMapping>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutDependency {
    pub formal_index: u32,
    pub value: AbstractValue,
    pub may_skip_write: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WriteEffect {
    pub place: PlaceKey,
    pub point: ProgramPoint,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SinkObligation {
    pub source: Source,
    pub first_failure: Operation,
    pub sink: Operation,
    pub obligation: Obligation,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AbstractOrigin {
    Formal(u32),
    PublicField {
        def_path: String,
        span: StableSpan,
    },
    PrivateField {
        def_path: String,
        span: StableSpan,
    },
    InternalLocal {
        function: FunctionKey,
        place: PlaceKey,
    },
    GenericCapability {
        token: String,
        span: StableSpan,
    },
    FfiOutput {
        token: String,
        span: StableSpan,
    },
    OpenBehaviorOutput {
        token: String,
        span: StableSpan,
    },
    Constant(u64),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbstractValue {
    pub origins: BTreeSet<AbstractOrigin>,
}

impl AbstractValue {
    pub fn new(origins: impl IntoIterator<Item = AbstractOrigin>) -> Self {
        Self {
            origins: origins.into_iter().collect(),
        }
    }

    pub fn substitute(&self, actuals: &[AbstractValue]) -> Self {
        let mut origins = BTreeSet::new();
        for origin in &self.origins {
            if let AbstractOrigin::Formal(index) = origin {
                if *index > 0 {
                    if let Some(actual) = actuals.get(*index as usize - 1) {
                        origins.extend(actual.origins.iter().cloned());
                        continue;
                    }
                }
            }
            origins.insert(origin.clone());
        }
        Self { origins }
    }
}

pub fn map_call_outputs(
    mappings: &BTreeSet<CallMapping>,
    return_value: &AbstractValue,
    out_values: &BTreeMap<u32, AbstractValue>,
    actuals: &[AbstractValue],
) -> BTreeMap<PlaceKey, AbstractValue> {
    let mut mapped = BTreeMap::<PlaceKey, AbstractValue>::new();
    for mapping in mappings {
        let (destination, value) = match mapping {
            CallMapping::ActualToFormal { .. } => continue,
            CallMapping::ReturnToDestination { destination } => {
                (destination, return_value.substitute(actuals))
            }
            CallMapping::OutToCaller {
                formal_index,
                caller_place,
            } => {
                let Some(value) = out_values.get(formal_index) else {
                    continue;
                };
                (caller_place, value.substitute(actuals))
            }
        };
        if !value.origins.is_empty() {
            mapped
                .entry(destination.clone())
                .or_default()
                .origins
                .extend(value.origins);
        }
    }
    mapped
}

pub fn out_values_from_dependencies(
    dependencies: &BTreeSet<OutDependency>,
) -> BTreeMap<u32, AbstractValue> {
    let mut values = BTreeMap::<u32, AbstractValue>::new();
    for dependency in dependencies {
        values
            .entry(dependency.formal_index)
            .or_default()
            .origins
            .extend(dependency.value.origins.iter().cloned());
    }
    values
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContractRequirement {
    pub seed_id: String,
    pub collection: Option<AbstractValue>,
    pub subject: AbstractValue,
    pub source_hint: Option<SourceKind>,
    pub internal_derivation: bool,
    pub source_span: StableSpan,
    pub first_failure: Operation,
    pub sink: Operation,
    pub predicates: BTreeSet<Predicate>,
    pub access_width: Option<u64>,
    pub rule: RuleId,
    pub return_exposure: bool,
    pub sink_function: FunctionKey,
}

/// A finite, conditional validation contract exported at a function boundary.
///
/// Task 6 production extraction exports only entry formals. Return and out
/// slots are reserved for the structural output work package and are rejected
/// by `export_entry_contract` until those mappings are populated from MIR.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValidationFact {
    pub requirement: ContractRequirement,
    pub subject: BoundarySlot,
    pub collection: Option<BoundarySlot>,
}

impl ValidationFact {
    pub fn export_entry_contract(
        requirement: ContractRequirement,
        subject: Option<BoundarySlot>,
        collection: Option<BoundarySlot>,
        dominates_sink: bool,
        current_after_writes: bool,
    ) -> Option<Self> {
        if !dominates_sink || !current_after_writes || requirement.predicates.len() != 1 {
            return None;
        }
        let predicate = *requirement.predicates.iter().next()?;
        if !matches!(
            predicate,
            Predicate::InBounds | Predicate::NonEmpty | Predicate::NonNull
        ) {
            return None;
        }
        let subject = match subject? {
            BoundarySlot::Formal(index) if index > 0 => BoundarySlot::Formal(index),
            _ => return None,
        };
        let collection = match collection {
            Some(BoundarySlot::Formal(index)) if index > 0 => Some(BoundarySlot::Formal(index)),
            Some(_) => return None,
            None => None,
        };
        if predicate == Predicate::InBounds && collection.is_none() {
            return None;
        }
        Some(Self {
            requirement,
            subject,
            collection,
        })
    }

    pub fn instantiate_requirement(&self, actuals: &[AbstractValue]) -> ContractRequirement {
        let mut requirement = self.requirement.clone();
        requirement.subject = requirement.subject.substitute(actuals);
        requirement.collection = requirement
            .collection
            .as_ref()
            .map(|collection| collection.substitute(actuals));
        requirement
    }

    pub fn matches_instantiated_requirement(
        &self,
        requirement: &ContractRequirement,
        actuals: &[AbstractValue],
    ) -> bool {
        let expected = self.instantiate_requirement(actuals);
        expected.predicates == requirement.predicates
            && expected.subject == requirement.subject
            && expected.collection == requirement.collection
    }
}

pub fn intersect_validation_paths<'a>(
    paths: impl IntoIterator<Item = &'a BTreeSet<ValidationFact>>,
) -> BTreeSet<ValidationFact> {
    let mut paths = paths.into_iter();
    let Some(first) = paths.next() else {
        return BTreeSet::new();
    };
    let mut common = first.clone();
    for path in paths {
        common.retain(|fact| path.contains(fact));
    }
    common
}

/// The exact unresolved terminal that keeps one route unsafe.
///
/// Hard and conditional terminals intentionally remain distinct even when
/// they refer to the same instantiated requirement. This lets Task 8 witness
/// selection follow the route that actually survived validation composition.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnresolvedRouteFact {
    Hard(ContractRequirement),
    Conditional(ValidationFact),
}

impl UnresolvedRouteFact {
    pub fn requirement(&self) -> &ContractRequirement {
        match self {
            Self::Hard(requirement) => requirement,
            Self::Conditional(validation) => &validation.requirement,
        }
    }
}

/// One finite caller-to-callee predecessor relation for an unresolved route.
///
/// The caller is the owner of the containing `FunctionSummary`; the callee is
/// explicit. Only one call edge is stored, never an expanded path or depth.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnresolvedPredecessor {
    pub caller_route: UnresolvedRouteFact,
    pub callee_route: UnresolvedRouteFact,
    pub callee: FunctionKey,
    pub call_point: ProgramPoint,
}

/// An exact route terminal paired with the summary that owns it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnresolvedRouteNode {
    pub owner: FunctionKey,
    pub route: UnresolvedRouteFact,
}

/// One selected caller-owned edge. This explicit owner is post-solve witness
/// state; it is deliberately absent from the per-summary predecessor fact.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnresolvedRouteStep {
    pub caller: FunctionKey,
    pub edge: UnresolvedPredecessor,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct UnresolvedRouteRankStep {
    callee: FunctionKey,
    call_span: StableSpan,
    call_block: u32,
    call_statement: u32,
    callee_route: UnresolvedRouteFact,
    caller: FunctionKey,
    caller_route: UnresolvedRouteFact,
}

impl From<&UnresolvedRouteStep> for UnresolvedRouteRankStep {
    fn from(step: &UnresolvedRouteStep) -> Self {
        Self {
            callee: step.edge.callee.clone(),
            call_span: step.edge.call_point.span.clone(),
            call_block: step.edge.call_point.block,
            call_statement: step.edge.call_point.statement,
            callee_route: step.edge.callee_route.clone(),
            caller: step.caller.clone(),
            caller_route: step.edge.caller_route.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedUnresolvedRoute {
    pub root: UnresolvedRouteNode,
    pub terminal: UnresolvedRouteNode,
    pub steps: Vec<UnresolvedRouteStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnresolvedRouteCycle {
    pub nodes: BTreeSet<UnresolvedRouteNode>,
    pub token: String,
    pub span: StableSpan,
}

fn summary_contains_route(summary: &FunctionSummary, route: &UnresolvedRouteFact) -> bool {
    match route {
        UnresolvedRouteFact::Hard(requirement) => summary.requirements.contains(requirement),
        UnresolvedRouteFact::Conditional(validation) => summary.validations.contains(validation),
    }
}

/// Selects a deterministic path only after unresolved validity is known.
///
/// Search state is finite because each `(summary owner, exact route fact)` is
/// visited at most once. The frontier ranks paths by edge count, then by the
/// stable callee def-path and call span/point sequence; exact route facts only
/// break ties at the same call edge. A terminal must be an exact fact in the
/// function's local seed summary; failure to find one is fail-closed.
pub fn select_unresolved_route(
    root: UnresolvedRouteNode,
    summaries: &BTreeMap<FunctionKey, FunctionSummary>,
    seeds: &BTreeMap<FunctionKey, FunctionSummary>,
) -> Result<SelectedUnresolvedRoute, String> {
    type Candidate = (
        usize,
        Vec<UnresolvedRouteRankStep>,
        UnresolvedRouteNode,
        Vec<UnresolvedRouteStep>,
    );
    let root_for_error = root.clone();
    let mut frontier = BTreeSet::<Candidate>::from([(0, Vec::new(), root, Vec::new())]);
    let mut visited = BTreeSet::<UnresolvedRouteNode>::new();
    while let Some(candidate) = frontier.iter().next().cloned() {
        frontier.remove(&candidate);
        let (_, ranks, node, steps) = candidate;
        if !visited.insert(node.clone()) {
            continue;
        }
        if seeds
            .get(&node.owner)
            .is_some_and(|seed| summary_contains_route(seed, &node.route))
        {
            return Ok(SelectedUnresolvedRoute {
                root: root_for_error,
                terminal: node,
                steps,
            });
        }
        let Some(summary) = summaries.get(&node.owner) else {
            continue;
        };
        for edge in summary
            .unresolved_predecessors
            .iter()
            .filter(|edge| edge.caller_route == node.route)
        {
            if edge.call_point.function != node.owner {
                return Err(format!(
                    "unresolved edge owner mismatch: summary '{}', call point '{}'",
                    node.owner.0, edge.call_point.function.0
                ));
            }
            let step = UnresolvedRouteStep {
                caller: node.owner.clone(),
                edge: edge.clone(),
            };
            let next = UnresolvedRouteNode {
                owner: edge.callee.clone(),
                route: edge.callee_route.clone(),
            };
            let mut next_steps = steps.clone();
            next_steps.push(step);
            let mut next_ranks = ranks.clone();
            next_ranks.push(UnresolvedRouteRankStep::from(
                next_steps
                    .last()
                    .expect("an expanded route has one new step"),
            ));
            frontier.insert((next_steps.len(), next_ranks, next, next_steps));
        }
    }
    Err(format!(
        "no exact unresolved terminal for '{}' route {:?}",
        root_for_error.owner.0, root_for_error.route
    ))
}

/// Resolves every exact exported route before selecting the preferred witness.
///
/// An untraceable alternative is an analysis-consistency failure, even when a
/// different alternative reaches a seed. This prevents a traceable guarded
/// route from hiding an unresolved route whose predecessor chain is missing.
pub fn select_preferred_unresolved_route(
    roots: &BTreeSet<UnresolvedRouteNode>,
    summaries: &BTreeMap<FunctionKey, FunctionSummary>,
    seeds: &BTreeMap<FunctionKey, FunctionSummary>,
) -> Result<SelectedUnresolvedRoute, String> {
    type Ranked = (
        usize,
        Vec<UnresolvedRouteRankStep>,
        UnresolvedRouteNode,
        UnresolvedRouteNode,
        Vec<UnresolvedRouteStep>,
    );
    let mut candidates = BTreeSet::<Ranked>::new();
    for root in roots {
        let selected = select_unresolved_route(root.clone(), summaries, seeds)?;
        let ranks = selected
            .steps
            .iter()
            .map(UnresolvedRouteRankStep::from)
            .collect::<Vec<_>>();
        candidates.insert((
            selected.steps.len(),
            ranks,
            selected.root,
            selected.terminal,
            selected.steps,
        ));
    }
    let Some((_, _, root, terminal, steps)) = candidates.into_iter().next() else {
        return Err("no exact unresolved root route was exported".into());
    };
    Ok(SelectedUnresolvedRoute {
        root,
        terminal,
        steps,
    })
}

fn route_successors(
    node: &UnresolvedRouteNode,
    summaries: &BTreeMap<FunctionKey, FunctionSummary>,
) -> Result<Vec<(UnresolvedRouteNode, UnresolvedRouteStep)>, String> {
    let Some(summary) = summaries.get(&node.owner) else {
        return Ok(Vec::new());
    };
    summary
        .unresolved_predecessors
        .iter()
        .filter(|edge| edge.caller_route == node.route)
        .map(|edge| {
            if edge.call_point.function != node.owner {
                return Err(format!(
                    "unresolved edge owner mismatch: summary '{}', call point '{}'",
                    node.owner.0, edge.call_point.function.0
                ));
            }
            Ok((
                UnresolvedRouteNode {
                    owner: edge.callee.clone(),
                    route: edge.callee_route.clone(),
                },
                UnresolvedRouteStep {
                    caller: node.owner.clone(),
                    edge: edge.clone(),
                },
            ))
        })
        .collect()
}

fn reachable_route_nodes(
    start: &UnresolvedRouteNode,
    adjacency: &BTreeMap<UnresolvedRouteNode, Vec<(UnresolvedRouteNode, UnresolvedRouteStep)>>,
    allowed: &BTreeSet<UnresolvedRouteNode>,
) -> BTreeSet<UnresolvedRouteNode> {
    let mut reached = BTreeSet::new();
    let mut frontier = BTreeSet::from([start.clone()]);
    while let Some(node) = frontier.pop_first() {
        if !allowed.contains(&node) || !reached.insert(node.clone()) {
            continue;
        }
        for (successor, _) in adjacency.get(&node).into_iter().flatten() {
            if !reached.contains(successor) {
                frontier.insert(successor.clone());
            }
        }
    }
    reached
}

fn route_distance(
    starts: &BTreeSet<UnresolvedRouteNode>,
    target: &UnresolvedRouteNode,
    adjacency: &BTreeMap<UnresolvedRouteNode, Vec<(UnresolvedRouteNode, UnresolvedRouteStep)>>,
    allowed: &BTreeSet<UnresolvedRouteNode>,
) -> usize {
    let mut distance = 0;
    let mut frontier = starts.clone();
    let mut visited = BTreeSet::new();
    loop {
        if frontier.contains(target) {
            return distance;
        }
        let mut next = BTreeSet::new();
        for node in frontier {
            if !visited.insert(node.clone()) {
                continue;
            }
            for (successor, _) in adjacency.get(&node).into_iter().flatten() {
                if allowed.contains(successor) && !visited.contains(successor) {
                    next.insert(successor.clone());
                }
            }
        }
        assert!(
            !next.is_empty(),
            "route-cycle candidates are restricted to nodes that reach the terminal"
        );
        frontier = next;
        distance += 1;
    }
}

/// Finds a cycle only in the exact unresolved state graph for this witness.
///
/// Nodes must be reachable from the selected root and able to reach its exact
/// terminal. Thus unrelated call-graph recursion cannot mark a finding as
/// recursive. When several cyclic SCCs remain, the one nearest the terminal
/// wins, followed by stable exact-node ordering.
pub fn select_unresolved_route_cycle(
    selected: &SelectedUnresolvedRoute,
    summaries: &BTreeMap<FunctionKey, FunctionSummary>,
) -> Result<Option<UnresolvedRouteCycle>, String> {
    let mut forward = BTreeSet::from([selected.root.clone()]);
    let mut adjacency =
        BTreeMap::<UnresolvedRouteNode, Vec<(UnresolvedRouteNode, UnresolvedRouteStep)>>::new();
    let mut frontier = BTreeSet::from([selected.root.clone()]);
    while let Some(node) = frontier.pop_first() {
        let successors = route_successors(&node, summaries)?;
        for (successor, _) in &successors {
            if forward.insert(successor.clone()) {
                frontier.insert(successor.clone());
            }
        }
        adjacency.insert(node, successors);
    }
    if !forward.contains(&selected.terminal) {
        return Err("selected exact terminal is unreachable from its root".into());
    }

    let mut can_reach_terminal = BTreeSet::from([selected.terminal.clone()]);
    loop {
        let before = can_reach_terminal.len();
        for (caller, successors) in &adjacency {
            if successors
                .iter()
                .any(|(successor, _)| can_reach_terminal.contains(successor))
            {
                can_reach_terminal.insert(caller.clone());
            }
        }
        if can_reach_terminal.len() == before {
            break;
        }
    }
    let corridor = forward
        .intersection(&can_reach_terminal)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut remaining = corridor.clone();
    let selected_nodes = std::iter::once(selected.root.clone())
        .chain(selected.steps.iter().map(|step| UnresolvedRouteNode {
            owner: step.edge.callee.clone(),
            route: step.edge.callee_route.clone(),
        }))
        .collect::<BTreeSet<_>>();
    let mut cyclic = Vec::<BTreeSet<UnresolvedRouteNode>>::new();
    while let Some(root) = remaining.pop_first() {
        let from_root = reachable_route_nodes(&root, &adjacency, &corridor);
        let component = from_root
            .into_iter()
            .filter(|candidate| {
                reachable_route_nodes(candidate, &adjacency, &corridor).contains(&root)
            })
            .collect::<BTreeSet<_>>();
        for member in &component {
            remaining.remove(member);
        }
        let self_edge = adjacency
            .get(&root)
            .is_some_and(|successors| successors.iter().any(|(successor, _)| successor == &root));
        if (component.len() > 1 || self_edge) && !component.is_disjoint(&selected_nodes) {
            cyclic.push(component);
        }
    }
    let Some(component) = cyclic.into_iter().min_by_key(|component| {
        (
            route_distance(component, &selected.terminal, &adjacency, &corridor),
            component.iter().cloned().collect::<Vec<_>>(),
        )
    }) else {
        return Ok(None);
    };
    let (_, cycle_edge) = component
        .iter()
        .flat_map(|caller| {
            adjacency
                .get(caller)
                .into_iter()
                .flatten()
                .filter(|(callee, _)| component.contains(callee))
                .map(move |(_, step)| {
                    (
                        (
                            step.caller.clone(),
                            step.edge.callee.clone(),
                            step.edge.call_point.span.clone(),
                            step.edge.call_point.block,
                            step.edge.call_point.statement,
                            step.edge.caller_route.clone(),
                            step.edge.callee_route.clone(),
                        ),
                        step,
                    )
                })
        })
        .min_by_key(|(rank, _)| rank.clone())
        .expect("a cyclic SCC contains a real internal edge");
    let members = component
        .iter()
        .map(|node| node.owner.clone())
        .collect::<BTreeSet<_>>();
    Ok(Some(UnresolvedRouteCycle {
        nodes: component,
        token: scc_cycle_token(&members.iter().cloned().collect::<Vec<_>>()),
        span: cycle_edge.edge.call_point.span.clone(),
    }))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionSummary {
    pub function: FunctionKey,
    pub sources: BTreeSet<Source>,
    pub flows: BTreeSet<FlowFact>,
    pub return_origins: BTreeSet<OriginKey>,
    pub out_dependencies: BTreeSet<OutDependency>,
    pub sink_obligations: BTreeSet<SinkObligation>,
    pub validations: BTreeSet<ValidationFact>,
    pub writes: BTreeSet<WriteEffect>,
    pub calls: BTreeSet<CallBoundary>,
    pub cycle_tokens: BTreeSet<String>,
    pub requirements: BTreeSet<ContractRequirement>,
    pub unresolved_predecessors: BTreeSet<UnresolvedPredecessor>,
    pub return_value: AbstractValue,
}

impl FunctionSummary {
    pub fn empty(function: FunctionKey) -> Self {
        Self {
            function,
            sources: BTreeSet::new(),
            flows: BTreeSet::new(),
            return_origins: BTreeSet::new(),
            out_dependencies: BTreeSet::new(),
            sink_obligations: BTreeSet::new(),
            validations: BTreeSet::new(),
            writes: BTreeSet::new(),
            calls: BTreeSet::new(),
            cycle_tokens: BTreeSet::new(),
            requirements: BTreeSet::new(),
            unresolved_predecessors: BTreeSet::new(),
            return_value: AbstractValue::default(),
        }
    }

    pub fn join(&mut self, other: &Self) -> bool {
        assert_eq!(
            self.function, other.function,
            "summaries for different functions cannot be joined"
        );
        let before = (
            self.sources.len(),
            self.flows.len(),
            self.return_origins.len(),
            self.out_dependencies.len(),
            self.sink_obligations.len(),
            self.validations.len(),
            self.writes.len(),
            self.calls.len(),
            self.cycle_tokens.len(),
            self.requirements.len(),
            self.unresolved_predecessors.len(),
            self.return_value.origins.len(),
        );
        self.sources.extend(other.sources.iter().cloned());
        self.flows.extend(other.flows.iter().cloned());
        self.return_origins
            .extend(other.return_origins.iter().cloned());
        self.out_dependencies
            .extend(other.out_dependencies.iter().cloned());
        self.sink_obligations
            .extend(other.sink_obligations.iter().cloned());
        self.validations.extend(other.validations.iter().cloned());
        self.writes.extend(other.writes.iter().cloned());
        self.calls.extend(other.calls.iter().cloned());
        self.cycle_tokens.extend(other.cycle_tokens.iter().cloned());
        self.requirements.extend(other.requirements.iter().cloned());
        self.unresolved_predecessors
            .extend(other.unresolved_predecessors.iter().cloned());
        self.return_value
            .origins
            .extend(other.return_value.origins.iter().cloned());
        before
            != (
                self.sources.len(),
                self.flows.len(),
                self.return_origins.len(),
                self.out_dependencies.len(),
                self.sink_obligations.len(),
                self.validations.len(),
                self.writes.len(),
                self.calls.len(),
                self.cycle_tokens.len(),
                self.requirements.len(),
                self.unresolved_predecessors.len(),
                self.return_value.origins.len(),
            )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pattern {
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
}

impl Pattern {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::P1 => "pattern1",
            Self::P2 => "pattern2",
            Self::P3 => "pattern3",
            Self::P4 => "pattern4",
            Self::P5 => "pattern5",
            Self::P6 => "pattern6",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailureClass {
    SinkPrecondition,
    InternalInvalidValue,
}

pub fn classify_primary(failure: FailureClass, source: SourceKind) -> Pattern {
    if failure == FailureClass::InternalInvalidValue {
        return Pattern::P3;
    }
    match source {
        SourceKind::FfiOutput | SourceKind::OpenBehaviorOutput => Pattern::P6,
        SourceKind::GenericNonEmptyCapability => Pattern::P5,
        SourceKind::LiteralPublicField => Pattern::P2,
        SourceKind::PublicParameter => Pattern::P1,
        SourceKind::InternalDerived => Pattern::P4,
        SourceKind::InternalUnsafeOrigin => Pattern::P3,
    }
}

pub fn classify_primary_with_secondary(
    failure: FailureClass,
    selected_source: SourceKind,
    all_sources: impl IntoIterator<Item = SourceKind>,
) -> (Pattern, BTreeSet<SourceKind>) {
    let mut secondary_sources = all_sources.into_iter().collect::<BTreeSet<_>>();
    secondary_sources.remove(&selected_source);
    (
        classify_primary(failure, selected_source),
        secondary_sources,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuleId {
    P1RawRead,
    P1GetUnchecked,
    P2RawRead,
    P2GetUnchecked,
    P3LifetimeTransmute,
    P3AssumeInitBool,
    P3UncheckedUtf8,
    P4Bounds,
    P4Offset,
    P51Nonempty,
    P6FfiOutParam,
    P6OpenTraitIndex,
}

impl RuleId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::P1RawRead => "P1.raw_read",
            Self::P1GetUnchecked => "P1.get_unchecked",
            Self::P2RawRead => "P2.raw_read",
            Self::P2GetUnchecked => "P2.get_unchecked",
            Self::P3LifetimeTransmute => "P3.lifetime_transmute",
            Self::P3AssumeInitBool => "P3.assume_init_bool",
            Self::P3UncheckedUtf8 => "P3.unchecked_utf8",
            Self::P4Bounds => "P4.bounds",
            Self::P4Offset => "P4.offset",
            Self::P51Nonempty => "P5.1.nonempty",
            Self::P6FfiOutParam => "P6.ffi_out_param",
            Self::P6OpenTraitIndex => "P6.open_trait_index",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WitnessStepKind {
    Entry,
    Source,
    LocalCall,
    SccCycle,
    Sink,
    Exposure,
}

impl WitnessStepKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entry => "entry",
            Self::Source => "source",
            Self::LocalCall => "local_call",
            Self::SccCycle => "scc_cycle",
            Self::Sink => "sink",
            Self::Exposure => "exposure",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WitnessStep {
    pub kind: WitnessStepKind,
    pub function: FunctionKey,
    pub span: StableSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropagationDepth {
    IntraProcedural,
    InterProcedural,
    Recursive,
}

impl PropagationDepth {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IntraProcedural => "intra_procedural",
            Self::InterProcedural => "inter_procedural",
            Self::Recursive => "recursive",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalWitness {
    pub depth: PropagationDepth,
    pub local_call_count: Option<u32>,
    pub boundaries: Vec<FunctionKey>,
    pub steps: Vec<WitnessStep>,
}

impl CanonicalWitness {
    pub fn new(mut steps: Vec<WitnessStep>, mut boundaries: Vec<FunctionKey>) -> Self {
        assert!(
            steps.len() >= 2,
            "a witness must contain an entry/source and sink"
        );
        boundaries.dedup();
        let recursive = steps
            .iter()
            .any(|step| step.kind == WitnessStepKind::SccCycle);
        let call_count = steps
            .iter()
            .filter(|step| step.kind == WitnessStepKind::LocalCall)
            .count() as u32;
        let (depth, local_call_count) = if recursive {
            (PropagationDepth::Recursive, None)
        } else if call_count == 0 {
            (PropagationDepth::IntraProcedural, Some(0))
        } else {
            (PropagationDepth::InterProcedural, Some(call_count))
        };
        if recursive {
            let mut retained_cycle = false;
            steps.retain(|step| {
                if step.kind != WitnessStepKind::SccCycle {
                    true
                } else if retained_cycle {
                    false
                } else {
                    retained_cycle = true;
                    true
                }
            });
        }
        Self {
            depth,
            local_call_count,
            boundaries,
            steps,
        }
    }

    fn rank(&self) -> (usize, &Vec<WitnessStep>, &Vec<FunctionKey>) {
        (self.steps.len(), &self.steps, &self.boundaries)
    }

    pub fn preferred_over(&self, other: &Self) -> bool {
        self.rank() < other.rank()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CausalKey {
    pub crate_key: String,
    pub public_root: FunctionKey,
    pub source_origin: String,
    pub first_contract_failure: String,
    pub sink_or_exposure: String,
    pub canonical_obligation: String,
}

impl CausalKey {
    pub fn from_parts(
        crate_key: impl Into<String>,
        public_root: FunctionKey,
        source: &Source,
        first_failure: &Operation,
        sink: &Operation,
        obligation: &Obligation,
    ) -> Self {
        Self {
            crate_key: crate_key.into(),
            public_root,
            source_origin: source.canonical_token(),
            first_contract_failure: first_failure.canonical_token(),
            sink_or_exposure: sink.canonical_token(),
            canonical_obligation: obligation.canonical_token(),
        }
    }

    pub fn canonical_json(&self) -> String {
        let string = |value: &str| serde_json::to_string(value).expect("strings serialize");
        format!(
            "{{\"canonical_obligation\":{},\"crate\":{},\"first_contract_failure\":{},\"public_root\":{},\"sink_or_exposure\":{},\"source_origin\":{}}}",
            string(&self.canonical_obligation),
            string(&self.crate_key),
            string(&self.first_contract_failure),
            string(&self.public_root.0),
            string(&self.sink_or_exposure),
            string(&self.source_origin),
        )
    }

    pub fn finding_id(&self) -> String {
        sha256_hex(self.canonical_json().as_bytes())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub causal_key: CausalKey,
    pub public_root_span: StableSpan,
    pub sink_obligation: SinkObligation,
    pub primary: Pattern,
    pub rule: RuleId,
    pub witness: CanonicalWitness,
    pub secondary_sources: BTreeSet<SourceKind>,
}

impl Finding {
    pub fn new(
        crate_key: impl Into<String>,
        public_root: FunctionKey,
        public_root_span: StableSpan,
        sink_obligation: SinkObligation,
        primary: Pattern,
        rule: RuleId,
        witness: CanonicalWitness,
        secondary_sources: BTreeSet<SourceKind>,
    ) -> Self {
        let causal_key = CausalKey::from_parts(
            crate_key,
            public_root,
            &sink_obligation.source,
            &sink_obligation.first_failure,
            &sink_obligation.sink,
            &sink_obligation.obligation,
        );
        Self {
            causal_key,
            public_root_span,
            sink_obligation,
            primary,
            rule,
            witness,
            secondary_sources,
        }
    }
}

pub fn deduplicate_findings(findings: impl IntoIterator<Item = Finding>) -> Vec<Finding> {
    let mut canonical = BTreeMap::<CausalKey, Finding>::new();
    for finding in findings {
        canonical
            .entry(finding.causal_key.clone())
            .and_modify(|current| {
                current
                    .secondary_sources
                    .extend(finding.secondary_sources.iter().copied());
                if finding.witness.preferred_over(&current.witness) {
                    current.witness = finding.witness.clone();
                }
            })
            .or_insert(finding);
    }
    canonical.into_values().collect()
}

pub fn scc_cycle_token(component: &[FunctionKey]) -> String {
    let mut members = component.to_vec();
    members.sort();
    members.dedup();
    format!(
        "scc:[{}]",
        members
            .iter()
            .map(|member| member.0.as_str())
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub fn strongly_connected_components(
    graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
) -> Vec<Vec<FunctionKey>> {
    struct Tarjan<'a> {
        graph: &'a BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
        next_index: usize,
        indices: BTreeMap<FunctionKey, usize>,
        lowlinks: BTreeMap<FunctionKey, usize>,
        stack: Vec<FunctionKey>,
        on_stack: BTreeSet<FunctionKey>,
        components: Vec<Vec<FunctionKey>>,
    }

    impl Tarjan<'_> {
        fn visit(&mut self, node: FunctionKey) {
            let index = self.next_index;
            self.next_index += 1;
            self.indices.insert(node.clone(), index);
            self.lowlinks.insert(node.clone(), index);
            self.stack.push(node.clone());
            self.on_stack.insert(node.clone());

            for successor in self.graph.get(&node).into_iter().flatten() {
                if !self.indices.contains_key(successor) {
                    self.visit(successor.clone());
                    let successor_lowlink = self.lowlinks[successor];
                    let node_lowlink = self.lowlinks[&node].min(successor_lowlink);
                    self.lowlinks.insert(node.clone(), node_lowlink);
                } else if self.on_stack.contains(successor) {
                    let successor_index = self.indices[successor];
                    let node_lowlink = self.lowlinks[&node].min(successor_index);
                    self.lowlinks.insert(node.clone(), node_lowlink);
                }
            }

            if self.lowlinks[&node] == self.indices[&node] {
                let mut component = Vec::new();
                loop {
                    let member = self.stack.pop().expect("Tarjan stack contains root");
                    self.on_stack.remove(&member);
                    component.push(member.clone());
                    if member == node {
                        break;
                    }
                }
                component.sort();
                self.components.push(component);
            }
        }
    }

    let mut complete = graph.clone();
    for successor in graph.values().flatten() {
        complete.entry(successor.clone()).or_default();
    }
    let mut tarjan = Tarjan {
        graph: &complete,
        next_index: 0,
        indices: BTreeMap::new(),
        lowlinks: BTreeMap::new(),
        stack: Vec::new(),
        on_stack: BTreeSet::new(),
        components: Vec::new(),
    };
    for node in complete.keys() {
        if !tarjan.indices.contains_key(node) {
            tarjan.visit(node.clone());
        }
    }
    tarjan
        .components
        .sort_by(|left, right| left[0].cmp(&right[0]));
    tarjan.components
}

fn callee_first_components(
    graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
    components: &[Vec<FunctionKey>],
) -> Vec<usize> {
    let mut component_by_node = BTreeMap::new();
    for (index, component) in components.iter().enumerate() {
        for node in component {
            component_by_node.insert(node.clone(), index);
        }
    }
    let mut dependencies = vec![BTreeSet::new(); components.len()];
    for (caller, callees) in graph {
        let Some(&caller_component) = component_by_node.get(caller) else {
            continue;
        };
        for callee in callees {
            let Some(&callee_component) = component_by_node.get(callee) else {
                continue;
            };
            if caller_component != callee_component {
                dependencies[caller_component].insert(callee_component);
            }
        }
    }
    fn visit(
        component: usize,
        dependencies: &[BTreeSet<usize>],
        visited: &mut BTreeSet<usize>,
        order: &mut Vec<usize>,
    ) {
        if !visited.insert(component) {
            return;
        }
        for dependency in &dependencies[component] {
            visit(*dependency, dependencies, visited, order);
        }
        order.push(component);
    }
    let mut component_indices = (0..components.len()).collect::<Vec<_>>();
    component_indices.sort_by(|left, right| components[*left][0].cmp(&components[*right][0]));
    let mut visited = BTreeSet::new();
    let mut order = Vec::new();
    for component in component_indices {
        visit(component, &dependencies, &mut visited, &mut order);
    }
    order
}

pub fn solve_summaries<F>(
    graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
    seeds: &BTreeMap<FunctionKey, FunctionSummary>,
    transfer: F,
) -> BTreeMap<FunctionKey, FunctionSummary>
where
    F: FnMut(&FunctionKey, &BTreeMap<FunctionKey, FunctionSummary>) -> FunctionSummary,
{
    solve_summaries_internal(graph, seeds, transfer, None)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SummarySolveStats {
    component_rounds: usize,
    summary_candidate_clones: usize,
}

#[cfg(test)]
fn solve_summaries_with_stats<F>(
    graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
    seeds: &BTreeMap<FunctionKey, FunctionSummary>,
    transfer: F,
) -> (BTreeMap<FunctionKey, FunctionSummary>, SummarySolveStats)
where
    F: FnMut(&FunctionKey, &BTreeMap<FunctionKey, FunctionSummary>) -> FunctionSummary,
{
    let mut stats = SummarySolveStats::default();
    let solved = solve_summaries_internal(graph, seeds, transfer, Some(&mut stats));
    (solved, stats)
}

fn solve_summaries_internal<F>(
    graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
    seeds: &BTreeMap<FunctionKey, FunctionSummary>,
    mut transfer: F,
    #[cfg(test)] mut stats: Option<&mut SummarySolveStats>,
    #[cfg(not(test))] _stats: Option<&mut ()>,
) -> BTreeMap<FunctionKey, FunctionSummary>
where
    F: FnMut(&FunctionKey, &BTreeMap<FunctionKey, FunctionSummary>) -> FunctionSummary,
{
    let mut complete_graph = graph.clone();
    for node in seeds.keys() {
        complete_graph.entry(node.clone()).or_default();
    }
    for callee in graph.values().flatten() {
        complete_graph.entry(callee.clone()).or_default();
    }
    let components = strongly_connected_components(&complete_graph);
    let order = callee_first_components(&complete_graph, &components);
    let mut solved = complete_graph
        .keys()
        .map(|node| {
            let summary = seeds
                .get(node)
                .cloned()
                .unwrap_or_else(|| FunctionSummary::empty(node.clone()));
            assert_eq!(
                summary.function, *node,
                "seed summary key must match function"
            );
            (node.clone(), summary)
        })
        .collect::<BTreeMap<_, _>>();
    for component_index in order {
        let component = &components[component_index];
        let recursive = component.len() > 1
            || component
                .first()
                .and_then(|node| complete_graph.get(node).map(|edges| edges.contains(node)))
                .unwrap_or(false);
        let cycle_token = recursive.then(|| scc_cycle_token(component));
        loop {
            #[cfg(test)]
            if let Some(stats) = stats.as_deref_mut() {
                stats.component_rounds += 1;
            }
            let mut updates = Vec::new();
            for node in component {
                let mut next = solved[node].clone();
                #[cfg(test)]
                if let Some(stats) = stats.as_deref_mut() {
                    stats.summary_candidate_clones += 1;
                }
                let derived = transfer(node, &solved);
                assert_eq!(
                    derived.function, *node,
                    "transfer summary key must match function"
                );
                let mut changed = next.join(&derived);
                if let Some(token) = &cycle_token {
                    changed |= next.cycle_tokens.insert(token.clone());
                }
                if changed {
                    updates.push((node.clone(), next));
                }
            }
            if updates.is_empty() {
                break;
            }
            for (node, next) in updates {
                solved.insert(node, next);
            }
        }
    }
    solved
}

fn sha256_hex(input: &[u8]) -> String {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let bit_length = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());
    let mut state = INITIAL;
    for block in padded.chunks_exact(64) {
        let mut schedule = [0_u32; 64];
        for (index, word) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(ROUND[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(line: u32) -> StableSpan {
        StableSpan::new(
            "src/lib.rs",
            StablePosition::new(line, 1),
            StablePosition::new(line, 5),
        )
    }

    fn point(function: &str, block: u32, statement: u32) -> ProgramPoint {
        ProgramPoint {
            function: FunctionKey::new(function),
            block,
            statement,
            span: span(block + 1),
        }
    }

    fn populated_summary(tag: &str, ordinal: u32) -> FunctionSummary {
        let function = FunctionKey::new("crate::entry");
        let source = Source {
            kind: SourceKind::PublicParameter,
            origin: OriginKey::new(tag),
            span: span(ordinal),
        };
        let operation = Operation {
            kind: OperationKind::RawRead,
            point: point("crate::entry", ordinal, 0),
        };
        let obligation = Obligation::new([Predicate::ValidForRead], OriginKey::new(tag));
        let requirement = ContractRequirement {
            seed_id: tag.to_owned(),
            collection: None,
            subject: AbstractValue::new([AbstractOrigin::Formal(ordinal)]),
            source_hint: Some(SourceKind::PublicParameter),
            internal_derivation: false,
            source_span: span(ordinal),
            first_failure: operation.clone(),
            sink: operation.clone(),
            predicates: BTreeSet::from([Predicate::NonNull]),
            access_width: None,
            rule: RuleId::P1RawRead,
            return_exposure: false,
            sink_function: function.clone(),
        };
        let mut summary = FunctionSummary::empty(function.clone());
        summary.sources.insert(source.clone());
        summary.flows.insert(FlowFact {
            from: PlaceKey::new(tag),
            to: PlaceKey::new("_0"),
            kind: FlowKind::Copy,
            point: operation.point.clone(),
        });
        summary.return_origins.insert(OriginKey::new(tag));
        summary.out_dependencies.insert(OutDependency {
            formal_index: ordinal,
            value: AbstractValue::new([AbstractOrigin::Formal(ordinal)]),
            may_skip_write: false,
        });
        summary.sink_obligations.insert(SinkObligation {
            source,
            first_failure: operation.clone(),
            sink: operation.clone(),
            obligation,
        });
        let validation = ValidationFact {
            requirement: requirement.clone(),
            subject: BoundarySlot::Formal(ordinal),
            collection: None,
        };
        summary.validations.insert(validation.clone());
        summary.writes.insert(WriteEffect {
            place: PlaceKey::new(tag),
            point: operation.point.clone(),
        });
        summary.calls.insert(CallBoundary {
            caller: function.clone(),
            callee: FunctionKey::new(tag),
            point: operation.point.clone(),
            mapping: BTreeSet::from([CallMapping::ReturnToDestination {
                destination: PlaceKey::new(tag),
            }]),
        });
        summary.cycle_tokens.insert(tag.to_owned());
        summary
            .unresolved_predecessors
            .insert(UnresolvedPredecessor {
                caller_route: UnresolvedRouteFact::Hard(requirement.clone()),
                callee_route: UnresolvedRouteFact::Conditional(validation),
                callee: FunctionKey::new(tag),
                call_point: operation.point.clone(),
            });
        summary.requirements.insert(requirement);
        summary
            .return_value
            .origins
            .insert(AbstractOrigin::Constant(ordinal as u64));
        summary
    }

    #[test]
    fn summary_join_is_monotone_idempotent_and_stably_sorted() {
        let mut left = populated_summary("z", 2);
        let right = populated_summary("a", 1);
        let before = left.clone();
        assert!(left.join(&right));
        assert!(before.sources.is_subset(&left.sources));
        assert!(before.flows.is_subset(&left.flows));
        assert!(before.return_origins.is_subset(&left.return_origins));
        assert!(before.out_dependencies.is_subset(&left.out_dependencies));
        assert!(before.sink_obligations.is_subset(&left.sink_obligations));
        assert!(before.validations.is_subset(&left.validations));
        assert!(before.writes.is_subset(&left.writes));
        assert!(before.calls.is_subset(&left.calls));
        assert!(before.cycle_tokens.is_subset(&left.cycle_tokens));
        assert!(before.requirements.is_subset(&left.requirements));
        assert!(before
            .unresolved_predecessors
            .is_subset(&left.unresolved_predecessors));
        assert!(before
            .return_value
            .origins
            .is_subset(&left.return_value.origins));
        let joined = left.clone();
        assert!(!left.join(&right));
        assert_eq!(left, joined);
        assert_eq!(
            left.sources
                .iter()
                .map(|fact| fact.origin.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.flows
                .iter()
                .map(|fact| fact.from.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.return_origins
                .iter()
                .map(|origin| origin.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.out_dependencies
                .iter()
                .flat_map(|fact| fact.value.origins.iter())
                .filter_map(|origin| match origin {
                    AbstractOrigin::Formal(index) => Some(*index),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            left.sink_obligations
                .iter()
                .map(|fact| fact.source.origin.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.validations
                .iter()
                .map(|fact| fact.requirement.seed_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.writes
                .iter()
                .map(|fact| fact.place.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.calls
                .iter()
                .map(|fact| fact.callee.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.cycle_tokens
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.requirements
                .iter()
                .map(|fact| fact.seed_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.unresolved_predecessors
                .iter()
                .map(|fact| fact.callee.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        assert_eq!(
            left.return_value.origins.iter().collect::<Vec<_>>(),
            vec![&AbstractOrigin::Constant(1), &AbstractOrigin::Constant(2)]
        );
    }

    fn bounds_requirement(tag: &str) -> ContractRequirement {
        let sink = Operation {
            kind: OperationKind::GetUnchecked,
            point: point("crate::sink", 4, 0),
        };
        ContractRequirement {
            seed_id: tag.to_owned(),
            collection: Some(AbstractValue::new([AbstractOrigin::Formal(1)])),
            subject: AbstractValue::new([AbstractOrigin::Formal(2)]),
            source_hint: None,
            internal_derivation: false,
            source_span: span(1),
            first_failure: sink.clone(),
            sink,
            predicates: BTreeSet::from([Predicate::InBounds]),
            access_width: None,
            rule: RuleId::P1GetUnchecked,
            return_exposure: false,
            sink_function: FunctionKey::new("crate::sink"),
        }
    }

    #[test]
    fn task8_hard_and_conditional_routes_keep_distinct_terminal_identity() {
        let requirement = bounds_requirement("same-terminal");
        let validation = ValidationFact::export_entry_contract(
            requirement.clone(),
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Formal(1)),
            true,
            true,
        )
        .unwrap();
        let hard = UnresolvedRouteFact::Hard(requirement.clone());
        let conditional = UnresolvedRouteFact::Conditional(validation);

        assert_eq!(hard.requirement(), &requirement);
        assert_eq!(conditional.requirement(), &requirement);
        assert_ne!(hard, conditional);
        assert_eq!(BTreeSet::from([hard, conditional]).len(), 2);
    }

    #[test]
    fn task8_unresolved_predecessors_join_by_finite_stable_union() {
        let function = FunctionKey::new("crate::caller");
        let requirement = bounds_requirement("route");
        let hard = UnresolvedRouteFact::Hard(requirement.clone());
        let conditional = UnresolvedRouteFact::Conditional(
            ValidationFact::export_entry_contract(
                requirement,
                Some(BoundarySlot::Formal(2)),
                Some(BoundarySlot::Formal(1)),
                true,
                true,
            )
            .unwrap(),
        );
        let earlier = UnresolvedPredecessor {
            caller_route: hard.clone(),
            callee_route: hard,
            callee: FunctionKey::new("crate::a"),
            call_point: point("crate::caller", 1, 0),
        };
        let later = UnresolvedPredecessor {
            caller_route: conditional.clone(),
            callee_route: conditional,
            callee: FunctionKey::new("crate::z"),
            call_point: point("crate::caller", 2, 0),
        };
        let mut left = FunctionSummary::empty(function.clone());
        left.unresolved_predecessors.insert(later.clone());
        let mut right = FunctionSummary::empty(function);
        right.unresolved_predecessors.insert(earlier.clone());

        let before = left.clone();
        assert!(left.join(&right));
        assert!(before
            .unresolved_predecessors
            .is_subset(&left.unresolved_predecessors));
        assert_eq!(
            left.unresolved_predecessors.iter().collect::<Vec<_>>(),
            vec![&earlier, &later]
        );
        let joined = left.clone();
        assert!(!left.join(&right));
        assert_eq!(left, joined);
    }

    fn task8_edge(
        caller: &str,
        caller_route: UnresolvedRouteFact,
        callee_route: UnresolvedRouteFact,
        callee: &str,
        line: u32,
    ) -> UnresolvedPredecessor {
        UnresolvedPredecessor {
            caller_route,
            callee_route,
            callee: FunctionKey::new(callee),
            call_point: point(caller, line, 0),
        }
    }

    fn task8_seed(owner: &str, route: UnresolvedRouteFact) -> FunctionSummary {
        let mut seed = FunctionSummary::empty(FunctionKey::new(owner));
        match route {
            UnresolvedRouteFact::Hard(requirement) => {
                seed.requirements.insert(requirement);
            }
            UnresolvedRouteFact::Conditional(validation) => {
                seed.validations.insert(validation);
            }
        }
        seed
    }

    #[test]
    fn task8_selector_ignores_a_short_satisfied_route_and_uses_the_long_unresolved_route() {
        let root = FunctionKey::new("crate::root");
        let middle = FunctionKey::new("crate::middle");
        let sink = FunctionKey::new("crate::sink");
        let root_route = UnresolvedRouteFact::Hard(bounds_requirement("root"));
        let middle_route = UnresolvedRouteFact::Hard(bounds_requirement("middle"));
        let sink_route = UnresolvedRouteFact::Hard(bounds_requirement("sink"));
        let first = task8_edge(
            &root.0,
            root_route.clone(),
            middle_route.clone(),
            &middle.0,
            9,
        );
        let second = task8_edge(
            &middle.0,
            middle_route.clone(),
            sink_route.clone(),
            &sink.0,
            10,
        );
        let mut root_summary = FunctionSummary::empty(root.clone());
        root_summary.unresolved_predecessors.insert(first);
        let mut middle_summary = FunctionSummary::empty(middle.clone());
        middle_summary.unresolved_predecessors.insert(second);
        // The lexical-short guarded call intentionally has no unresolved edge.
        let summaries = BTreeMap::from([
            (root.clone(), root_summary),
            (middle.clone(), middle_summary),
            (sink.clone(), FunctionSummary::empty(sink.clone())),
        ]);
        let seeds = BTreeMap::from([(sink.clone(), task8_seed(&sink.0, sink_route))]);

        let selected = select_unresolved_route(
            UnresolvedRouteNode {
                owner: root,
                route: root_route,
            },
            &summaries,
            &seeds,
        )
        .unwrap();
        assert_eq!(selected.steps.len(), 2);
        assert_eq!(
            selected
                .steps
                .iter()
                .map(|step| step.edge.callee.0.as_str())
                .collect::<Vec<_>>(),
            vec!["crate::middle", "crate::sink"]
        );
    }

    #[test]
    fn task8_selector_never_crosses_route_variant_or_subject_identity() {
        let root = FunctionKey::new("crate::root");
        let sink = FunctionKey::new("crate::sink");
        let requirement = bounds_requirement("same");
        let hard = UnresolvedRouteFact::Hard(requirement.clone());
        let conditional = UnresolvedRouteFact::Conditional(
            ValidationFact::export_entry_contract(
                requirement.clone(),
                Some(BoundarySlot::Formal(2)),
                Some(BoundarySlot::Formal(1)),
                true,
                true,
            )
            .unwrap(),
        );
        let mut other_requirement = requirement;
        other_requirement.subject = AbstractValue::new([AbstractOrigin::Formal(3)]);
        let other_subject = UnresolvedRouteFact::Hard(other_requirement.clone());
        let exact_terminal = UnresolvedRouteFact::Hard(bounds_requirement("exact-terminal"));
        let mut summary = FunctionSummary::empty(root.clone());
        summary.unresolved_predecessors.extend([
            task8_edge(
                &root.0,
                conditional.clone(),
                conditional.clone(),
                &sink.0,
                1,
            ),
            task8_edge(
                &root.0,
                other_subject.clone(),
                other_subject.clone(),
                &sink.0,
                2,
            ),
            task8_edge(&root.0, hard.clone(), exact_terminal.clone(), &sink.0, 3),
        ]);
        let summaries = BTreeMap::from([
            (root.clone(), summary),
            (sink.clone(), FunctionSummary::empty(sink.clone())),
        ]);
        let seeds = BTreeMap::from([(sink.clone(), {
            let mut seed = task8_seed(&sink.0, conditional);
            seed.requirements.insert(other_requirement);
            seed.requirements
                .insert(exact_terminal.requirement().clone());
            seed
        })]);

        let selected = select_unresolved_route(
            UnresolvedRouteNode {
                owner: root,
                route: hard,
            },
            &summaries,
            &seeds,
        )
        .unwrap();
        assert_eq!(selected.steps.len(), 1);
        assert_eq!(selected.terminal.route, exact_terminal);
    }

    #[test]
    fn task8_selector_fails_closed_when_no_exact_seed_terminal_exists() {
        let root = FunctionKey::new("crate::root");
        let route = UnresolvedRouteFact::Hard(bounds_requirement("missing"));
        assert!(select_unresolved_route(
            UnresolvedRouteNode {
                owner: root.clone(),
                route,
            },
            &BTreeMap::from([(root.clone(), FunctionSummary::empty(root))]),
            &BTreeMap::new(),
        )
        .is_err());
    }

    #[test]
    fn task8_selector_rejects_a_predecessor_not_owned_by_its_summary() {
        let root = FunctionKey::new("crate::root");
        let sink = FunctionKey::new("crate::sink");
        let route = UnresolvedRouteFact::Hard(bounds_requirement("owner"));
        let mut summary = FunctionSummary::empty(root.clone());
        summary.unresolved_predecessors.insert(task8_edge(
            "crate::wrong",
            route.clone(),
            route.clone(),
            &sink.0,
            1,
        ));
        let error = select_unresolved_route(
            UnresolvedRouteNode {
                owner: root.clone(),
                route: route.clone(),
            },
            &BTreeMap::from([(root, summary)]),
            &BTreeMap::from([(sink.clone(), task8_seed(&sink.0, route))]),
        )
        .unwrap_err();
        assert!(error.contains("owner mismatch"));
    }

    #[test]
    fn task8_selector_terminates_on_direct_and_mutual_route_cycles() {
        let a = FunctionKey::new("crate::a");
        let b = FunctionKey::new("crate::b");
        let start = UnresolvedRouteFact::Hard(bounds_requirement("start"));
        let through_b = UnresolvedRouteFact::Hard(bounds_requirement("through-b"));
        let terminal = UnresolvedRouteFact::Hard(bounds_requirement("terminal"));
        let mut a_summary = FunctionSummary::empty(a.clone());
        a_summary.unresolved_predecessors.extend([
            task8_edge(&a.0, start.clone(), start.clone(), &a.0, 1),
            task8_edge(&a.0, start.clone(), through_b.clone(), &b.0, 2),
        ]);
        let mut b_summary = FunctionSummary::empty(b.clone());
        b_summary.unresolved_predecessors.insert(task8_edge(
            &b.0,
            through_b,
            terminal.clone(),
            &a.0,
            3,
        ));
        let summaries = BTreeMap::from([(a.clone(), a_summary), (b.clone(), b_summary)]);
        let seeds = BTreeMap::from([(a.clone(), task8_seed(&a.0, terminal.clone()))]);

        let selected = select_unresolved_route(
            UnresolvedRouteNode {
                owner: a,
                route: start,
            },
            &summaries,
            &seeds,
        )
        .unwrap();
        assert_eq!(selected.steps.len(), 2);
        assert_eq!(selected.terminal.route, terminal);
    }

    #[test]
    fn task8_selector_equal_length_choice_is_lexical_and_insertion_order_independent() {
        let root = FunctionKey::new("crate::root");
        let a = FunctionKey::new("crate::a");
        let z = FunctionKey::new("crate::z");
        let start = UnresolvedRouteFact::Hard(bounds_requirement("start-order"));
        // Deliberately oppose semantic route ordering and call-edge ordering:
        // the `a` callee must win even though its terminal route sorts later.
        let terminal_a = UnresolvedRouteFact::Hard(bounds_requirement("z-route"));
        let terminal_z = UnresolvedRouteFact::Hard(bounds_requirement("a-route"));
        let to_a = task8_edge(&root.0, start.clone(), terminal_a.clone(), &a.0, 9);
        let to_z = task8_edge(&root.0, start.clone(), terminal_z.clone(), &z.0, 1);
        let seeds = BTreeMap::from([
            (a.clone(), task8_seed(&a.0, terminal_a)),
            (z.clone(), task8_seed(&z.0, terminal_z)),
        ]);
        let select = |edges: [UnresolvedPredecessor; 2]| {
            let mut summary = FunctionSummary::empty(root.clone());
            summary.unresolved_predecessors.extend(edges);
            select_unresolved_route(
                UnresolvedRouteNode {
                    owner: root.clone(),
                    route: start.clone(),
                },
                &BTreeMap::from([(root.clone(), summary)]),
                &seeds,
            )
            .unwrap()
        };

        let forward = select([to_z.clone(), to_a.clone()]);
        let reverse = select([to_a, to_z]);
        assert_eq!(forward, reverse);
        assert_eq!(forward.steps[0].edge.callee, a);
    }

    #[test]
    fn task8_preferred_selector_requires_every_exact_start_to_reach_a_seed() {
        let root = FunctionKey::new("crate::root");
        let middle = FunctionKey::new("crate::middle");
        let sink = FunctionKey::new("crate::sink");
        let requirement = bounds_requirement("shared-root-requirement");
        let hard = UnresolvedRouteFact::Hard(requirement.clone());
        let conditional = UnresolvedRouteFact::Conditional(
            ValidationFact::export_entry_contract(
                requirement,
                Some(BoundarySlot::Formal(2)),
                Some(BoundarySlot::Formal(1)),
                true,
                true,
            )
            .unwrap(),
        );
        let hard_middle = UnresolvedRouteFact::Hard(bounds_requirement("hard-middle"));
        let hard_terminal = UnresolvedRouteFact::Hard(bounds_requirement("hard-terminal"));
        let conditional_terminal =
            UnresolvedRouteFact::Hard(bounds_requirement("conditional-terminal"));
        let mut root_summary = FunctionSummary::empty(root.clone());
        root_summary.unresolved_predecessors.extend([
            task8_edge(&root.0, hard.clone(), hard_middle.clone(), &middle.0, 1),
            task8_edge(
                &root.0,
                conditional.clone(),
                conditional_terminal.clone(),
                &sink.0,
                2,
            ),
        ]);
        let mut middle_summary = FunctionSummary::empty(middle.clone());
        middle_summary.unresolved_predecessors.insert(task8_edge(
            &middle.0,
            hard_middle,
            hard_terminal.clone(),
            &sink.0,
            3,
        ));
        let summaries = BTreeMap::from([
            (root.clone(), root_summary),
            (middle, middle_summary),
            (sink.clone(), FunctionSummary::empty(sink.clone())),
        ]);
        let roots = BTreeSet::from([
            UnresolvedRouteNode {
                owner: root.clone(),
                route: hard,
            },
            UnresolvedRouteNode {
                owner: root,
                route: conditional,
            },
        ]);
        let complete_seeds = BTreeMap::from([(sink.clone(), {
            let mut seed = task8_seed(&sink.0, hard_terminal);
            seed.requirements
                .insert(conditional_terminal.requirement().clone());
            seed
        })]);
        let selected =
            select_preferred_unresolved_route(&roots, &summaries, &complete_seeds).unwrap();
        assert_eq!(selected.steps.len(), 1);
        assert_eq!(selected.terminal.route, conditional_terminal);

        let missing_one_seed = BTreeMap::from([(
            sink.clone(),
            task8_seed(&sink.0, selected.terminal.route.clone()),
        )]);
        assert!(select_preferred_unresolved_route(&roots, &summaries, &missing_one_seed).is_err());
    }

    #[test]
    fn task8_route_cycle_uses_only_exact_corridor_scc_and_a_real_recursive_edge() {
        let a = FunctionKey::new("crate::a");
        let b = FunctionKey::new("crate::b");
        let sink = FunctionKey::new("crate::sink");
        let r1 = UnresolvedRouteFact::Hard(bounds_requirement("r1"));
        let r2 = UnresolvedRouteFact::Hard(bounds_requirement("r2"));
        let r3 = UnresolvedRouteFact::Hard(bounds_requirement("r3"));
        let r4 = UnresolvedRouteFact::Hard(bounds_requirement("r4"));
        let terminal = UnresolvedRouteFact::Hard(bounds_requirement("terminal-cycle"));
        let unrelated = UnresolvedRouteFact::Hard(bounds_requirement("unrelated-cycle"));
        let mut a_summary = FunctionSummary::empty(a.clone());
        a_summary.unresolved_predecessors.extend([
            task8_edge(&a.0, r1.clone(), r2.clone(), &b.0, 30),
            task8_edge(&a.0, r3.clone(), r4.clone(), &b.0, 10),
            task8_edge(&a.0, unrelated.clone(), unrelated.clone(), &a.0, 1),
        ]);
        let mut b_summary = FunctionSummary::empty(b.clone());
        b_summary.unresolved_predecessors.extend([
            task8_edge(&b.0, r2, r3.clone(), &a.0, 40),
            task8_edge(&b.0, r4.clone(), r3, &a.0, 20),
            task8_edge(&b.0, r4, terminal.clone(), &sink.0, 50),
        ]);
        let summaries = BTreeMap::from([
            (a.clone(), a_summary),
            (b.clone(), b_summary),
            (sink.clone(), FunctionSummary::empty(sink.clone())),
        ]);
        let selected = select_unresolved_route(
            UnresolvedRouteNode {
                owner: a.clone(),
                route: r1,
            },
            &summaries,
            &BTreeMap::from([(sink.clone(), task8_seed(&sink.0, terminal))]),
        )
        .unwrap();
        let cycle = select_unresolved_route_cycle(&selected, &summaries)
            .unwrap()
            .unwrap();
        assert_eq!(cycle.token, "scc:[crate::a,crate::b]");
        assert_eq!(cycle.span.start.line, 11);
        assert_eq!(cycle.nodes.len(), 2);
        assert!(!cycle.nodes.iter().any(|node| node.route == unrelated));
    }

    #[test]
    fn task8_route_cycle_detects_direct_recursion_but_not_an_unrelated_branch() {
        let root = FunctionKey::new("crate::root");
        let route = UnresolvedRouteFact::Hard(bounds_requirement("direct-cycle"));
        let unrelated = UnresolvedRouteFact::Hard(bounds_requirement("unrelated-direct"));
        let selected = SelectedUnresolvedRoute {
            root: UnresolvedRouteNode {
                owner: root.clone(),
                route: route.clone(),
            },
            terminal: UnresolvedRouteNode {
                owner: root.clone(),
                route: route.clone(),
            },
            steps: Vec::new(),
        };
        let mut direct = FunctionSummary::empty(root.clone());
        direct.unresolved_predecessors.insert(task8_edge(
            &root.0,
            route.clone(),
            route,
            &root.0,
            7,
        ));
        let cycle =
            select_unresolved_route_cycle(&selected, &BTreeMap::from([(root.clone(), direct)]))
                .unwrap()
                .unwrap();
        assert_eq!(cycle.token, "scc:[crate::root]");
        assert_eq!(cycle.span.start.line, 8);

        let mut unrelated_only = FunctionSummary::empty(root.clone());
        unrelated_only.unresolved_predecessors.insert(task8_edge(
            &root.0,
            unrelated.clone(),
            unrelated,
            &root.0,
            3,
        ));
        assert_eq!(
            select_unresolved_route_cycle(&selected, &BTreeMap::from([(root, unrelated_only)]),)
                .unwrap(),
            None
        );
    }

    #[test]
    fn task8_recursive_detour_does_not_mark_the_preferred_direct_route() {
        let root = FunctionKey::new("crate::root");
        let detour = FunctionKey::new("crate::detour");
        let sink = FunctionKey::new("crate::sink");
        let start = UnresolvedRouteFact::Hard(bounds_requirement("detour-start"));
        let recursive = UnresolvedRouteFact::Hard(bounds_requirement("detour-cycle"));
        let terminal = UnresolvedRouteFact::Hard(bounds_requirement("detour-terminal"));
        let mut root_summary = FunctionSummary::empty(root.clone());
        root_summary.unresolved_predecessors.extend([
            task8_edge(&root.0, start.clone(), terminal.clone(), &sink.0, 1),
            task8_edge(&root.0, start.clone(), recursive.clone(), &detour.0, 2),
        ]);
        let mut detour_summary = FunctionSummary::empty(detour.clone());
        detour_summary.unresolved_predecessors.extend([
            task8_edge(
                &detour.0,
                recursive.clone(),
                recursive.clone(),
                &detour.0,
                3,
            ),
            task8_edge(&detour.0, recursive, terminal.clone(), &sink.0, 4),
        ]);
        let summaries = BTreeMap::from([
            (root.clone(), root_summary),
            (detour, detour_summary),
            (sink.clone(), FunctionSummary::empty(sink.clone())),
        ]);
        let selected = select_unresolved_route(
            UnresolvedRouteNode {
                owner: root,
                route: start,
            },
            &summaries,
            &BTreeMap::from([(sink.clone(), task8_seed(&sink.0, terminal))]),
        )
        .unwrap();
        assert_eq!(selected.steps.len(), 1);
        assert_eq!(selected.terminal.owner, sink);
        assert_eq!(
            select_unresolved_route_cycle(&selected, &summaries).unwrap(),
            None
        );
    }

    #[test]
    fn entry_validation_contract_requires_formals_dominance_and_current_versions() {
        let requirement = bounds_requirement("bounds");
        let exported = ValidationFact::export_entry_contract(
            requirement.clone(),
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Formal(1)),
            true,
            true,
        );
        assert!(exported.is_some());
        assert!(ValidationFact::export_entry_contract(
            requirement.clone(),
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Formal(1)),
            false,
            true,
        )
        .is_none());
        assert!(ValidationFact::export_entry_contract(
            requirement.clone(),
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Formal(1)),
            true,
            false,
        )
        .is_none());
        assert!(ValidationFact::export_entry_contract(
            requirement.clone(),
            Some(BoundarySlot::Return),
            Some(BoundarySlot::Formal(1)),
            true,
            true,
        )
        .is_none());
        assert!(ValidationFact::export_entry_contract(
            requirement,
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Out(1)),
            true,
            true,
        )
        .is_none());
    }

    #[test]
    fn validation_contract_substitutes_actuals_and_requires_same_value_pair() {
        let contract = ValidationFact::export_entry_contract(
            bounds_requirement("bounds"),
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Formal(1)),
            true,
            true,
        )
        .unwrap();
        let actuals = vec![
            AbstractValue::new([AbstractOrigin::Formal(7)]),
            AbstractValue::new([AbstractOrigin::Formal(8)]),
        ];
        let instantiated = contract.instantiate_requirement(&actuals);
        assert_eq!(
            instantiated.collection,
            Some(AbstractValue::new([AbstractOrigin::Formal(7)]))
        );
        assert_eq!(
            instantiated.subject,
            AbstractValue::new([AbstractOrigin::Formal(8)])
        );
        assert!(contract.matches_instantiated_requirement(&instantiated, &actuals));

        let mut wrong_collection = instantiated.clone();
        wrong_collection.collection = Some(AbstractValue::new([AbstractOrigin::Formal(9)]));
        assert!(!contract.matches_instantiated_requirement(&wrong_collection, &actuals));

        let mut wrong_subject = instantiated.clone();
        wrong_subject.subject = AbstractValue::new([AbstractOrigin::Formal(9)]);
        assert!(!contract.matches_instantiated_requirement(&wrong_subject, &actuals));
    }

    #[test]
    fn validation_contracts_intersect_across_all_feasible_paths() {
        let common = ValidationFact::export_entry_contract(
            bounds_requirement("common"),
            Some(BoundarySlot::Formal(2)),
            Some(BoundarySlot::Formal(1)),
            true,
            true,
        )
        .unwrap();
        let only_left = ValidationFact::export_entry_contract(
            bounds_requirement("left"),
            Some(BoundarySlot::Formal(3)),
            Some(BoundarySlot::Formal(1)),
            true,
            true,
        )
        .unwrap();
        let left = BTreeSet::from([common.clone(), only_left]);
        let right = BTreeSet::from([common.clone()]);
        assert_eq!(
            intersect_validation_paths([&left, &right]),
            BTreeSet::from([common])
        );
        assert!(intersect_validation_paths([&left, &BTreeSet::new()]).is_empty());
    }

    #[test]
    fn call_mapping_substitutes_formals_into_return_and_projected_out_places() {
        let public_field = AbstractOrigin::PublicField {
            def_path: "crate::Config::pointer".into(),
            span: span(2),
        };
        let ffi_output = AbstractOrigin::FfiOutput {
            token: "ffi-out:fixture".into(),
            span: span(3),
        };
        let actuals = vec![
            AbstractValue::new([public_field.clone()]),
            AbstractValue::new([AbstractOrigin::Constant(7)]),
        ];
        let return_value =
            AbstractValue::new([AbstractOrigin::Formal(2), AbstractOrigin::Constant(9)]);
        let out_values = BTreeMap::from([(
            0,
            AbstractValue::new([AbstractOrigin::Formal(1), ffi_output.clone()]),
        )]);
        let mappings = BTreeSet::from([
            CallMapping::ActualToFormal {
                actual: PlaceKey::new("_2"),
                formal_index: 1,
            },
            CallMapping::ReturnToDestination {
                destination: PlaceKey::new("_3"),
            },
            CallMapping::OutToCaller {
                formal_index: 0,
                caller_place: PlaceKey::new("_4.*"),
            },
        ]);
        let mapped = map_call_outputs(&mappings, &return_value, &out_values, &actuals);
        assert_eq!(mapped.len(), 2);
        assert_eq!(
            mapped[&PlaceKey::new("_3")],
            AbstractValue::new([AbstractOrigin::Constant(7), AbstractOrigin::Constant(9)])
        );
        assert_eq!(
            mapped[&PlaceKey::new("_4.*")],
            AbstractValue::new([public_field, ffi_output])
        );
        assert!(!mapped.contains_key(&PlaceKey::new("_2")));
    }

    #[test]
    fn constant_return_does_not_inherit_unreturned_actual_origins() {
        let mappings = BTreeSet::from([CallMapping::ReturnToDestination {
            destination: PlaceKey::new("_3"),
        }]);
        let actuals = vec![AbstractValue::new([AbstractOrigin::Formal(1)])];
        let mapped = map_call_outputs(
            &mappings,
            &AbstractValue::new([AbstractOrigin::Constant(0)]),
            &BTreeMap::new(),
            &actuals,
        );

        assert_eq!(
            mapped[&PlaceKey::new("_3")],
            AbstractValue::new([AbstractOrigin::Constant(0)])
        );
    }

    #[test]
    fn returned_formal_reaches_only_the_return_destination() {
        let mappings = BTreeSet::from([
            CallMapping::ActualToFormal {
                actual: PlaceKey::new("_1"),
                formal_index: 1,
            },
            CallMapping::ReturnToDestination {
                destination: PlaceKey::new("_4"),
            },
        ]);
        let actual = AbstractValue::new([AbstractOrigin::PublicField {
            def_path: "crate::Config::index".into(),
            span: span(4),
        }]);
        let mapped = map_call_outputs(
            &mappings,
            &AbstractValue::new([AbstractOrigin::Formal(1)]),
            &BTreeMap::new(),
            &[actual.clone()],
        );

        assert_eq!(mapped, BTreeMap::from([(PlaceKey::new("_4"), actual)]));
    }

    #[test]
    fn structured_out_dependencies_map_only_the_selected_projected_actual() {
        let dependencies = BTreeSet::from([OutDependency {
            formal_index: 1,
            value: AbstractValue::new([AbstractOrigin::Formal(2)]),
            may_skip_write: false,
        }]);
        let out_values = out_values_from_dependencies(&dependencies);
        let mappings = BTreeSet::from([CallMapping::OutToCaller {
            formal_index: 1,
            caller_place: PlaceKey::new("(*_7).0"),
        }]);
        let actuals = vec![
            AbstractValue::new([AbstractOrigin::Constant(11)]),
            AbstractValue::new([AbstractOrigin::Formal(3)]),
        ];

        assert_eq!(
            map_call_outputs(&mappings, &AbstractValue::default(), &out_values, &actuals,),
            BTreeMap::from([(
                PlaceKey::new("(*_7).0"),
                AbstractValue::new([AbstractOrigin::Formal(3)]),
            )])
        );
    }

    #[test]
    fn primary_precedence_uses_failure_then_source_provenance() {
        assert_eq!(
            classify_primary(
                FailureClass::InternalInvalidValue,
                SourceKind::PublicParameter
            ),
            Pattern::P3
        );
        assert_eq!(
            classify_primary(
                FailureClass::SinkPrecondition,
                SourceKind::OpenBehaviorOutput
            ),
            Pattern::P6
        );
        assert_eq!(
            classify_primary(
                FailureClass::SinkPrecondition,
                SourceKind::GenericNonEmptyCapability
            ),
            Pattern::P5
        );
        assert_eq!(
            classify_primary(
                FailureClass::SinkPrecondition,
                SourceKind::LiteralPublicField
            ),
            Pattern::P2
        );
        assert_eq!(
            classify_primary(FailureClass::SinkPrecondition, SourceKind::PublicParameter),
            Pattern::P1
        );
        assert_eq!(
            classify_primary(FailureClass::SinkPrecondition, SourceKind::InternalDerived),
            Pattern::P4
        );
    }

    #[test]
    fn primary_selection_preserves_all_other_sources_as_stable_secondary_metadata() {
        let (primary, secondary) = classify_primary_with_secondary(
            FailureClass::SinkPrecondition,
            SourceKind::LiteralPublicField,
            [
                SourceKind::InternalDerived,
                SourceKind::LiteralPublicField,
                SourceKind::PublicParameter,
                SourceKind::PublicParameter,
            ],
        );
        assert_eq!(primary, Pattern::P2);
        assert_eq!(
            secondary.into_iter().collect::<Vec<_>>(),
            vec![SourceKind::PublicParameter, SourceKind::InternalDerived]
        );
    }

    #[test]
    fn causal_dedup_chooses_shortest_then_lexical_witness() {
        let source = Source {
            kind: SourceKind::PublicParameter,
            origin: OriginKey::new("arg:1"),
            span: span(2),
        };
        let sink = Operation {
            kind: OperationKind::GetUnchecked,
            point: point("crate::sink", 3, 0),
        };
        let sink_obligation = SinkObligation {
            source: source.clone(),
            first_failure: sink.clone(),
            sink: sink.clone(),
            obligation: Obligation::new([Predicate::InBounds], OriginKey::new("arg:1")),
        };
        let key = CausalKey::from_parts(
            "crate",
            FunctionKey::new("crate::entry"),
            &source,
            &sink,
            &sink,
            &sink_obligation.obligation,
        );
        let step = |kind, function, line| WitnessStep {
            kind,
            function: FunctionKey::new(function),
            span: span(line),
        };
        let longer = Finding::new(
            "crate",
            FunctionKey::new("crate::entry"),
            span(1),
            sink_obligation.clone(),
            Pattern::P1,
            RuleId::P1GetUnchecked,
            CanonicalWitness::new(
                vec![
                    step(WitnessStepKind::Entry, "crate::entry", 1),
                    step(WitnessStepKind::LocalCall, "crate::middle", 2),
                    step(WitnessStepKind::Sink, "crate::sink", 3),
                ],
                vec![FunctionKey::new("crate::middle")],
            ),
            BTreeSet::new(),
        );
        let shorter = Finding::new(
            "crate",
            FunctionKey::new("crate::entry"),
            span(1),
            sink_obligation,
            Pattern::P1,
            RuleId::P1GetUnchecked,
            CanonicalWitness::new(
                vec![
                    step(WitnessStepKind::Entry, "crate::entry", 1),
                    step(WitnessStepKind::Sink, "crate::sink", 3),
                ],
                vec![],
            ),
            BTreeSet::from([SourceKind::InternalDerived]),
        );
        assert_eq!(longer.causal_key, key);
        assert_eq!(shorter.causal_key, key);
        let findings = deduplicate_findings([longer, shorter]);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].witness.steps.len(), 2);
        assert!(findings[0]
            .secondary_sources
            .contains(&SourceKind::InternalDerived));
    }

    #[test]
    fn tarjan_and_fixed_point_handle_self_and_mutual_recursion_without_a_round_cap() {
        let a = FunctionKey::new("a");
        let b = FunctionKey::new("b");
        let c = FunctionKey::new("c");
        let graph = BTreeMap::from([
            (a.clone(), BTreeSet::from([b.clone()])),
            (b.clone(), BTreeSet::from([a.clone(), c.clone()])),
            (c.clone(), BTreeSet::from([c.clone()])),
        ]);
        let components = strongly_connected_components(&graph);
        assert!(components.contains(&vec![a.clone(), b.clone()]));
        assert!(components.contains(&vec![c.clone()]));
        let mut seeds = BTreeMap::new();
        let mut c_seed = FunctionSummary::empty(c.clone());
        c_seed.return_origins.insert(OriginKey::new("seed"));
        seeds.insert(c.clone(), c_seed);
        let solved = solve_summaries(&graph, &seeds, |node, current| {
            let mut derived = FunctionSummary::empty(node.clone());
            for callee in graph.get(node).into_iter().flatten() {
                derived
                    .return_origins
                    .extend(current[callee].return_origins.iter().cloned());
            }
            derived
        });
        for node in [&a, &b, &c] {
            assert!(solved[node]
                .return_origins
                .contains(&OriginKey::new("seed")));
            assert_eq!(solved[node].cycle_tokens.len(), 1);
        }
        assert_eq!(solved[&a].cycle_tokens, solved[&b].cycle_tokens);
        assert_ne!(solved[&a].cycle_tokens, solved[&c].cycle_tokens);
    }

    #[test]
    fn independent_components_clone_only_their_own_summary_candidates() {
        let nodes = (0..96)
            .map(|index| FunctionKey::new(format!("crate::f{index:03}")))
            .collect::<Vec<_>>();
        let graph = nodes
            .iter()
            .cloned()
            .map(|node| (node, BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        let seeds = nodes
            .iter()
            .cloned()
            .map(|node| (node.clone(), FunctionSummary::empty(node)))
            .collect::<BTreeMap<_, _>>();
        let (solved, stats) = solve_summaries_with_stats(&graph, &seeds, |node, _| {
            let mut derived = FunctionSummary::empty(node.clone());
            derived.return_origins.insert(OriginKey::new("grown"));
            derived
        });

        assert_eq!(solved.len(), nodes.len());
        assert!(solved
            .values()
            .all(|summary| summary.return_origins.contains(&OriginKey::new("grown"))));
        assert_eq!(stats.component_rounds, nodes.len() * 2);
        assert_eq!(stats.summary_candidate_clones, nodes.len() * 2);
    }

    #[test]
    fn staged_component_updates_match_the_reference_jacobi_solver_exactly() {
        fn reference<F>(
            graph: &BTreeMap<FunctionKey, BTreeSet<FunctionKey>>,
            seeds: &BTreeMap<FunctionKey, FunctionSummary>,
            mut transfer: F,
        ) -> BTreeMap<FunctionKey, FunctionSummary>
        where
            F: FnMut(&FunctionKey, &BTreeMap<FunctionKey, FunctionSummary>) -> FunctionSummary,
        {
            let mut complete_graph = graph.clone();
            for node in seeds.keys() {
                complete_graph.entry(node.clone()).or_default();
            }
            for callee in graph.values().flatten() {
                complete_graph.entry(callee.clone()).or_default();
            }
            let components = strongly_connected_components(&complete_graph);
            let order = callee_first_components(&complete_graph, &components);
            let mut solved = complete_graph
                .keys()
                .map(|node| {
                    (
                        node.clone(),
                        seeds
                            .get(node)
                            .cloned()
                            .unwrap_or_else(|| FunctionSummary::empty(node.clone())),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            for component_index in order {
                let component = &components[component_index];
                let recursive = component.len() > 1
                    || component
                        .first()
                        .is_some_and(|node| complete_graph[node].contains(node));
                let cycle_token = recursive.then(|| scc_cycle_token(component));
                loop {
                    let snapshot = solved.clone();
                    let mut changed = false;
                    for node in component {
                        let mut next = snapshot[node].clone();
                        if let Some(seed) = seeds.get(node) {
                            next.join(seed);
                        }
                        next.join(&transfer(node, &snapshot));
                        if let Some(token) = &cycle_token {
                            next.cycle_tokens.insert(token.clone());
                        }
                        if next != solved[node] {
                            solved.insert(node.clone(), next);
                            changed = true;
                        }
                    }
                    if !changed {
                        break;
                    }
                }
            }
            solved
        }

        let a = FunctionKey::new("crate::a");
        let b = FunctionKey::new("crate::b");
        let c = FunctionKey::new("crate::c");
        let d = FunctionKey::new("crate::d");
        let graph = BTreeMap::from([
            (a.clone(), BTreeSet::from([b.clone(), c.clone()])),
            (b.clone(), BTreeSet::from([c.clone()])),
            (c.clone(), BTreeSet::from([b.clone(), d.clone()])),
            (d.clone(), BTreeSet::from([d.clone()])),
        ]);
        let mut seeds = graph
            .keys()
            .cloned()
            .map(|node| (node.clone(), FunctionSummary::empty(node)))
            .collect::<BTreeMap<_, _>>();
        seeds
            .get_mut(&d)
            .expect("d seed exists")
            .return_origins
            .insert(OriginKey::new("seed::d"));
        seeds
            .get_mut(&b)
            .expect("b seed exists")
            .cycle_tokens
            .insert("seed-token".to_owned());
        let transfer = |node: &FunctionKey, current: &BTreeMap<FunctionKey, FunctionSummary>| {
            let mut derived = FunctionSummary::empty(node.clone());
            for callee in graph.get(node).into_iter().flatten() {
                derived
                    .return_origins
                    .extend(current[callee].return_origins.iter().cloned());
                derived
                    .cycle_tokens
                    .extend(current[callee].cycle_tokens.iter().cloned());
            }
            derived
        };

        let expected = reference(&graph, &seeds, transfer);
        let actual = solve_summaries(&graph, &seeds, transfer);
        assert_eq!(actual, expected);
    }

    #[test]
    fn recursive_witness_saturates_to_one_cycle_token() {
        let cycle = WitnessStep {
            kind: WitnessStepKind::SccCycle,
            function: FunctionKey::new("scc:[a,b]"),
            span: span(2),
        };
        let witness = CanonicalWitness::new(
            vec![
                WitnessStep {
                    kind: WitnessStepKind::Entry,
                    function: FunctionKey::new("a"),
                    span: span(1),
                },
                cycle.clone(),
                cycle,
                WitnessStep {
                    kind: WitnessStepKind::Sink,
                    function: FunctionKey::new("b"),
                    span: span(3),
                },
            ],
            vec![FunctionKey::new("a"), FunctionKey::new("b")],
        );
        assert_eq!(witness.depth, PropagationDepth::Recursive);
        assert_eq!(witness.local_call_count, None);
        assert_eq!(
            witness
                .steps
                .iter()
                .filter(|step| step.kind == WitnessStepKind::SccCycle)
                .count(),
            1
        );
    }

    #[test]
    fn canonical_key_is_stable_and_sha256_backed() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let source = Source {
            kind: SourceKind::PublicParameter,
            origin: OriginKey::new("arg:1"),
            span: span(1),
        };
        let sink = Operation {
            kind: OperationKind::RawRead,
            point: point("crate::entry", 0, 0),
        };
        let key = CausalKey::from_parts(
            "crate",
            FunctionKey::new("crate::entry"),
            &source,
            &sink,
            &sink,
            &Obligation::new([Predicate::ValidForRead], OriginKey::new("arg:1")),
        );
        assert_eq!(key.finding_id().len(), 64);
        assert_eq!(key.finding_id(), key.finding_id());
        assert!(key
            .canonical_json()
            .starts_with("{\"canonical_obligation\":"));
    }
}
