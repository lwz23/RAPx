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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValidationFact {
    pub place: PlaceKey,
    pub origin: OriginKey,
    pub predicate: Predicate,
    pub established_at: ProgramPoint,
    pub place_version: u32,
}

impl ValidationFact {
    pub fn entails(
        &self,
        place: &PlaceKey,
        origin: &OriginKey,
        predicate: Predicate,
        current_place_version: u32,
    ) -> bool {
        self.place == *place
            && self.origin == *origin
            && self.predicate == predicate
            && self.place_version == current_place_version
    }
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
    pub origin: OriginKey,
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
        }
    }

    pub fn join(&mut self, other: &Self) -> bool {
        assert_eq!(
            self.function, other.function,
            "summaries for different functions cannot be joined"
        );
        let before = self.clone();
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
        *self != before
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
    pub primary: Pattern,
    pub rule: RuleId,
    pub witness: CanonicalWitness,
    pub secondary_sources: BTreeSet<SourceKind>,
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
            let snapshot = solved.clone();
            let mut changed = false;
            for node in component {
                let mut next = snapshot[node].clone();
                if let Some(seed) = seeds.get(node) {
                    next.join(seed);
                }
                let derived = transfer(node, &snapshot);
                assert_eq!(
                    derived.function, *node,
                    "transfer summary key must match function"
                );
                next.join(&derived);
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

    #[test]
    fn summary_join_is_monotone_idempotent_and_stably_sorted() {
        let key = FunctionKey::new("crate::entry");
        let mut left = FunctionSummary::empty(key.clone());
        left.return_origins.insert(OriginKey::new("z"));
        let mut right = FunctionSummary::empty(key);
        right.return_origins.insert(OriginKey::new("a"));
        assert!(left.join(&right));
        assert!(!left.join(&right));
        assert_eq!(
            left.return_origins
                .iter()
                .map(|origin| origin.0.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
    }

    #[test]
    fn validation_entailment_binds_place_origin_predicate_and_version() {
        let fact = ValidationFact {
            place: PlaceKey::new("_2"),
            origin: OriginKey::new("arg:1"),
            predicate: Predicate::InBounds,
            established_at: point("crate::entry", 1, 0),
            place_version: 3,
        };
        assert!(fact.entails(
            &PlaceKey::new("_2"),
            &OriginKey::new("arg:1"),
            Predicate::InBounds,
            3
        ));
        assert!(!fact.entails(
            &PlaceKey::new("_3"),
            &OriginKey::new("arg:1"),
            Predicate::InBounds,
            3
        ));
        assert!(!fact.entails(
            &PlaceKey::new("_2"),
            &OriginKey::new("arg:1"),
            Predicate::InBounds,
            4
        ));
    }

    #[test]
    fn call_mapping_distinguishes_actual_return_and_out_state() {
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
        assert_eq!(mappings.len(), 3);
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
        let key = CausalKey::from_parts(
            "crate",
            FunctionKey::new("crate::entry"),
            &source,
            &sink,
            &sink,
            &Obligation::new([Predicate::InBounds], OriginKey::new("arg:1")),
        );
        let step = |kind, function, line| WitnessStep {
            kind,
            function: FunctionKey::new(function),
            span: span(line),
        };
        let longer = Finding {
            causal_key: key.clone(),
            primary: Pattern::P1,
            rule: RuleId::P1GetUnchecked,
            witness: CanonicalWitness::new(
                vec![
                    step(WitnessStepKind::Entry, "crate::entry", 1),
                    step(WitnessStepKind::LocalCall, "crate::middle", 2),
                    step(WitnessStepKind::Sink, "crate::sink", 3),
                ],
                vec![FunctionKey::new("crate::middle")],
            ),
            secondary_sources: BTreeSet::new(),
        };
        let shorter = Finding {
            causal_key: key,
            primary: Pattern::P1,
            rule: RuleId::P1GetUnchecked,
            witness: CanonicalWitness::new(
                vec![
                    step(WitnessStepKind::Entry, "crate::entry", 1),
                    step(WitnessStepKind::Sink, "crate::sink", 3),
                ],
                vec![],
            ),
            secondary_sources: BTreeSet::from([SourceKind::InternalDerived]),
        };
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
