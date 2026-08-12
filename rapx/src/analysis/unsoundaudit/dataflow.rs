use rustc_index::IndexVec;
use rustc_middle::mir::{BasicBlock, START_BLOCK};
use std::collections::{BTreeMap, BTreeSet};

/// A finite, commutative may-join used by the CFG lattice.
///
/// Implementations must be associative, commutative, idempotent, and finite for
/// the set of facts extracted from one MIR body. The solver deliberately has no
/// iteration cap and therefore relies on those lattice laws.
pub(super) trait MayJoin {
    /// Joins `other` into `self` and reports whether `self` changed.
    fn join_may(&mut self, other: &Self) -> bool;
}

impl<T> MayJoin for BTreeSet<T>
where
    T: Clone + Ord,
{
    fn join_may(&mut self, other: &Self) -> bool {
        let before = self.len();
        self.extend(other.iter().cloned());
        self.len() != before
    }
}

/// A stable identity for one syntactic MIR write site.
///
/// A token is reused every time a loop is revisited. It must never be replaced
/// by a dynamic counter, which would make the lattice infinite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct StaticWriteToken {
    block: u32,
    statement: u32,
}

impl StaticWriteToken {
    pub(super) fn new(block: u32, statement: u32) -> Self {
        Self { block, statement }
    }
}

/// A semantic validation binding. Evidence locations are intentionally absent
/// from its identity so equivalent checks established on distinct paths can
/// survive predecessor intersection.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct BoundValidation<K, P> {
    predicate: P,
    bound_versions: BTreeMap<K, BTreeSet<StaticWriteToken>>,
}

impl<K, P> BoundValidation<K, P>
where
    K: Clone + Ord,
    P: Ord,
{
    pub(super) fn new(
        predicate: P,
        bound_places: impl IntoIterator<Item = K>,
    ) -> Self {
        Self {
            predicate,
            bound_versions: bound_places
                .into_iter()
                .map(|place| (place, BTreeSet::new()))
                .collect(),
        }
    }

    fn bind_versions(&mut self, versions: &BTreeMap<K, BTreeSet<StaticWriteToken>>) {
        for (place, version) in &mut self.bound_versions {
            *version = versions.get(place).cloned().unwrap_or_default();
        }
    }

    fn binds(&self, place: &K) -> bool {
        self.bound_versions.contains_key(place)
    }

    pub(super) fn predicate(&self) -> &P {
        &self.predicate
    }

    pub(super) fn bound_versions(&self) -> &BTreeMap<K, BTreeSet<StaticWriteToken>> {
        &self.bound_versions
    }

    fn same_binding(&self, other: &Self) -> bool {
        self.predicate == other.predicate
            && self.bound_versions.keys().eq(other.bound_versions.keys())
    }

    fn merge_versions(&mut self, other: &Self) {
        for (place, versions) in &other.bound_versions {
            self.bound_versions
                .get_mut(place)
                .expect("equal validation bindings have equal place keys")
                .extend(versions.iter().copied());
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BlockState<K, V, P> {
    may_values: BTreeMap<K, V>,
    must_validations: BTreeSet<BoundValidation<K, P>>,
    versions: BTreeMap<K, BTreeSet<StaticWriteToken>>,
}

impl<K, V, P> BlockState<K, V, P>
where
    K: Clone + Ord,
    V: Clone + Eq + MayJoin,
    P: Clone + Ord,
{
    pub(super) fn empty() -> Self {
        Self {
            may_values: BTreeMap::new(),
            must_validations: BTreeSet::new(),
            versions: BTreeMap::new(),
        }
    }

    pub(super) fn may_values(&self) -> &BTreeMap<K, V> {
        &self.may_values
    }

    pub(super) fn must_validations(&self) -> &BTreeSet<BoundValidation<K, P>> {
        &self.must_validations
    }

    /// Queries a validation against the current reaching definitions at this
    /// program point. Both the semantic binding and all bound versions must
    /// match exactly.
    pub(super) fn has_current_validation(
        &self,
        predicate: &P,
        bound_places: impl IntoIterator<Item = K>,
    ) -> bool {
        let expected = bound_places
            .into_iter()
            .map(|place| {
                let versions = self.versions.get(&place).cloned().unwrap_or_default();
                (place, versions)
            })
            .collect::<BTreeMap<_, _>>();
        self.must_validations.iter().any(|validation| {
            validation.predicate() == predicate && validation.bound_versions() == &expected
        })
    }

    pub(super) fn versions(&self) -> &BTreeMap<K, BTreeSet<StaticWriteToken>> {
        &self.versions
    }

    /// Strongly assigns the value at one program point. May-union happens only
    /// when predecessor states are joined.
    pub(super) fn set_value(&mut self, place: K, value: V) {
        self.may_values.insert(place, value);
    }

    pub(super) fn establish(&mut self, mut validation: BoundValidation<K, P>) {
        validation.bind_versions(&self.versions);
        self.must_validations.insert(validation);
    }

    /// Applies a syntactic write. Reapplying the same token is idempotent, while
    /// every validation bound to the written storage is invalidated.
    pub(super) fn record_write(&mut self, place: K, token: StaticWriteToken) {
        self.versions.insert(place.clone(), BTreeSet::from([token]));
        self.must_validations
            .retain(|validation| !validation.binds(&place));
    }

    /// Joins reachable predecessor states. An empty iterator means unreachable
    /// and returns `None`; it is not treated as a reachable empty state.
    pub(super) fn join_predecessors<'a>(
        predecessors: impl IntoIterator<Item = &'a Self>,
    ) -> Option<Self>
    where
        K: 'a,
        V: 'a,
        P: 'a,
    {
        let mut predecessors = predecessors.into_iter();
        let mut joined = predecessors.next()?.clone();
        for predecessor in predecessors {
            for (place, value) in &predecessor.may_values {
                match joined.may_values.get_mut(place) {
                    Some(current) => {
                        current.join_may(value);
                    }
                    None => {
                        joined.may_values.insert(place.clone(), value.clone());
                    }
                }
            }
            let mut must = BTreeSet::new();
            for mut validation in joined.must_validations {
                let Some(other) = predecessor
                    .must_validations
                    .iter()
                    .find(|candidate| validation.same_binding(candidate))
                else {
                    continue;
                };
                validation.merge_versions(other);
                must.insert(validation);
            }
            joined.must_validations = must;
            for (place, versions) in &predecessor.versions {
                joined
                    .versions
                    .entry(place.clone())
                    .or_default()
                    .extend(versions.iter().copied());
            }
        }
        Some(joined)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DataflowResult<K, V, P> {
    pub(super) entry: IndexVec<BasicBlock, Option<BlockState<K, V, P>>>,
    pub(super) before: BTreeMap<(BasicBlock, usize), BlockState<K, V, P>>,
    pub(super) exit: IndexVec<BasicBlock, Option<BlockState<K, V, P>>>,
    pub(super) edge: BTreeMap<(BasicBlock, BasicBlock), BlockState<K, V, P>>,
}

/// Solves a finite forward CFG lattice to equality in deterministic block order.
///
/// Both transfer functions must be deterministic and monotone over `BlockState`.
/// Task-specific implementations may strongly replace the value at a fixed MIR
/// definition, but the value computed there must itself be monotone in the input
/// state. This is a hard contract of this crate-private solver.
pub(super) fn solve_cfg<K, V, P, F, E>(
    successors: &IndexVec<BasicBlock, BTreeSet<BasicBlock>>,
    operation_counts: &IndexVec<BasicBlock, usize>,
    entry_seed: BlockState<K, V, P>,
    mut transfer: F,
    mut edge_transfer: E,
) -> DataflowResult<K, V, P>
where
    K: Clone + Ord,
    V: Clone + Eq + MayJoin,
    P: Clone + Ord,
    F: FnMut(BasicBlock, usize, &mut BlockState<K, V, P>),
    E: FnMut(BasicBlock, BasicBlock, &mut BlockState<K, V, P>),
{
    assert_eq!(
        successors.len(),
        operation_counts.len(),
        "CFG and operation-count vectors must have the same block count"
    );
    assert!(
        !successors.is_empty(),
        "a MIR CFG must contain the start block"
    );

    let mut predecessors = IndexVec::from_elem_n(BTreeSet::new(), successors.len());
    for (block, targets) in successors.iter_enumerated() {
        for target in targets {
            assert!(
                target.index() < successors.len(),
                "CFG successor is outside the block vector"
            );
            predecessors[*target].insert(block);
        }
    }

    let mut result = DataflowResult {
        entry: IndexVec::from_elem_n(None, successors.len()),
        before: BTreeMap::new(),
        exit: IndexVec::from_elem_n(None, successors.len()),
        edge: BTreeMap::new(),
    };
    result.entry[START_BLOCK] = Some(entry_seed.clone());
    let mut worklist = BTreeSet::from([START_BLOCK]);

    while let Some(block) = worklist.pop_first() {
        let Some(mut state) = result.entry[block].clone() else {
            continue;
        };
        for operation in 0..operation_counts[block] {
            result.before.insert((block, operation), state.clone());
            transfer(block, operation, &mut state);
        }
        if result.exit[block].as_ref() == Some(&state) {
            continue;
        }
        result.exit[block] = Some(state);

        for target in &successors[block] {
            let mut outgoing = result.exit[block]
                .clone()
                .expect("a processed reachable block has an exit state");
            edge_transfer(block, *target, &mut outgoing);
            let edge_changed = result.edge.get(&(block, *target)) != Some(&outgoing);
            if edge_changed {
                result.edge.insert((block, *target), outgoing);
            }
            let mut incoming = Vec::new();
            if *target == START_BLOCK {
                incoming.push(entry_seed.clone());
            }
            incoming.extend(
                predecessors[*target]
                    .iter()
                    .filter_map(|predecessor| result.edge.get(&(*predecessor, *target)).cloned()),
            );
            let next_entry = BlockState::join_predecessors(incoming.iter());
            if result.entry[*target] != next_entry {
                result.entry[*target] = next_entry;
                worklist.insert(*target);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::{
        solve_cfg, BlockState, BoundValidation, MayJoin, StaticWriteToken,
    };
    use rustc_index::IndexVec;
    use rustc_middle::mir::BasicBlock;
    use std::collections::{BTreeMap, BTreeSet};

    type State = BlockState<String, BTreeSet<String>, String>;

    fn block(index: usize) -> BasicBlock {
        BasicBlock::from_usize(index)
    }

    fn graph(
        block_count: usize,
        edges: &[(usize, usize)],
    ) -> IndexVec<BasicBlock, BTreeSet<BasicBlock>> {
        let mut graph = IndexVec::from_elem_n(BTreeSet::new(), block_count);
        for (from, to) in edges {
            graph[block(*from)].insert(block(*to));
        }
        graph
    }

    #[test]
    fn diamond_join_keeps_may_values_and_must_validations_separate() {
        let mut left = State::empty();
        left.set_value("index".into(), BTreeSet::from(["arg1".into()]));
        left.establish(BoundValidation::new(
            "index-in-slice".into(),
            ["index".into(), "slice".into()],
        ));

        let mut right = State::empty();
        right.set_value("index".into(), BTreeSet::from(["constant".into()]));

        let joined = BlockState::join_predecessors([&left, &right]).unwrap();
        assert_eq!(
            joined.may_values()["index"],
            BTreeSet::from(["arg1".into(), "constant".into()])
        );
        assert!(joined.must_validations().is_empty());
    }

    #[test]
    fn may_join_is_idempotent() {
        let mut values = BTreeSet::from(["arg1"]);
        assert!(!values.join_may(&BTreeSet::from(["arg1"])));
    }

    #[test]
    fn relevant_write_kills_bound_validation_with_a_static_token() {
        let mut state = State::empty();
        state.establish(BoundValidation::new(
            "index-in-slice".into(),
            ["index".into(), "slice".into()],
        ));
        state.record_write("index".into(), StaticWriteToken::new(2, 4));
        assert!(state.must_validations().is_empty());
    }

    #[test]
    fn write_on_one_diamond_branch_prevents_must_validation_at_merge() {
        let validation = || {
            BoundValidation::new(
                "in-bounds".to_string(),
                ["index".to_string(), "slice".to_string()],
            )
        };
        let mut left = State::empty();
        left.establish(validation());
        let mut right = left.clone();
        right.record_write("index".into(), StaticWriteToken::new(2, 1));

        let joined = BlockState::join_predecessors([&left, &right]).unwrap();
        assert!(joined.must_validations().is_empty());
        assert_eq!(
            joined.versions()["index"],
            BTreeSet::from([StaticWriteToken::new(2, 1)])
        );
    }

    #[test]
    fn reapplying_one_write_site_is_idempotent_and_later_site_wins() {
        let mut state = State::empty();
        state.record_write("index".into(), StaticWriteToken::new(1, 2));
        let once = state.clone();
        state.record_write("index".into(), StaticWriteToken::new(1, 2));
        assert_eq!(state, once);
        state.record_write("index".into(), StaticWriteToken::new(3, 0));
        assert_eq!(
            state.versions()["index"],
            BTreeSet::from([StaticWriteToken::new(3, 0)])
        );
    }

    #[test]
    fn join_unions_finite_reaching_write_tokens() {
        let mut left = State::empty();
        left.record_write("index".into(), StaticWriteToken::new(1, 2));
        let mut right = State::empty();
        right.record_write("index".into(), StaticWriteToken::new(3, 1));
        let joined = BlockState::join_predecessors([&right, &left]).unwrap();
        assert_eq!(
            joined.versions()["index"],
            BTreeSet::from([
                StaticWriteToken::new(1, 2),
                StaticWriteToken::new(3, 1),
            ])
        );
    }

    #[test]
    fn entry_formals_are_seeded_and_unreachable_blocks_remain_none() {
        let successors = graph(3, &[(0, 1)]);
        let operations = IndexVec::from_elem_n(0, 3);
        let mut seed = State::empty();
        seed.set_value("arg1".into(), BTreeSet::from(["formal1".into()]));
        let result = solve_cfg(
            &successors,
            &operations,
            seed,
            |_, _, _| {},
            |_, _, _| {},
        );
        assert_eq!(
            result.entry[block(0)]
                .as_ref()
                .unwrap()
                .may_values()["arg1"],
            BTreeSet::from(["formal1".into()])
        );
        assert!(result.entry[block(2)].is_none());
    }

    #[test]
    fn diamond_solver_intersects_branch_validations() {
        let successors = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let operations = IndexVec::from_raw(vec![0, 1, 1, 0]);
        let result = solve_cfg(
            &successors,
            &operations,
            State::empty(),
            |bb, _, state| {
                state.set_value(
                    "index".into(),
                    BTreeSet::from([format!("branch{}", bb.index())]),
                );
                if bb == block(1) {
                    state.establish(BoundValidation::new(
                        "in-bounds".into(),
                        ["index".into(), "slice".into()],
                    ));
                }
            },
            |_, _, _| {},
        );
        let merged = result.entry[block(3)].as_ref().unwrap();
        assert_eq!(
            merged.may_values()["index"],
            BTreeSet::from(["branch1".into(), "branch2".into()])
        );
        assert!(merged.must_validations().is_empty());
    }

    #[test]
    fn edge_refinement_is_applied_only_to_the_selected_successor() {
        let successors = graph(3, &[(0, 1), (0, 2)]);
        let operations = IndexVec::from_elem_n(0, 3);
        let result = solve_cfg(
            &successors,
            &operations,
            State::empty(),
            |_, _, _| {},
            |from, to, state| {
                if from == block(0) && to == block(1) {
                    state.establish(BoundValidation::new(
                        "in-bounds".into(),
                        ["index".into(), "slice".into()],
                    ));
                }
            },
        );
        assert_eq!(
            result.entry[block(1)]
                .as_ref()
                .unwrap()
                .must_validations()
                .len(),
            1
        );
        assert!(result.entry[block(2)]
            .as_ref()
            .unwrap()
            .must_validations()
            .is_empty());
    }

    #[test]
    fn validations_reestablished_after_distinct_branch_writes_survive_join() {
        let validation = || {
            BoundValidation::new(
                "in-bounds".to_string(),
                ["index".to_string(), "slice".to_string()],
            )
        };
        let mut left = State::empty();
        left.record_write("index".into(), StaticWriteToken::new(1, 0));
        left.establish(validation());
        let mut right = State::empty();
        right.record_write("index".into(), StaticWriteToken::new(2, 0));
        right.establish(validation());

        let joined = BlockState::join_predecessors([&left, &right]).unwrap();
        assert_eq!(joined.must_validations().len(), 1);
        assert!(joined.has_current_validation(
            &"in-bounds".to_string(),
            ["index".to_string(), "slice".to_string()]
        ));
        assert_eq!(
            joined.versions()["index"],
            BTreeSet::from([
                StaticWriteToken::new(1, 0),
                StaticWriteToken::new(2, 0),
            ])
        );
    }

    #[test]
    fn current_validation_query_rejects_stale_or_wrong_bindings() {
        let mut state = State::empty();
        state.establish(BoundValidation::new(
            "in-bounds".to_string(),
            ["index".to_string(), "slice".to_string()],
        ));
        assert!(state.has_current_validation(
            &"in-bounds".to_string(),
            ["index".to_string(), "slice".to_string()]
        ));
        assert!(!state.has_current_validation(
            &"nonnull".to_string(),
            ["index".to_string(), "slice".to_string()]
        ));
        assert!(!state.has_current_validation(
            &"in-bounds".to_string(),
            ["other".to_string(), "slice".to_string()]
        ));
        state.record_write("index".into(), StaticWriteToken::new(4, 2));
        assert!(!state.has_current_validation(
            &"in-bounds".to_string(),
            ["index".to_string(), "slice".to_string()]
        ));
    }

    #[test]
    fn loop_with_one_static_write_token_reaches_equality() {
        let successors = graph(3, &[(0, 1), (1, 1), (1, 2)]);
        let operations = IndexVec::from_raw(vec![0, 1, 0]);
        let mut transfers = 0_u32;
        let result = solve_cfg(
            &successors,
            &operations,
            State::empty(),
            |bb, _, state| {
                transfers += 1;
                state.record_write(
                    "index".into(),
                    StaticWriteToken::new(bb.index() as u32, 0),
                );
            },
            |_, _, _| {},
        );
        assert_eq!(
            result.exit[block(1)]
                .as_ref()
                .unwrap()
                .versions()["index"],
            BTreeSet::from([StaticWriteToken::new(1, 0)])
        );
        assert!(transfers < 10, "the finite loop should stabilize promptly");
    }

    #[test]
    fn worklist_result_is_independent_of_edge_insertion_order() {
        let solve = |edges: &[(usize, usize)]| {
            let successors = graph(4, edges);
            let operations = IndexVec::from_raw(vec![1, 1, 1, 0]);
            solve_cfg(
                &successors,
                &operations,
                State::empty(),
                |bb, _, state| {
                    state.set_value(
                        "seen".into(),
                        BTreeSet::from([bb.index().to_string()]),
                    );
                },
                |_, _, _| {},
            )
        };
        let forward = solve(&[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let reverse = solve(&[(2, 3), (1, 3), (0, 2), (0, 1)]);
        assert_eq!(forward, reverse);

        let before_keys = forward.before.keys().copied().collect::<Vec<_>>();
        let mut sorted = before_keys.clone();
        sorted.sort();
        assert_eq!(before_keys, sorted);
    }

    #[test]
    fn predecessor_join_is_monotone_and_idempotent() {
        let mut first = State::empty();
        first.set_value("x".into(), BTreeSet::from(["a".into()]));
        first.establish(BoundValidation::new(
            "in-bounds".into(),
            ["x".into(), "slice".into()],
        ));
        let once = BlockState::join_predecessors([&first]).unwrap();
        let twice = BlockState::join_predecessors([&once, &once]).unwrap();
        assert_eq!(once, twice);

        let mut second = State::empty();
        second.set_value("x".into(), BTreeSet::from(["b".into()]));
        second.record_write("x".into(), StaticWriteToken::new(4, 0));
        let expanded = BlockState::join_predecessors([&once, &second]).unwrap();
        assert!(expanded.may_values()["x"].is_superset(&once.may_values()["x"]));
        assert!(expanded
            .must_validations()
            .is_subset(once.must_validations()));
        assert!(expanded.versions()["x"].contains(&StaticWriteToken::new(4, 0)));
        assert_eq!(
            BlockState::<String, BTreeSet<String>, String>::join_predecessors([]),
            None
        );
    }

    #[test]
    fn equal_semantic_validations_from_distinct_paths_survive_intersection() {
        let validation = || {
            BoundValidation::new(
                "in-bounds".to_string(),
                ["index".to_string(), "slice".to_string()],
            )
        };
        let mut left = State::empty();
        left.establish(validation());
        let mut right = State::empty();
        right.establish(validation());
        let joined = BlockState::join_predecessors([&left, &right]).unwrap();
        assert_eq!(joined.must_validations().len(), 1);

        let expected_versions = BTreeMap::<String, BTreeSet<StaticWriteToken>>::new();
        assert_eq!(joined.versions(), &expected_versions);
    }
}
