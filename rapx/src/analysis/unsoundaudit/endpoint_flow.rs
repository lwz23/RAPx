use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::summary::{FunctionKey, ProgramPoint, StableSpan};
use super::unsafe_site::OperandRole;

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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum EndpointNode {
    Formal {
        owner: FunctionKey,
        index: u32,
    },
    PublicField {
        def_path: String,
        span: StableSpan,
    },
    ExternalBehavior {
        owner: FunctionKey,
        kind: ExternalSourceKind,
        token: String,
        span: StableSpan,
    },
    InternalDerivation {
        owner: FunctionKey,
        token: String,
        point: ProgramPoint,
    },
    GenericCapability {
        owner: FunctionKey,
        token: String,
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
    ActualToFormal,
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

    pub fn canonical_route(
        &self,
        source: &EndpointNode,
        target: &EndpointNode,
    ) -> Option<Vec<EndpointEdge>> {
        if source == target {
            return self.nodes.contains(source).then(Vec::new);
        }

        let mut outgoing = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
        let mut incoming = BTreeMap::<&EndpointNode, Vec<&EndpointEdge>>::new();
        for edge in &self.edges {
            outgoing.entry(&edge.from).or_default().push(edge);
            incoming.entry(&edge.to).or_default().push(edge);
        }
        let distance_from_source = graph_distances(source, &outgoing, false);
        let distance_to_target = graph_distances(target, &incoming, true);
        let distance = *distance_from_source.get(target)?;
        let mut route = Vec::with_capacity(distance);
        let mut cursor = source;
        for step in 0..distance {
            let edge = outgoing.get(cursor)?.iter().copied().find(|edge| {
                distance_from_source.get(&edge.to) == Some(&(step + 1))
                    && distance_to_target.get(&edge.to) == Some(&(distance - step - 1))
            })?;
            route.push(edge.clone());
            cursor = &edge.to;
        }
        Some(route)
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
            | EndpointNode::Return { owner }
            | EndpointNode::Out { owner, .. }
            | EndpointNode::UnsafeOperand { owner, .. }
            | EndpointNode::UnsafeResult { owner, .. }
            | EndpointNode::InternalDerivation { owner, .. }
            | EndpointNode::GenericCapability { owner, .. }
            | EndpointNode::ExternalBehavior { owner, .. }
            | EndpointNode::Exposure { owner, .. } => owner.clone(),
            EndpointNode::PublicField { .. } => FunctionKey::new("crate::field"),
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
        let other = EndpointFlowGraph {
            nodes: BTreeSet::new(),
            edges: BTreeSet::from([second_edge.clone()]),
        };
        assert!(graph.join(&other));
        assert!(!graph.join(&other));
        assert_eq!(graph.edges, BTreeSet::from([first_edge, second_edge]));
    }
}
