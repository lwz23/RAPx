use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::contract_catalog::{ContractDescriptor, FailurePhase, StructuralSiteCandidate};
use super::endpoint_flow::{EndpointEdge, EndpointFlowGraph, EndpointNode};
use super::summary::{FunctionKey, ProgramPoint, StableSpan};
use super::unsafe_site::{
    binding::BoundUnsafeSite, CausalSourceKind, ClassificationConfidence, OperandRole,
    PrimaryPattern,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CandidateOwner {
    pub function: FunctionKey,
    pub root_span: StableSpan,
    pub public_safe: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CandidateCallEdge {
    pub caller: FunctionKey,
    pub callee: FunctionKey,
    pub point: ProgramPoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CandidateExposure {
    pub node: EndpointNode,
    pub span: StableSpan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StructuralFindingCandidate {
    pub primary_pattern: PrimaryPattern,
    pub mechanism_id: String,
    pub classification_confidence: ClassificationConfidence,
    pub public_root: FunctionKey,
    pub public_root_span: StableSpan,
    pub call_route: Vec<CandidateCallEdge>,
    pub source: EndpointNode,
    pub source_kind: CausalSourceKind,
    pub source_span: StableSpan,
    pub first_failure: StableSpan,
    pub site: BoundUnsafeSite,
    pub operand_role: Option<OperandRole>,
    pub exposure: Option<CandidateExposure>,
    pub contract: ContractDescriptor,
    pub endpoint_route: Vec<EndpointEdge>,
}

pub(super) fn build_structural_candidates(
    graph: &EndpointFlowGraph,
    structural_sites: &BTreeMap<String, StructuralSiteCandidate>,
    contracts: &BTreeMap<String, ContractDescriptor>,
    bound_sites: &BTreeMap<String, BoundUnsafeSite>,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
) -> Vec<StructuralFindingCandidate> {
    let mut result = Vec::new();
    for (site_id, structural) in structural_sites {
        let Some(site) = bound_sites.get(site_id) else {
            continue;
        };
        let Some(contract) = contracts.get(site_id) else {
            continue;
        };

        if let Some(mut exposures) = p3_exposures(graph, site, structural, contract, owners, calls)
        {
            result.append(&mut exposures);
            continue;
        }

        let mut roles = structural.relevant_roles.clone();
        roles.extend(contract.relevant_roles.iter().cloned());
        let mut classified = Vec::new();
        for role in roles {
            let target = EndpointNode::UnsafeOperand {
                owner: site.seed.owner.clone(),
                site_id: site_id.clone(),
                role: role.clone(),
            };
            if !graph.nodes.contains(&target) {
                continue;
            }
            for source in &graph.nodes {
                let Some((pattern, source_kind, mechanism, rank)) = source_classification(source)
                else {
                    continue;
                };
                let Some(endpoint_route) = graph.canonical_route(source, &target) else {
                    continue;
                };
                let roots = candidate_roots(source, &site.seed.owner, owners, calls);
                let Some(source_span) = node_span(source, owners) else {
                    continue;
                };
                for (root, root_span, call_route) in roots {
                    classified.push((
                        rank,
                        StructuralFindingCandidate {
                            primary_pattern: pattern,
                            mechanism_id: mechanism.to_string(),
                            classification_confidence: ClassificationConfidence::Structural,
                            public_root: root,
                            public_root_span: root_span,
                            call_route,
                            source: source.clone(),
                            source_kind,
                            source_span: source_span.clone(),
                            first_failure: site.seed.raw_span.stable.clone(),
                            site: site.clone(),
                            operand_role: Some(role.clone()),
                            exposure: None,
                            contract: contract.clone(),
                            endpoint_route: endpoint_route.clone(),
                        },
                    ));
                }
            }
        }
        let Some(best_rank) = classified.iter().map(|(rank, _)| *rank).min() else {
            continue;
        };
        result.extend(
            classified
                .into_iter()
                .filter(|(rank, _)| *rank == best_rank)
                .map(|(_, candidate)| candidate),
        );
    }
    result.sort_by_key(candidate_key);
    result.dedup_by(|left, right| candidate_key(left) == candidate_key(right));
    result
}

fn p3_exposures(
    graph: &EndpointFlowGraph,
    site: &BoundUnsafeSite,
    structural: &StructuralSiteCandidate,
    contract: &ContractDescriptor,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
) -> Option<Vec<StructuralFindingCandidate>> {
    if site.result_place.is_none()
        || (!structural.validity_sensitive_result
            && contract.failure_phase != FailurePhase::ImmediateValidity)
    {
        return None;
    }
    let source = EndpointNode::UnsafeResult {
        owner: site.seed.owner.clone(),
        site_id: site.seed.site_id(),
    };
    if !graph.nodes.contains(&source) {
        return None;
    }
    let mut result = Vec::new();
    for exposure_node in &graph.nodes {
        let exposure_owner = match exposure_node {
            EndpointNode::Return { owner } | EndpointNode::Out { owner, .. } => {
                if !owners.get(owner).is_some_and(|info| info.public_safe) {
                    continue;
                }
                owner
            }
            EndpointNode::Exposure { owner, .. } => owner,
            _ => continue,
        };
        let Some(endpoint_route) = graph.canonical_route(&source, exposure_node) else {
            continue;
        };
        let Some(exposure_span) = node_span(exposure_node, owners) else {
            continue;
        };
        let roots = match exposure_node {
            EndpointNode::Return { owner } | EndpointNode::Out { owner, .. }
                if owners.get(owner).is_some_and(|info| info.public_safe) =>
            {
                vec![(owner.clone(), owners[owner].root_span.clone(), Vec::new())]
            }
            _ => reachable_public_roots(exposure_owner, owners, calls),
        };
        for (root, root_span, call_route) in roots {
            result.push(StructuralFindingCandidate {
                primary_pattern: PrimaryPattern::P3,
                mechanism_id: "p3_internal_unsafe_exposure_v1".to_string(),
                classification_confidence: ClassificationConfidence::Structural,
                public_root: root,
                public_root_span: root_span,
                call_route,
                source: source.clone(),
                source_kind: CausalSourceKind::InternalUnsafeOrigin,
                source_span: site.seed.raw_span.stable.clone(),
                first_failure: site.seed.raw_span.stable.clone(),
                site: site.clone(),
                operand_role: None,
                exposure: Some(CandidateExposure {
                    node: exposure_node.clone(),
                    span: exposure_span.clone(),
                }),
                contract: contract.clone(),
                endpoint_route: endpoint_route.clone(),
            });
        }
    }
    (!result.is_empty()).then_some(result)
}

fn source_classification(
    source: &EndpointNode,
) -> Option<(PrimaryPattern, CausalSourceKind, &'static str, u8)> {
    match source {
        EndpointNode::ExternalBehavior { .. } => Some((
            PrimaryPattern::P6,
            CausalSourceKind::ExternalBehaviorOutput,
            "p6_external_behavior_flow_v1",
            0,
        )),
        EndpointNode::GenericCapability { .. } => Some((
            PrimaryPattern::P5,
            CausalSourceKind::GenericCapabilityGap,
            "p5_generic_capability_gap_v1",
            1,
        )),
        EndpointNode::PublicField { .. } => Some((
            PrimaryPattern::P2,
            CausalSourceKind::PublicField,
            "p2_public_state_flow_v1",
            2,
        )),
        EndpointNode::InternalDerivation { .. } => Some((
            PrimaryPattern::P4,
            CausalSourceKind::InternalDerivation,
            "p4_internal_derivation_v1",
            3,
        )),
        EndpointNode::Formal { .. } => Some((
            PrimaryPattern::P1,
            CausalSourceKind::PublicParameter,
            "p1_public_parameter_flow_v1",
            4,
        )),
        EndpointNode::UnsafeOperand { .. }
        | EndpointNode::UnsafeResult { .. }
        | EndpointNode::Return { .. }
        | EndpointNode::Out { .. }
        | EndpointNode::Exposure { .. } => None,
    }
}

fn candidate_roots(
    source: &EndpointNode,
    site_owner: &FunctionKey,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
) -> Vec<(FunctionKey, StableSpan, Vec<CandidateCallEdge>)> {
    if let EndpointNode::Formal { owner, .. } = source {
        if owners.get(owner).is_some_and(|info| info.public_safe) {
            let route = canonical_call_route(owner, site_owner, calls);
            return route
                .map(|route| vec![(owner.clone(), owners[owner].root_span.clone(), route)])
                .unwrap_or_default();
        }
        return Vec::new();
    }
    reachable_public_roots(site_owner, owners, calls)
}

fn reachable_public_roots(
    target: &FunctionKey,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
    calls: &BTreeSet<CandidateCallEdge>,
) -> Vec<(FunctionKey, StableSpan, Vec<CandidateCallEdge>)> {
    owners
        .values()
        .filter(|owner| owner.public_safe)
        .filter_map(|owner| {
            canonical_call_route(&owner.function, target, calls)
                .map(|route| (owner.function.clone(), owner.root_span.clone(), route))
        })
        .collect()
}

fn canonical_call_route(
    root: &FunctionKey,
    target: &FunctionKey,
    calls: &BTreeSet<CandidateCallEdge>,
) -> Option<Vec<CandidateCallEdge>> {
    if root == target {
        return Some(Vec::new());
    }
    let mut outgoing = BTreeMap::<FunctionKey, Vec<&CandidateCallEdge>>::new();
    for edge in calls {
        outgoing.entry(edge.caller.clone()).or_default().push(edge);
    }
    let mut visited = BTreeSet::from([root.clone()]);
    let mut queue = VecDeque::from([(root.clone(), Vec::new())]);
    while let Some((owner, route)) = queue.pop_front() {
        for edge in outgoing.get(&owner).into_iter().flatten() {
            if !visited.insert(edge.callee.clone()) {
                continue;
            }
            let mut next = route.clone();
            next.push((*edge).clone());
            if &edge.callee == target {
                return Some(next);
            }
            queue.push_back((edge.callee.clone(), next));
        }
    }
    None
}

fn node_span(
    node: &EndpointNode,
    owners: &BTreeMap<FunctionKey, CandidateOwner>,
) -> Option<StableSpan> {
    match node {
        EndpointNode::Formal { owner, .. }
        | EndpointNode::Return { owner }
        | EndpointNode::Out { owner, .. }
        | EndpointNode::UnsafeOperand { owner, .. }
        | EndpointNode::UnsafeResult { owner, .. } => {
            owners.get(owner).map(|owner| owner.root_span.clone())
        }
        EndpointNode::PublicField { span, .. }
        | EndpointNode::ExternalBehavior { span, .. }
        | EndpointNode::GenericCapability { span, .. } => Some(span.clone()),
        EndpointNode::InternalDerivation { point, .. } | EndpointNode::Exposure { point, .. } => {
            Some(point.span.clone())
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
        candidate.site.seed.site_id(),
        candidate.source,
        role,
        exposure
    )
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
        }
    }

    fn owner_info(owner: &FunctionKey, public_safe: bool) -> CandidateOwner {
        CandidateOwner {
            function: owner.clone(),
            root_span: span(1),
            public_safe,
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
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        )
        .pop()
        .unwrap()
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
                    def_path: "crate::State::field".into(),
                    span: span(2),
                },
                PrimaryPattern::P2,
            ),
            (
                EndpointNode::InternalDerivation {
                    owner: owner.clone(),
                    token: "offset".into(),
                    point: point(&owner, 2),
                },
                PrimaryPattern::P4,
            ),
            (
                EndpointNode::GenericCapability {
                    owner: owner.clone(),
                    token: "T: nonempty".into(),
                    span: span(2),
                },
                PrimaryPattern::P5,
            ),
            (
                EndpointNode::ExternalBehavior {
                    owner: owner.clone(),
                    kind: ExternalSourceKind::Callback,
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
                token: "derived".into(),
                point: point(&owner, 2),
            },
            EndpointNode::PublicField {
                def_path: "crate::State::field".into(),
                span: span(3),
            },
            EndpointNode::GenericCapability {
                owner: owner.clone(),
                token: "T".into(),
                span: span(4),
            },
            EndpointNode::ExternalBehavior {
                owner: owner.clone(),
                kind: ExternalSourceKind::Iterator,
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
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &BTreeMap::from([(owner.clone(), owner_info(&owner, true))]),
            &BTreeSet::new(),
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].primary_pattern, PrimaryPattern::P4);
        assert_eq!(candidates[0].endpoint_route.len(), 1);
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
    fn private_sink_uses_each_real_public_call_root_without_a_depth_cap() {
        let functions = (0..=64)
            .map(|index| FunctionKey::new(format!("crate::f{index}")))
            .collect::<Vec<_>>();
        let site = bound_site(&functions[64], None);
        let site_id = site.seed.site_id();
        let source = EndpointNode::InternalDerivation {
            owner: functions[64].clone(),
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
            &BTreeMap::from([(site_id.clone(), descriptor(FailurePhase::Precondition))]),
            &BTreeMap::from([(site_id, site)]),
            &owners,
            &calls,
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].call_route.len(), 64);
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
