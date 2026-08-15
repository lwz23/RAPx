use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::{json, Value};

use super::candidate::StructuralFindingCandidate;
use super::contract_catalog::{ContractDescriptor, FailurePhase};
use super::endpoint_flow::{EndpointFlowGraph, EndpointNode, ExternalSourceKind};
use super::mir::AnalysisOutput;
use super::summary::{sha256_hex, FunctionKey, StableSpan};
use super::unsafe_site::binding::{BoundUnsafeSite, SiteBindingRecord};
use super::unsafe_site::{
    self, BindingStatus, CausalSourceKind, ClassificationConfidence, ContractPrecision,
    CoverageFailure, CoverageFailureKind, OperandRole, PrimaryPattern, UnsafeSiteKind,
    UnsafeSiteSeed,
};
use super::UnitMetadata;

pub(super) fn build_receipt(
    metadata: UnitMetadata,
    analysis: AnalysisOutput,
) -> Result<Value, String> {
    let adapter = unsafe_site::adapter_for_commit(&metadata.rap_compiler_commit)?;
    adapter.validate()?;
    let snapshot = unsafe_site::census_snapshot()?;
    let sites = census_sites(&snapshot.owners)?;
    let failures = snapshot.failures.iter().cloned().collect::<Vec<_>>();
    let status = if failures.is_empty() {
        "complete"
    } else {
        "partial"
    };

    let source_root = selected_source_root(&metadata.project_root)?;
    let code = build_code_dictionary(
        &source_root,
        &analysis.structural_candidates,
        &analysis.owner_code_spans,
    )?;
    let mut findings = build_findings(
        &analysis.structural_candidates,
        &analysis.endpoint_graph,
        &analysis.owner_spans,
        &code.owner_code_ids,
        &code.safety_doc_ids,
    )?;
    findings.sort_by_key(|finding| {
        finding["finding_id"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    });
    findings.dedup_by(|left, right| left["finding_id"] == right["finding_id"]);

    let finding_counts = finding_counts_by_site(&findings);
    let site_records = build_site_records(
        &sites,
        &analysis.binding_records,
        &failures,
        &finding_counts,
    )?;
    let census = census_json(&sites, &site_records);
    let coverage_failures = failures
        .iter()
        .map(coverage_failure_json)
        .collect::<Vec<_>>();
    let counts = pattern_counts(&findings)?;

    let mut emitted_kinds = adapter
        .emitted_kinds
        .iter()
        .map(|kind| kind.as_str())
        .collect::<Vec<_>>();
    emitted_kinds.sort();
    let manifest_path =
        normalize_metadata_path(&metadata.manifest_path, Path::new(&metadata.project_root))?;
    let source_path =
        normalize_metadata_path(&metadata.source_path, Path::new(&metadata.project_root))?;
    let rustc_invocation_id = semantic_rustc_invocation_id(&metadata, &manifest_path, &source_path);

    Ok(json!({
        "schema_version": "rap-unit-v3",
        "source": "rap",
        "success": true,
        "analysis_status": status,
        "unit_id": require_nonempty(&metadata.unit_id, "unit id")?,
        "rustc_invocation_id": rustc_invocation_id,
        "package_id": require_nonempty(&metadata.package_id, "package id")?,
        "package_name": require_nonempty(&metadata.package_name, "package name")?,
        "package_version": require_nonempty(&metadata.package_version, "package version")?,
        "crate_name": require_nonempty(&metadata.crate_name, "crate name")?,
        "crate_types": metadata.crate_types,
        "target_kind": require_nonempty(&metadata.target_kind, "target kind")?,
        "target_triple": require_nonempty(&metadata.target_triple, "target triple")?,
        "manifest_path": manifest_path,
        "source_path": source_path,
        "cargo_supplied_rustc_commit": metadata.cargo_supplied_rustc_commit,
        "rap_compiler_commit": metadata.rap_compiler_commit,
        "rustc_commit": adapter.rustc_commit,
        "toolchain_adapter": {
            "adapter_id": adapter.adapter_id,
            "rustc_commit": adapter.rustc_commit,
            "checker_sha256": adapter.checker_sha256,
            "emitted_kinds": emitted_kinds,
        },
        "unsafe_site_census": census,
        "unsafe_sites": site_records,
        "coverage_failures": coverage_failures,
        "observed_counts": {
            "semantics": if status == "complete" { "exact" } else { "lower_bound" },
            "pattern1": counts[0],
            "pattern2": counts[1],
            "pattern3": counts[2],
            "pattern4": counts[3],
            "pattern5": counts[4],
            "pattern6": counts[5],
        },
        "finding_count": findings.len(),
        "findings": findings,
        "code_dictionary": code.blocks,
    }))
}

fn semantic_rustc_invocation_id(
    metadata: &UnitMetadata,
    manifest_path: &str,
    source_path: &str,
) -> String {
    let crate_types = metadata.crate_types.join(",");
    let digest = sha256_hex(&length_prefixed(&[
        &metadata.package_name,
        &metadata.package_version,
        &metadata.crate_name,
        &crate_types,
        &metadata.target_kind,
        &metadata.target_triple,
        manifest_path,
        source_path,
        &metadata.extra_filename,
    ]));
    format!("rustc-{}", &digest[..16])
}

fn census_sites(
    owners: &BTreeMap<FunctionKey, unsafe_site::OwnerCensus>,
) -> Result<BTreeMap<String, UnsafeSiteSeed>, String> {
    let mut sites = BTreeMap::new();
    for owner in owners.values() {
        owner.validate()?;
        for site in &owner.sites {
            let site_id = site.site_id();
            if sites.insert(site_id.clone(), site.clone()).is_some() {
                return Err(format!("duplicate unsafe site id in census: {site_id}"));
            }
        }
    }
    Ok(sites)
}

fn build_site_records(
    sites: &BTreeMap<String, UnsafeSiteSeed>,
    records: &BTreeMap<String, SiteBindingRecord>,
    failures: &[CoverageFailure],
    finding_counts: &BTreeMap<String, usize>,
) -> Result<Vec<Value>, String> {
    let failures_by_site = failures
        .iter()
        .filter_map(|failure| {
            failure
                .site_id
                .as_ref()
                .map(|site_id| (site_id.clone(), failure))
        })
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::with_capacity(sites.len());
    for (site_id, site) in sites {
        let record = records
            .get(site_id)
            .ok_or_else(|| format!("unsafe site {site_id} lacks a terminal binding record"))?;
        if record.site_id != *site_id {
            return Err("unsafe-site binding record key differs from its site id".to_string());
        }
        let candidate_count = if record.status == BindingStatus::BoundExact {
            finding_counts.get(site_id).copied().unwrap_or(0)
        } else {
            failures_by_site
                .get(site_id)
                .ok_or_else(|| format!("non-bound unsafe site {site_id} lacks a coverage failure"))?
                .candidate_count
        };
        result.push(json!({
            "site_id": site_id,
            "kind": site.kind.as_str(),
            "owner": site.owner.0,
            "span": span_json(&site.raw_span.stable),
            "binding": binding_status(record.status),
            "candidate_count": candidate_count,
        }));
    }
    Ok(result)
}

fn census_json(sites: &BTreeMap<String, UnsafeSiteSeed>, records: &[Value]) -> Value {
    let mut status_counts = BTreeMap::<&str, usize>::new();
    for record in records {
        *status_counts
            .entry(record["binding"].as_str().unwrap_or_default())
            .or_default() += 1;
    }
    let mut by_kind = BTreeMap::<&str, usize>::new();
    for site in sites.values() {
        *by_kind.entry(site.kind.as_str()).or_default() += 1;
    }
    json!({
        "discovered": sites.len(),
        "bound_exact": status_counts.get("bound_exact").copied().unwrap_or(0),
        "unbound": status_counts.get("unbound").copied().unwrap_or(0),
        "ambiguous": status_counts.get("ambiguous").copied().unwrap_or(0),
        "elided": status_counts.get("elided").copied().unwrap_or(0),
        "by_kind": by_kind.into_iter().map(|(kind, count)| json!({
            "kind": kind,
            "count": count,
        })).collect::<Vec<_>>(),
    })
}

fn build_findings(
    candidates: &[StructuralFindingCandidate],
    graph: &EndpointFlowGraph,
    owner_spans: &BTreeMap<FunctionKey, StableSpan>,
    code_ids: &BTreeMap<FunctionKey, String>,
    safety_doc_ids: &BTreeMap<FunctionKey, String>,
) -> Result<Vec<Value>, String> {
    candidates
        .iter()
        .map(|candidate| finding_json(candidate, graph, owner_spans, code_ids, safety_doc_ids))
        .collect()
}

fn finding_json(
    candidate: &StructuralFindingCandidate,
    graph: &EndpointFlowGraph,
    owner_spans: &BTreeMap<FunctionKey, StableSpan>,
    code_ids: &BTreeMap<FunctionKey, String>,
    safety_doc_ids: &BTreeMap<FunctionKey, String>,
) -> Result<Value, String> {
    let root_code = code_ids.get(&candidate.public_root).ok_or_else(|| {
        format!(
            "missing code dictionary entry for public root {}",
            candidate.public_root.0
        )
    })?;
    let source_identity = endpoint_identity(&candidate.source);
    let failure_kind = failure_kind(&candidate.contract, candidate.site.seed.kind);
    let exposure_identity = candidate
        .exposure
        .as_ref()
        .map(|exposure| endpoint_identity(&exposure.node));
    let finding_id = finding_id(
        candidate.primary_pattern,
        &candidate.site.seed.site_id(),
        &source_identity,
        &failure_kind,
        &candidate.first_failure,
        &candidate.public_root,
        exposure_identity.as_deref(),
    );
    let propagation = propagation_json(candidate, graph, owner_spans);
    let mut code_refs = finding_owners(candidate)
        .into_iter()
        .filter_map(|owner| code_ids.get(&owner).cloned())
        .collect::<Vec<_>>();
    if let Some(code_id) = candidate
        .site
        .seed
        .callee
        .as_ref()
        .and_then(|callee| code_ids.get(&FunctionKey::new(callee.clone())))
    {
        code_refs.push(code_id.clone());
    }
    let safety_doc_id = candidate
        .site
        .seed
        .callee
        .as_ref()
        .and_then(|callee| safety_doc_ids.get(&FunctionKey::new(callee.clone())));
    if let Some(code_id) = safety_doc_id {
        code_refs.push(code_id.clone());
    }
    code_refs.sort();
    code_refs.dedup();
    if !code_refs.contains(root_code) {
        code_refs.push(root_code.clone());
        code_refs.sort();
    }

    Ok(json!({
        "finding_id": finding_id,
        "primary_pattern": primary_pattern(candidate.primary_pattern),
        "mechanism_id": candidate.mechanism_id,
        "classification_confidence": classification_confidence(candidate.classification_confidence),
        "public_safe_root": {
            "owner": candidate.public_root.0,
            "span": span_json(&candidate.public_root_span),
            "code_id": root_code,
        },
        "source": {
            "kind": source_kind(candidate.source_kind),
            "identity": source_identity,
            "span": span_json(&candidate.source_span),
        },
        "first_contract_failure": {
            "kind": failure_kind,
            "span": span_json(&candidate.first_failure),
        },
        "unsafe_site": unsafe_site_json(&candidate.site),
        "exposure": candidate.exposure.as_ref().map(|exposure| json!({
            "kind": exposure_kind(&exposure.node),
            "span": span_json(&exposure.span),
        })),
        "contract": contract_json(&candidate.contract, &candidate.site, safety_doc_id),
        "propagation": propagation,
        "code_refs": code_refs,
    }))
}

fn finding_id(
    pattern: PrimaryPattern,
    site_id: &str,
    source_identity: &str,
    failure_kind: &str,
    failure_span: &StableSpan,
    root: &FunctionKey,
    exposure_identity: Option<&str>,
) -> String {
    sha256_hex(&length_prefixed(&[
        primary_pattern(pattern),
        site_id,
        source_identity,
        failure_kind,
        &failure_span.token(),
        &root.0,
        exposure_identity.unwrap_or(""),
    ]))
}

fn unsafe_site_json(site: &BoundUnsafeSite) -> Value {
    let mut operands = site
        .operands
        .iter()
        .map(|operand| operand_role(&operand.role))
        .collect::<Vec<_>>();
    operands.sort();
    operands.dedup();
    json!({
        "site_id": site.seed.site_id(),
        "kind": site.seed.kind.as_str(),
        "owner": site.seed.owner.0,
        "span": span_json(&site.seed.raw_span.stable),
        "callee": site.seed.callee,
        "operands": operands,
        "destination": site.result_place.as_ref().map(|place| place.0.clone()),
        "binding": "bound_exact",
    })
}

fn contract_json(
    contract: &ContractDescriptor,
    site: &BoundUnsafeSite,
    safety_doc_id: Option<&String>,
) -> Value {
    let mut roles = contract
        .relevant_roles
        .iter()
        .map(operand_role)
        .collect::<Vec<_>>();
    if roles.is_empty() {
        roles.push(
            site.operands
                .first()
                .map(|operand| operand_role(&operand.role))
                .unwrap_or_else(|| "result".to_string()),
        );
    }
    roles.sort();
    roles.dedup();
    let obligation = (contract.precision != ContractPrecision::Opaque).then(|| {
        json!({
            "predicates": contract.predicates.iter().map(|predicate| predicate.as_str()).collect::<Vec<_>>()
        })
    });
    let reason = if site.seed.callee.is_some() {
        "unsafe callee has no local adjacent # Safety documentation in the selected source root"
    } else {
        "language-level unsafe site has no callable Safety documentation"
    };
    json!({
        "precision": contract_precision(contract.precision),
        "model_id": contract.model_id,
        "relevant_roles": roles,
        "obligation": obligation,
        "safety_doc": {
            "status": if safety_doc_id.is_some() { "available" } else { "unavailable" },
            "code_id": safety_doc_id,
            "reason": if safety_doc_id.is_some() { Value::Null } else { Value::String(reason.to_string()) },
        },
    })
}

fn propagation_json(
    candidate: &StructuralFindingCandidate,
    graph: &EndpointFlowGraph,
    owner_spans: &BTreeMap<FunctionKey, StableSpan>,
) -> Value {
    let mut witness = vec![witness_step(
        "entry",
        &candidate.public_root,
        &candidate.public_root_span,
    )];
    let source_owner = endpoint_owner(&candidate.source);
    let source_before_calls = source_owner
        .as_ref()
        .is_some_and(|owner| owner == &candidate.public_root);
    if candidate.primary_pattern != PrimaryPattern::P3 && source_before_calls {
        witness.push(witness_step(
            "source",
            source_owner.as_ref().unwrap_or(&candidate.public_root),
            &candidate.source_span,
        ));
    }
    for edge in &candidate.call_route {
        witness.push(witness_step("local_call", &edge.callee, &edge.point.span));
    }
    if candidate.primary_pattern != PrimaryPattern::P3 && !source_before_calls {
        witness.push(witness_step(
            "source",
            source_owner.as_ref().unwrap_or(&candidate.site.seed.owner),
            &candidate.source_span,
        ));
    }

    let (recursive, cycle_step) = selected_cycle(candidate, graph, owner_spans);
    if let Some(cycle_step) = cycle_step {
        witness.push(cycle_step);
    }
    witness.push(witness_step(
        "unsafe_site",
        &candidate.site.seed.owner,
        &candidate.site.seed.raw_span.stable,
    ));
    if let Some(exposure) = &candidate.exposure {
        witness.push(witness_step(
            "exposure",
            endpoint_owner(&exposure.node)
                .as_ref()
                .unwrap_or(&candidate.public_root),
            &exposure.span,
        ));
    }

    let mut boundaries = candidate
        .call_route
        .iter()
        .flat_map(|edge| [edge.caller.0.clone(), edge.callee.0.clone()])
        .collect::<Vec<_>>();
    boundaries.sort();
    boundaries.dedup();
    let depth = if recursive {
        "recursive"
    } else if !candidate.call_route.is_empty()
        || candidate
            .endpoint_route
            .iter()
            .any(|edge| endpoint_owner(&edge.from) != endpoint_owner(&edge.to))
    {
        "inter_procedural"
    } else {
        "intra_procedural"
    };
    json!({
        "depth": depth,
        "boundaries": boundaries,
        "witness": witness,
    })
}

fn selected_cycle(
    candidate: &StructuralFindingCandidate,
    graph: &EndpointFlowGraph,
    owner_spans: &BTreeMap<FunctionKey, StableSpan>,
) -> (bool, Option<Value>) {
    let route_nodes = std::iter::once(&candidate.source)
        .chain(
            candidate
                .endpoint_route
                .iter()
                .flat_map(|edge| [&edge.from, &edge.to]),
        )
        .collect::<BTreeSet<_>>();
    for component in graph.cyclic_components() {
        if component.iter().all(|node| !route_nodes.contains(node)) {
            continue;
        }
        if let Some(edge) = graph.edges.iter().find(|edge| {
            component.contains(&edge.from) && component.contains(&edge.to) && edge.point.is_some()
        }) {
            let point = edge.point.as_ref().unwrap();
            return (
                true,
                Some(witness_step("scc_cycle", &point.function, &point.span)),
            );
        }
        if let Some(owner) = component.iter().find_map(endpoint_owner) {
            if let Some(span) = owner_spans.get(&owner) {
                return (true, Some(witness_step("scc_cycle", &owner, span)));
            }
        }
    }
    (false, None)
}

fn witness_step(kind: &str, owner: &FunctionKey, span: &StableSpan) -> Value {
    json!({
        "kind": kind,
        "owner": owner.0,
        "span": span_json(span),
    })
}

struct CodeDictionary {
    blocks: Vec<Value>,
    owner_code_ids: BTreeMap<FunctionKey, String>,
    safety_doc_ids: BTreeMap<FunctionKey, String>,
}

fn build_code_dictionary(
    source_root: &Path,
    candidates: &[StructuralFindingCandidate],
    owner_code_spans: &BTreeMap<FunctionKey, StableSpan>,
) -> Result<CodeDictionary, String> {
    let local_callees = candidates
        .iter()
        .filter_map(|candidate| candidate.site.seed.callee.as_ref())
        .map(|callee| FunctionKey::new(callee.clone()))
        .filter(|callee| owner_code_spans.contains_key(callee))
        .collect::<BTreeSet<_>>();
    let mut owners = candidates
        .iter()
        .flat_map(finding_owners)
        .collect::<BTreeSet<_>>();
    owners.extend(local_callees.iter().cloned());
    let mut blocks = Vec::new();
    let mut owner_code_ids = BTreeMap::new();
    let mut safety_doc_ids = BTreeMap::new();
    for owner in owners {
        let span = owner_code_spans
            .get(&owner)
            .ok_or_else(|| format!("missing local body span for causal function {}", owner.0))?;
        let source_path = source_root.join(&span.path);
        let source = fs::read_to_string(&source_path).map_err(|error| {
            format!(
                "failed to read causal source '{}': {error}",
                source_path.display()
            )
        })?;
        let text = slice_span(&source, span)?;
        let text_sha = sha256_hex(text.as_bytes());
        let code_id = sha256_hex(&length_prefixed(&[
            "function_body",
            &owner.0,
            &span.token(),
            &text_sha,
        ]));
        owner_code_ids.insert(owner.clone(), code_id.clone());
        blocks.push(json!({
            "code_id": code_id,
            "kind": "function_body",
            "owner": owner.0,
            "path": span.path,
            "span": span_json(span),
            "sha256": text_sha,
            "text": text,
        }));
        if local_callees.contains(&owner) {
            if let Some((doc_span, doc_text)) = adjacent_safety_doc(&source, span)? {
                let doc_sha = sha256_hex(doc_text.as_bytes());
                let doc_id = sha256_hex(&length_prefixed(&[
                    "safety_doc",
                    &owner.0,
                    &doc_span.token(),
                    &doc_sha,
                ]));
                safety_doc_ids.insert(owner.clone(), doc_id.clone());
                blocks.push(json!({
                    "code_id": doc_id,
                    "kind": "safety_doc",
                    "owner": owner.0,
                    "path": doc_span.path,
                    "span": span_json(&doc_span),
                    "sha256": doc_sha,
                    "text": doc_text,
                }));
            }
        }
    }
    blocks.sort_by_key(|block| block["code_id"].as_str().unwrap_or_default().to_string());
    Ok(CodeDictionary {
        blocks,
        owner_code_ids,
        safety_doc_ids,
    })
}

fn adjacent_safety_doc(
    source: &str,
    function_span: &StableSpan,
) -> Result<Option<(StableSpan, String)>, String> {
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let function_line = usize::try_from(function_span.start.line - 1).unwrap();
    if function_line >= lines.len() {
        return Err(format!(
            "function body starts outside its source file: {}",
            function_span.token()
        ));
    }
    let mut cursor = function_line;
    while cursor > 0 && lines[cursor - 1].trim_start().starts_with("#[") {
        cursor -= 1;
    }
    let docs_end = cursor;
    while cursor > 0 && lines[cursor - 1].trim_start().starts_with("///") {
        cursor -= 1;
    }
    if cursor == docs_end {
        return Ok(None);
    }
    let text = lines[cursor..docs_end].concat();
    if !text.contains("# Safety") {
        return Ok(None);
    }
    let last = lines[docs_end - 1].trim_end_matches(['\r', '\n']);
    let start_line = u32::try_from(cursor + 1)
        .map_err(|_| "Safety documentation start line exceeds u32".to_string())?;
    let end_line = u32::try_from(docs_end)
        .map_err(|_| "Safety documentation end line exceeds u32".to_string())?;
    let end_column = u32::try_from(last.len() + 1)
        .map_err(|_| "Safety documentation end column exceeds u32".to_string())?;
    let span = StableSpan::new(
        function_span.path.clone(),
        super::summary::StablePosition::new(start_line, 1),
        super::summary::StablePosition::new(end_line, end_column),
    );
    Ok(Some((span.clone(), slice_span(source, &span)?)))
}

fn finding_owners(candidate: &StructuralFindingCandidate) -> BTreeSet<FunctionKey> {
    let mut owners = BTreeSet::from([
        candidate.public_root.clone(),
        candidate.site.seed.owner.clone(),
    ]);
    for edge in &candidate.call_route {
        owners.insert(edge.caller.clone());
        owners.insert(edge.callee.clone());
    }
    for edge in &candidate.endpoint_route {
        if let Some(owner) = endpoint_owner(&edge.from) {
            owners.insert(owner);
        }
        if let Some(owner) = endpoint_owner(&edge.to) {
            owners.insert(owner);
        }
    }
    if let Some(exposure) = &candidate.exposure {
        if let Some(owner) = endpoint_owner(&exposure.node) {
            owners.insert(owner);
        }
    }
    owners
}

fn slice_span(source: &str, span: &StableSpan) -> Result<String, String> {
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let start_line = usize::try_from(span.start.line - 1).unwrap();
    let end_line = usize::try_from(span.end.line - 1).unwrap();
    let start_column = usize::try_from(span.start.column - 1).unwrap();
    let end_column = usize::try_from(span.end.column - 1).unwrap();
    let start = line_offset(&lines, start_line, start_column, span)?;
    let end = line_offset(&lines, end_line, end_column, span)?;
    source.get(start..end).map(str::to_string).ok_or_else(|| {
        format!(
            "causal function span is not on UTF-8 boundaries: {}",
            span.token()
        )
    })
}

fn line_offset(
    lines: &[&str],
    line: usize,
    column: usize,
    span: &StableSpan,
) -> Result<usize, String> {
    let text = lines.get(line).ok_or_else(|| {
        format!(
            "causal function span line is outside source: {}",
            span.token()
        )
    })?;
    if column > text.len() {
        return Err(format!(
            "causal function span column is outside source: {}",
            span.token()
        ));
    }
    Ok(lines
        .iter()
        .take(line)
        .map(|line| line.len())
        .sum::<usize>()
        + column)
}

fn selected_source_root(project_root: &str) -> Result<PathBuf, String> {
    let root = env::var_os("UNSOUND_SCANNER_SOURCE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(project_root));
    root.canonicalize().map_err(|error| {
        format!(
            "failed to canonicalize UnsoundAudit v3 source root '{}': {error}",
            root.display()
        )
    })
}

fn normalize_metadata_path(value: &str, root: &Path) -> Result<String, String> {
    if value.is_empty() {
        return Err("receipt metadata path is empty".to_string());
    }
    let path = Path::new(value);
    let relative = if path.is_absolute() {
        let root = root.canonicalize().map_err(|error| {
            format!(
                "failed to canonicalize project root '{}': {error}",
                root.display()
            )
        })?;
        let path = path.canonicalize().map_err(|error| {
            format!(
                "failed to canonicalize metadata path '{}': {error}",
                path.display()
            )
        })?;
        path.strip_prefix(&root)
            .map_err(|_| format!("metadata path '{}' escapes project root", path.display()))?
            .to_path_buf()
    } else {
        path.to_path_buf()
    };
    normalized_relative_path(&relative)
}

fn normalized_relative_path(path: &Path) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .ok_or_else(|| "metadata path is not UTF-8".to_string())?;
                if part.is_empty() || part.contains('\\') || part.contains('\0') {
                    return Err("metadata path is not normalized".to_string());
                }
                parts.push(part);
            }
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err("metadata path is not a normalized relative path".to_string());
            }
        }
    }
    if parts.is_empty() {
        return Err("metadata path is empty".to_string());
    }
    let value = parts.join("/");
    if value.as_bytes().get(1) == Some(&b':') {
        return Err("metadata path has a Windows drive prefix".to_string());
    }
    Ok(value)
}

fn endpoint_identity(node: &EndpointNode) -> String {
    match node {
        EndpointNode::Formal { owner, index } => format!("{}:formal:{index}", owner.0),
        EndpointNode::PublicField { def_path, .. } => format!("public_field:{def_path}"),
        EndpointNode::ExternalBehavior {
            owner, kind, token, ..
        } => {
            format!("{}:external:{}:{token}", owner.0, external_kind(*kind))
        }
        EndpointNode::InternalDerivation {
            owner,
            token,
            point,
        } => {
            format!("{}:derivation:{token}@{}", owner.0, point.span.token())
        }
        EndpointNode::GenericCapability { owner, token, .. } => {
            format!("{}:generic_capability:{token}", owner.0)
        }
        EndpointNode::UnsafeOperand {
            owner,
            site_id,
            role,
        } => {
            format!(
                "{}:unsafe_operand:{site_id}:{}",
                owner.0,
                operand_role(role)
            )
        }
        EndpointNode::UnsafeResult { owner, site_id } => {
            format!("{}:unsafe_result:{site_id}", owner.0)
        }
        EndpointNode::Return { owner } => format!("{}:return", owner.0),
        EndpointNode::Out { owner, index } => format!("{}:out:{index}", owner.0),
        EndpointNode::Exposure {
            owner,
            token,
            point,
        } => {
            format!("{}:exposure:{token}@{}", owner.0, point.span.token())
        }
    }
}

fn endpoint_owner(node: &EndpointNode) -> Option<FunctionKey> {
    match node {
        EndpointNode::Formal { owner, .. }
        | EndpointNode::ExternalBehavior { owner, .. }
        | EndpointNode::InternalDerivation { owner, .. }
        | EndpointNode::GenericCapability { owner, .. }
        | EndpointNode::UnsafeOperand { owner, .. }
        | EndpointNode::UnsafeResult { owner, .. }
        | EndpointNode::Return { owner }
        | EndpointNode::Out { owner, .. }
        | EndpointNode::Exposure { owner, .. } => Some(owner.clone()),
        EndpointNode::PublicField { .. } => None,
    }
}

fn finding_counts_by_site(findings: &[Value]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for finding in findings {
        if let Some(site_id) = finding["unsafe_site"]["site_id"].as_str() {
            *counts.entry(site_id.to_string()).or_default() += 1;
        }
    }
    counts
}

fn pattern_counts(findings: &[Value]) -> Result<[usize; 6], String> {
    let mut counts = [0; 6];
    for finding in findings {
        let index = match finding["primary_pattern"].as_str() {
            Some("P1") => 0,
            Some("P2") => 1,
            Some("P3") => 2,
            Some("P4") => 3,
            Some("P5") => 4,
            Some("P6") => 5,
            _ => return Err("finding has an invalid primary pattern".to_string()),
        };
        counts[index] += 1;
    }
    Ok(counts)
}

fn coverage_failure_json(failure: &CoverageFailure) -> Value {
    json!({
        "reason": coverage_failure_kind(failure.kind),
        "owner": failure.owner.0,
        "site_id": failure.site_id,
        "candidate_count": failure.candidate_count,
    })
}

fn span_json(span: &StableSpan) -> Value {
    json!({
        "path": span.path,
        "start": {"line": span.start.line, "column": span.start.column},
        "end": {"line": span.end.line, "column": span.end.column},
    })
}

fn require_nonempty<'a>(value: &'a str, label: &str) -> Result<&'a str, String> {
    (!value.is_empty())
        .then_some(value)
        .ok_or_else(|| format!("missing {label}"))
}

fn length_prefixed(fields: &[&str]) -> Vec<u8> {
    let mut result = Vec::new();
    for field in fields {
        result.extend_from_slice(field.len().to_string().as_bytes());
        result.push(b':');
        result.extend_from_slice(field.as_bytes());
    }
    result
}

fn primary_pattern(pattern: PrimaryPattern) -> &'static str {
    match pattern {
        PrimaryPattern::P1 => "P1",
        PrimaryPattern::P2 => "P2",
        PrimaryPattern::P3 => "P3",
        PrimaryPattern::P4 => "P4",
        PrimaryPattern::P5 => "P5",
        PrimaryPattern::P6 => "P6",
    }
}

fn source_kind(kind: CausalSourceKind) -> &'static str {
    match kind {
        CausalSourceKind::PublicParameter => "public_parameter",
        CausalSourceKind::PublicField => "public_field",
        CausalSourceKind::InternalUnsafeOrigin => "internal_unsafe_origin",
        CausalSourceKind::InternalDerivation => "internal_derivation",
        CausalSourceKind::GenericCapabilityGap => "generic_capability_gap",
        CausalSourceKind::ExternalBehaviorOutput => "external_behavior_output",
    }
}

fn classification_confidence(confidence: ClassificationConfidence) -> &'static str {
    match confidence {
        ClassificationConfidence::Exact => "exact",
        ClassificationConfidence::Structural => "structural",
    }
}

fn contract_precision(precision: ContractPrecision) -> &'static str {
    match precision {
        ContractPrecision::Precise => "precise",
        ContractPrecision::Generic => "generic",
        ContractPrecision::Opaque => "opaque",
    }
}

fn failure_kind(contract: &ContractDescriptor, site_kind: UnsafeSiteKind) -> String {
    let phase = match contract.failure_phase {
        FailurePhase::Precondition => "precondition",
        FailurePhase::ImmediateValidity => "immediate_validity",
        FailurePhase::ExposureInvariant => "exposure_invariant",
    };
    format!("{phase}:{}", site_kind.as_str())
}

fn binding_status(status: BindingStatus) -> &'static str {
    match status {
        BindingStatus::BoundExact => "bound_exact",
        BindingStatus::Unbound => "unbound",
        BindingStatus::Ambiguous => "ambiguous",
        BindingStatus::Elided => "elided",
    }
}

fn coverage_failure_kind(kind: CoverageFailureKind) -> &'static str {
    match kind {
        CoverageFailureKind::CollectorMissingAfterSteal => "collector_missing_after_steal",
        CoverageFailureKind::CollectorFailed => "collector_failed",
        CoverageFailureKind::UnboundUnsafeSite => "unbound_unsafe_site",
        CoverageFailureKind::AmbiguousUnsafeSite => "ambiguous_unsafe_site",
        CoverageFailureKind::ElidedUnsafeSite => "elided_unsafe_site",
    }
}

fn operand_role(role: &OperandRole) -> String {
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

fn external_kind(kind: ExternalSourceKind) -> &'static str {
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

fn exposure_kind(node: &EndpointNode) -> &'static str {
    match node {
        EndpointNode::Return { .. } => "public_return",
        EndpointNode::Out { .. } => "public_out",
        EndpointNode::Exposure { .. } => "public_store",
        _ => "public_exposure",
    }
}

#[cfg(test)]
mod tests {
    use super::super::summary::StablePosition;
    use super::*;

    fn span(line: u32) -> StableSpan {
        StableSpan::new(
            "src/lib.rs",
            StablePosition::new(line, 1),
            StablePosition::new(line, 2),
        )
    }

    #[test]
    fn finding_identity_excludes_witness_code_and_contract_model_version() {
        let root = FunctionKey::new("sample::root");
        let first = finding_id(
            PrimaryPattern::P1,
            &"a".repeat(64),
            "sample::root:formal:1",
            "precondition:unsafe_call",
            &span(2),
            &root,
            None,
        );
        let same = finding_id(
            PrimaryPattern::P1,
            &"a".repeat(64),
            "sample::root:formal:1",
            "precondition:unsafe_call",
            &span(2),
            &root,
            None,
        );
        assert_eq!(first, same);
        assert_ne!(
            first,
            finding_id(
                PrimaryPattern::P1,
                &"b".repeat(64),
                "sample::root:formal:1",
                "precondition:unsafe_call",
                &span(2),
                &root,
                None,
            )
        );
    }

    #[test]
    fn source_slicing_uses_one_based_exclusive_columns() {
        let source = "fn a() {}\nfn b() {}\n";
        let body = StableSpan::new(
            "src/lib.rs",
            StablePosition::new(2, 1),
            StablePosition::new(2, 10),
        );
        assert_eq!(slice_span(source, &body).unwrap(), "fn b() {}");
    }

    #[test]
    fn normalized_metadata_paths_reject_parent_and_drive_forms() {
        assert_eq!(
            normalized_relative_path(Path::new("src/lib.rs")).unwrap(),
            "src/lib.rs"
        );
        assert!(normalized_relative_path(Path::new("../src/lib.rs")).is_err());
        assert!(normalized_relative_path(Path::new("C:src/lib.rs")).is_err());
    }

    #[test]
    fn semantic_invocation_identity_excludes_fresh_target_specific_wrapper_id() {
        let metadata = UnitMetadata {
            project_root: "/workspace/sample".to_string(),
            unit_id: "sample-unit".to_string(),
            rustc_invocation_id: "rustc-fresh-target-one".to_string(),
            package_id: "sample@0.1.0#/workspace/sample/Cargo.toml".to_string(),
            package_name: "sample".to_string(),
            package_version: "0.1.0".to_string(),
            crate_name: "sample".to_string(),
            crate_types: vec!["lib".to_string()],
            target_kind: "lib".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            manifest_path: "/workspace/sample/Cargo.toml".to_string(),
            source_path: "/workspace/sample/src/lib.rs".to_string(),
            extra_filename: "sample-hash".to_string(),
            rustc_path: "/toolchain/rustc".to_string(),
            cargo_supplied_rustc_path: "/toolchain/rustc".to_string(),
            cargo_supplied_rustc_commit: "a".repeat(40),
            rap_compiler_commit: "b".repeat(40),
        };
        let first = semantic_rustc_invocation_id(&metadata, "Cargo.toml", "src/lib.rs");
        let mut second_metadata = metadata;
        second_metadata.rustc_invocation_id = "rustc-fresh-target-two".to_string();
        let second = semantic_rustc_invocation_id(&second_metadata, "Cargo.toml", "src/lib.rs");
        assert_eq!(first, second);
        assert!(first.starts_with("rustc-"));
        assert_eq!(first.len(), 22);
    }

    #[test]
    fn adjacent_local_safety_docs_are_captured_without_source_discovery() {
        let source = "/// Reads one byte.\n/// # Safety\n/// `pointer` must be valid.\n#[inline]\nunsafe fn read(pointer: *const u8) -> u8 { *pointer }\n";
        let function = StableSpan::new(
            "src/lib.rs",
            StablePosition::new(5, 1),
            StablePosition::new(5, 57),
        );
        let (doc_span, text) = adjacent_safety_doc(source, &function).unwrap().unwrap();
        assert_eq!(doc_span.start, StablePosition::new(1, 1));
        assert_eq!(doc_span.end.line, 3);
        assert_eq!(
            text,
            "/// Reads one byte.\n/// # Safety\n/// `pointer` must be valid."
        );

        let no_safety =
            "/// Reads one byte.\nunsafe fn read(pointer: *const u8) -> u8 { *pointer }\n";
        let function = StableSpan::new(
            "src/lib.rs",
            StablePosition::new(2, 1),
            StablePosition::new(2, 57),
        );
        assert!(adjacent_safety_doc(no_safety, &function).unwrap().is_none());
    }
}
