use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::thread;

use serde_json::{json, Value};

use crate::rap_warn;

use super::candidate::{
    CandidateCallKind, StructuralFindingCandidate, SuppressedStructuralCandidate,
    UnestablishedStructuralCandidate,
};
use super::contract_catalog::{ContractDescriptor, FailurePhase, StaticContractImplication};
use super::endpoint_flow::{
    AccessExtent, BehaviorController, BehaviorOpenness, CapabilityRequirement, CapabilityWitness,
    EndpointEdge, EndpointFlowGraph, EndpointNode, ExternalSourceKind, StorageProvenance,
};
use super::mir::AnalysisOutput;
use super::missing_sanitizer::MissingSanitizerWarning;
use super::perf_trace::PerfTrace;
use super::summary::{sha256_hex, FunctionKey, ProgramPoint, StableSpan};
use super::unsafe_site::binding::{BoundUnsafeSite, SiteBindingRecord};
use super::unsafe_site::{
    self, BindingStatus, CausalSourceKind, ClassificationConfidence, ContractPrecision,
    CoverageFailure, CoverageFailureKind, OperandRole, PrimaryPattern, UnsafeSiteKind,
    UnsafeSiteSeed,
};
use super::UnitMetadata;

#[path = "v3_output_reporting.rs"]
mod reporting;

use reporting::{build_reports_with_unestablished, MvpReports, SuppressionRecord};

const MVP_OUTPUT_ENV: &str = "RAP_UNSOUNDAUDIT_OUTPUT";
const RECEIPT_MAX_WORKERS: usize = 8;
const RECEIPT_CHUNK_SIZE: usize = 256;
const RECEIPT_WORKER_STACK_BYTES: usize = 1024 * 1024;

struct SuppressionPlan {
    record: SuppressionRecord,
    original_index: usize,
}

pub(super) fn emit_mvp_output(
    analysis: &mut AnalysisOutput,
    trace: &mut PerfTrace,
) -> Result<(), String> {
    // The formal rap-unit-v3 writer runs after this development receipt.  Its
    // input must remain intact when requested; the MVP-only path can instead
    // consume rich analysis payloads as soon as their compact receipt state is
    // available.
    let preserve_formal_analysis = env::var_os("UNSOUND_SCANNER_RAP_JSON_OUT").is_some();
    let project_root = env::var("UNSOUND_SCANNER_PROJECT_ROOT")
        .ok()
        .or_else(|| {
            env::current_dir()
                .ok()
                .map(|path| path.display().to_string())
        })
        .unwrap_or_default();
    let source_root = selected_source_root(&project_root)?;
    let candidate_count =
        analysis.structural_candidates.len() + analysis.suppressed_structural_candidates.len();
    trace.event(
        "receipt_code_dictionary",
        "begin",
        [("candidate_count", candidate_count as u64)],
    )?;
    trace.event(
        "receipt_code_owner_collect",
        "begin",
        [("candidate_count", candidate_count as u64)],
    )?;
    let (owners, local_callees) = collect_code_dictionary_inputs_ordered(
        analysis.structural_candidates.iter().chain(
            analysis
                .suppressed_structural_candidates
                .iter()
                .map(|suppressed| &suppressed.candidate),
        ),
        configured_receipt_workers(),
        RECEIPT_CHUNK_SIZE,
    )?;
    let local_callees = local_callees
        .into_iter()
        .filter(|callee| analysis.owner_code_spans.contains_key(callee))
        .collect::<BTreeSet<_>>();
    trace.event(
        "receipt_code_owner_collect",
        "end",
        [
            ("owner_count", owners.len() as u64),
            ("local_callee_count", local_callees.len() as u64),
        ],
    )?;
    trace.event(
        "receipt_code_blocks",
        "begin",
        [("owner_count", owners.len() as u64)],
    )?;
    let code = build_code_dictionary_for_owners(
        &source_root,
        owners,
        &local_callees,
        &analysis.owner_code_spans,
        |source_path| {
            fs::read_to_string(source_path).map_err(|error| {
                format!(
                    "failed to read causal source '{}': {error}",
                    source_path.display()
                )
            })
        },
    )?;
    trace.event(
        "receipt_code_blocks",
        "end",
        [("code_block_count", code.blocks.len() as u64)],
    )?;
    trace.event(
        "receipt_code_dictionary",
        "end",
        [("code_block_count", code.blocks.len() as u64)],
    )?;
    trace.event("receipt_cycle_index", "begin", [])?;
    let cycle_index = selected_cycle_index_for_analysis(analysis, preserve_formal_analysis);
    trace.event("receipt_cycle_index", "end", [])?;
    let considered_count = analysis.structural_candidates.len()
        + analysis.suppressed_structural_candidates.len()
        + analysis.unestablished_structural_candidates.len();
    trace.event(
        "receipt_active_materialize",
        "begin",
        [(
            "candidate_count",
            analysis.structural_candidates.len() as u64,
        )],
    )?;
    let mut findings = Vec::with_capacity(analysis.structural_candidates.len());
    let mut replacement_ids = BTreeMap::<String, String>::new();
    if preserve_formal_analysis {
        materialize_active_candidates_ordered(
            analysis
                .structural_candidates
                .iter()
                .map(ActiveCandidateInput::Borrowed),
            &cycle_index,
            &code,
            &mut findings,
            &mut replacement_ids,
            trace,
        )?;
    } else {
        materialize_active_candidates_ordered(
            std::mem::take(&mut analysis.structural_candidates)
                .into_iter()
                .map(ActiveCandidateInput::Owned),
            &cycle_index,
            &code,
            &mut findings,
            &mut replacement_ids,
            trace,
        )?;
    }
    trace.event(
        "receipt_active_materialize",
        "end",
        [("candidate_count", findings.len() as u64)],
    )?;
    trace.event(
        "receipt_active_order",
        "begin",
        [("candidate_count", findings.len() as u64)],
    )?;
    findings.sort_unstable_by(|left, right| {
        left["finding_id"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["finding_id"].as_str().unwrap_or_default())
    });
    ensure_unique_finding_ids(&findings, "active MVP")?;
    trace.event(
        "receipt_active_order",
        "end",
        [("candidate_count", findings.len() as u64)],
    )?;

    trace.event(
        "receipt_suppressed_plan",
        "begin",
        [(
            "candidate_count",
            analysis.suppressed_structural_candidates.len() as u64,
        )],
    )?;
    let suppressed_structural_candidates = if preserve_formal_analysis {
        // Formal output owns `analysis` next.  A simultaneous development
        // package is intentionally the conservative, higher-memory case.
        analysis.suppressed_structural_candidates.clone()
    } else {
        std::mem::take(&mut analysis.suppressed_structural_candidates)
    };
    let mut suppressions = Vec::with_capacity(suppressed_structural_candidates.len());
    for (original_index, suppressed) in suppressed_structural_candidates.iter().enumerate() {
        let candidate_id = mvp_candidate_finding_id(&suppressed.candidate);
        let replacement_candidate_id = replacement_ids
            .get(&replacement_key(&suppressed.candidate))
            .filter(|replacement| replacement.as_str() != candidate_id)
            .cloned();
        suppressions.push(SuppressionPlan {
            record: SuppressionRecord::new(
                candidate_id,
                suppressed.reason,
                replacement_candidate_id,
            ),
            original_index,
        });
        if trace.is_enabled() && suppressions.len().is_power_of_two() {
            trace.event(
                "receipt_suppressed_plan",
                "checkpoint",
                [("completed", suppressions.len() as u64)],
            )?;
        }
    }
    trace.event(
        "receipt_suppressed_plan",
        "end",
        [("candidate_count", suppressions.len() as u64)],
    )?;
    drop(replacement_ids);
    trace.event(
        "receipt_suppressed_order",
        "begin",
        [("candidate_count", suppressions.len() as u64)],
    )?;
    suppressions
        .sort_unstable_by(|left, right| left.record.sort_key().cmp(&right.record.sort_key()));
    trace.event(
        "receipt_suppressed_order",
        "end",
        [("candidate_count", suppressions.len() as u64)],
    )?;
    let unestablished = if preserve_formal_analysis {
        analysis.unestablished_structural_candidates.clone()
    } else {
        std::mem::take(&mut analysis.unestablished_structural_candidates)
    };
    let unestablished = prepare_unestablished_decisions(unestablished)?;
    let unestablished_ids = unestablished
        .iter()
        .map(|decision| decision.decision_id.as_str())
        .collect::<Vec<_>>();

    trace.event(
        "receipt_reports",
        "begin",
        [
            ("active_count", findings.len() as u64),
            ("suppressed_count", suppressions.len() as u64),
            ("unestablished_count", unestablished.len() as u64),
        ],
    )?;
    let reports = build_reports_with_unestablished(
        &findings,
        suppressions.iter().map(|suppression| &suppression.record),
        &unestablished_ids,
        considered_count,
    )?;
    trace.event(
        "receipt_reports",
        "end",
        [
            ("family_count", reports.families.len() as u64),
            ("suppressed_count", reports.suppressed_count as u64),
            ("unestablished_count", reports.unestablished_count as u64),
        ],
    )?;
    for family in &reports.families {
        rap_warn!("{}", mvp_family_summary_line(family)?);
    }

    trace.event("receipt_census", "begin", [])?;
    let snapshot = unsafe_site::census_snapshot()?;
    let sites = census_sites(&snapshot.owners)?;
    let failures = snapshot.failures.iter().cloned().collect::<Vec<_>>();
    let finding_counts = finding_counts_by_site(&findings);
    let site_records = build_site_records(
        &sites,
        &analysis.binding_records,
        &failures,
        &finding_counts,
    )?;
    let census = census_json(&sites, &site_records);
    trace.event(
        "receipt_census",
        "end",
        [
            ("site_count", sites.len() as u64),
            ("failure_count", failures.len() as u64),
        ],
    )?;
    if failures.is_empty() {
        rap_warn!(
            "UnsoundAudit v3: {} active candidate(s) in {} family/families, {} structurally suppressed, {} unestablished, coverage complete",
            reports.active_count,
            reports.families.len(),
            reports.suppressed_count,
            reports.unestablished_count,
        );
    } else {
        rap_warn!(
            "UnsoundAudit v3: {} active candidate(s) in {} family/families, {} structurally suppressed, {} unestablished, coverage partial ({} gap(s)); results are a lower bound",
            reports.active_count,
            reports.families.len(),
            reports.suppressed_count,
            reports.unestablished_count,
            failures.len()
        );
    }

    let Some(output_dir) = env::var_os(MVP_OUTPUT_ENV).map(PathBuf::from) else {
        return Ok(());
    };
    let warnings = analysis
        .missing_sanitizer_warnings
        .iter()
        .map(missing_sanitizer_warning_json)
        .collect::<Vec<_>>();
    write_mvp_package(
        &output_dir,
        findings,
        reports,
        suppressions,
        suppressed_structural_candidates,
        unestablished,
        &cycle_index,
        census,
        &failures,
        &code.owner_code_ids,
        &code.safety_doc_ids,
        &code.blocks,
        &warnings,
        trace,
    )
}

fn selected_cycle_index_for_analysis(
    analysis: &mut AnalysisOutput,
    preserve_formal_analysis: bool,
) -> SelectedCycleIndex {
    if preserve_formal_analysis {
        SelectedCycleIndex::new(&analysis.endpoint_graph, &analysis.owner_spans)
    } else {
        SelectedCycleIndex::from_graph(
            std::mem::take(&mut analysis.endpoint_graph),
            &analysis.owner_spans,
        )
    }
}

enum ActiveCandidateInput<'candidate> {
    Borrowed(&'candidate StructuralFindingCandidate),
    Owned(StructuralFindingCandidate),
}

impl ActiveCandidateInput<'_> {
    fn candidate(&self) -> &StructuralFindingCandidate {
        match self {
            Self::Borrowed(candidate) => candidate,
            Self::Owned(candidate) => candidate,
        }
    }
}

struct MaterializedActiveCandidate {
    finding: Value,
    replacement_key: String,
}

fn materialize_active_candidate(
    candidate: &StructuralFindingCandidate,
    cycle_index: &SelectedCycleIndex,
    code: &CodeDictionary,
) -> Result<MaterializedActiveCandidate, String> {
    let finding = mvp_finding_json(
        candidate,
        cycle_index,
        &code.owner_code_ids,
        &code.safety_doc_ids,
    )?;
    finding["finding_id"]
        .as_str()
        .ok_or_else(|| "MVP finding lacks finding_id".to_string())?;
    Ok(MaterializedActiveCandidate {
        finding,
        replacement_key: replacement_key(candidate),
    })
}

fn materialize_active_candidates_ordered<'candidate>(
    candidates: impl IntoIterator<Item = ActiveCandidateInput<'candidate>>,
    cycle_index: &SelectedCycleIndex,
    code: &CodeDictionary,
    findings: &mut Vec<Value>,
    replacement_ids: &mut BTreeMap<String, String>,
    trace: &mut PerfTrace,
) -> Result<(), String> {
    let process = |candidates: Vec<ActiveCandidateInput<'candidate>>| {
        candidates
            .into_iter()
            .map(|candidate| materialize_active_candidate(candidate.candidate(), cycle_index, code))
            .collect::<Result<Vec<_>, _>>()
    };
    let mut consume = |materialized: Vec<MaterializedActiveCandidate>| {
        let previous = findings.len();
        for materialized in materialized {
            let finding_id = materialized.finding["finding_id"]
                .as_str()
                .ok_or_else(|| "MVP finding lacks finding_id".to_string())?;
            replacement_ids
                .entry(materialized.replacement_key)
                .and_modify(|selected| {
                    if finding_id < selected.as_str() {
                        *selected = finding_id.to_string();
                    }
                })
                .or_insert_with(|| finding_id.to_string());
            findings.push(materialized.finding);
        }
        trace_power_of_two_checkpoints(
            trace,
            "receipt_active_materialize",
            previous,
            findings.len(),
        )
    };
    process_ordered_chunks(
        candidates,
        configured_receipt_workers(),
        RECEIPT_CHUNK_SIZE,
        "active-materialize",
        &process,
        &mut consume,
    )
}

fn trace_power_of_two_checkpoints(
    trace: &mut PerfTrace,
    phase: &str,
    previous: usize,
    completed: usize,
) -> Result<(), String> {
    if !trace.is_enabled() || completed <= previous {
        return Ok(());
    }
    let mut checkpoint = 1usize;
    while checkpoint <= completed {
        if checkpoint > previous {
            trace.event(phase, "checkpoint", [("completed", checkpoint as u64)])?;
        }
        let Some(next) = checkpoint.checked_mul(2) else {
            break;
        };
        checkpoint = next;
    }
    Ok(())
}

fn mvp_family_summary_line(family: &Value) -> Result<String, String> {
    let required = |path: &[&str]| -> Result<&str, String> {
        let mut value = family;
        for key in path {
            value = value
                .get(*key)
                .ok_or_else(|| format!("MVP family lacks {}", path.join(".")))?;
        }
        value
            .as_str()
            .ok_or_else(|| format!("MVP family field {} is not a string", path.join(".")))
    };
    Ok(format!(
        "[{}] family={} raw={} roots={} subject={} site={}:{}",
        required(&["key", "primary_pattern"])?,
        required(&["family_id"])?,
        family["raw_count"]
            .as_u64()
            .ok_or_else(|| "MVP family raw_count is not an integer".to_string())?,
        family["public_root_fanout"]
            .as_u64()
            .ok_or_else(|| "MVP family public_root_fanout is not an integer".to_string())?,
        required(&["key", "subject_role"])?,
        required(&["representatives", "shortest", "unsafe_site", "span", "path"])?,
        family["representatives"]["shortest"]["unsafe_site"]["span"]["start"]["line"]
            .as_u64()
            .ok_or_else(|| "MVP family unsafe-site line is not an integer".to_string())?,
    ))
}

fn write_mvp_package(
    output_dir: &Path,
    findings: Vec<Value>,
    reports: MvpReports,
    suppressions: Vec<SuppressionPlan>,
    suppressed_candidates: Vec<SuppressedStructuralCandidate>,
    unestablished: Vec<UnestablishedStructuralCandidate>,
    cycle_index: &SelectedCycleIndex,
    census: Value,
    failures: &[CoverageFailure],
    code_ids: &BTreeMap<FunctionKey, String>,
    safety_doc_ids: &BTreeMap<FunctionKey, String>,
    code_blocks: &[Value],
    warnings: &[Value],
    trace: &mut PerfTrace,
) -> Result<(), String> {
    let (families, candidate_accounting, suppressed_count, unestablished_count) =
        reports.into_write_parts(&findings)?;
    if suppressed_count != suppressions.len() {
        return Err(format!(
            "MVP report suppression count drift: report={}, planned={}",
            suppressed_count,
            suppressions.len()
        ));
    }
    if suppressions.len() != suppressed_candidates.len() {
        return Err(format!(
            "MVP suppression plan drift: plans={}, candidates={}",
            suppressions.len(),
            suppressed_candidates.len()
        ));
    }
    if unestablished_count != unestablished.len() {
        return Err(format!(
            "MVP report unestablished count drift: report={}, planned={}",
            unestablished_count,
            unestablished.len()
        ));
    }
    fs::create_dir_all(output_dir).map_err(|error| {
        format!(
            "failed to create MVP output directory '{}': {error}",
            output_dir.display()
        )
    })?;
    let code_dir = output_dir.join("code");
    fs::create_dir_all(&code_dir).map_err(|error| {
        format!(
            "failed to create MVP code directory '{}': {error}",
            code_dir.display()
        )
    })?;

    trace.event(
        "receipt_write_candidates",
        "begin",
        [("row_count", findings.len() as u64)],
    )?;
    write_jsonl(&output_dir.join("candidates.jsonl"), "candidate", &findings)?;
    write_jsonl(
        &output_dir.join("missing_sanitizer_warnings.jsonl"),
        "missing sanitizer warning",
        warnings,
    )?;
    let finding_count = findings.len();
    trace.event(
        "receipt_write_candidates",
        "end",
        [("row_count", finding_count as u64)],
    )?;
    drop(findings);
    trace.event(
        "receipt_write_families",
        "begin",
        [("row_count", families.len() as u64)],
    )?;
    write_jsonl(&output_dir.join("families.jsonl"), "family", &families)?;
    trace.event(
        "receipt_write_families",
        "end",
        [("row_count", families.len() as u64)],
    )?;
    drop(families);
    trace.event(
        "receipt_write_unestablished",
        "begin",
        [("row_count", unestablished.len() as u64)],
    )?;
    let unestablished_count = unestablished.len();
    write_unestablished_jsonl(&output_dir.join("unestablished.jsonl"), unestablished)?;
    trace.event(
        "receipt_write_unestablished",
        "end",
        [("row_count", unestablished_count as u64)],
    )?;
    trace.event(
        "receipt_write_suppressions",
        "begin",
        [("row_count", suppressions.len() as u64)],
    )?;
    let suppression_count = suppressions.len();
    write_suppression_jsonl(
        &output_dir.join("suppression-ledger.jsonl"),
        suppressions,
        suppressed_candidates,
        cycle_index,
        code_ids,
        safety_doc_ids,
        trace,
    )?;
    trace.event(
        "receipt_write_suppressions",
        "end",
        [("row_count", suppression_count as u64)],
    )?;

    trace.event(
        "receipt_write_code",
        "begin",
        [("block_count", code_blocks.len() as u64)],
    )?;
    for block in code_blocks {
        let code_id = block["code_id"]
            .as_str()
            .ok_or_else(|| "MVP code block lacks code_id".to_string())?;
        if code_id.len() != 64 || !code_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("MVP code block has a non-canonical code_id".to_string());
        }
        let text = block["text"]
            .as_str()
            .ok_or_else(|| format!("MVP code block {code_id} lacks text"))?;
        fs::write(code_dir.join(format!("{code_id}.rs")), text)
            .map_err(|error| format!("failed to write MVP code block {code_id}: {error}"))?;
    }
    trace.event(
        "receipt_write_code",
        "end",
        [("block_count", code_blocks.len() as u64)],
    )?;

    trace.event("receipt_write_coverage", "begin", [])?;
    let coverage = mvp_coverage_json(census, failures, candidate_accounting);
    let mut coverage_bytes = serde_json::to_vec_pretty(&coverage)
        .map_err(|error| format!("failed to serialize MVP coverage: {error}"))?;
    coverage_bytes.push(b'\n');
    fs::write(output_dir.join("coverage.json"), coverage_bytes)
        .map_err(|error| format!("failed to write MVP coverage: {error}"))?;
    trace.event("receipt_write_coverage", "end", [])
}

fn mvp_coverage_json(
    census: Value,
    failures: &[CoverageFailure],
    candidate_accounting: Value,
) -> Value {
    json!({
        "schema_version": "rap-unsoundaudit-coverage-v3-mvp",
        "complete": failures.is_empty(),
        "semantics": if failures.is_empty() { "exact" } else { "lower_bound" },
        "candidate_accounting": candidate_accounting,
        "unsafe_site_census": census,
        "coverage_failures": failures.iter().map(coverage_failure_json).collect::<Vec<_>>(),
    })
}

fn missing_sanitizer_warning_json(warning: &MissingSanitizerWarning) -> Value {
    json!({
        "public_root_owner": warning.public_root.0,
        "unsafe_site_id": warning.site_id,
        "predicate": warning.predicate.as_str(),
        "sink_model": warning.sink_model,
    })
}

fn write_jsonl(path: &Path, label: &str, rows: &[Value]) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|error| {
        format!(
            "failed to create MVP {label} output '{}': {error}",
            path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    write_json_rows_ordered(
        &mut writer,
        label,
        rows.iter(),
        configured_receipt_workers(),
        RECEIPT_CHUNK_SIZE,
    )?;
    writer
        .flush()
        .map_err(|error| format!("failed to flush MVP {label}: {error}"))
}

struct SerializedJsonChunk {
    bytes: Vec<u8>,
    row_count: usize,
}

fn configured_receipt_workers() -> usize {
    thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
        .min(RECEIPT_MAX_WORKERS)
}

fn write_json_rows_ordered<'row>(
    writer: &mut impl Write,
    label: &str,
    rows: impl IntoIterator<Item = &'row Value>,
    worker_count: usize,
    chunk_size: usize,
) -> Result<(), String> {
    let process = |rows: Vec<&Value>| {
        let row_count = rows.len();
        let mut bytes = Vec::new();
        for row in rows {
            let mut row_bytes = serde_json::to_vec(row)
                .map_err(|error| format!("failed to serialize MVP {label}: {error}"))?;
            bytes.append(&mut row_bytes);
            bytes.push(b'\n');
        }
        Ok(SerializedJsonChunk { row_count, bytes })
    };
    let mut consume = |chunk: SerializedJsonChunk| {
        writer
            .write_all(&chunk.bytes)
            .map_err(|error| format!("failed to write MVP {label}: {error}"))
    };
    process_ordered_chunks(
        rows,
        worker_count,
        chunk_size,
        label,
        &process,
        &mut consume,
    )
}

fn next_ordered_chunk<T>(items: &mut impl Iterator<Item = T>, chunk_size: usize) -> Option<Vec<T>> {
    let chunk = items.by_ref().take(chunk_size).collect::<Vec<_>>();
    (!chunk.is_empty()).then_some(chunk)
}

fn record_earliest_failure(failure: &mut Option<(usize, String)>, ordinal: usize, error: String) {
    if failure
        .as_ref()
        .is_none_or(|(selected_ordinal, _)| ordinal < *selected_ordinal)
    {
        *failure = Some((ordinal, error));
    }
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

/// Run bounded chunks concurrently while exposing results to the caller in
/// strict input order.  Only the main thread invokes `consume`; after joining
/// one ordinal it replenishes the worker window before consuming that result,
/// so ordered writes overlap with computation of the next chunk.
fn process_ordered_chunks<T, R>(
    items: impl IntoIterator<Item = T>,
    worker_count: usize,
    chunk_size: usize,
    worker_label: &str,
    process: &(impl Fn(Vec<T>) -> Result<R, String> + Sync),
    consume: &mut impl FnMut(R) -> Result<(), String>,
) -> Result<(), String>
where
    T: Send,
    R: Send,
{
    if worker_count == 0 {
        return Err("ordered receipt worker count must be positive".to_string());
    }
    if chunk_size == 0 {
        return Err("ordered receipt chunk size must be positive".to_string());
    }
    let mut items = items.into_iter();
    thread::scope(|scope| {
        let spawn = |ordinal, chunk| {
            let name = format!("rap-receipt-{worker_label}-{ordinal}");
            thread::Builder::new()
                .name(name)
                .stack_size(RECEIPT_WORKER_STACK_BYTES)
                .spawn_scoped(scope, move || process(chunk))
        };
        let mut handles = VecDeque::new();
        let mut next_ordinal = 0usize;
        let mut failure = None::<(usize, String)>;

        for _ in 0..worker_count {
            let Some(chunk) = next_ordered_chunk(&mut items, chunk_size) else {
                break;
            };
            let ordinal = next_ordinal;
            next_ordinal += 1;
            match spawn(ordinal, chunk) {
                Ok(handle) => handles.push_back((ordinal, handle)),
                Err(error) => {
                    record_earliest_failure(
                        &mut failure,
                        ordinal,
                        format!(
                            "failed to spawn ordered MVP {worker_label} chunk {ordinal}: {error}"
                        ),
                    );
                    break;
                }
            }
        }

        while let Some((ordinal, handle)) = handles.pop_front() {
            let joined = match handle.join() {
                Ok(result) => result,
                Err(payload) => Err(format!(
                    "ordered MVP {worker_label} worker panicked in chunk {ordinal}: {}",
                    panic_payload_message(payload.as_ref())
                )),
            };
            let still_precedes_failure = failure
                .as_ref()
                .is_none_or(|(failure_ordinal, _)| ordinal < *failure_ordinal);
            if !still_precedes_failure {
                continue;
            }
            let output = match joined {
                Ok(output) => output,
                Err(error) => {
                    record_earliest_failure(&mut failure, ordinal, error);
                    continue;
                }
            };

            // Replenish before the main-thread consume/write to overlap the
            // next computation with this ordinal's ordered output.
            if failure.is_none() {
                if let Some(chunk) = next_ordered_chunk(&mut items, chunk_size) {
                    let next = next_ordinal;
                    next_ordinal += 1;
                    match spawn(next, chunk) {
                        Ok(handle) => handles.push_back((next, handle)),
                        Err(error) => record_earliest_failure(
                            &mut failure,
                            next,
                            format!(
                                "failed to spawn ordered MVP {worker_label} chunk {next}: {error}"
                            ),
                        ),
                    }
                }
            }
            if let Err(error) = consume(output) {
                record_earliest_failure(&mut failure, ordinal, error);
            }
        }

        match failure {
            Some((_, error)) => Err(error),
            None => Ok(()),
        }
    })
}

fn write_suppression_jsonl(
    path: &Path,
    suppressions: Vec<SuppressionPlan>,
    mut suppressed_candidates: Vec<SuppressedStructuralCandidate>,
    cycle_index: &SelectedCycleIndex,
    code_ids: &BTreeMap<FunctionKey, String>,
    safety_doc_ids: &BTreeMap<FunctionKey, String>,
    trace: &mut PerfTrace,
) -> Result<(), String> {
    trace.event(
        "receipt_suppressed_permute",
        "begin",
        [("row_count", suppressions.len() as u64)],
    )?;
    reorder_suppressed_candidates(&suppressions, &mut suppressed_candidates)?;
    trace.event(
        "receipt_suppressed_permute",
        "end",
        [("row_count", suppressions.len() as u64)],
    )?;
    let file = fs::File::create(path).map_err(|error| {
        format!(
            "failed to create MVP suppression ledger output '{}': {error}",
            path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    let mut completed = 0usize;
    let process = |rows: Vec<(SuppressionPlan, SuppressedStructuralCandidate)>| {
        let row_count = rows.len();
        let mut bytes = Vec::new();
        for (suppression, suppressed) in rows {
            let candidate =
                mvp_finding_json(&suppressed.candidate, cycle_index, code_ids, safety_doc_ids)?;
            let evidence = Value::Object(
                suppressed
                    .evidence
                    .into_iter()
                    .map(|(key, value)| (key, Value::String(value)))
                    .collect(),
            );
            let row = suppression.record.into_json(candidate, evidence)?;
            let mut row_bytes = serde_json::to_vec(&row)
                .map_err(|error| format!("failed to serialize MVP suppression ledger: {error}"))?;
            bytes.append(&mut row_bytes);
            bytes.push(b'\n');
        }
        Ok(SerializedJsonChunk { bytes, row_count })
    };
    let mut consume = |chunk: SerializedJsonChunk| {
        writer
            .write_all(&chunk.bytes)
            .map_err(|error| format!("failed to write MVP suppression ledger: {error}"))?;
        let previous = completed;
        completed += chunk.row_count;
        trace_power_of_two_checkpoints(trace, "receipt_write_suppressions", previous, completed)
    };
    process_ordered_chunks(
        suppressions.into_iter().zip(suppressed_candidates),
        configured_receipt_workers(),
        RECEIPT_CHUNK_SIZE,
        "suppression-ledger",
        &process,
        &mut consume,
    )?;
    writer
        .flush()
        .map_err(|error| format!("failed to flush MVP suppression ledger: {error}"))
}

fn validate_unestablished_decisions(
    decisions: &[UnestablishedStructuralCandidate],
) -> Result<(), String> {
    let mut previous = None::<&str>;
    for decision in decisions {
        if decision.decision_id.len() != 64
            || !decision
                .decision_id
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("unestablished decision has a non-canonical decision_id".to_string());
        }
        if previous.is_some_and(|selected| selected >= decision.decision_id.as_str()) {
            return Err("unestablished decision IDs are duplicated or not sorted".to_string());
        }
        // Every frozen pattern may retain an exact but unresolved candidate.
        // Pattern-specific construction still owns the semantic gate; the
        // writer only enforces deterministic four-state accounting.
        if decision.controller.is_empty() || decision.capabilities.is_empty() {
            return Err(
                "unestablished decision has empty controller/capability identity".to_string(),
            );
        }
        if decision
            .secondary_reasons
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || decision
                .secondary_reasons
                .iter()
                .any(|reason| *reason <= decision.primary_reason)
        {
            return Err(
                "unestablished decision secondary reasons violate stable precedence".to_string(),
            );
        }
        previous = Some(&decision.decision_id);
    }
    Ok(())
}

fn prepare_unestablished_decisions(
    mut decisions: Vec<UnestablishedStructuralCandidate>,
) -> Result<Vec<UnestablishedStructuralCandidate>, String> {
    decisions.sort_unstable_by(|left, right| left.decision_id.cmp(&right.decision_id));
    validate_unestablished_decisions(&decisions)?;
    Ok(decisions)
}

fn unestablished_json(decision: UnestablishedStructuralCandidate) -> Value {
    json!({
        "schema_version": "rap-unsoundaudit-unestablished-v1",
        "decision_id": decision.decision_id,
        "attempted_pattern": primary_pattern(decision.attempted_pattern),
        "public_root": decision.public_root.0,
        "unsafe_site_id": decision.site_id,
        "operand_role": operand_role(&decision.operand_role),
        "source": {
            "identity": decision.source_identity,
            "kind": source_kind(decision.source_kind),
        },
        "controller": decision.controller,
        "capabilities": decision.capabilities,
        "requirement": decision.requirement,
        "primary_reason": decision.primary_reason.as_str(),
        "secondary_reasons": decision.secondary_reasons
            .into_iter()
            .map(|reason| reason.as_str())
            .collect::<Vec<_>>(),
        "exposure": null,
    })
}

fn write_unestablished_rows_ordered(
    writer: &mut impl Write,
    decisions: impl IntoIterator<Item = UnestablishedStructuralCandidate>,
    worker_count: usize,
    chunk_size: usize,
) -> Result<(), String> {
    let process = |decisions: Vec<UnestablishedStructuralCandidate>| {
        let row_count = decisions.len();
        let mut bytes = Vec::new();
        for decision in decisions {
            let mut row = serde_json::to_vec(&unestablished_json(decision)).map_err(|error| {
                format!("failed to serialize MVP unestablished decision: {error}")
            })?;
            bytes.append(&mut row);
            bytes.push(b'\n');
        }
        Ok(SerializedJsonChunk { bytes, row_count })
    };
    let mut consume = |chunk: SerializedJsonChunk| {
        writer
            .write_all(&chunk.bytes)
            .map_err(|error| format!("failed to write MVP unestablished decisions: {error}"))
    };
    process_ordered_chunks(
        decisions,
        worker_count,
        chunk_size,
        "unestablished",
        &process,
        &mut consume,
    )
}

fn write_unestablished_jsonl(
    path: &Path,
    decisions: Vec<UnestablishedStructuralCandidate>,
) -> Result<(), String> {
    validate_unestablished_decisions(&decisions)?;
    let file = fs::File::create(path).map_err(|error| {
        format!(
            "failed to create MVP unestablished output '{}': {error}",
            path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    write_unestablished_rows_ordered(
        &mut writer,
        decisions,
        configured_receipt_workers(),
        RECEIPT_CHUNK_SIZE,
    )?;
    writer
        .flush()
        .map_err(|error| format!("failed to flush MVP unestablished decisions: {error}"))
}

fn reorder_suppressed_candidates<T>(
    plans: &[SuppressionPlan],
    candidates: &mut [T],
) -> Result<(), String> {
    if plans.len() != candidates.len() {
        return Err("MVP suppression reorder length drift".to_string());
    }
    let mut target_at_position = vec![usize::MAX; plans.len()];
    for (target, plan) in plans.iter().enumerate() {
        let Some(slot) = target_at_position.get_mut(plan.original_index) else {
            return Err("MVP suppression plan has an out-of-range original index".to_string());
        };
        if *slot != usize::MAX {
            return Err("MVP suppression plan repeats an original index".to_string());
        }
        *slot = target;
    }
    if target_at_position
        .iter()
        .any(|target| *target == usize::MAX)
    {
        return Err("MVP suppression plan omits an original index".to_string());
    }
    for position in 0..target_at_position.len() {
        while target_at_position[position] != position {
            let target = target_at_position[position];
            candidates.swap(position, target);
            target_at_position.swap(position, target);
        }
    }
    Ok(())
}

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
    let active_candidates = analysis.structural_candidates.iter().collect::<Vec<_>>();
    let code = build_code_dictionary(&source_root, &active_candidates, &analysis.owner_code_spans)?;
    let mut findings = build_findings(
        &analysis.structural_candidates,
        &analysis.endpoint_graph,
        &analysis.owner_spans,
        &code.owner_code_ids,
        &code.safety_doc_ids,
    )?;
    findings.sort_unstable_by(|left, right| {
        let left = left["finding_id"].as_str().unwrap_or_default();
        let right = right["finding_id"].as_str().unwrap_or_default();
        left.cmp(right)
    });
    ensure_unique_finding_ids(&findings, "rap-unit-v3")?;

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
        "missing_sanitizer_warnings": analysis
            .missing_sanitizer_warnings
            .iter()
            .map(missing_sanitizer_warning_json)
            .collect::<Vec<_>>(),
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
        if !matches!(
            record.status,
            BindingStatus::BoundExact | BindingStatus::OptimizedAwayExact
        ) && !failures_by_site.contains_key(site_id)
        {
            return Err(format!(
                "non-bound unsafe site {site_id} lacks a coverage failure"
            ));
        }
        let candidate_count = finding_counts.get(site_id).copied().unwrap_or(0);
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
        "optimized_away_exact": status_counts.get("optimized_away_exact").copied().unwrap_or(0),
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
    let cycle_index = SelectedCycleIndex::new(graph, owner_spans);
    candidates
        .iter()
        .map(|candidate| finding_json(candidate, &cycle_index, code_ids, safety_doc_ids))
        .collect()
}

fn mvp_finding_json(
    candidate: &StructuralFindingCandidate,
    cycle_index: &SelectedCycleIndex,
    code_ids: &BTreeMap<FunctionKey, String>,
    safety_doc_ids: &BTreeMap<FunctionKey, String>,
) -> Result<Value, String> {
    validate_mvp_exposure(candidate.primary_pattern, candidate.exposure.is_some())?;
    let mut finding = finding_json(candidate, cycle_index, code_ids, safety_doc_ids)?;
    let subject_role = candidate
        .operand_role
        .as_ref()
        .map(operand_role)
        .unwrap_or_else(|| "result".to_string());
    finding
        .as_object_mut()
        .ok_or_else(|| "MVP finding is not a JSON object".to_string())?
        .insert("subject_role".to_string(), Value::String(subject_role));
    if let Some(exposure) = &candidate.exposure {
        finding["exposure"]["identity"] = Value::String(endpoint_identity(&exposure.node));
    }
    finding["contract"]["static_implication"] =
        static_contract_implication_json(candidate.contract().static_implication);
    Ok(finding)
}

fn validate_mvp_exposure(pattern: PrimaryPattern, has_exposure: bool) -> Result<(), String> {
    if pattern != PrimaryPattern::P3 && has_exposure {
        return Err(format!(
            "non-P3 {} candidate carries a P3-only exposure",
            primary_pattern(pattern)
        ));
    }
    Ok(())
}

fn mvp_candidate_finding_id(candidate: &StructuralFindingCandidate) -> String {
    let subject_role = candidate
        .operand_role
        .as_ref()
        .map(operand_role)
        .unwrap_or_else(|| "result".to_string());
    let source_identity = endpoint_identity(candidate.source.as_ref());
    let failure_kind = failure_kind(candidate.contract(), candidate.site().seed.kind);
    let exposure_identity = candidate
        .exposure
        .as_ref()
        .map(|exposure| endpoint_identity(&exposure.node));
    candidate_finding_id_from_parts(
        candidate,
        &subject_role,
        &source_identity,
        &failure_kind,
        exposure_identity.as_deref(),
    )
}

fn candidate_finding_id_from_parts(
    candidate: &StructuralFindingCandidate,
    subject_role: &str,
    source_identity: &str,
    failure_kind: &str,
    exposure_identity: Option<&str>,
) -> String {
    finding_id(
        candidate.primary_pattern,
        &candidate.site().seed.site_id(),
        subject_role,
        source_identity,
        failure_kind,
        candidate.first_failure(),
        candidate.public_root.as_ref(),
        exposure_identity,
    )
}

fn static_contract_implication_json(implication: StaticContractImplication) -> Value {
    match implication {
        StaticContractImplication::NotApplicable => json!({
            "status": "not_applicable",
            "evidence": null,
            "gap": null,
        }),
        StaticContractImplication::Proven(evidence) => json!({
            "status": "proven",
            "evidence": evidence.as_str(),
            "gap": null,
        }),
        StaticContractImplication::Unproven(gap) => json!({
            "status": "unproven",
            "evidence": null,
            "gap": gap.as_str(),
        }),
    }
}

fn replacement_key(candidate: &StructuralFindingCandidate) -> String {
    let role = candidate
        .operand_role
        .as_ref()
        .map(operand_role)
        .unwrap_or_else(|| "result".to_string());
    let exposure = candidate
        .exposure
        .as_ref()
        .map(|exposure| endpoint_identity(&exposure.node))
        .unwrap_or_default();
    format!(
        "{}|{}|{}|{}",
        candidate.site().seed.site_id(),
        candidate.public_root.0,
        role,
        exposure,
    )
}

fn finding_json(
    candidate: &StructuralFindingCandidate,
    cycle_index: &SelectedCycleIndex,
    code_ids: &BTreeMap<FunctionKey, String>,
    safety_doc_ids: &BTreeMap<FunctionKey, String>,
) -> Result<Value, String> {
    let root_code = code_ids
        .get(candidate.public_root.as_ref())
        .ok_or_else(|| {
            format!(
                "missing code dictionary entry for public root {}",
                candidate.public_root.0
            )
        })?;
    let source_identity = endpoint_identity(candidate.source.as_ref());
    let failure_kind = failure_kind(candidate.contract(), candidate.site().seed.kind);
    let exposure_identity = candidate
        .exposure
        .as_ref()
        .map(|exposure| endpoint_identity(&exposure.node));
    let subject_role = candidate
        .operand_role
        .as_ref()
        .map(operand_role)
        .unwrap_or_else(|| "result".to_string());
    let finding_id = candidate_finding_id_from_parts(
        candidate,
        &subject_role,
        &source_identity,
        &failure_kind,
        exposure_identity.as_deref(),
    );
    let propagation = propagation_json(candidate, cycle_index);
    let mut selected_code_ids = Vec::new();
    for_each_finding_owner(candidate, |owner| {
        if let Some(code_id) = code_ids.get(owner) {
            selected_code_ids.push(code_id);
        }
    });
    if let Some(code_id) = candidate
        .site()
        .seed
        .callee
        .as_ref()
        .and_then(|callee| code_ids.get(&FunctionKey::new(callee.clone())))
    {
        selected_code_ids.push(code_id);
    }
    let safety_doc_id = candidate
        .site()
        .seed
        .callee
        .as_ref()
        .and_then(|callee| safety_doc_ids.get(&FunctionKey::new(callee.clone())));
    if let Some(code_id) = safety_doc_id {
        selected_code_ids.push(code_id);
    }
    selected_code_ids.push(root_code);
    selected_code_ids.sort_unstable();
    selected_code_ids.dedup();
    let code_refs = selected_code_ids.into_iter().cloned().collect::<Vec<_>>();

    Ok(json!({
        "finding_id": finding_id,
        "primary_pattern": primary_pattern(candidate.primary_pattern),
        "mechanism_id": candidate.mechanism_id(),
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
            "generic_capability": generic_capability_json(candidate.source.as_ref()),
        },
        "first_contract_failure": {
            "kind": failure_kind,
            "span": span_json(candidate.first_failure()),
        },
        "unsafe_site": unsafe_site_json(candidate.site()),
        "exposure": candidate.exposure.as_ref().map(|exposure| json!({
            "kind": exposure_kind(&exposure.node),
            "span": span_json(&exposure.span),
        })),
        "contract": contract_json(candidate.contract(), candidate.site(), safety_doc_id),
        "propagation": propagation,
        "code_refs": code_refs,
    }))
}

fn generic_capability_json(source: &EndpointNode) -> Option<Value> {
    let EndpointNode::GenericCapability { evidence, .. } = source else {
        return None;
    };
    let associated_items = match &evidence.kind {
        super::endpoint_flow::GenericCapabilityKind::AssociatedSequenceShape {
            associated_items,
        } => associated_items.iter().cloned().collect::<Vec<_>>(),
        super::endpoint_flow::GenericCapabilityKind::PointerLayoutReinterpretation
        | super::endpoint_flow::GenericCapabilityKind::RepresentationPreservingPointerView => {
            Vec::new()
        }
    };
    Some(json!({
        "kind": evidence.kind.as_str(),
        "controllers": evidence.controllers().iter().map(|controller| json!({
            "kind": controller.kind.as_str(),
            "def_path": controller.def_path.clone(),
        })).collect::<Vec<_>>(),
        "associated_items": associated_items,
        "terms": evidence.terms.iter().map(|(role, term)| json!({
            "role": role.as_str(),
            "form": term.form.as_str(),
            "controllers": term.controllers.iter().map(|controller| json!({
                "kind": controller.kind.as_str(),
                "def_path": controller.def_path.clone(),
            })).collect::<Vec<_>>(),
            "residual_aliases": term.residual_aliases.iter().cloned().collect::<Vec<_>>(),
            "equality_class": term.equality_class,
            "type_facts": {
                "size": term.type_facts.size,
                "align": term.type_facts.align,
                "inhabited": term.type_facts.inhabited,
            },
        })).collect::<Vec<_>>(),
        "requirement": capability_requirement_json(&evidence.requirement),
        "witness": capability_witness_json(&evidence.witness),
    }))
}

fn capability_requirement_json(requirement: &CapabilityRequirement) -> Value {
    match requirement {
        CapabilityRequirement::AssociatedSequenceNonEmpty => json!({
            "kind": "associated_sequence_nonempty",
        }),
        CapabilityRequirement::PointerLayout { extent } => json!({
            "kind": "pointer_layout",
            "extent": match extent {
                AccessExtent::NotApplicable => json!({"kind": "not_applicable"}),
                AccessExtent::Zero => json!({"kind": "zero"}),
                AccessExtent::Positive(width) => json!({"kind": "positive", "width": width}),
                AccessExtent::Symbolic { controllers } => json!({
                    "kind": "symbolic",
                    "controllers": controllers.iter().map(|controller| json!({
                        "kind": controller.kind.as_str(),
                        "def_path": controller.def_path.clone(),
                    })).collect::<Vec<_>>(),
                }),
                AccessExtent::Unknown => json!({"kind": "unknown"}),
            },
        }),
        CapabilityRequirement::RepresentationPreservingView => json!({
            "kind": "representation_preserving_view",
        }),
    }
}

fn capability_witness_json(witness: &CapabilityWitness) -> Value {
    match witness {
        CapabilityWitness::Proven {
            assignment,
            violations,
        } => json!({
            "status": "proven",
            "assignment": assignment.iter().map(|(role, value)| json!({
                "role": role.as_str(),
                "type": value.as_str(),
            })).collect::<Vec<_>>(),
            "violations": violations.iter().map(|violation| violation.as_str()).collect::<Vec<_>>(),
        }),
        CapabilityWitness::ImpossibleByCompilerFacts { reason } => json!({
            "status": "impossible_by_compiler_facts",
            "reason": reason.as_str(),
        }),
        CapabilityWitness::Unestablished { reason } => json!({
            "status": "unestablished",
            "reason": reason.as_str(),
        }),
    }
}

fn finding_id(
    pattern: PrimaryPattern,
    site_id: &str,
    subject_role: &str,
    source_identity: &str,
    failure_kind: &str,
    failure_span: &StableSpan,
    root: &FunctionKey,
    exposure_identity: Option<&str>,
) -> String {
    sha256_hex(&length_prefixed(&[
        primary_pattern(pattern),
        site_id,
        subject_role,
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
    cycle_index: &SelectedCycleIndex,
) -> Value {
    let mut witness = vec![witness_step(
        "entry",
        candidate.public_root.as_ref(),
        &candidate.public_root_span,
    )];
    let source_owner = endpoint_owner_ref(candidate.source.as_ref());
    let source_before_calls =
        source_owner.is_some_and(|owner| owner == candidate.public_root.as_ref());
    if candidate.primary_pattern != PrimaryPattern::P3 && source_before_calls {
        witness.push(witness_step(
            "source",
            source_owner.unwrap_or(candidate.public_root.as_ref()),
            &candidate.source_span,
        ));
    }
    for edge in candidate.call_route.iter() {
        let kind = match edge.kind {
            CandidateCallKind::ResolvedLocal => "local_call",
            CandidateCallKind::Callback => "callback",
        };
        witness.push(witness_step(kind, &edge.callee, &edge.point.span));
    }
    if candidate.primary_pattern != PrimaryPattern::P3 && !source_before_calls {
        witness.push(witness_step(
            "source",
            source_owner.unwrap_or(&candidate.site().seed.owner),
            &candidate.source_span,
        ));
    }

    let (recursive, cycle_step) = cycle_index.selected_cycle(candidate);
    if let Some(cycle_step) = cycle_step {
        witness.push(cycle_step);
    }
    witness.push(witness_step(
        "unsafe_site",
        &candidate.site().seed.owner,
        &candidate.site().seed.raw_span.stable,
    ));
    if let Some(exposure) = &candidate.exposure {
        witness.push(witness_step(
            "exposure",
            endpoint_owner_ref(&exposure.node).unwrap_or(candidate.public_root.as_ref()),
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
            .any(|edge| endpoint_owner_ref(&edge.from) != endpoint_owner_ref(&edge.to))
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CompactNodeId(u32);

impl CompactNodeId {
    fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("endpoint cycle index node count exceeds u32"))
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Compact sparse rows of node IDs.  Rows preserve the same lexical neighbor
/// order as the old `BTreeMap<EndpointNode, BTreeSet<EndpointNode>>` graph, but
/// never clone a rich endpoint into adjacency storage.
struct CompactNodeRows {
    offsets: Vec<usize>,
    nodes: Vec<CompactNodeId>,
}

impl CompactNodeRows {
    fn from_edges(
        node_count: usize,
        edges: &[(CompactNodeId, CompactNodeId)],
        reverse: bool,
    ) -> Self {
        let mut counts = vec![0usize; node_count];
        for &(from, to) in edges {
            let row = if reverse { to } else { from };
            counts[row.index()] = counts[row.index()]
                .checked_add(1)
                .expect("endpoint cycle adjacency row length overflows usize");
        }
        let mut offsets = Vec::<usize>::with_capacity(node_count + 1);
        offsets.push(0);
        for count in counts {
            offsets.push(
                offsets
                    .last()
                    .copied()
                    .unwrap_or_default()
                    .checked_add(count)
                    .expect("endpoint cycle adjacency length overflows usize"),
            );
        }
        let mut cursors = offsets[..node_count].to_vec();
        let mut nodes = vec![CompactNodeId(0); edges.len()];
        for &(from, to) in edges {
            let (row, node) = if reverse { (to, from) } else { (from, to) };
            let cursor = &mut cursors[row.index()];
            nodes[*cursor] = node;
            *cursor += 1;
        }
        Self { offsets, nodes }
    }

    fn row(&self, node: CompactNodeId) -> &[CompactNodeId] {
        &self.nodes[self.offsets[node.index()]..self.offsets[node.index() + 1]]
    }
}

struct SelectedCycleIndex {
    // Only cyclic nodes survive construction.  They remain in the original
    // BTree node order, so a binary search replaces the old rich-key BTreeMap.
    cyclic_nodes: Vec<EndpointNode>,
    component_by_node: Vec<u32>,
    witnesses: Vec<Option<Value>>,
}

impl SelectedCycleIndex {
    fn new(graph: &EndpointFlowGraph, owner_spans: &BTreeMap<FunctionKey, StableSpan>) -> Self {
        Self::from_parts(
            graph.nodes.iter().cloned().collect(),
            &graph.edges,
            owner_spans,
        )
    }

    fn from_graph(
        graph: EndpointFlowGraph,
        owner_spans: &BTreeMap<FunctionKey, StableSpan>,
    ) -> Self {
        let EndpointFlowGraph { nodes, edges } = graph;
        let index = Self::from_parts(nodes.into_iter().collect(), &edges, owner_spans);
        // In the MVP-only path the rich edge set is released here, before any
        // finding JSON is materialized.
        drop(edges);
        index
    }

    fn from_parts(
        nodes: Vec<EndpointNode>,
        edges: &BTreeSet<EndpointEdge>,
        owner_spans: &BTreeMap<FunctionKey, StableSpan>,
    ) -> Self {
        let mut indexed_edges = Vec::with_capacity(edges.len());
        let mut self_edges = vec![false; nodes.len()];
        for edge in edges {
            let from = CompactNodeId::from_index(
                nodes
                    .binary_search(&edge.from)
                    .expect("endpoint edge source must be present in graph nodes"),
            );
            let to = CompactNodeId::from_index(
                nodes
                    .binary_search(&edge.to)
                    .expect("endpoint edge target must be present in graph nodes"),
            );
            self_edges[from.index()] |= from == to;
            indexed_edges.push((from, to));
        }
        let outgoing = CompactNodeRows::from_edges(nodes.len(), &indexed_edges, false);
        let incoming = CompactNodeRows::from_edges(nodes.len(), &indexed_edges, true);

        // Iterative Kosaraju with the exact root, neighbor and finish order of
        // EndpointFlowGraph::cyclic_components.
        let mut visited = vec![false; nodes.len()];
        let mut finish_order = Vec::with_capacity(nodes.len());
        for root_index in 0..nodes.len() {
            if visited[root_index] {
                continue;
            }
            visited[root_index] = true;
            let root = CompactNodeId::from_index(root_index);
            let mut stack = vec![(root, false)];
            while let Some((node, exiting)) = stack.pop() {
                if exiting {
                    finish_order.push(node);
                    continue;
                }
                stack.push((node, true));
                for &successor in outgoing.row(node).iter().rev() {
                    if !visited[successor.index()] {
                        visited[successor.index()] = true;
                        stack.push((successor, false));
                    }
                }
            }
        }

        visited.fill(false);
        let mut components = Vec::<Vec<CompactNodeId>>::new();
        while let Some(root) = finish_order.pop() {
            if visited[root.index()] {
                continue;
            }
            visited[root.index()] = true;
            let mut component = Vec::new();
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                component.push(node);
                for &predecessor in incoming.row(node).iter().rev() {
                    if !visited[predecessor.index()] {
                        visited[predecessor.index()] = true;
                        stack.push(predecessor);
                    }
                }
            }
            component.sort_unstable();
            if component.len() > 1 || self_edges[component[0].index()] {
                components.push(component);
            }
        }
        components.sort_unstable();

        let mut compact_component_by_node = vec![None::<u32>; nodes.len()];
        for (component_index, component) in components.iter().enumerate() {
            let component_index =
                u32::try_from(component_index).expect("endpoint cycle component count exceeds u32");
            for &node in component {
                compact_component_by_node[node.index()] = Some(component_index);
            }
        }

        let mut witnesses = vec![None; components.len()];
        for (edge, &(from, to)) in edges.iter().zip(&indexed_edges) {
            let Some(component) = compact_component_by_node[from.index()] else {
                continue;
            };
            let component_index = component as usize;
            if witnesses[component_index].is_some()
                || compact_component_by_node[to.index()] != Some(component)
            {
                continue;
            }
            if let Some(point) = &edge.point {
                witnesses[component_index] =
                    Some(witness_step("scc_cycle", &point.function, &point.span));
            }
        }
        for (component_index, component) in components.iter().enumerate() {
            if witnesses[component_index].is_some() {
                continue;
            }
            witnesses[component_index] = component
                .iter()
                .map(|node| &nodes[node.index()])
                .find_map(endpoint_owner_ref)
                .and_then(|owner| {
                    owner_spans
                        .get(owner)
                        .map(|span| witness_step("scc_cycle", owner, span))
                });
        }

        let mut cyclic_nodes = Vec::new();
        let mut component_by_node = Vec::new();
        for (node, component) in nodes.into_iter().zip(compact_component_by_node) {
            if let Some(component) = component {
                cyclic_nodes.push(node);
                component_by_node.push(component);
            }
        }
        Self {
            cyclic_nodes,
            component_by_node,
            witnesses,
        }
    }

    fn selected_cycle(&self, candidate: &StructuralFindingCandidate) -> (bool, Option<Value>) {
        self.selected_cycle_for_nodes(
            std::iter::once(candidate.source.as_ref()).chain(
                candidate
                    .endpoint_route
                    .iter()
                    .flat_map(|edge| [&edge.from, &edge.to]),
            ),
        )
    }

    fn selected_cycle_for_nodes<'a>(
        &self,
        nodes: impl Iterator<Item = &'a EndpointNode>,
    ) -> (bool, Option<Value>) {
        let component = nodes
            .filter_map(|node| {
                self.cyclic_nodes
                    .binary_search(node)
                    .ok()
                    .map(|index| self.component_by_node[index])
            })
            // The old per-finding walk continued past an intersecting
            // component when neither an edge point nor its first endpoint
            // owner had a span.  Preserve that detail by selecting the first
            // intersecting component that actually has a witness.
            .filter(|component| self.witnesses[*component as usize].is_some())
            .min()
            .map(|component| component as usize);
        match component {
            Some(index) => (true, self.witnesses[index].clone()),
            None => (false, None),
        }
    }
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

#[derive(Default)]
struct CodeDictionaryInputs {
    owners: BTreeSet<FunctionKey>,
    local_callees: BTreeSet<FunctionKey>,
}

fn collect_code_dictionary_inputs_ordered<'candidate>(
    candidates: impl IntoIterator<Item = &'candidate StructuralFindingCandidate>,
    worker_count: usize,
    chunk_size: usize,
) -> Result<(BTreeSet<FunctionKey>, BTreeSet<FunctionKey>), String> {
    let process = |candidates: Vec<&'candidate StructuralFindingCandidate>| {
        let mut inputs = CodeDictionaryInputs::default();
        for candidate in candidates {
            if let Some(callee) = &candidate.site().seed.callee {
                inputs
                    .local_callees
                    .insert(FunctionKey::new(callee.clone()));
            }
            for_each_finding_owner(candidate, |owner| {
                inputs.owners.insert(owner.clone());
            });
        }
        Ok(inputs)
    };
    let mut owners = BTreeSet::new();
    let mut local_callees = BTreeSet::new();
    let mut consume = |mut inputs: CodeDictionaryInputs| {
        owners.append(&mut inputs.owners);
        local_callees.append(&mut inputs.local_callees);
        Ok(())
    };
    process_ordered_chunks(
        candidates,
        worker_count,
        chunk_size,
        "code-owner-collect",
        &process,
        &mut consume,
    )?;
    Ok((owners, local_callees))
}

fn build_code_dictionary(
    source_root: &Path,
    candidates: &[&StructuralFindingCandidate],
    owner_code_spans: &BTreeMap<FunctionKey, StableSpan>,
) -> Result<CodeDictionary, String> {
    let local_callee_names = candidates
        .iter()
        .copied()
        .filter_map(|candidate| candidate.site().seed.callee.as_ref())
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let local_callees = local_callee_names
        .into_iter()
        .map(FunctionKey::new)
        .filter(|callee| owner_code_spans.contains_key(callee))
        .collect::<BTreeSet<_>>();
    let mut owners = BTreeSet::new();
    for candidate in candidates.iter().copied() {
        for_each_finding_owner(candidate, |owner| {
            if !owners.contains(owner) {
                owners.insert(owner.clone());
            }
        });
    }
    owners.extend(local_callees.iter().cloned());
    build_code_dictionary_for_owners(
        source_root,
        owners,
        &local_callees,
        owner_code_spans,
        |source_path| {
            fs::read_to_string(source_path).map_err(|error| {
                format!(
                    "failed to read causal source '{}': {error}",
                    source_path.display()
                )
            })
        },
    )
}

fn build_code_dictionary_for_owners(
    source_root: &Path,
    owners: BTreeSet<FunctionKey>,
    local_callees: &BTreeSet<FunctionKey>,
    owner_code_spans: &BTreeMap<FunctionKey, StableSpan>,
    mut read_source: impl FnMut(&Path) -> Result<String, String>,
) -> Result<CodeDictionary, String> {
    let mut owners_by_path = BTreeMap::<String, Vec<FunctionKey>>::new();
    for owner in owners {
        let span = owner_code_spans
            .get(&owner)
            .ok_or_else(|| format!("missing local body span for causal function {}", owner.0))?;
        owners_by_path
            .entry(span.path.clone())
            .or_default()
            .push(owner);
    }
    let mut blocks = Vec::new();
    let mut owner_code_ids = BTreeMap::new();
    let mut safety_doc_ids = BTreeMap::new();
    for (relative_path, owners) in owners_by_path {
        let source_path = source_root.join(relative_path);
        let source = read_source(&source_path)?;
        let source_file = SourceFile::new(&source);
        for owner in owners {
            let span = owner_code_spans
                .get(&owner)
                .expect("code dictionary owners were validated while grouping by source path");
            let text = source_file.slice_span(span)?;
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
                if let Some((doc_span, doc_text)) = source_file.adjacent_safety_doc(span)? {
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
    }
    blocks.sort_unstable_by(|left, right| {
        left["code_id"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["code_id"].as_str().unwrap_or_default())
    });
    Ok(CodeDictionary {
        blocks,
        owner_code_ids,
        safety_doc_ids,
    })
}

struct SourceFile<'source> {
    source: &'source str,
    lines: Vec<&'source str>,
    line_starts: Vec<usize>,
}

impl<'source> SourceFile<'source> {
    fn new(source: &'source str) -> Self {
        let lines = source.split_inclusive('\n').collect::<Vec<_>>();
        let mut cursor = 0usize;
        let line_starts = lines
            .iter()
            .map(|line| {
                let start = cursor;
                cursor += line.len();
                start
            })
            .collect();
        Self {
            source,
            lines,
            line_starts,
        }
    }

    fn adjacent_safety_doc(
        &self,
        function_span: &StableSpan,
    ) -> Result<Option<(StableSpan, String)>, String> {
        let function_line = usize::try_from(function_span.start.line - 1).unwrap();
        if function_line >= self.lines.len() {
            return Err(format!(
                "function body starts outside its source file: {}",
                function_span.token()
            ));
        }
        let mut cursor = function_line;
        while cursor > 0 && self.lines[cursor - 1].trim_start().starts_with("#[") {
            cursor -= 1;
        }
        let docs_end = cursor;
        while cursor > 0 && self.lines[cursor - 1].trim_start().starts_with("///") {
            cursor -= 1;
        }
        if cursor == docs_end {
            return Ok(None);
        }
        let text = self.lines[cursor..docs_end].concat();
        if !text.contains("# Safety") {
            return Ok(None);
        }
        let last = self.lines[docs_end - 1].trim_end_matches(['\r', '\n']);
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
        Ok(Some((span.clone(), self.slice_span(&span)?)))
    }

    fn slice_span(&self, span: &StableSpan) -> Result<String, String> {
        let start_line = usize::try_from(span.start.line - 1).unwrap();
        let end_line = usize::try_from(span.end.line - 1).unwrap();
        let start_column = usize::try_from(span.start.column - 1).unwrap();
        let end_column = usize::try_from(span.end.column - 1).unwrap();
        let start = self.line_offset(start_line, start_column, span)?;
        let end = self.line_offset(end_line, end_column, span)?;
        self.source
            .get(start..end)
            .map(str::to_string)
            .ok_or_else(|| {
                format!(
                    "causal function span is not on UTF-8 boundaries: {}",
                    span.token()
                )
            })
    }

    fn line_offset(&self, line: usize, column: usize, span: &StableSpan) -> Result<usize, String> {
        let text = self.lines.get(line).ok_or_else(|| {
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
        Ok(self.line_starts[line] + column)
    }
}

#[cfg(test)]
fn adjacent_safety_doc(
    source: &str,
    function_span: &StableSpan,
) -> Result<Option<(StableSpan, String)>, String> {
    SourceFile::new(source).adjacent_safety_doc(function_span)
}

fn for_each_finding_owner<'candidate>(
    candidate: &'candidate StructuralFindingCandidate,
    mut visit: impl FnMut(&'candidate FunctionKey),
) {
    visit(candidate.public_root.as_ref());
    visit(&candidate.site().seed.owner);
    for edge in candidate.call_route.iter() {
        visit(&edge.caller);
        visit(&edge.callee);
    }
    for edge in candidate.endpoint_route.iter() {
        if let Some(owner) = endpoint_owner_ref(&edge.from) {
            visit(owner);
        }
        if let Some(owner) = endpoint_owner_ref(&edge.to) {
            visit(owner);
        }
    }
    if let Some(exposure) = &candidate.exposure {
        if let Some(owner) = endpoint_owner_ref(&exposure.node) {
            visit(owner);
        }
    }
}

#[cfg(test)]
fn slice_span(source: &str, span: &StableSpan) -> Result<String, String> {
    SourceFile::new(source).slice_span(span)
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
        match path.strip_prefix(&root) {
            Ok(relative) => relative.to_path_buf(),
            Err(_) if path.file_name().is_some_and(|name| name == "Cargo.toml") => {
                let parent = path.parent().and_then(Path::file_name).ok_or_else(|| {
                    format!("metadata path '{}' escapes project root", path.display())
                })?;
                let name = path.file_name().ok_or_else(|| {
                    format!("metadata path '{}' escapes project root", path.display())
                })?;
                PathBuf::from("_isolate").join(parent).join(name)
            }
            Err(_) => {
                return Err(format!(
                    "metadata path '{}' escapes project root",
                    path.display()
                ));
            }
        }
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
        EndpointNode::Capture { owner, index } => format!("{}:capture:{index}", owner.0),
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
            "public_field:owner={}:field={}:access={}:base={}:storage={}:versions={}:adt={}:persistent={}@{}",
            access_owner.0,
            def_path,
            access_path.0,
            base.0,
            storage_provenance_identity(storage),
            storage_versions
                .iter()
                .map(|version| version.stable_identity())
                .collect::<Vec<_>>()
                .join(","),
            adt_kind.as_str(),
            persistent,
            span.token(),
        ),
        EndpointNode::ExternalBehavior {
            owner,
            kind,
            controller,
            token,
            span,
        } => {
            format!(
                "{}:external:{}:{token}:controller={}@{}",
                owner.0,
                external_kind(*kind),
                behavior_controller_identity(controller),
                span.token(),
            )
        }
        EndpointNode::InternalDerivation {
            owner,
            kind,
            token,
            point,
        } => {
            format!(
                "{}:derivation:{}:{token}@{}",
                owner.0,
                kind.as_str(),
                program_point_identity(point),
            )
        }
        EndpointNode::GenericCapability {
            owner,
            evidence,
            span,
        } => {
            format!(
                "{}:generic_capability:{}@{}",
                owner.0,
                evidence.source_identity(),
                span.token(),
            )
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
        EndpointNode::CallResult { callsite, output } => format!(
            "call_result:{}:{}",
            callsite.stable_identity(),
            match output {
                super::endpoint_flow::CallSiteOutput::Return => "return".to_string(),
                super::endpoint_flow::CallSiteOutput::Out(index) => format!("out:{index}"),
            }
        ),
        EndpointNode::Exposure {
            owner,
            token,
            point,
        } => {
            format!(
                "{}:exposure:{token}@{}",
                owner.0,
                program_point_identity(point)
            )
        }
    }
}

fn program_point_identity(point: &ProgramPoint) -> String {
    format!(
        "{}:{}:{}@{}",
        point.function.0,
        point.block,
        point.statement,
        point.span.token()
    )
}

fn behavior_controller_identity(controller: &BehaviorController) -> String {
    match controller {
        BehaviorController::OpenTrait {
            trait_id,
            openness,
            resolved_target,
            receiver,
        } => format!(
            "open_trait:{}:{}:{}:{}",
            trait_id.0,
            match openness {
                BehaviorOpenness::ProvenOpen => "proven_open",
                BehaviorOpenness::ProvenClosed => "proven_closed",
                BehaviorOpenness::Unknown => "unknown",
            },
            resolved_target
                .as_ref()
                .map(|target| target.0.as_str())
                .unwrap_or("none"),
            receiver.stable_identity(),
        ),
        BehaviorController::PublicFormal { owner, index } => {
            format!("public_formal:{}:{index}", owner.0)
        }
        BehaviorController::PublicField {
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
            "public_field:owner={}:field={}:access={}:base={}:storage={}:versions={}:adt={}:persistent={}@{}",
            access_owner.0,
            def_path,
            access_path.0,
            base.0,
            storage_provenance_identity(storage),
            storage_versions
                .iter()
                .map(|version| version.stable_identity())
                .collect::<Vec<_>>()
                .join(","),
            adt_kind.as_str(),
            persistent,
            span.token(),
        ),
        BehaviorController::InternalFunction { function } => {
            format!("internal_function:{}", function.0)
        }
        BehaviorController::InternalClosure { function } => {
            format!("internal_closure:{}", function.0)
        }
        BehaviorController::FfiHost => "ffi_host".to_string(),
        BehaviorController::Unknown => "unknown".to_string(),
    }
}

fn storage_provenance_identity(storage: &StorageProvenance) -> String {
    match storage {
        StorageProvenance::Untracked => "untracked".to_string(),
        StorageProvenance::DefinitelyLocal => "definitely_local".to_string(),
        StorageProvenance::EntryFormal { owner, index } => {
            format!("entry_formal:{}:{index}", owner.0)
        }
        StorageProvenance::MaybeExternal => "maybe_external".to_string(),
    }
}

fn endpoint_owner_ref(node: &EndpointNode) -> Option<&FunctionKey> {
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

fn ensure_unique_finding_ids(findings: &[Value], label: &str) -> Result<(), String> {
    let mut previous = None::<(&str, &Value)>;
    let mut ordered = true;
    for finding in findings {
        let finding_id = finding["finding_id"]
            .as_str()
            .filter(|finding_id| !finding_id.is_empty())
            .ok_or_else(|| format!("{label} finding lacks finding_id"))?;
        if let Some((previous_id, previous_finding)) = previous {
            if previous_id == finding_id {
                let collision = if previous_finding == finding {
                    "duplicate candidate"
                } else {
                    "identity collision between different candidates"
                };
                return Err(format!("{label} {collision}: {finding_id}"));
            }
            ordered &= previous_id < finding_id;
        }
        previous = Some((finding_id, finding));
    }
    if ordered {
        return Ok(());
    }

    let mut by_id = BTreeMap::<&str, &Value>::new();
    for finding in findings {
        let finding_id = finding["finding_id"]
            .as_str()
            .expect("finding IDs were validated above");
        if let Some(previous) = by_id.insert(finding_id, finding) {
            let collision = if previous == finding {
                "duplicate candidate"
            } else {
                "identity collision between different candidates"
            };
            return Err(format!("{label} {collision}: {finding_id}"));
        }
    }
    Ok(())
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
        ClassificationConfidence::Heuristic => "heuristic",
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
        BindingStatus::OptimizedAwayExact => "optimized_away_exact",
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
    use super::super::dataflow::StaticWriteToken;
    use super::super::endpoint_flow::{
        AccessExtent, CapabilityImpossibleReason, CapabilityRequirement, CapabilityTerm,
        CapabilityTermForm, CapabilityTermRole, CapabilityTypeFacts, CapabilityUnestablishedReason,
        CapabilityWitness, DerivationKind, EndpointEdge, EndpointEdgeKind,
        GenericCapabilityEvidence, GenericCapabilityKind, GenericParamId, GenericParamKind,
        PublicFieldAdtKind, StorageProvenance,
    };
    use super::super::summary::{PlaceKey, ProgramPoint, StablePosition};
    use super::super::unsafe_site::RawSpanAnchor;
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
            "argument:0",
            "sample::root:formal:1",
            "precondition:unsafe_call",
            &span(2),
            &root,
            None,
        );
        let same = finding_id(
            PrimaryPattern::P1,
            &"a".repeat(64),
            "argument:0",
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
                "argument:0",
                "sample::root:formal:1",
                "precondition:unsafe_call",
                &span(2),
                &root,
                None,
            )
        );
        assert_ne!(
            first,
            finding_id(
                PrimaryPattern::P1,
                &"a".repeat(64),
                "argument:1",
                "sample::root:formal:1",
                "precondition:unsafe_call",
                &span(2),
                &root,
                None,
            )
        );
    }

    #[test]
    fn candidate_identity_collisions_fail_closed_instead_of_deduplicating() {
        let first = json!({"finding_id": "a".repeat(64), "subject_role": "argument:0"});
        let same = first.clone();
        assert!(ensure_unique_finding_ids(&[first.clone(), same], "test")
            .unwrap_err()
            .contains("duplicate candidate"));

        let different = json!({"finding_id": "a".repeat(64), "subject_role": "argument:1"});
        assert!(ensure_unique_finding_ids(&[first, different], "test")
            .unwrap_err()
            .contains("identity collision between different candidates"));
    }

    #[test]
    fn generic_capability_source_span_prevents_same_evidence_identity_collisions() {
        let owner = FunctionKey::new("sample::root");
        let evidence = GenericCapabilityEvidence {
            kind: GenericCapabilityKind::PointerLayoutReinterpretation,
            terms: BTreeMap::from([(
                CapabilityTermRole::TargetPointee,
                CapabilityTerm {
                    form: CapabilityTermForm::DirectParam,
                    controllers: BTreeSet::from([GenericParamId {
                        def_path: "sample::root::T".to_string(),
                        kind: GenericParamKind::Type,
                    }]),
                    residual_aliases: BTreeSet::new(),
                    equality_class: None,
                    type_facts: CapabilityTypeFacts::default(),
                },
            )]),
            requirement: CapabilityRequirement::PointerLayout {
                extent: AccessExtent::Unknown,
            },
            witness: CapabilityWitness::ImpossibleByCompilerFacts {
                reason: CapabilityImpossibleReason::ConcreteRootConfiguration,
            },
        };
        let first = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence: evidence.clone(),
            span: span(3),
        };
        let second = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence,
            span: span(4),
        };
        let first_source = endpoint_identity(&first);
        let second_source = endpoint_identity(&second);
        assert_ne!(first_source, second_source);
        assert!(first_source.ends_with("@src/lib.rs:3:1-3:2"));
        assert!(second_source.ends_with("@src/lib.rs:4:1-4:2"));

        let site_id = "a".repeat(64);
        let first_finding = finding_id(
            PrimaryPattern::P5,
            &site_id,
            "argument:0",
            &first_source,
            "precondition:unsafe_call",
            &span(8),
            &owner,
            None,
        );
        let second_finding = finding_id(
            PrimaryPattern::P5,
            &site_id,
            "argument:0",
            &second_source,
            "precondition:unsafe_call",
            &span(8),
            &owner,
            None,
        );
        assert_ne!(first_finding, second_finding);
    }

    #[test]
    fn generic_capability_source_identity_ignores_selected_witness_but_reports_it() {
        let owner = FunctionKey::new("sample::root");
        let base = GenericCapabilityEvidence {
            kind: GenericCapabilityKind::PointerLayoutReinterpretation,
            terms: BTreeMap::from([(
                CapabilityTermRole::TargetPointee,
                CapabilityTerm {
                    form: CapabilityTermForm::DirectParam,
                    controllers: BTreeSet::from([GenericParamId {
                        def_path: "sample::root::T".to_string(),
                        kind: GenericParamKind::Type,
                    }]),
                    residual_aliases: BTreeSet::new(),
                    equality_class: None,
                    type_facts: CapabilityTypeFacts::default(),
                },
            )]),
            requirement: CapabilityRequirement::PointerLayout {
                extent: AccessExtent::Positive(8),
            },
            witness: CapabilityWitness::ImpossibleByCompilerFacts {
                reason: CapabilityImpossibleReason::ConcreteRootConfiguration,
            },
        };
        let mut alternate = base.clone();
        alternate.witness = CapabilityWitness::Unestablished {
            reason: CapabilityUnestablishedReason::NoFiniteWitness,
        };

        assert_eq!(base.source_identity(), alternate.source_identity());
        assert_ne!(base.stable_identity(), alternate.stable_identity());

        let first = EndpointNode::GenericCapability {
            owner: owner.clone(),
            evidence: base,
            span: span(3),
        };
        let second = EndpointNode::GenericCapability {
            owner,
            evidence: alternate,
            span: span(3),
        };
        let first_identity = endpoint_identity(&first);
        let second_identity = endpoint_identity(&second);
        assert_eq!(first_identity, second_identity);
        assert_eq!(
            finding_id(
                PrimaryPattern::P5,
                &"a".repeat(64),
                "place",
                &first_identity,
                "precondition:raw_pointer_dereference",
                &span(8),
                &FunctionKey::new("sample::root"),
                None,
            ),
            finding_id(
                PrimaryPattern::P5,
                &"a".repeat(64),
                "place",
                &second_identity,
                "precondition:raw_pointer_dereference",
                &span(8),
                &FunctionKey::new("sample::root"),
                None,
            )
        );
        assert_ne!(
            generic_capability_json(&first).unwrap()["witness"],
            generic_capability_json(&second).unwrap()["witness"]
        );
    }

    #[test]
    fn typed_field_and_derivation_facts_participate_in_endpoint_identity() {
        let owner = FunctionKey::new("sample::root");
        let field = |access_path: &str, versions| EndpointNode::PublicField {
            access_owner: owner.clone(),
            def_path: "sample::State::value".to_string(),
            access_path: PlaceKey::new(access_path),
            base: PlaceKey::new("_1"),
            storage: StorageProvenance::EntryFormal {
                owner: owner.clone(),
                index: 1,
            },
            storage_versions: versions,
            adt_kind: PublicFieldAdtKind::Struct,
            persistent: true,
            span: span(3),
        };
        let first_field = endpoint_identity(&field(
            "_1.0",
            BTreeSet::from([StaticWriteToken::new(0, 1)]),
        ));
        let other_access = endpoint_identity(&field(
            "_1.1",
            BTreeSet::from([StaticWriteToken::new(0, 1)]),
        ));
        let other_version = endpoint_identity(&field(
            "_1.0",
            BTreeSet::from([StaticWriteToken::new(0, 2)]),
        ));
        assert_ne!(first_field, other_access);
        assert_ne!(first_field, other_version);
        assert!(first_field.contains("storage=entry_formal:sample::root:1"));

        let point = ProgramPoint {
            function: owner.clone(),
            block: 0,
            statement: 1,
            span: span(4),
        };
        let index = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Index,
            token: "_2".to_string(),
            point: point.clone(),
        };
        let alignment = EndpointNode::InternalDerivation {
            owner: owner.clone(),
            kind: DerivationKind::Alignment,
            token: "_2".to_string(),
            point: point.clone(),
        };
        assert_ne!(endpoint_identity(&index), endpoint_identity(&alignment));

        let later_at_same_span = EndpointNode::InternalDerivation {
            owner,
            kind: DerivationKind::Index,
            token: "_2".to_string(),
            point: ProgramPoint {
                statement: 2,
                ..point
            },
        };
        assert_ne!(
            endpoint_identity(&index),
            endpoint_identity(&later_at_same_span)
        );
    }

    #[test]
    fn mvp_contract_reports_static_proof_and_unresolved_gap_without_string_guessing() {
        use super::super::contract_catalog::{StaticContractEvidence, StaticContractGap};

        assert_eq!(
            static_contract_implication_json(StaticContractImplication::Proven(
                StaticContractEvidence::TargetAllBitsValidByteForm,
            )),
            json!({
                "status": "proven",
                "evidence": "target_all_bits_valid_byte_form",
                "gap": null,
            })
        );
        assert_eq!(
            static_contract_implication_json(StaticContractImplication::Unproven(
                StaticContractGap::TargetRestrictedValidity,
            )),
            json!({
                "status": "unproven",
                "evidence": null,
                "gap": "target_restricted_validity",
            })
        );
        assert_eq!(
            static_contract_implication_json(StaticContractImplication::NotApplicable)["status"],
            "not_applicable"
        );
    }

    #[test]
    fn external_source_identity_keeps_typed_controller_alternatives_distinct() {
        let owner = FunctionKey::new("sample::root");
        let make = |controller| EndpointNode::ExternalBehavior {
            owner: owner.clone(),
            kind: ExternalSourceKind::FunctionPointer,
            controller,
            token: "call-result".to_string(),
            span: span(3),
        };
        let unknown = endpoint_identity(&make(BehaviorController::Unknown));
        let internal = endpoint_identity(&make(BehaviorController::InternalFunction {
            function: FunctionKey::new("sample::identity"),
        }));
        assert_ne!(unknown, internal);
        assert!(internal.contains("internal_function:sample::identity"));

        let at_other_span = EndpointNode::ExternalBehavior {
            owner,
            kind: ExternalSourceKind::FunctionPointer,
            controller: BehaviorController::Unknown,
            token: "call-result".to_string(),
            span: span(4),
        };
        assert_ne!(unknown, endpoint_identity(&at_other_span));
    }

    #[test]
    fn public_field_behavior_controller_identity_keeps_storage_context() {
        let owner = FunctionKey::new("sample::root");
        let make = |access_path: &str, version| BehaviorController::PublicField {
            access_owner: owner.clone(),
            def_path: "sample::State::callback".to_string(),
            access_path: PlaceKey::new(access_path),
            base: PlaceKey::new("_1"),
            storage: StorageProvenance::EntryFormal {
                owner: owner.clone(),
                index: 1,
            },
            storage_versions: BTreeSet::from([StaticWriteToken::new(0, version)]),
            adt_kind: PublicFieldAdtKind::Struct,
            persistent: true,
            span: span(3),
        };
        let first = behavior_controller_identity(&make("_1.0", 1));
        assert!(first.contains("owner=sample::root"));
        assert!(first.contains("access=_1.0"));
        assert!(first.contains("entry_formal:sample::root:1"));
        assert_ne!(first, behavior_controller_identity(&make("_1.1", 1)));
        assert_ne!(first, behavior_controller_identity(&make("_1.0", 2)));
    }

    #[test]
    fn exposure_identity_includes_the_exact_program_point() {
        let owner = FunctionKey::new("sample::root");
        let point = ProgramPoint {
            function: owner.clone(),
            block: 1,
            statement: 2,
            span: span(7),
        };
        let first = EndpointNode::Exposure {
            owner: owner.clone(),
            token: "_3".to_string(),
            point: point.clone(),
        };
        let second = EndpointNode::Exposure {
            owner,
            token: "_3".to_string(),
            point: ProgramPoint {
                statement: 3,
                ..point
            },
        };
        assert_ne!(endpoint_identity(&first), endpoint_identity(&second));
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
    fn grouped_code_dictionary_matches_ungrouped_bytes_and_reads_each_file_once() {
        let root = Path::new("/virtual-source-root");
        let first_source =
            "/// Reads one byte.\n/// # Safety\n#[inline]\nunsafe fn a() {}\nfn b() {}\n";
        let second_source = "fn c() {}\n";
        let owner_a = FunctionKey::new("sample::a");
        let owner_b = FunctionKey::new("sample::b");
        let owner_c = FunctionKey::new("sample::c");
        let line_span = |path: &str, line: u32, text: &str| {
            StableSpan::new(
                path,
                StablePosition::new(line, 1),
                StablePosition::new(line, u32::try_from(text.len() + 1).unwrap()),
            )
        };
        let spans = BTreeMap::from([
            (
                owner_a.clone(),
                line_span("src/lib.rs", 4, "unsafe fn a() {}"),
            ),
            (owner_b.clone(), line_span("src/lib.rs", 5, "fn b() {}")),
            (owner_c.clone(), line_span("src/other.rs", 1, "fn c() {}")),
        ]);
        let owners = BTreeSet::from([owner_a.clone(), owner_b.clone(), owner_c.clone()]);
        let local_callees = BTreeSet::from([owner_a, owner_c]);
        let sources = BTreeMap::from([
            (root.join("src/lib.rs"), first_source.to_string()),
            (root.join("src/other.rs"), second_source.to_string()),
        ]);

        let mut grouped_reads = BTreeMap::<PathBuf, usize>::new();
        let grouped = build_code_dictionary_for_owners(
            root,
            owners.clone(),
            &local_callees,
            &spans,
            |path| {
                *grouped_reads.entry(path.to_path_buf()).or_default() += 1;
                sources
                    .get(path)
                    .cloned()
                    .ok_or_else(|| format!("missing test source {}", path.display()))
            },
        )
        .unwrap();

        // Compose the same dictionary one owner at a time.  This is the old
        // read/split lifecycle while retaining the exact block construction
        // rules, and therefore provides a byte-for-byte differential oracle.
        let mut ungrouped_reads = BTreeMap::<PathBuf, usize>::new();
        let mut ungrouped_blocks = Vec::new();
        let mut ungrouped_owner_ids = BTreeMap::new();
        let mut ungrouped_safety_ids = BTreeMap::new();
        for owner in owners {
            let partial = build_code_dictionary_for_owners(
                root,
                BTreeSet::from([owner]),
                &local_callees,
                &spans,
                |path| {
                    *ungrouped_reads.entry(path.to_path_buf()).or_default() += 1;
                    sources
                        .get(path)
                        .cloned()
                        .ok_or_else(|| format!("missing test source {}", path.display()))
                },
            )
            .unwrap();
            ungrouped_blocks.extend(partial.blocks);
            ungrouped_owner_ids.extend(partial.owner_code_ids);
            ungrouped_safety_ids.extend(partial.safety_doc_ids);
        }
        ungrouped_blocks.sort_unstable_by(|left, right| {
            left["code_id"]
                .as_str()
                .unwrap_or_default()
                .cmp(right["code_id"].as_str().unwrap_or_default())
        });

        assert_eq!(
            serde_json::to_vec(&grouped.blocks).unwrap(),
            serde_json::to_vec(&ungrouped_blocks).unwrap()
        );
        assert_eq!(grouped.owner_code_ids, ungrouped_owner_ids);
        assert_eq!(grouped.safety_doc_ids, ungrouped_safety_ids);
        assert_eq!(grouped_reads[&root.join("src/lib.rs")], 1);
        assert_eq!(grouped_reads[&root.join("src/other.rs")], 1);
        assert_eq!(ungrouped_reads[&root.join("src/lib.rs")], 2);
        assert_eq!(ungrouped_reads[&root.join("src/other.rs")], 1);
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

    #[test]
    fn cached_cycle_selection_matches_the_exhaustive_per_finding_oracle() {
        let owner_a = FunctionKey::new("sample::a");
        let owner_b = FunctionKey::new("sample::b");
        let owner_c = FunctionKey::new("sample::c");
        let a0 = EndpointNode::Formal {
            owner: owner_a.clone(),
            index: 0,
        };
        let a1 = EndpointNode::Return {
            owner: owner_a.clone(),
        };
        let b0 = EndpointNode::Formal {
            owner: owner_b.clone(),
            index: 0,
        };
        let c0 = EndpointNode::Formal {
            owner: owner_c.clone(),
            index: 0,
        };
        let owner_d = FunctionKey::new("sample::d_missing_span");
        let owner_e = FunctionKey::new("sample::e_has_span");
        let d0 = EndpointNode::Formal {
            owner: owner_d.clone(),
            index: 0,
        };
        let e0 = EndpointNode::Formal {
            owner: owner_e.clone(),
            index: 0,
        };
        let owner_f = FunctionKey::new("sample::f");
        let f0 = EndpointNode::Formal {
            owner: owner_f.clone(),
            index: 0,
        };
        let f1 = EndpointNode::Return {
            owner: owner_f.clone(),
        };
        let point = |function: FunctionKey, line| ProgramPoint {
            function,
            block: 0,
            statement: 0,
            span: span(line),
        };
        let mut graph = EndpointFlowGraph::default();
        graph.insert_edge(EndpointEdge {
            from: a0.clone(),
            to: a1.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(owner_a.clone(), 10)),
        });
        graph.insert_edge(EndpointEdge {
            from: a1.clone(),
            to: a0.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(owner_a.clone(), 11)),
        });
        graph.insert_edge(EndpointEdge {
            from: b0.clone(),
            to: b0.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(owner_b.clone(), 20)),
        });
        graph.insert_node(c0.clone());
        graph.insert_edge(EndpointEdge {
            from: d0.clone(),
            to: e0.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: None,
        });
        graph.insert_edge(EndpointEdge {
            from: e0.clone(),
            to: d0.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: None,
        });
        graph.insert_edge(EndpointEdge {
            from: f0.clone(),
            to: f1.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: None,
        });
        graph.insert_edge(EndpointEdge {
            from: f1.clone(),
            to: f0.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: Some(point(owner_f.clone(), 30)),
        });
        let owner_spans = BTreeMap::from([
            (owner_a, span(1)),
            (owner_b, span(2)),
            (owner_c, span(3)),
            (owner_e, span(4)),
            (owner_f, span(5)),
        ]);
        let index = SelectedCycleIndex::new(&graph, &owner_spans);
        let consumed_index = SelectedCycleIndex::from_graph(graph.clone(), &owner_spans);

        let exhaustive = |route_nodes: &[&EndpointNode]| {
            let route_nodes = route_nodes.iter().copied().collect::<BTreeSet<_>>();
            for component in graph.cyclic_components() {
                if component.iter().all(|node| !route_nodes.contains(node)) {
                    continue;
                }
                if let Some(edge) = graph.edges.iter().find(|edge| {
                    component.contains(&edge.from)
                        && component.contains(&edge.to)
                        && edge.point.is_some()
                }) {
                    let point = edge.point.as_ref().unwrap();
                    return (
                        true,
                        Some(witness_step("scc_cycle", &point.function, &point.span)),
                    );
                }
                if let Some(owner) = component.iter().find_map(endpoint_owner_ref) {
                    if let Some(span) = owner_spans.get(owner) {
                        return (true, Some(witness_step("scc_cycle", owner, span)));
                    }
                }
            }
            (false, None)
        };

        let nodes = [&a0, &a1, &b0, &c0, &d0, &e0, &f0, &f1];
        for selected in 0usize..(1usize << nodes.len()) {
            let route = nodes
                .iter()
                .enumerate()
                .filter_map(|(index, node)| ((selected & (1 << index)) != 0).then_some(*node))
                .collect::<Vec<_>>();
            let expected = exhaustive(&route);
            assert_eq!(
                index.selected_cycle_for_nodes(route.iter().copied()),
                expected
            );
            assert_eq!(
                consumed_index.selected_cycle_for_nodes(route.iter().copied()),
                expected
            );
        }
        assert_eq!(exhaustive(&[&d0, &e0]), (false, None));
        assert_eq!(
            index.selected_cycle_for_nodes([&d0, &e0].into_iter()),
            (false, None)
        );
    }

    #[test]
    fn cycle_index_consumes_mvp_graph_but_preserves_formal_graph() {
        let owner = FunctionKey::new("sample::cycle");
        let node = EndpointNode::Formal {
            owner: owner.clone(),
            index: 0,
        };
        let mut graph = EndpointFlowGraph::default();
        graph.insert_edge(EndpointEdge {
            from: node.clone(),
            to: node.clone(),
            kind: EndpointEdgeKind::Transparent,
            point: Some(ProgramPoint {
                function: owner.clone(),
                block: 0,
                statement: 0,
                span: span(9),
            }),
        });
        let owner_spans = BTreeMap::from([(owner, span(1))]);
        let analysis = |endpoint_graph: EndpointFlowGraph| AnalysisOutput {
            structural_candidates: Vec::new(),
            suppressed_structural_candidates: Vec::new(),
            unestablished_structural_candidates: Vec::new(),
            missing_sanitizer_warnings: Vec::new(),
            binding_records: BTreeMap::new(),
            endpoint_graph,
            owner_spans: owner_spans.clone(),
            owner_code_spans: BTreeMap::new(),
        };

        let mut formal = analysis(graph.clone());
        let formal_index = selected_cycle_index_for_analysis(&mut formal, true);
        assert_eq!(formal.endpoint_graph, graph);
        assert!(
            formal_index
                .selected_cycle_for_nodes(std::iter::once(&node))
                .0
        );

        let mut mvp_only = analysis(graph);
        let mvp_index = selected_cycle_index_for_analysis(&mut mvp_only, false);
        assert!(mvp_only.endpoint_graph.nodes.is_empty());
        assert!(mvp_only.endpoint_graph.edges.is_empty());
        assert!(mvp_index.selected_cycle_for_nodes(std::iter::once(&node)).0);
    }

    #[test]
    fn non_bound_site_keeps_binding_match_count_out_of_finding_count() {
        let seed = UnsafeSiteSeed {
            adapter_id: "adapter".to_string(),
            owner: FunctionKey::new("sample::root"),
            owner_def_path_hash: "owner-hash".to_string(),
            ordinal: 0,
            kind: UnsafeSiteKind::UnsafeCall,
            raw_span: RawSpanAnchor::new(span(1), 0, 1, "#0"),
            hir_context: "hir:1".to_string(),
            callee: Some("sample::unsafe_call".to_string()),
            operands: Vec::new(),
            result: None,
        };
        let site_id = seed.site_id();
        let sites = BTreeMap::from([(site_id.clone(), seed)]);
        let records = BTreeMap::from([(
            site_id.clone(),
            SiteBindingRecord {
                site_id: site_id.clone(),
                status: BindingStatus::Ambiguous,
                built_candidate_count: 2,
                solver_candidate_count: 0,
                built: None,
                solver: None,
                solver_event: None,
            },
        )]);
        let failures = vec![CoverageFailure {
            kind: CoverageFailureKind::AmbiguousUnsafeSite,
            owner: FunctionKey::new("sample::root"),
            site_id: Some(site_id),
            candidate_count: 2,
        }];

        let result = build_site_records(&sites, &records, &failures, &BTreeMap::new()).unwrap();
        assert_eq!(result[0]["binding"], "ambiguous");
        assert_eq!(result[0]["candidate_count"], 0);
        assert_eq!(coverage_failure_json(&failures[0])["candidate_count"], 2);
    }

    #[test]
    fn mvp_summary_is_compact_and_contains_the_actionable_chain() {
        let family = json!({
            "family_id": "a".repeat(64),
            "key": {"primary_pattern": "P4", "subject_role": "index"},
            "raw_count": 7,
            "public_root_fanout": 3,
            "representatives": {
                "shortest": {
                    "unsafe_site": {
                        "span": {
                            "path": "src/lib.rs",
                            "start": {"line": 17, "column": 5}
                        }
                    }
                }
            }
        });
        assert_eq!(
            mvp_family_summary_line(&family).unwrap(),
            format!(
                "[P4] family={} raw=7 roots=3 subject=index site=src/lib.rs:17",
                "a".repeat(64)
            )
        );
    }

    #[test]
    fn mvp_coverage_persists_writer_candidate_accounting() {
        let accounting = json!({
            "active_count": 2,
            "pure_suppressed_count": 1,
            "reclassified_count": 1,
            "unestablished_count": 0,
            "suppressed_count": 2,
            "considered_count": 4,
        });
        let coverage = mvp_coverage_json(json!({"total_sites": 3}), &[], accounting.clone());
        assert_eq!(coverage["candidate_accounting"], accounting);
        assert_eq!(coverage["complete"], true);
        assert_eq!(coverage["semantics"], "exact");
    }

    fn unestablished_decision(
        id: char,
        primary_reason: super::super::influence::UnestablishedReason,
        secondary_reasons: Vec<super::super::influence::UnestablishedReason>,
    ) -> UnestablishedStructuralCandidate {
        UnestablishedStructuralCandidate {
            decision_id: id.to_string().repeat(64),
            attempted_pattern: PrimaryPattern::P6,
            public_root: std::sync::Arc::new(FunctionKey::new("sample::root")),
            site_id: "f".repeat(64),
            operand_role: OperandRole::Argument(0),
            source_identity: format!("sample::root:external:{id}"),
            source_kind: CausalSourceKind::ExternalBehaviorOutput,
            controller: "ffi_host".to_string(),
            capabilities: "address,typed_value".to_string(),
            requirement: Some("argument:0:address".to_string()),
            primary_reason,
            secondary_reasons,
        }
    }

    #[test]
    fn unestablished_rows_are_sorted_byte_stable_and_exposure_free() {
        use super::super::influence::UnestablishedReason;

        let first = unestablished_decision(
            'a',
            UnestablishedReason::MissingExactContext,
            vec![UnestablishedReason::OpaqueTransform],
        );
        let second = unestablished_decision(
            'b',
            UnestablishedReason::UnmodeledUnsafeCall,
            vec![UnestablishedReason::ValidationStateUnknown],
        );
        let expected_rows =
            prepare_unestablished_decisions(vec![first.clone(), second.clone()]).unwrap();
        let mut expected = Vec::new();
        write_unestablished_rows_ordered(&mut expected, expected_rows, 1, 1).unwrap();

        for worker_count in [1, 3, 8] {
            let rows =
                prepare_unestablished_decisions(vec![second.clone(), first.clone()]).unwrap();
            let mut actual = Vec::new();
            write_unestablished_rows_ordered(&mut actual, rows, worker_count, 1).unwrap();
            assert_eq!(actual, expected);
        }
        for line in String::from_utf8(expected).unwrap().lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            assert_eq!(row["schema_version"], "rap-unsoundaudit-unestablished-v1");
            assert_eq!(row["exposure"], Value::Null);
        }
    }

    #[test]
    fn unestablished_rows_preserve_every_attempted_pattern() {
        use super::super::influence::UnestablishedReason;

        for (index, pattern) in [
            PrimaryPattern::P1,
            PrimaryPattern::P2,
            PrimaryPattern::P3,
            PrimaryPattern::P4,
            PrimaryPattern::P5,
            PrimaryPattern::P6,
        ]
        .into_iter()
        .enumerate()
        {
            let mut decision = unestablished_decision(
                char::from(b'a' + index as u8),
                UnestablishedReason::MissingExactContext,
                Vec::new(),
            );
            decision.attempted_pattern = pattern;
            assert_eq!(
                unestablished_json(decision)["attempted_pattern"],
                primary_pattern(pattern)
            );
        }
    }

    #[test]
    fn unestablished_rows_reject_duplicate_ids_and_reason_precedence_drift() {
        use super::super::influence::UnestablishedReason;

        let first =
            unestablished_decision('a', UnestablishedReason::MissingExactContext, Vec::new());
        assert!(prepare_unestablished_decisions(vec![first.clone(), first])
            .unwrap_err()
            .contains("duplicated"));

        let invalid = unestablished_decision(
            'b',
            UnestablishedReason::OpaqueTransform,
            vec![UnestablishedReason::MissingExactContext],
        );
        assert!(prepare_unestablished_decisions(vec![invalid])
            .unwrap_err()
            .contains("precedence"));
    }

    #[test]
    fn mvp_rejects_exposure_on_every_non_p3_pattern() {
        for pattern in [
            PrimaryPattern::P1,
            PrimaryPattern::P2,
            PrimaryPattern::P4,
            PrimaryPattern::P5,
            PrimaryPattern::P6,
        ] {
            assert!(validate_mvp_exposure(pattern, true)
                .unwrap_err()
                .contains("P3-only exposure"));
        }
        validate_mvp_exposure(PrimaryPattern::P3, true).unwrap();
        validate_mvp_exposure(PrimaryPattern::P1, false).unwrap();
    }

    #[test]
    fn suppression_plan_permutation_keeps_sorted_records_with_original_payloads() {
        let plans = vec![
            SuppressionPlan {
                record: SuppressionRecord::new(
                    "a".repeat(64),
                    super::super::unsafe_site::SuppressionReason::ContractDischarged,
                    None,
                ),
                original_index: 2,
            },
            SuppressionPlan {
                record: SuppressionRecord::new(
                    "b".repeat(64),
                    super::super::unsafe_site::SuppressionReason::ContractDischarged,
                    Some("d".repeat(64)),
                ),
                original_index: 0,
            },
            SuppressionPlan {
                record: SuppressionRecord::new(
                    "c".repeat(64),
                    super::super::unsafe_site::SuppressionReason::ContractImpliedByType,
                    None,
                ),
                original_index: 1,
            },
        ];
        let mut payloads = vec!["original-0", "original-1", "original-2"];
        reorder_suppressed_candidates(&plans, &mut payloads).unwrap();
        assert_eq!(payloads, ["original-2", "original-0", "original-1"]);
    }

    fn ordered_json_fixture_rows(size: usize) -> Vec<Value> {
        let templates = [
            json!({
                "candidate_id": "a".repeat(64),
                "text": "Unicode 雪; controls \n\t\u{0000}; slash \\\"",
                "replacement_candidate_id": null,
            }),
            json!({
                "candidate_id": "b".repeat(64),
                "candidate": {
                    "primary_pattern": "P1",
                    "propagation": {
                        "depth": "recursive",
                        "witness": [{"kind": "scc_cycle", "owner": "sample::cycle"}],
                    },
                },
            }),
            json!({
                "candidate_id": "c".repeat(64),
                "candidate": {
                    "primary_pattern": "P3",
                    "exposure": {"identity": "sample::root:exposure:_3"},
                },
            }),
            json!({
                "candidate_id": "d".repeat(64),
                "candidate": {
                    "primary_pattern": "P5",
                    "source": {
                        "generic_capability": {
                            "kind": "pointer_layout_reinterpretation",
                            "witness": {"status": "proven", "violations": ["alignment"]},
                        },
                    },
                },
            }),
        ];
        (0..size)
            .map(|ordinal| {
                let mut row = templates[ordinal % templates.len()].clone();
                row["ordinal"] = json!(ordinal);
                row
            })
            .collect()
    }

    fn serial_jsonl_bytes(rows: &[Value]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            serde_json::to_writer(&mut bytes, row).unwrap();
            bytes.push(b'\n');
        }
        bytes
    }

    #[test]
    fn bounded_ordered_json_chunks_match_serial_golden_across_boundaries() {
        for size in [0, 1, 255, 256, 257] {
            let rows = ordered_json_fixture_rows(size);
            let expected = serial_jsonl_bytes(&rows);
            for worker_count in [1, 3, 8] {
                for chunk_size in [1, 2, 256] {
                    let mut actual = Vec::new();
                    write_json_rows_ordered(
                        &mut actual,
                        "test-golden",
                        rows.iter(),
                        worker_count,
                        chunk_size,
                    )
                    .unwrap();
                    assert_eq!(
                        actual, expected,
                        "byte drift for size={size}, workers={worker_count}, chunk={chunk_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn ordered_chunks_report_the_earliest_input_error_after_a_later_chunk_finishes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let later_finished = Arc::new(AtomicBool::new(false));
        let process_flag = Arc::clone(&later_finished);
        let process = move |chunk: Vec<usize>| -> Result<usize, String> {
            match chunk[0] {
                0 => {
                    while !process_flag.load(Ordering::Acquire) {
                        thread::yield_now();
                    }
                    Err("earliest input error".to_string())
                }
                1 => {
                    process_flag.store(true, Ordering::Release);
                    Ok(1)
                }
                _ => Err("later input error".to_string()),
            }
        };
        let mut consumed = Vec::new();
        let error = process_ordered_chunks(
            0usize..3,
            3,
            1,
            "error-order-test",
            &process,
            &mut |value| {
                consumed.push(value);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(error, "earliest input error");
        assert!(later_finished.load(Ordering::Acquire));
        assert!(consumed.is_empty());
    }

    #[test]
    fn ordered_chunks_turn_the_earliest_worker_panic_into_a_fail_closed_error() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let later_finished = Arc::new(AtomicBool::new(false));
        let process_flag = Arc::clone(&later_finished);
        let process = move |chunk: Vec<usize>| -> Result<usize, String> {
            match chunk[0] {
                0 => {
                    while !process_flag.load(Ordering::Acquire) {
                        thread::yield_now();
                    }
                    panic!("earliest worker panic")
                }
                _ => {
                    process_flag.store(true, Ordering::Release);
                    Ok(chunk[0])
                }
            }
        };
        let error =
            process_ordered_chunks(0usize..2, 2, 1, "panic-order-test", &process, &mut |_| {
                Ok(())
            })
            .unwrap_err();
        assert!(error.contains("chunk 0"));
        assert!(error.contains("earliest worker panic"));
        assert!(later_finished.load(Ordering::Acquire));
    }

    #[test]
    fn parallel_receipt_payloads_satisfy_the_scoped_thread_bounds() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<SuppressionPlan>();
        assert_send::<SuppressedStructuralCandidate>();
        assert_send::<ActiveCandidateInput<'static>>();
        assert_sync::<SelectedCycleIndex>();
        assert_sync::<CodeDictionary>();
        assert_sync::<Value>();
    }
}
