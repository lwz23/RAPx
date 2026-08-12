mod mir;
mod summary;

use crate::utils::unit_output::{
    atomic_write, build_verification_receipt_from_environment, embedded_rap_compiler_commit,
    validate_commit_hash,
};
use rustc_middle::ty::TyCtxt;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::env;
use std::path::PathBuf;
use summary::{deduplicate_findings, Finding, Operation, Pattern, StableSpan};

const LIMITATION: &str =
    "fixture-closed P1-P6 heuristic candidate generator; unregistered forms are unsupported";

struct UnitMetadata {
    project_root: String,
    unit_id: String,
    rustc_invocation_id: String,
    package_id: String,
    package_name: String,
    package_version: String,
    crate_name: String,
    crate_types: Vec<String>,
    target_kind: String,
    target_triple: String,
    manifest_path: String,
    source_path: String,
    extra_filename: String,
    rustc_path: String,
    cargo_supplied_rustc_path: String,
    cargo_supplied_rustc_commit: String,
    rap_compiler_commit: String,
}

pub struct LwzCheck<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

impl<'tcx> LwzCheck<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx }
    }

    pub fn start(&mut self) {
        let result =
            mir::analyze(self.tcx).and_then(|findings| self.maybe_write_json_summary(findings));
        if let Err(error) = result {
            crate::utils::log::rap_error_and_exit(error);
        }
    }

    fn maybe_write_json_summary(&self, findings: Vec<Finding>) -> Result<(), String> {
        let Some(output_path) = env::var_os("UNSOUND_SCANNER_RAP_JSON_OUT") else {
            return Ok(());
        };
        let output_path = PathBuf::from(output_path);
        let summary = self.build_json_summary(findings)?;
        let mut bytes = serde_json::to_vec_pretty(&summary).map_err(|error| {
            format!(
                "failed to serialize RAP JSON output '{}': {error}",
                output_path.display()
            )
        })?;
        bytes.push(b'\n');
        atomic_write(&output_path, &bytes).map_err(|error| {
            format!(
                "failed to atomically write RAP JSON output '{}': {error}",
                output_path.display()
            )
        })
    }

    fn build_json_summary(&self, findings: Vec<Finding>) -> Result<Value, String> {
        // The verification challenge remains fail-closed, but its machine-specific
        // evidence is intentionally not embedded in the closed rap-unit-v2 schema.
        let _ = build_verification_receipt_from_environment()?;
        let rap_compiler_commit = embedded_rap_compiler_commit()?.to_ascii_lowercase();
        let cargo_supplied_rustc_commit =
            env::var("UNSOUND_SCANNER_CARGO_SUPPLIED_RUSTC_COMMIT")
                .map_err(|_| "missing Cargo-supplied rustc commit".to_string())?;
        let cargo_supplied_rustc_commit = validate_commit_hash(
            Some(&cargo_supplied_rustc_commit),
            "Cargo-supplied rustc commit",
        )?
        .to_ascii_lowercase();

        let mut findings = deduplicate_findings(findings);
        findings.sort_by_key(|finding| finding.causal_key.canonical_json());
        let mut counts = [0_usize; 6];
        for finding in &findings {
            counts[pattern_index(finding.primary)] += 1;
        }
        let findings = findings.iter().map(finding_json).collect::<Vec<_>>();

        let mut crate_types = env::var("UNSOUND_SCANNER_RAP_CRATE_TYPES")
            .unwrap_or_default()
            .split(',')
            .filter(|crate_type| !crate_type.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        crate_types.sort();
        crate_types.dedup();
        if crate_types.is_empty() {
            return Err("missing RAP crate type metadata".to_string());
        }

        let metadata = UnitMetadata {
            project_root: env::var("UNSOUND_SCANNER_PROJECT_ROOT")
                .ok()
                .or_else(|| {
                    env::current_dir()
                        .ok()
                        .map(|path| path.display().to_string())
                })
                .unwrap_or_default(),
            unit_id: environment("UNSOUND_SCANNER_RAP_UNIT_ID"),
            rustc_invocation_id: environment("UNSOUND_SCANNER_RAP_RUSTC_INVOCATION_ID"),
            package_id: environment("UNSOUND_SCANNER_RAP_PACKAGE_ID"),
            package_name: environment("UNSOUND_SCANNER_RAP_PACKAGE_NAME"),
            package_version: environment("UNSOUND_SCANNER_RAP_PACKAGE_VERSION"),
            crate_name: environment("UNSOUND_SCANNER_RAP_CRATE_NAME"),
            crate_types,
            target_kind: environment("UNSOUND_SCANNER_RAP_TARGET_KIND"),
            target_triple: environment("UNSOUND_SCANNER_RAP_TARGET_TRIPLE"),
            manifest_path: environment("UNSOUND_SCANNER_RAP_MANIFEST_PATH"),
            source_path: environment("UNSOUND_SCANNER_RAP_SOURCE_PATH"),
            extra_filename: environment("UNSOUND_SCANNER_RAP_EXTRA_FILENAME"),
            rustc_path: environment("UNSOUND_SCANNER_RAP_RUSTC_PATH"),
            cargo_supplied_rustc_path: environment("UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH"),
            cargo_supplied_rustc_commit,
            rap_compiler_commit,
        };

        Ok(unit_json(metadata, counts, findings))
    }
}

fn unit_json(metadata: UnitMetadata, counts: [usize; 6], findings: Vec<Value>) -> Value {
    json!({
        "schema_version": "rap-unit-v2",
        "project_root": metadata.project_root,
        "source": "rap",
        "success": true,
        "unit_id": metadata.unit_id,
        "rustc_invocation_id": metadata.rustc_invocation_id,
        "package_id": metadata.package_id,
        "package_name": metadata.package_name,
        "package_version": metadata.package_version,
        "crate_name": metadata.crate_name,
        "crate_types": metadata.crate_types,
        "target_kind": metadata.target_kind,
        "target_triple": metadata.target_triple,
        "manifest_path": metadata.manifest_path,
        "source_path": metadata.source_path,
        "extra_filename": metadata.extra_filename,
        "rustc_path": metadata.rustc_path,
        "cargo_supplied_rustc_path": metadata.cargo_supplied_rustc_path,
        "cargo_supplied_rustc_commit": metadata.cargo_supplied_rustc_commit,
        "rap_compiler_commit": metadata.rap_compiler_commit,
        "rustc_commit": metadata.rap_compiler_commit,
        "pattern_counts": {
            "pattern1": counts[0],
            "pattern2": counts[1],
            "pattern3": counts[2],
            "pattern4": counts[3],
            "pattern5": counts[4],
            "pattern6": counts[5],
        },
        "finding_count": findings.len(),
        "findings": findings,
    })
}

fn environment(name: &str) -> String {
    env::var(name).unwrap_or_default()
}

fn pattern_index(pattern: Pattern) -> usize {
    match pattern {
        Pattern::P1 => 0,
        Pattern::P2 => 1,
        Pattern::P3 => 2,
        Pattern::P4 => 3,
        Pattern::P5 => 4,
        Pattern::P6 => 5,
    }
}

fn span_json(span: &StableSpan) -> Value {
    json!({
        "path": span.path,
        "start": { "line": span.start.line, "column": span.start.column },
        "end": { "line": span.end.line, "column": span.end.column },
    })
}

fn operation_json(operation: &Operation) -> Value {
    json!({
        "kind": operation.kind.as_str(),
        "span": span_json(&operation.point.span),
    })
}

fn finding_json(finding: &Finding) -> Value {
    let evidence = &finding.sink_obligation;
    let mut predicates = evidence
        .obligation
        .predicates
        .iter()
        .map(|predicate| predicate.as_str())
        .collect::<Vec<_>>();
    predicates.sort();
    predicates.dedup();
    let mut secondary_sources = finding
        .secondary_sources
        .iter()
        .map(|source| source.as_str())
        .collect::<Vec<_>>();
    secondary_sources.sort();
    secondary_sources.dedup();
    let boundaries = finding
        .witness
        .boundaries
        .iter()
        .map(|boundary| boundary.0.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let witness = finding
        .witness
        .steps
        .iter()
        .map(|step| {
            json!({
                "kind": step.kind.as_str(),
                "function": step.function.0,
                "span": span_json(&step.span),
            })
        })
        .collect::<Vec<_>>();

    let mut finding_value = Map::from_iter([
        (
            "finding_id".to_string(),
            Value::String(finding.causal_key.finding_id()),
        ),
        (
            "causal_key".to_string(),
            json!({
                "crate": finding.causal_key.crate_key,
                "public_root": finding.causal_key.public_root.0,
                "source_origin": finding.causal_key.source_origin,
                "first_contract_failure": finding.causal_key.first_contract_failure,
                "sink_or_exposure": finding.causal_key.sink_or_exposure,
                "canonical_obligation": finding.causal_key.canonical_obligation,
            }),
        ),
        (
            "primary_pattern".to_string(),
            Value::String(finding.primary.as_str().to_string()),
        ),
        (
            "rule_id".to_string(),
            Value::String(finding.rule.as_str().to_string()),
        ),
        (
            "public_safe_root".to_string(),
            json!({
                "def_path": finding.causal_key.public_root.0,
                "span": span_json(&finding.public_root_span),
            }),
        ),
        (
            "source".to_string(),
            json!({
                "kind": evidence.source.kind.as_str(),
                "origin_key": evidence.source.origin.0,
                "span": span_json(&evidence.source.span),
            }),
        ),
        (
            "first_contract_failure".to_string(),
            operation_json(&evidence.first_failure),
        ),
        ("sink".to_string(), operation_json(&evidence.sink)),
        (
            "obligation".to_string(),
            json!({
                "predicates": predicates,
                "subject": evidence.obligation.subject.0,
            }),
        ),
        (
            "validation_status".to_string(),
            json!({ "status": "missing", "facts": [] }),
        ),
        (
            "propagation".to_string(),
            json!({
                "depth": finding.witness.depth.as_str(),
                "local_call_count": finding.witness.local_call_count,
                "boundaries": boundaries,
                "witness": witness,
            }),
        ),
        (
            "secondary_source_kinds".to_string(),
            json!(secondary_sources),
        ),
        (
            "heuristic_candidate_generator".to_string(),
            Value::Bool(true),
        ),
        ("limitations".to_string(), json!([LIMITATION])),
    ]);
    if finding.primary == Pattern::P5 {
        finding_value.insert(
            "family_support".to_string(),
            Value::String("partial".to_string()),
        );
    }
    Value::Object(finding_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use summary::{
        CanonicalWitness, FunctionKey, Obligation, OperationKind, OriginKey, Predicate,
        ProgramPoint, RuleId, SinkObligation, Source, SourceKind, StablePosition, WitnessStep,
        WitnessStepKind,
    };

    fn keys(value: &Value) -> BTreeSet<String> {
        value
            .as_object()
            .expect("test value must be an object")
            .keys()
            .cloned()
            .collect()
    }

    fn metadata() -> UnitMetadata {
        UnitMetadata {
            project_root: "fixture".to_string(),
            unit_id: "unit".to_string(),
            rustc_invocation_id: "invocation".to_string(),
            package_id: "package 0.1.0".to_string(),
            package_name: "package".to_string(),
            package_version: "0.1.0".to_string(),
            crate_name: "fixture".to_string(),
            crate_types: vec!["lib".to_string()],
            target_kind: "lib".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            manifest_path: "Cargo.toml".to_string(),
            source_path: "src/lib.rs".to_string(),
            extra_filename: String::new(),
            rustc_path: "rustc".to_string(),
            cargo_supplied_rustc_path: "rustc".to_string(),
            cargo_supplied_rustc_commit: "a".repeat(40),
            rap_compiler_commit: "b".repeat(40),
        }
    }

    fn span(line: u32) -> StableSpan {
        StableSpan::new(
            "src/lib.rs",
            StablePosition::new(line, 1),
            StablePosition::new(line, 2),
        )
    }

    fn finding(pattern: Pattern) -> Finding {
        let function = FunctionKey::new("fixture::root");
        let source_span = span(1);
        let sink_span = span(2);
        let source = Source {
            kind: if pattern == Pattern::P5 {
                SourceKind::GenericNonEmptyCapability
            } else {
                SourceKind::PublicParameter
            },
            origin: OriginKey::new("arg1"),
            span: source_span.clone(),
        };
        let point = ProgramPoint {
            function: function.clone(),
            block: 0,
            statement: 0,
            span: sink_span.clone(),
        };
        let operation = Operation {
            kind: OperationKind::GetUnchecked,
            point,
        };
        let sink_obligation = SinkObligation {
            source,
            first_failure: operation.clone(),
            sink: operation,
            obligation: Obligation::new([Predicate::NonEmpty], OriginKey::new("arg1")),
        };
        let witness = CanonicalWitness::new(
            vec![
                WitnessStep {
                    kind: WitnessStepKind::Entry,
                    function: function.clone(),
                    span: source_span.clone(),
                },
                WitnessStep {
                    kind: WitnessStepKind::Sink,
                    function: function.clone(),
                    span: sink_span,
                },
            ],
            Vec::new(),
        );
        Finding::new(
            "fixture",
            function,
            source_span,
            sink_obligation,
            pattern,
            if pattern == Pattern::P5 {
                RuleId::P51Nonempty
            } else {
                RuleId::P1GetUnchecked
            },
            witness,
            BTreeSet::new(),
        )
    }

    #[test]
    fn top_level_and_pattern_count_keys_are_exact() {
        let value = unit_json(metadata(), [0; 6], Vec::new());
        assert_eq!(
            keys(&value),
            [
                "cargo_supplied_rustc_commit",
                "cargo_supplied_rustc_path",
                "crate_name",
                "crate_types",
                "extra_filename",
                "finding_count",
                "findings",
                "manifest_path",
                "package_id",
                "package_name",
                "package_version",
                "pattern_counts",
                "project_root",
                "rap_compiler_commit",
                "rustc_commit",
                "rustc_invocation_id",
                "rustc_path",
                "schema_version",
                "source",
                "source_path",
                "success",
                "target_kind",
                "target_triple",
                "unit_id",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        );
        assert_eq!(
            keys(&value["pattern_counts"]),
            ["pattern1", "pattern2", "pattern3", "pattern4", "pattern5", "pattern6",]
                .into_iter()
                .map(str::to_string)
                .collect()
        );
    }

    #[test]
    fn finding_keys_and_p5_family_support_are_exact() {
        let base_keys = [
            "causal_key",
            "finding_id",
            "first_contract_failure",
            "heuristic_candidate_generator",
            "limitations",
            "obligation",
            "primary_pattern",
            "propagation",
            "public_safe_root",
            "rule_id",
            "secondary_source_kinds",
            "sink",
            "source",
            "validation_status",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();

        for pattern in [
            Pattern::P1,
            Pattern::P2,
            Pattern::P3,
            Pattern::P4,
            Pattern::P6,
        ] {
            let value = finding_json(&finding(pattern));
            assert_eq!(keys(&value), base_keys, "unexpected keys for {pattern:?}");
            assert!(value.get("family_support").is_none());
        }

        let p5 = finding_json(&finding(Pattern::P5));
        let mut p5_keys = base_keys;
        p5_keys.insert("family_support".to_string());
        assert_eq!(keys(&p5), p5_keys);
        assert_eq!(p5["family_support"], "partial");
    }
}
