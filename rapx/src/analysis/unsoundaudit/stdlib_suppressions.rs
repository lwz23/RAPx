use super::summary::{sha256_hex, Finding, RuleId, SourceKind};
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub(super) const SUPPRESSION_LEDGER_SCHEMA: &str = "unsoundaudit-v2-stdlib-suppression-ledger-v1";
pub(super) const ALLOWLIST_ENTRIES_SHA256: &str =
    "95b5cf503b9942aac2c787053c1f0c6c40b8caa07472965e8059ea8becfacd21";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StdlibSuppression {
    pub rustc_commit: &'static str,
    pub crate_key: &'static str,
    pub rule_id: RuleId,
    pub sink_token: &'static str,
    pub first_failure_token: &'static str,
    pub canonical_obligation: &'static str,
    pub source_kind: SourceKind,
    pub family_id: &'static str,
    pub evidence_sha256: &'static str,
    pub rationale: &'static str,
}

#[derive(Clone, Debug)]
pub(super) struct SuppressedFinding {
    pub finding: Finding,
    pub entry: &'static StdlibSuppression,
}

include!("stdlib_suppressions_table.rs");

fn operation_json(operation: &super::summary::Operation) -> Value {
    json!({
        "kind": operation.kind.as_str(),
        "span": {
            "path": operation.point.span.path,
            "start": {
                "line": operation.point.span.start.line,
                "column": operation.point.span.start.column,
            },
            "end": {
                "line": operation.point.span.end.line,
                "column": operation.point.span.end.column,
            },
        },
    })
}

pub(super) fn finding_family_id(finding: &Finding) -> String {
    let evidence = &finding.sink_obligation;
    let predicates = evidence
        .obligation
        .predicates
        .iter()
        .map(|predicate| predicate.as_str())
        .collect::<Vec<_>>();
    let key = json!({
        "unit": finding.causal_key.crate_key,
        "rule_id": finding.rule.as_str(),
        "sink": operation_json(&evidence.sink),
        "first_contract_failure": operation_json(&evidence.first_failure),
        "predicates": predicates,
        "subject": evidence.obligation.subject.0,
        "source_kind": evidence.source.kind.as_str(),
        "propagation_depth": finding.witness.depth.as_str(),
        "validation_status": "missing",
    });
    let canonical = serde_json::to_vec(&key).expect("suppression family key serializes");
    sha256_hex(&canonical)
}

fn entry_json(entry: &StdlibSuppression) -> Value {
    json!({
        "rustc_commit": entry.rustc_commit,
        "crate_key": entry.crate_key,
        "rule_id": entry.rule_id.as_str(),
        "sink_token": entry.sink_token,
        "first_failure_token": entry.first_failure_token,
        "canonical_obligation": entry.canonical_obligation,
        "source_kind": entry.source_kind.as_str(),
        "family_id": entry.family_id,
        "evidence_sha256": entry.evidence_sha256,
        "rationale": entry.rationale,
    })
}

fn computed_allowlist_sha256() -> String {
    let entries = SUPPRESSIONS.iter().map(entry_json).collect::<Vec<_>>();
    let canonical = serde_json::to_vec(&entries).expect("suppression allowlist serializes");
    sha256_hex(&canonical)
}

pub(super) fn validate_allowlist(entries: &[StdlibSuppression]) -> Result<(), String> {
    if entries.is_empty() {
        return Err("stdlib suppression allowlist is empty".to_string());
    }
    let mut identities = BTreeSet::new();
    let mut families = BTreeSet::new();
    for entry in entries {
        if entry.rustc_commit.len() != 40
            || !entry
                .rustc_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || !matches!(entry.crate_key, "core" | "alloc" | "std")
            || entry.family_id.len() != 64
            || entry.evidence_sha256.len() != 64
            || entry.rationale.trim().is_empty()
        {
            return Err("stdlib suppression entry has invalid closed fields".to_string());
        }
        let identity = (
            entry.rustc_commit,
            entry.crate_key,
            entry.rule_id,
            entry.sink_token,
            entry.first_failure_token,
            entry.canonical_obligation,
            entry.source_kind,
            entry.family_id,
        );
        if !identities.insert(identity) || !families.insert(entry.family_id) {
            return Err("stdlib suppression allowlist contains a duplicate".to_string());
        }
    }
    Ok(())
}

fn exact_match(
    entry: &StdlibSuppression,
    rustc_commit: &str,
    finding: &Finding,
    family_id: &str,
) -> bool {
    let evidence = &finding.sink_obligation;
    entry.rustc_commit == rustc_commit
        && entry.crate_key == finding.causal_key.crate_key
        && entry.rule_id == finding.rule
        && entry.sink_token == evidence.sink.canonical_token()
        && entry.first_failure_token == evidence.first_failure.canonical_token()
        && entry.canonical_obligation == evidence.obligation.canonical_token()
        && entry.source_kind == evidence.source.kind
        && entry.family_id == family_id
}

pub(super) fn partition_findings(
    findings: Vec<Finding>,
    rustc_commit: &str,
) -> Result<(Vec<Finding>, Vec<SuppressedFinding>), String> {
    validate_allowlist(SUPPRESSIONS)?;
    if computed_allowlist_sha256() != ALLOWLIST_ENTRIES_SHA256 {
        return Err("compiled stdlib suppression allowlist hash drifted".to_string());
    }
    let mut active = Vec::new();
    let mut suppressed = Vec::new();
    for finding in findings {
        let family_id = finding_family_id(&finding);
        let matches = SUPPRESSIONS
            .iter()
            .filter(|entry| exact_match(entry, rustc_commit, &finding, &family_id))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => active.push(finding),
            [entry] => suppressed.push(SuppressedFinding {
                finding,
                entry: *entry,
            }),
            _ => return Err("one finding matched multiple stdlib suppressions".to_string()),
        }
    }
    Ok((active, suppressed))
}

pub(super) fn ledger_json(
    crate_key: &str,
    cargo_supplied_rustc_commit: &str,
    rap_compiler_commit: &str,
    raw_finding_count: usize,
    active_finding_count: usize,
    suppressed: &[SuppressedFinding],
) -> Value {
    let rows = suppressed
        .iter()
        .map(|row| {
            json!({
                "family_id": row.entry.family_id,
                "evidence_sha256": row.entry.evidence_sha256,
                "rationale": row.entry.rationale,
                "allowlist_entry": entry_json(row.entry),
                "finding": super::finding_json(&row.finding),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": SUPPRESSION_LEDGER_SCHEMA,
        "crate_key": crate_key,
        "cargo_supplied_rustc_commit": cargo_supplied_rustc_commit,
        "rap_compiler_commit": rap_compiler_commit,
        "rustc_commit": rap_compiler_commit,
        "allowlist_entries_sha256": ALLOWLIST_ENTRIES_SHA256,
        "raw_finding_count": raw_finding_count,
        "suppressed_finding_count": rows.len(),
        "active_finding_count": active_finding_count,
        "suppressed_findings": rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::unsoundaudit::summary::{
        CanonicalWitness, FunctionKey, Obligation, Operation, OperationKind, OriginKey, Pattern,
        Predicate, ProgramPoint, SinkObligation, Source, StablePosition, StableSpan, WitnessStep,
        WitnessStepKind,
    };

    const PINNED: &str = "1bc403daadbebb553ccc211a0a8eebb73989665f";

    fn span(path: &str, line: u32, start: u32, end: u32) -> StableSpan {
        StableSpan::new(
            path,
            StablePosition::new(line, start),
            StablePosition::new(line, end),
        )
    }

    fn cstring_finding() -> Finding {
        let function = FunctionKey::new("ffi::c_str::CString::as_bytes");
        let root_span = span("library/alloc/src/ffi/c_str.rs", 540, 5, 36);
        let sink_span = span("library/alloc/src/ffi/c_str.rs", 542, 18, 66);
        let operation = Operation {
            kind: OperationKind::GetUnchecked,
            point: ProgramPoint {
                function: function.clone(),
                block: 0,
                statement: 0,
                span: sink_span.clone(),
            },
        };
        Finding::new(
            "alloc",
            function.clone(),
            root_span.clone(),
            SinkObligation {
                source: Source {
                    kind: SourceKind::InternalDerived,
                    origin: OriginKey::new("private-field:boxed::Box::0"),
                    span: span("library/alloc/src/ffi/c_str.rs", 542, 45, 55),
                },
                first_failure: operation.clone(),
                sink: operation,
                obligation: Obligation::new(
                    [Predicate::InBounds],
                    OriginKey::new("private-field:boxed::Box::0"),
                ),
            },
            Pattern::P4,
            RuleId::P4Bounds,
            CanonicalWitness::new(
                vec![
                    WitnessStep {
                        kind: WitnessStepKind::Entry,
                        function: function.clone(),
                        span: root_span,
                    },
                    WitnessStep {
                        kind: WitnessStepKind::Sink,
                        function,
                        span: sink_span,
                    },
                ],
                Vec::new(),
            ),
            BTreeSet::new(),
        )
    }

    fn exact_entry() -> StdlibSuppression {
        StdlibSuppression {
            rustc_commit: PINNED,
            crate_key: "alloc",
            rule_id: RuleId::P4Bounds,
            sink_token: "get_unchecked@library/alloc/src/ffi/c_str.rs:542:18-542:66",
            first_failure_token: "get_unchecked@library/alloc/src/ffi/c_str.rs:542:18-542:66",
            canonical_obligation: "in_bounds:private-field:boxed::Box::0",
            source_kind: SourceKind::InternalDerived,
            family_id: "4e8bd8f71b2a466b81f05ae0a79ee7cf69ff089cc91f86811e7252d6c4d5af11",
            evidence_sha256: "25c7a13bef9b37b543021657381864657aa2d86c89385028be5d1e3a51990e78",
            rationale: "CString keeps one trailing NUL.",
        }
    }

    #[test]
    fn family_id_matches_the_frozen_python_family_key() {
        assert_eq!(
            finding_family_id(&cstring_finding()),
            "4e8bd8f71b2a466b81f05ae0a79ee7cf69ff089cc91f86811e7252d6c4d5af11"
        );
    }

    #[test]
    fn exact_match_rejects_every_changed_identity_field() {
        let finding = cstring_finding();
        let family = finding_family_id(&finding);
        let entry = exact_entry();
        assert!(exact_match(&entry, PINNED, &finding, &family));

        let mut wrong = entry;
        wrong.rustc_commit = "0000000000000000000000000000000000000000";
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
        let mut wrong = entry;
        wrong.crate_key = "core";
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
        let mut wrong = entry;
        wrong.rule_id = RuleId::P4Offset;
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
        let mut wrong = entry;
        wrong.sink_token = "get_unchecked@library/alloc/src/ffi/c_str.rs:542:19-542:66";
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
        let mut wrong = entry;
        wrong.first_failure_token = "get_unchecked@library/alloc/src/ffi/c_str.rs:542:18-542:65";
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
        let mut wrong = entry;
        wrong.canonical_obligation = "in_bounds:private-field:boxed::Box::1";
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
        let mut wrong = entry;
        wrong.source_kind = SourceKind::PublicParameter;
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
        let mut wrong = entry;
        wrong.family_id = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(!exact_match(&wrong, PINNED, &finding, &family));
    }

    #[test]
    fn compiled_allowlist_is_complete_unique_and_hash_bound() {
        validate_allowlist(SUPPRESSIONS).unwrap();
        assert_eq!(SUPPRESSIONS.len(), 54);
        assert_eq!(computed_allowlist_sha256(), ALLOWLIST_ENTRIES_SHA256);

        let entry = exact_entry();
        assert!(validate_allowlist(&[entry, entry]).is_err());
    }

    #[test]
    fn suppression_is_post_finding_and_wrong_commit_preserves_the_id() {
        let finding = cstring_finding();
        let finding_id = finding.causal_key.finding_id();
        let (active, suppressed) = partition_findings(vec![finding.clone()], PINNED).unwrap();
        assert!(active.is_empty());
        assert_eq!(suppressed.len(), 1);
        assert_eq!(suppressed[0].finding.causal_key.finding_id(), finding_id);

        let (active, suppressed) =
            partition_findings(vec![finding], "0000000000000000000000000000000000000000").unwrap();
        assert_eq!(active.len(), 1);
        assert!(suppressed.is_empty());
        assert_eq!(active[0].causal_key.finding_id(), finding_id);
    }

    #[test]
    fn ledger_reconciles_raw_active_and_complete_suppressed_findings() {
        let finding = cstring_finding();
        let (_, suppressed) = partition_findings(vec![finding], PINNED).unwrap();
        let ledger = ledger_json("alloc", PINNED, PINNED, 1, 0, &suppressed);
        assert_eq!(ledger["schema_version"], SUPPRESSION_LEDGER_SCHEMA);
        assert_eq!(ledger["raw_finding_count"], 1);
        assert_eq!(ledger["active_finding_count"], 0);
        assert_eq!(ledger["suppressed_finding_count"], 1);
        assert_eq!(
            ledger["suppressed_findings"][0]["finding"]["finding_id"],
            suppressed[0].finding.causal_key.finding_id()
        );
        assert_eq!(
            ledger["suppressed_findings"][0]["family_id"],
            finding_family_id(&suppressed[0].finding)
        );
    }
}
