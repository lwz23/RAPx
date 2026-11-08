use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

use super::super::summary::sha256_hex;
use super::super::unsafe_site::SuppressionReason;

const FAMILY_SCHEMA: &str = "rap-unsoundaudit-family-v3-mvp";
const SUPPRESSION_SCHEMA: &str = "rap-unsoundaudit-suppression-v3-mvp";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SuppressionRecord {
    candidate_id: String,
    reason: SuppressionReason,
    replacement_candidate_id: Option<String>,
}

impl SuppressionRecord {
    #[allow(dead_code)]
    pub(super) fn new(
        candidate_id: String,
        reason: SuppressionReason,
        replacement_candidate_id: Option<String>,
    ) -> Self {
        Self {
            candidate_id,
            reason,
            replacement_candidate_id,
        }
    }

    pub(super) fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    fn validate(&self, active_ids: &[&str]) -> Result<(), String> {
        let candidate_id = self.candidate_id();
        validate_sha256(candidate_id, "suppressed candidate id")?;
        if let Some(replacement) = &self.replacement_candidate_id {
            validate_sha256(replacement, "replacement candidate id")?;
            if active_ids.binary_search(&replacement.as_str()).is_err() {
                return Err(format!(
                    "suppression record for {candidate_id} references non-active replacement {replacement}"
                ));
            }
        }
        Ok(())
    }

    pub(super) fn sort_key(&self) -> (&str, &str, &str) {
        (
            &self.candidate_id,
            self.reason.as_str(),
            self.replacement_candidate_id.as_deref().unwrap_or_default(),
        )
    }

    pub(super) fn into_json(self, candidate: Value, evidence: Value) -> Result<Value, String> {
        let Self {
            candidate_id,
            reason,
            replacement_candidate_id,
        } = self;
        let materialized_id = finding_id(&candidate, "suppressed candidate")?;
        if materialized_id != candidate_id {
            return Err(format!(
                "suppressed candidate identity drifted during materialization: planned={candidate_id}, materialized={materialized_id}"
            ));
        }
        if !matches!(&evidence, Value::Object(fields) if !fields.is_empty()) {
            return Err(format!(
                "suppression record for {candidate_id} has empty or non-object structural evidence"
            ));
        }
        let mut row = Map::new();
        row.insert(
            "schema_version".to_string(),
            Value::String(SUPPRESSION_SCHEMA.to_string()),
        );
        row.insert("candidate_id".to_string(), Value::String(candidate_id));
        row.insert("candidate".to_string(), candidate);
        row.insert(
            "reason".to_string(),
            Value::String(reason.as_str().to_string()),
        );
        row.insert("evidence".to_string(), evidence);
        row.insert(
            "replacement_candidate_id".to_string(),
            replacement_candidate_id.map_or(Value::Null, Value::String),
        );
        Ok(Value::Object(row))
    }
}

#[derive(Debug)]
pub(super) struct MvpReports {
    pub(super) families: Vec<Value>,
    pub(super) active_count: usize,
    pub(super) suppressed_count: usize,
    pub(super) unestablished_count: usize,
    considered_count: usize,
    pure_suppressed_count: usize,
    reclassified_count: usize,
    active_ids: Vec<String>,
}

impl MvpReports {
    pub(super) fn reconcile_with_active(&self, active: &[Value]) -> Result<(), String> {
        let active_ids = unique_finding_id_refs(active, "active")?;
        if self.active_count != active.len() {
            return Err(format!(
                "MVP report active count drift: report={}, candidates={}",
                self.active_count,
                active.len()
            ));
        }
        if self.active_count + self.suppressed_count + self.unestablished_count
            != self.considered_count
        {
            return Err(format!(
                "MVP report considered count drift: active={} + suppressed={} + unestablished={} != considered={}",
                self.active_count,
                self.suppressed_count,
                self.unestablished_count,
                self.considered_count
            ));
        }
        if self.pure_suppressed_count + self.reclassified_count != self.suppressed_count {
            return Err(format!(
                "MVP report suppression partition drift: pure={} + reclassified={} != suppressed={}",
                self.pure_suppressed_count, self.reclassified_count, self.suppressed_count
            ));
        }
        if !active_ids
            .iter()
            .copied()
            .eq(self.active_ids.iter().map(String::as_str))
        {
            return Err(
                "MVP report active candidate identities drifted after aggregation".to_string(),
            );
        }
        Ok(())
    }

    pub(super) fn candidate_accounting_json(&self) -> Value {
        json!({
            "active_count": self.active_count,
            "pure_suppressed_count": self.pure_suppressed_count,
            "reclassified_count": self.reclassified_count,
            "unestablished_count": self.unestablished_count,
            "suppressed_count": self.suppressed_count,
            "considered_count": self.considered_count,
        })
    }

    pub(super) fn into_write_parts(
        self,
        active: &[Value],
    ) -> Result<(Vec<Value>, Value, usize, usize), String> {
        self.reconcile_with_active(active)?;
        let accounting = self.candidate_accounting_json();
        Ok((
            self.families,
            accounting,
            self.suppressed_count,
            self.unestablished_count,
        ))
    }
}

pub(super) fn build_reports<'suppression>(
    active: &[Value],
    suppressions: impl IntoIterator<Item = &'suppression SuppressionRecord>,
    considered_count: usize,
) -> Result<MvpReports, String> {
    build_reports_with_unestablished(active, suppressions, &[], considered_count)
}

pub(super) fn build_reports_with_unestablished<'suppression>(
    active: &[Value],
    suppressions: impl IntoIterator<Item = &'suppression SuppressionRecord>,
    unestablished_ids: &[&str],
    considered_count: usize,
) -> Result<MvpReports, String> {
    let active_id_refs = unique_finding_id_refs(active, "active")?;
    let active_ids = active_id_refs
        .iter()
        .map(|candidate_id| (*candidate_id).to_string())
        .collect::<Vec<_>>();
    let mut suppressed_ids = BTreeSet::new();
    let mut suppressed_count = 0usize;
    let mut pure_suppressed_count = 0usize;
    for suppression in suppressions {
        suppression.validate(&active_id_refs)?;
        let candidate_id = suppression.candidate_id();
        if active_id_refs.binary_search(&candidate_id).is_ok() {
            return Err(format!(
                "candidate {candidate_id} is both active and suppressed"
            ));
        }
        if !suppressed_ids.insert(candidate_id) {
            return Err(format!(
                "duplicate suppressed candidate id in MVP ledger: {candidate_id}"
            ));
        }
        suppressed_count += 1;
        if suppression.replacement_candidate_id.is_none() {
            pure_suppressed_count += 1;
        }
    }
    let mut previous_unestablished = None;
    for decision_id in unestablished_ids {
        validate_sha256(decision_id, "unestablished decision id")?;
        if previous_unestablished.is_some_and(|previous| previous >= *decision_id) {
            return Err(
                "unestablished decision identities are duplicated or not sorted".to_string(),
            );
        }
        if active_id_refs.binary_search(decision_id).is_ok() {
            return Err(format!(
                "decision {decision_id} is both active and unestablished"
            ));
        }
        if suppressed_ids.contains(decision_id) {
            return Err(format!(
                "decision {decision_id} is both suppressed and unestablished"
            ));
        }
        previous_unestablished = Some(*decision_id);
    }
    let unestablished_count = unestablished_ids.len();
    if active.len() + suppressed_count + unestablished_count != considered_count {
        return Err(format!(
            "MVP candidate reconciliation failed: active={} + suppressed={} + unestablished={} != considered={considered_count}",
            active.len(),
            suppressed_count,
            unestablished_count,
        ));
    }

    drop(suppressed_ids);
    drop(active_id_refs);
    let families = build_families(active)?;
    let reclassified_count = suppressed_count - pure_suppressed_count;
    Ok(MvpReports {
        families,
        active_count: active.len(),
        suppressed_count,
        unestablished_count,
        considered_count,
        pure_suppressed_count,
        reclassified_count,
        active_ids,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FamilyKey {
    primary_pattern: String,
    unsafe_site_id: String,
    subject_role: String,
    failure_kind: String,
    failure_span: String,
    contract_model_id: String,
    contract_predicates: Vec<String>,
    source_kind: String,
    source_identity: String,
    exposure_kind: String,
    exposure_identity: String,
    exposure_span: String,
}

impl FamilyKey {
    fn from_finding(finding: &Value) -> Result<Self, String> {
        let primary_pattern =
            required_str(finding, "/primary_pattern", "primary pattern")?.to_string();
        let (exposure_kind, exposure_identity, exposure_span) = if primary_pattern == "P5" {
            let exposure = finding
                .get("exposure")
                .ok_or_else(|| "MVP P5 finding lacks exposure".to_string())?;
            if exposure.is_null() {
                (String::new(), String::new(), String::new())
            } else {
                (
                    required_str(exposure, "/kind", "exposure kind")?.to_string(),
                    required_str(exposure, "/identity", "exposure identity")?.to_string(),
                    span_token(
                        exposure
                            .get("span")
                            .ok_or_else(|| "MVP P5 finding exposure lacks span".to_string())?,
                        "exposure span",
                    )?,
                )
            }
        } else {
            (String::new(), String::new(), String::new())
        };
        let mut contract_predicates = finding
            .pointer("/contract/obligation/predicates")
            .map(|predicates| {
                predicates
                    .as_array()
                    .ok_or_else(|| "contract predicates are not an array".to_string())?
                    .iter()
                    .map(|predicate| {
                        predicate
                            .as_str()
                            .filter(|predicate| !predicate.is_empty())
                            .map(str::to_string)
                            .ok_or_else(|| {
                                "contract predicate is not a non-empty string".to_string()
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        contract_predicates.sort();
        contract_predicates.dedup();

        Ok(Self {
            primary_pattern,
            unsafe_site_id: required_str(finding, "/unsafe_site/site_id", "unsafe site id")?
                .to_string(),
            subject_role: required_str(finding, "/subject_role", "selected subject role")?
                .to_string(),
            failure_kind: required_str(
                finding,
                "/first_contract_failure/kind",
                "first failure kind",
            )?
            .to_string(),
            failure_span: span_token(
                finding
                    .pointer("/first_contract_failure/span")
                    .ok_or_else(|| "MVP finding lacks first failure span".to_string())?,
                "first failure span",
            )?,
            contract_model_id: optional_str(finding, "/contract/model_id", "contract model id")?
                .unwrap_or_default()
                .to_string(),
            contract_predicates,
            source_kind: required_str(finding, "/source/kind", "source kind")?.to_string(),
            source_identity: required_str(finding, "/source/identity", "source identity")?
                .to_string(),
            exposure_kind,
            exposure_identity,
            exposure_span,
        })
    }

    fn family_id(&self) -> String {
        let prefix = [
            self.primary_pattern.as_str(),
            self.unsafe_site_id.as_str(),
            self.subject_role.as_str(),
            self.failure_kind.as_str(),
            self.failure_span.as_str(),
            self.contract_model_id.as_str(),
        ];
        let suffix = [
            self.source_kind.as_str(),
            self.source_identity.as_str(),
            self.exposure_kind.as_str(),
            self.exposure_identity.as_str(),
            self.exposure_span.as_str(),
        ];
        sha256_hex(&length_prefixed(
            prefix
                .into_iter()
                .chain(self.contract_predicates.iter().map(String::as_str))
                .chain(suffix),
        ))
    }

    fn to_json(&self) -> Value {
        json!({
            "primary_pattern": self.primary_pattern,
            "unsafe_site_id": self.unsafe_site_id,
            "subject_role": self.subject_role,
            "first_contract_failure": {
                "kind": self.failure_kind,
                "span": self.failure_span,
            },
            "contract": {
                "model_id": if self.contract_model_id.is_empty() {
                    Value::Null
                } else {
                    Value::String(self.contract_model_id.clone())
                },
                "predicates": self.contract_predicates,
            },
            "source": {
                "kind": self.source_kind,
                "identity": self.source_identity,
            },
            "exposure": if self.exposure_kind.is_empty() {
                Value::Null
            } else {
                json!({
                    "kind": self.exposure_kind,
                    "identity": self.exposure_identity,
                    "span": self.exposure_span,
                })
            },
        })
    }
}

#[derive(Clone, Copy)]
struct FamilyMember<'finding> {
    finding_id: &'finding str,
    witness_len: usize,
    finding: &'finding Value,
}

struct FamilyAccumulator<'finding> {
    key: FamilyKey,
    raw_count: usize,
    root_identities: BTreeSet<String>,
    shortest: FamilyMember<'finding>,
    longest: FamilyMember<'finding>,
}

fn build_families(active: &[Value]) -> Result<Vec<Value>, String> {
    let mut families = BTreeMap::<String, FamilyAccumulator<'_>>::new();
    for finding in active {
        let key = FamilyKey::from_finding(finding)?;
        validate_sha256(&key.unsafe_site_id, "unsafe site id")?;
        let family_id = key.family_id();
        let member = FamilyMember {
            finding_id: finding_id(finding, "active candidate")?,
            witness_len: validate_witness(finding)?,
            finding,
        };
        match families.get_mut(&family_id) {
            Some(family) => {
                if family.key != key {
                    return Err(format!(
                        "family id collision for structurally different key {family_id}"
                    ));
                }
                family.raw_count += 1;
                family.root_identities.insert(root_identity(finding)?);
                if (member.witness_len, member.finding_id)
                    < (family.shortest.witness_len, family.shortest.finding_id)
                {
                    family.shortest = member;
                }
                if member.witness_len > family.longest.witness_len
                    || (member.witness_len == family.longest.witness_len
                        && member.finding_id < family.longest.finding_id)
                {
                    family.longest = member;
                }
            }
            None => {
                families.insert(
                    family_id,
                    FamilyAccumulator {
                        key,
                        raw_count: 1,
                        root_identities: BTreeSet::from([root_identity(finding)?]),
                        shortest: member,
                        longest: member,
                    },
                );
            }
        }
    }

    families
        .into_iter()
        .map(|(family_id, family)| {
            Ok(json!({
                "schema_version": FAMILY_SCHEMA,
                "family_id": family_id,
                "key": family.key.to_json(),
                "raw_count": family.raw_count,
                "public_root_fanout": family.root_identities.len(),
                "representatives": {
                    "shortest": family.shortest.finding,
                    "longest": family.longest.finding,
                },
            }))
        })
        .collect()
}

fn unique_finding_id_refs<'finding>(
    findings: &'finding [Value],
    label: &str,
) -> Result<Vec<&'finding str>, String> {
    let mut ids = Vec::with_capacity(findings.len());
    for finding in findings {
        let id = finding_id(finding, label)?;
        ids.push(id);
    }
    if !ids.windows(2).all(|pair| pair[0] < pair[1]) {
        ids.sort_unstable();
    }
    if let Some(duplicate) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(format!("duplicate {label} candidate id: {}", duplicate[0]));
    }
    Ok(ids)
}

fn finding_id<'a>(finding: &'a Value, label: &str) -> Result<&'a str, String> {
    let id = required_str(finding, "/finding_id", &format!("{label} finding id"))?;
    validate_sha256(id, &format!("{label} finding id"))?;
    Ok(id)
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not a canonical lowercase SHA-256"));
    }
    Ok(())
}

fn root_identity(finding: &Value) -> Result<String, String> {
    let owner = required_str(finding, "/public_safe_root/owner", "public root owner")?;
    let span = span_token(
        finding
            .pointer("/public_safe_root/span")
            .ok_or_else(|| "MVP finding lacks public root span".to_string())?,
        "public root span",
    )?;
    Ok(String::from_utf8(length_prefixed([owner, span.as_str()]))
        .expect("length-prefixed UTF-8 fields remain UTF-8"))
}

fn validate_witness(finding: &Value) -> Result<usize, String> {
    let finding_id = finding_id(finding, "active candidate")?;
    let witness = finding
        .pointer("/propagation/witness")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("active candidate {finding_id} lacks a witness array"))?;
    if witness.is_empty() {
        return Err(format!(
            "active candidate {finding_id} has an empty witness"
        ));
    }
    for (index, step) in witness.iter().enumerate() {
        required_str(step, "/kind", "witness step kind")?;
        required_str(step, "/owner", "witness step owner")?;
        validate_span(
            step.get("span")
                .ok_or_else(|| format!("witness step {index} lacks span"))?,
            "witness step span",
        )?;
    }
    Ok(witness.len())
}

fn span_token(span: &Value, label: &str) -> Result<String, String> {
    let (path, start_line, start_column, end_line, end_column) = validated_span(span, label)?;
    Ok(format!(
        "{path}:{start_line}:{start_column}-{end_line}:{end_column}"
    ))
}

fn validate_span(span: &Value, label: &str) -> Result<(), String> {
    validated_span(span, label).map(|_| ())
}

fn validated_span<'span>(
    span: &'span Value,
    label: &str,
) -> Result<(&'span str, u64, u64, u64, u64), String> {
    let path = span
        .pointer("/path")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| format!("MVP finding {label} path is missing or not a non-empty string"))?;
    let coordinate = |pointer: &str, field: &str| {
        span.pointer(pointer)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("MVP finding {label} {field} is missing or not an integer"))
    };
    let start_line = coordinate("/start/line", "start line")?;
    let start_column = coordinate("/start/column", "start column")?;
    let end_line = coordinate("/end/line", "end line")?;
    let end_column = coordinate("/end/column", "end column")?;
    if path.is_empty() || start_line == 0 || start_column == 0 || end_line == 0 || end_column == 0 {
        return Err(format!("{label} is not canonical"));
    }
    if (end_line, end_column) < (start_line, start_column) {
        return Err(format!("{label} ends before it starts"));
    }
    Ok((path, start_line, start_column, end_line, end_column))
}

fn required_str<'a>(value: &'a Value, pointer: &str, label: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("MVP finding {label} is missing or not a non-empty string"))
}

fn optional_str<'a>(
    value: &'a Value,
    pointer: &str,
    label: &str,
) -> Result<Option<&'a str>, String> {
    match value.pointer(pointer) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value)),
        _ => Err(format!(
            "MVP finding {label} is neither null nor a non-empty string"
        )),
    }
}

fn length_prefixed<'field>(fields: impl IntoIterator<Item = &'field str>) -> Vec<u8> {
    let mut result = Vec::new();
    for field in fields {
        result.extend_from_slice(field.len().to_string().as_bytes());
        result.push(b':');
        result.extend_from_slice(field.as_bytes());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(line: u64) -> Value {
        json!({
            "path": "src/lib.rs",
            "start": {"line": line, "column": 1},
            "end": {"line": line, "column": 2},
        })
    }

    fn witness(kind: &str, owner: &str, line: u64) -> Value {
        json!({"kind": kind, "owner": owner, "span": span(line)})
    }

    fn finding(id: char, root: &str, role: &str, witness_len: usize) -> Value {
        let mut witness_steps = vec![witness("entry", root, 1)];
        for line in 2..=witness_len {
            witness_steps.push(witness("local_call", "sample::middle", line as u64));
        }
        json!({
            "finding_id": id.to_string().repeat(64),
            "primary_pattern": "P1",
            "mechanism_id": "p1_public_parameter_flow_v1",
            "classification_confidence": "structural",
            "subject_role": role,
            "public_safe_root": {"owner": root, "span": span(1), "code_id": "c".repeat(64)},
            "source": {"kind": "public_parameter", "identity": "sample::sink:formal:0", "span": span(2)},
            "first_contract_failure": {"kind": "precondition:raw_pointer_dereference", "span": span(8)},
            "unsafe_site": {
                "site_id": "d".repeat(64),
                "kind": "raw_pointer_dereference",
                "owner": "sample::sink",
                "span": span(8),
                "callee": null,
                "operands": ["place"],
                "destination": null,
                "binding": "bound_exact"
            },
            "exposure": null,
            "contract": {
                "precision": "generic",
                "model_id": "raw_deref_generic_v1",
                "relevant_roles": ["place"],
                "obligation": {"predicates": ["initialized", "aligned"]},
                "safety_doc": {"status": "unavailable", "code_id": null, "reason": "language site"}
            },
            "propagation": {"depth": "inter_procedural", "boundaries": [], "witness": witness_steps},
            "code_refs": ["c".repeat(64)],
        })
    }

    #[test]
    fn families_exclude_root_witness_and_code_and_are_input_order_independent() {
        let short = finding('a', "sample::root_a", "place", 2);
        let mut long = finding('b', "sample::root_b", "place", 4);
        long["code_refs"] = json!(["e".repeat(64)]);

        let forward = build_reports(
            &[short.clone(), long.clone()],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        let reverse =
            build_reports(&[long, short], std::iter::empty::<&SuppressionRecord>(), 2).unwrap();
        assert_eq!(
            serde_json::to_vec(&forward.families).unwrap(),
            serde_json::to_vec(&reverse.families).unwrap()
        );
        assert_eq!(forward.families.len(), 1);
        let family = &forward.families[0];
        assert_eq!(family["raw_count"], 2);
        assert_eq!(family["public_root_fanout"], 2);
        assert_eq!(
            family["representatives"]["shortest"]["finding_id"],
            "a".repeat(64)
        );
        assert_eq!(
            family["representatives"]["longest"]["finding_id"],
            "b".repeat(64)
        );
    }

    #[test]
    fn family_representative_ties_use_the_smallest_finding_id() {
        let equal_b = finding('b', "sample::root_b", "place", 3);
        let equal_a = finding('a', "sample::root_a", "place", 3);
        let equal = build_reports(
            &[equal_b, equal_a],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        assert_eq!(
            equal.families[0]["representatives"]["shortest"]["finding_id"],
            "a".repeat(64)
        );
        assert_eq!(
            equal.families[0]["representatives"]["longest"]["finding_id"],
            "a".repeat(64)
        );

        let short = finding('c', "sample::root_short", "place", 2);
        let long_b = finding('b', "sample::root_b", "place", 4);
        let long_a = finding('a', "sample::root_a", "place", 4);
        let mixed = build_reports(
            &[long_b, short, long_a],
            std::iter::empty::<&SuppressionRecord>(),
            3,
        )
        .unwrap();
        assert_eq!(
            mixed.families[0]["representatives"]["shortest"]["finding_id"],
            "c".repeat(64)
        );
        assert_eq!(
            mixed.families[0]["representatives"]["longest"]["finding_id"],
            "a".repeat(64)
        );
    }

    #[test]
    fn borrowed_family_id_encoding_matches_the_legacy_owned_fields() {
        let finding = finding('a', "sample::root", "place", 2);
        let key = FamilyKey::from_finding(&finding).unwrap();
        let mut legacy_fields = vec![
            key.primary_pattern.clone(),
            key.unsafe_site_id.clone(),
            key.subject_role.clone(),
            key.failure_kind.clone(),
            key.failure_span.clone(),
            key.contract_model_id.clone(),
        ];
        legacy_fields.extend(key.contract_predicates.iter().cloned());
        legacy_fields.extend([
            key.source_kind.clone(),
            key.source_identity.clone(),
            key.exposure_kind.clone(),
            key.exposure_identity.clone(),
            key.exposure_span.clone(),
        ]);
        let mut legacy_bytes = Vec::new();
        for field in legacy_fields {
            legacy_bytes.extend_from_slice(field.len().to_string().as_bytes());
            legacy_bytes.push(b':');
            legacy_bytes.extend_from_slice(field.as_bytes());
        }
        assert_eq!(key.family_id(), sha256_hex(&legacy_bytes));
    }

    #[test]
    fn selected_subject_role_and_contract_predicates_are_part_of_the_family_key() {
        let first = finding('a', "sample::root", "place", 2);
        let mut different_role = finding('b', "sample::root", "address", 2);
        different_role["contract"]["obligation"]["predicates"] = json!(["aligned", "initialized"]);
        let reports = build_reports(
            &[first, different_role],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        assert_eq!(reports.families.len(), 2);
        assert_ne!(
            reports.families[0]["key"]["subject_role"],
            reports.families[1]["key"]["subject_role"]
        );
    }

    #[test]
    fn predicate_order_does_not_split_a_family() {
        let first = finding('a', "sample::root_a", "place", 2);
        let mut second = finding('b', "sample::root_b", "place", 3);
        second["contract"]["obligation"]["predicates"] = json!(["aligned", "initialized"]);
        assert_eq!(
            build_reports(
                &[first, second],
                std::iter::empty::<&SuppressionRecord>(),
                2,
            )
            .unwrap()
            .families
            .len(),
            1
        );
    }

    #[test]
    fn capability_witness_evidence_does_not_split_a_source_family() {
        let mut first = finding('a', "sample::root_a", "place", 2);
        first["primary_pattern"] = json!("P4");
        first["source"] = json!({
            "kind": "generic_capability_gap",
            "identity": "sample::sink:generic_capability:layout-obligation@src/lib.rs:3:1-3:2",
            "span": span(3),
            "generic_capability": {
                "witness": {"status": "proven", "assignment": [{"role": "target_pointee", "type": "u64"}]},
            },
        });
        let mut second = finding('b', "sample::root_b", "place", 3);
        second["primary_pattern"] = json!("P4");
        second["source"] = json!({
            "kind": "generic_capability_gap",
            "identity": "sample::sink:generic_capability:layout-obligation@src/lib.rs:3:1-3:2",
            "span": span(3),
            "generic_capability": {
                "witness": {"status": "unestablished", "reason": "no_finite_witness"},
            },
        });

        let reports = build_reports(
            &[first, second],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        assert_eq!(reports.families.len(), 1);
        assert_eq!(reports.families[0]["raw_count"], 2);
    }

    #[test]
    fn p5_exposure_identity_is_part_of_the_family_key() {
        let mut first = finding('a', "sample::root", "result", 2);
        first["primary_pattern"] = json!("P5");
        first["exposure"] = json!({
            "kind": "public_store",
            "identity": "sample::root:exposure:_3@sample::root:1:2@src/lib.rs:12:1-12:2",
            "span": span(12),
        });
        let mut second = first.clone();
        second["finding_id"] = json!("b".repeat(64));
        second["exposure"]["identity"] =
            json!("sample::root:exposure:_3@sample::root:1:3@src/lib.rs:12:1-12:2");

        let forward = build_reports(
            &[first.clone(), second.clone()],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        let reverse = build_reports(
            &[second, first],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        assert_eq!(forward.families.len(), 2);
        assert_eq!(
            serde_json::to_vec(&forward.families).unwrap(),
            serde_json::to_vec(&reverse.families).unwrap()
        );
    }

    #[test]
    fn non_p5_exposure_is_ignored_and_does_not_destabilize_family_order() {
        let mut first = finding('a', "sample::root_a", "place", 2);
        first["exposure"] = json!({
            "kind": "public_return",
            "identity": "sample::root_a:exposure:_0@src/lib.rs:12:1-12:2",
            "span": span(12),
        });
        let mut second = finding('b', "sample::root_b", "place", 3);
        second["exposure"] = json!({
            "kind": "public_store",
            "identity": "sample::root_b:exposure:_3@src/lib.rs:18:1-18:2",
            "span": span(18),
        });

        let forward = build_reports(
            &[first.clone(), second.clone()],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        let reverse = build_reports(
            &[second, first],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap();
        assert_eq!(forward.families.len(), 1);
        assert_eq!(forward.families[0]["key"]["exposure"], Value::Null);
        assert_eq!(
            serde_json::to_vec(&forward.families).unwrap(),
            serde_json::to_vec(&reverse.families).unwrap()
        );
    }

    #[test]
    fn suppression_ledger_preserves_candidate_reason_evidence_and_replacement() {
        let active = finding('a', "sample::root", "place", 2);
        let suppressed = finding('b', "sample::root", "place", 2);
        let evidence = json!({"source_validity": "all_bits", "target_validity": "all_bits"});
        let record = SuppressionRecord::new(
            "b".repeat(64),
            SuppressionReason::ContractImpliedByType,
            Some("a".repeat(64)),
        );
        let ledger_row = record
            .clone()
            .into_json(suppressed.clone(), evidence)
            .unwrap();
        let reports = build_reports(&[active], [&record], 2).unwrap();
        assert_eq!(reports.active_count, 1);
        assert_eq!(reports.suppressed_count, 1);
        assert_eq!(ledger_row["candidate"], suppressed);
        assert_eq!(ledger_row["reason"], "contract_implied_by_type");
        assert_eq!(ledger_row["replacement_candidate_id"], "a".repeat(64));
        assert_eq!(
            reports.candidate_accounting_json(),
            json!({
                "active_count": 1,
                "pure_suppressed_count": 0,
                "reclassified_count": 1,
                "unestablished_count": 0,
                "suppressed_count": 1,
                "considered_count": 2,
            })
        );
    }

    #[test]
    fn candidate_accounting_distinguishes_pure_suppression_from_reclassification() {
        let active = finding('a', "sample::root", "place", 2);
        let pure =
            SuppressionRecord::new("b".repeat(64), SuppressionReason::ContractDischarged, None);
        let reclassified = SuppressionRecord::new(
            "c".repeat(64),
            SuppressionReason::PointerViewHasNoCapabilityGap,
            Some("a".repeat(64)),
        );
        let reports = build_reports(&[active], [&pure, &reclassified], 3).unwrap();
        assert_eq!(
            reports.candidate_accounting_json(),
            json!({
                "active_count": 1,
                "pure_suppressed_count": 1,
                "reclassified_count": 1,
                "unestablished_count": 0,
                "suppressed_count": 2,
                "considered_count": 3,
            })
        );
    }

    #[test]
    fn candidate_accounting_includes_a_disjoint_unestablished_population() {
        let active = finding('a', "sample::root", "place", 2);
        let pure =
            SuppressionRecord::new("b".repeat(64), SuppressionReason::ContractDischarged, None);
        let reclassified = SuppressionRecord::new(
            "c".repeat(64),
            SuppressionReason::PointerViewHasNoCapabilityGap,
            Some("a".repeat(64)),
        );
        let ids = ["d".repeat(64), "e".repeat(64), "f".repeat(64)];
        let id_refs = ids.iter().map(String::as_str).collect::<Vec<_>>();
        let reports =
            build_reports_with_unestablished(&[active], [&pure, &reclassified], &id_refs, 6)
                .unwrap();
        assert_eq!(
            reports.candidate_accounting_json(),
            json!({
                "active_count": 1,
                "pure_suppressed_count": 1,
                "reclassified_count": 1,
                "unestablished_count": 3,
                "suppressed_count": 2,
                "considered_count": 6,
            })
        );
    }

    #[test]
    fn reconciliation_rejects_unestablished_overlap_and_order_drift() {
        let active = finding('a', "sample::root", "place", 2);
        let suppressed =
            SuppressionRecord::new("b".repeat(64), SuppressionReason::ContractDischarged, None);
        assert!(build_reports_with_unestablished(
            &[active.clone()],
            [&suppressed],
            &[&"a".repeat(64)],
            3,
        )
        .unwrap_err()
        .contains("both active and unestablished"));
        assert!(build_reports_with_unestablished(
            &[active.clone()],
            [&suppressed],
            &[&"b".repeat(64)],
            3,
        )
        .unwrap_err()
        .contains("both suppressed and unestablished"));
        assert!(build_reports_with_unestablished(
            &[active],
            [&suppressed],
            &[&"d".repeat(64), &"c".repeat(64)],
            4,
        )
        .unwrap_err()
        .contains("duplicated or not sorted"));
    }

    #[test]
    fn reconciliation_rejects_silent_count_drift_and_identity_overlap() {
        let active = finding('a', "sample::root", "place", 2);
        assert!(build_reports(
            &[active.clone()],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap_err()
        .contains("reconciliation failed"));

        let overlap =
            SuppressionRecord::new("a".repeat(64), SuppressionReason::ContractDischarged, None);
        assert!(build_reports(&[active], [&overlap], 2)
            .unwrap_err()
            .contains("both active and suppressed"));
    }

    #[test]
    fn same_identity_with_different_candidate_json_fails_closed() {
        let first = finding('a', "sample::root", "argument:0", 2);
        let mut different = first.clone();
        different["subject_role"] = json!("argument:1");
        assert!(build_reports(
            &[first, different],
            std::iter::empty::<&SuppressionRecord>(),
            2,
        )
        .unwrap_err()
        .contains("duplicate active candidate id"));
    }

    #[test]
    fn reconciliation_rejects_active_identity_drift_after_aggregation() {
        let active = finding('a', "sample::root", "place", 2);
        let reports = build_reports(
            &[active.clone()],
            std::iter::empty::<&SuppressionRecord>(),
            1,
        )
        .unwrap();
        let mut changed = active;
        changed["finding_id"] = json!("b".repeat(64));
        assert!(reports
            .reconcile_with_active(&[changed])
            .unwrap_err()
            .contains("identities drifted"));
    }

    #[test]
    fn replacement_must_name_an_active_candidate() {
        let active = finding('a', "sample::root", "place", 2);
        let record = SuppressionRecord::new(
            "b".repeat(64),
            SuppressionReason::PointerViewHasNoCapabilityGap,
            Some("c".repeat(64)),
        );
        assert!(build_reports(&[active], [&record], 2)
            .unwrap_err()
            .contains("non-active replacement"));
    }

    #[test]
    fn suppression_reason_vocabulary_is_closed_and_stable() {
        let reasons = [
            SuppressionReason::ContractImpliedByType,
            SuppressionReason::PointerViewHasNoCapabilityGap,
            SuppressionReason::RootHasNoRelevantCapabilitySurface,
            SuppressionReason::NonpersistentPublicField,
            SuppressionReason::TraitNotDownstreamImplementable,
            SuppressionReason::InternalBehaviorController,
            SuppressionReason::UnrealizableCallContext,
            SuppressionReason::ContractDischarged,
        ];
        assert_eq!(
            reasons.map(SuppressionReason::as_str),
            [
                "contract_implied_by_type",
                "pointer_view_has_no_capability_gap",
                "root_has_no_relevant_capability_surface",
                "nonpersistent_public_field",
                "trait_not_downstream_implementable",
                "internal_behavior_controller",
                "unrealizable_call_context",
                "contract_discharged",
            ]
        );
    }
}
