//! Compiler-aligned unsafe-site descriptors.
//!
//! These types are deliberately owned. A collector must not retain THIR or
//! `Ty<'tcx>` values after the wrapped rustc query returns.

use std::collections::BTreeSet;
use std::env;
use std::path::PathBuf;

use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::Span;
use serde_json::{json, Value};

use super::summary::{sha256_hex, FunctionKey, StableSpan};

pub(super) mod binding;
mod current_2026;
mod legacy_2024;
mod store;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnsafeSiteKind {
    UnsafeCall,
    InlineAssembly,
    RestrictedValidityInitialization,
    UnsafeFieldInitialization,
    UnsafeFieldUse,
    MutableStaticAccess,
    ExternStaticAccess,
    RawPointerDereference,
    UnionFieldAccess,
    LayoutConstrainedMutation,
    LayoutConstrainedBorrow,
    TargetFeatureCall,
    UnsafeBinderCast,
    ExplicitDropCall,
}

impl UnsafeSiteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnsafeCall => "unsafe_call",
            Self::InlineAssembly => "inline_assembly",
            Self::RestrictedValidityInitialization => "restricted_validity_initialization",
            Self::UnsafeFieldInitialization => "unsafe_field_initialization",
            Self::UnsafeFieldUse => "unsafe_field_use",
            Self::MutableStaticAccess => "mutable_static_access",
            Self::ExternStaticAccess => "extern_static_access",
            Self::RawPointerDereference => "raw_pointer_dereference",
            Self::UnionFieldAccess => "union_field_access",
            Self::LayoutConstrainedMutation => "layout_constrained_mutation",
            Self::LayoutConstrainedBorrow => "layout_constrained_borrow",
            Self::TargetFeatureCall => "target_feature_call",
            Self::UnsafeBinderCast => "unsafe_binder_cast",
            Self::ExplicitDropCall => "explicit_drop_call",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RawSpanAnchor {
    pub stable: StableSpan,
    pub lo_byte: u32,
    pub hi_byte: u32,
    pub syntax_context: String,
}

impl RawSpanAnchor {
    pub fn new(
        stable: StableSpan,
        lo_byte: u32,
        hi_byte: u32,
        syntax_context: impl Into<String>,
    ) -> Self {
        assert!(hi_byte >= lo_byte, "raw span end must not precede start");
        Self {
            stable,
            lo_byte,
            hi_byte,
            syntax_context: syntax_context.into(),
        }
    }

    fn canonical_token(&self) -> String {
        format!(
            "{}@{}-{}#{}",
            self.stable.token(),
            self.lo_byte,
            self.hi_byte,
            self.syntax_context
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperandRole {
    Receiver,
    Argument(u32),
    Place,
    Address,
    Index,
    Length,
    Size,
    Alignment,
    Layout,
    Discriminant,
    Out,
    Callee,
    Other(u32),
}

impl OperandRole {
    fn canonical_token(&self) -> String {
        match self {
            Self::Receiver => "receiver".to_string(),
            Self::Argument(index) => format!("argument:{index}"),
            Self::Place => "place".to_string(),
            Self::Address => "address".to_string(),
            Self::Index => "index".to_string(),
            Self::Length => "length".to_string(),
            Self::Size => "size".to_string(),
            Self::Alignment => "alignment".to_string(),
            Self::Layout => "layout".to_string(),
            Self::Discriminant => "discriminant".to_string(),
            Self::Out => "out".to_string(),
            Self::Callee => "callee".to_string(),
            Self::Other(index) => format!("other:{index}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperandAnchor {
    pub role: OperandRole,
    pub span: RawSpanAnchor,
    pub type_fingerprint: String,
    pub shape: String,
}

impl OperandAnchor {
    fn canonical_token(&self) -> String {
        format!(
            "{}@{}:{}:{}",
            self.role.canonical_token(),
            self.span.canonical_token(),
            self.type_fingerprint,
            self.shape
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResultAnchor {
    pub span: RawSpanAnchor,
    pub type_fingerprint: String,
    pub shape: String,
}

impl ResultAnchor {
    fn canonical_token(&self) -> String {
        format!(
            "{}:{}:{}",
            self.span.canonical_token(),
            self.type_fingerprint,
            self.shape
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnsafeSiteSeed {
    pub adapter_id: String,
    pub owner: FunctionKey,
    pub owner_def_path_hash: String,
    pub ordinal: u32,
    pub kind: UnsafeSiteKind,
    pub raw_span: RawSpanAnchor,
    pub hir_context: String,
    pub callee: Option<String>,
    pub operands: Vec<OperandAnchor>,
    pub result: Option<ResultAnchor>,
}

impl UnsafeSiteSeed {
    pub fn site_id(&self) -> String {
        let fields = vec![
            self.adapter_id.clone(),
            self.owner_def_path_hash.clone(),
            self.ordinal.to_string(),
            self.kind.as_str().to_string(),
        ];
        sha256_hex(&length_prefixed(&fields))
    }
}

fn length_prefixed(fields: &[String]) -> Vec<u8> {
    let mut canonical = Vec::new();
    for field in fields {
        canonical.extend_from_slice(field.len().to_string().as_bytes());
        canonical.push(b':');
        canonical.extend_from_slice(field.as_bytes());
    }
    canonical
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerCensus {
    pub owner: FunctionKey,
    pub collector_ran: bool,
    pub sites: Vec<UnsafeSiteSeed>,
}

impl OwnerCensus {
    pub fn validate(&self) -> Result<(), String> {
        if !self.collector_ran && !self.sites.is_empty() {
            return Err("owner census cannot contain sites when collector did not run".to_string());
        }
        let mut ordinals = BTreeSet::new();
        let mut site_ids = BTreeSet::new();
        let mut previous = None;
        for site in &self.sites {
            if site.owner != self.owner {
                return Err("unsafe site owner differs from owner census".to_string());
            }
            if previous.is_some_and(|value| site.ordinal <= value) {
                return Err("unsafe sites must be sorted by unique traversal ordinal".to_string());
            }
            previous = Some(site.ordinal);
            if !ordinals.insert(site.ordinal) || !site_ids.insert(site.site_id()) {
                return Err("unsafe site ordinal and id must be unique per owner".to_string());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterManifest {
    pub adapter_id: String,
    pub rustc_commit: String,
    pub checker_sha256: String,
    pub emitted_kinds: BTreeSet<UnsafeSiteKind>,
}

pub fn adapter_for_commit(rustc_commit: &str) -> Result<AdapterManifest, String> {
    match rustc_commit {
        legacy_2024::RUSTC_COMMIT => Ok(legacy_2024::manifest()),
        current_2026::RUSTC_COMMIT => Ok(current_2026::manifest()),
        _ => Err(format!(
            "unsupported rustc commit {rustc_commit}; UnsoundAudit v3 supports only {} and {}",
            legacy_2024::RUSTC_COMMIT,
            current_2026::RUSTC_COMMIT
        )),
    }
}

pub(crate) fn begin_census_invocation(rustc_commit: &str) -> Result<u64, String> {
    adapter_for_commit(rustc_commit)?.validate()?;
    store::begin_global_invocation()
}

pub(crate) fn census_generation() -> Result<u64, String> {
    store::active_generation()
}

pub(crate) fn census_store() -> &'static store::CensusStore {
    store::global_store()
}

#[rustversion::before(1.96)]
pub(crate) fn install_census_provider(
    providers: &mut rustc_middle::util::Providers,
) -> Result<(), String> {
    legacy_2024::install(providers)
}

#[rustversion::since(1.96)]
pub(crate) fn install_census_provider(
    providers: &mut rustc_middle::util::Providers,
) -> Result<(), String> {
    current_2026::install(providers)
}

#[rustversion::before(1.96)]
pub(crate) fn completeness_sweep(tcx: TyCtxt<'_>) {
    legacy_2024::sweep(tcx);
}

#[rustversion::since(1.96)]
pub(crate) fn completeness_sweep(tcx: TyCtxt<'_>) {
    current_2026::sweep(tcx);
}

pub(crate) fn census_snapshot() -> Result<store::CensusSnapshot, String> {
    let generation = census_generation()?;
    census_store().snapshot(generation)
}

pub(super) fn record_built_mir_events(
    tcx: TyCtxt<'_>,
    owner_id: rustc_span::def_id::LocalDefId,
) -> Result<(), String> {
    let generation = census_generation()?;
    let owner = FunctionKey::new(tcx.def_path_str(owner_id.to_def_id()));
    if census_store()
        .snapshot(generation)?
        .owners
        .get(&owner)
        .is_none_or(|census| census.sites.is_empty())
    {
        return Ok(());
    }
    if census_store().contains_built_events(generation, &owner)? {
        return Ok(());
    }
    let body = tcx.mir_drops_elaborated_and_const_checked(owner_id);
    if body.is_stolen() {
        return record_built_inventory_failure(
            generation,
            owner,
            "built MIR was stolen before unsafe-site binding inventory",
        );
    }
    let body = body.borrow();
    let events = match binding::enumerate_mir_events(tcx, owner_id, &body, binding::MirStage::Built)
    {
        Ok(events) => events,
        Err(error) => {
            drop(body);
            return record_built_inventory_failure(generation, owner, &error);
        }
    };
    census_store().record_built_events(generation, owner, events)
}

fn record_built_inventory_failure(
    generation: u64,
    owner: FunctionKey,
    reason: &str,
) -> Result<(), String> {
    log::error!(
        "unsafe-site built MIR inventory failed for {}: {reason}",
        owner.0
    );
    census_store().record_failure(
        generation,
        CoverageFailure {
            kind: CoverageFailureKind::CollectorFailed,
            owner: owner.clone(),
            site_id: None,
            candidate_count: 0,
        },
    )?;
    census_store().record_built_events(generation, owner, Vec::new())
}

pub(crate) fn census_only_from_environment() -> Result<bool, String> {
    match env::var_os("UNSOUND_SCANNER_CENSUS_ONLY") {
        None => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err("UNSOUND_SCANNER_CENSUS_ONLY must be unset or exactly '1'".to_string()),
    }
}

pub(crate) fn maybe_write_census_sidecar() -> Result<(), String> {
    let Some(output_path) = env::var_os("UNSOUND_SCANNER_CENSUS_JSON_OUT") else {
        return Ok(());
    };
    let snapshot = census_snapshot()?;
    let owners = snapshot
        .owners
        .values()
        .map(|owner| {
            json!({
                "owner": owner.owner.0,
                "collector_ran": owner.collector_ran,
                "sites": owner.sites.iter().map(site_json).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let failures = snapshot
        .failures
        .iter()
        .map(|failure| {
            json!({
                "kind": coverage_failure_kind(failure.kind),
                "owner": failure.owner.0,
                "site_id": failure.site_id,
                "candidate_count": failure.candidate_count,
            })
        })
        .collect::<Vec<_>>();
    let payload = json!({
        "schema_version": "unsafe-site-census-v1",
        "generation": snapshot.generation,
        "owners": owners,
        "coverage_failures": failures,
    });
    let mut bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|error| format!("failed to serialize unsafe-site census: {error}"))?;
    bytes.push(b'\n');
    crate::utils::unit_output::atomic_write(&PathBuf::from(output_path), &bytes)
        .map_err(|error| format!("failed to write unsafe-site census sidecar: {error}"))
}

fn site_json(site: &UnsafeSiteSeed) -> Value {
    json!({
        "site_id": site.site_id(),
        "adapter_id": site.adapter_id,
        "owner": site.owner.0,
        "owner_def_path_hash": site.owner_def_path_hash,
        "ordinal": site.ordinal,
        "kind": site.kind.as_str(),
        "span": raw_span_json(&site.raw_span),
        "hir_context": site.hir_context,
        "callee": site.callee,
        "operands": site.operands.iter().map(|operand| json!({
            "role": operand.role.canonical_token(),
            "span": raw_span_json(&operand.span),
            "type_fingerprint": operand.type_fingerprint,
            "shape": operand.shape,
        })).collect::<Vec<_>>(),
        "result": site.result.as_ref().map(|result| json!({
            "span": raw_span_json(&result.span),
            "type_fingerprint": result.type_fingerprint,
            "shape": result.shape,
        })),
    })
}

fn raw_span_json(span: &RawSpanAnchor) -> Value {
    json!({
        "path": span.stable.path,
        "start": {"line": span.stable.start.line, "column": span.stable.start.column},
        "end": {"line": span.stable.end.line, "column": span.stable.end.column},
        "lo_byte": span.lo_byte,
        "hi_byte": span.hi_byte,
        "syntax_context": span.syntax_context,
    })
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

#[rustversion::before(1.96)]
fn type_fingerprint<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> String {
    format!("{:?}", tcx.erase_regions(ty))
}

#[rustversion::since(1.96)]
fn type_fingerprint<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> String {
    format!("{:?}", tcx.erase_and_anonymize_regions(ty))
}

fn raw_span_anchor(tcx: TyCtxt<'_>, raw: Span) -> Result<RawSpanAnchor, String> {
    let project_root = env::var_os("UNSOUND_SCANNER_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let source_root = super::mir::select_source_root(
        &project_root,
        env::var_os("UNSOUND_SCANNER_SOURCE_ROOT").map(PathBuf::from),
    )?;
    let source_map = tcx.sess.source_map();
    let display = raw.source_callsite();
    let (file, lo_line, lo_column, hi_line, hi_column) = source_map.span_to_location_info(display);
    let local = file
        .as_ref()
        .map(|file| super::mir::local_file_name(&file.name));
    let remapped = file
        .as_ref()
        .map(|file| super::mir::remapped_file_name(&file.name));
    let stable = super::mir::stable_span_from_locations(
        &source_root,
        local.as_deref(),
        remapped.as_deref(),
        lo_line,
        lo_column,
        hi_line,
        hi_column,
    )?;
    Ok(RawSpanAnchor::new(
        stable,
        raw.lo().0,
        raw.hi().0,
        format!("{:?}", raw.ctxt()),
    ))
}

impl AdapterManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.adapter_id.is_empty()
            || self.rustc_commit.len() != 40
            || self.checker_sha256.len() != 64
            || self.emitted_kinds.is_empty()
        {
            return Err("adapter manifest is incomplete".to_string());
        }
        if !self
            .rustc_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || !self
                .checker_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("adapter hashes must be lowercase hexadecimal".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingStatus {
    BoundExact,
    Unbound,
    Ambiguous,
    Elided,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoverageFailureKind {
    CollectorMissingAfterSteal,
    CollectorFailed,
    UnboundUnsafeSite,
    AmbiguousUnsafeSite,
    ElidedUnsafeSite,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoverageFailure {
    pub kind: CoverageFailureKind,
    pub owner: FunctionKey,
    pub site_id: Option<String>,
    pub candidate_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContractPrecision {
    Precise,
    Generic,
    Opaque,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AnalysisStatus {
    Complete,
    Partial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CountSemantics {
    Exact,
    LowerBound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimaryPattern {
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClassificationConfidence {
    Exact,
    Structural,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalSourceKind {
    PublicParameter,
    PublicField,
    InternalUnsafeOrigin,
    InternalDerivation,
    GenericCapabilityGap,
    ExternalBehaviorOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingConfidence {
    BoundExact,
    LowConfidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsafeSiteReference {
    pub site_id: String,
    pub kind: UnsafeSiteKind,
    pub owner: FunctionKey,
    pub span: StableSpan,
    pub callee: Option<String>,
    pub operands: Vec<String>,
    pub destination: Option<String>,
    pub binding: BindingConfidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FindingRoot {
    pub owner: FunctionKey,
    pub span: StableSpan,
    pub code_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FindingSource {
    pub kind: CausalSourceKind,
    pub identity: String,
    pub span: StableSpan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FindingFailure {
    pub kind: String,
    pub span: StableSpan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SafetyDocReference {
    Available { code_id: String },
    Unavailable { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FindingContract {
    pub precision: ContractPrecision,
    pub model_id: Option<String>,
    pub relevant_roles: Vec<String>,
    pub obligation: Option<Value>,
    pub safety_doc: SafetyDocReference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropagationDepth {
    IntraProcedural,
    InterProcedural,
    Recursive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum V3WitnessStepKind {
    Entry,
    Source,
    LocalCall,
    SccCycle,
    UnsafeSite,
    Exposure,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V3WitnessStep {
    pub kind: V3WitnessStepKind,
    pub owner: FunctionKey,
    pub span: StableSpan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FindingPropagation {
    pub depth: PropagationDepth,
    pub boundaries: Vec<FunctionKey>,
    pub witness: Vec<V3WitnessStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V3Finding {
    pub finding_id: String,
    pub primary_pattern: PrimaryPattern,
    pub mechanism_id: String,
    pub classification_confidence: ClassificationConfidence,
    pub public_safe_root: FindingRoot,
    pub source: FindingSource,
    pub first_contract_failure: FindingFailure,
    pub unsafe_site: UnsafeSiteReference,
    pub exposure: Option<FindingFailure>,
    pub contract: FindingContract,
    pub propagation: FindingPropagation,
    pub code_refs: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CodeBlockKind {
    FunctionBody,
    Signature,
    SafetyDoc,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeBlock {
    pub code_id: String,
    pub kind: CodeBlockKind,
    pub owner: String,
    pub path: String,
    pub span: StableSpan,
    pub sha256: String,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::unsoundaudit::summary::StablePosition;
    use std::sync::Arc;
    use std::thread;

    fn span(line: u32) -> RawSpanAnchor {
        RawSpanAnchor::new(
            StableSpan::new(
                "src/lib.rs",
                StablePosition::new(line, 1),
                StablePosition::new(line, 5),
            ),
            line * 10,
            line * 10 + 4,
            "#0",
        )
    }

    fn seed(ordinal: u32, kind: UnsafeSiteKind) -> UnsafeSiteSeed {
        UnsafeSiteSeed {
            adapter_id: "rustc-2024-1bc403".to_string(),
            owner: FunctionKey::new("sample::root"),
            owner_def_path_hash: "a".repeat(32),
            ordinal,
            kind,
            raw_span: span(ordinal + 1),
            hir_context: "hir:42".to_string(),
            callee: Some("core::ptr::read".to_string()),
            operands: vec![OperandAnchor {
                role: OperandRole::Argument(0),
                span: span(ordinal + 1),
                type_fingerprint: "*const u8".to_string(),
                shape: "deref".to_string(),
            }],
            result: Some(ResultAnchor {
                span: span(ordinal + 1),
                type_fingerprint: "u8".to_string(),
                shape: "value".to_string(),
            }),
        }
    }

    #[test]
    fn site_id_is_stable_and_sensitive_to_identity_fields() {
        let original = seed(0, UnsafeSiteKind::UnsafeCall);
        assert_eq!(original.site_id(), original.clone().site_id());

        let mut changed = original.clone();
        changed.ordinal = 1;
        assert_ne!(original.site_id(), changed.site_id());
        changed = original.clone();
        changed.kind = UnsafeSiteKind::RawPointerDereference;
        assert_ne!(original.site_id(), changed.site_id());
        changed = original.clone();
        changed.adapter_id = "rustc-current-ba28ff".to_string();
        assert_ne!(original.site_id(), changed.site_id());
        changed = original.clone();
        changed.owner_def_path_hash = "b".repeat(32);
        assert_ne!(original.site_id(), changed.site_id());

        changed = original.clone();
        changed.raw_span = span(99);
        changed.callee = Some("core::ptr::write".to_string());
        assert_eq!(
            original.site_id(),
            changed.site_id(),
            "source rendering and contract metadata do not participate in site identity"
        );
    }

    #[test]
    fn owner_census_accepts_empty_or_sorted_unique_sites() {
        OwnerCensus {
            owner: FunctionKey::new("sample::root"),
            collector_ran: true,
            sites: Vec::new(),
        }
        .validate()
        .unwrap();

        OwnerCensus {
            owner: FunctionKey::new("sample::root"),
            collector_ran: true,
            sites: vec![
                seed(0, UnsafeSiteKind::UnsafeCall),
                seed(1, UnsafeSiteKind::RawPointerDereference),
            ],
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn owner_census_rejects_uncollected_duplicate_or_unsorted_sites() {
        let site = seed(0, UnsafeSiteKind::UnsafeCall);
        assert!(OwnerCensus {
            owner: FunctionKey::new("sample::root"),
            collector_ran: false,
            sites: vec![site.clone()],
        }
        .validate()
        .is_err());
        assert!(OwnerCensus {
            owner: FunctionKey::new("sample::root"),
            collector_ran: true,
            sites: vec![site.clone(), site],
        }
        .validate()
        .is_err());
        assert!(OwnerCensus {
            owner: FunctionKey::new("sample::root"),
            collector_ran: true,
            sites: vec![
                seed(1, UnsafeSiteKind::RawPointerDereference),
                seed(0, UnsafeSiteKind::UnsafeCall),
            ],
        }
        .validate()
        .is_err());
    }

    #[test]
    fn adapter_manifests_are_exact_and_unsupported_commits_fail_closed() {
        let legacy = adapter_for_commit(legacy_2024::RUSTC_COMMIT).unwrap();
        assert_eq!(legacy.adapter_id, "rustc-nightly-2024-10-12-1bc403");
        assert_eq!(
            legacy.checker_sha256,
            "bde4d27635e54d8ca0582ef8decd204c3089591e5c540322891d7aaf1a182023"
        );
        assert!(legacy
            .emitted_kinds
            .contains(&UnsafeSiteKind::RestrictedValidityInitialization));
        assert!(legacy
            .emitted_kinds
            .contains(&UnsafeSiteKind::LayoutConstrainedBorrow));
        assert!(!legacy
            .emitted_kinds
            .contains(&UnsafeSiteKind::UnsafeFieldUse));
        legacy.validate().unwrap();

        let current = adapter_for_commit(current_2026::RUSTC_COMMIT).unwrap();
        assert_eq!(current.adapter_id, "rustc-nightly-2026-08-14-ba28ff");
        assert_eq!(
            current.checker_sha256,
            "25b4b8e22ff7d561dda2a9abf429ab5bef035d246ec0200b9b887f7713fca346"
        );
        assert!(current
            .emitted_kinds
            .contains(&UnsafeSiteKind::UnsafeBinderCast));
        assert!(current
            .emitted_kinds
            .contains(&UnsafeSiteKind::ExplicitDropCall));
        assert!(!current
            .emitted_kinds
            .contains(&UnsafeSiteKind::RestrictedValidityInitialization));
        current.validate().unwrap();

        let error = adapter_for_commit("0000000000000000000000000000000000000000").unwrap_err();
        assert!(error.contains("unsupported rustc commit"));
    }

    #[test]
    fn census_store_is_generation_scoped_thread_safe_and_owner_sorted() {
        let store = Arc::new(store::CensusStore::default());
        store.begin(7).unwrap();
        let mut workers = Vec::new();
        for index in (0..16).rev() {
            let store = Arc::clone(&store);
            workers.push(thread::spawn(move || {
                let owner = FunctionKey::new(format!("sample::owner_{index:02}"));
                store
                    .record_owner(
                        7,
                        OwnerCensus {
                            owner,
                            collector_ran: true,
                            sites: Vec::new(),
                        },
                    )
                    .unwrap();
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        let snapshot = store.snapshot(7).unwrap();
        let owners = snapshot
            .owners
            .keys()
            .map(|owner| owner.0.clone())
            .collect::<Vec<_>>();
        let mut sorted = owners.clone();
        sorted.sort();
        assert_eq!(owners, sorted);
        assert_eq!(owners.len(), 16);
        assert!(snapshot.failures.is_empty());

        assert!(store.snapshot(8).is_err());
        assert!(store
            .record_owner(
                8,
                OwnerCensus {
                    owner: FunctionKey::new("sample::wrong_generation"),
                    collector_ran: true,
                    sites: Vec::new(),
                },
            )
            .is_err());
    }
}
