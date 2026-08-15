//! Compiler-aligned unsafe-site descriptors.
//!
//! These types are deliberately owned. A collector must not retain THIR or
//! `Ty<'tcx>` values after the wrapped rustc query returns.

use std::collections::BTreeSet;

use serde_json::Value;

use super::summary::{sha256_hex, FunctionKey, StableSpan};

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
    pub syntax_context: u32,
}

impl RawSpanAnchor {
    pub fn new(stable: StableSpan, lo_byte: u32, hi_byte: u32, syntax_context: u32) -> Self {
        assert!(hi_byte >= lo_byte, "raw span end must not precede start");
        Self {
            stable,
            lo_byte,
            hi_byte,
            syntax_context,
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

    fn span(line: u32) -> RawSpanAnchor {
        RawSpanAnchor::new(
            StableSpan::new(
                "src/lib.rs",
                StablePosition::new(line, 1),
                StablePosition::new(line, 5),
            ),
            line * 10,
            line * 10 + 4,
            0,
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
}
