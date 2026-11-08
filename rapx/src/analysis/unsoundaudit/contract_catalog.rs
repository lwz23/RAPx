use std::collections::BTreeSet;

use super::unsafe_site::{
    binding::{BoundUnsafeSite, TypeValidityClass},
    ContractPrecision, OperandRole, UnsafeSiteKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum KnownContractModel {
    GetUnchecked,
    VecGetUnchecked,
    PtrRead,
    ReadUnaligned,
    NonNullNewUnchecked,
    AssumeInit,
    FromUtf8Unchecked,
    StringFromUtf8Unchecked,
    SliceFromRawParts,
    CStrFromPtr,
    CStringFromRaw,
    BoxFromRaw,
    VecFromRawParts,
    PtrOffset,
    PtrCopy,
    Transmute,
    ConditionTrue,
}

impl KnownContractModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GetUnchecked => "slice_get_unchecked_v1",
            Self::VecGetUnchecked => "vec_get_unchecked_v1",
            Self::PtrRead => "ptr_read_v1",
            Self::ReadUnaligned => "ptr_read_unaligned_v1",
            Self::NonNullNewUnchecked => "nonnull_new_unchecked_v1",
            Self::AssumeInit => "maybe_uninit_assume_init_v1",
            Self::FromUtf8Unchecked => "str_from_utf8_unchecked_v1",
            Self::StringFromUtf8Unchecked => "string_from_utf8_unchecked_v1",
            Self::SliceFromRawParts => "slice_from_raw_parts_v1",
            Self::CStrFromPtr => "cstr_from_ptr_v1",
            Self::CStringFromRaw => "cstring_from_raw_v1",
            Self::BoxFromRaw => "box_from_raw_v1",
            Self::VecFromRawParts => "vec_from_raw_parts_v1",
            Self::PtrOffset => "ptr_offset_v1",
            Self::PtrCopy => "ptr_copy_v1",
            Self::Transmute => "transmute_validity_v1",
            Self::ConditionTrue => "condition_true_v1",
        }
    }

    pub fn from_model_id(id: &str) -> Option<Self> {
        match id {
            "slice_get_unchecked_v1" => Some(Self::GetUnchecked),
            "vec_get_unchecked_v1" => Some(Self::VecGetUnchecked),
            "ptr_read_v1" => Some(Self::PtrRead),
            "ptr_read_unaligned_v1" => Some(Self::ReadUnaligned),
            "nonnull_new_unchecked_v1" => Some(Self::NonNullNewUnchecked),
            "maybe_uninit_assume_init_v1" => Some(Self::AssumeInit),
            "str_from_utf8_unchecked_v1" => Some(Self::FromUtf8Unchecked),
            "string_from_utf8_unchecked_v1" => Some(Self::StringFromUtf8Unchecked),
            "slice_from_raw_parts_v1" => Some(Self::SliceFromRawParts),
            "cstr_from_ptr_v1" => Some(Self::CStrFromPtr),
            "cstring_from_raw_v1" => Some(Self::CStringFromRaw),
            "box_from_raw_v1" => Some(Self::BoxFromRaw),
            "vec_from_raw_parts_v1" => Some(Self::VecFromRawParts),
            "ptr_offset_v1" => Some(Self::PtrOffset),
            "ptr_copy_v1" => Some(Self::PtrCopy),
            "transmute_validity_v1" => Some(Self::Transmute),
            "condition_true_v1" => Some(Self::ConditionTrue),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum FailurePhase {
    Precondition,
    ImmediateValidity,
    ExposureInvariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum ContractPredicate {
    ValidForRead,
    InBounds,
    RangeInBounds,
    NonNull,
    Initialized,
    ValidForType,
    ValidUtf8,
    Aligned,
    LayoutCompatible,
    ConditionTrue,
}

impl ContractPredicate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ValidForRead => "valid_for_read",
            Self::InBounds => "in_bounds",
            Self::RangeInBounds => "range_in_bounds",
            Self::NonNull => "non_null",
            Self::Initialized => "initialized",
            Self::ValidForType => "valid_for_type",
            Self::ValidUtf8 => "valid_utf8",
            Self::Aligned => "aligned",
            Self::LayoutCompatible => "layout_compatible",
            Self::ConditionTrue => "condition_true",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum DischargeModel {
    None,
    ExactValidation,
    ExactRelation,
    ExactInitialization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum StaticContractEvidence {
    TargetAllBitsValidScalar,
    TargetAllBitsValidByteForm,
    TargetAllBitsValidRawPointer,
    TargetAllBitsValidStructural,
    SameAllBitsValidStructure,
}

impl StaticContractEvidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TargetAllBitsValidScalar => "target_all_bits_valid_scalar",
            Self::TargetAllBitsValidByteForm => "target_all_bits_valid_byte_form",
            Self::TargetAllBitsValidRawPointer => "target_all_bits_valid_raw_pointer",
            Self::TargetAllBitsValidStructural => "target_all_bits_valid_structural",
            Self::SameAllBitsValidStructure => "same_all_bits_valid_structure",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum StaticContractGap {
    SourceMayContainUninitializedBytes,
    LifetimeOrProvenanceSensitive,
    TargetRestrictedValidity,
    StructuralValidityMismatch,
    UnknownTypeValidity,
}

impl StaticContractGap {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SourceMayContainUninitializedBytes => "source_may_contain_uninitialized_bytes",
            Self::LifetimeOrProvenanceSensitive => "lifetime_or_provenance_sensitive",
            Self::TargetRestrictedValidity => "target_restricted_validity",
            Self::StructuralValidityMismatch => "structural_validity_mismatch",
            Self::UnknownTypeValidity => "unknown_type_validity",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum StaticContractImplication {
    NotApplicable,
    Proven(StaticContractEvidence),
    Unproven(StaticContractGap),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ContractDescriptor {
    pub precision: ContractPrecision,
    pub model_id: Option<String>,
    pub relevant_roles: BTreeSet<OperandRole>,
    pub predicates: BTreeSet<ContractPredicate>,
    pub failure_phase: FailurePhase,
    pub discharge: DischargeModel,
    pub static_implication: StaticContractImplication,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StructuralSiteCandidate {
    pub site_id: String,
    pub relevant_roles: BTreeSet<OperandRole>,
    pub validity_sensitive_result: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ContractCatalog {
    models_enabled: bool,
}

impl ContractCatalog {
    pub fn empty() -> Self {
        Self {
            models_enabled: false,
        }
    }

    pub fn full() -> Self {
        Self {
            models_enabled: true,
        }
    }

    pub fn describe(
        self,
        site: &BoundUnsafeSite,
        known: Option<KnownContractModel>,
    ) -> ContractDescriptor {
        let structural = structural_candidate(site);
        let roles = structural
            .as_ref()
            .map(|candidate| candidate.relevant_roles.clone())
            .unwrap_or_default();
        let opaque = || ContractDescriptor {
            precision: ContractPrecision::Opaque,
            model_id: None,
            relevant_roles: roles.clone(),
            predicates: BTreeSet::new(),
            failure_phase: FailurePhase::Precondition,
            discharge: DischargeModel::None,
            static_implication: StaticContractImplication::NotApplicable,
        };
        if !self.models_enabled {
            return opaque();
        }
        if site.seed.kind == UnsafeSiteKind::UnsafeCall {
            if let Some(model) = known {
                let (selected, predicates, failure_phase, discharge) = match model {
                    KnownContractModel::GetUnchecked | KnownContractModel::VecGetUnchecked => (
                        select_get_unchecked_subject_role(site),
                        BTreeSet::from([ContractPredicate::InBounds]),
                        FailurePhase::Precondition,
                        DischargeModel::ExactValidation,
                    ),
                    KnownContractModel::SliceFromRawParts => (
                        select_pointer_and_extent_roles(site, 0, 1),
                        BTreeSet::from([
                            ContractPredicate::NonNull,
                            ContractPredicate::Aligned,
                            ContractPredicate::RangeInBounds,
                            ContractPredicate::ValidForRead,
                        ]),
                        FailurePhase::Precondition,
                        DischargeModel::None,
                    ),
                    KnownContractModel::CStrFromPtr
                    | KnownContractModel::CStringFromRaw
                    | KnownContractModel::BoxFromRaw => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([ContractPredicate::NonNull]),
                        FailurePhase::Precondition,
                        DischargeModel::None,
                    ),
                    KnownContractModel::VecFromRawParts => (
                        select_pointer_len_cap_roles(site),
                        BTreeSet::from([
                            ContractPredicate::NonNull,
                            ContractPredicate::RangeInBounds,
                            ContractPredicate::LayoutCompatible,
                        ]),
                        FailurePhase::Precondition,
                        DischargeModel::None,
                    ),
                    KnownContractModel::PtrOffset => (
                        select_pointer_and_extent_roles(site, 0, 1),
                        BTreeSet::from([
                            ContractPredicate::NonNull,
                            ContractPredicate::RangeInBounds,
                        ]),
                        FailurePhase::Precondition,
                        DischargeModel::None,
                    ),
                    KnownContractModel::PtrCopy => (
                        select_ptr_copy_roles(site),
                        BTreeSet::from([
                            ContractPredicate::NonNull,
                            ContractPredicate::RangeInBounds,
                            ContractPredicate::ValidForRead,
                        ]),
                        FailurePhase::Precondition,
                        DischargeModel::None,
                    ),
                    KnownContractModel::PtrRead => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([
                            ContractPredicate::ValidForRead,
                            ContractPredicate::RangeInBounds,
                            ContractPredicate::NonNull,
                            ContractPredicate::Aligned,
                            ContractPredicate::Initialized,
                            ContractPredicate::ValidForType,
                        ]),
                        FailurePhase::Precondition,
                        DischargeModel::None,
                    ),
                    KnownContractModel::ReadUnaligned => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([
                            ContractPredicate::ValidForRead,
                            ContractPredicate::RangeInBounds,
                            ContractPredicate::NonNull,
                            ContractPredicate::Initialized,
                            ContractPredicate::ValidForType,
                        ]),
                        FailurePhase::Precondition,
                        DischargeModel::None,
                    ),
                    KnownContractModel::NonNullNewUnchecked => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([ContractPredicate::NonNull]),
                        FailurePhase::Precondition,
                        DischargeModel::ExactValidation,
                    ),
                    KnownContractModel::AssumeInit => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([
                            ContractPredicate::Initialized,
                            ContractPredicate::ValidForType,
                        ]),
                        FailurePhase::ImmediateValidity,
                        DischargeModel::ExactInitialization,
                    ),
                    KnownContractModel::FromUtf8Unchecked
                    | KnownContractModel::StringFromUtf8Unchecked => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([ContractPredicate::ValidUtf8]),
                        FailurePhase::ImmediateValidity,
                        DischargeModel::ExactValidation,
                    ),
                    KnownContractModel::Transmute => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([ContractPredicate::ValidForType]),
                        FailurePhase::ImmediateValidity,
                        DischargeModel::None,
                    ),
                    KnownContractModel::ConditionTrue => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([ContractPredicate::ConditionTrue]),
                        FailurePhase::Precondition,
                        DischargeModel::ExactRelation,
                    ),
                };
                return ContractDescriptor {
                    precision: ContractPrecision::Precise,
                    model_id: Some(model.as_str().to_string()),
                    relevant_roles: selected,
                    predicates,
                    failure_phase,
                    discharge,
                    static_implication: if model == KnownContractModel::Transmute {
                        transmute_validity_implication(site)
                    } else {
                        StaticContractImplication::NotApplicable
                    },
                };
            }
        }
        let generic = match site.seed.kind {
            UnsafeSiteKind::RawPointerDereference => Some((
                "raw_pointer_access_v1",
                BTreeSet::from([
                    ContractPredicate::ValidForRead,
                    ContractPredicate::Aligned,
                    ContractPredicate::Initialized,
                ]),
                FailurePhase::Precondition,
            )),
            UnsafeSiteKind::RestrictedValidityInitialization
            | UnsafeSiteKind::UnsafeFieldInitialization
            | UnsafeSiteKind::UnionFieldAccess
            | UnsafeSiteKind::UnsafeBinderCast => Some((
                "language_validity_v1",
                BTreeSet::from([ContractPredicate::ValidForType]),
                FailurePhase::ImmediateValidity,
            )),
            UnsafeSiteKind::UnsafeFieldUse => Some((
                "unsafe_field_validity_v1",
                BTreeSet::from([ContractPredicate::ValidForType]),
                FailurePhase::Precondition,
            )),
            UnsafeSiteKind::LayoutConstrainedMutation | UnsafeSiteKind::LayoutConstrainedBorrow => {
                Some((
                    "layout_constrained_access_v1",
                    BTreeSet::from([
                        ContractPredicate::Aligned,
                        ContractPredicate::LayoutCompatible,
                    ]),
                    FailurePhase::Precondition,
                ))
            }
            UnsafeSiteKind::MutableStaticAccess | UnsafeSiteKind::ExternStaticAccess => Some((
                "static_memory_access_v1",
                BTreeSet::from([ContractPredicate::ValidForRead]),
                FailurePhase::Precondition,
            )),
            UnsafeSiteKind::UnsafeCall
            | UnsafeSiteKind::InlineAssembly
            | UnsafeSiteKind::TargetFeatureCall
            | UnsafeSiteKind::ExplicitDropCall => None,
        };
        let Some((model_id, predicates, failure_phase)) = generic else {
            return opaque();
        };
        ContractDescriptor {
            precision: ContractPrecision::Generic,
            model_id: Some(model_id.to_string()),
            relevant_roles: roles,
            predicates,
            failure_phase,
            discharge: DischargeModel::None,
            static_implication: StaticContractImplication::NotApplicable,
        }
    }
}

fn transmute_validity_implication(site: &BoundUnsafeSite) -> StaticContractImplication {
    let source = site.type_validity.operand(&OperandRole::Argument(0));
    let target = site.type_validity.result;

    if matches!(source, TypeValidityClass::UninitializedOrPadding) {
        return StaticContractImplication::Unproven(
            StaticContractGap::SourceMayContainUninitializedBytes,
        );
    }
    if matches!(target, TypeValidityClass::ReferenceOrFunctionPointer) {
        return StaticContractImplication::Unproven(
            StaticContractGap::LifetimeOrProvenanceSensitive,
        );
    }
    if matches!(
        target,
        TypeValidityClass::RestrictedScalar
            | TypeValidityClass::RestrictedStructural
            | TypeValidityClass::RestrictedEnum
    ) {
        return StaticContractImplication::Unproven(StaticContractGap::TargetRestrictedValidity);
    }
    if source == TypeValidityClass::Unknown || target == TypeValidityClass::Unknown {
        return StaticContractImplication::Unproven(StaticContractGap::UnknownTypeValidity);
    }
    if !source.guarantees_initialized_bytes() {
        return StaticContractImplication::Unproven(
            StaticContractGap::SourceMayContainUninitializedBytes,
        );
    }
    match target {
        TypeValidityClass::AllBitsValidScalar => {
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidScalar)
        }
        TypeValidityClass::AllBitsValidBytes => {
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidByteForm)
        }
        TypeValidityClass::AllBitsValidRawPointer => {
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidRawPointer)
        }
        // rustc has already established equal transmute sizes. Once every
        // source byte is initialized, an all-bits-valid structural target
        // admits every such bit pattern; the source's structural shape is
        // irrelevant to the target validity obligation.
        TypeValidityClass::AllBitsValidStructural => StaticContractImplication::Proven(
            if source == TypeValidityClass::AllBitsValidStructural {
                StaticContractEvidence::SameAllBitsValidStructure
            } else {
                StaticContractEvidence::TargetAllBitsValidStructural
            },
        ),
        TypeValidityClass::UninitializedOrPadding => StaticContractImplication::Unproven(
            StaticContractGap::SourceMayContainUninitializedBytes,
        ),
        TypeValidityClass::ReferenceOrFunctionPointer => {
            StaticContractImplication::Unproven(StaticContractGap::LifetimeOrProvenanceSensitive)
        }
        TypeValidityClass::RestrictedScalar
        | TypeValidityClass::RestrictedStructural
        | TypeValidityClass::RestrictedEnum => {
            StaticContractImplication::Unproven(StaticContractGap::TargetRestrictedValidity)
        }
        TypeValidityClass::Unknown => {
            StaticContractImplication::Unproven(StaticContractGap::UnknownTypeValidity)
        }
    }
}

fn select_pointer_and_extent_roles(
    site: &BoundUnsafeSite,
    pointer_index: u32,
    extent_index: u32,
) -> BTreeSet<OperandRole> {
    let mut roles = select_roles(site, &[OperandRole::Argument(pointer_index)]);
    roles.extend(select_roles(
        site,
        &[
            OperandRole::Argument(extent_index),
            OperandRole::Index,
            OperandRole::Length,
            OperandRole::Size,
        ],
    ));
    roles
}

fn select_pointer_len_cap_roles(site: &BoundUnsafeSite) -> BTreeSet<OperandRole> {
    let mut roles = select_roles(site, &[OperandRole::Argument(0)]);
    roles.extend(select_roles(
        site,
        &[
            OperandRole::Argument(1),
            OperandRole::Argument(2),
            OperandRole::Index,
            OperandRole::Length,
            OperandRole::Size,
        ],
    ));
    roles
}

fn select_ptr_copy_roles(site: &BoundUnsafeSite) -> BTreeSet<OperandRole> {
    let mut roles = select_roles(site, &[OperandRole::Argument(0), OperandRole::Argument(1)]);
    roles.extend(select_roles(
        site,
        &[
            OperandRole::Argument(2),
            OperandRole::Length,
            OperandRole::Size,
        ],
    ));
    roles
}

fn select_roles(site: &BoundUnsafeSite, expected: &[OperandRole]) -> BTreeSet<OperandRole> {
    site.operands
        .iter()
        .filter(|operand| expected.contains(&operand.role))
        .map(|operand| operand.role.clone())
        .collect()
}

fn select_get_unchecked_subject_role(site: &BoundUnsafeSite) -> BTreeSet<OperandRole> {
    // Bounds are a relation between the exact collection and index.  Keeping
    // both roles also exposes type-level collection capabilities (for example,
    // whether an associated slice can be empty) to the P5 classifier.  The
    // exact validation binding still has to prove the complete relation before
    // this contract can be discharged.
    select_roles(site, &[OperandRole::Argument(0), OperandRole::Argument(1)])
}

pub(super) fn structural_candidate(site: &BoundUnsafeSite) -> Option<StructuralSiteCandidate> {
    let relevant_roles = site
        .operands
        .iter()
        .filter(|operand| {
            operand.role != OperandRole::Callee
                && (is_memory_or_validity_type(&operand.type_fingerprint)
                    || is_structural_memory_role(&operand.role))
        })
        .map(|operand| operand.role.clone())
        .collect::<BTreeSet<_>>();
    let validity_sensitive_result = site
        .seed
        .result
        .as_ref()
        .is_some_and(|result| is_memory_or_validity_type(&result.type_fingerprint));
    let language_site_is_structural = !matches!(
        site.seed.kind,
        UnsafeSiteKind::UnsafeCall
            | UnsafeSiteKind::InlineAssembly
            | UnsafeSiteKind::TargetFeatureCall
            | UnsafeSiteKind::ExplicitDropCall
    );
    (!relevant_roles.is_empty() || validity_sensitive_result || language_site_is_structural).then(
        || StructuralSiteCandidate {
            site_id: site.seed.site_id(),
            relevant_roles,
            validity_sensitive_result,
        },
    )
}

pub(super) fn structural_candidate_with_contract(
    site: &BoundUnsafeSite,
    contract: &ContractDescriptor,
) -> Option<StructuralSiteCandidate> {
    structural_candidate(site).or_else(|| {
        (site.result_place.is_some()
            && contract.failure_phase == FailurePhase::ImmediateValidity
            && contract
                .predicates
                .contains(&ContractPredicate::ValidForType))
        .then(|| StructuralSiteCandidate {
            site_id: site.seed.site_id(),
            relevant_roles: contract.relevant_roles.clone(),
            validity_sensitive_result: true,
        })
    })
}

fn is_structural_memory_role(role: &OperandRole) -> bool {
    matches!(
        role,
        OperandRole::Address
            | OperandRole::Index
            | OperandRole::Length
            | OperandRole::Size
            | OperandRole::Alignment
            | OperandRole::Layout
            | OperandRole::Discriminant
            | OperandRole::Out
    )
}

fn is_memory_or_validity_type(ty: &str) -> bool {
    let compact = ty.replace(' ', "");
    compact.starts_with('&')
        || compact.starts_with("*const")
        || compact.starts_with("*mut")
        || compact.starts_with('[')
        || compact.contains("str")
        || compact.contains("MaybeUninit<")
        || compact.contains("NonNull<")
        || compact.contains("NonZero")
        || matches!(compact.as_str(), "bool" | "char")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::unsoundaudit::summary::{
        FunctionKey, PlaceKey, StablePosition, StableSpan,
    };
    use crate::analysis::unsoundaudit::unsafe_site::{
        binding::{MirEventLocation, MirOperandAnchor, MirStage, MirTypeValidity},
        RawSpanAnchor, ResultAnchor, UnsafeSiteSeed,
    };

    fn span() -> StableSpan {
        StableSpan::new(
            "src/lib.rs",
            StablePosition::new(1, 1),
            StablePosition::new(1, 2),
        )
    }

    fn site(
        kind: UnsafeSiteKind,
        operands: &[(OperandRole, &str)],
        result: Option<&str>,
    ) -> BoundUnsafeSite {
        let location = MirEventLocation {
            stage: MirStage::Solver,
            block: 0,
            statement: 0,
            event: 0,
        };
        BoundUnsafeSite {
            seed: UnsafeSiteSeed {
                adapter_id: "test".into(),
                owner: FunctionKey::new("crate::f"),
                owner_def_path_hash: "owner".into(),
                ordinal: 1,
                kind,
                raw_span: RawSpanAnchor::new(span(), 0, 1, "#0"),
                hir_context: "HirId(1)".into(),
                callee: None,
                operands: Vec::new(),
                result: result.map(|ty| ResultAnchor {
                    span: RawSpanAnchor::new(span(), 0, 1, "#0"),
                    type_fingerprint: ty.into(),
                    shape: "call".into(),
                }),
            },
            built: MirEventLocation {
                stage: MirStage::Built,
                ..location.clone()
            },
            solver: location,
            operands: operands
                .iter()
                .enumerate()
                .map(|(index, (role, ty))| MirOperandAnchor {
                    role: role.clone(),
                    type_fingerprint: (*ty).into(),
                    place: Some(PlaceKey::new(format!("_{index}"))),
                })
                .collect(),
            result_place: result.map(|_| PlaceKey::new("_9")),
            type_validity: MirTypeValidity::default(),
        }
    }

    fn transmute_site(source: TypeValidityClass, target: TypeValidityClass) -> BoundUnsafeSite {
        let mut site = site(
            UnsafeSiteKind::UnsafeCall,
            &[(OperandRole::Argument(0), "Source")],
            Some("Target"),
        );
        site.type_validity = MirTypeValidity {
            operands: vec![(OperandRole::Argument(0), source)],
            result: target,
        };
        site
    }

    #[test]
    fn catalog_refines_metadata_without_changing_structural_candidates() {
        let site = site(
            UnsafeSiteKind::UnsafeCall,
            &[
                (OperandRole::Callee, "unsafe fn(&[u8], usize)"),
                (OperandRole::Argument(0), "&[u8]"),
                (OperandRole::Argument(1), "usize"),
            ],
            Some("u8"),
        );
        let candidate = structural_candidate(&site).unwrap();
        let opaque =
            ContractCatalog::empty().describe(&site, Some(KnownContractModel::GetUnchecked));
        let precise =
            ContractCatalog::full().describe(&site, Some(KnownContractModel::GetUnchecked));
        assert_eq!(candidate.site_id, site.seed.site_id());
        assert_eq!(opaque.precision, ContractPrecision::Opaque);
        assert_eq!(precise.precision, ContractPrecision::Precise);
        assert_eq!(precise.model_id.as_deref(), Some("slice_get_unchecked_v1"));
        assert_eq!(
            precise.relevant_roles,
            BTreeSet::from([OperandRole::Argument(0), OperandRole::Argument(1)])
        );
        assert_eq!(
            precise.predicates,
            BTreeSet::from([ContractPredicate::InBounds])
        );
        assert_eq!(structural_candidate(&site), Some(candidate));
    }

    #[test]
    fn get_unchecked_keeps_both_sides_of_the_bounds_relation() {
        let mut site = site(
            UnsafeSiteKind::UnsafeCall,
            &[
                (OperandRole::Argument(0), "&[u8]"),
                (OperandRole::Argument(1), "usize"),
            ],
            Some("u8"),
        );
        let variable =
            ContractCatalog::full().describe(&site, Some(KnownContractModel::GetUnchecked));
        assert_eq!(
            variable.relevant_roles,
            BTreeSet::from([OperandRole::Argument(0), OperandRole::Argument(1)])
        );
        site.operands
            .iter_mut()
            .find(|operand| operand.role == OperandRole::Argument(1))
            .unwrap()
            .place = None;
        let literal =
            ContractCatalog::full().describe(&site, Some(KnownContractModel::GetUnchecked));
        assert_eq!(
            literal.relevant_roles,
            BTreeSet::from([OperandRole::Argument(0), OperandRole::Argument(1)])
        );
    }

    #[test]
    fn generic_language_sites_do_not_depend_on_an_api_registry() {
        let site = site(
            UnsafeSiteKind::RawPointerDereference,
            &[(OperandRole::Address, "*const u32")],
            Some("u32"),
        );
        let candidate = structural_candidate(&site).unwrap();
        let full = ContractCatalog::full().describe(&site, None);
        assert_eq!(
            candidate.relevant_roles,
            BTreeSet::from([OperandRole::Address])
        );
        assert_eq!(full.precision, ContractPrecision::Generic);
        assert_eq!(
            full.predicates,
            BTreeSet::from([
                ContractPredicate::ValidForRead,
                ContractPredicate::Aligned,
                ContractPredicate::Initialized,
            ])
        );
    }

    #[test]
    fn opaque_calls_track_memory_roles_but_leave_unrelated_scalars_census_only() {
        let memory = site(
            UnsafeSiteKind::UnsafeCall,
            &[
                (OperandRole::Callee, "unsafe fn(*mut u8, usize)"),
                (OperandRole::Argument(0), "*mut u8"),
                (OperandRole::Argument(1), "usize"),
            ],
            None,
        );
        assert_eq!(
            structural_candidate(&memory).unwrap().relevant_roles,
            BTreeSet::from([OperandRole::Argument(0)])
        );
        let scalar = site(
            UnsafeSiteKind::UnsafeCall,
            &[
                (OperandRole::Callee, "unsafe fn(u32)"),
                (OperandRole::Argument(0), "u32"),
            ],
            Some("u32"),
        );
        assert!(structural_candidate(&scalar).is_none());
        assert_eq!(
            ContractCatalog::full().describe(&memory, None).precision,
            ContractPrecision::Opaque
        );
    }

    #[test]
    fn validity_contract_keeps_a_user_enum_result_in_the_structural_lane() {
        let site = site(
            UnsafeSiteKind::UnsafeCall,
            &[(OperandRole::Argument(0), "u8")],
            Some("crate::Tiny"),
        );
        assert!(structural_candidate(&site).is_none());
        let contract = ContractCatalog::full().describe(&site, Some(KnownContractModel::Transmute));
        let candidate = structural_candidate_with_contract(&site, &contract).unwrap();
        assert!(candidate.validity_sensitive_result);
        assert_eq!(candidate.site_id, site.seed.site_id());
    }

    #[test]
    fn exact_models_have_closed_failure_and_discharge_semantics() {
        let site = site(
            UnsafeSiteKind::UnsafeCall,
            &[(OperandRole::Argument(0), "*const u8")],
            Some("u8"),
        );
        let cases = [
            (
                KnownContractModel::PtrRead,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::ReadUnaligned,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::NonNullNewUnchecked,
                FailurePhase::Precondition,
                DischargeModel::ExactValidation,
            ),
            (
                KnownContractModel::AssumeInit,
                FailurePhase::ImmediateValidity,
                DischargeModel::ExactInitialization,
            ),
            (
                KnownContractModel::FromUtf8Unchecked,
                FailurePhase::ImmediateValidity,
                DischargeModel::ExactValidation,
            ),
            (
                KnownContractModel::StringFromUtf8Unchecked,
                FailurePhase::ImmediateValidity,
                DischargeModel::ExactValidation,
            ),
            (
                KnownContractModel::VecGetUnchecked,
                FailurePhase::Precondition,
                DischargeModel::ExactValidation,
            ),
            (
                KnownContractModel::SliceFromRawParts,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::CStrFromPtr,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::CStringFromRaw,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::BoxFromRaw,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::VecFromRawParts,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::PtrOffset,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::PtrCopy,
                FailurePhase::Precondition,
                DischargeModel::None,
            ),
            (
                KnownContractModel::Transmute,
                FailurePhase::ImmediateValidity,
                DischargeModel::None,
            ),
            (
                KnownContractModel::ConditionTrue,
                FailurePhase::Precondition,
                DischargeModel::ExactRelation,
            ),
        ];
        for (model, phase, discharge) in cases {
            let descriptor = ContractCatalog::full().describe(&site, Some(model));
            assert_eq!(descriptor.precision, ContractPrecision::Precise);
            assert_eq!(descriptor.failure_phase, phase);
            assert_eq!(descriptor.discharge, discharge);
        }
    }

    #[test]
    fn condition_true_selects_only_the_boolean_argument() {
        let site = site(
            UnsafeSiteKind::UnsafeCall,
            &[
                (OperandRole::Callee, "unsafe fn(bool)"),
                (OperandRole::Argument(0), "bool"),
            ],
            Some("()"),
        );
        let contract =
            ContractCatalog::full().describe(&site, Some(KnownContractModel::ConditionTrue));
        assert_eq!(contract.precision, ContractPrecision::Precise);
        assert_eq!(contract.model_id.as_deref(), Some("condition_true_v1"));
        assert_eq!(
            contract.relevant_roles,
            BTreeSet::from([OperandRole::Argument(0)])
        );
        assert_eq!(
            contract.predicates,
            BTreeSet::from([ContractPredicate::ConditionTrue])
        );
        assert_eq!(contract.failure_phase, FailurePhase::Precondition);
        assert_eq!(contract.discharge, DischargeModel::ExactRelation);
    }

    #[test]
    fn pointer_reads_model_alignment_only_when_the_operation_requires_it() {
        let site = site(
            UnsafeSiteKind::UnsafeCall,
            &[(OperandRole::Argument(0), "*const T")],
            Some("T"),
        );
        let aligned = ContractCatalog::full().describe(&site, Some(KnownContractModel::PtrRead));
        assert_eq!(
            aligned.predicates,
            BTreeSet::from([
                ContractPredicate::ValidForRead,
                ContractPredicate::RangeInBounds,
                ContractPredicate::NonNull,
                ContractPredicate::Aligned,
                ContractPredicate::Initialized,
                ContractPredicate::ValidForType,
            ])
        );
        let unaligned =
            ContractCatalog::full().describe(&site, Some(KnownContractModel::ReadUnaligned));
        assert_eq!(
            unaligned.predicates,
            BTreeSet::from([
                ContractPredicate::ValidForRead,
                ContractPredicate::RangeInBounds,
                ContractPredicate::NonNull,
                ContractPredicate::Initialized,
                ContractPredicate::ValidForType,
            ])
        );
        assert!(!unaligned.predicates.contains(&ContractPredicate::Aligned));
    }

    #[test]
    fn transmute_proves_only_initialized_sources_into_all_bits_valid_targets() {
        let scalar = transmute_site(
            TypeValidityClass::RestrictedScalar,
            TypeValidityClass::AllBitsValidScalar,
        );
        let scalar_contract =
            ContractCatalog::full().describe(&scalar, Some(KnownContractModel::Transmute));
        assert_eq!(
            scalar_contract.static_implication,
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidScalar)
        );

        let bytes = transmute_site(
            TypeValidityClass::AllBitsValidStructural,
            TypeValidityClass::AllBitsValidBytes,
        );
        let bytes_contract =
            ContractCatalog::full().describe(&bytes, Some(KnownContractModel::Transmute));
        assert_eq!(
            bytes_contract.static_implication,
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidByteForm)
        );

        let structural = transmute_site(
            TypeValidityClass::AllBitsValidStructural,
            TypeValidityClass::AllBitsValidStructural,
        );
        let structural_contract =
            ContractCatalog::full().describe(&structural, Some(KnownContractModel::Transmute));
        assert_eq!(
            structural_contract.static_implication,
            StaticContractImplication::Proven(StaticContractEvidence::SameAllBitsValidStructure)
        );

        let bytes_to_structural = transmute_site(
            TypeValidityClass::AllBitsValidBytes,
            TypeValidityClass::AllBitsValidStructural,
        );
        let bytes_to_structural = ContractCatalog::full()
            .describe(&bytes_to_structural, Some(KnownContractModel::Transmute));
        assert_eq!(
            bytes_to_structural.static_implication,
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidStructural)
        );
    }

    #[test]
    fn transmute_does_not_prove_uninitialized_padding_or_unknown_sources() {
        for (source, expected_gap) in [
            (
                TypeValidityClass::UninitializedOrPadding,
                StaticContractGap::SourceMayContainUninitializedBytes,
            ),
            (
                TypeValidityClass::Unknown,
                StaticContractGap::UnknownTypeValidity,
            ),
        ] {
            let site = transmute_site(source, TypeValidityClass::AllBitsValidScalar);
            let contract =
                ContractCatalog::full().describe(&site, Some(KnownContractModel::Transmute));
            assert_eq!(
                contract.static_implication,
                StaticContractImplication::Unproven(expected_gap)
            );
        }
    }

    #[test]
    fn transmute_keeps_restricted_and_reference_targets_unproven() {
        for (target, expected_gap) in [
            (
                TypeValidityClass::RestrictedScalar,
                StaticContractGap::TargetRestrictedValidity,
            ),
            (
                TypeValidityClass::RestrictedStructural,
                StaticContractGap::TargetRestrictedValidity,
            ),
            (
                TypeValidityClass::RestrictedEnum,
                StaticContractGap::TargetRestrictedValidity,
            ),
            (
                TypeValidityClass::ReferenceOrFunctionPointer,
                StaticContractGap::LifetimeOrProvenanceSensitive,
            ),
        ] {
            let site = transmute_site(TypeValidityClass::AllBitsValidScalar, target);
            let contract =
                ContractCatalog::full().describe(&site, Some(KnownContractModel::Transmute));
            assert_eq!(
                contract.static_implication,
                StaticContractImplication::Unproven(expected_gap)
            );
        }
    }

    #[test]
    fn transmute_raw_pointer_and_integer_runtime_bit_validity_is_implied() {
        let pointer_to_integer = transmute_site(
            TypeValidityClass::AllBitsValidRawPointer,
            TypeValidityClass::AllBitsValidScalar,
        );
        let pointer_to_integer = ContractCatalog::full()
            .describe(&pointer_to_integer, Some(KnownContractModel::Transmute));
        assert_eq!(
            pointer_to_integer.static_implication,
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidScalar)
        );

        let integer_to_pointer = transmute_site(
            TypeValidityClass::AllBitsValidScalar,
            TypeValidityClass::AllBitsValidRawPointer,
        );
        let integer_to_pointer = ContractCatalog::full()
            .describe(&integer_to_pointer, Some(KnownContractModel::Transmute));
        assert_eq!(
            integer_to_pointer.static_implication,
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidRawPointer)
        );
    }

    #[test]
    fn transmute_reference_and_function_pointer_validity_is_directional() {
        let reference_to_integer = transmute_site(
            TypeValidityClass::ReferenceOrFunctionPointer,
            TypeValidityClass::AllBitsValidScalar,
        );
        let reference_to_integer = ContractCatalog::full()
            .describe(&reference_to_integer, Some(KnownContractModel::Transmute));
        assert_eq!(
            reference_to_integer.static_implication,
            StaticContractImplication::Proven(StaticContractEvidence::TargetAllBitsValidScalar)
        );

        let integer_to_reference = transmute_site(
            TypeValidityClass::AllBitsValidScalar,
            TypeValidityClass::ReferenceOrFunctionPointer,
        );
        let integer_to_reference = ContractCatalog::full()
            .describe(&integer_to_reference, Some(KnownContractModel::Transmute));
        assert_eq!(
            integer_to_reference.static_implication,
            StaticContractImplication::Unproven(StaticContractGap::LifetimeOrProvenanceSensitive)
        );
    }

    #[test]
    fn empty_catalog_cannot_claim_a_static_transmute_proof() {
        let site = transmute_site(
            TypeValidityClass::AllBitsValidScalar,
            TypeValidityClass::AllBitsValidScalar,
        );
        let contract =
            ContractCatalog::empty().describe(&site, Some(KnownContractModel::Transmute));
        assert_eq!(
            contract.static_implication,
            StaticContractImplication::NotApplicable
        );
    }
}
