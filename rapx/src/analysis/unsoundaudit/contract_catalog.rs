use std::collections::BTreeSet;

use super::unsafe_site::{
    binding::BoundUnsafeSite, ContractPrecision, OperandRole, UnsafeSiteKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum KnownContractModel {
    GetUnchecked,
    ReadUnaligned,
    NonNullNewUnchecked,
    AssumeInit,
    FromUtf8Unchecked,
}

impl KnownContractModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GetUnchecked => "slice_get_unchecked_v1",
            Self::ReadUnaligned => "ptr_read_unaligned_v1",
            Self::NonNullNewUnchecked => "nonnull_new_unchecked_v1",
            Self::AssumeInit => "maybe_uninit_assume_init_v1",
            Self::FromUtf8Unchecked => "str_from_utf8_unchecked_v1",
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ContractDescriptor {
    pub precision: ContractPrecision,
    pub model_id: Option<String>,
    pub relevant_roles: BTreeSet<OperandRole>,
    pub predicates: BTreeSet<ContractPredicate>,
    pub failure_phase: FailurePhase,
    pub discharge: DischargeModel,
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
        };
        if !self.models_enabled {
            return opaque();
        }
        if site.seed.kind == UnsafeSiteKind::UnsafeCall {
            if let Some(model) = known {
                let (selected, predicates, failure_phase, discharge) = match model {
                    KnownContractModel::GetUnchecked => (
                        select_get_unchecked_subject_role(site),
                        BTreeSet::from([ContractPredicate::InBounds]),
                        FailurePhase::Precondition,
                        DischargeModel::ExactValidation,
                    ),
                    KnownContractModel::ReadUnaligned => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([
                            ContractPredicate::RangeInBounds,
                            ContractPredicate::Aligned,
                        ]),
                        FailurePhase::Precondition,
                        DischargeModel::ExactRelation,
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
                    KnownContractModel::FromUtf8Unchecked => (
                        select_roles(site, &[OperandRole::Argument(0)]),
                        BTreeSet::from([ContractPredicate::ValidUtf8]),
                        FailurePhase::ImmediateValidity,
                        DischargeModel::ExactValidation,
                    ),
                };
                return ContractDescriptor {
                    precision: ContractPrecision::Precise,
                    model_id: Some(model.as_str().to_string()),
                    relevant_roles: selected,
                    predicates,
                    failure_phase,
                    discharge,
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
        }
    }
}

fn select_roles(site: &BoundUnsafeSite, expected: &[OperandRole]) -> BTreeSet<OperandRole> {
    site.operands
        .iter()
        .filter(|operand| expected.contains(&operand.role))
        .map(|operand| operand.role.clone())
        .collect()
}

fn select_get_unchecked_subject_role(site: &BoundUnsafeSite) -> BTreeSet<OperandRole> {
    let index_is_literal = site.operands.iter().any(|operand| {
        operand.role == OperandRole::Argument(1) && operand.place.is_none()
    });
    select_roles(
        site,
        &[if index_is_literal {
            OperandRole::Argument(0)
        } else {
            OperandRole::Argument(1)
        }],
    )
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
        binding::{MirEventLocation, MirOperandAnchor, MirStage},
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
        }
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
            BTreeSet::from([OperandRole::Argument(1)])
        );
        assert_eq!(
            precise.predicates,
            BTreeSet::from([ContractPredicate::InBounds])
        );
        assert_eq!(structural_candidate(&site), Some(candidate));
    }

    #[test]
    fn get_unchecked_uses_the_index_subject_except_for_a_literal_index() {
        let mut site = site(
            UnsafeSiteKind::UnsafeCall,
            &[
                (OperandRole::Argument(0), "&[u8]"),
                (OperandRole::Argument(1), "usize"),
            ],
            Some("u8"),
        );
        let variable = ContractCatalog::full().describe(&site, Some(KnownContractModel::GetUnchecked));
        assert_eq!(
            variable.relevant_roles,
            BTreeSet::from([OperandRole::Argument(1)])
        );
        site.operands
            .iter_mut()
            .find(|operand| operand.role == OperandRole::Argument(1))
            .unwrap()
            .place = None;
        let literal = ContractCatalog::full().describe(&site, Some(KnownContractModel::GetUnchecked));
        assert_eq!(
            literal.relevant_roles,
            BTreeSet::from([OperandRole::Argument(0)])
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
    fn exact_models_have_closed_failure_and_discharge_semantics() {
        let site = site(
            UnsafeSiteKind::UnsafeCall,
            &[(OperandRole::Argument(0), "*const u8")],
            Some("u8"),
        );
        let cases = [
            (
                KnownContractModel::ReadUnaligned,
                FailurePhase::Precondition,
                DischargeModel::ExactRelation,
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
        ];
        for (model, phase, discharge) in cases {
            let descriptor = ContractCatalog::full().describe(&site, Some(model));
            assert_eq!(descriptor.precision, ContractPrecision::Precise);
            assert_eq!(descriptor.failure_phase, phase);
            assert_eq!(descriptor.discharge, discharge);
        }
    }
}
