use super::contract_catalog::KnownContractModel;
use super::summary::FunctionKey;
use super::unsafe_site::UnsafeSiteKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum WarningPredicate {
    NonNull,
    InBounds,
}

impl WarningPredicate {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::NonNull => "nonnull",
            Self::InBounds => "in_bounds",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SinkObligation {
    pub predicate: WarningPredicate,
    pub sink_model: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct MissingSanitizerWarning {
    pub public_root: FunctionKey,
    pub site_id: String,
    pub predicate: WarningPredicate,
    pub sink_model: String,
}

pub(super) fn classify_sink(
    kind: UnsafeSiteKind,
    model: Option<KnownContractModel>,
) -> Option<SinkObligation> {
    match model {
        Some(KnownContractModel::GetUnchecked | KnownContractModel::VecGetUnchecked) => {
            Some(SinkObligation {
                predicate: WarningPredicate::InBounds,
                sink_model: model.expect("matched get_unchecked model").as_str(),
            })
        }
        Some(
            KnownContractModel::PtrRead
            | KnownContractModel::ReadUnaligned
            | KnownContractModel::PtrOffset
            | KnownContractModel::PtrCopy
            | KnownContractModel::SliceFromRawParts
            | KnownContractModel::CStrFromPtr,
        ) => Some(SinkObligation {
            predicate: WarningPredicate::NonNull,
            sink_model: model.expect("matched pointer model").as_str(),
        }),
        Some(
            KnownContractModel::NonNullNewUnchecked
            | KnownContractModel::AssumeInit
            | KnownContractModel::FromUtf8Unchecked
            | KnownContractModel::StringFromUtf8Unchecked
            | KnownContractModel::CStringFromRaw
            | KnownContractModel::BoxFromRaw
            | KnownContractModel::VecFromRawParts
            | KnownContractModel::Transmute
            | KnownContractModel::ConditionTrue,
        ) => None,
        None => (kind == UnsafeSiteKind::RawPointerDereference).then_some(SinkObligation {
            predicate: WarningPredicate::NonNull,
            sink_model: "raw_pointer_dereference",
        }),
    }
}

#[cfg(test)]
pub(super) fn should_emit_warning(
    has_active_pattern_finding: bool,
    sanitizer_dominates: bool,
) -> bool {
    !has_active_pattern_finding && !sanitizer_dominates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_unchecked_is_index_only() {
        let obligation = classify_sink(
            UnsafeSiteKind::UnsafeCall,
            Some(KnownContractModel::GetUnchecked),
        )
        .expect("get_unchecked is in v1");
        assert_eq!(obligation.predicate, WarningPredicate::InBounds);
        assert_eq!(obligation.sink_model, "slice_get_unchecked_v1");
        assert!(classify_sink(
            UnsafeSiteKind::UnsafeCall,
            Some(KnownContractModel::VecGetUnchecked),
        )
        .is_some());
    }

    #[test]
    fn pointer_models_are_nonnull_and_ignore_index_roles() {
        for model in [
            KnownContractModel::PtrRead,
            KnownContractModel::ReadUnaligned,
            KnownContractModel::PtrOffset,
            KnownContractModel::PtrCopy,
            KnownContractModel::SliceFromRawParts,
            KnownContractModel::CStrFromPtr,
        ] {
            let obligation = classify_sink(UnsafeSiteKind::UnsafeCall, Some(model))
                .unwrap_or_else(|| panic!("{model:?} should be a pointer sink"));
            assert_eq!(obligation.predicate, WarningPredicate::NonNull);
            assert_eq!(obligation.sink_model, model.as_str());
        }
    }

    #[test]
    fn language_deref_is_pointer_nonnull() {
        let obligation =
            classify_sink(UnsafeSiteKind::RawPointerDereference, None).expect("raw deref is in v1");
        assert_eq!(obligation.predicate, WarningPredicate::NonNull);
        assert_eq!(obligation.sink_model, "raw_pointer_dereference");
    }

    #[test]
    fn unrelated_contracts_and_kinds_are_out_of_v1() {
        assert!(classify_sink(
            UnsafeSiteKind::UnsafeCall,
            Some(KnownContractModel::Transmute),
        )
        .is_none());
        assert!(classify_sink(
            UnsafeSiteKind::UnsafeCall,
            Some(KnownContractModel::FromUtf8Unchecked),
        )
        .is_none());
        assert!(classify_sink(UnsafeSiteKind::UnionFieldAccess, None).is_none());
        assert!(classify_sink(UnsafeSiteKind::UnsafeCall, None).is_none());
    }

    #[test]
    fn residual_warning_loses_to_active_pattern_and_to_dominating_check() {
        assert!(should_emit_warning(false, false));
        assert!(!should_emit_warning(true, false));
        assert!(!should_emit_warning(false, true));
        assert!(!should_emit_warning(true, true));
    }
}
