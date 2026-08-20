use std::collections::{BTreeMap, BTreeSet};
use std::ops::{BitOr, BitOrAssign};
use std::sync::Arc;

use super::endpoint_flow::GenericParamId;
use super::summary::FunctionKey;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct InfluenceCapabilities(u8);

impl InfluenceCapabilities {
    pub(super) const ADDRESS: Self = Self(1 << 0);
    pub(super) const INDEX_OR_LENGTH: Self = Self(1 << 1);
    pub(super) const TYPED_VALUE: Self = Self(1 << 2);
    pub(super) const VALIDITY_BITS: Self = Self(1 << 3);
    pub(super) const BOOLEAN: Self = Self(1 << 4);
    pub(super) const CALLABLE_RESULT: Self = Self(1 << 5);

    pub(super) fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub(super) fn stable_identity(self) -> String {
        [
            (Self::ADDRESS, "address"),
            (Self::INDEX_OR_LENGTH, "index_or_length"),
            (Self::TYPED_VALUE, "typed_value"),
            (Self::VALIDITY_BITS, "validity_bits"),
            (Self::BOOLEAN, "boolean"),
            (Self::CALLABLE_RESULT, "callable_result"),
        ]
        .into_iter()
        .filter_map(|(capability, name)| self.intersects(capability).then_some(name))
        .collect::<Vec<_>>()
        .join(",")
    }
}

impl BitOr for InfluenceCapabilities {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for InfluenceCapabilities {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum InfluenceOrigin {
    PublicFormal,
    PersistentPublicField,
    FfiReturn,
    FfiOut,
    FfiCallbackInput,
    OpenTraitReturn,
    PublicCallbackReturn,
    PublicFunctionPointerReturn,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum OpenBehaviorReceiver {
    Parameter(GenericParamId),
    Dynamic,
    Alias(BTreeSet<GenericParamId>),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum InfluenceController {
    FfiHost,
    PublicFormal {
        owner: FunctionKey,
        index: u32,
    },
    PersistentPublicField {
        access_owner: FunctionKey,
        field: String,
    },
    OpenTrait {
        trait_id: FunctionKey,
        receiver: OpenBehaviorReceiver,
    },
    PublicCallback,
    Internal,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum TokenPrecision {
    StrongBoundary,
    ExactDerived,
    Incomplete,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct InfluenceToken {
    pub origin: InfluenceOrigin,
    pub controller: InfluenceController,
    pub capabilities: InfluenceCapabilities,
    pub precision: TokenPrecision,
}

impl InfluenceToken {
    pub(super) fn ffi(origin: InfluenceOrigin, capabilities: InfluenceCapabilities) -> Self {
        debug_assert!(matches!(
            origin,
            InfluenceOrigin::FfiReturn
                | InfluenceOrigin::FfiOut
                | InfluenceOrigin::FfiCallbackInput
        ));
        Self {
            origin,
            controller: InfluenceController::FfiHost,
            capabilities,
            precision: TokenPrecision::StrongBoundary,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TokenId(u32);

#[derive(Default)]
pub(super) struct InfluenceArena {
    ids: BTreeMap<Arc<InfluenceToken>, TokenId>,
    values: Vec<Arc<InfluenceToken>>,
}

impl InfluenceArena {
    pub(super) fn intern(&mut self, token: InfluenceToken) -> TokenId {
        if let Some(id) = self.ids.get(&token) {
            return *id;
        }
        let id =
            TokenId(u32::try_from(self.values.len()).expect("influence token arena exceeds u32"));
        let token = Arc::new(token);
        self.values.push(token.clone());
        self.ids.insert(token, id);
        id
    }

    pub(super) fn get(&self, id: TokenId) -> &InfluenceToken {
        &self.values[id.0 as usize]
    }

    pub(super) fn len(&self) -> usize {
        self.values.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SinkRequirement {
    Address,
    IndexOrAddress,
    ValidityBits,
    TypedContent,
    Boolean,
    CallableResult,
}

impl SinkRequirement {
    pub(super) fn matches(&self, capabilities: InfluenceCapabilities) -> RequirementMatch {
        let required = match self {
            Self::Address => InfluenceCapabilities::ADDRESS,
            Self::IndexOrAddress => {
                InfluenceCapabilities::INDEX_OR_LENGTH | InfluenceCapabilities::ADDRESS
            }
            Self::ValidityBits => InfluenceCapabilities::VALIDITY_BITS,
            Self::TypedContent => InfluenceCapabilities::TYPED_VALUE,
            Self::Boolean => InfluenceCapabilities::BOOLEAN,
            Self::CallableResult => InfluenceCapabilities::CALLABLE_RESULT,
        };
        if capabilities.intersects(required) {
            RequirementMatch::Compatible
        } else {
            RequirementMatch::ProvenIncompatible
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RequirementMatch {
    Compatible,
    ProvenIncompatible,
    Unknown,
}

/// Declaration order is the stable primary-reason precedence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum UnestablishedReason {
    MissingExactContext,
    IncompleteNormalization,
    UnresolvedController,
    UnmodeledUnsafeCall,
    UnknownCapabilityMatch,
    OpaqueTransform,
    HavocedProvenance,
    ValidationStateUnknown,
}

impl UnestablishedReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::MissingExactContext => "missing_exact_context",
            Self::IncompleteNormalization => "incomplete_normalization",
            Self::UnresolvedController => "unresolved_controller",
            Self::UnmodeledUnsafeCall => "unmodeled_unsafe_call",
            Self::UnknownCapabilityMatch => "unknown_capability_match",
            Self::OpaqueTransform => "opaque_transform",
            Self::HavocedProvenance => "havoced_provenance",
            Self::ValidationStateUnknown => "validation_state_unknown",
        }
    }
}

pub(super) fn primary_unestablished_reason(
    reasons: &BTreeSet<UnestablishedReason>,
) -> Option<UnestablishedReason> {
    reasons.iter().next().copied()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn boolean_does_not_satisfy_address_requirement() {
        let token = InfluenceCapabilities::BOOLEAN | InfluenceCapabilities::TYPED_VALUE;
        assert_eq!(
            SinkRequirement::Address.matches(token),
            RequirementMatch::ProvenIncompatible,
        );
    }

    #[test]
    fn token_interner_deduplicates_complete_semantic_values() {
        let mut arena = InfluenceArena::default();
        let token = InfluenceToken::ffi(
            InfluenceOrigin::FfiReturn,
            InfluenceCapabilities::ADDRESS | InfluenceCapabilities::VALIDITY_BITS,
        );
        assert_eq!(arena.intern(token.clone()), arena.intern(token));
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn unestablished_reason_precedence_is_stable() {
        let reasons = BTreeSet::from([
            UnestablishedReason::OpaqueTransform,
            UnestablishedReason::UnresolvedController,
            UnestablishedReason::MissingExactContext,
        ]);
        assert_eq!(
            primary_unestablished_reason(&reasons),
            Some(UnestablishedReason::MissingExactContext),
        );
    }
}
