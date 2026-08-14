use super::summary::PlaceKey;
use std::collections::BTreeSet;

pub(super) const RELATIONAL_REFINEMENT_VERSION: &str = "lightweight-relations-v1";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RelationTerm {
    Place(PlaceKey),
    Length(PlaceKey),
    Capacity(PlaceKey),
    Constant(u64),
    Formal(u32),
    Return,
    Out(u32),
}

impl RelationTerm {
    pub(super) fn storage_place(&self) -> Option<&PlaceKey> {
        match self {
            Self::Place(place) | Self::Length(place) | Self::Capacity(place) => Some(place),
            Self::Constant(_) | Self::Formal(_) | Self::Return | Self::Out(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RelationKind {
    Eq(RelationTerm, RelationTerm),
    Le(RelationTerm, RelationTerm),
    Lt(RelationTerm, RelationTerm),
    AddConst {
        result: RelationTerm,
        base: RelationTerm,
        addend: u64,
    },
    Sub {
        result: RelationTerm,
        upper: RelationTerm,
        lower: RelationTerm,
    },
    SumEq {
        total: RelationTerm,
        left: RelationTerm,
        right: RelationTerm,
    },
    Within {
        base: PlaceKey,
        start: RelationTerm,
        end: RelationTerm,
    },
    InitializedWithin {
        base: PlaceKey,
        start: RelationTerm,
        end: RelationTerm,
    },
    CapacityFor {
        base: PlaceKey,
        end: RelationTerm,
    },
    SameObject(RelationTerm, RelationTerm),
}

impl RelationKind {
    pub(super) fn storage_places(&self) -> Vec<PlaceKey> {
        let mut places = BTreeSet::new();
        fn insert_term(places: &mut BTreeSet<PlaceKey>, term: &RelationTerm) {
            if let Some(place) = term.storage_place() {
                places.insert(place.clone());
            }
        }
        match self {
            Self::Eq(left, right)
            | Self::Le(left, right)
            | Self::Lt(left, right)
            | Self::SameObject(left, right) => {
                insert_term(&mut places, left);
                insert_term(&mut places, right);
            }
            Self::AddConst { result, base, .. } => {
                insert_term(&mut places, result);
                insert_term(&mut places, base);
            }
            Self::Sub {
                result,
                upper,
                lower,
            } => {
                insert_term(&mut places, result);
                insert_term(&mut places, upper);
                insert_term(&mut places, lower);
            }
            Self::SumEq { total, left, right } => {
                insert_term(&mut places, total);
                insert_term(&mut places, left);
                insert_term(&mut places, right);
            }
            Self::Within { base, start, end } | Self::InitializedWithin { base, start, end } => {
                places.insert(base.clone());
                insert_term(&mut places, start);
                insert_term(&mut places, end);
            }
            Self::CapacityFor { base, end } => {
                places.insert(base.clone());
                insert_term(&mut places, end);
            }
        }
        places.into_iter().collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RelationFact {
    pub(super) kind: RelationKind,
}

impl RelationFact {
    pub(super) fn new(kind: RelationKind) -> Self {
        Self { kind }
    }

    pub(super) fn storage_places(&self) -> Vec<PlaceKey> {
        self.kind.storage_places()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SummaryRelationTerm {
    Formal(u32),
    Return,
    Out(u32),
    LengthFormal(u32),
    CapacityFormal(u32),
    Constant(u64),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SummaryRelationKind {
    Eq(SummaryRelationTerm, SummaryRelationTerm),
    Le(SummaryRelationTerm, SummaryRelationTerm),
    Lt(SummaryRelationTerm, SummaryRelationTerm),
    AddConst {
        result: SummaryRelationTerm,
        base: SummaryRelationTerm,
        addend: u64,
    },
    Sub {
        result: SummaryRelationTerm,
        upper: SummaryRelationTerm,
        lower: SummaryRelationTerm,
    },
    SumEq {
        total: SummaryRelationTerm,
        left: SummaryRelationTerm,
        right: SummaryRelationTerm,
    },
    Within {
        base_formal: u32,
        start: SummaryRelationTerm,
        end: SummaryRelationTerm,
    },
    InitializedWithin {
        base_formal: u32,
        start: SummaryRelationTerm,
        end: SummaryRelationTerm,
    },
    CapacityFor {
        base_formal: u32,
        end: SummaryRelationTerm,
    },
    SameObject(SummaryRelationTerm, SummaryRelationTerm),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RelationalPostcondition {
    pub(super) relation: SummaryRelationKind,
}

impl RelationalPostcondition {
    pub(super) fn unconditional(relation: SummaryRelationKind) -> Self {
        Self { relation }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct RelationProof {
    relations: BTreeSet<RelationKind>,
}

impl RelationProof {
    pub(super) fn from_facts(facts: impl IntoIterator<Item = RelationFact>) -> Self {
        Self::from_kinds(facts.into_iter().map(|fact| fact.kind))
    }

    pub(super) fn from_kinds(kinds: impl IntoIterator<Item = RelationKind>) -> Self {
        let mut proof = Self {
            relations: kinds.into_iter().collect(),
        };
        proof.close();
        proof
    }

    pub(super) fn proves_le(&self, left: &RelationTerm, right: &RelationTerm) -> bool {
        left == right
            || self
                .relations
                .contains(&RelationKind::Le(left.clone(), right.clone()))
            || self
                .relations
                .contains(&RelationKind::Lt(left.clone(), right.clone()))
            || self
                .relations
                .contains(&RelationKind::Eq(left.clone(), right.clone()))
            || self
                .relations
                .contains(&RelationKind::Eq(right.clone(), left.clone()))
    }

    pub(super) fn proves_lt(&self, left: &RelationTerm, right: &RelationTerm) -> bool {
        self.relations
            .contains(&RelationKind::Lt(left.clone(), right.clone()))
    }

    pub(super) fn proves_in_bounds(&self, subject: &RelationTerm, collection: &PlaceKey) -> bool {
        self.proves_lt(subject, &RelationTerm::Length(collection.clone()))
    }

    pub(super) fn proves_within(
        &self,
        base: &PlaceKey,
        start: &RelationTerm,
        end: &RelationTerm,
    ) -> bool {
        self.relations.contains(&RelationKind::Within {
            base: base.clone(),
            start: start.clone(),
            end: end.clone(),
        }) || (self.proves_le(start, end)
            && self.proves_le(end, &RelationTerm::Length(base.clone())))
    }

    pub(super) fn proves_initialized_within(
        &self,
        base: &PlaceKey,
        start: &RelationTerm,
        end: &RelationTerm,
    ) -> bool {
        self.relations.contains(&RelationKind::InitializedWithin {
            base: base.clone(),
            start: start.clone(),
            end: end.clone(),
        })
    }

    pub(super) fn proves_capacity_for(&self, base: &PlaceKey, end: &RelationTerm) -> bool {
        self.relations.contains(&RelationKind::CapacityFor {
            base: base.clone(),
            end: end.clone(),
        }) || self.proves_le(end, &RelationTerm::Capacity(base.clone()))
    }

    pub(super) fn proves_increment_without_overflow(&self, base: &RelationTerm) -> bool {
        self.relations.iter().any(|relation| {
            matches!(relation, RelationKind::Lt(left, RelationTerm::Length(_)) if left == base)
        })
    }

    pub(super) fn facts(&self) -> impl Iterator<Item = RelationFact> + '_ {
        self.relations.iter().cloned().map(RelationFact::new)
    }

    fn close(&mut self) {
        loop {
            let before = self.relations.len();
            let snapshot = self.relations.iter().cloned().collect::<Vec<_>>();

            for relation in &snapshot {
                match relation {
                    RelationKind::Eq(left, right) | RelationKind::SameObject(left, right) => {
                        self.relations
                            .insert(RelationKind::Eq(right.clone(), left.clone()));
                        self.relations
                            .insert(RelationKind::Le(left.clone(), right.clone()));
                        self.relations
                            .insert(RelationKind::Le(right.clone(), left.clone()));
                    }
                    RelationKind::Lt(left, right) => {
                        self.relations
                            .insert(RelationKind::Le(left.clone(), right.clone()));
                    }
                    RelationKind::AddConst {
                        result,
                        base,
                        addend,
                    } if *addend == 0 => {
                        self.relations
                            .insert(RelationKind::Eq(result.clone(), base.clone()));
                    }
                    RelationKind::SumEq { total, left, right } => {
                        self.relations
                            .insert(RelationKind::Le(left.clone(), total.clone()));
                        self.relations
                            .insert(RelationKind::Le(right.clone(), total.clone()));
                    }
                    RelationKind::Sub { result, upper, .. } => {
                        self.relations
                            .insert(RelationKind::Le(result.clone(), upper.clone()));
                    }
                    RelationKind::Within { base, start, end } => {
                        self.relations
                            .insert(RelationKind::Le(start.clone(), end.clone()));
                        self.relations.insert(RelationKind::Le(
                            end.clone(),
                            RelationTerm::Length(base.clone()),
                        ));
                    }
                    RelationKind::InitializedWithin { base, start, end } => {
                        self.relations.insert(RelationKind::Within {
                            base: base.clone(),
                            start: start.clone(),
                            end: end.clone(),
                        });
                    }
                    RelationKind::CapacityFor { base, end } => {
                        self.relations.insert(RelationKind::Le(
                            end.clone(),
                            RelationTerm::Capacity(base.clone()),
                        ));
                    }
                    RelationKind::Le(_, _) | RelationKind::AddConst { .. } => {}
                }
            }

            let ordered = self
                .relations
                .iter()
                .filter_map(|relation| match relation {
                    RelationKind::Le(left, right) => Some((left.clone(), right.clone(), false)),
                    RelationKind::Lt(left, right) => Some((left.clone(), right.clone(), true)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for (left, middle, first_strict) in &ordered {
                for (candidate_middle, right, second_strict) in &ordered {
                    if middle != candidate_middle {
                        continue;
                    }
                    let relation = if *first_strict || *second_strict {
                        RelationKind::Lt(left.clone(), right.clone())
                    } else {
                        RelationKind::Le(left.clone(), right.clone())
                    };
                    self.relations.insert(relation);
                }
            }

            let additions = self
                .relations
                .iter()
                .filter_map(|relation| match relation {
                    RelationKind::AddConst {
                        result,
                        base,
                        addend: 1,
                    } => Some((result.clone(), base.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for (result, base) in additions {
                for relation in self.relations.clone() {
                    if let RelationKind::Lt(left, upper) = relation {
                        if left == base {
                            self.relations
                                .insert(RelationKind::Le(result.clone(), upper));
                        }
                    }
                }
            }

            if self.relations.len() == before {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn place(name: &str) -> RelationTerm {
        RelationTerm::Place(PlaceKey::new(name))
    }

    fn length(name: &str) -> RelationTerm {
        RelationTerm::Length(PlaceKey::new(name))
    }

    #[test]
    fn exact_base_is_required_to_prove_in_bounds() {
        let proof = RelationProof::from_kinds([RelationKind::Lt(place("i"), length("slice_a"))]);
        assert!(proof.proves_in_bounds(&place("i"), &PlaceKey::new("slice_a")));
        assert!(!proof.proves_in_bounds(&place("i"), &PlaceKey::new("slice_b")));
    }

    #[test]
    fn closure_proves_transitive_bounds_and_checked_increment() {
        let proof = RelationProof::from_kinds([
            RelationKind::Lt(place("i"), place("n")),
            RelationKind::Le(place("n"), length("slice")),
            RelationKind::AddConst {
                result: place("next"),
                base: place("i"),
                addend: 1,
            },
        ]);
        assert!(proof.proves_in_bounds(&place("i"), &PlaceKey::new("slice")));
        assert!(proof.proves_le(&place("next"), &length("slice")));
    }

    #[test]
    fn contained_and_initialized_ranges_remain_collection_specific() {
        let proof = RelationProof::from_kinds([RelationKind::InitializedWithin {
            base: PlaceKey::new("array"),
            start: place("start"),
            end: place("end"),
        }]);
        assert!(proof.proves_within(&PlaceKey::new("array"), &place("start"), &place("end"),));
        assert!(proof.proves_initialized_within(
            &PlaceKey::new("array"),
            &place("start"),
            &place("end"),
        ));
        assert!(!proof.proves_within(&PlaceKey::new("other"), &place("start"), &place("end"),));
    }

    #[test]
    fn sum_sub_and_capacity_closure_is_finite_and_idempotent() {
        let kinds = [
            RelationKind::SumEq {
                total: place("total"),
                left: place("front"),
                right: place("back"),
            },
            RelationKind::Sub {
                result: place("diff"),
                upper: RelationTerm::Capacity(PlaceKey::new("vec")),
                lower: RelationTerm::Length(PlaceKey::new("vec")),
            },
            RelationKind::CapacityFor {
                base: PlaceKey::new("vec"),
                end: place("desired"),
            },
        ];
        let first = RelationProof::from_kinds(kinds.clone());
        let second = RelationProof::from_kinds(kinds);
        assert_eq!(first, second);
        assert!(first.proves_le(&place("front"), &place("total")));
        assert!(first.proves_capacity_for(&PlaceKey::new("vec"), &place("desired")));
    }

    #[test]
    fn relation_storage_places_are_sorted_and_unique() {
        let fact = RelationFact::new(RelationKind::Within {
            base: PlaceKey::new("slice"),
            start: RelationTerm::Place(PlaceKey::new("index")),
            end: RelationTerm::Length(PlaceKey::new("slice")),
        });
        assert_eq!(
            fact.storage_places(),
            vec![PlaceKey::new("index"), PlaceKey::new("slice")],
        );
    }
}
