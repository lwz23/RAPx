use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use std::collections::HashMap;
#[derive(Hash, Eq, PartialEq, Debug)]
pub enum LifetimeKind {
    Static,
    Bound(usize),
    Any(usize),
}

#[derive(Hash, Eq, PartialEq, Debug, Copy, Clone)]
pub struct Lifetime {
    // pub kind: LifetimeKind,
    pub id: usize,
}
