use rustc_hir::def_id::DefId;
use std::collections::HashSet;

/// Represents an internal unsafe function (safe signature with unsafe blocks)
#[derive(Clone)]
pub struct InternalUnsafe {
    /// The function's DefId
    pub def_id: DefId,
    
    /// The function's name
    pub name: String,
    
    /// Whether the function is public
    pub is_public: bool,
    
    /// Set of functions that call this internal unsafe function
    pub callers: HashSet<DefId>,
}