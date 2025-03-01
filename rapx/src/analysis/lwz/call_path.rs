use rustc_hir::def_id::DefId;

/// Represents a call path from a public function to an internal unsafe function
#[derive(Clone)]
pub struct CallPath {
    /// The path from source to destination
    pub path: Vec<DefId>,
    
    /// The source (public) function
    pub source: DefId,
    
    /// The destination (internal unsafe) function
    pub destination: DefId,
}