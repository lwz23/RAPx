//! Internal unsafe function analyzer
//! 
//! This module finds internal unsafe functions (declared safe but with unsafe blocks)
//! and traces call paths from public APIs to these internal unsafe functions.

use rustc_hir::{def_id::DefId, BodyId, Node, BlockCheckMode};
use rustc_middle::ty::TyCtxt;
use rustc_middle::hir::nested_filter;
use std::collections::{HashMap, HashSet};

use crate::rap_info;

/// Visitor to detect unsafe blocks within function bodies
pub struct ContainsUnsafe<'tcx> {
    tcx: TyCtxt<'tcx>,
    function_unsafe: bool,
    block_unsafe: bool,
}

impl<'tcx> ContainsUnsafe<'tcx> {
    pub fn contains_unsafe(tcx: TyCtxt<'tcx>, body_id: BodyId) -> (bool, bool) {
        let mut visitor = ContainsUnsafe {
            tcx,
            function_unsafe: false,
            block_unsafe: false,
        };

        let body = visitor.tcx.hir().body(body_id);
        visitor.function_unsafe = visitor.body_unsafety(&body);
        visitor.visit_body(body);

        (visitor.function_unsafe, visitor.block_unsafe)
    }

    fn body_unsafety(&self, body: &'tcx rustc_hir::Body<'tcx>) -> bool {
        let did = body.value.hir_id.owner.to_def_id();
        let sig = self.tcx.fn_sig(did);
        if let rustc_hir::Safety::Unsafe = sig.skip_binder().safety() {
            return true;
        }
        false
    }
    
    fn visit_body(&mut self, body: &rustc_hir::Body<'_>) {
        // Visit the expression to detect unsafe blocks
        self.visit_expr(&body.value);
    }
    
    fn visit_expr(&mut self, expr: &rustc_hir::Expr<'_>) {
        // Process this expression
        match expr.kind {
            rustc_hir::ExprKind::Block(ref block, _) => {
                self.visit_block(block);
            }
            _ => {
                // Recursively visit all child expressions
                rustc_hir::intravisit::walk_expr(self, expr);
            }
        }
    }
    
    fn visit_block(&mut self, block: &rustc_hir::Block<'_>) {
        if matches!(block.rules, BlockCheckMode::UnsafeBlock(_)) {
            self.block_unsafe = true;
        }
        
        // Continue walking if we haven't found an unsafe block yet
        if !self.block_unsafe {
            for stmt in block.stmts {
                match stmt.kind {
                    rustc_hir::StmtKind::Expr(ref expr) | rustc_hir::StmtKind::Semi(ref expr) => {
                        self.visit_expr(expr);
                    },
                    _ => {}
                }
            }
            
            if let Some(ref expr) = block.expr {
                self.visit_expr(expr);
            }
        }
    }
}

/// Represents an internal unsafe function
#[derive(Clone, Debug)]
struct InternalUnsafe {
    def_id: DefId,
    name: String,
    is_public: bool,
}

/// The main analysis struct that checks for internal unsafe functions
pub struct LwzCheck<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    internal_unsafe_fns: HashMap<DefId, InternalUnsafe>,
    pub_fns: HashSet<DefId>,
    call_graph: HashMap<DefId, Vec<DefId>>,
    paths: HashMap<DefId, Vec<Vec<DefId>>>,
}

impl<'tcx> LwzCheck<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            internal_unsafe_fns: HashMap::new(),
            pub_fns: HashSet::new(),
            call_graph: HashMap::new(),
            paths: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        rap_info!("Starting LWZ analysis to detect internal unsafe functions...");
        rap_info!("LWZ is enabled");
        
        self.collect_functions();
        self.build_call_graph();
        self.find_paths();
        self.report_findings();
    }

    fn collect_functions(&mut self) {
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();
            let is_fn = matches!(self.tcx.def_kind(def_id), 
                rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn);
            
            if !is_fn {
                continue;
            }
            
            // Get the function's HIR body
            if let Some(body_id) = self.tcx.hir().maybe_body_owned_by(local_def_id) {
                // Use ContainsUnsafe to check if this is an internal unsafe function
                let (function_unsafe, block_unsafe) = ContainsUnsafe::contains_unsafe(self.tcx, body_id);
                
                // Only interested in functions that are not declared unsafe but contain unsafe blocks
                if !function_unsafe && block_unsafe {
                    // This is an internal unsafe function
                    let internal_unsafe = InternalUnsafe {
                        def_id,
                        name: self.get_fn_name(def_id),
                        is_public: self.is_public_fn(def_id),
                    };
                    
                    self.internal_unsafe_fns.insert(def_id, internal_unsafe);
                    
                    // Debug output
                    rap_info!("Found internal unsafe function: {} (public: {})",
                              internal_unsafe.name, internal_unsafe.is_public);
                }
                
                // Check if function is public
                if self.is_public_fn(def_id) {
                    self.pub_fns.insert(def_id);
                }
            }
        }
        
        rap_info!("Found {} internal unsafe functions and {} public functions",
            self.internal_unsafe_fns.len(), self.pub_fns.len());
    }

    fn build_call_graph(&mut self) {
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();
            
            if !self.tcx.is_mir_available(def_id) {
                continue;
            }
            
            let callees = self.get_callees(def_id);
            if !callees.is_empty() {
                self.call_graph.insert(def_id, callees);
            }
        }
    }

    fn find_paths(&mut self) {
        for &pub_fn in &self.pub_fns {
            let mut paths = Vec::new();
            let mut visited = HashSet::new();
            let mut current_path = Vec::new();
            
            self.find_paths_dfs(pub_fn, &mut paths, &mut visited, &mut current_path);
            
            if !paths.is_empty() {
                self.paths.insert(pub_fn, paths);
            }
        }
    }

    fn find_paths_dfs(
        &self,
        current: DefId,
        paths: &mut Vec<Vec<DefId>>,
        visited: &mut HashSet<DefId>,
        current_path: &mut Vec<DefId>
    ) {
        if visited.contains(&current) {
            return;
        }
        
        visited.insert(current);
        current_path.push(current);
        
        if self.internal_unsafe_fns.contains_key(&current) {
            paths.push(current_path.clone());
        } else if let Some(callees) = self.call_graph.get(&current) {
            for &callee in callees {
                self.find_paths_dfs(callee, paths, visited, current_path);
            }
        }
        
        current_path.pop();
        visited.remove(&current);
    }

    fn report_findings(&self) {
        let mut found_paths = false;
        
        for (&pub_fn, paths) in &self.paths {
            found_paths = true;
            
            let pub_fn_name = self.get_fn_name(pub_fn);
            rap_info!("Public function '{}' calls internal unsafe functions through {} path(s):", 
                     pub_fn_name, paths.len());
            
            for (i, path) in paths.iter().enumerate() {
                let path_str = path.iter()
                    .map(|&def_id| self.get_fn_name(def_id))
                    .collect::<Vec<_>>()
                    .join(" -> ");
                
                rap_info!("  Path {}: {}", i + 1, path_str);
            }
        }
        
        if !found_paths {
            rap_info!("No paths from public functions to internal unsafe functions were found.");
        }
    }

    // Helper functions
    fn is_public_fn(&self, def_id: DefId) -> bool {
        // Check visibility
        self.tcx.visibility(def_id).is_public()
    }

    fn get_fn_name(&self, def_id: DefId) -> String {
        self.tcx.def_path_str(def_id)
    }

    fn get_callees(&self, def_id: DefId) -> Vec<DefId> {
        use rustc_middle::mir::{Operand, TerminatorKind};
        
        let mut callees = Vec::new();
        
        if !self.tcx.is_mir_available(def_id) {
            return callees;
        }
        
        let body = self.tcx.optimized_mir(def_id);
        
        // Correctly iterate through basic blocks
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { func, .. } = &terminator.kind {
                    if let Operand::Constant(constant) = func {
                        if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) = constant.const_.ty().kind() {
                            callees.push(*callee_def_id);
                        }
                    }
                }
            }
        }
        
        callees
    }
}