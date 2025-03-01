use rustc_hir::{
    intravisit::{self, Visitor, NestedVisitorMap},
    BlockCheckMode, Body, HirId, Item, 
    ItemKind, Safety, def_id::LocalDefId
};
use rustc_middle::ty::TyCtxt;
use std::collections::{HashSet, HashMap};

/// Visitor to find unsafe blocks within safe functions
pub struct UnsafeBlockVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    unsafe_blocks: HashSet<HirId>,
}

impl<'tcx> UnsafeBlockVisitor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            unsafe_blocks: HashSet::new(),
        }
    }

    pub fn visit_crate(&mut self) {
        self.tcx.hir().visit_all_item_likes_in_crate(self);
    }

    pub fn get_unsafe_blocks(&self) -> &HashSet<HirId> {
        &self.unsafe_blocks
    }
}

impl<'tcx> Visitor<'tcx> for UnsafeBlockVisitor<'tcx> {
    type NestedFilter = rustc_middle::hir::nested_filter::All;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.nested_filter())
    }

    fn visit_block(&mut self, block: &'tcx rustc_hir::Block<'tcx>) {
        if matches!(block.rules, BlockCheckMode::UnsafeBlock(_)) {
            self.unsafe_blocks.insert(block.hir_id);
        }
        intravisit::walk_block(self, block);
    }

    fn visit_body(&mut self, body: &Body<'_>) {
        intravisit::walk_body(self, body);
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        match &item.kind {
            ItemKind::Fn(sig, _, body_id) => {
                // We're only interested in safe functions with unsafe blocks
                if sig.header.safety == Safety::Safe {
                    self.visit_body(self.tcx.hir().body(*body_id));
                }
            }
            _ => intravisit::walk_item(self, item),
        }
    }
}

/// Visitor to build the call graph
pub struct CallGraphVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    call_graph: HashMap<LocalDefId, HashSet<LocalDefId>>,
}

impl<'tcx> CallGraphVisitor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            call_graph: HashMap::new(),
        }
    }

    pub fn visit_crate(&mut self) {
        self.tcx.hir().visit_all_item_likes_in_crate(self);
    }

    pub fn get_call_graph(&self) -> &HashMap<LocalDefId, HashSet<LocalDefId>> {
        &self.call_graph
    }
}

impl<'tcx> Visitor<'tcx> for CallGraphVisitor<'tcx> {
    type NestedFilter = rustc_middle::hir::nested_filter::All;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.nested_filter())
    }

    fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'tcx>) {
        // Record function calls
        // This is a simplified approach - for a complete implementation,
        // you'd need to handle different kinds of calls (method calls, direct calls, etc.)
        // and resolve the actual functions being called.
        if let rustc_hir::ExprKind::Call(_func, _) = &expr.kind {
            // Handle the call - this is simplified
            // In a real implementation, you'd resolve the function being called
            // and record it in the call graph
        }

        intravisit::walk_expr(self, expr);
    }
}