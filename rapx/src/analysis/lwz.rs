//! Internal unsafe function analyzer
//! 
//! This module finds internal unsafe functions (declared safe but with unsafe blocks)
//! and traces call paths from public APIs to these internal unsafe functions.

use rustc_hir::{
    def_id::DefId, 
    BodyId, 
    BlockCheckMode,
    intravisit::{self, Visitor},
};
use rustc_middle::ty::TyCtxt;
use rustc_middle::hir::nested_filter;
use rustc_middle::mir::{Operand, TerminatorKind};
use std::collections::{HashMap, HashSet, VecDeque};

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
        visitor.function_unsafe = visitor.body_unsafety(body);
        intravisit::walk_body(&mut visitor, body);

        (visitor.function_unsafe, visitor.block_unsafe)
    }

    fn body_unsafety(&self, body: &rustc_hir::Body<'_>) -> bool {
        let did = body.value.hir_id.owner.to_def_id();
        let sig = self.tcx.fn_sig(did);
        if let rustc_hir::Safety::Unsafe = sig.skip_binder().safety() {
            return true;
        }
        false
    }
}

// Implement the Visitor trait for ContainsUnsafe
impl<'tcx> Visitor<'tcx> for ContainsUnsafe<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_block(&mut self, block: &'tcx rustc_hir::Block<'tcx>) {
        if let BlockCheckMode::UnsafeBlock(_) = block.rules {
            self.block_unsafe = true;
        }
        // Only continue traversing if we haven't found an unsafe block yet
        if !self.block_unsafe {
            intravisit::walk_block(self, block);
        }
    }
}

/// 表示一个具体的不安全操作
#[derive(Debug, Clone)]
struct UnsafeOperation {
    /// 操作类型（函数调用、解引用等）
    operation_type: String,
    /// 操作的详细描述（比如被调用的函数名）
    operation_detail: String,
}

/// Represents an internal unsafe function
#[derive(Clone, Debug)]
struct InternalUnsafe {
    def_id: DefId,
    name: String,
    is_public: bool,
    is_in_pub_mod: bool,
    /// 存储该函数中的不安全操作
    unsafe_operations: Vec<UnsafeOperation>,
}

/// The main analysis struct that checks for internal unsafe functions
pub struct LwzCheck<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    internal_unsafe_fns: HashMap<DefId, InternalUnsafe>,
    pub_fns_in_pub_mods: HashSet<DefId>,
    call_graph: HashMap<DefId, Vec<DefId>>,
    reverse_call_graph: HashMap<DefId, Vec<DefId>>,
    shortest_paths: HashMap<DefId, Vec<DefId>>,
}

impl<'tcx> LwzCheck<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            internal_unsafe_fns: HashMap::new(),
            pub_fns_in_pub_mods: HashSet::new(),
            call_graph: HashMap::new(),
            reverse_call_graph: HashMap::new(),
            shortest_paths: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        rap_info!("Starting LWZ analysis to detect internal unsafe functions...");
        
        self.collect_functions();
        self.build_call_graphs();
        self.find_shortest_paths();
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
            
            // Get the function's HIR body ID
            if let Some(body_id) = self.tcx.hir().maybe_body_owned_by(local_def_id) {
                // 检查函数是否声明为unsafe
                let function_unsafe = {
                    let did = body_id.value.hir_id.owner.to_def_id();
                    let sig = self.tcx.fn_sig(did);
                    matches!(sig.skip_binder().safety(), rustc_hir::Safety::Unsafe)
                };
                
                // 检查函数体内是否包含unsafe块
                let mut visitor = ContainsUnsafe {
                    tcx: self.tcx,
                    function_unsafe: false,
                    block_unsafe: false,
                };
                intravisit::walk_body(&mut visitor, body_id);
                let block_unsafe = visitor.block_unsafe;
                
                // Only interested in functions that are not declared unsafe but contain unsafe blocks
                if !function_unsafe && block_unsafe {
                    // This is an internal unsafe function
                    let is_public = self.is_public_fn(def_id);
                    let is_in_pub_mod = self.is_in_public_module(def_id);
                    
                    // 提取不安全操作
                    let unsafe_operations = self.extract_unsafe_operations(def_id);
                    
                    let internal_unsafe = InternalUnsafe {
                        def_id,
                        name: self.get_fn_name(def_id),
                        is_public,
                        is_in_pub_mod,
                        unsafe_operations,
                    };
                    
                    // Debug output
                    // rap_info!("Found internal unsafe function: {} (public: {}, in pub mod: {})",
                    //           internal_unsafe.name, internal_unsafe.is_public, internal_unsafe.is_in_pub_mod);
                    
                    self.internal_unsafe_fns.insert(def_id, internal_unsafe);
                    
                    // If this internal unsafe function is already a pub function in a pub module,
                    // add it directly to our pub_fns_in_pub_mods set
                    if is_public && is_in_pub_mod {
                        self.pub_fns_in_pub_mods.insert(def_id);
                    }
                }
                
                // Check if function is public and in a public module
                if self.is_public_fn(def_id) && self.is_in_public_module(def_id) {
                    self.pub_fns_in_pub_mods.insert(def_id);
                }
            }
        }
        
        rap_info!("Found {} internal unsafe functions and {} public functions in public modules",
            self.internal_unsafe_fns.len(), self.pub_fns_in_pub_mods.len());
    }

    fn build_call_graphs(&mut self) {
        // Build forward call graph
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();
            
            if !self.tcx.is_mir_available(def_id) {
                continue;
            }
            
            let callees = self.get_callees(def_id);
            if !callees.is_empty() {
                self.call_graph.insert(def_id, callees.clone());
                
                // Build reverse call graph at the same time
                for callee in callees {
                    self.reverse_call_graph
                        .entry(callee)
                        .or_insert_with(Vec::new)
                        .push(def_id);
                }
            }
        }
    }

    fn find_shortest_paths(&mut self) {
        // For each internal unsafe function, find the shortest path from a pub function in a pub module
        for internal_unsafe_fn in self.internal_unsafe_fns.keys() {
            // Skip if the internal unsafe function is directly a pub function in a pub module
            if self.pub_fns_in_pub_mods.contains(internal_unsafe_fn) {
                // No need to find a path, it's already a pub function
                self.shortest_paths.insert(*internal_unsafe_fn, vec![*internal_unsafe_fn]);
                continue;
            }
            
            // Use BFS to find the shortest path from any pub function to this internal unsafe function
            let path = self.find_shortest_path_to_pub_fn(*internal_unsafe_fn);
            
            if let Some(path) = path {
                self.shortest_paths.insert(*internal_unsafe_fn, path);
            }
        }
    }
    
    // Find shortest path from internal unsafe function to any pub function in a pub module using BFS
    fn find_shortest_path_to_pub_fn(&self, from: DefId) -> Option<Vec<DefId>> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent_map: HashMap<DefId, DefId> = HashMap::new();
        
        queue.push_back(from);
        visited.insert(from);
        
        while let Some(current) = queue.pop_front() {
            // Check if this function is a pub function in a pub module
            if self.pub_fns_in_pub_mods.contains(&current) && current != from {
                // Reconstruct path from pub function to internal unsafe function
                let mut path = Vec::new();
                let mut curr = current;
                
                // Start with the pub function
                path.push(curr);
                
                // Reconstruct path back to the internal unsafe function
                while curr != from {
                    curr = *parent_map.get(&curr).unwrap();
                    path.push(curr);
                }
                
                // Reverse to get path from pub function to internal unsafe function
                path.reverse();
                return Some(path);
            }
            
            // Add all callers to the queue
            if let Some(callers) = self.reverse_call_graph.get(&current) {
                for &caller in callers {
                    if !visited.contains(&caller) {
                        visited.insert(caller);
                        queue.push_back(caller);
                        parent_map.insert(caller, current);
                    }
                }
            }
        }
        
        None
    }

    fn report_findings(&self) {
        rap_info!("===== Internal Unsafe Functions Report =====");
        
        if self.shortest_paths.is_empty() {
            rap_info!("No paths from public functions to internal unsafe functions were found.");
            return;
        }
        
        let mut count = 0;
        
        // First, report internal unsafe functions that are directly pub functions in pub modules
        for (&internal_fn, internal_unsafe) in &self.internal_unsafe_fns {
            if self.pub_fns_in_pub_mods.contains(&internal_fn) {
                count += 1;
                rap_info!("{}: Internal unsafe function is directly a public function in a public module: {}", 
                         count, self.get_fn_name(internal_fn));
                
                // 显示不安全操作
                if !internal_unsafe.unsafe_operations.is_empty() {
                    rap_info!("(unsafe operations: ");
                    for (i, op) in internal_unsafe.unsafe_operations.iter().enumerate() {
                        rap_info!("({}) {}, ", i+1, op.operation_detail);
                    }
                    rap_info!(")");
                }
            }
        }
        
        // Then report paths from pub functions to internal unsafe functions
        for (&internal_fn, path) in &self.shortest_paths {
            // 只跳过那些"本身就是公共函数且在公共模块中"的内部不安全函数
            // 注意：我们需要保留"公共模块中的私有内部不安全函数"的路径
            let internal_unsafe = self.internal_unsafe_fns.get(&internal_fn);
            if let Some(internal_unsafe) = internal_unsafe {
                // 只有当函数既在公共模块中，又是公共函数时，才跳过
                if internal_unsafe.is_public && internal_unsafe.is_in_pub_mod {
                    continue;
                }
            }
            
            count += 1;
            
            if path.len() > 1 {
                let path_str = path.iter()
                    .map(|&def_id| self.get_fn_name(def_id))
                    .collect::<Vec<_>>()
                    .join(" -> ");
                
                rap_info!("{}: Path from public function to internal unsafe function: {}", 
                         count, path_str);
                
                // 显示不安全操作
                if let Some(internal_unsafe) = internal_unsafe {
                    if !internal_unsafe.unsafe_operations.is_empty() {
                        rap_info!("(unsafe operations: ");
                        for (i, op) in internal_unsafe.unsafe_operations.iter().enumerate() {
                            rap_info!("({}) {}, ", i+1, op.operation_detail);
                        }
                        rap_info!(")");
                    }
                }
            }
        }
        
        rap_info!("Total paths reported: {}", count);
    }

    // Helper functions
    fn is_public_fn(&self, def_id: DefId) -> bool {
        // Check visibility
        self.tcx.visibility(def_id).is_public()
    }
    
    fn is_in_public_module(&self, def_id: DefId) -> bool {
        // Get the module of this function
        if let Some(local_def_id) = def_id.as_local() {
            let hir_id = self.tcx.local_def_id_to_hir_id(local_def_id);
            
            // 使用parent_module方法并传递正确类型的参数
            let mod_def_id = self.tcx.parent_module(hir_id);
            
            // 检查模块是否是public
            self.tcx.visibility(mod_def_id).is_public()
        } else {
            // 如果不是本地def_id，假设为非公开模块
            false
        }
    }

    fn get_fn_name(&self, def_id: DefId) -> String {
        self.tcx.def_path_str(def_id)
    }

    fn get_callees(&self, def_id: DefId) -> Vec<DefId> {
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

    /// 提取函数中的不安全操作
    fn extract_unsafe_operations(&self, def_id: DefId) -> Vec<UnsafeOperation> {
        let mut operations = Vec::new();
        
        if !self.tcx.is_mir_available(def_id) {
            return operations;
        }
        
        let body = self.tcx.optimized_mir(def_id);
        
        // 遍历所有基本块
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                // 检查函数调用
                if let TerminatorKind::Call { func, args, .. } = &terminator.kind {
                    if let Operand::Constant(constant) = func {
                        if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) = constant.const_.ty().kind() {
                            // 获取被调用函数的名称
                            let callee_name = self.get_fn_name(*callee_def_id);
                            
                            // 检查是否为unsafe函数调用
                            let is_unsafe_fn = self.is_unsafe_fn(*callee_def_id);
                            
                            if is_unsafe_fn {
                                // 添加函数调用类型的不安全操作
                                let operation = UnsafeOperation {
                                    operation_type: "unsafe function call".to_string(),
                                    operation_detail: callee_name.clone(),
                                };
                                operations.push(operation);
                            }
                            
                            // 记录函数参数中的指针操作
                            for (i, arg) in args.iter().enumerate() {
                                if let Operand::Copy(place) | Operand::Move(place) = &arg.node {
                                    let ty = place.ty(&body.local_decls, self.tcx).ty;
                                    
                                    // 检查是否为原始指针操作
                                    if let rustc_middle::ty::TyKind::RawPtr(..) = ty.kind() {
                                        let operation = UnsafeOperation {
                                            operation_type: "raw pointer operation".to_string(),
                                            operation_detail: format!("arg {} in call to {}", i, callee_name),
                                        };
                                        operations.push(operation);
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 检查内联汇编
                if let TerminatorKind::InlineAsm { .. } = &terminator.kind {
                    let operation = UnsafeOperation {
                        operation_type: "inline assembly".to_string(),
                        operation_detail: "inline assembly code".to_string(),
                    };
                    operations.push(operation);
                }
            }
            
            // 检查语句中的不安全操作
            for _statement in &block_data.statements {
                // TODO: 可以进一步细化对语句中不安全操作的检测
                // 例如检测对原始指针的解引用等
            }
        }
        
        operations
    }
    
    /// 判断函数是否是unsafe函数
    fn is_unsafe_fn(&self, def_id: DefId) -> bool {
        // 检查函数是否可用
        if !self.tcx.is_mir_available(def_id) {
            return false;
        }
        
        // 获取函数签名
        if let Some(local_def_id) = def_id.as_local() {
            let hir = self.tcx.hir();
            if let Some(node) = hir.get_if_local(def_id) {
                if let rustc_hir::Node::Item(item) = node {
                    if let rustc_hir::ItemKind::Fn(sig, ..) = &item.kind {
                        return matches!(sig.header.safety, rustc_hir::Safety::Unsafe);
                    }
                }
            }
        } else {
            // 对于非本地函数，使用其他方法获取函数签名
            let sig = self.tcx.fn_sig(def_id);
            return matches!(sig.skip_binder().safety(), rustc_hir::Safety::Unsafe);
        }
        
        false
    }
}