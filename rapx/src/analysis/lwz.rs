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
    /// 操作的详细描述（比如被调用的函数名）
    operation_detail: String,
}

/// Represents an internal unsafe function
#[derive(Clone, Debug)]
struct InternalUnsafe {
    def_id: DefId,
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
    /// 存储符合pattern1的函数和对应的unsafe操作
    pattern1_matches: HashMap<DefId, Vec<UnsafeOperation>>,
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
            pattern1_matches: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        rap_info!("Starting LWZ analysis to detect internal unsafe functions...");
        
        self.collect_functions();
        self.build_call_graphs();
        self.find_shortest_paths();
        self.detect_pattern1_matches();
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
                    rap_info!("unsafe operations: ");
                    for (i, op) in internal_unsafe.unsafe_operations.iter().enumerate() {
                        rap_info!("({}) {}, ", i+1, op.operation_detail);
                    }
                    rap_info!("\n");
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
                        rap_info!("unsafe operations: ");
                        for (i, op) in internal_unsafe.unsafe_operations.iter().enumerate() {
                            rap_info!("({}) {}, ", i+1, op.operation_detail);
                        }
                        rap_info!("\n");
                    }
                }
            }
        }
        
        rap_info!("Total paths reported: {}", count);
        
        // 输出pattern1匹配结果
        if !self.pattern1_matches.is_empty() {
            rap_info!("\n===== Pattern1 (Parameter Directly Used in Unsafe Operation) Report =====");
            let mut pattern1_count = 0;
            
            for (&fn_id, ops) in &self.pattern1_matches {
                pattern1_count += 1;
                let fn_name = self.get_fn_name(fn_id);
                
                rap_info!("{}: Public function with direct parameter to unsafe operation: {}", pattern1_count, fn_name);
                
                // 显示不安全操作
                if !ops.is_empty() {
                    rap_info!("unsafe operations: ");
                    for (i, op) in ops.iter().enumerate() {
                        rap_info!("({}) {}, ", i+1, op.operation_detail);
                    }
                    rap_info!("\n");
                }
            }
            
            rap_info!("Total pattern1 functions found: {}", pattern1_count);
        }
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
            // 注意：避免在impl块内部项上调用visibility，因为这可能导致ICE
            // 对于impl块中的项，查看其所属类型或模块的可见性
            let parent_vis = self.tcx.visibility(mod_def_id);
            return parent_vis.is_public();
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
        
        // 获取MIR
        let body = self.tcx.optimized_mir(def_id);
        let fn_name = self.get_fn_name(def_id);
        
        // DEBUG输出
        self.debug_log(format!("开始提取不安全操作 - 函数: {}", fn_name));
        
        // 遍历所有基本块 
        for block_data in body.basic_blocks.iter() {
            // 检查语句中的不安全操作
            for statement in &block_data.statements {
                // 检查statement中的不安全操作
                match &statement.kind {
                    // 检查Assign语句中可能的不安全操作
                    rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) => {
                        // 检查左侧的place是否包含解引用裸指针
                        self.check_place_for_raw_ptr_deref(place, &body.local_decls, &mut operations);
                        
                        // 检查右值中的不安全操作
                        match rvalue {
                            // 检查使用原始指针
                            rustc_middle::mir::Rvalue::Use(operand) => {
                                if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                                    // 检查操作数中是否包含解引用裸指针
                                    self.check_place_for_raw_ptr_deref(source_place, &body.local_decls, &mut operations);
                                }
                            },
                            // 检查原始指针算术操作
                            rustc_middle::mir::Rvalue::BinaryOp(op, box (left, _)) => {
                                if let rustc_middle::mir::BinOp::Offset = op {
                                    if let Operand::Copy(place) | Operand::Move(place) = left {
                                        let ty = place.ty(&body.local_decls, self.tcx).ty;
                                        if let rustc_middle::ty::TyKind::RawPtr(..) = ty.kind() {
                                            let operation = UnsafeOperation {
                                                operation_detail: "offset operation on raw pointer".to_string(),
                                            };
                                            operations.push(operation);
                                        }
                                    }
                                }
                            },
                            // 检查RawPtr操作
                            rustc_middle::mir::Rvalue::RawPtr(_, _place) => {
                                // 创建原始指针是安全的，但解引用是不安全的
                                // 这里我们只记录创建操作，不标记为不安全
                            },
                            // 检查Ref操作，可能涉及到裸指针
                            rustc_middle::mir::Rvalue::Ref(_, _, place) => {
                                // 检查是否对裸指针进行了引用操作
                                let ty = place.ty(&body.local_decls, self.tcx).ty;
                                if let rustc_middle::ty::TyKind::RawPtr(..) = ty.kind() {
                                    // 对裸指针取引用可能是不安全的，取决于上下文
                                    self.check_place_for_raw_ptr_deref(place, &body.local_decls, &mut operations);
                                }
                            },
                            // 其他可能的不安全操作
                            _ => {}
                        }
                    },
                    // 检查Intrinsic调用，比如copy_nonoverlapping
                    rustc_middle::mir::StatementKind::Intrinsic(box intrinsic) => {
                        // 内部函数调用通常是unsafe的
                        let operation = UnsafeOperation {
                            operation_detail: format!("intrinsic call: {:?}", intrinsic),
                        };
                        operations.push(operation);
                    },
                    _ => {}
                }
            }

            // 检查终结符中的不安全操作
            if let Some(terminator) = &block_data.terminator {
                match &terminator.kind {
                    // 检查函数调用
                    TerminatorKind::Call { func, args, .. } => {
                        // 处理函数调用
                        if let Operand::Constant(constant) = func {
                            if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) = constant.const_.ty().kind() {
                                // 检查被调用函数是否标记为unsafe
                                if self.is_unsafe_fn(*callee_def_id) {
                                    let callee_name = self.get_fn_name(*callee_def_id);
                                    
                                    // DEBUG输出
                                    self.debug_log(format!("发现unsafe函数调用: {} 在函数 {}", callee_name, fn_name));
                                    
                                    // 提取函数参数信息
                                    let mut arg_descriptions = Vec::new();
                                    for (i, arg) in args.iter().enumerate() {
                                        match &arg.node {
                                            Operand::Copy(place) | Operand::Move(place) => {
                                                // 直接获取原始参数变量编号，不进行额外处理
                                                let arg_str = format!("_{}", place.local.as_usize());
                                                arg_descriptions.push(arg_str.clone());
                                                
                                                // DEBUG输出
                                                self.debug_log(format!("参数 #{}: {} (本地变量ID: {})", 
                                                          i, arg_str, place.local.as_usize()));
                                            },
                                            Operand::Constant(constant) => {
                                                // 处理常量参数
                                                let arg_str = format!("{:?}", constant.const_);
                                                arg_descriptions.push(arg_str.clone());
                                                
                                                // DEBUG输出
                                                self.debug_log(format!("参数 #{}: {} (常量)", i, arg_str));
                                            },
                                        }
                                    }
                                    
                                    // 构建参数字符串
                                    let args_str = if !arg_descriptions.is_empty() {
                                        arg_descriptions.join(", ")
                                    } else {
                                        String::new()
                                    };
                                    
                                    let operation = UnsafeOperation {
                                        operation_detail: format!("{}({})", callee_name, args_str),
                                    };
                                    operations.push(operation);
                                }
                            }
                        } else if let Operand::Copy(place) | Operand::Move(place) = func {
                            // 处理方法调用
                            // 尝试确定方法是否是不安全的
                            if let Some(method_def_id) = self.resolve_method(place, &body.local_decls) {
                                if self.is_unsafe_fn(method_def_id) {
                                    let method_name = self.get_fn_name(method_def_id);
                                    
                                    // 提取参数信息
                                    let mut arg_descriptions = Vec::new();
                                    for arg in args.iter() {
                                        match &arg.node {
                                            Operand::Copy(place) | Operand::Move(place) => {
                                                // 直接获取原始参数变量编号
                                                let arg_str = format!("_{}", place.local.as_usize());
                                                arg_descriptions.push(arg_str);
                                            },
                                            Operand::Constant(constant) => {
                                                // 处理常量参数
                                                arg_descriptions.push(format!("{:?}", constant.const_));
                                            },
                                        }
                                    }
                                    
                                    // 构建参数字符串
                                    let args_str = if !arg_descriptions.is_empty() {
                                        arg_descriptions.join(", ")
                                    } else {
                                        String::new()
                                    };
                                    
                                    let operation = UnsafeOperation {
                                        operation_detail: format!("{}({})", method_name, args_str),
                                    };
                                    operations.push(operation);
                                }
                            }
                            
                            // 检查函数指针是否是裸指针解引用
                            self.check_place_for_raw_ptr_deref(place, &body.local_decls, &mut operations);
                        }
                    },
                    // 检查内联汇编
                    TerminatorKind::InlineAsm { .. } => {
                        let operation = UnsafeOperation {
                            operation_detail: "inline assembly".to_string(),
                        };
                        operations.push(operation);
                    },
                    // 检查其他可能的不安全操作类型
                    _ => {}
                }
            }
        }
        
        // DEBUG输出
        self.debug_log(format!("函数 {} 中共发现 {} 个不安全操作", fn_name, operations.len()));
        for (i, op) in operations.iter().enumerate() {
            self.debug_log(format!("   不安全操作 #{}: {}", i+1, op.operation_detail));
        }
        
        operations
    }
    
    /// 检查Place中是否包含解引用裸指针的操作
    fn check_place_for_raw_ptr_deref(&self, 
                                     place: &rustc_middle::mir::Place<'tcx>, 
                                     local_decls: &rustc_middle::mir::LocalDecls<'tcx>, 
                                     operations: &mut Vec<UnsafeOperation>) {
        // 检查投影序列中是否有解引用操作
        for (i, proj) in place.projection.iter().enumerate() {
            if let rustc_middle::mir::ProjectionElem::Deref = proj {
                // 获取被解引用的类型
                let prefix_place = rustc_middle::mir::Place {
                    local: place.local,
                    projection: self.tcx.mk_place_elems(&place.projection[0..i]),
                };
                
                let prefix_ty = prefix_place.ty(local_decls, self.tcx).ty;
                
                // 检查是否是裸指针类型
                if let rustc_middle::ty::TyKind::RawPtr(..) = prefix_ty.kind() {
                    // 尝试从 MIR 获取变量名信息
                    let var_name = self.get_place_description(&prefix_place, local_decls);
                    
                    let operation = UnsafeOperation {
                        operation_detail: format!("*{}", var_name),
                    };
                    operations.push(operation);
                }
            }
        }
    }

    /// 尝试从 Place 获取变量描述
    fn get_place_description(&self, place: &rustc_middle::mir::Place<'tcx>, 
                            local_decls: &rustc_middle::mir::LocalDecls<'tcx>) -> String {
        // 获取局部变量名
        let local = place.local;
        let local_decl = &local_decls[local];
        
        // 尝试获取变量名
        let base_name = match local_decl.local_info {
            rustc_middle::mir::ClearCrossCrate::Set(ref info) => {
                match **info {
                    rustc_middle::mir::LocalInfo::User(ref binding) => {
                        // BindingForm没有实现ToString，使用Debug格式化
                        format!("{:?}", binding)
                    },
                    rustc_middle::mir::LocalInfo::BlockTailTemp(ref _block) => {
                        format!("_temp_{}", local.as_usize())
                    },
                    rustc_middle::mir::LocalInfo::Boring => {
                        format!("_var_{}", local.as_usize())
                    },
                    // 匹配任何其他变体
                    _ => format!("_var_{}", local.as_usize()),
                }
            },
            rustc_middle::mir::ClearCrossCrate::Clear => {
                // 如果信息被清除，使用默认格式
                if local.as_usize() == 0 {
                    // 返回值特殊处理
                    "_return".to_string()
                } else {
                    // 为临时变量使用下标
                    format!("_{}", local.as_usize())
                }
            }
        };
        
        // 处理投影
        let mut result = base_name;
        for elem in place.projection.iter() {
            match elem {
                rustc_middle::mir::ProjectionElem::Deref => {
                    result = format!("*{}", result);
                },
                rustc_middle::mir::ProjectionElem::Field(field, _) => {
                    result = format!("{}.{}", result, field.index());
                },
                rustc_middle::mir::ProjectionElem::Index(idx) => {
                    result = format!("{}[_{:?}]", result, idx);
                },
                _ => {}
            }
        }
        
        result
    }

    /// 判断函数是否是unsafe函数
    fn is_unsafe_fn(&self, def_id: DefId) -> bool {
        // 参考 unsafety_isolation 模块的实现
        if self.tcx.is_mir_available(def_id) {
            let poly_fn_sig = self.tcx.fn_sig(def_id);
            let fn_sig = poly_fn_sig.skip_binder();
            return fn_sig.safety() == rustc_hir::Safety::Unsafe;
        }
        false
    }

    /// 尝试解析方法调用对应的函数定义ID
    fn resolve_method(&self, place: &rustc_middle::mir::Place<'tcx>, _local_decls: &rustc_middle::mir::LocalDecls<'tcx>) -> Option<DefId> {
        // 这是一个简化的实现，实际上解析方法调用比较复杂
        // 在实际场景中，可能需要使用typeck结果或其他机制来准确解析方法
        // 此实现仅作为示例
        if let Some(field) = place.projection.last() {
            if let rustc_middle::mir::ProjectionElem::Field(_, _) = field {
                // 这里应该有更复杂的逻辑来解析方法调用
                // 但由于限制，我们返回None
                return None;
            }
        }
        None
    }

    /// 检测符合pattern1的函数
    /// pattern1: pub函数的参数直接传入unsafe操作
    fn detect_pattern1_matches(&mut self) {
        self.debug_log("开始检测pattern1匹配...");
        
        for (&def_id, internal_unsafe) in &self.internal_unsafe_fns {
            if self.is_public_fn(def_id) {
                let fn_name = self.get_fn_name(def_id);
                
                self.debug_log(format!("检查公共函数: {}", fn_name));
                
                // 处理函数参数的情况
                let mut pattern1_ops = Vec::new();
                
                // 获取MIR以分析参数
                if let Some(body) = self.tcx.is_mir_available(def_id).then(|| self.tcx.optimized_mir(def_id)) {
                    let param_count = body.arg_count;
                    
                    self.debug_log(format!("函数 {} 有 {} 个参数", fn_name, param_count));
                    
                    // 检查所有不安全操作，是否有直接或间接使用参数的情况
                    for op in &internal_unsafe.unsafe_operations {
                        self.debug_log(format!("检查不安全操作: {}", op.operation_detail));
                        
                        // 1. 检查解引用参数的情况
                        if op.operation_detail.starts_with("*_") {
                            // 从操作详情中提取变量编号
                            if let Some(var_number) = self.extract_var_number(&op.operation_detail) {
                                // 检查变量编号是否在参数范围内，或者是否是参数的复制
                                if self.is_param_or_copy(body, var_number, param_count) {
                                    pattern1_ops.push(op.clone());
                                    self.debug_log(format!("发现pattern1解引用参数: 变量 {} 在函数 {}", var_number, fn_name));
                                } else {
                                    self.debug_log(format!("变量 {} 不是参数或参数复制", var_number));
                                }
                            }
                        } 
                        // 2. 检查函数调用参数的情况
                        else if op.operation_detail.contains("(") && op.operation_detail.contains(")") {
                            self.debug_log(format!("分析函数调用: {}", op.operation_detail));
                            
                            if self.check_function_call_args(&op.operation_detail, body, param_count) {
                                pattern1_ops.push(op.clone());
                                self.debug_log(format!("发现pattern1函数调用参数在函数 {}", fn_name));
                            }
                        }
                    }
                }
                
                // 如果找到pattern1操作，保存结果
                if !pattern1_ops.is_empty() {
                    self.pattern1_matches.insert(def_id, pattern1_ops.clone());
                    self.debug_log(format!("函数 {} 匹配pattern1，共 {} 个匹配操作", fn_name, pattern1_ops.len()));
                } else {
                    self.debug_log(format!("函数 {} 没有匹配pattern1", fn_name));
                }
            }
        }
        
        self.debug_log(format!("pattern1检测完成，共发现 {} 个匹配函数", self.pattern1_matches.len()));
    }
    
    /// 检查变量是否是函数参数或参数的复制
    fn is_param_or_copy(&self, body: &rustc_middle::mir::Body<'tcx>, var_num: usize, param_count: usize) -> bool {
        self.is_param_or_copy_with_visited(body, var_num, param_count, &mut HashSet::new())
    }

    fn is_param_or_copy_with_visited(&self, 
                                    body: &rustc_middle::mir::Body<'tcx>, 
                                    var_num: usize, 
                                    param_count: usize,
                                    visited: &mut HashSet<usize>) -> bool {
        // 如果已经访问过这个变量，避免循环
        if !visited.insert(var_num) {
            return false;
        }
        
        // 如果变量本身就是参数，直接返回true
        if var_num > 0 && var_num <= param_count {
            return true;
        }
        
        // 检查变量是否是参数的复制
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    // 检查赋值的左侧是否是我们关注的变量
                    if place.local.as_usize() == var_num {
                        // 检查右侧是否是对参数的引用或复制
                        match rvalue {
                            rustc_middle::mir::Rvalue::Use(operand) => {
                                if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                                    let source_local = source_place.local.as_usize();
                                    // 检查源变量是否是参数
                                    if source_local > 0 && source_local <= param_count {
                                        // DEBUG输出
                                        self.debug_log(format!("变量 {} 是参数 {} 的直接复制", var_num, source_local));
                                        return true;
                                    }
                                    // 递归检查源变量是否是参数的复制
                                    if self.is_param_or_copy_with_visited(body, source_local, param_count, visited) {
                                        // DEBUG输出
                                        self.debug_log(format!("变量 {} 是间接复制自参数", var_num));
                                        return true;
                                    }
                                }
                            },
                            rustc_middle::mir::Rvalue::Ref(_, _, place) => {
                                let source_local = place.local.as_usize();
                                // 检查源变量是否是参数或参数的复制
                                if source_local > 0 && source_local <= param_count {
                                    // DEBUG输出
                                    self.debug_log(format!("变量 {} 是对参数 {} 的引用", var_num, source_local));
                                    return true;
                                }
                                if self.is_param_or_copy_with_visited(body, source_local, param_count, visited) {
                                    // DEBUG输出
                                    self.debug_log(format!("变量 {} 是对参数复制的引用", var_num));
                                    return true;
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
        
        false
    }

    // 从操作详情中提取变量编号（如从 "*_1" 提取出 1）
    fn extract_var_number(&self, op_detail: &str) -> Option<usize> {
        let prefixes = ["*_", "copy _", "move _"];
        
        for prefix in &prefixes {
            if op_detail.starts_with(prefix) {
                let var_part = &op_detail[prefix.len()..];
                // 使用 split_at_first_non_digit 辅助函数获取数字部分
                let digit_end = var_part.find(|c: char| !c.is_ascii_digit()).unwrap_or(var_part.len());
                return var_part[..digit_end].parse::<usize>().ok();
            }
        }
        None
    }

    // 添加一个辅助函数控制调试输出
    fn debug_log(&self, msg: impl AsRef<str>) {
        // 可以根据环境变量或其他条件决定是否输出调试信息
        rap_info!("DEBUG: {}", msg.as_ref());
    }

    // 添加辅助函数检查函数调用中的参数
    fn check_function_call_args(&self, 
                               op_detail: &str, 
                               body: &rustc_middle::mir::Body<'tcx>, 
                               param_count: usize) -> bool {
        if let Some(start_pos) = op_detail.find('(') {
            if let Some(end_pos) = op_detail.rfind(')') {
                if start_pos < end_pos {
                    let args_str = &op_detail[start_pos+1..end_pos];
                    // 分割参数
                    for arg in args_str.split(',') {
                        let arg = arg.trim();
                        if arg.starts_with("_") {
                            // 尝试提取参数编号
                            if let Ok(arg_num) = arg[1..].parse::<usize>() {
                                // 检查编号是否在参数范围内，或者是否是参数的复制
                                if self.is_param_or_copy(body, arg_num, param_count) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

}