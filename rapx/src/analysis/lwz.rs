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
use rustc_target::abi::VariantIdx;
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

/// 表示污点传播路径中的函数
#[derive(Debug, Clone)]
struct PatternCarrier {
    /// 函数的DefId
    def_id: DefId,
    /// 被污染的参数索引（从1开始）
    tainted_param_idx: usize,
    /// 污染最终流向的不安全操作
    unsafe_ops: Vec<UnsafeOperation>,
    /// 识别的模式类型（1=Pattern1, 2=Pattern2）
    pattern_type: u8,
}

/// 存储过程间分析结果
#[derive(Debug, Clone)]
struct InterproceduralMatch {
    /// 公开函数
    pub_fn_id: DefId,
    /// 调用链路径
    call_path: Vec<DefId>,
    /// 最终的不安全操作
    unsafe_ops: Vec<UnsafeOperation>,
    /// 基于哪种模式传播（1=Pattern1, 2=Pattern2）
    base_pattern: u8,
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

/// 表示结构体字段的信息
#[derive(Debug, Clone)]
struct FieldInfo {
    /// 字段在结构体中的索引
    index: usize,
    /// 字段名称
    name: String,
    /// 字段是否为pub
    is_public: bool,
}

/// 表示一个公共结构体的信息
#[derive(Debug, Clone)]
struct PubStructInfo {
    def_id: DefId,
    /// 结构体是否在公共模块中
    is_in_pub_mod: bool,
    /// 存储结构体的公共字段信息
    pub_fields: HashMap<usize, FieldInfo>,
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
    /// 存储公共结构体信息
    pub_structs: HashMap<DefId, PubStructInfo>,
    /// 存储符合pattern2的函数和对应的unsafe操作与字段信息
    pattern2_matches: HashMap<DefId, Vec<(UnsafeOperation, String)>>,
    /// 存储所有的"模式载体"函数
    pattern_carriers: HashMap<DefId, PatternCarrier>,
    /// 存储过程间分析的结果
    interprocedural_matches: HashMap<DefId, InterproceduralMatch>,
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
            pub_structs: HashMap::new(),
            pattern2_matches: HashMap::new(),
            pattern_carriers: HashMap::new(),
            interprocedural_matches: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        // rap_info!("Starting LWZ analysis to detect internal unsafe functions...");
        
        self.collect_functions();
        self.build_call_graphs();
        self.find_shortest_paths();
        
        // 第一阶段：识别所有"模式载体"
        self.identify_pattern_carriers();
        
        // 第二阶段：基于"模式载体"进行过程间分析
        self.perform_interprocedural_analysis();
        
        // 原有的分析
        self.detect_pattern1_matches();
        self.collect_pub_structs();
        self.detect_pattern2_matches();
        
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
        
        // rap_info!("Found {} internal unsafe functions and {} public functions in public modules",
        //     self.internal_unsafe_fns.len(), self.pub_fns_in_pub_mods.len());
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
        // rap_info!("===== Internal Unsafe Functions Report =====");
        
        // if self.shortest_paths.is_empty() {
        //     rap_info!("No paths from public functions to internal unsafe functions were found.");
        //     return;
        // }
        
        // let mut count = 0;
        
        // // First, report internal unsafe functions that are directly pub functions in pub modules
        // for (&internal_fn, internal_unsafe) in &self.internal_unsafe_fns {
        //     if self.pub_fns_in_pub_mods.contains(&internal_fn) {
        //         count += 1;
        //         rap_info!("{}: Internal unsafe function is directly a public function in a public module: {}", 
        //                  count, self.get_fn_name(internal_fn));
                
        //         // 显示不安全操作
        //         if !internal_unsafe.unsafe_operations.is_empty() {
        //             rap_info!("unsafe operations: ");
        //             for (i, op) in internal_unsafe.unsafe_operations.iter().enumerate() {
        //                 rap_info!("({}) {}, ", i+1, op.operation_detail);
        //             }
        //             rap_info!("\n");
        //         }
        //     }
        // }
        
        // // Then report paths from pub functions to internal unsafe functions
        // for (&internal_fn, path) in &self.shortest_paths {
        //     // 只跳过那些"本身就是公共函数且在公共模块中"的内部不安全函数
        //     // 注意：我们需要保留"公共模块中的私有内部不安全函数"的路径
        //     let internal_unsafe = self.internal_unsafe_fns.get(&internal_fn);
        //     if let Some(internal_unsafe) = internal_unsafe {
        //         // 只有当函数既在公共模块中，又是公共函数时，才跳过
        //         if internal_unsafe.is_public && internal_unsafe.is_in_pub_mod {
        //             continue;
        //         }
        //     }
            
        //     count += 1;
            
        //     if path.len() > 1 {
        //         let path_str = path.iter()
        //             .map(|&def_id| self.get_fn_name(def_id))
        //             .collect::<Vec<_>>()
        //             .join(" -> ");
                
        //         rap_info!("{}: Path from public function to internal unsafe function: {}", 
        //                  count, path_str);
                
        //         // 显示不安全操作
        //         if let Some(internal_unsafe) = internal_unsafe {
        //             if !internal_unsafe.unsafe_operations.is_empty() {
        //                 rap_info!("unsafe operations: ");
        //                 for (i, op) in internal_unsafe.unsafe_operations.iter().enumerate() {
        //                     rap_info!("({}) {}, ", i+1, op.operation_detail);
        //                 }
        //                 rap_info!("\n");
        //             }
        //         }
        //     }
        // }
        
        // rap_info!("Total paths reported: {}", count);
        
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
        
        // 输出pattern2匹配结果
        if !self.pattern2_matches.is_empty() {
            rap_info!("\n===== Pattern2 (Public Struct Field Used in Unsafe Operation) Report =====");
            let mut pattern2_count = 0;
            
            for (&fn_id, ops_with_fields) in &self.pattern2_matches {
                pattern2_count += 1;
                let fn_name = self.get_fn_name(fn_id);
                
                rap_info!("{}: Public function using public struct field in unsafe operation: {}", 
                         pattern2_count, fn_name);
                
                // 显示不安全操作和相关字段
                if !ops_with_fields.is_empty() {
                    rap_info!("unsafe operations with struct fields: ");
                    for (i, (op, field_path)) in ops_with_fields.iter().enumerate() {
                        rap_info!("({}) Field {} used in {}, ", i+1, field_path, op.operation_detail);
                    }
                    rap_info!("\n");
                }
            }
            
            rap_info!("Total pattern2 functions found: {}", pattern2_count);
        }
        
        // 报告过程间分析结果
        if !self.interprocedural_matches.is_empty() {
            rap_info!("\n===== 过程间污点传播分析报告 =====");
            let mut count = 0;
            
            for (&fn_id, interprocedural_match) in &self.interprocedural_matches {
                count += 1;
                let fn_name = self.get_fn_name(fn_id);
                
                // 构建调用链路径字符串
                let path_str = interprocedural_match.call_path.iter()
                    .map(|&def_id| self.get_fn_name(def_id))
                    .collect::<Vec<_>>()
                    .join(" -> ");
                
                rap_info!("{}: 过程间污点传播: {}", count, path_str);
                
                // 显示不安全操作
                if !interprocedural_match.unsafe_ops.is_empty() {
                    rap_info!("unsafe operations: ");
                    for (i, op) in interprocedural_match.unsafe_ops.iter().enumerate() {
                        rap_info!("({}) {}, ", i+1, op.operation_detail);
                    }
                    rap_info!("\n");
                }
            }
            
            rap_info!("总计发现 {} 个过程间污点传播", count);
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
        
        // 安全地获取MIR
        let body = match self.get_mir_safely(def_id) {
            Some(body) => body,
            None => {
                //self.debug_log(format!("无法安全获取函数 {} 的MIR，跳过调用图构建", 
                //               self.get_fn_name(def_id)));
                return callees;
            }
        };
        
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

    /// 安全地获取MIR，捕获可能的panic
    fn get_mir_safely(&self, def_id: DefId) -> Option<&rustc_middle::mir::Body<'tcx>> {
        use std::panic::{self, AssertUnwindSafe};
        
        // 检查是否是常量、常量字段或常量表达式
        let is_const = self.tcx.is_const_fn_raw(def_id) || 
                      self.tcx.is_const_fn(def_id) || 
                      matches!(self.tcx.def_kind(def_id), 
                          rustc_hir::def::DefKind::Const | 
                          rustc_hir::def::DefKind::AssocConst | 
                          rustc_hir::def::DefKind::Static { safety: _, mutability: _, nested: _ });
                          
        // 检查def_id是否包含常量表达式的特殊标记
        let def_path = self.tcx.def_path_str(def_id);
        let is_const_expr = def_path.contains("::{constant#") || 
                           def_path.contains("::promoted[") ||
                           def_path.contains("]::") || 
                           def_path.ends_with("}");
        
        // 对于常量或常量表达式，使用mir_for_ctfe而不是optimized_mir
        if is_const || is_const_expr {
            let def_kind = self.tcx.def_kind(def_id);
            if matches!(def_kind, rustc_hir::def::DefKind::Const | rustc_hir::def::DefKind::Static { safety: _, mutability: _, nested: _ }) {
                // 对于常量和静态变量，使用mir_for_ctfe
                return Some(self.tcx.mir_for_ctfe(def_id));
            } else {
                // 对于其他类型的常量表达式，可能无法安全获取MIR
                //self.debug_log(format!("跳过常量表达式的MIR获取: {:?}", def_id));
                return None;
            }
        }
        
        // 对于非常量，使用try_catch_unwind避免面向panic编程
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            if self.tcx.is_mir_available(def_id) {
                Some(self.tcx.optimized_mir(def_id))
            } else {
                None
            }
        }));
        
        match result {
            Ok(Some(mir)) => Some(mir),
            Ok(None) => None,
            Err(_) => {
                // 记录错误并返回None
                //self.debug_log(format!("获取def_id {:?}的MIR时发生panic，已跳过", def_id));
                None
            }
        }
    }

    /// 提取函数中的不安全操作
    fn extract_unsafe_operations(&self, def_id: DefId) -> Vec<UnsafeOperation> {
        let mut operations = Vec::new();
        
        // 使用安全的MIR获取方法
        let body = match self.get_mir_safely(def_id) {
            Some(body) => body,
            None => {
                //self.debug_log(format!("无法安全获取函数 {} 的MIR，跳过不安全操作分析", 
                //               self.get_fn_name(def_id)));
                return operations;
            }
        };
        
        let fn_name = self.get_fn_name(def_id);
        
        // DEBUG输出
        //self.debug_log(format!("开始提取不安全操作 - 函数: {}", fn_name));
        
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
                                    //self.debug_log(format!("发现unsafe函数调用: {} 在函数 {}", callee_name, fn_name));
                                    
                                    // 提取函数参数信息
                                    let mut arg_descriptions = Vec::new();
                                    for (i, arg) in args.iter().enumerate() {
                                        match &arg.node {
                                            Operand::Copy(place) | Operand::Move(place) => {
                                                // 直接获取原始参数变量编号，不进行额外处理
                                                let arg_str = format!("_{}", place.local.as_usize());
                                                arg_descriptions.push(arg_str.clone());
                                                
                                                // DEBUG输出
                                                //self.debug_log(format!("参数 #{}: {} (本地变量ID: {})", 
                                                          //i, arg_str, place.local.as_usize()));
                                            },
                                            Operand::Constant(constant) => {
                                                // 处理常量参数
                                                let arg_str = format!("{:?}", constant.const_);
                                                arg_descriptions.push(arg_str.clone());
                                                
                                                // DEBUG输出
                                                //self.debug_log(format!("参数 #{}: {} (常量)", i, arg_str));
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
        //(format!("函数 {} 中共发现 {} 个不安全操作", fn_name, operations.len()));
        for (i, op) in operations.iter().enumerate() {
            //self.debug_log(format!("   不安全操作 #{}: {}", i+1, op.operation_detail));
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
    /// pattern1: pub函数的参数(非self.<field>)直接传入unsafe操作
    fn detect_pattern1_matches(&mut self) {
        //self.debug_log("开始检测pattern1匹配...");
        
        for (&def_id, internal_unsafe) in &self.internal_unsafe_fns {
            if self.is_public_fn(def_id) {
                let fn_name = self.get_fn_name(def_id);
                
                //self.debug_log(format!("检查公共函数: {}", fn_name));
                
                // 处理函数参数的情况
                let mut pattern1_ops = Vec::new();
                
                // 获取MIR以分析参数
                if let Some(body) = self.get_mir_safely(def_id) {
                    let param_count = body.arg_count;
                    
                    //self.debug_log(format!("函数 {} 有 {} 个参数", fn_name, param_count));
                    
                    // 判断是否是结构体impl方法
                    let is_method = param_count > 0 && self.get_impl_self_type(def_id).is_some();
                    let self_param = if is_method { Some(1) } else { None };
                    
                    // 检查所有不安全操作，是否有直接或间接使用参数的情况
                    for op in &internal_unsafe.unsafe_operations {
                        //self.debug_log(format!("检查不安全操作: {}", op.operation_detail));
                        
                        // 1. 检查解引用参数的情况
                        if op.operation_detail.starts_with("*_") {
                            // 从操作详情中提取变量编号
                            if let Some(var_number) = self.extract_var_number(&op.operation_detail) {
                                // 检查变量编号是否在参数范围内，但不是self参数
                                let is_pattern1 = self.is_non_self_param_or_copy(body, var_number, param_count, self_param);
                                if is_pattern1 {
                                    // 检查变量是否经过净化操作
                                    if !self.is_var_sanitized(body, var_number, def_id) {
                                        pattern1_ops.push(op.clone());
                                        //self.debug_log(format!("发现pattern1解引用参数: 变量 {} 在函数 {}", var_number, fn_name));
                                    } else {
                                       // self.debug_log(format!("变量 {} 在函数 {} 中已经过净化，跳过", var_number, fn_name));
                                    }
                                } else {
                                    //self.debug_log(format!("变量 {} 不是参数、或是self参数/字段", var_number));
                                }
                            }
                        } 
                        // 2. 检查函数调用参数的情况
                        else if op.operation_detail.contains("(") && op.operation_detail.contains(")") {
                            //self.debug_log(format!("分析函数调用: {}", op.operation_detail));
                            
                            if self.check_function_call_non_self_args_with_sanitization(&op.operation_detail, body, param_count, self_param, def_id) {
                                pattern1_ops.push(op.clone());
                                //self.debug_log(format!("发现pattern1函数调用参数在函数 {}", fn_name));
                            }
                        }
                    }
                }
                
                // 如果找到pattern1操作，保存结果
                if !pattern1_ops.is_empty() {
                    self.pattern1_matches.insert(def_id, pattern1_ops.clone());
                    //self.debug_log(format!("函数 {} 匹配pattern1，共 {} 个匹配操作", fn_name, pattern1_ops.len()));
                } else {
                    //self.debug_log(format!("函数 {} 没有匹配pattern1", fn_name));
                }
            }
        }
        
        //self.debug_log(format!("pattern1检测完成，共发现 {} 个匹配函数", self.pattern1_matches.len()));
    }
    
    /// 检查函数调用中是否包含非self参数，并且参数没有经过净化
    fn check_function_call_non_self_args_with_sanitization(&self, 
                                          op_detail: &str, 
                                          body: &rustc_middle::mir::Body<'tcx>, 
                                          param_count: usize,
                                          self_param: Option<usize>,
                                          def_id: DefId) -> bool {
        let fn_name = self.get_fn_name(def_id);
        //self.debug_log(format!("检查函数调用参数: {} (函数: {})", op_detail, fn_name));
        
        // 不再基于函数名称匹配判断是否为不安全函数
        // 所有出现在extract_unsafe_operations中的操作都应该被视为不安全的
        let is_unsafe_op = true;
        
        if let Some(start_pos) = op_detail.find('(') {
            if let Some(end_pos) = op_detail.rfind(')') {
                if start_pos < end_pos {
                    let args_str = &op_detail[start_pos+1..end_pos];
                    //self.debug_log(format!("  解析参数字符串: {}", args_str));
                    
                    // 分割参数
                    for arg in args_str.split(',') {
                        let arg = arg.trim();
                        //self.debug_log(format!("  检查参数: {}", arg));
                        
                        if arg.starts_with("_") {
                            // 尝试提取参数编号
                            if let Ok(arg_num) = arg[1..].parse::<usize>() {
                                //self.debug_log(format!("  参数编号: {}", arg_num));
                                
                                // 检查是否是非self参数或从非self参数复制
                                if self.is_non_self_param_or_copy(body, arg_num, param_count, self_param) {
                                    //self.debug_log(format!("  变量 {} 是非self参数或从非self参数复制", arg_num));
                                    
                                    // 检查变量是否经过净化操作
                                    if !self.is_var_sanitized(body, arg_num, def_id) {
                                        // 直接传递给不安全操作的参数
                                        //self.debug_log(format!("  变量 {} 未经过净化，匹配pattern1", arg_num));
                                        return true;
                                    } else {
                                        //self.debug_log(format!("  变量 {} 在函数调用中已经过净化，跳过", arg_num));
                                    }
                                } else {
                                    //self.debug_log(format!("  变量 {} 不是非self参数或从非self参数复制", arg_num));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        //self.debug_log("  函数调用不匹配pattern1");
        false
    }
    
    /// 检查变量是否是非self参数或非self参数的复制
    fn is_non_self_param_or_copy(&self, body: &rustc_middle::mir::Body<'tcx>, var_num: usize, param_count: usize, self_param: Option<usize>) -> bool {
        // 排除self参数
        if let Some(self_idx) = self_param {
            if var_num == self_idx {
                return false;
            }
        }
        
        // 检查是否是普通参数
        if var_num > 0 && var_num <= param_count {
            if let Some(self_idx) = self_param {
                return var_num != self_idx; // 确保不是self参数
            }
            return true;
        }
        
        // 检查是否源自普通参数的复制
        let mut visited = HashSet::new();
        self.is_non_self_param_or_copy_with_visited(body, var_num, param_count, self_param, &mut visited)
    }

    /// 递归检查变量是否源自非self参数
    fn is_non_self_param_or_copy_with_visited(&self, 
                                 body: &rustc_middle::mir::Body<'tcx>, 
                                 var_num: usize, 
                                 param_count: usize,
                                 self_param: Option<usize>,
                                 visited: &mut HashSet<usize>) -> bool {
        // 防止循环递归
        if !visited.insert(var_num) {
            return false;
        }
        
        // 再次检查是否是self参数
        if let Some(self_idx) = self_param {
            if var_num == self_idx {
                return false;
            }
        }
        
        // 检查变量是否直接从self参数的字段获取
        // 如果变量是从self.<field>获取的，应该排除
        if self.is_from_self_field(body, var_num, self_param) {
            return false;
        }
        
        // 检查赋值来源
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    // 检查赋值目标是否是当前变量
                    if place.local.as_usize() == var_num {
                        match rvalue {
                            rustc_middle::mir::Rvalue::Use(operand) => {
                                if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                                    let source_local = source_place.local.as_usize();
                                    
                                    // 检查源变量是否是参数
                                    if source_local > 0 && source_local <= param_count {
                                        // 确保不是self参数
                                        if let Some(self_idx) = self_param {
                                            if source_local != self_idx {
                                                return true; // 是普通参数
                                            }
                                        } else {
                                            return true; // 没有self参数，所有参数都是普通参数
                                        }
                                    }
                                    
                                    // 递归检查源变量
                                    if self.is_non_self_param_or_copy_with_visited(body, source_local, param_count, self_param, visited) {
                                        return true;
                                    }
                                }
                            },
                            rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                                let source_local = source_place.local.as_usize();
                                
                                // 检查源变量是否是参数
                                if source_local > 0 && source_local <= param_count {
                                    // 确保不是self参数
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return true; // 是普通参数的引用
                                        }
                                    } else {
                                        return true; // 没有self参数
                                    }
                                }
                                
                                // 递归检查
                                if self.is_non_self_param_or_copy_with_visited(body, source_local, param_count, self_param, visited) {
                                    return true;
                                }
                            },
                            // 添加对裸指针转换的检测
                            rustc_middle::mir::Rvalue::RawPtr(_, source_place) => {
                                let source_local = source_place.local.as_usize();
                                
                                // 检查源变量是否是参数
                                if source_local > 0 && source_local <= param_count {
                                    // 确保不是self参数
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return true; // 是普通参数转换为裸指针
                                        }
                                    } else {
                                        return true; // 没有self参数
                                    }
                                }
                                
                                // 递归检查
                                if self.is_non_self_param_or_copy_with_visited(body, source_local, param_count, self_param, visited) {
                                    return true;
                                }
                            },
                            // 增加对结构体构造(Aggregate)的支持
                            rustc_middle::mir::Rvalue::Aggregate(_, operands) => {
                                // 检查每个操作数是否含有非self参数
                                for operand in operands.iter() {
                                    match operand {
                                        Operand::Copy(source_place) | Operand::Move(source_place) => {
                                            let source_local = source_place.local.as_usize();
                                            
                                            // 检查源变量是否是参数
                                            if source_local > 0 && source_local <= param_count {
                                                // 确保不是self参数
                                                if let Some(self_idx) = self_param {
                                                    if source_local != self_idx {
                                                        return true; // 是通过结构体传递的普通参数
                                                    }
                                                } else {
                                                    return true; // 没有self参数
                                                }
                                            }
                                            
                                            // 递归检查
                                            if self.is_non_self_param_or_copy_with_visited(body, source_local, param_count, self_param, visited) {
                                                return true;
                                            }
                                        },
                                        _ => {}
                                    }
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
        
        // 检查函数调用中的参数
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                if let rustc_middle::mir::TerminatorKind::Call { destination, args, .. } = &terminator.kind {
                    // 检查结果是否赋值给当前变量
                    if destination.local.as_usize() == var_num {
                        // 检查参数中是否有非self参数
                        for arg in args {
                            if let Operand::Copy(place) | Operand::Move(place) = &arg.node {
                                let source_local = place.local.as_usize();
                                
                                // 检查源变量是否是参数
                                if source_local > 0 && source_local <= param_count {
                                    // 确保不是self参数
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return true; // 是通过函数调用传递的普通参数
                                        }
                                    } else {
                                        return true; // 没有self参数
                                    }
                                }
                                
                                // 递归检查
                                if self.is_non_self_param_or_copy_with_visited(body, source_local, param_count, self_param, visited) {
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

    /// 检查变量是否来自self字段
    fn is_from_self_field(&self, body: &rustc_middle::mir::Body<'tcx>, var_num: usize, self_param: Option<usize>) -> bool {
        // 如果没有self参数，直接返回false
        let self_idx = match self_param {
            Some(idx) => idx,
            None => return false,
        };
        
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    if place.local.as_usize() != var_num {
                        continue;
                    }
                    
                    match rvalue {
                        // 直接使用self字段的情况
                        rustc_middle::mir::Rvalue::Use(operand) => {
                            if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                                if source_place.local.as_usize() == self_idx && !source_place.projection.is_empty() {
                                    // 这里检测到从self字段获取值
                                    return true;
                                }
                            }
                        },
                        // 引用self字段的情况
                        rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                            if source_place.local.as_usize() == self_idx && !source_place.projection.is_empty() {
                                // 从self字段获取引用
                                return true;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        
        false
    }

    /// 从操作详情中提取变量编号（如从 "*_1" 提取出 1）
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

    /// 收集所有公共结构体及其公共字段的信息
    fn collect_pub_structs(&mut self) {
        //self.debug_log("开始收集公共结构体信息...");
        
        // 遍历所有本地定义
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();
            
            // 先检查是否是结构体类型，避免对非结构体调用type_of
            let def_kind = self.tcx.def_kind(def_id);
            if def_kind != rustc_hir::def::DefKind::Struct {
                continue;
            }
            
            // 使用catch_unwind来避免整个函数因为处理一个结构体而崩溃
            let collect_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // 检查结构体是否是公共的
                let is_public = self.tcx.visibility(def_id).is_public();
                let is_in_pub_mod = self.is_in_public_module(def_id);
                
                if is_public {
                    let mut pub_struct_info = PubStructInfo {
                        def_id,
                        is_in_pub_mod,
                        pub_fields: HashMap::new(),
                    };
                    
                    // 安全地获取结构体的类型信息
                    let adt_def = self.tcx.adt_def(def_id);
                    let struct_name = self.get_fn_name(def_id);
                    
                    // 遍历结构体字段
                    let variant_idx = VariantIdx::from_usize(0);
                    if let Some(variant) = adt_def.variants().get(variant_idx) {
                        for (idx, field) in variant.fields.iter().enumerate() {
                            // 使用另一个catch_unwind来单独处理每个字段
                            let field_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let field_def_id = field.did;
                                let field_vis = self.tcx.visibility(field_def_id);
                                let field_name = field.ident(self.tcx).to_string();
                                
                                // 只记录公共字段
                                if field_vis.is_public() {
                                    // 我们不再尝试获取字段类型，因为这可能导致泛型结构体的panic
                                    pub_struct_info.pub_fields.insert(idx, FieldInfo {
                                        index: idx,
                                        name: field_name.clone(),
                                        is_public: true,
                                    });
                                    return Some(field_name);
                                }
                                None
                            }));
                            
                            // 如果处理字段时发生panic，记录日志并继续处理下一个字段
                            if field_result.is_err() {
                                //self.debug_log(format!("处理结构体 {} 的字段 {} 时发生panic", struct_name, idx));
                            }
                        }
                    }
                    
                    // 只保存有公共字段的公共结构体
                    if !pub_struct_info.pub_fields.is_empty() {
                        //self.debug_log(format!("发现公共结构体: {} 有 {} 个公共字段", 
                                          //struct_name, pub_struct_info.pub_fields.len()));
                        return Some((def_id, pub_struct_info));
                    }
                }
                None
            }));
            
            // 如果处理结构体时发生panic，记录日志并继续处理下一个结构体
            match collect_result {
                Ok(Some((def_id, pub_struct_info))) => {
                    self.pub_structs.insert(def_id, pub_struct_info);
                },
                Ok(None) => {}, // 不是我们关心的结构体，继续处理下一个
                Err(_) => {
                    //self.debug_log(format!("处理DefId({:?})结构体时发生panic，已跳过", def_id));
                }
            }
        }
        
        //self.debug_log(format!("公共结构体收集完成，共 {} 个", self.pub_structs.len()));
    }
    
    /// 安全地获取字段类型，避免编译器内部panic
    fn get_field_type_safely(&self, field: &rustc_middle::ty::FieldDef) -> Option<rustc_middle::ty::Ty<'tcx>> {
        use std::panic::{self, AssertUnwindSafe};
        
        // 使用空的泛型参数列表
        let empty_substs = rustc_middle::ty::List::empty();
        
        // 使用catch_unwind捕获可能的panic
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            field.ty(self.tcx, empty_substs)
        }));
        
        match result {
            Ok(ty) => Some(ty),
            Err(_) => {
                // 记录错误并返回None
                //self.debug_log(format!("获取字段 {:?} 的类型时发生panic，已跳过", field.did));
                None
            }
        }
    }
    
    /// 检测符合pattern2的函数：pub结构体的pub字段传入impl的pub函数的unsafe操作
    fn detect_pattern2_matches(&mut self) {
        //self.debug_log("开始检测pattern2匹配...");
        
        // 遍历所有内部不安全函数
        for (&def_id, internal_unsafe) in &self.internal_unsafe_fns {
            // 只检查公共函数
            if self.is_public_fn(def_id) {
                let fn_name = self.get_fn_name(def_id);
                //self.debug_log(format!("检查公共函数: {}", fn_name));
                
                // 安全地获取MIR
                let body = match self.get_mir_safely(def_id) {
                    Some(body) => body,
                    None => {
                        //self.debug_log(format!("无法安全获取函数 {} 的MIR，跳过pattern2检测", fn_name));
                        continue;
                    }
                };
                
                // 检查此函数是否是某个结构体的impl方法
                if let Some(impl_self_ty) = self.get_impl_self_type(def_id) {
                    // 尝试获取结构体类型
                    if let Some(struct_def_id) = self.extract_struct_from_type(impl_self_ty) {
                        // 检查是否是我们记录的公共结构体
                        if let Some(struct_info) = self.pub_structs.get(&struct_def_id) {
                            let struct_name = self.get_fn_name(struct_def_id);
                            //self.debug_log(format!("函数 {} 是结构体 {} 的方法", fn_name, struct_name));
                            
                            // Debug点1: 输出识别到的pub结构体的pub字段
                            for (field_idx, field_info) in &struct_info.pub_fields {
                                //self.debug_log(format!("【Debug1】识别到pub结构体 {} 的pub字段: {}.{}", 
                                //              struct_name, struct_name, field_info.name));
                            }
                            
                            // 检查函数参数中是否有对该结构体的引用
                            let self_param = self.find_self_param(body);
                            if let Some(self_param) = self_param {
                                // Debug点2: 输出识别到的unsafe操作
                                for op in &internal_unsafe.unsafe_operations {
                                    //self.debug_log(format!("【Debug2】识别到unsafe操作: {}", op.operation_detail));
                                }
                                
                                // 分析每个不安全操作
                                let mut pattern2_ops = Vec::new();
                                
                                for op in &internal_unsafe.unsafe_operations {
                                    // 在函数体中查找结构体字段的访问
                                    let field_accesses = self.find_struct_field_accesses(body, self_param);
                                    
                                    // 检查不安全操作是否使用了结构体的公共字段
                                    for (field_idx, field_var, field_path) in &field_accesses {
                                        // 检查该字段是否是公共字段
                                        if let Some(field_info) = struct_info.pub_fields.get(field_idx) {
                                            // Debug点3: 输出检查字段是否用于unsafe操作的信息
                                            //self.debug_log(format!("【Debug3】检查字段 {}.{} (变量 _{}) 是否用于unsafe操作 {}", 
                                            //              struct_name, field_info.name, field_var, op.operation_detail));
                                            
                                            // 检查该字段是否传递给了不安全操作
                                            if self.is_var_used_in_unsafe_op(body, *field_var, &op.operation_detail) {
                                                // 检查变量是否经过净化操作
                                                if !self.is_var_sanitized(body, *field_var, def_id) {
                                                    let field_name = &field_info.name;
                                                    pattern2_ops.push((op.clone(), format!("{}.{}", struct_name, field_name)));
                                                    //self.debug_log(format!(
                                                        //"发现pattern2：结构体 {} 的公共字段 {} 传递给不安全操作 {}", 
                                                        //struct_name, field_name, op.operation_detail));
                                                } else {
                                                    //self.debug_log(format!(
                                                        //"结构体 {} 的字段 {} 经过净化，跳过", 
                                                        //struct_name, field_info.name));
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // 保存结果
                                if !pattern2_ops.is_empty() {
                                    self.pattern2_matches.insert(def_id, pattern2_ops);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        //self.debug_log(format!("pattern2检测完成，共发现 {} 个匹配函数", self.pattern2_matches.len()));
    }
    
    /// 查找函数中的self参数
    fn find_self_param(&self, body: &rustc_middle::mir::Body<'tcx>) -> Option<usize> {
        // 一般来说，self参数是第一个参数(索引1)
        if body.arg_count >= 1 {
            return Some(1);
        }
        None
    }
    
    /// 查找对结构体字段的访问
    /// 返回 (字段索引, 字段被赋值到的变量, 访问路径)
    fn find_struct_field_accesses(&self, body: &rustc_middle::mir::Body<'tcx>, self_param: usize) 
        -> Vec<(usize, usize, String)> {
        let mut field_accesses = Vec::new();
        // 用于跟踪变量之间的映射关系，记录哪些变量包含了结构体的哪个字段
        let mut var_to_field_map: HashMap<usize, (usize, String)> = HashMap::new();
        
        // 遍历所有基本块和语句
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    // 查找类似 *_3 = &((**_1).0) 的模式，其中_1是self参数
                    if let rustc_middle::mir::Rvalue::Ref(_, _, source_place) = rvalue {
                        // 检查是否访问了结构体字段
                        if let Some((base, proj)) = self.get_base_and_projection(source_place) {
                            if base.as_usize() == self_param {
                                // 检查投影是否包含字段访问
                                for proj_elem in proj {
                                    if let rustc_middle::mir::ProjectionElem::Field(field_index, _) = proj_elem {
                                        let field_idx = field_index.index();
                                        let dest_var = place.local.as_usize();
                                        let access_path = format!("(self).{}", field_idx);
                                        
                                        field_accesses.push((field_idx, dest_var, access_path.clone()));
                                        var_to_field_map.insert(dest_var, (field_idx, access_path.clone()));
                                        //self.debug_log(format!("发现字段引用访问: 字段 {} 被赋值到变量 _{}", field_idx, dest_var));
                                    }
                                }
                            }
                        }
                    }
                    // 检查直接使用字段的情况 (例如通过self.name.as_slice())
                    else if let rustc_middle::mir::Rvalue::Use(operand) = rvalue {
                        if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                            if let Some((base, proj)) = self.get_base_and_projection(source_place) {
                                if base.as_usize() == self_param {
                                    // 检查投影是否包含字段访问
                                    for proj_elem in proj {
                                        if let rustc_middle::mir::ProjectionElem::Field(field_index, _) = proj_elem {
                                            let field_idx = field_index.index();
                                            let dest_var = place.local.as_usize();
                                            let access_path = format!("(self).{}", field_idx);
                                            
                                            field_accesses.push((field_idx, dest_var, access_path.clone()));
                                            var_to_field_map.insert(dest_var, (field_idx, access_path.clone()));
                                            //self.debug_log(format!("发现字段直接使用: 字段 {} 被赋值到变量 _{}", field_idx, dest_var));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // 检查语句中的rvalue详细表示，寻找 deref_copy 模式
                    // 例如 _2 = deref_copy ((*_1).2: *const i32) 这种形式
                    let rvalue_str = format!("{:?}", rvalue);
                    if rvalue_str.contains("deref_copy") && rvalue_str.contains(&format!("_{})", self_param)) {
                        // 尝试提取字段索引
                        // 通常格式为 deref_copy ((*_1).2: *const i32)，我们需要提取"2"
                        if let Some(field_start) = rvalue_str.find(").") {
                            let field_substr = &rvalue_str[field_start+2..];
                            if let Some(field_end) = field_substr.find(":") {
                                if let Ok(field_idx) = field_substr[..field_end].trim().parse::<usize>() {
                                    let dest_var = place.local.as_usize();
                                    let access_path = format!("(self).{}", field_idx);
                                    
                                    field_accesses.push((field_idx, dest_var, access_path.clone()));
                                    var_to_field_map.insert(dest_var, (field_idx, access_path.clone()));
                                    //self.debug_log(format!("发现字段deref_copy访问: 字段 {} 被赋值到变量 _{}", field_idx, dest_var));
                                }
                            }
                        }
                    }
                    
                    // 跟踪变量间的复制关系，处理变量之间的数据流
                    let dest_var = place.local.as_usize();
                    if let rustc_middle::mir::Rvalue::Use(operand) = rvalue {
                        if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                            let source_var = source_place.local.as_usize();
                            // 先获取字段信息的副本，再插入新条目
                            let field_info_opt = var_to_field_map.get(&source_var).cloned();
                            if let Some((field_idx, access_path)) = field_info_opt {
                                var_to_field_map.insert(dest_var, (field_idx, access_path.clone()));
                                //self.debug_log(format!("变量传递: 变量 _{} (字段 {}) 被复制到变量 _{}", 
                                              //source_var, field_idx, dest_var));
                            }
                        }
                    }
                }
            }
            
            // 检查终结符中是否有对变量的解引用操作
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { destination, .. } = &terminator.kind {
                    let dest_var = destination.local.as_usize();
                    let terminator_str = format!("{:?}", terminator);
                    
                    // 检查是否有对结构体字段的解引用操作
                    if terminator_str.contains("deref") && terminator_str.contains(&format!("_{})", self_param)) {
                        // 类似于前面的代码，提取字段索引
                        if let Some(field_start) = terminator_str.find(").") {
                            let field_substr = &terminator_str[field_start+2..];
                            if let Some(field_end) = field_substr.find(":") {
                                if let Ok(field_idx) = field_substr[..field_end].trim().parse::<usize>() {
                                    let access_path = format!("(self).{}", field_idx);
                                    field_accesses.push((field_idx, dest_var, access_path.clone()));
                                    var_to_field_map.insert(dest_var, (field_idx, access_path.clone()));
                                    //self.debug_log(format!("发现终结符中的字段访问: 字段 {} 被赋值到变量 _{}", field_idx, dest_var));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 第二遍扫描，检查是否有对映射变量的解引用操作
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                // 检查是否存在对已知字段变量的解引用操作
                let terminator_str = format!("{:?}", terminator);
                for (&var, &(field_idx, ref _access_path)) in &var_to_field_map {
                    if terminator_str.contains(&format!("(*_{})", var)) {
                        //self.debug_log(format!("发现终结符中对字段变量的解引用: 变量 _{} (字段 {})", var, field_idx));
                        // 这里不需要添加新的field_accesses，因为我们已经有了对应的映射
                    }
                }
            }
            
            // 检查语句中是否有对映射变量的解引用操作
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    let rvalue_str = format!("{:?}", rvalue);
                    let dest_var = place.local.as_usize();
                    
                    // 检查是否存在从字段变量到解引用的直接数据流
                    if rvalue_str.contains("copy (*_") {
                        for (&var, &(field_idx, ref access_path)) in &var_to_field_map {
                            if rvalue_str.contains(&format!("copy (*_{})", var)) {
                                //self.debug_log(format!("发现语句中对字段变量的解引用: 变量 _{} (字段 {}) 被解引用到变量 _{}", 
                                              //var, field_idx, dest_var));
                                // 在这种情况下，我们也应该考虑将dest_var标记为对应字段的访问
                                field_accesses.push((field_idx, var, access_path.clone()));
                            }
                        }
                    }
                }
            }
        }
        
        field_accesses
    }
    
    /// 获取place的基本变量和投影
    fn get_base_and_projection(&self, place: &rustc_middle::mir::Place<'tcx>) 
        -> Option<(rustc_middle::mir::Local, &[rustc_middle::mir::PlaceElem<'tcx>])> {
        Some((place.local, place.projection.as_ref()))
    }
    
    /// 检查变量是否被用于不安全操作
    fn is_var_used_in_unsafe_op(&self, body: &rustc_middle::mir::Body<'tcx>, var: usize, op_detail: &str) -> bool {
        // 先检查操作描述中是否直接包含该变量
        let var_str = format!("_{}", var);
        if op_detail.contains(&var_str) {
            return true;
        }
        
        // 使用更全面的数据流分析
        let mut visited = HashSet::new();
        self.check_var_flows_to_unsafe_op(body, var, op_detail, &mut visited)
    }
    
    /// 检查变量是否通过数据流传递给了不安全操作
    fn check_var_flows_to_unsafe_op(&self, 
                                    body: &rustc_middle::mir::Body<'tcx>, 
                                    var: usize, 
                                    op_detail: &str,
                                    visited: &mut HashSet<usize>) -> bool {
        // 避免循环
        if !visited.insert(var) {
            return false;
        }
        
        // 直接检查操作描述
        let var_str = format!("_{}", var);
        if op_detail.contains(&var_str) {
            return true;
        }
        
        // 检查该变量的值是否传递给了其他变量
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    match rvalue {
                        rustc_middle::mir::Rvalue::Use(operand) => {
                            if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                                if source_place.local.as_usize() == var {
                                    // 该变量的值被复制/移动到另一个变量
                                    let dest_var = place.local.as_usize();
                                    if self.check_var_flows_to_unsafe_op(body, dest_var, op_detail, visited) {
                                        return true;
                                    }
                                }
                            }
                        },
                        rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                            if source_place.local.as_usize() == var {
                                // 取该变量的引用
                                let dest_var = place.local.as_usize();
                                
                                // 特殊处理：在不安全操作中检查数组引用
                                // 例如 &self.sysname 变成 &_5，然后传递给 from_utf8_unchecked
                                // 在操作描述中会看到 from_utf8_unchecked(_3)，但实际上_3是&_5
                                let ref_var_pattern = format!("_{}", dest_var);
                                
                                // 1. 首先检查目标变量是否直接用于不安全操作
                                if self.check_var_flows_to_unsafe_op(body, dest_var, op_detail, visited) {
                                    return true;
                                }
                                
                                // 2. 检查操作中是否有形如(_3)的模式，其中_3可能是&_5的引用
                                for (i, _) in op_detail.match_indices("(") {
                                    if i + 1 < op_detail.len() {
                                        let end_idx = op_detail[i..].find(")").map(|pos| i + pos).unwrap_or(op_detail.len());
                                        let arg_str = &op_detail[i+1..end_idx];
                                        
                                        // 检查参数中是否有目标变量的引用
                                        if arg_str.contains(&ref_var_pattern) {
                                            //self.debug_log(format!("变量 {} 的引用 {} 被用作参数传递给不安全操作: {}", 
                                                          //var, dest_var, op_detail));
                                            return true;
                                        }
                                    }
                                }
                                
                                // 3. 检查参数变量和操作是否有关联
                                // 寻找函数调用
                                for other_block in body.basic_blocks.iter() {
                                    if let Some(terminator) = &other_block.terminator {
                                        if let TerminatorKind::Call { func: _, args, destination, .. } = &terminator.kind {
                                            // 检查任何参数是否引用了该变量
                                            for arg in args {
                                                if let Operand::Copy(place) | Operand::Move(place) = &arg.node {
                                                    let place_str = format!("{:?}", place);
                                                    if place_str.contains(&ref_var_pattern) {
                                                        // 如果该参数被传递给不安全函数调用，特别是如果这个调用是目标不安全操作
                                                        let term_str = format!("{:?}", terminator);
                                                        if term_str.contains(op_detail) {
                                                            //self.debug_log(format!("字段引用 {} 作为参数传递给调用: {}", 
                                                                      //ref_var_pattern, term_str));
                                                            return true;
                                                        }
                                                        
                                                        // 如果不是同一个调用，检查结果是否流向不安全操作
                                                        let result_var = destination.local.as_usize();
                                                        if self.check_var_flows_to_unsafe_op(body, result_var, op_detail, visited) {
                                                            return true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // 4. 对于Utsname::name这样的特殊情况，直接检查类型和操作
                                // 如果操作是 from_utf8_unchecked 且变量是数组引用
                                if op_detail.contains("from_utf8_unchecked") {
                                    // 检查变量类型是否是数组引用
                                    if let Some(ty) = body.local_decls.get(rustc_middle::mir::Local::from_usize(var)) {
                                        let ty_str = format!("{:?}", ty.ty);
                                        if ty_str.contains("[u8;") || ty_str.contains("&[u8") {
                                            //self.debug_log(format!("数组引用变量 {} (类型 {}) 流向 from_utf8_unchecked", var, ty_str));
                                            return true;
                                        }
                                    }
                                    
                                    // 检查字段的引用是否被传递给另一个变量
                                    for other_stmt in &block_data.statements {
                                        if let rustc_middle::mir::StatementKind::Assign(box (other_place, other_rvalue)) = &other_stmt.kind {
                                            let other_rvalue_str = format!("{:?}", other_rvalue);
                                            if other_rvalue_str.contains(&ref_var_pattern) {
                                                // 字段引用被传递给其他变量
                                                let next_var = other_place.local.as_usize();
                                                if self.check_var_flows_to_unsafe_op(body, next_var, op_detail, visited) {
                                                    return true;
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // 5. 继续跟踪传递关系
                                // 查找所有将引用转换为值的地方
                                for other_block in body.basic_blocks.iter() {
                                    for other_stmt in &other_block.statements {
                                        if let rustc_middle::mir::StatementKind::Assign(box (other_place, other_rvalue)) = &other_stmt.kind {
                                            // 检查是否是从引用加载值
                                            if let rustc_middle::mir::Rvalue::Use(Operand::Copy(ref_place) | Operand::Move(ref_place)) = other_rvalue {
                                                // 检查加载的是否是我们跟踪的变量
                                                let ref_place_str = format!("{:?}", ref_place);
                                                if ref_place_str.contains(&ref_var_pattern) {
                                                    // 被加载到的新变量
                                                    let new_var = other_place.local.as_usize();
                                                    if self.check_var_flows_to_unsafe_op(body, new_var, op_detail, visited) {
                                                        return true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        _ => {}
                    }
                }
            }
            
            // 检查终结器中的函数调用
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { func: _, args, .. } = &terminator.kind {
                    for arg in args {
                        if let Operand::Copy(place) | Operand::Move(place) = &arg.node {
                            if place.local.as_usize() == var {
                                // 如果变量作为参数传递给函数，就认为它可能流入不安全操作
                                // 简化处理，不再尝试追踪destination
                                //self.debug_log(format!("变量 {} 作为参数传递给函数，可能流入不安全操作", var));
                                return true;
                            }
                        }
                    }
                    
                    // 检查是否有引用被传递给函数调用
                    let term_str = format!("{:?}", terminator);
                    let var_ref_pattern = format!("&_{}", var);
                    if term_str.contains(&var_ref_pattern) {
                        //self.debug_log(format!("变量 {} 的引用被传递给函数调用: {}", var, term_str));
                        return true;
                    }
                    
                    // 特殊处理from_utf8_unchecked调用
                    if term_str.contains("from_utf8_unchecked") && term_str.contains(&format!("_{}", var)) {
                        //self.debug_log(format!("变量 {} 被传递给 from_utf8_unchecked 调用: {}", var, term_str));
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// 检查变量是否经过了净化函数
    /// 净化函数是指函数名中包含"valid"、"check"、"is_"等关键词的函数
    fn is_var_sanitized(&self, body: &rustc_middle::mir::Body<'tcx>, var: usize, def_id: DefId) -> bool {
        let fn_name = self.get_fn_name(def_id);
        //self.debug_log(format!("检查变量 {} 在函数 {} 中是否经过净化", var, fn_name));
        
        // 首先检查是否有sanitizer函数调用
        let mut sanitizer_found = false;
        
        // 遍历所有基本块
        for (block_idx, block_data) in body.basic_blocks.iter().enumerate() {
            // 检查终结符中的函数调用
            if let Some(terminator) = &block_data.terminator {
                match &terminator.kind {
                    // 检查函数调用
                    TerminatorKind::Call { func, args, .. } => {
                        // 处理函数调用
                        if let Operand::Constant(constant) = func {
                            if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) = constant.const_.ty().kind() {
                                let callee_fn_name = self.get_fn_name(*callee_def_id);
                                //self.debug_log(format!("  检查函数调用: {} 在基本块 {}", callee_fn_name, block_idx));
                                
                                // 检查函数名是否含有净化关键词
                                if self.is_sanitizer_function_name(&callee_fn_name) {
                                    // 检查参数中是否包含目标变量
                                    for arg in args.iter() {
                                        if let Operand::Copy(place) | Operand::Move(place) = &arg.node {
                                            if place.local.as_usize() == var {
                                                //self.debug_log(format!("  变量 {} 作为参数传递给净化函数 {}", var, callee_fn_name));
                                                sanitizer_found = true;
                                            }
                                        }
                                    }
                                    
                                    // 如果没有直接作为参数，检查函数调用结果是否在条件判断中使用
                                    if !sanitizer_found && self.check_result_used_in_condition(body, block_idx) {
                                        //self.debug_log(format!("  净化函数 {} 的结果用于条件判断", callee_fn_name));
                                        sanitizer_found = true;
                                    }
                                }
                            }
                        }
                    },
                    // 检查SwitchInt终结符，表示变量被用于match或if条件判断
                    TerminatorKind::SwitchInt { discr, .. } => {
                        if let Operand::Copy(place) | Operand::Move(place) = discr {
                            // 直接检查条件变量
                            if place.local.as_usize() == var {
                                //self.debug_log(format!("  变量 {} 直接用于条件判断", var));
                                sanitizer_found = true;
                            } else {
                                // 检查条件表达式中是否包含目标变量
                                let terminator_str = format!("{:?}", terminator);
                                if terminator_str.contains(&format!("_{}", var)) {
                                    //self.debug_log(format!("  变量 {} 在条件表达式中: {}", var, terminator_str));
                                    sanitizer_found = true;
                                }
                                
                                // 特别处理：检查条件是sanitizer函数调用的结果
                                let cond_var = place.local.as_usize();
                                if self.is_var_from_sanitizer_call(body, cond_var, var) {
                                    //self.debug_log(format!("  条件变量 {} 来自验证变量 {} 的sanitizer函数调用", cond_var, var));
                                    sanitizer_found = true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
            
            // 检查块中的语句
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (_place, rvalue)) = &statement.kind {
                    let rvalue_str = format!("{:?}", rvalue);
                    
                    // 检查rvalue中是否包含目标变量和sanitizer关键词
                    if rvalue_str.contains(&format!("_{}", var)) && self.is_sanitizer_function_name(&rvalue_str) {
                        //self.debug_log(format!("  变量 {} 在sanitizer表达式中: {}", var, rvalue_str));
                        sanitizer_found = true;
                    }
                }
            }
        }
        
        if sanitizer_found {
            //self.debug_log(format!("  变量 {} 在函数 {} 中已经过净化", var, fn_name));
            return true;
        }
        
        // 如果是特殊函数，检查函数名本身
        if self.is_sanitizer_function_name(&fn_name) {
            // 是检查是否是参数，1~arg_count的变量是参数
            if var > 0 && var <= body.arg_count {
                //self.debug_log(format!("  变量 {} 是净化函数 {} 的参数", var, fn_name));
                return true;
            }
        }
        
        //self.debug_log(format!("  变量 {} 在函数 {} 中未检测到净化操作", var, fn_name));
        false
    }

    /// 检查变量是否来自sanitizer函数调用
    fn is_var_from_sanitizer_call(&self, body: &rustc_middle::mir::Body<'tcx>, var: usize, target_var: usize) -> bool {
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    if place.local.as_usize() == var {
                        // 变量是否来自函数调用结果
                        let rvalue_str = format!("{:?}", rvalue);
                        
                        // 检查rvalue是否包含目标变量和安全检查关键词
                        if rvalue_str.contains(&format!("_{}", target_var)) && 
                           (rvalue_str.contains("is_ascii") || rvalue_str.contains("from_utf8")) {
                            //self.debug_log(format!("  变量 {} 来自对变量 {} 的安全检查: {}", 
                                           //var, target_var, rvalue_str));
                            return true;
                        }
                    }
                }
            }
            
            // 检查终结符中的函数调用
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { func, args, destination, .. } = &terminator.kind {
                    if destination.local.as_usize() == var {
                        // 函数调用结果赋值给了目标变量
                        
                        // 检查是否是sanitizer函数
                        if let Operand::Constant(constant) = func {
                            if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) = constant.const_.ty().kind() {
                                let callee_fn_name = self.get_fn_name(*callee_def_id);
                                
                                if self.is_sanitizer_function_name(&callee_fn_name) {
                                    // 检查参数中是否包含目标变量
                                    for arg in args.iter() {
                                        if let Operand::Copy(place) | Operand::Move(place) = &arg.node {
                                            if place.local.as_usize() == target_var {
                                                //self.debug_log(format!("  变量 {} 是对变量 {} 的sanitizer函数 {} 调用结果", 
                                                               //var, target_var, callee_fn_name));
                                                return true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        false
    }
    
    /// 检查函数调用结果是否用于条件判断
    fn check_result_used_in_condition(&self, body: &rustc_middle::mir::Body<'tcx>, block_idx: usize) -> bool {
        // 获取当前块的终结符
        if let Some(block_data) = body.basic_blocks.get(rustc_middle::mir::BasicBlock::from_usize(block_idx)) {
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { destination, .. } = &terminator.kind {
                    let result_var = destination.local.as_usize();
                    
                    // 遍历所有块，查找使用结果变量的条件判断
                    for next_block_data in body.basic_blocks.iter() {
                        if let Some(next_terminator) = &next_block_data.terminator {
                            if let TerminatorKind::SwitchInt { discr, .. } = &next_terminator.kind {
                                if let Operand::Copy(place) | Operand::Move(place) = discr {
                                    if place.local.as_usize() == result_var {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        false
    }

    /// 判断函数名是否为净化函数（包含"valid"、"check"、"is_"等关键词）
    fn is_sanitizer_function_name(&self, name: &str) -> bool {
        let sanitizer_keywords = [
            "valid", "check", "is_", "has_", "ensure", "verify", "safe","is_null"
        ];
        
        // 排除本身是unsafe函数的情况
        let unsafe_functions = [
            "unchecked", "raw_parts", "transmute", "copy_nonoverlapping", 
            "write", "read", "ptr", "get_unchecked", "from_utf8_unchecked"
        ];
        
        // 如果函数名包含unsafe关键词，不应该算作sanitizer
        let name_lower = name.to_lowercase();
        
        // 首先检查是否包含unsafe关键词
        for unsafe_keyword in &unsafe_functions {
            if name_lower.contains(unsafe_keyword) {
                //self.debug_log(format!("函数名 '{}' 包含unsafe关键词 '{}', 不视为sanitizer", name, unsafe_keyword));
                return false;
            }
        }
        
        // 特殊处理：将from_utf8视为sanitizer，但from_utf8_unchecked不是
        if name_lower.contains("from_utf8") && !name_lower.contains("unchecked") {
            //self.debug_log(format!("函数名 '{}' 匹配sanitizer函数 'from_utf8'", name));
            return true;
        }
        
        // 然后检查是否包含sanitizer关键词
        for keyword in &sanitizer_keywords {
            if name_lower.contains(keyword) {
                //self.debug_log(format!("函数名 '{}' 匹配sanitizer关键词 '{}'", name, keyword));
                return true;
            }
        }
        
        false
    }
    
    /// 从类型中提取结构体DefId
    fn extract_struct_from_type(&self, ty: rustc_middle::ty::Ty<'tcx>) -> Option<DefId> {
        match ty.kind() {
            rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_struct() => Some(adt_def.did()),
            rustc_middle::ty::TyKind::Ref(_, inner_ty, _) => self.extract_struct_from_type(*inner_ty),
            // 新增对数组类型的支持
            rustc_middle::ty::TyKind::Array(elem_ty, _) => self.extract_struct_from_type(*elem_ty),
            rustc_middle::ty::TyKind::Slice(elem_ty) => self.extract_struct_from_type(*elem_ty),
            _ => None,
        }
    }
    
    /// 获取方法的实现类型（如果是impl方法）
    fn get_impl_self_type(&self, def_id: DefId) -> Option<rustc_middle::ty::Ty<'tcx>> {
        if let Some(impl_def_id) = self.tcx.impl_of_method(def_id) {
            // 获取impl的自身类型
            return Some(self.tcx.type_of(impl_def_id).skip_binder());
        }
        None
    }

    /// 识别所有"模式载体"函数
    fn identify_pattern_carriers(&mut self) {
        //self.debug_log("开始识别模式载体...");
        
        // 遍历所有内部不安全函数，不限于公共函数
        for (&def_id, internal_unsafe) in &self.internal_unsafe_fns {
            let fn_name = self.get_fn_name(def_id);
            
            //self.debug_log(format!("检查函数: {}", fn_name));
            
            // 处理函数参数的情况
            let mut pattern1_ops = Vec::new();
            let mut tainted_param_idx = 0;
            
            // 获取MIR以分析参数
            if let Some(body) = self.get_mir_safely(def_id) {
                let param_count = body.arg_count;
                
                // 判断是否是结构体impl方法
                let is_method = param_count > 0 && self.get_impl_self_type(def_id).is_some();
                let self_param = if is_method { Some(1) } else { None };
                
                // 检查所有不安全操作，是否有直接或间接使用参数的情况
                for op in &internal_unsafe.unsafe_operations {
                    //self.debug_log(format!("检查不安全操作: {}", op.operation_detail));
                    
                    // 1. 检查解引用参数的情况
                    if op.operation_detail.starts_with("*_") {
                        // 从操作详情中提取变量编号
                        if let Some(var_number) = self.extract_var_number(&op.operation_detail) {
                            // 检查变量编号是否在参数范围内，但不是self参数
                            let is_pattern1 = self.is_non_self_param_or_copy(body, var_number, param_count, self_param);
                            if is_pattern1 {
                                // 检查变量是否经过净化操作
                                if !self.is_var_sanitized(body, var_number, def_id) {
                                    pattern1_ops.push(op.clone());
                                    // 查找源参数
                                    if let Some(source_param) = self.find_source_parameter(body, var_number, param_count, self_param) {
                                        tainted_param_idx = source_param;
                                        //self.debug_log(format!("发现从参数 {} 到不安全操作的传播路径", source_param));
                                    }
                                }
                            }
                        }
                    } 
                    // 2. 检查函数调用参数的情况
                    else if op.operation_detail.contains("(") && op.operation_detail.contains(")") {
                        if let Some(source_param) = self.check_function_call_tainted_param(
                            &op.operation_detail, body, param_count, self_param, def_id) {
                            pattern1_ops.push(op.clone());
                            tainted_param_idx = source_param;
                            //self.debug_log(format!("发现从参数 {} 到函数调用的传播路径", source_param));
                        }
                    }
                }
            }
            
            // 如果找到pattern1操作，保存结果为模式载体
            if !pattern1_ops.is_empty() && tainted_param_idx > 0 {
                let carrier = PatternCarrier {
                    def_id,
                    tainted_param_idx,
                    unsafe_ops: pattern1_ops.clone(),
                    pattern_type: 1, // Pattern1
                };
                self.pattern_carriers.insert(def_id, carrier);
                //self.debug_log(format!("函数 {} 被识别为Pattern1载体，污染参数索引: {}", fn_name, tainted_param_idx));
            }
        }
        
        //self.debug_log(format!("模式载体识别完成，共发现 {} 个", self.pattern_carriers.len()));
    }
    
    /// 查找变量的源参数
    fn find_source_parameter(&self, 
                           body: &rustc_middle::mir::Body<'tcx>, 
                           var_num: usize, 
                           param_count: usize,
                           self_param: Option<usize>) -> Option<usize> {
        // 如果变量本身是参数
        if var_num > 0 && var_num <= param_count {
            // 确保不是self参数
            if let Some(self_idx) = self_param {
                if var_num != self_idx {
                    return Some(var_num);
                }
            } else {
                return Some(var_num);
            }
        }
        
        // 否则回溯查找变量的来源
        let mut visited = HashSet::new();
        self.find_source_parameter_with_visited(body, var_num, param_count, self_param, &mut visited)
    }

    /// 递归回溯查找变量的源参数
    fn find_source_parameter_with_visited(&self, 
                                        body: &rustc_middle::mir::Body<'tcx>, 
                                        var_num: usize, 
                                        param_count: usize,
                                        self_param: Option<usize>,
                                        visited: &mut HashSet<usize>) -> Option<usize> {
        // 防止循环递归
        if !visited.insert(var_num) {
            return None;
        }
        
        // 检查变量是否是参数
        if var_num > 0 && var_num <= param_count {
            if let Some(self_idx) = self_param {
                if var_num != self_idx {
                    // 是普通参数
                    return Some(var_num);
                }
            } else {
                // 没有self参数，所有参数都是普通参数
                return Some(var_num);
            }
        }
        
        // 检查赋值来源
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) = &statement.kind {
                    // 检查赋值目标是否是当前变量
                    if place.local.as_usize() == var_num {
                        match rvalue {
                            rustc_middle::mir::Rvalue::Use(operand) => {
                                if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                                    let source_local = source_place.local.as_usize();
                                    
                                    // 检查源变量是否是参数
                                    if source_local > 0 && source_local <= param_count {
                                        // 确保不是self参数
                                        if let Some(self_idx) = self_param {
                                            if source_local != self_idx {
                                                return Some(source_local);
                                            }
                                        } else {
                                            return Some(source_local);
                                        }
                                    }
                                    
                                    // 递归检查源变量
                                    if let Some(source_param) = self.find_source_parameter_with_visited(
                                        body, source_local, param_count, self_param, visited) {
                                        return Some(source_param);
                                    }
                                }
                            },
                            rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                                let source_local = source_place.local.as_usize();
                                
                                // 检查源变量是否是参数
                                if source_local > 0 && source_local <= param_count {
                                    // 确保不是self参数
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return Some(source_local);
                                        }
                                    } else {
                                        return Some(source_local);
                                    }
                                }
                                
                                // 递归检查
                                if let Some(source_param) = self.find_source_parameter_with_visited(
                                    body, source_local, param_count, self_param, visited) {
                                    return Some(source_param);
                                }
                            },
                            // 添加对裸指针转换的检测
                            rustc_middle::mir::Rvalue::RawPtr(_, source_place) => {
                                let source_local = source_place.local.as_usize();
                                
                                // 检查源变量是否是参数
                                if source_local > 0 && source_local <= param_count {
                                    // 确保不是self参数
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return Some(source_local);
                                        }
                                    } else {
                                        return Some(source_local);
                                    }
                                }
                                
                                if let Some(source_param) = self.find_source_parameter_with_visited(
                                    body, source_local, param_count, self_param, visited) {
                                    return Some(source_param);
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
        
        None
    }
    
    /// 检查函数调用中的污染参数
    fn check_function_call_tainted_param(&self, 
                                      op_detail: &str, 
                                      body: &rustc_middle::mir::Body<'tcx>, 
                                      param_count: usize,
                                      self_param: Option<usize>,
                                      def_id: DefId) -> Option<usize> {
        //self.debug_log(format!("检查函数调用中的污染参数: {} (函数ID: {:?})", op_detail, def_id));
        
        // 不再基于函数名称判断是否为不安全操作
        // 所有在extract_unsafe_operations中提取出的操作都应被视为不安全的
        let is_unsafe_op = true;
        
        if let Some(start_pos) = op_detail.find('(') {
            if let Some(end_pos) = op_detail.rfind(')') {
                if start_pos < end_pos {
                    let args_str = &op_detail[start_pos+1..end_pos];
                    //self.debug_log(format!("  解析参数字符串: {}", args_str));
                    
                    // 分割参数
                    for arg in args_str.split(',') {
                        let arg = arg.trim();
                        //self.debug_log(format!("  检查参数: {}", arg));
                        
                        if arg.starts_with("_") {
                            // 尝试提取参数编号
                            if let Ok(arg_num) = arg[1..].parse::<usize>() {
                                //self.debug_log(format!("  参数编号: {}", arg_num));
                                
                                // 检查是否是非self参数或从非self参数复制
                                if self.is_non_self_param_or_copy(body, arg_num, param_count, self_param) {
                                    //self.debug_log(format!("  变量 {} 是非self参数或从非self参数复制", arg_num));
                                    
                                    // 检查变量是否经过净化操作
                                    if !self.is_var_sanitized(body, arg_num, def_id) {
                                        // 找出源参数
                                        if let Some(source_param) = self.find_source_parameter(body, arg_num, param_count, self_param) {
                                            //self.debug_log(format!("  变量 {} 未经过净化，来源于参数 {}", arg_num, source_param));
                                            return Some(source_param);
                                        }
                                    } else {
                                        //self.debug_log(format!("  变量 {} 在函数调用中已经过净化，跳过", arg_num));
                                    }
                                } else {
                                    //self.debug_log(format!("  变量 {} 不是非self参数或从非self参数复制", arg_num));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        //self.debug_log("  函数调用不匹配污染参数条件");
        None
    }
    
    /// 执行过程间分析
    fn perform_interprocedural_analysis(&mut self) {
        //self.debug_log("开始执行过程间分析...");
        
        // 首先清空pattern_carriers，确保它包含最新的carrier
        self.pattern_carriers.clear();
        self.identify_pattern_carriers();
        
        // 第一阶段：对所有模式载体进行处理
        let mut matches_to_add = Vec::new();
        let mut intermediate_carriers = Vec::new();
        
        // 收集模式载体和它们的调用者
        let mut carrier_callers = Vec::new();
        let carrier_def_ids: Vec<DefId> = self.pattern_carriers.keys().cloned().collect();
        
        for carrier_def_id in carrier_def_ids {
            let carrier = self.pattern_carriers.get(&carrier_def_id).unwrap().clone();
            
            // 使用反向调用图找到所有调用者
            if let Some(callers) = self.reverse_call_graph.get(&carrier_def_id) {
                for &caller_def_id in callers {
                    carrier_callers.push((caller_def_id, carrier_def_id, carrier.clone()));
                }
            }
        }
        
        // 处理直接的调用关系
        for (caller_def_id, carrier_def_id, carrier) in carrier_callers {
            // 检查调用者是否是公共函数
            if self.is_public_fn(caller_def_id) {
                // 分析调用点
                if let Some(body) = self.get_mir_safely(caller_def_id) {
                    // 查找调用点并映射参数
                    if let Some(call_arg) = self.find_caller_arg_local(
                        body, carrier_def_id, carrier.tainted_param_idx) {
                        
                        let param_count = body.arg_count;
                        let is_method = param_count > 0 && self.get_impl_self_type(caller_def_id).is_some();
                        let self_param = if is_method { Some(1) } else { None };
                        
                        // 检查调用者将什么参数传递给了载体的污染参数
                        if self.is_non_self_param_or_copy(body, call_arg, param_count, self_param) {
                            // 检查参数是否经过净化
                            if !self.is_var_sanitized(body, call_arg, caller_def_id) {
                                // 构建调用路径并检查是否有效
                                let mut call_path = vec![caller_def_id];
                                let path_valid = self.build_call_path(carrier_def_id, &mut call_path);
                                
                                // 只有当路径有效时，才添加匹配
                                if path_valid {
                                    // 找到了过程间模式匹配
                                    let interprocedural_match = InterproceduralMatch {
                                        pub_fn_id: caller_def_id,
                                        call_path,
                                        unsafe_ops: carrier.unsafe_ops.clone(),
                                        base_pattern: carrier.pattern_type,
                                    };
                                    
                                    matches_to_add.push((caller_def_id, interprocedural_match));
                                }
                            }
                        }
                    }
                }
            } else {
                // 对于非公共函数的调用者，记录为中间载体
                if let Some(body) = self.get_mir_safely(caller_def_id) {
                    if let Some(call_arg) = self.find_caller_arg_local(
                        body, carrier_def_id, carrier.tainted_param_idx) {
                        
                        let param_count = body.arg_count;
                        let is_method = param_count > 0 && self.get_impl_self_type(caller_def_id).is_some();
                        let self_param = if is_method { Some(1) } else { None };
                        
                        // 检查调用者将什么参数传递给了载体的污染参数
                        if self.is_non_self_param_or_copy(body, call_arg, param_count, self_param) {
                            // 检查参数是否经过净化
                            if !self.is_var_sanitized(body, call_arg, caller_def_id) {
                                if let Some(source_param) = self.find_source_parameter(body, call_arg, param_count, self_param) {
                                    // 创建中间载体
                                    let intermediate_carrier = PatternCarrier {
                                        def_id: caller_def_id,
                                        tainted_param_idx: source_param,
                                        unsafe_ops: carrier.unsafe_ops.clone(),
                                        pattern_type: carrier.pattern_type,
                                    };
                                    // 添加到中间载体列表
                                    intermediate_carriers.push((caller_def_id, intermediate_carrier));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 添加第一阶段发现的匹配和中间载体
        for (def_id, match_info) in matches_to_add {
            self.interprocedural_matches.insert(def_id, match_info);
        }
        
        for (def_id, carrier) in intermediate_carriers {
            self.pattern_carriers.insert(def_id, carrier);
        }
        
        // 第二阶段：递归分析中间调用链
        let mut iteration = 0;
        let max_iterations = 10; // 防止无限循环
        let mut found_new_match = true;
        
        while found_new_match && iteration < max_iterations {
            found_new_match = false;
            iteration += 1;
            
            // 收集当前所有模式载体和已匹配结果
            let current_carriers = self.pattern_carriers.clone();
            let current_matches = self.interprocedural_matches.clone();
            let mut new_matches = Vec::new();
            let mut new_carriers = Vec::new();
            
            // 创建一个包含所有已处理函数的集合，用于跳过检查
            let mut processed_fns = HashSet::new();
            
            // 添加已经匹配或作为载体的函数
            for &def_id in current_matches.keys() {
                processed_fns.insert(def_id);
            }
            
            for &def_id in current_carriers.keys() {
                processed_fns.insert(def_id);
            }
            
            for (carrier_def_id, carrier) in current_carriers {
                // 跳过已经作为匹配结果的载体
                if current_matches.contains_key(&carrier_def_id) {
                    continue;
                }
                
                // 使用反向调用图找到所有调用者
                if let Some(callers) = self.reverse_call_graph.get(&carrier_def_id) {
                    for &caller_def_id in callers {
                        // 跳过已经分析过的函数
                        if processed_fns.contains(&caller_def_id) {
                            continue;
                        }
                        
                        // 标记当前函数为已处理
                        processed_fns.insert(caller_def_id);
                        
                        // 检查调用者
                        if let Some(body) = self.get_mir_safely(caller_def_id) {
                            if let Some(call_arg) = self.find_caller_arg_local(
                                body, carrier_def_id, carrier.tainted_param_idx) {
                                
                                let param_count = body.arg_count;
                                let is_method = param_count > 0 && self.get_impl_self_type(caller_def_id).is_some();
                                let self_param = if is_method { Some(1) } else { None };
                                
                                if self.is_non_self_param_or_copy(body, call_arg, param_count, self_param) {
                                    if !self.is_var_sanitized(body, call_arg, caller_def_id) {
                                        // 检查是否是公共函数
                                        if self.is_public_fn(caller_def_id) {
                                            // 构建完整的调用链并检查是否有效
                                            let mut call_path = vec![caller_def_id];
                                            let path_valid = self.build_call_path(carrier_def_id, &mut call_path);
                                            
                                            // 只有当路径有效时，才添加匹配
                                            if path_valid {
                                                let interprocedural_match = InterproceduralMatch {
                                                    pub_fn_id: caller_def_id,
                                                    call_path,
                                                    unsafe_ops: carrier.unsafe_ops.clone(),
                                                    base_pattern: carrier.pattern_type,
                                                };
                                                
                                                new_matches.push((caller_def_id, interprocedural_match));
                                                found_new_match = true;
                                            }
                                        } else if let Some(source_param) = self.find_source_parameter(body, call_arg, param_count, self_param) {
                                            // 创建新的中间载体
                                            let new_carrier = PatternCarrier {
                                                def_id: caller_def_id,
                                                tainted_param_idx: source_param,
                                                unsafe_ops: carrier.unsafe_ops.clone(),
                                                pattern_type: carrier.pattern_type,
                                            };
                                            
                                            new_carriers.push((caller_def_id, new_carrier));
                                            found_new_match = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // 添加新发现的匹配和中间载体
            for (def_id, match_info) in new_matches {
                self.interprocedural_matches.insert(def_id, match_info);
            }
            
            for (def_id, carrier) in new_carriers {
                self.pattern_carriers.insert(def_id, carrier);
            }
        }
        
        //self.debug_log(format!("过程间分析完成，共发现 {} 个匹配", self.interprocedural_matches.len()));
    }
    
    /// 构建完整的调用路径，并检查路径中是否有除起始节点外的其他公共函数
    /// 返回 true 表示路径有效（不包含其他公共函数），false 表示路径无效（包含其他公共函数）
    fn build_call_path(&self, start_def_id: DefId, call_path: &mut Vec<DefId>) -> bool {
        // 首先添加起始函数
        call_path.push(start_def_id);
        
        // 检查当前函数是否是公共函数
        let is_pub = self.is_public_fn(start_def_id);
        
        // 如果当前函数是模式载体，并且它调用了其他模式载体，继续递归构建路径
        if let Some(carrier) = self.pattern_carriers.get(&start_def_id) {
            // 查找此载体的调用对象
            let carriers: Vec<DefId> = self.pattern_carriers.keys().cloned().collect();
            for target_def_id in carriers {
                // 跳过自己
                if target_def_id == start_def_id {
                    continue;
                }
                
                // 检查是否有调用关系
                if let Some(callees) = self.call_graph.get(&start_def_id) {
                    if callees.contains(&target_def_id) {
                        // 检查参数传递
                        if let Some(body) = self.get_mir_safely(start_def_id) {
                            if let Some(target_carrier) = self.pattern_carriers.get(&target_def_id) {
                                if let Some(_) = self.find_caller_arg_local(
                                    body, target_def_id, target_carrier.tainted_param_idx) {
                                    // 递归构建路径，如果子路径无效，则整个路径无效
                                    let sub_path_valid = self.build_call_path(target_def_id, call_path);
                                    if !sub_path_valid {
                                        return false;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 如果当前函数不是起始节点，并且是公共函数，则路径无效
        if start_def_id != call_path[0] && is_pub {
            return false;
        }
        
        // 路径有效
        true
    }
    
    /// 在调用者中找到对应模式载体污染参数的变量
    fn find_caller_arg_local(&self, 
                           body: &rustc_middle::mir::Body<'tcx>,
                           callee_def_id: DefId,
                           callee_param_idx: usize) -> Option<usize> {
        // 遍历所有基本块查找函数调用
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { func, args, .. } = &terminator.kind {
                    // 检查是否调用目标函数
                    if let Operand::Constant(constant) = func {
                        if let rustc_middle::ty::TyKind::FnDef(func_def_id, _) = constant.const_.ty().kind() {
                            if *func_def_id == callee_def_id {
                                // 找到了调用点，检查参数索引是否有效
                                let param_idx = callee_param_idx - 1; // 转为0-based索引
                                if param_idx < args.len() {
                                    // 获取参数变量
                                    if let Operand::Copy(place) | Operand::Move(place) = &args[param_idx].node {
                                        //self.debug_log(format!(
                                           // "找到调用点: 参数索引 {} 对应变量 _{}",
                                            //callee_param_idx, place.local.as_usize()));
                                        return Some(place.local.as_usize());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}