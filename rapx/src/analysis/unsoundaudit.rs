#[rustversion::since(1.96)]
use rustc_abi::VariantIdx;
use rustc_hir::Node;
use rustc_hir::{
    def_id::DefId,
    intravisit::{self, Visitor},
    BlockCheckMode, BodyId,
};
use rustc_middle::hir::nested_filter;
use rustc_middle::mir::{Operand, TerminatorKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
#[rustversion::before(1.96)]
use rustc_target::abi::VariantIdx;
use std::collections::{HashMap, HashSet, VecDeque};

use rustc_hir::def_id::LocalDefId;
use rustc_middle::middle::privacy::Level;

// Pattern 4 imports (HIR based)
use rustc_hir::{Expr, ExprKind, QPath};

use crate::utils::log::{span_to_filename, span_to_line_number};
use crate::utils::unit_output::{
    atomic_write, build_verification_receipt_from_environment, embedded_rap_compiler_commit,
    validate_commit_hash, BuildVerificationReceipt,
};
use crate::{rap_error, rap_info};
use serde_json::{json, Value};
use std::env;
use std::path::PathBuf;

#[rustversion::before(1.83)]
macro_rules! fn_sig_is_unsafe {
    ($signature:expr) => {
        ($signature).unsafety() == rustc_hir::Unsafety::Unsafe
    };
}

#[rustversion::since(1.83)]
macro_rules! fn_sig_is_unsafe {
    ($signature:expr) => {
        ($signature).safety() == rustc_hir::Safety::Unsafe
    };
}

#[rustversion::before(1.83)]
macro_rules! mir_call_arg_operand {
    ($argument:expr) => {
        $argument
    };
}

#[rustversion::since(1.83)]
macro_rules! mir_call_arg_operand {
    ($argument:expr) => {
        &($argument).node
    };
}

#[rustversion::before(1.83)]
macro_rules! mir_raw_ptr_rvalue {
    ($mutability:pat, $place:pat) => {
        rustc_middle::mir::Rvalue::AddressOf($mutability, $place)
    };
}

#[rustversion::since(1.83)]
macro_rules! mir_raw_ptr_rvalue {
    ($mutability:pat, $place:pat) => {
        rustc_middle::mir::Rvalue::RawPtr($mutability, $place)
    };
}

fn build_verification_receipt_value(receipt: &BuildVerificationReceipt) -> Value {
    json!({
        "schema": "rap-build-verification-receipt-v1",
        "challenge": receipt.challenge,
        "challenge_input_sha256": receipt.challenge_input_sha256,
        "exact_toolchain": receipt.exact_toolchain,
        "cargo_path": receipt.cargo_path,
        "cargo_sha256": receipt.cargo_sha256,
        "cargo_identity": {
            "component": receipt.cargo_identity.component,
            "release": receipt.cargo_identity.release,
            "commit_hash": receipt.cargo_identity.commit_hash,
            "commit_date": receipt.cargo_identity.commit_date,
            "host": receipt.cargo_identity.host,
            "first_line": receipt.cargo_identity.first_line,
        },
        "rustc_path": receipt.rustc_path,
        "rustc_sha256": receipt.rustc_sha256,
        "rustc_identity": {
            "component": receipt.rustc_identity.component,
            "release": receipt.rustc_identity.release,
            "commit_hash": receipt.rustc_identity.commit_hash,
            "commit_date": receipt.rustc_identity.commit_date,
            "host": receipt.rustc_identity.host,
            "first_line": receipt.rustc_identity.first_line,
        },
        "cargo_rapx_path": receipt.cargo_rapx_path,
        "cargo_rapx_sha256": receipt.cargo_rapx_sha256,
        "rapx_path": receipt.rapx_path,
        "rapx_sha256": receipt.rapx_sha256,
        "rap_source_path": receipt.rap_source_path,
        "rap_source_sha256": receipt.rap_source_sha256,
        "builder_sidecar_path": receipt.builder_sidecar_path,
        "builder_sidecar_sha256": receipt.builder_sidecar_sha256,
        "fixture_project_root": receipt.fixture_project_root,
        "fixture_manifest_sha256": receipt.fixture_manifest_sha256,
        "fixture_source_sha256": receipt.fixture_source_sha256,
        "fixture_lock_sha256": receipt.fixture_lock_sha256,
        "loader_environment_variable": receipt.loader_environment_variable,
        "loader_path": receipt.loader_path,
        "loader_contents_sha256": receipt.loader_contents_sha256,
    })
}

pub struct ContainsUnsafe<'tcx> {
    tcx: TyCtxt<'tcx>,
    function_unsafe: bool,
    block_unsafe: bool,
}

#[rustversion::before(1.96)]
fn hir_body<'tcx>(tcx: TyCtxt<'tcx>, body_id: BodyId) -> &'tcx rustc_hir::Body<'tcx> {
    tcx.hir().body(body_id)
}

#[rustversion::since(1.96)]
fn hir_body<'tcx>(tcx: TyCtxt<'tcx>, body_id: BodyId) -> &'tcx rustc_hir::Body<'tcx> {
    tcx.hir_body(body_id)
}

#[rustversion::before(1.83)]
fn hir_maybe_body_owned_by<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_def_id: LocalDefId,
) -> Option<&'tcx rustc_hir::Body<'tcx>> {
    tcx.hir()
        .maybe_body_owned_by(local_def_id)
        .map(|body_id| tcx.hir().body(body_id))
}

#[rustversion::all(since(1.83), before(1.96))]
fn hir_maybe_body_owned_by<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_def_id: LocalDefId,
) -> Option<&'tcx rustc_hir::Body<'tcx>> {
    tcx.hir().maybe_body_owned_by(local_def_id)
}

#[rustversion::since(1.96)]
fn hir_maybe_body_owned_by<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_def_id: LocalDefId,
) -> Option<&'tcx rustc_hir::Body<'tcx>> {
    tcx.hir_maybe_body_owned_by(local_def_id)
}

#[rustversion::before(1.96)]
fn impl_of_assoc_compat(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    tcx.impl_of_method(def_id)
}

#[rustversion::since(1.96)]
fn impl_of_assoc_compat(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    tcx.inherent_impl_of_assoc(def_id)
        .or_else(|| tcx.trait_impl_of_assoc(def_id))
}

#[rustversion::before(1.96)]
fn is_const_fn_compat(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_const_fn_raw(def_id) || tcx.is_const_fn(def_id)
}

#[rustversion::since(1.96)]
fn is_const_fn_compat(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_const_fn(def_id)
}

#[rustversion::before(1.83)]
fn is_const_or_assoc_const_or_static(kind: rustc_hir::def::DefKind) -> bool {
    matches!(
        kind,
        rustc_hir::def::DefKind::Const
            | rustc_hir::def::DefKind::AssocConst
            | rustc_hir::def::DefKind::Static(_)
    )
}

#[rustversion::all(since(1.83), before(1.96))]
fn is_const_or_assoc_const_or_static(kind: rustc_hir::def::DefKind) -> bool {
    matches!(
        kind,
        rustc_hir::def::DefKind::Const
            | rustc_hir::def::DefKind::AssocConst
            | rustc_hir::def::DefKind::Static {
                safety: _,
                mutability: _,
                nested: _
            }
    )
}

#[rustversion::since(1.96)]
fn is_const_or_assoc_const_or_static(kind: rustc_hir::def::DefKind) -> bool {
    matches!(
        kind,
        rustc_hir::def::DefKind::Const { .. }
            | rustc_hir::def::DefKind::AssocConst { .. }
            | rustc_hir::def::DefKind::Static {
                safety: _,
                mutability: _,
                nested: _
            }
    )
}

#[rustversion::before(1.83)]
fn is_const_or_static(kind: rustc_hir::def::DefKind) -> bool {
    matches!(
        kind,
        rustc_hir::def::DefKind::Const
            | rustc_hir::def::DefKind::Static(_)
    )
}

#[rustversion::all(since(1.83), before(1.96))]
fn is_const_or_static(kind: rustc_hir::def::DefKind) -> bool {
    matches!(
        kind,
        rustc_hir::def::DefKind::Const
            | rustc_hir::def::DefKind::Static {
                safety: _,
                mutability: _,
                nested: _
            }
    )
}

#[rustversion::before(1.83)]
fn hir_node<'tcx>(tcx: TyCtxt<'tcx>, hir_id: rustc_hir::HirId) -> Node<'tcx> {
    tcx.hir().get(hir_id)
}

#[rustversion::since(1.83)]
fn hir_node<'tcx>(tcx: TyCtxt<'tcx>, hir_id: rustc_hir::HirId) -> Node<'tcx> {
    tcx.hir_node(hir_id)
}

#[rustversion::since(1.96)]
fn is_const_or_static(kind: rustc_hir::def::DefKind) -> bool {
    matches!(
        kind,
        rustc_hir::def::DefKind::Const { .. }
            | rustc_hir::def::DefKind::Static {
                safety: _,
                mutability: _,
                nested: _
            }
    )
}

impl<'tcx> ContainsUnsafe<'tcx> {
    pub fn contains_unsafe(tcx: TyCtxt<'tcx>, body_id: BodyId) -> (bool, bool) {
        let mut visitor = ContainsUnsafe {
            tcx,
            function_unsafe: false,
            block_unsafe: false,
        };

        let body = hir_body(visitor.tcx, body_id);
        visitor.function_unsafe = visitor.body_unsafety(body);
        intravisit::walk_body(&mut visitor, body);

        (visitor.function_unsafe, visitor.block_unsafe)
    }

    fn body_unsafety(&self, body: &rustc_hir::Body<'_>) -> bool {
        let did = body.value.hir_id.owner.to_def_id();
        let sig = self.tcx.fn_sig(did);
        if fn_sig_is_unsafe!(sig.skip_binder()) {
            return true;
        }
        false
    }
}

impl<'tcx> Visitor<'tcx> for ContainsUnsafe<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    #[rustversion::before(1.96)]
    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    #[rustversion::since(1.96)]
    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_block(&mut self, block: &'tcx rustc_hir::Block<'tcx>) {
        if let BlockCheckMode::UnsafeBlock(_) = block.rules {
            self.block_unsafe = true;
        }
        if !self.block_unsafe {
            intravisit::walk_block(self, block);
        }
    }
}

#[derive(Debug, Clone)]
struct UnsafeOperation {
    operation_detail: String,
}

#[derive(Debug, Clone)]
struct PatternCarrier {
    def_id: DefId,
    tainted_param_idx: usize,
    unsafe_ops: Vec<UnsafeOperation>,
    pattern_type: u8,
}

#[derive(Debug, Clone)]
struct InterproceduralMatch {
    pub_fn_id: DefId,
    call_path: Vec<DefId>,
    unsafe_ops: Vec<UnsafeOperation>,
    base_pattern: u8,
}

#[derive(Clone, Debug)]
struct InternalUnsafe {
    def_id: DefId,
    is_public: bool,
    is_in_pub_mod: bool,
    unsafe_operations: Vec<UnsafeOperation>,
}

#[derive(Debug, Clone)]
struct FieldInfo {
    index: usize,
    name: String,
    is_public: bool,
}

#[derive(Debug, Clone)]
struct PubStructInfo {
    def_id: DefId,
    is_in_pub_mod: bool,
    pub_fields: HashMap<usize, FieldInfo>,
}

// Struct to hold Pattern 4 findings
#[derive(Debug, Clone)]
struct Pattern4Finding {
    desc: String,
    // span: Span, // Optional: could store span for more precise error reporting location
}

pub struct LwzCheck<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    internal_unsafe_fns: HashMap<DefId, InternalUnsafe>,
    pub_fns_in_pub_mods: HashSet<DefId>,
    call_graph: HashMap<DefId, Vec<DefId>>,
    reverse_call_graph: HashMap<DefId, Vec<DefId>>,
    shortest_paths: HashMap<DefId, Vec<DefId>>,
    pattern1_matches: HashMap<DefId, Vec<UnsafeOperation>>,
    pub_structs: HashMap<DefId, PubStructInfo>,
    pattern2_matches: HashMap<DefId, Vec<(UnsafeOperation, String)>>,
    pattern_carriers: HashMap<DefId, PatternCarrier>,
    interprocedural_matches: HashMap<DefId, InterproceduralMatch>,
    // Added for Pattern 4
    pattern4_matches: HashMap<DefId, Vec<Pattern4Finding>>,
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
            pattern4_matches: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        self.collect_functions();
        self.build_call_graphs();
        self.find_shortest_paths();
        self.identify_pattern_carriers();
        self.perform_interprocedural_analysis();
        self.detect_pattern1_matches();
        self.collect_pub_structs();
        self.detect_pattern2_matches();

        // Added: Run Pattern 4 detection (AST/HIR based)
        self.detect_pattern4_matches();

        self.report_findings();
        if let Err(err) = self.maybe_write_json_summary() {
            crate::utils::log::rap_error_and_exit(err);
        }
    }

    fn collect_functions(&mut self) {
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();
            let is_fn = matches!(
                self.tcx.def_kind(def_id),
                rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn
            );

            if !is_fn {
                continue;
            }

            if let Some(body_id) = hir_maybe_body_owned_by(self.tcx, local_def_id) {
                let function_unsafe = {
                    let did = body_id.value.hir_id.owner.to_def_id();
                    let sig = self.tcx.fn_sig(did);
                    fn_sig_is_unsafe!(sig.skip_binder())
                };

                let mut visitor = ContainsUnsafe {
                    tcx: self.tcx,
                    function_unsafe: false,
                    block_unsafe: false,
                };
                intravisit::walk_body(&mut visitor, body_id);
                let block_unsafe = visitor.block_unsafe;

                if !function_unsafe && block_unsafe {
                    let is_public = self.is_public_fn(def_id);
                    let is_in_pub_mod = self.is_in_public_module(def_id);
                    let unsafe_operations = self.extract_unsafe_operations(def_id);

                    let internal_unsafe = InternalUnsafe {
                        def_id,
                        is_public,
                        is_in_pub_mod,
                        unsafe_operations,
                    };

                    self.internal_unsafe_fns.insert(def_id, internal_unsafe);

                    if is_public && is_in_pub_mod {
                        self.pub_fns_in_pub_mods.insert(def_id);
                    }
                }

                if self.is_public_fn(def_id) && self.is_in_public_module(def_id) {
                    self.pub_fns_in_pub_mods.insert(def_id);
                }
            }
        }
    }

    fn build_call_graphs(&mut self) {
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();

            if !self.tcx.is_mir_available(def_id) {
                continue;
            }

            let callees = self.get_callees(def_id);
            if !callees.is_empty() {
                self.call_graph.insert(def_id, callees.clone());

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
        for internal_unsafe_fn in self.internal_unsafe_fns.keys() {
            if self.pub_fns_in_pub_mods.contains(internal_unsafe_fn) {
                self.shortest_paths
                    .insert(*internal_unsafe_fn, vec![*internal_unsafe_fn]);
                continue;
            }

            let path = self.find_shortest_path_to_pub_fn(*internal_unsafe_fn);

            if let Some(path) = path {
                self.shortest_paths.insert(*internal_unsafe_fn, path);
            }
        }
    }

    fn find_shortest_path_to_pub_fn(&self, from: DefId) -> Option<Vec<DefId>> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent_map: HashMap<DefId, DefId> = HashMap::new();

        queue.push_back(from);
        visited.insert(from);

        while let Some(current) = queue.pop_front() {
            if self.pub_fns_in_pub_mods.contains(&current) && current != from {
                let mut path = Vec::new();
                let mut curr = current;
                path.push(curr);
                while curr != from {
                    curr = *parent_map.get(&curr).unwrap();
                    path.push(curr);
                }
                path.reverse();
                return Some(path);
            }
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
        if !self.pattern1_matches.is_empty() {
            rap_info!(
                "\n===== Pattern1 (Parameter Directly Used in Unsafe Operation) Report ====="
            );
            let mut pattern1_count = 0;

            for (&fn_id, ops) in &self.pattern1_matches {
                pattern1_count += 1;
                let fn_name = self.get_fn_name(fn_id);

                rap_info!(
                    "{}: Public function with direct parameter to unsafe operation: {}",
                    pattern1_count,
                    fn_name
                );

                if !ops.is_empty() {
                    rap_info!("unsafe operations: ");
                    for (i, op) in ops.iter().enumerate() {
                        rap_info!("({}) {}, ", i + 1, op.operation_detail);
                    }
                    rap_info!("\n");
                }

                // 新增：打印函数源码
                self.print_fn_source(fn_id);
                rap_info!("--------------------------------------------------\n");
            }

            rap_info!("Total pattern1 functions found: {}", pattern1_count);
        }

        if !self.pattern2_matches.is_empty() {
            rap_info!(
                "\n===== Pattern2 (Public Struct Field Used in Unsafe Operation) Report ====="
            );
            let mut pattern2_count = 0;

            for (&fn_id, ops_with_fields) in &self.pattern2_matches {
                pattern2_count += 1;
                let fn_name = self.get_fn_name(fn_id);

                rap_info!(
                    "{}: Public function using public struct field in unsafe operation: {}",
                    pattern2_count,
                    fn_name
                );

                if !ops_with_fields.is_empty() {
                    rap_info!("unsafe operations with struct fields: ");
                    for (i, (op, field_path)) in ops_with_fields.iter().enumerate() {
                        rap_info!(
                            "({}) Field {} used in {}, ",
                            i + 1,
                            field_path,
                            op.operation_detail
                        );
                    }
                    rap_info!("\n");
                }

                // 新增：打印结构体源码
                if let Some(struct_def_id) = self.get_struct_def_id_from_fn(fn_id) {
                    self.print_struct_source(struct_def_id);
                    rap_info!("\n");
                }

                // 新增：打印函数源码
                self.print_fn_source(fn_id);
                rap_info!("--------------------------------------------------\n");
            }

            rap_info!("Total pattern2 functions found: {}", pattern2_count);
        }

        if !self.interprocedural_matches.is_empty() {
            rap_info!("\n===== Pattern3 (Interprocedural Taint Propagation) Report =====");
            let mut count = 0;

            for (&fn_id, interprocedural_match) in &self.interprocedural_matches {
                count += 1;
                let fn_name = self.get_fn_name(fn_id);

                let path_str = interprocedural_match
                    .call_path
                    .iter()
                    .map(|&def_id| self.get_fn_name(def_id))
                    .collect::<Vec<_>>()
                    .join(" -> ");

                rap_info!("{}: Interprocedural Taint Propagation: {}", count, path_str);

                if !interprocedural_match.unsafe_ops.is_empty() {
                    rap_info!("unsafe operations: ");
                    for (i, op) in interprocedural_match.unsafe_ops.iter().enumerate() {
                        rap_info!("({}) {}, ", i + 1, op.operation_detail);
                    }
                    rap_info!("\n");
                }

                // 修改：打印传播路径上所有函数的源码
                self.print_call_path_sources(&interprocedural_match.call_path);
                rap_info!("--------------------------------------------------\n");
            }

            rap_info!("Total pattern3 functions found: {}", count);
        }

        // Added Report for Pattern 4
        if !self.pattern4_matches.is_empty() {
            rap_info!("\n===== Pattern4 (Post-condition Unsoundness / AST-based) Report =====");
            let mut pattern4_count = 0;

            for (&fn_id, findings) in &self.pattern4_matches {
                pattern4_count += 1;
                let fn_name = self.get_fn_name(fn_id);

                rap_info!(
                    "{}: Public function with potential post-condition vulnerabilities: {}",
                    pattern4_count,
                    fn_name
                );

                for (i, finding) in findings.iter().enumerate() {
                    rap_info!("({}) {}", i + 1, finding.desc);
                }
                rap_info!("\n");

                // Print function source
                self.print_fn_source(fn_id);
                rap_info!("--------------------------------------------------\n");
            }

            rap_info!("Total pattern4 functions found: {}", pattern4_count);
        }
    }

    fn maybe_write_json_summary(&self) -> Result<(), String> {
        let Some(output_path) = env::var_os("UNSOUND_SCANNER_RAP_JSON_OUT") else {
            return Ok(());
        };

        let output_path = PathBuf::from(output_path);
        let summary = self.build_json_summary()?;
        let mut bytes = serde_json::to_vec_pretty(&summary).map_err(|err| {
            format!(
                "Failed to serialize RAP JSON output '{}': {}",
                output_path.display(),
                err
            )
        })?;
        bytes.push(b'\n');
        atomic_write(&output_path, &bytes).map_err(|err| {
            format!(
                "Failed to atomically write RAP JSON output '{}': {}",
                output_path.display(),
                err
            )
        })
    }

    fn build_json_summary(&self) -> Result<Value, String> {
        let build_verification = build_verification_receipt_from_environment()?;
        let rap_compiler_commit = embedded_rap_compiler_commit()?;
        let cargo_supplied_rustc_commit =
            env::var("UNSOUND_SCANNER_CARGO_SUPPLIED_RUSTC_COMMIT")
                .map_err(|_| "missing Cargo-supplied rustc commit".to_string())?;
        let cargo_supplied_rustc_commit = validate_commit_hash(
            Some(&cargo_supplied_rustc_commit),
            "Cargo-supplied rustc commit",
        )?;
        let mut findings = Vec::new();
        findings.extend(self.collect_pattern1_json_findings());
        findings.extend(self.collect_pattern2_json_findings());
        findings.extend(self.collect_pattern3_json_findings());
        findings.extend(self.collect_pattern4_json_findings());

        findings.sort_by(|left, right| {
            let left_key = (
                left.get("pattern")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                left.get("context_name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                left.get("file_path_or_null")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                left.get("line_or_null")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
            );
            let right_key = (
                right
                    .get("pattern")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                right
                    .get("context_name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                right
                    .get("file_path_or_null")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                right
                    .get("line_or_null")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
            );
            left_key.cmp(&right_key)
        });

        let mut summary = json!({
            "schema_version": "rap-unit-v1",
            "project_root": env::var("UNSOUND_SCANNER_PROJECT_ROOT")
                .ok()
                .or_else(|| {
                    env::current_dir()
                        .ok()
                        .map(|path| path.display().to_string())
                })
                .unwrap_or_default(),
            "source": "rap",
            "success": true,
            "unit_id": env::var("UNSOUND_SCANNER_RAP_UNIT_ID").unwrap_or_default(),
            "rustc_invocation_id": env::var("UNSOUND_SCANNER_RAP_RUSTC_INVOCATION_ID").unwrap_or_default(),
            "package_id": env::var("UNSOUND_SCANNER_RAP_PACKAGE_ID").unwrap_or_default(),
            "package_name": env::var("UNSOUND_SCANNER_RAP_PACKAGE_NAME").unwrap_or_default(),
            "package_version": env::var("UNSOUND_SCANNER_RAP_PACKAGE_VERSION").unwrap_or_default(),
            "crate_name": env::var("UNSOUND_SCANNER_RAP_CRATE_NAME").unwrap_or_default(),
            "crate_types": env::var("UNSOUND_SCANNER_RAP_CRATE_TYPES")
                .unwrap_or_default()
                .split(',')
                .filter(|crate_type| !crate_type.is_empty())
                .collect::<Vec<_>>(),
            "target_kind": env::var("UNSOUND_SCANNER_RAP_TARGET_KIND").unwrap_or_default(),
            "target_triple": env::var("UNSOUND_SCANNER_RAP_TARGET_TRIPLE").unwrap_or_default(),
            "manifest_path": env::var("UNSOUND_SCANNER_RAP_MANIFEST_PATH").unwrap_or_default(),
            "source_path": env::var("UNSOUND_SCANNER_RAP_SOURCE_PATH").unwrap_or_default(),
            "extra_filename": env::var("UNSOUND_SCANNER_RAP_EXTRA_FILENAME").unwrap_or_default(),
            "rustc_path": env::var("UNSOUND_SCANNER_RAP_RUSTC_PATH").unwrap_or_default(),
            "cargo_supplied_rustc_path": env::var("UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH").unwrap_or_default(),
            "cargo_supplied_rustc_commit": cargo_supplied_rustc_commit,
            "rap_compiler_commit": rap_compiler_commit.clone(),
            "rustc_commit": rap_compiler_commit,
            "pattern_counts": {
                "pattern1": self.pattern1_matches.len(),
                "pattern2": self.pattern2_matches.len(),
                "pattern3": self.interprocedural_matches.len(),
                "pattern4": self.pattern4_matches.len(),
            },
            "finding_count": findings.len(),
            "findings": findings,
        });
        if let Some(receipt) = build_verification {
            let summary = summary
                .as_object_mut()
                .expect("RAP JSON summary is always an object");
            summary.insert(
                "producer".to_string(),
                Value::String("rapx::analysis::UnsoundAudit".to_string()),
            );
            summary.insert(
                "build_verification".to_string(),
                build_verification_receipt_value(&receipt),
            );
        }
        Ok(summary)
    }

    fn collect_pattern1_json_findings(&self) -> Vec<Value> {
        let mut entries = self
            .pattern1_matches
            .iter()
            .map(|(&fn_id, ops)| (self.get_fn_name(fn_id), fn_id, ops.clone()))
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));

        entries
            .into_iter()
            .map(|(fn_name, fn_id, ops)| {
                let (file_path, line) = self.get_item_location(fn_id);
                let struct_source = self
                    .get_struct_def_id_from_fn(fn_id)
                    .and_then(|def_id| self.get_source_snippet(def_id));
                let unsafe_ops = ops
                    .iter()
                    .map(|op| op.operation_detail.clone())
                    .collect::<Vec<_>>();
                let unsafe_operation = (!unsafe_ops.is_empty()).then(|| unsafe_ops.join(" | "));

                self.make_json_finding_value(
                    "pattern1",
                    fn_name.clone(),
                    None,
                    fn_name.clone(),
                    self.context_kind_for_fn(fn_id).to_string(),
                    file_path,
                    line,
                    unsafe_operation,
                    format!(
                        "Public function with direct parameter to unsafe operation: {}",
                        fn_name
                    ),
                    json!({
                        "function_source": self.get_source_snippet(fn_id),
                        "struct_source": struct_source,
                        "callee_source": Value::Null,
                    }),
                )
            })
            .collect()
    }

    fn collect_pattern2_json_findings(&self) -> Vec<Value> {
        let mut entries = self
            .pattern2_matches
            .iter()
            .map(|(&fn_id, ops_with_fields)| {
                (self.get_fn_name(fn_id), fn_id, ops_with_fields.clone())
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));

        entries
            .into_iter()
            .map(|(fn_name, fn_id, ops_with_fields)| {
                let (file_path, line) = self.get_item_location(fn_id);
                let struct_source = self
                    .get_struct_def_id_from_fn(fn_id)
                    .and_then(|def_id| self.get_source_snippet(def_id));
                let field_paths = ops_with_fields
                    .iter()
                    .map(|(_, field_path)| field_path.clone())
                    .collect::<Vec<_>>();
                let unsafe_ops = ops_with_fields
                    .iter()
                    .map(|(op, _)| op.operation_detail.clone())
                    .collect::<Vec<_>>();

                self.make_json_finding_value(
                    "pattern2",
                    fn_name.clone(),
                    (!field_paths.is_empty()).then(|| field_paths.join(" | ")),
                    fn_name.clone(),
                    self.context_kind_for_fn(fn_id).to_string(),
                    file_path,
                    line,
                    (!unsafe_ops.is_empty()).then(|| unsafe_ops.join(" | ")),
                    format!(
                        "Public function using public struct field in unsafe operation: {}",
                        fn_name
                    ),
                    json!({
                        "function_source": self.get_source_snippet(fn_id),
                        "struct_source": struct_source,
                        "callee_source": Value::Null,
                    }),
                )
            })
            .collect()
    }

    fn collect_pattern3_json_findings(&self) -> Vec<Value> {
        let mut entries = self
            .interprocedural_matches
            .iter()
            .map(|(&fn_id, interprocedural_match)| {
                (
                    self.get_fn_name(fn_id),
                    fn_id,
                    interprocedural_match.clone(),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));

        entries
            .into_iter()
            .map(|(fn_name, fn_id, interprocedural_match)| {
                let (file_path, line) = self.get_item_location(fn_id);
                let call_path = interprocedural_match
                    .call_path
                    .iter()
                    .map(|def_id| self.get_fn_name(*def_id))
                    .collect::<Vec<_>>();
                let unsafe_ops = interprocedural_match
                    .unsafe_ops
                    .iter()
                    .map(|op| op.operation_detail.clone())
                    .collect::<Vec<_>>();
                let call_path_sources = interprocedural_match
                    .call_path
                    .iter()
                    .map(|def_id| {
                        json!({
                            "name": self.get_fn_name(*def_id),
                            "source": self.get_source_snippet(*def_id),
                        })
                    })
                    .collect::<Vec<_>>();

                self.make_json_finding_value(
                    "pattern3",
                    fn_name.clone(),
                    Some(call_path.join(" -> ")),
                    fn_name.clone(),
                    self.context_kind_for_fn(fn_id).to_string(),
                    file_path,
                    line,
                    (!unsafe_ops.is_empty()).then(|| unsafe_ops.join(" | ")),
                    format!(
                        "Interprocedural taint propagation from public entry to unsafe operation: {}",
                        fn_name
                    ),
                    json!({
                        "function_source": self.get_source_snippet(fn_id),
                        "struct_source": Value::Null,
                        "callee_source": Value::Null,
                        "call_path_sources": call_path_sources,
                    }),
                )
            })
            .collect()
    }

    fn collect_pattern4_json_findings(&self) -> Vec<Value> {
        let mut entries = self
            .pattern4_matches
            .iter()
            .map(|(&fn_id, findings)| (self.get_fn_name(fn_id), fn_id, findings.clone()))
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));

        entries
            .into_iter()
            .map(|(fn_name, fn_id, findings)| {
                let (file_path, line) = self.get_item_location(fn_id);
                let descriptions = findings
                    .iter()
                    .map(|finding| finding.desc.clone())
                    .collect::<Vec<_>>();
                let struct_source = self
                    .get_struct_def_id_from_fn(fn_id)
                    .and_then(|def_id| self.get_source_snippet(def_id));

                self.make_json_finding_value(
                    "pattern4",
                    fn_name.clone(),
                    (!descriptions.is_empty()).then(|| descriptions.join(" | ")),
                    fn_name.clone(),
                    self.context_kind_for_fn(fn_id).to_string(),
                    file_path,
                    line,
                    None,
                    format!(
                        "Public function with potential post-condition vulnerabilities: {}",
                        fn_name
                    ),
                    json!({
                        "function_source": self.get_source_snippet(fn_id),
                        "struct_source": struct_source,
                        "callee_source": Value::Null,
                    }),
                )
            })
            .collect()
    }

    fn make_json_finding_value(
        &self,
        pattern: &str,
        item_name: String,
        element_name: Option<String>,
        context_name: String,
        context_kind: String,
        file_path: Option<String>,
        line: Option<usize>,
        unsafe_operation: Option<String>,
        summary: String,
        snippets: Value,
    ) -> Value {
        let finding_id = format!(
            "{}|{}|{}|{}|{}|{}",
            pattern,
            file_path.clone().unwrap_or_default(),
            item_name,
            context_name,
            line.unwrap_or_default(),
            unsafe_operation.clone().unwrap_or_default(),
        );

        json!({
            "finding_id": finding_id,
            "pattern": pattern,
            "item_name": item_name,
            "element_name_or_null": element_name,
            "context_name": context_name,
            "context_kind": context_kind,
            "file_path_or_null": file_path,
            "line_or_null": line,
            "unsafe_operation_or_null": unsafe_operation,
            "summary": summary,
            "snippets": snippets,
        })
    }

    fn context_kind_for_fn(&self, def_id: DefId) -> &'static str {
        if self.get_impl_self_type(def_id).is_some() {
            "Method"
        } else {
            "Function"
        }
    }

    fn get_item_span(&self, def_id: DefId) -> Option<Span> {
        let local_def_id = def_id.as_local()?;
        let hir_id = self.tcx.local_def_id_to_hir_id(local_def_id);
        let node: Node<'_> = hir_node(self.tcx, hir_id);
        match node {
            Node::Item(item) => Some(item.span),
            Node::ImplItem(impl_item) => Some(impl_item.span),
            Node::TraitItem(trait_item) => Some(trait_item.span),
            Node::ForeignItem(foreign_item) => Some(foreign_item.span),
            Node::Expr(expr) => Some(expr.span),
            _ => None,
        }
    }

    fn get_source_snippet(&self, def_id: DefId) -> Option<String> {
        let span = self.get_item_span(def_id)?;
        self.tcx.sess.source_map().span_to_snippet(span).ok()
    }

    fn get_item_location(&self, def_id: DefId) -> (Option<String>, Option<usize>) {
        let span = if let Some(span) = self.get_item_span(def_id) {
            span
        } else {
            return (None, None);
        };

        (
            Some(span_to_filename(span)),
            Some(span_to_line_number(span)),
        )
    }

    fn is_public_fn(&self, def_id: DefId) -> bool {
        if let Some(local_def_id) = def_id.as_local() {
            let vis_map = self.tcx.effective_visibilities(());
            if let Some(eff_vis) = vis_map.effective_vis(local_def_id) {
                // `Reexported` 过于严格，会忽略二进制 crate 中的 `pub fn`（它们不会被再导出）。
                // 使用 Reachable 以覆盖 crate 对外可达的函数，避免漏报 Pattern4。
                return eff_vis.is_public_at_level(Level::Reachable);
            }
        }
        false
    }

    fn print_fn_source(&self, def_id: DefId) {
        if let Some(local_def_id) = def_id.as_local() {
            let hir_id = self.tcx.local_def_id_to_hir_id(local_def_id);

            // 新 API: hir_node()
            let node: Node<'_> = hir_node(self.tcx, hir_id);

            // 取 span
            let span: Span = match node {
                Node::Item(item) => item.span,
                Node::ImplItem(impl_item) => impl_item.span,
                Node::TraitItem(trait_item) => trait_item.span,
                Node::ForeignItem(foreign_item) => foreign_item.span,
                Node::Expr(expr) => expr.span,
                _ => return,
            };

            // 从 SourceMap 拿源码片段
            let source_map = self.tcx.sess.source_map();
            if let Ok(snippet) = source_map.span_to_snippet(span) {
                rap_info!("{}", snippet);
            } else {
                rap_info!("[WARN] Could not extract function source.");
            }
        }
    }

    fn print_struct_source(&self, struct_def_id: DefId) {
        if let Some(local_def_id) = struct_def_id.as_local() {
            let hir_id = self.tcx.local_def_id_to_hir_id(local_def_id);

            // 新 API: hir_node()
            let node: Node<'_> = hir_node(self.tcx, hir_id);

            // 取 span
            let span: Span = match node {
                Node::Item(item) => item.span,
                _ => return,
            };

            // 从 SourceMap 拿源码片段
            let source_map = self.tcx.sess.source_map();
            if let Ok(snippet) = source_map.span_to_snippet(span) {
                rap_info!("{}", snippet);
            } else {
                rap_info!("[WARN] Could not extract struct source.");
            }
        }
    }

    fn get_struct_def_id_from_fn(&self, fn_def_id: DefId) -> Option<DefId> {
        if let Some(impl_self_ty) = self.get_impl_self_type(fn_def_id) {
            if let Some(struct_def_id) = self.extract_struct_from_type(impl_self_ty) {
                return Some(struct_def_id);
            }
        }
        None
    }

    fn print_call_path_sources(&self, call_path: &[DefId]) {
        for (i, &def_id) in call_path.iter().enumerate() {
            let fn_name = self.get_fn_name(def_id);
            if i > 0 {
                rap_info!("\n");
            }
            rap_info!("Function {}: {}", i + 1, fn_name);
            self.print_fn_source(def_id);
        }
    }

    fn is_in_public_module(&self, def_id: DefId) -> bool {
        if let Some(local_def_id) = def_id.as_local() {
            let hir_id = self.tcx.local_def_id_to_hir_id(local_def_id);
            let mod_def_id: LocalDefId = self.tcx.parent_module(hir_id).to_def_id().expect_local();
            let vis_map = self.tcx.effective_visibilities(());
            if let Some(eff_vis) = vis_map.effective_vis(mod_def_id) {
                return eff_vis.is_public_at_level(Level::Reexported);
            }
        }
        false
    }

    fn get_fn_name(&self, def_id: DefId) -> String {
        self.tcx.def_path_str(def_id)
    }

    fn get_callees(&self, def_id: DefId) -> Vec<DefId> {
        let mut callees = Vec::new();
        let body = match self.get_mir_safely(def_id) {
            Some(body) => body,
            None => {
                return callees;
            }
        };
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { func, .. } = &terminator.kind {
                    if let Operand::Constant(constant) = func {
                        if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) =
                            constant.const_.ty().kind()
                        {
                            callees.push(*callee_def_id);
                        }
                    }
                }
            }
        }
        callees
    }

    fn get_mir_safely(&self, def_id: DefId) -> Option<&rustc_middle::mir::Body<'tcx>> {
        use std::panic::{self, AssertUnwindSafe};
        let is_const = is_const_fn_compat(self.tcx, def_id)
            || is_const_or_assoc_const_or_static(self.tcx.def_kind(def_id));
        let def_path = self.tcx.def_path_str(def_id);
        let is_const_expr = def_path.contains("::{constant#")
            || def_path.contains("::promoted[")
            || def_path.contains("]::")
            || def_path.ends_with("}");
        if is_const || is_const_expr {
            let def_kind = self.tcx.def_kind(def_id);
            if is_const_or_static(def_kind) {
                return Some(self.tcx.mir_for_ctfe(def_id));
            } else {
                return None;
            }
        }
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
            Err(_) => None,
        }
    }

    fn extract_unsafe_operations(&self, def_id: DefId) -> Vec<UnsafeOperation> {
        let mut operations = Vec::new();
        let body = match self.get_mir_safely(def_id) {
            Some(body) => body,
            None => {
                return operations;
            }
        };
        let fn_name = self.get_fn_name(def_id);
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                match &statement.kind {
                    rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) => {
                        self.check_place_for_raw_ptr_deref(
                            place,
                            &body.local_decls,
                            &mut operations,
                        );
                        match rvalue {
                            rustc_middle::mir::Rvalue::Use(operand) => {
                                if let Operand::Copy(source_place) | Operand::Move(source_place) =
                                    operand
                                {
                                    self.check_place_for_raw_ptr_deref(
                                        source_place,
                                        &body.local_decls,
                                        &mut operations,
                                    );
                                }
                            }
                            rustc_middle::mir::Rvalue::BinaryOp(op, box (left, _)) => {
                                if let rustc_middle::mir::BinOp::Offset = op {
                                    if let Operand::Copy(place) | Operand::Move(place) = left {
                                        let ty = place.ty(&body.local_decls, self.tcx).ty;
                                        if let rustc_middle::ty::TyKind::RawPtr(..) = ty.kind() {
                                            let operation = UnsafeOperation {
                                                operation_detail: "offset operation on raw pointer"
                                                    .to_string(),
                                            };
                                            operations.push(operation);
                                        }
                                    }
                                }
                            }
                            mir_raw_ptr_rvalue!(_, _place) => {}
                            rustc_middle::mir::Rvalue::Ref(_, _, place) => {
                                let ty = place.ty(&body.local_decls, self.tcx).ty;
                                if let rustc_middle::ty::TyKind::RawPtr(..) = ty.kind() {
                                    self.check_place_for_raw_ptr_deref(
                                        place,
                                        &body.local_decls,
                                        &mut operations,
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    rustc_middle::mir::StatementKind::Intrinsic(box intrinsic) => {
                        let operation = UnsafeOperation {
                            operation_detail: format!("intrinsic call: {:?}", intrinsic),
                        };
                        operations.push(operation);
                    }
                    _ => {}
                }
            }
            if let Some(terminator) = &block_data.terminator {
                match &terminator.kind {
                    TerminatorKind::Call { func, args, .. } => {
                        if let Operand::Constant(constant) = func {
                            if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) =
                                constant.const_.ty().kind()
                            {
                                if self.is_unsafe_fn(*callee_def_id) {
                                    let callee_name = self.get_fn_name(*callee_def_id);
                                    let mut arg_descriptions = Vec::new();
                                    for (i, arg) in args.iter().enumerate() {
                                        match mir_call_arg_operand!(arg) {
                                            Operand::Copy(place) | Operand::Move(place) => {
                                                let arg_str =
                                                    format!("_{}", place.local.as_usize());
                                                arg_descriptions.push(arg_str.clone());
                                            }
                                            Operand::Constant(constant) => {
                                                let arg_str = format!("{:?}", constant.const_);
                                                arg_descriptions.push(arg_str.clone());
                                            }
                                            _ => {}
                                        }
                                    }
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
                            if let Some(method_def_id) =
                                self.resolve_method(place, &body.local_decls)
                            {
                                if self.is_unsafe_fn(method_def_id) {
                                    let method_name = self.get_fn_name(method_def_id);
                                    let mut arg_descriptions = Vec::new();
                                    for arg in args.iter() {
                                        match mir_call_arg_operand!(arg) {
                                            Operand::Copy(place) | Operand::Move(place) => {
                                                let arg_str =
                                                    format!("_{}", place.local.as_usize());
                                                arg_descriptions.push(arg_str);
                                            }
                                            Operand::Constant(constant) => {
                                                arg_descriptions
                                                    .push(format!("{:?}", constant.const_));
                                            }
                                            _ => {}
                                        }
                                    }
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
                            self.check_place_for_raw_ptr_deref(
                                place,
                                &body.local_decls,
                                &mut operations,
                            );
                        }
                    }
                    TerminatorKind::InlineAsm { .. } => {
                        let operation = UnsafeOperation {
                            operation_detail: "inline assembly".to_string(),
                        };
                        operations.push(operation);
                    }
                    _ => {}
                }
            }
        }
        for (i, op) in operations.iter().enumerate() {}
        operations
    }

    fn check_place_for_raw_ptr_deref(
        &self,
        place: &rustc_middle::mir::Place<'tcx>,
        local_decls: &rustc_middle::mir::LocalDecls<'tcx>,
        operations: &mut Vec<UnsafeOperation>,
    ) {
        for (i, proj) in place.projection.iter().enumerate() {
            if let rustc_middle::mir::ProjectionElem::Deref = proj {
                let prefix_place = rustc_middle::mir::Place {
                    local: place.local,
                    projection: self.tcx.mk_place_elems(&place.projection[0..i]),
                };

                let prefix_ty = prefix_place.ty(local_decls, self.tcx).ty;

                if let rustc_middle::ty::TyKind::RawPtr(..) = prefix_ty.kind() {
                    let var_name = self.get_place_description(&prefix_place, local_decls);

                    let operation = UnsafeOperation {
                        operation_detail: format!("*{}", var_name),
                    };
                    operations.push(operation);
                }
            }
        }
    }

    fn get_place_description(
        &self,
        place: &rustc_middle::mir::Place<'tcx>,
        local_decls: &rustc_middle::mir::LocalDecls<'tcx>,
    ) -> String {
        let local = place.local;
        let local_decl = &local_decls[local];

        let base_name = match local_decl.local_info {
            rustc_middle::mir::ClearCrossCrate::Set(ref info) => match **info {
                rustc_middle::mir::LocalInfo::User(ref binding) => {
                    format!("{:?}", binding)
                }
                rustc_middle::mir::LocalInfo::BlockTailTemp(ref _block) => {
                    format!("_temp_{}", local.as_usize())
                }
                rustc_middle::mir::LocalInfo::Boring => {
                    format!("_var_{}", local.as_usize())
                }
                _ => format!("_var_{}", local.as_usize()),
            },
            rustc_middle::mir::ClearCrossCrate::Clear => {
                if local.as_usize() == 0 {
                    "_return".to_string()
                } else {
                    format!("_{}", local.as_usize())
                }
            }
        };

        let mut result = base_name;
        for elem in place.projection.iter() {
            match elem {
                rustc_middle::mir::ProjectionElem::Deref => {
                    result = format!("*{}", result);
                }
                rustc_middle::mir::ProjectionElem::Field(field, _) => {
                    result = format!("{}.{}", result, field.index());
                }
                rustc_middle::mir::ProjectionElem::Index(idx) => {
                    result = format!("{}[_{:?}]", result, idx);
                }
                _ => {}
            }
        }

        result
    }

    fn is_unsafe_fn(&self, def_id: DefId) -> bool {
        if self.tcx.is_mir_available(def_id) {
            let poly_fn_sig = self.tcx.fn_sig(def_id);
            let fn_sig = poly_fn_sig.skip_binder();
            return fn_sig_is_unsafe!(fn_sig);
        }
        false
    }

    fn resolve_method(
        &self,
        place: &rustc_middle::mir::Place<'tcx>,
        _local_decls: &rustc_middle::mir::LocalDecls<'tcx>,
    ) -> Option<DefId> {
        if let Some(field) = place.projection.last() {
            if let rustc_middle::mir::ProjectionElem::Field(_, _) = field {
                return None;
            }
        }
        None
    }

    fn detect_pattern1_matches(&mut self) {
        for (&def_id, internal_unsafe) in &self.internal_unsafe_fns {
            if self.is_public_fn(def_id) {
                let fn_name = self.get_fn_name(def_id);

                let mut pattern1_ops = Vec::new();

                if let Some(body) = self.get_mir_safely(def_id) {
                    let param_count = body.arg_count;

                    let is_method = param_count > 0 && self.get_impl_self_type(def_id).is_some();
                    let self_param = if is_method { Some(1) } else { None };

                    for op in &internal_unsafe.unsafe_operations {
                        if op.operation_detail.starts_with("*_") {
                            if let Some(var_number) = self.extract_var_number(&op.operation_detail)
                            {
                                let is_pattern1 = self.is_non_self_param_or_copy(
                                    body,
                                    var_number,
                                    param_count,
                                    self_param,
                                );
                                if is_pattern1 {
                                    if !self.is_var_sanitized(body, var_number, def_id) {
                                        pattern1_ops.push(op.clone());
                                    }
                                }
                            }
                        } else if op.operation_detail.contains("(")
                            && op.operation_detail.contains(")")
                        {
                            if self.check_function_call_non_self_args_with_sanitization(
                                &op.operation_detail,
                                body,
                                param_count,
                                self_param,
                                def_id,
                            ) {
                                pattern1_ops.push(op.clone());
                            }
                        }
                    }
                }

                if !pattern1_ops.is_empty() {
                    self.pattern1_matches.insert(def_id, pattern1_ops.clone());
                }
            }
        }
    }

    fn check_function_call_non_self_args_with_sanitization(
        &self,
        op_detail: &str,
        body: &rustc_middle::mir::Body<'tcx>,
        param_count: usize,
        self_param: Option<usize>,
        def_id: DefId,
    ) -> bool {
        let fn_name = self.get_fn_name(def_id);

        let is_unsafe_op = true;

        if let Some(start_pos) = op_detail.find('(') {
            if let Some(end_pos) = op_detail.rfind(')') {
                if start_pos < end_pos {
                    let args_str = &op_detail[start_pos + 1..end_pos];
                    for arg in args_str.split(',') {
                        let arg = arg.trim();
                        if arg.starts_with("_") {
                            if let Ok(arg_num) = arg[1..].parse::<usize>() {
                                if self.is_non_self_param_or_copy(
                                    body,
                                    arg_num,
                                    param_count,
                                    self_param,
                                ) {
                                    if !self.is_var_sanitized(body, arg_num, def_id) {
                                        return true;
                                    } else {
                                    }
                                } else {
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn is_non_self_param_or_copy(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var_num: usize,
        param_count: usize,
        self_param: Option<usize>,
    ) -> bool {
        if let Some(self_idx) = self_param {
            if var_num == self_idx {
                return false;
            }
        }
        if var_num > 0 && var_num <= param_count {
            if let Some(self_idx) = self_param {
                return var_num != self_idx;
            }
            return true;
        }
        let mut visited = HashSet::new();
        self.is_non_self_param_or_copy_with_visited(
            body,
            var_num,
            param_count,
            self_param,
            &mut visited,
        )
    }

    fn is_non_self_param_or_copy_with_visited(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var_num: usize,
        param_count: usize,
        self_param: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool {
        if !visited.insert(var_num) {
            return false;
        }
        if let Some(self_idx) = self_param {
            if var_num == self_idx {
                return false;
            }
        }
        if self.is_from_self_field(body, var_num, self_param) {
            return false;
        }
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                    &statement.kind
                {
                    if place.local.as_usize() == var_num {
                        match rvalue {
                            rustc_middle::mir::Rvalue::Use(operand) => {
                                if let Operand::Copy(source_place) | Operand::Move(source_place) =
                                    operand
                                {
                                    let source_local = source_place.local.as_usize();
                                    if source_local > 0 && source_local <= param_count {
                                        if let Some(self_idx) = self_param {
                                            if source_local != self_idx {
                                                return true;
                                            }
                                        } else {
                                            return true;
                                        }
                                    }
                                    if self.is_non_self_param_or_copy_with_visited(
                                        body,
                                        source_local,
                                        param_count,
                                        self_param,
                                        visited,
                                    ) {
                                        return true;
                                    }
                                }
                            }
                            rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                                let source_local = source_place.local.as_usize();
                                if source_local > 0 && source_local <= param_count {
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return true;
                                        }
                                    } else {
                                        return true;
                                    }
                                }
                                if self.is_non_self_param_or_copy_with_visited(
                                    body,
                                    source_local,
                                    param_count,
                                    self_param,
                                    visited,
                                ) {
                                    return true;
                                }
                            }
                            mir_raw_ptr_rvalue!(_, source_place) => {
                                let source_local = source_place.local.as_usize();
                                if source_local > 0 && source_local <= param_count {
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return true;
                                        }
                                    } else {
                                        return true;
                                    }
                                }
                                if self.is_non_self_param_or_copy_with_visited(
                                    body,
                                    source_local,
                                    param_count,
                                    self_param,
                                    visited,
                                ) {
                                    return true;
                                }
                            }
                            rustc_middle::mir::Rvalue::Aggregate(_, operands) => {
                                for operand in operands.iter() {
                                    match operand {
                                        Operand::Copy(source_place)
                                        | Operand::Move(source_place) => {
                                            let source_local = source_place.local.as_usize();
                                            if source_local > 0 && source_local <= param_count {
                                                if let Some(self_idx) = self_param {
                                                    if source_local != self_idx {
                                                        return true;
                                                    }
                                                } else {
                                                    return true;
                                                }
                                            }
                                            if self.is_non_self_param_or_copy_with_visited(
                                                body,
                                                source_local,
                                                param_count,
                                                self_param,
                                                visited,
                                            ) {
                                                return true;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                if let rustc_middle::mir::TerminatorKind::Call {
                    destination, args, ..
                } = &terminator.kind
                {
                    if destination.local.as_usize() == var_num {
                        for arg in args {
                            if let Operand::Copy(place) | Operand::Move(place) =
                                mir_call_arg_operand!(arg)
                            {
                                let source_local = place.local.as_usize();
                                if source_local > 0 && source_local <= param_count {
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return true;
                                        }
                                    } else {
                                        return true;
                                    }
                                }
                                if self.is_non_self_param_or_copy_with_visited(
                                    body,
                                    source_local,
                                    param_count,
                                    self_param,
                                    visited,
                                ) {
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

    fn is_from_self_field(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var_num: usize,
        self_param: Option<usize>,
    ) -> bool {
        let self_idx = match self_param {
            Some(idx) => idx,
            None => return false,
        };

        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                    &statement.kind
                {
                    if place.local.as_usize() != var_num {
                        continue;
                    }

                    match rvalue {
                        rustc_middle::mir::Rvalue::Use(operand) => {
                            if let Operand::Copy(source_place) | Operand::Move(source_place) =
                                operand
                            {
                                if source_place.local.as_usize() == self_idx
                                    && !source_place.projection.is_empty()
                                {
                                    return true;
                                }
                            }
                        }
                        rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                            if source_place.local.as_usize() == self_idx
                                && !source_place.projection.is_empty()
                            {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        false
    }

    fn extract_var_number(&self, op_detail: &str) -> Option<usize> {
        let prefixes = ["*_", "copy _", "move _"];

        for prefix in &prefixes {
            if op_detail.starts_with(prefix) {
                let var_part = &op_detail[prefix.len()..];
                let digit_end = var_part
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(var_part.len());
                return var_part[..digit_end].parse::<usize>().ok();
            }
        }
        None
    }

    fn debug_log(&self, msg: impl AsRef<str>) {
        rap_info!("DEBUG: {}", msg.as_ref());
    }

    fn collect_pub_structs(&mut self) {
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();

            let def_kind = self.tcx.def_kind(def_id);
            if def_kind != rustc_hir::def::DefKind::Struct {
                continue;
            }

            let collect_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let is_public = self.tcx.visibility(def_id).is_public();
                let is_in_pub_mod = self.is_in_public_module(def_id);

                if is_public {
                    let mut pub_struct_info = PubStructInfo {
                        def_id,
                        is_in_pub_mod,
                        pub_fields: HashMap::new(),
                    };

                    let adt_def = self.tcx.adt_def(def_id);
                    let struct_name = self.get_fn_name(def_id);

                    let variant_idx = VariantIdx::from_usize(0);
                    if let Some(variant) = adt_def.variants().get(variant_idx) {
                        for (idx, field) in variant.fields.iter().enumerate() {
                            let field_result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let field_def_id = field.did;
                                    let field_vis = self.tcx.visibility(field_def_id);
                                    let field_name = field.ident(self.tcx).to_string();

                                    if field_vis.is_public() {
                                        pub_struct_info.pub_fields.insert(
                                            idx,
                                            FieldInfo {
                                                index: idx,
                                                name: field_name.clone(),
                                                is_public: true,
                                            },
                                        );
                                        return Some(field_name);
                                    }
                                    None
                                }));

                            if field_result.is_err() {}
                        }
                    }

                    if !pub_struct_info.pub_fields.is_empty() {
                        return Some((def_id, pub_struct_info));
                    }
                }
                None
            }));

            match collect_result {
                Ok(Some((def_id, pub_struct_info))) => {
                    self.pub_structs.insert(def_id, pub_struct_info);
                }
                Ok(None) => {}
                Err(_) => {}
            }
        }
    }

    fn get_field_type_safely(
        &self,
        field: &rustc_middle::ty::FieldDef,
    ) -> Option<rustc_middle::ty::Ty<'tcx>> {
        use std::panic::{self, AssertUnwindSafe};

        let empty_substs = rustc_middle::ty::List::empty();

        let result = panic::catch_unwind(AssertUnwindSafe(|| field.ty(self.tcx, empty_substs)));

        match result {
            Ok(ty) => Some(ty),
            Err(_) => None,
        }
    }

    fn detect_pattern2_matches(&mut self) {
        for (&def_id, internal_unsafe) in &self.internal_unsafe_fns {
            if self.is_public_fn(def_id) {
                let fn_name = self.get_fn_name(def_id);
                let body = match self.get_mir_safely(def_id) {
                    Some(body) => body,
                    None => {
                        continue;
                    }
                };
                if let Some(impl_self_ty) = self.get_impl_self_type(def_id) {
                    if let Some(struct_def_id) = self.extract_struct_from_type(impl_self_ty) {
                        if let Some(struct_info) = self.pub_structs.get(&struct_def_id) {
                            let struct_name = self.get_fn_name(struct_def_id);
                            for (_field_idx, _field_info) in &struct_info.pub_fields {}
                            let self_param = self.find_self_param(body);
                            if let Some(self_param) = self_param {
                                for _op in &internal_unsafe.unsafe_operations {}
                                let mut pattern2_ops = Vec::new();
                                for op in &internal_unsafe.unsafe_operations {
                                    let field_accesses =
                                        self.find_struct_field_accesses(body, self_param);
                                    for (field_idx, field_var, _field_path) in &field_accesses {
                                        if let Some(field_info) =
                                            struct_info.pub_fields.get(field_idx)
                                        {
                                            if self.is_var_used_in_unsafe_op(
                                                body,
                                                *field_var,
                                                &op.operation_detail,
                                            ) {
                                                if !self.is_var_sanitized(body, *field_var, def_id)
                                                {
                                                    let field_name = &field_info.name;
                                                    pattern2_ops.push((
                                                        op.clone(),
                                                        format!("{}.{}", struct_name, field_name),
                                                    ));
                                                } else {
                                                }
                                            }
                                        }
                                    }
                                }
                                if !pattern2_ops.is_empty() {
                                    self.pattern2_matches.insert(def_id, pattern2_ops);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn find_self_param(&self, body: &rustc_middle::mir::Body<'tcx>) -> Option<usize> {
        if body.arg_count >= 1 {
            return Some(1);
        }
        None
    }

    fn find_struct_field_accesses(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        self_param: usize,
    ) -> Vec<(usize, usize, String)> {
        let mut field_accesses = Vec::new();
        let mut var_to_field_map: HashMap<usize, (usize, String)> = HashMap::new();
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                    &statement.kind
                {
                    if let rustc_middle::mir::Rvalue::Ref(_, _, source_place) = rvalue {
                        if let Some((base, proj)) = self.get_base_and_projection(source_place) {
                            if base.as_usize() == self_param {
                                for proj_elem in proj {
                                    if let rustc_middle::mir::ProjectionElem::Field(
                                        field_index,
                                        _,
                                    ) = proj_elem
                                    {
                                        let field_idx = field_index.index();
                                        let dest_var = place.local.as_usize();
                                        let access_path = format!("(self).{}", field_idx);
                                        field_accesses.push((
                                            field_idx,
                                            dest_var,
                                            access_path.clone(),
                                        ));
                                        var_to_field_map
                                            .insert(dest_var, (field_idx, access_path.clone()));
                                    }
                                }
                            }
                        }
                    } else if let rustc_middle::mir::Rvalue::Use(operand) = rvalue {
                        if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                            if let Some((base, proj)) = self.get_base_and_projection(source_place) {
                                if base.as_usize() == self_param {
                                    for proj_elem in proj {
                                        if let rustc_middle::mir::ProjectionElem::Field(
                                            field_index,
                                            _,
                                        ) = proj_elem
                                        {
                                            let field_idx = field_index.index();
                                            let dest_var = place.local.as_usize();
                                            let access_path = format!("(self).{}", field_idx);
                                            field_accesses.push((
                                                field_idx,
                                                dest_var,
                                                access_path.clone(),
                                            ));
                                            var_to_field_map
                                                .insert(dest_var, (field_idx, access_path.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let rvalue_str = format!("{:?}", rvalue);
                    if rvalue_str.contains("deref_copy")
                        && rvalue_str.contains(&format!("_{})", self_param))
                    {
                        if let Some(field_start) = rvalue_str.find(").") {
                            let field_substr = &rvalue_str[field_start + 2..];
                            if let Some(field_end) = field_substr.find(":") {
                                if let Ok(field_idx) =
                                    field_substr[..field_end].trim().parse::<usize>()
                                {
                                    let dest_var = place.local.as_usize();
                                    let access_path = format!("(self).{}", field_idx);
                                    field_accesses.push((field_idx, dest_var, access_path.clone()));
                                    var_to_field_map
                                        .insert(dest_var, (field_idx, access_path.clone()));
                                }
                            }
                        }
                    }
                    let dest_var = place.local.as_usize();
                    if let rustc_middle::mir::Rvalue::Use(operand) = rvalue {
                        if let Operand::Copy(source_place) | Operand::Move(source_place) = operand {
                            let source_var = source_place.local.as_usize();
                            let field_info_opt = var_to_field_map.get(&source_var).cloned();
                            if let Some((field_idx, access_path)) = field_info_opt {
                                var_to_field_map.insert(dest_var, (field_idx, access_path.clone()));
                            }
                        }
                    }
                }
            }
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { destination, .. } = &terminator.kind {
                    let dest_var = destination.local.as_usize();
                    let terminator_str = format!("{:?}", terminator);
                    if terminator_str.contains("deref")
                        && terminator_str.contains(&format!("_{})", self_param))
                    {
                        if let Some(field_start) = terminator_str.find(").") {
                            let field_substr = &terminator_str[field_start + 2..];
                            if let Some(field_end) = field_substr.find(":") {
                                if let Ok(field_idx) =
                                    field_substr[..field_end].trim().parse::<usize>()
                                {
                                    let access_path = format!("(self).{}", field_idx);
                                    field_accesses.push((field_idx, dest_var, access_path.clone()));
                                    var_to_field_map
                                        .insert(dest_var, (field_idx, access_path.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
        for block_data in body.basic_blocks.iter() {
            if let Some(terminator) = &block_data.terminator {
                let terminator_str = format!("{:?}", terminator);
                for (&var, &(field_idx, ref _access_path)) in &var_to_field_map {
                    if terminator_str.contains(&format!("(*_{})", var)) {}
                }
            }
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                    &statement.kind
                {
                    let rvalue_str = format!("{:?}", rvalue);
                    let dest_var = place.local.as_usize();
                    if rvalue_str.contains("copy (*_") {
                        for (&var, &(field_idx, ref access_path)) in &var_to_field_map {
                            if rvalue_str.contains(&format!("copy (*_{})", var)) {
                                field_accesses.push((field_idx, var, access_path.clone()));
                            }
                        }
                    }
                }
            }
        }
        field_accesses
    }

    fn get_base_and_projection(
        &self,
        place: &rustc_middle::mir::Place<'tcx>,
    ) -> Option<(
        rustc_middle::mir::Local,
        &[rustc_middle::mir::PlaceElem<'tcx>],
    )> {
        Some((place.local, place.projection.as_ref()))
    }

    fn is_var_used_in_unsafe_op(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var: usize,
        op_detail: &str,
    ) -> bool {
        let var_str = format!("_{}", var);
        if op_detail.contains(&var_str) {
            return true;
        }
        let mut visited = HashSet::new();
        self.check_var_flows_to_unsafe_op(body, var, op_detail, &mut visited)
    }

    fn check_var_flows_to_unsafe_op(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var: usize,
        op_detail: &str,
        visited: &mut HashSet<usize>,
    ) -> bool {
        if !visited.insert(var) {
            return false;
        }
        let var_str = format!("_{}", var);
        if op_detail.contains(&var_str) {
            return true;
        }
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                    &statement.kind
                {
                    match rvalue {
                        rustc_middle::mir::Rvalue::Use(operand) => {
                            if let Operand::Copy(source_place) | Operand::Move(source_place) =
                                operand
                            {
                                if source_place.local.as_usize() == var {
                                    let dest_var = place.local.as_usize();
                                    if self.check_var_flows_to_unsafe_op(
                                        body, dest_var, op_detail, visited,
                                    ) {
                                        return true;
                                    }
                                }
                            }
                        }
                        rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                            if source_place.local.as_usize() == var {
                                let dest_var = place.local.as_usize();
                                let ref_var_pattern = format!("_{}", dest_var);
                                if self.check_var_flows_to_unsafe_op(
                                    body, dest_var, op_detail, visited,
                                ) {
                                    return true;
                                }
                                for (i, _) in op_detail.match_indices("(") {
                                    if i + 1 < op_detail.len() {
                                        let end_idx = op_detail[i..]
                                            .find(")")
                                            .map(|pos| i + pos)
                                            .unwrap_or(op_detail.len());
                                        let arg_str = &op_detail[i + 1..end_idx];
                                        if arg_str.contains(&ref_var_pattern) {
                                            return true;
                                        }
                                    }
                                }
                                for other_block in body.basic_blocks.iter() {
                                    if let Some(terminator) = &other_block.terminator {
                                        if let TerminatorKind::Call {
                                            func: _,
                                            args,
                                            destination,
                                            ..
                                        } = &terminator.kind
                                        {
                                            for arg in args {
                                                if let Operand::Copy(place) | Operand::Move(place) =
                                                    mir_call_arg_operand!(arg)
                                                {
                                                    let place_str = format!("{:?}", place);
                                                    if place_str.contains(&ref_var_pattern) {
                                                        let term_str = format!("{:?}", terminator);
                                                        if term_str.contains(op_detail) {
                                                            return true;
                                                        }
                                                        let result_var =
                                                            destination.local.as_usize();
                                                        if self.check_var_flows_to_unsafe_op(
                                                            body, result_var, op_detail, visited,
                                                        ) {
                                                            return true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if op_detail.contains("from_utf8_unchecked") {
                                    if let Some(ty) = body
                                        .local_decls
                                        .get(rustc_middle::mir::Local::from_usize(var))
                                    {
                                        let ty_str = format!("{:?}", ty.ty);
                                        if ty_str.contains("[u8;") || ty_str.contains("&[u8") {
                                            return true;
                                        }
                                    }
                                    for other_stmt in &block_data.statements {
                                        if let rustc_middle::mir::StatementKind::Assign(box (
                                            other_place,
                                            other_rvalue,
                                        )) = &other_stmt.kind
                                        {
                                            let other_rvalue_str = format!("{:?}", other_rvalue);
                                            if other_rvalue_str.contains(&ref_var_pattern) {
                                                let next_var = other_place.local.as_usize();
                                                if self.check_var_flows_to_unsafe_op(
                                                    body, next_var, op_detail, visited,
                                                ) {
                                                    return true;
                                                }
                                            }
                                        }
                                    }
                                }
                                for other_block in body.basic_blocks.iter() {
                                    for other_stmt in &other_block.statements {
                                        if let rustc_middle::mir::StatementKind::Assign(box (
                                            other_place,
                                            other_rvalue,
                                        )) = &other_stmt.kind
                                        {
                                            if let rustc_middle::mir::Rvalue::Use(
                                                Operand::Copy(ref_place) | Operand::Move(ref_place),
                                            ) = other_rvalue
                                            {
                                                let ref_place_str = format!("{:?}", ref_place);
                                                if ref_place_str.contains(&ref_var_pattern) {
                                                    let new_var = other_place.local.as_usize();
                                                    if self.check_var_flows_to_unsafe_op(
                                                        body, new_var, op_detail, visited,
                                                    ) {
                                                        return true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { func: _, args, .. } = &terminator.kind {
                    for arg in args {
                        if let Operand::Copy(place) | Operand::Move(place) =
                            mir_call_arg_operand!(arg)
                        {
                            if place.local.as_usize() == var {
                                return true;
                            }
                        }
                    }
                    let term_str = format!("{:?}", terminator);
                    let var_ref_pattern = format!("&_{}", var);
                    if term_str.contains(&var_ref_pattern) {
                        return true;
                    }
                    if term_str.contains("from_utf8_unchecked")
                        && term_str.contains(&format!("_{}", var))
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn is_var_sanitized(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var: usize,
        def_id: DefId,
    ) -> bool {
        let fn_name = self.get_fn_name(def_id);
        let mut sanitizer_found = false;
        for (block_idx, block_data) in body.basic_blocks.iter().enumerate() {
            if let Some(terminator) = &block_data.terminator {
                match &terminator.kind {
                    TerminatorKind::Call { func, args, .. } => {
                        if let Operand::Constant(constant) = func {
                            if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) =
                                constant.const_.ty().kind()
                            {
                                let callee_fn_name = self.get_fn_name(*callee_def_id);
                                if self.is_sanitizer_function_name(&callee_fn_name) {
                                    for arg in args.iter() {
                                        if let Operand::Copy(place) | Operand::Move(place) =
                                            mir_call_arg_operand!(arg)
                                        {
                                            if place.local.as_usize() == var {
                                                sanitizer_found = true;
                                            }
                                        }
                                    }
                                    if !sanitizer_found
                                        && self.check_result_used_in_condition(body, block_idx)
                                    {
                                        sanitizer_found = true;
                                    }
                                }
                            }
                        }
                    }
                    TerminatorKind::SwitchInt { discr, .. } => {
                        if let Operand::Copy(place) | Operand::Move(place) = discr {
                            if place.local.as_usize() == var {
                                sanitizer_found = true;
                            } else {
                                let terminator_str = format!("{:?}", terminator);
                                if terminator_str.contains(&format!("_{}", var)) {
                                    sanitizer_found = true;
                                }
                                let cond_var = place.local.as_usize();
                                if self.is_var_from_sanitizer_call(body, cond_var, var) {
                                    sanitizer_found = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (_place, rvalue)) =
                    &statement.kind
                {
                    let rvalue_str = format!("{:?}", rvalue);
                    if rvalue_str.contains(&format!("_{}", var))
                        && self.is_sanitizer_function_name(&rvalue_str)
                    {
                        sanitizer_found = true;
                    }
                }
            }
        }
        if sanitizer_found {
            return true;
        }
        if self.is_sanitizer_function_name(&fn_name) {
            if var > 0 && var <= body.arg_count {
                return true;
            }
        }
        false
    }

    fn is_var_from_sanitizer_call(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var: usize,
        target_var: usize,
    ) -> bool {
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                    &statement.kind
                {
                    if place.local.as_usize() == var {
                        let rvalue_str = format!("{:?}", rvalue);
                        if rvalue_str.contains(&format!("_{}", target_var))
                            && (rvalue_str.contains("is_ascii") || rvalue_str.contains("from_utf8"))
                        {
                            return true;
                        }
                    }
                }
            }
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } = &terminator.kind
                {
                    if destination.local.as_usize() == var {
                        if let Operand::Constant(constant) = func {
                            if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) =
                                constant.const_.ty().kind()
                            {
                                let callee_fn_name = self.get_fn_name(*callee_def_id);
                                if self.is_sanitizer_function_name(&callee_fn_name) {
                                    for arg in args.iter() {
                                        if let Operand::Copy(place) | Operand::Move(place) =
                                            mir_call_arg_operand!(arg)
                                        {
                                            if place.local.as_usize() == target_var {
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

    fn check_result_used_in_condition(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        block_idx: usize,
    ) -> bool {
        if let Some(block_data) = body
            .basic_blocks
            .get(rustc_middle::mir::BasicBlock::from_usize(block_idx))
        {
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { destination, .. } = &terminator.kind {
                    let result_var = destination.local.as_usize();
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

    fn is_sanitizer_function_name(&self, name: &str) -> bool {
        let sanitizer_keywords = [
            "valid", "check", "is_", "has_", "ensure", "verify", "safe", "is_null",
        ];
        let unsafe_functions = [
            "unchecked",
            "raw_parts",
            "transmute",
            "copy_nonoverlapping",
            "write",
            "read",
            "ptr",
            "get_unchecked",
            "from_utf8_unchecked",
        ];
        let name_lower = name.to_lowercase();
        for unsafe_keyword in &unsafe_functions {
            if name_lower.contains(unsafe_keyword) {
                return false;
            }
        }
        if name_lower.contains("from_utf8") && !name_lower.contains("unchecked") {
            return true;
        }
        for keyword in &sanitizer_keywords {
            if name_lower.contains(keyword) {
                return true;
            }
        }
        false
    }

    fn extract_struct_from_type(&self, ty: rustc_middle::ty::Ty<'tcx>) -> Option<DefId> {
        match ty.kind() {
            rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_struct() => Some(adt_def.did()),
            rustc_middle::ty::TyKind::Ref(_, inner_ty, _) => {
                self.extract_struct_from_type(*inner_ty)
            }
            rustc_middle::ty::TyKind::Array(elem_ty, _) => self.extract_struct_from_type(*elem_ty),
            rustc_middle::ty::TyKind::Slice(elem_ty) => self.extract_struct_from_type(*elem_ty),
            _ => None,
        }
    }

    fn get_impl_self_type(&self, def_id: DefId) -> Option<rustc_middle::ty::Ty<'tcx>> {
        if let Some(impl_def_id) = impl_of_assoc_compat(self.tcx, def_id) {
            return Some(self.tcx.type_of(impl_def_id).skip_binder());
        }
        None
    }

    fn identify_pattern_carriers(&mut self) {
        for (&def_id, internal_unsafe) in &self.internal_unsafe_fns {
            let fn_name = self.get_fn_name(def_id);
            let mut pattern1_ops = Vec::new();
            let mut tainted_param_idx = 0;
            if let Some(body) = self.get_mir_safely(def_id) {
                let param_count = body.arg_count;
                let is_method = param_count > 0 && self.get_impl_self_type(def_id).is_some();
                let self_param = if is_method { Some(1) } else { None };
                for op in &internal_unsafe.unsafe_operations {
                    if op.operation_detail.starts_with("*_") {
                        if let Some(var_number) = self.extract_var_number(&op.operation_detail) {
                            let is_pattern1 = self.is_non_self_param_or_copy(
                                body,
                                var_number,
                                param_count,
                                self_param,
                            );
                            if is_pattern1 {
                                if !self.is_var_sanitized(body, var_number, def_id) {
                                    pattern1_ops.push(op.clone());
                                    if let Some(source_param) = self.find_source_parameter(
                                        body,
                                        var_number,
                                        param_count,
                                        self_param,
                                    ) {
                                        tainted_param_idx = source_param;
                                    }
                                }
                            }
                        }
                    } else if op.operation_detail.contains("(") && op.operation_detail.contains(")")
                    {
                        if let Some(source_param) = self.check_function_call_tainted_param(
                            &op.operation_detail,
                            body,
                            param_count,
                            self_param,
                            def_id,
                        ) {
                            pattern1_ops.push(op.clone());
                            tainted_param_idx = source_param;
                        }
                    }
                }
                'block_loop: for (block_idx, block_data) in body.basic_blocks.iter().enumerate() {
                    for (stmt_idx, statement) in block_data.statements.iter().enumerate() {
                        if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                            &statement.kind
                        {
                            if let rustc_middle::mir::Rvalue::Cast(_, ref operand, cast_ty) = rvalue
                            {
                                if let rustc_middle::ty::TyKind::RawPtr(..) = cast_ty.kind() {
                                    if let Operand::Copy(source_place)
                                    | Operand::Move(source_place) = operand
                                    {
                                        let source_local = source_place.local.as_usize();
                                        if self.is_non_self_param_or_copy(
                                            body,
                                            source_local,
                                            param_count,
                                            self_param,
                                        ) {
                                            if !self.is_var_sanitized(body, source_local, def_id) {
                                                let ptr_local = place.local.as_usize();
                                                if let Some(source_param) = self
                                                    .find_source_parameter(
                                                        body,
                                                        source_local,
                                                        param_count,
                                                        self_param,
                                                    )
                                                {
                                                    let deref_op = UnsafeOperation {
                                                        operation_detail: format!(
                                                            "*_{} (raw pointer cast from param)",
                                                            ptr_local
                                                        ),
                                                    };
                                                    pattern1_ops.push(deref_op);
                                                    tainted_param_idx = source_param;
                                                    break 'block_loop;
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
            if !pattern1_ops.is_empty() && tainted_param_idx > 0 {
                let carrier = PatternCarrier {
                    def_id,
                    tainted_param_idx,
                    unsafe_ops: pattern1_ops.clone(),
                    pattern_type: 1,
                };
                self.pattern_carriers.insert(def_id, carrier);
            }
        }
    }

    fn find_source_parameter(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var_num: usize,
        param_count: usize,
        self_param: Option<usize>,
    ) -> Option<usize> {
        if var_num > 0 && var_num <= param_count {
            if let Some(self_idx) = self_param {
                if var_num != self_idx {
                    return Some(var_num);
                }
            } else {
                return Some(var_num);
            }
        }
        let mut visited = HashSet::new();
        self.find_source_parameter_with_visited(
            body,
            var_num,
            param_count,
            self_param,
            &mut visited,
        )
    }

    fn find_source_parameter_with_visited(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        var_num: usize,
        param_count: usize,
        self_param: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> Option<usize> {
        if !visited.insert(var_num) {
            return None;
        }
        if var_num > 0 && var_num <= param_count {
            if let Some(self_idx) = self_param {
                if var_num != self_idx {
                    return Some(var_num);
                }
            } else {
                return Some(var_num);
            }
        }
        for block_data in body.basic_blocks.iter() {
            for statement in &block_data.statements {
                if let rustc_middle::mir::StatementKind::Assign(box (place, rvalue)) =
                    &statement.kind
                {
                    if place.local.as_usize() == var_num {
                        match rvalue {
                            rustc_middle::mir::Rvalue::Use(operand) => {
                                if let Operand::Copy(source_place) | Operand::Move(source_place) =
                                    operand
                                {
                                    let source_local = source_place.local.as_usize();
                                    if source_local > 0 && source_local <= param_count {
                                        if let Some(self_idx) = self_param {
                                            if source_local != self_idx {
                                                return Some(source_local);
                                            }
                                        } else {
                                            return Some(source_local);
                                        }
                                    }
                                    if let Some(source_param) = self
                                        .find_source_parameter_with_visited(
                                            body,
                                            source_local,
                                            param_count,
                                            self_param,
                                            visited,
                                        )
                                    {
                                        return Some(source_param);
                                    }
                                }
                            }
                            rustc_middle::mir::Rvalue::Ref(_, _, source_place) => {
                                let source_local = source_place.local.as_usize();
                                if source_local > 0 && source_local <= param_count {
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return Some(source_local);
                                        }
                                    } else {
                                        return Some(source_local);
                                    }
                                }
                                if let Some(source_param) = self.find_source_parameter_with_visited(
                                    body,
                                    source_local,
                                    param_count,
                                    self_param,
                                    visited,
                                ) {
                                    return Some(source_param);
                                }
                            }
                            mir_raw_ptr_rvalue!(_, source_place) => {
                                let source_local = source_place.local.as_usize();
                                if source_local > 0 && source_local <= param_count {
                                    if let Some(self_idx) = self_param {
                                        if source_local != self_idx {
                                            return Some(source_local);
                                        }
                                    } else {
                                        return Some(source_local);
                                    }
                                }
                                if let Some(source_param) = self.find_source_parameter_with_visited(
                                    body,
                                    source_local,
                                    param_count,
                                    self_param,
                                    visited,
                                ) {
                                    return Some(source_param);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        None
    }

    fn check_function_call_tainted_param(
        &self,
        op_detail: &str,
        body: &rustc_middle::mir::Body<'tcx>,
        param_count: usize,
        self_param: Option<usize>,
        def_id: DefId,
    ) -> Option<usize> {
        let is_unsafe_op = true;
        if let Some(start_pos) = op_detail.find('(') {
            if let Some(end_pos) = op_detail.rfind(')') {
                if start_pos < end_pos {
                    let args_str = &op_detail[start_pos + 1..end_pos];
                    for arg in args_str.split(',') {
                        let arg = arg.trim();
                        if arg.starts_with("_") {
                            if let Ok(arg_num) = arg[1..].parse::<usize>() {
                                if self.is_non_self_param_or_copy(
                                    body,
                                    arg_num,
                                    param_count,
                                    self_param,
                                ) {
                                    if !self.is_var_sanitized(body, arg_num, def_id) {
                                        if let Some(source_param) = self.find_source_parameter(
                                            body,
                                            arg_num,
                                            param_count,
                                            self_param,
                                        ) {
                                            return Some(source_param);
                                        }
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

    fn perform_interprocedural_analysis(&mut self) {
        self.pattern_carriers.clear();
        self.identify_pattern_carriers();
        let mut matches_to_add = Vec::new();
        let mut intermediate_carriers = Vec::new();
        let mut carrier_callers = Vec::new();
        let carrier_def_ids: Vec<DefId> = self.pattern_carriers.keys().cloned().collect();
        for carrier_def_id in carrier_def_ids {
            let carrier = self.pattern_carriers.get(&carrier_def_id).unwrap().clone();
            if let Some(callers) = self.reverse_call_graph.get(&carrier_def_id) {
                for &caller_def_id in callers {
                    carrier_callers.push((caller_def_id, carrier_def_id, carrier.clone()));
                }
            }
        }
        for (caller_def_id, carrier_def_id, carrier) in carrier_callers {
            if self.is_public_fn(caller_def_id) {
                if let Some(body) = self.get_mir_safely(caller_def_id) {
                    if let Some(call_arg) =
                        self.find_caller_arg_local(body, carrier_def_id, carrier.tainted_param_idx)
                    {
                        let param_count = body.arg_count;
                        let is_method =
                            param_count > 0 && self.get_impl_self_type(caller_def_id).is_some();
                        let self_param = if is_method { Some(1) } else { None };
                        if self.is_non_self_param_or_copy(body, call_arg, param_count, self_param) {
                            if !self.is_var_sanitized(body, call_arg, caller_def_id) {
                                let mut call_path = vec![caller_def_id];
                                let path_valid =
                                    self.build_call_path(carrier_def_id, &mut call_path);
                                if path_valid {
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
                if let Some(body) = self.get_mir_safely(caller_def_id) {
                    if let Some(call_arg) =
                        self.find_caller_arg_local(body, carrier_def_id, carrier.tainted_param_idx)
                    {
                        let param_count = body.arg_count;
                        let is_method =
                            param_count > 0 && self.get_impl_self_type(caller_def_id).is_some();
                        let self_param = if is_method { Some(1) } else { None };
                        if self.is_non_self_param_or_copy(body, call_arg, param_count, self_param) {
                            if !self.is_var_sanitized(body, call_arg, caller_def_id) {
                                if let Some(source_param) = self.find_source_parameter(
                                    body,
                                    call_arg,
                                    param_count,
                                    self_param,
                                ) {
                                    let intermediate_carrier = PatternCarrier {
                                        def_id: caller_def_id,
                                        tainted_param_idx: source_param,
                                        unsafe_ops: carrier.unsafe_ops.clone(),
                                        pattern_type: carrier.pattern_type,
                                    };
                                    intermediate_carriers
                                        .push((caller_def_id, intermediate_carrier));
                                }
                            }
                        }
                    }
                }
            }
        }
        for (def_id, match_info) in matches_to_add {
            self.interprocedural_matches.insert(def_id, match_info);
        }
        for (def_id, carrier) in intermediate_carriers {
            self.pattern_carriers.insert(def_id, carrier);
        }
        let mut iteration = 0;
        let max_iterations = 10;
        let mut found_new_match = true;
        while found_new_match && iteration < max_iterations {
            found_new_match = false;
            iteration += 1;
            let current_carriers = self.pattern_carriers.clone();
            let current_matches = self.interprocedural_matches.clone();
            let mut new_matches = Vec::new();
            let mut new_carriers = Vec::new();
            let mut processed_fns = HashSet::new();
            for &def_id in current_matches.keys() {
                processed_fns.insert(def_id);
            }
            for &def_id in current_carriers.keys() {
                processed_fns.insert(def_id);
            }
            for (carrier_def_id, carrier) in current_carriers {
                if current_matches.contains_key(&carrier_def_id) {
                    continue;
                }
                if let Some(callers) = self.reverse_call_graph.get(&carrier_def_id) {
                    for &caller_def_id in callers {
                        if processed_fns.contains(&caller_def_id) {
                            continue;
                        }
                        processed_fns.insert(caller_def_id);
                        if let Some(body) = self.get_mir_safely(caller_def_id) {
                            if let Some(call_arg) = self.find_caller_arg_local(
                                body,
                                carrier_def_id,
                                carrier.tainted_param_idx,
                            ) {
                                let param_count = body.arg_count;
                                let is_method = param_count > 0
                                    && self.get_impl_self_type(caller_def_id).is_some();
                                let self_param = if is_method { Some(1) } else { None };
                                if self.is_non_self_param_or_copy(
                                    body,
                                    call_arg,
                                    param_count,
                                    self_param,
                                ) {
                                    if !self.is_var_sanitized(body, call_arg, caller_def_id) {
                                        if self.is_public_fn(caller_def_id) {
                                            let mut call_path = vec![caller_def_id];
                                            let path_valid = self
                                                .build_call_path(carrier_def_id, &mut call_path);
                                            if path_valid {
                                                let interprocedural_match = InterproceduralMatch {
                                                    pub_fn_id: caller_def_id,
                                                    call_path,
                                                    unsafe_ops: carrier.unsafe_ops.clone(),
                                                    base_pattern: carrier.pattern_type,
                                                };
                                                new_matches
                                                    .push((caller_def_id, interprocedural_match));
                                                found_new_match = true;
                                            }
                                        } else if let Some(source_param) = self
                                            .find_source_parameter(
                                                body,
                                                call_arg,
                                                param_count,
                                                self_param,
                                            )
                                        {
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
            for (def_id, match_info) in new_matches {
                self.interprocedural_matches.insert(def_id, match_info);
            }
            for (def_id, carrier) in new_carriers {
                self.pattern_carriers.insert(def_id, carrier);
            }
        }
    }

    fn build_call_path(&self, start_def_id: DefId, call_path: &mut Vec<DefId>) -> bool {
        call_path.push(start_def_id);
        let is_pub = self.is_public_fn(start_def_id);
        if let Some(carrier) = self.pattern_carriers.get(&start_def_id) {
            let carriers: Vec<DefId> = self.pattern_carriers.keys().cloned().collect();
            for target_def_id in carriers {
                if target_def_id == start_def_id {
                    continue;
                }
                if let Some(callees) = self.call_graph.get(&start_def_id) {
                    if callees.contains(&target_def_id) {
                        if let Some(body) = self.get_mir_safely(start_def_id) {
                            if let Some(target_carrier) = self.pattern_carriers.get(&target_def_id)
                            {
                                if let Some(_) = self.find_caller_arg_local(
                                    body,
                                    target_def_id,
                                    target_carrier.tainted_param_idx,
                                ) {
                                    let sub_path_valid =
                                        self.build_call_path(target_def_id, call_path);
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
        if start_def_id != call_path[0] && is_pub {
            return false;
        }
        true
    }

    fn find_caller_arg_local(
        &self,
        body: &rustc_middle::mir::Body<'tcx>,
        callee_def_id: DefId,
        callee_param_idx: usize,
    ) -> Option<usize> {
        for (_block_idx, block_data) in body.basic_blocks.iter().enumerate() {
            if let Some(terminator) = &block_data.terminator {
                if let TerminatorKind::Call { func, args, .. } = &terminator.kind {
                    if let Operand::Constant(constant) = func {
                        if let rustc_middle::ty::TyKind::FnDef(func_def_id, _) =
                            constant.const_.ty().kind()
                        {
                            if *func_def_id == callee_def_id {
                                let param_idx = callee_param_idx - 1;
                                if param_idx < args.len() {
                                    if let Operand::Copy(place) | Operand::Move(place) =
                                        mir_call_arg_operand!(&args[param_idx])
                                    {
                                        let var_num = place.local.as_usize();
                                        return Some(var_num);
                                    }
                                }
                            }
                        }
                    }
                    if let Some(callees) = self.call_graph.get(&body.source.def_id()) {
                        if callees.contains(&callee_def_id) {
                            if args.len() >= callee_param_idx {
                                let param_idx = callee_param_idx - 1;
                                if let Operand::Copy(place) | Operand::Move(place) =
                                    mir_call_arg_operand!(&args[param_idx])
                                {
                                    let var_num = place.local.as_usize();
                                    return Some(var_num);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    // =========================================================================
    // Pattern 4: Post-condition / AST-based Logic (Ported from main.rs)
    // =========================================================================

    fn detect_pattern4_matches(&mut self) {
        // Iterate over all local definitions, finding functions
        for local_def_id in self.tcx.iter_local_def_id() {
            let def_id = local_def_id.to_def_id();

            // Check if function is public (main.rs constraint)
            if !self.is_public_fn(def_id) {
                continue;
            }

            // Ensure it's a function
            let is_fn = matches!(
                self.tcx.def_kind(def_id),
                rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn
            );
            if !is_fn {
                continue;
            }

            // Get the HIR body
            if let Some(body) = hir_maybe_body_owned_by(self.tcx, local_def_id) {
                // --- 1. Determine Return Type Risk ---
                let mut returns_risk = false;
                let sig = self.tcx.fn_sig(def_id).skip_binder();
                let output = sig.output();

                let output_str = format!("{:?}", output);
                if output_str.contains("&")
                    || output_str.contains("String")
                    || output_str.contains("Vec")
                {
                    returns_risk = true;
                }

                // --- 2. Visit Body (AST/HIR traversal) ---
                let mut visitor = PostCondVisitor::new(self.tcx);
                // Walk the HIR body (simulating syn::visit::Visit)
                intravisit::walk_body(&mut visitor, body);

                // --- 3. Analyze Findings from Visitor ---
                let mut findings = Vec::new();

                // 3a. Check Returns: direct call or tainted variable
                let mut flagged = false;

                for (ret_expr_str, line) in &visitor.returns {
                    // Check direct high risk call in return
                    if visitor.expr_str_contains_high_risk(ret_expr_str) {
                        let desc = format!(
                            "pub fn returns expression that directly contains high-risk call around line {}",
                            line
                        );
                        findings.push(Pattern4Finding { desc });
                        flagged = true;
                    }

                    // Check if returned identifier is tainted
                    // (Simple check: if ret_expr_str matches a tainted variable name)
                    if let Some(tags) = visitor.origins.get(ret_expr_str) {
                        if tags.contains("unsafe_origin")
                            || tags.contains("set_len_related")
                            || tags.contains("transmute_lifetime")
                        {
                            let desc = format!(
                                "pub fn returns variable '{}' originating from unsafe operation (tags={:?}) around line {}",
                                ret_expr_str, tags, line
                            );
                            findings.push(Pattern4Finding { desc });
                            flagged = true;
                        }
                    }
                }

                // 3b. If returns risky type and not yet flagged, inspect unsafe blocks
                if returns_risk && !flagged {
                    for ub_span in &visitor.unsafe_blocks {
                        if self.unsafe_block_protected(*ub_span, &visitor) {
                            continue;
                        }

                        // Check if unsafe block contains high risk call
                        // (We look at calls collected inside the visitor that happened within the unsafe block's span roughly)
                        // Heuristic: check if visitor.calls contains high risk ones that are roughly in the same location
                        // Since we don't have perfect span containment check here easily, we check if ANY high risk call exists
                        // and if the unsafe block is unprotected.

                        // A more precise way mimicking main.rs: check calls inside the unsafe block
                        // We will use the stored 'calls' in visitor, check if any high risk call exists
                        // Note: ideally we check if the call is *inside* the UB.
                        // Simplified port: if UB is unprotected and we found high risk calls in the function, flag it.

                        if visitor.calls.iter().any(|(name, _)| {
                            [
                                "from_raw",
                                "transmute",
                                "set_len",
                                "assume_init",
                                "from_raw_parts",
                                "read_unaligned",
                                "alloc",
                            ]
                            .iter()
                            .any(|hr| name.contains(hr))
                        }) {
                            let desc = format!(
                                "pub fn contains unsafe high-risk call without clear protective-check and returns risky type"
                            );
                            // Avoid duplicates
                            if !findings.iter().any(|f| f.desc == desc) {
                                findings.push(Pattern4Finding { desc });
                            }
                        }
                    }
                }

                // 3c. set_len post-check heuristic
                for (name, line) in &visitor.calls {
                    if name.contains("set_len") {
                        if !visitor.has_init_after_set_len(*line) {
                            let desc = format!("pub fn calls set_len at line {} without observed initialization writes afterward", line);
                            findings.push(Pattern4Finding { desc });
                        }
                    }
                }

                if !findings.is_empty() {
                    self.pattern4_matches.insert(def_id, findings);
                }
            }
        }
    }

    // Helper to check protection (Port of unsafe_block_protected)
    fn unsafe_block_protected(&self, _ub_span: Span, visitor: &PostCondVisitor<'_>) -> bool {
        // Logic from main.rs:
        // 1) inside then-branch of protective if
        // 2) early-return check before it

        // 1. Check if UB is inside a protective IF block
        // Since we don't have the full hierarchy stored as tree in visitor (we flattened it),
        // we can check if the UB's span is contained within any 'protective' if block collected.
        // Simplified: Check if we encountered any protective condition in the function.
        // This is a loose approximation of main.rs logic suitable for HIR flattening without complex CFG.

        for (cond_str, _then_span, _is_protective) in &visitor.if_nodes {
            if visitor.cond_is_protective_str(cond_str) {
                return true;
            }
        }

        false
    }
}

// =============================================================================
// Helper Structures for Pattern 4 (HIR Visitor)
// =============================================================================

struct PostCondVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    // Data collected
    let_bindings: Vec<(String, String)>, // (lhs_name, rhs_expr_str/desc)
    assignments: Vec<(String, String)>,  // (lhs_name, rhs_expr_str/desc)
    returns: Vec<(String, usize)>,       // (expr_str, line)
    calls: Vec<(String, usize)>,         // (func_name, line)
    unsafe_blocks: Vec<Span>,
    if_nodes: Vec<(String, Span, bool)>, // (cond_str, then_span, is_protective)

    // Origins (Dataflow state)
    origins: HashMap<String, HashSet<String>>,
}

impl<'tcx> PostCondVisitor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            let_bindings: Vec::new(),
            assignments: Vec::new(),
            returns: Vec::new(),
            calls: Vec::new(),
            unsafe_blocks: Vec::new(),
            if_nodes: Vec::new(),
            origins: HashMap::new(),
        }
    }

    fn has_init_after_set_len(&self, set_len_line: usize) -> bool {
        for (name, line) in &self.calls {
            if *line > set_len_line
                && (name.contains("ptr::write")
                    || name.contains("copy_nonoverlapping")
                    || name.contains("write_unaligned"))
            {
                return true;
            }
        }
        false
    }

    // Heuristic string check for high risk
    fn expr_str_contains_high_risk(&self, s: &str) -> bool {
        let risks = [
            "from_raw_parts",
            "from_raw_parts_mut",
            "from_utf8_unchecked",
            "transmute",
            "Box::from_raw",
            "set_len",
            "assume_init",
            // 扩展：捕获分配与未对齐读场景
            "alloc",
            "read_unaligned",
        ];
        risks.iter().any(|r| s.contains(r))
    }

    fn cond_is_protective_str(&self, s: &str) -> bool {
        let keys = [
            "from_utf8",
            "is_ok",
            "is_null",
            "len",
            "validate",
            "check",
            "is_ascii",
            "is_empty",
        ];
        keys.iter().any(|k| s.contains(k))
    }
}

impl<'tcx> Visitor<'tcx> for PostCondVisitor<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    #[rustversion::before(1.96)]
    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    #[rustversion::since(1.96)]
    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    // Visit assignments (x = ...)
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        let source_map = self.tcx.sess.source_map();
        let line = source_map.lookup_char_pos(expr.span.lo()).line;

        match expr.kind {
            ExprKind::Assign(lhs, rhs, _) => {
                // Try to get LHS name
                if let ExprKind::Path(QPath::Resolved(_, path)) = lhs.kind {
                    if let Some(segment) = path.segments.last() {
                        let lhs_name = segment.ident.name.to_string();
                        let rhs_desc = format!("{:?}", rhs);

                        // Propagate taint
                        if self.expr_str_contains_high_risk(&rhs_desc) {
                            self.origins
                                .entry(lhs_name.clone())
                                .or_default()
                                .insert("unsafe_origin".to_string());
                        }
                        // Check if RHS is a variable that is already tainted
                        if let ExprKind::Path(QPath::Resolved(_, rhs_path)) = rhs.kind {
                            if let Some(r_seg) = rhs_path.segments.last() {
                                let r_name = r_seg.ident.name.to_string();
                                if let Some(tags) = self.origins.get(&r_name).cloned() {
                                    self.origins
                                        .entry(lhs_name.clone())
                                        .or_default()
                                        .extend(tags);
                                }
                            }
                        }

                        self.assignments.push((lhs_name, rhs_desc));
                    }
                }
            }
            ExprKind::Ret(Some(ret_val)) => {
                // Try to get return value name or description
                let ret_desc = if let ExprKind::Path(QPath::Resolved(_, path)) = ret_val.kind {
                    path.segments
                        .last()
                        .map(|s| s.ident.name.to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                } else {
                    // For complex expressions, just check string rep
                    // In a real tool, we might use span_to_snippet
                    if let Ok(snip) = source_map.span_to_snippet(ret_val.span) {
                        snip
                    } else {
                        format!("{:?}", ret_val)
                    }
                };
                self.returns.push((ret_desc, line));
            }
            // Implicit return (last expression in block) check
            // (Skipped for brevity in this port, relying on explicit Returns or main body flow)
            ExprKind::Call(func, _args) => {
                // Get function name
                if let ExprKind::Path(QPath::Resolved(_, path)) = func.kind {
                    let fn_name = path
                        .segments
                        .iter()
                        .map(|s| s.ident.name.to_string())
                        .collect::<Vec<_>>()
                        .join("::");
                    self.calls.push((fn_name, line));
                }
            }
            ExprKind::MethodCall(segment, _receiver, _args, _) => {
                let method_name = segment.ident.name.to_string();
                self.calls.push((method_name, line));
            }
            ExprKind::Block(block, _) => {
                if let BlockCheckMode::UnsafeBlock(_) = block.rules {
                    self.unsafe_blocks.push(block.span);
                }
            }
            ExprKind::If(cond, _then, _else) => {
                // Analyze condition string for protective patterns
                if let Ok(cond_snip) = source_map.span_to_snippet(cond.span) {
                    let is_prot = self.cond_is_protective_str(&cond_snip);
                    // _then block span
                    // We assume the block follows the condition
                    self.if_nodes.push((cond_snip, expr.span, is_prot));
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }
}
