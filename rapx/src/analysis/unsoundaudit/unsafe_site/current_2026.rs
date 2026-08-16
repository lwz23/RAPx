use std::collections::BTreeSet;

use super::{AdapterManifest, UnsafeSiteKind};

pub const RUSTC_COMMIT: &str = "ba28ff76f353a722f31c4f3dd2ac4e437d36411b";
const CHECKER_SHA256: &str = "25b4b8e22ff7d561dda2a9abf429ab5bef035d246ec0200b9b887f7713fca346";

pub fn manifest() -> AdapterManifest {
    AdapterManifest {
        adapter_id: "rustc-nightly-2026-08-14-ba28ff".to_string(),
        rustc_commit: RUSTC_COMMIT.to_string(),
        checker_sha256: CHECKER_SHA256.to_string(),
        emitted_kinds: BTreeSet::from([
            UnsafeSiteKind::UnsafeCall,
            UnsafeSiteKind::InlineAssembly,
            UnsafeSiteKind::UnsafeFieldInitialization,
            UnsafeSiteKind::UnsafeFieldUse,
            UnsafeSiteKind::MutableStaticAccess,
            UnsafeSiteKind::ExternStaticAccess,
            UnsafeSiteKind::RawPointerDereference,
            UnsafeSiteKind::UnionFieldAccess,
            UnsafeSiteKind::TargetFeatureCall,
            UnsafeSiteKind::UnsafeBinderCast,
            UnsafeSiteKind::ExplicitDropCall,
        ]),
    }
}

#[rustversion::since(1.96)]
mod collector {
    use std::mem;
    use std::sync::OnceLock;

    use rustc_ast::AsmMacro;
    use rustc_data_structures::steal::Steal;
    use rustc_hir::attrs::find_attr;
    use rustc_hir::attrs::lang_items::LangItem;
    use rustc_hir::def::DefKind;
    use rustc_hir::{self as hir, HirId};
    use rustc_middle::middle::codegen_fn_attrs::TargetFeatureKind;
    use rustc_middle::mir::Body;
    use rustc_middle::thir::visit::{self, Visitor};
    use rustc_middle::thir::*;
    use rustc_middle::ty::{self, TyCtxt};
    use rustc_middle::util::Providers;
    use rustc_span::def_id::LocalDefId;
    use rustc_span::Span;

    use super::super::{
        census_generation, census_store, raw_span_anchor, type_fingerprint, CoverageFailure,
        CoverageFailureKind, FunctionKey, OperandAnchor, OperandRole, OwnerCensus, ResultAnchor,
        UnsafeSiteKind, UnsafeSiteSeed,
    };
    use super::manifest;

    type CheckUnsafetyProvider = for<'tcx> fn(TyCtxt<'tcx>, LocalDefId);
    type MirBuiltProvider = for<'tcx> fn(TyCtxt<'tcx>, LocalDefId) -> &'tcx Steal<Body<'tcx>>;
    static ORIGINAL_PROVIDER: OnceLock<CheckUnsafetyProvider> = OnceLock::new();
    static ORIGINAL_MIR_BUILT_PROVIDER: OnceLock<MirBuiltProvider> = OnceLock::new();

    pub fn install(providers: &mut Providers) -> Result<(), String> {
        let original = providers.queries.check_unsafety;
        if ORIGINAL_PROVIDER.get().is_some() {
            return Err("current check_unsafety provider installed twice".to_string());
        }
        ORIGINAL_PROVIDER
            .set(original)
            .map_err(|_| "current check_unsafety provider installed twice".to_string())?;
        ORIGINAL_MIR_BUILT_PROVIDER
            .set(providers.queries.mir_built)
            .map_err(|_| "current mir_built provider installed twice".to_string())?;
        providers.queries.check_unsafety = wrapped_check_unsafety;
        providers.queries.mir_built = wrapped_mir_built;
        Ok(())
    }

    fn wrapped_mir_built(tcx: TyCtxt<'_>, def: LocalDefId) -> &Steal<Body<'_>> {
        let provider = ORIGINAL_MIR_BUILT_PROVIDER
            .get()
            .copied()
            .unwrap_or_else(|| {
                crate::utils::log::rap_error_and_exit("missing original mir_built provider")
            });
        let body = provider(tcx, def);
        // Some compiler queries build a typeck child's MIR before its parent's unsafety query
        // visits that child.  At this point the THIR is still available: census it before this
        // MIR result is returned to a caller that may steal the THIR.
        let mut deferred_built_inventory = Vec::new();
        collect_census_tree(tcx, def, &mut deferred_built_inventory);
        if body.is_stolen() {
            super::super::record_stolen_built_mir(tcx, def).unwrap_or_else(|error| {
                crate::utils::log::rap_error_and_exit(format!(
                    "current mir_built inventory failed: {error}"
                ))
            });
        } else {
            let borrowed = body.borrow();
            super::super::record_fresh_built_mir_events(tcx, def, &borrowed).unwrap_or_else(
                |error| {
                    crate::utils::log::rap_error_and_exit(format!(
                        "current mir_built inventory failed: {error}"
                    ))
                },
            );
        }
        body
    }

    fn wrapped_check_unsafety(tcx: TyCtxt<'_>, def: LocalDefId) {
        collect_and_record(tcx, def);
        let provider = ORIGINAL_PROVIDER.get().copied().unwrap_or_else(|| {
            crate::utils::log::rap_error_and_exit("missing original check_unsafety provider")
        });
        provider(tcx, def);
    }

    fn collect_and_record(tcx: TyCtxt<'_>, def: LocalDefId) {
        let mut built_inventory_order = Vec::new();
        collect_census_tree(tcx, def, &mut built_inventory_order);
        for owner in built_inventory_order {
            super::super::record_built_mir_events(tcx, owner).unwrap_or_else(|error| {
                crate::utils::log::rap_error_and_exit(format!(
                    "current unsafe-site built MIR inventory failed: {error}"
                ))
            });
        }
    }

    fn collect_census_tree(
        tcx: TyCtxt<'_>,
        def: LocalDefId,
        built_inventory_order: &mut Vec<LocalDefId>,
    ) {
        let generation = census_generation().unwrap_or_else(|error| {
            crate::utils::log::rap_error_and_exit(format!("unsafe census store: {error}"))
        });
        let owner = FunctionKey::new(tcx.def_path_str(def.to_def_id()));
        if census_store()
            .contains_owner(generation, &owner)
            .unwrap_or_else(|error| crate::utils::log::rap_error_and_exit(error))
        {
            built_inventory_order.push(def);
            return;
        }
        match collect_owner(tcx, def) {
            Ok((census, inner_bodies)) => {
                census_store()
                    .record_owner(generation, census)
                    .unwrap_or_else(|error| crate::utils::log::rap_error_and_exit(error));
                built_inventory_order.push(def);
                // Capture the complete THIR subtree before forcing any MIR query. Building a
                // parent's MIR may steal a typeck child's THIR; after the owned census is stored,
                // MIR can safely be inventoried in this parent-before-child order.
                for inner_body in inner_bodies {
                    collect_census_tree(tcx, inner_body, built_inventory_order);
                }
            }
            Err(error) => {
                census_store()
                    .record_owner(
                        generation,
                        OwnerCensus {
                            owner: owner.clone(),
                            collector_ran: false,
                            sites: Vec::new(),
                        },
                    )
                    .unwrap_or_else(|store_error| {
                        crate::utils::log::rap_error_and_exit(store_error)
                    });
                census_store()
                    .record_failure(
                        generation,
                        CoverageFailure {
                            kind: error.failure_kind(),
                            owner,
                            site_id: None,
                            candidate_count: 0,
                        },
                    )
                    .unwrap_or_else(|store_error| {
                        crate::utils::log::rap_error_and_exit(store_error)
                    });
                log::error!("current unsafe census failed: {}", error.message());
            }
        }
    }

    enum CollectOwnerError {
        Stolen(String),
        Failed(String),
    }

    impl CollectOwnerError {
        fn failure_kind(&self) -> CoverageFailureKind {
            match self {
                Self::Stolen(_) => CoverageFailureKind::CollectorMissingAfterSteal,
                Self::Failed(_) => CoverageFailureKind::CollectorFailed,
            }
        }

        fn message(&self) -> &str {
            match self {
                Self::Stolen(message) | Self::Failed(message) => message,
            }
        }
    }

    fn collect_owner(
        tcx: TyCtxt<'_>,
        def: LocalDefId,
    ) -> Result<(OwnerCensus, Vec<LocalDefId>), CollectOwnerError> {
        let owner = FunctionKey::new(tcx.def_path_str(def.to_def_id()));
        if find_attr!(tcx, def, CustomMir(..) => ()).is_some() {
            return Ok((
                OwnerCensus {
                    owner,
                    collector_ran: true,
                    sites: Vec::new(),
                },
                Vec::new(),
            ));
        }
        let Ok((thir, body_expr)) = tcx.thir_body(def) else {
            return Ok((
                OwnerCensus {
                    owner,
                    collector_ran: true,
                    sites: Vec::new(),
                },
                Vec::new(),
            ));
        };
        if thir.is_stolen() {
            return Err(CollectOwnerError::Stolen(format!(
                "THIR for {} was stolen before census",
                owner.0
            )));
        }
        let thir = thir.borrow();
        let hir_context = tcx.local_def_id_to_hir_id(def);
        let mut visitor = CensusVisitor {
            tcx,
            thir: &thir,
            owner: owner.clone(),
            owner_def_path_hash: tcx.def_path_hash(def.to_def_id()).0.to_hex(),
            hir_context,
            ordinal: 0,
            sites: Vec::new(),
            errors: Vec::new(),
            suppress_builtin: false,
            assignment_info: false,
            in_union_destructure: false,
            body_target_features: &tcx.body_codegen_attrs(def.to_def_id()).target_features,
            inner_bodies: Vec::new(),
        };
        for param in &thir.params {
            if let Some(pattern) = param.pat.as_deref() {
                visitor.visit_pat(pattern);
            }
        }
        visitor.visit_expr(&thir[body_expr]);
        if !visitor.errors.is_empty() {
            return Err(CollectOwnerError::Failed(visitor.errors.join("; ")));
        }
        let CensusVisitor {
            sites,
            inner_bodies,
            ..
        } = visitor;
        let census = OwnerCensus {
            owner,
            collector_ran: true,
            sites,
        };
        census.validate().map_err(CollectOwnerError::Failed)?;
        Ok((census, inner_bodies))
    }

    struct CensusVisitor<'a, 'tcx> {
        tcx: TyCtxt<'tcx>,
        thir: &'a Thir<'tcx>,
        owner: FunctionKey,
        owner_def_path_hash: String,
        hir_context: HirId,
        ordinal: u32,
        sites: Vec<UnsafeSiteSeed>,
        errors: Vec<String>,
        suppress_builtin: bool,
        assignment_info: bool,
        in_union_destructure: bool,
        body_target_features: &'tcx [rustc_middle::middle::codegen_fn_attrs::TargetFeature],
        inner_bodies: Vec<LocalDefId>,
    }

    impl<'a, 'tcx> CensusVisitor<'a, 'tcx> {
        fn record(
            &mut self,
            kind: UnsafeSiteKind,
            span: Span,
            callee: Option<String>,
            operands: Vec<OperandAnchor>,
            result: Option<ResultAnchor>,
        ) {
            if self.suppress_builtin {
                return;
            }
            let raw_span = match raw_span_anchor(self.tcx, span) {
                Ok(span) => span,
                Err(error) => {
                    self.errors.push(error);
                    return;
                }
            };
            let ordinal = self.ordinal;
            self.ordinal = self.ordinal.checked_add(1).unwrap_or_else(|| {
                crate::utils::log::rap_error_and_exit("unsafe site ordinal overflow")
            });
            self.sites.push(UnsafeSiteSeed {
                adapter_id: manifest().adapter_id,
                owner: self.owner.clone(),
                owner_def_path_hash: self.owner_def_path_hash.clone(),
                ordinal,
                kind,
                raw_span,
                hir_context: format!("{:?}", self.hir_context),
                callee,
                operands,
                result,
            });
        }

        fn anchor(&mut self, id: ExprId, role: OperandRole) -> Option<OperandAnchor> {
            let expr = &self.thir[id];
            match raw_span_anchor(self.tcx, expr.span) {
                Ok(span) => Some(OperandAnchor {
                    role,
                    span,
                    type_fingerprint: type_fingerprint(self.tcx, expr.ty),
                    shape: expr_shape(&expr.kind).to_string(),
                }),
                Err(error) => {
                    self.errors.push(error);
                    None
                }
            }
        }

        fn result(&mut self, expr: &Expr<'tcx>) -> Option<ResultAnchor> {
            match raw_span_anchor(self.tcx, expr.span) {
                Ok(span) => Some(ResultAnchor {
                    span,
                    type_fingerprint: type_fingerprint(self.tcx, expr.ty),
                    shape: expr_shape(&expr.kind).to_string(),
                }),
                Err(error) => {
                    self.errors.push(error);
                    None
                }
            }
        }

        fn visit_inner_body(&mut self, def: LocalDefId) {
            self.inner_bodies.push(def);
        }
    }

    fn expr_shape(kind: &ExprKind<'_>) -> &'static str {
        match kind {
            ExprKind::Call { .. } => "call",
            ExprKind::Deref { .. } => "deref",
            ExprKind::Field { .. } => "field",
            ExprKind::Assign { .. } => "assign",
            ExprKind::AssignOp { .. } => "assign_op",
            ExprKind::Adt(..) => "adt",
            ExprKind::InlineAsm(..) => "inline_asm",
            ExprKind::StaticRef { .. } => "static_ref",
            ExprKind::ThreadLocalRef(..) => "thread_local_ref",
            ExprKind::PlaceUnwrapUnsafeBinder { .. } => "place_unwrap_unsafe_binder",
            ExprKind::ValueUnwrapUnsafeBinder { .. } => "value_unwrap_unsafe_binder",
            ExprKind::WrapUnsafeBinder { .. } => "wrap_unsafe_binder",
            _ => "expression",
        }
    }

    impl<'a, 'tcx> Visitor<'a, 'tcx> for CensusVisitor<'a, 'tcx> {
        fn thir(&self) -> &'a Thir<'tcx> {
            self.thir
        }

        fn visit_block(&mut self, block: &'a Block) {
            let previous = self.suppress_builtin;
            if matches!(block.safety_mode, BlockSafety::BuiltinUnsafe) {
                self.suppress_builtin = true;
            }
            visit::walk_block(self, block);
            self.suppress_builtin = previous;
        }

        fn visit_pat(&mut self, pat: &'a Pat<'tcx>) {
            if self.in_union_destructure
                && matches!(
                    pat.kind,
                    PatKind::Binding { .. }
                        | PatKind::Constant { .. }
                        | PatKind::Variant { .. }
                        | PatKind::Leaf { .. }
                        | PatKind::Deref { .. }
                        | PatKind::DerefPattern { .. }
                        | PatKind::Range { .. }
                        | PatKind::Slice { .. }
                        | PatKind::Array { .. }
                        | PatKind::Guard { .. }
                        | PatKind::Never
                )
            {
                self.record(
                    UnsafeSiteKind::UnionFieldAccess,
                    pat.span,
                    None,
                    Vec::new(),
                    None,
                );
                return;
            }
            match &pat.kind {
                PatKind::Leaf { subpatterns, .. } => {
                    if let ty::Adt(def, ..) = pat.ty.kind() {
                        for field in subpatterns {
                            if def.non_enum_variant().fields[field.field]
                                .safety
                                .is_unsafe()
                            {
                                self.record(
                                    UnsafeSiteKind::UnsafeFieldUse,
                                    field.pattern.span,
                                    Some(self.tcx.def_path_str(def.did())),
                                    Vec::new(),
                                    None,
                                );
                            }
                        }
                        if def.is_union() {
                            let previous = mem::replace(&mut self.in_union_destructure, true);
                            visit::walk_pat(self, pat);
                            self.in_union_destructure = previous;
                        } else {
                            visit::walk_pat(self, pat);
                        }
                    } else {
                        visit::walk_pat(self, pat);
                    }
                }
                PatKind::Variant {
                    adt_def,
                    variant_index,
                    subpatterns,
                    ..
                } => {
                    for field in subpatterns {
                        if adt_def.variant(*variant_index).fields[field.field]
                            .safety
                            .is_unsafe()
                        {
                            self.record(
                                UnsafeSiteKind::UnsafeFieldUse,
                                field.pattern.span,
                                Some(self.tcx.def_path_str(adt_def.did())),
                                Vec::new(),
                                None,
                            );
                        }
                    }
                    visit::walk_pat(self, pat);
                }
                _ => visit::walk_pat(self, pat),
            }
        }

        fn visit_expr(&mut self, expr: &'a Expr<'tcx>) {
            match expr.kind {
                ExprKind::Scope { value, hir_id, .. } => {
                    let previous = self.hir_context;
                    self.hir_context = hir_id;
                    self.visit_expr(&self.thir[value]);
                    self.hir_context = previous;
                    return;
                }
                ExprKind::Call { fun, ref args, .. } => {
                    let function_ty = self.thir[fun].ty;
                    let signature = function_ty.fn_sig(self.tcx);
                    let (callee_features, safe_target_features): (&[_], _) =
                        match *function_ty.kind() {
                            ty::FnDef(def, ..) => {
                                let attrs = self.tcx.codegen_fn_attrs(def);
                                (&attrs.target_features, attrs.safe_target_features)
                            }
                            _ => (&[], false),
                        };
                    let mut sites = Vec::new();
                    if signature.safety().is_unsafe() && !safe_target_features {
                        let callee = match function_ty.kind() {
                            ty::FnDef(def, _) => Some(self.tcx.def_path_str(*def)),
                            _ => None,
                        };
                        sites.push((UnsafeSiteKind::UnsafeCall, callee));
                    } else if let ty::FnDef(def, _) = function_ty.kind() {
                        if !self
                            .tcx
                            .is_target_feature_call_safe(callee_features, self.body_target_features)
                        {
                            let _missing = callee_features
                                .iter()
                                .filter(|feature| feature.kind == TargetFeatureKind::Enabled)
                                .count();
                            sites.push((
                                UnsafeSiteKind::TargetFeatureCall,
                                Some(self.tcx.def_path_str(*def)),
                            ));
                        }
                        if let Some(trait_def) = self.tcx.trait_of_assoc(*def) {
                            if self.tcx.is_lang_item(trait_def, LangItem::Drop) {
                                sites.push((
                                    UnsafeSiteKind::ExplicitDropCall,
                                    Some(self.tcx.def_path_str(*def)),
                                ));
                            }
                        }
                    }
                    if !sites.is_empty() {
                        let mut operands = Vec::new();
                        if let Some(anchor) = self.anchor(fun, OperandRole::Callee) {
                            operands.push(anchor);
                        }
                        for (index, argument) in args.iter().copied().enumerate() {
                            if let Some(anchor) =
                                self.anchor(argument, OperandRole::Argument(index as u32))
                            {
                                operands.push(anchor);
                            }
                        }
                        let result = self.result(expr);
                        for (kind, callee) in sites {
                            self.record(kind, expr.span, callee, operands.clone(), result.clone());
                        }
                    }
                }
                ExprKind::RawBorrow { arg, .. } => {
                    if let ExprKind::Scope { value: scoped, .. } = self.thir[arg].kind {
                        if let ExprKind::Deref { arg: deref } = self.thir[scoped].kind {
                            visit::walk_expr(self, &self.thir[deref]);
                            return;
                        }
                    }
                    let mut peeled = arg;
                    loop {
                        let ExprKind::Scope { value: scoped, .. } = self.thir[peeled].kind else {
                            break;
                        };
                        let ExprKind::Field { lhs, .. } = self.thir[scoped].kind else {
                            break;
                        };
                        let ty::Adt(def, _) = self.thir[lhs].ty.kind() else {
                            break;
                        };
                        if !def.is_union() {
                            break;
                        }
                        peeled = lhs;
                    }
                    visit::walk_expr(self, &self.thir[peeled]);
                    return;
                }
                ExprKind::Deref { arg } => {
                    if let ExprKind::StaticRef { def_id, .. } | ExprKind::ThreadLocalRef(def_id) =
                        self.thir[arg].kind
                    {
                        let kind = if self.tcx.is_mutable_static(def_id) {
                            Some(UnsafeSiteKind::MutableStaticAccess)
                        } else if self.tcx.is_foreign_item(def_id)
                            && !matches!(
                                self.tcx.def_kind(def_id),
                                DefKind::Static {
                                    safety: hir::Safety::Safe,
                                    ..
                                }
                            )
                        {
                            Some(UnsafeSiteKind::ExternStaticAccess)
                        } else {
                            None
                        };
                        if let Some(kind) = kind {
                            let operands =
                                self.anchor(arg, OperandRole::Place).into_iter().collect();
                            let result = self.result(expr);
                            self.record(
                                kind,
                                expr.span,
                                Some(self.tcx.def_path_str(def_id)),
                                operands,
                                result,
                            );
                        }
                    } else if self.thir[arg].ty.is_raw_ptr() {
                        let operands = self.anchor(arg, OperandRole::Address).into_iter().collect();
                        let result = self.result(expr);
                        self.record(
                            UnsafeSiteKind::RawPointerDereference,
                            expr.span,
                            None,
                            operands,
                            result,
                        );
                    }
                }
                ExprKind::InlineAsm(box InlineAsmExpr {
                    asm_macro,
                    ref operands,
                    ..
                }) => {
                    if matches!(asm_macro, AsmMacro::Asm) {
                        let result = self.result(expr);
                        self.record(
                            UnsafeSiteKind::InlineAssembly,
                            expr.span,
                            None,
                            Vec::new(),
                            result,
                        );
                    }
                    for operand in &**operands {
                        use rustc_middle::thir::InlineAsmOperand::*;
                        match operand {
                            In { expr, .. }
                            | Out {
                                expr: Some(expr), ..
                            }
                            | InOut { expr, .. } => self.visit_expr(&self.thir[*expr]),
                            SplitInOut {
                                in_expr, out_expr, ..
                            } => {
                                self.visit_expr(&self.thir[*in_expr]);
                                if let Some(out_expr) = out_expr {
                                    self.visit_expr(&self.thir[*out_expr]);
                                }
                            }
                            Label { block } => {
                                let previous = mem::replace(&mut self.suppress_builtin, false);
                                visit::walk_block(self, &self.thir[*block]);
                                self.suppress_builtin = previous;
                            }
                            _ => {}
                        }
                    }
                    return;
                }
                ExprKind::Adt(box AdtExpr {
                    adt_def,
                    variant_index,
                    ..
                }) if adt_def.variant(variant_index).has_unsafe_fields() => {
                    let result = self.result(expr);
                    self.record(
                        UnsafeSiteKind::UnsafeFieldInitialization,
                        expr.span,
                        Some(self.tcx.def_path_str(adt_def.did())),
                        Vec::new(),
                        result,
                    );
                }
                ExprKind::Closure(box ClosureExpr { closure_id, .. }) => {
                    self.visit_inner_body(closure_id);
                }
                ExprKind::ConstBlock { did, .. } => self.visit_inner_body(did.expect_local()),
                ExprKind::Field {
                    lhs,
                    variant_index,
                    name,
                } => {
                    if let ty::Adt(def, _) = self.thir[lhs].ty.kind() {
                        let kind = if def.variant(variant_index).fields[name].safety.is_unsafe() {
                            Some(UnsafeSiteKind::UnsafeFieldUse)
                        } else if def.is_union() && !self.assignment_info {
                            Some(UnsafeSiteKind::UnionFieldAccess)
                        } else {
                            None
                        };
                        if let Some(kind) = kind {
                            let operands =
                                self.anchor(lhs, OperandRole::Place).into_iter().collect();
                            let result = self.result(expr);
                            self.record(kind, expr.span, None, operands, result);
                        }
                    }
                }
                ExprKind::Assign { lhs, rhs } => {
                    self.assignment_info = true;
                    visit::walk_expr(self, &self.thir[lhs]);
                    self.assignment_info = false;
                    visit::walk_expr(self, &self.thir[rhs]);
                    return;
                }
                ExprKind::PlaceUnwrapUnsafeBinder { .. }
                | ExprKind::ValueUnwrapUnsafeBinder { .. }
                | ExprKind::WrapUnsafeBinder { .. } => {
                    let result = self.result(expr);
                    self.record(
                        UnsafeSiteKind::UnsafeBinderCast,
                        expr.span,
                        None,
                        Vec::new(),
                        result,
                    );
                }
                _ => {}
            }
            visit::walk_expr(self, expr);
        }
    }

    pub fn sweep(tcx: TyCtxt<'_>) {
        for def in tcx.hir_body_owners() {
            if !tcx.is_typeck_child(def.to_def_id()) {
                collect_and_record(tcx, def);
            }
        }
    }
}

#[rustversion::since(1.96)]
pub use collector::{install, sweep};
