#![feature(rustc_private)]
#![feature(box_patterns)]

#[macro_use]
pub mod utils;
pub mod analysis;

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_abi;

#[cfg(feature = "full_analyses")]
use analysis::api_dep::ApiDep;
#[cfg(feature = "full_analyses")]
use analysis::core::alias::mop::MopAlias;
#[cfg(feature = "full_analyses")]
use analysis::core::call_graph::CallGraph;
#[cfg(feature = "full_analyses")]
use analysis::core::dataflow::DataFlow;
#[cfg(feature = "full_analyses")]
use analysis::opt::Opt;
#[cfg(feature = "full_analyses")]
use analysis::rcanary::rCanary;
#[cfg(feature = "full_analyses")]
use analysis::safedrop::SafeDrop;
#[cfg(feature = "full_analyses")]
use analysis::senryx::SenryxCheck;
#[cfg(feature = "full_analyses")]
use analysis::unsafety_isolation::{UigInstruction, UnsafetyIsolationCheck};
#[cfg(feature = "full_analyses")]
use analysis::utils::show_mir::ShowMir;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;
use rustc_interface::Config;
#[rustversion::before(1.96)]
use rustc_data_structures::sync::Lrc;
#[rustversion::before(1.96)]
use rustc_interface::Queries;
use rustc_middle::ty::TyCtxt;
#[rustversion::before(1.96)]
use rustc_middle::util::Providers;
#[rustversion::before(1.96)]
use rustc_session::search_paths::PathKind;
#[rustversion::before(1.96)]
use std::path::PathBuf;

// Insert rustc arguments at the beginning of the argument list that RAP wants to be
// set per default, for maximal validation power.
pub static RAP_DEFAULT_ARGS: &[&str] = &["-Zalways-encode-mir", "-Zmir-opt-level=0"];

pub type Elapsed = (i64, i64);

#[derive(Debug, Copy, Clone, Hash)]
pub struct RapCallback {
    rcanary: bool,
    safedrop: bool,
    annotation: bool,
    unsafety_isolation: usize,
    mop: bool,
    callgraph: bool,
    api_dep: bool,
    show_mir: bool,
    dataflow: usize,
    opt: bool,
    unsoundaudit: bool,
}

impl Default for RapCallback {
    fn default() -> Self {
        Self {
            rcanary: false,
            safedrop: false,
            annotation: false,
            unsafety_isolation: 0,
            mop: false,
            callgraph: false,
            api_dep: false,
            show_mir: false,
            dataflow: 0,
            opt: false,
            unsoundaudit: false,
        }
    }
}

impl Callbacks for RapCallback {
    fn config(&mut self, config: &mut Config) {
        configure_override_queries(config);
    }

    #[rustversion::before(1.96)]
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &Compiler,
        queries: &'tcx Queries<'tcx>,
    ) -> Compilation {
        rap_trace!("Execute after_analysis() of compiler callbacks");
        queries
            .global_ctxt()
            .unwrap()
            .enter(|tcx| start_analyzer(tcx, *self));
        rap_trace!("analysis done");
        Compilation::Continue
    }

    #[rustversion::since(1.96)]
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        rap_trace!("Execute after_analysis() of compiler callbacks");
        start_analyzer(tcx, *self);
        rap_trace!("analysis done");
        Compilation::Continue
    }
}

#[rustversion::before(1.96)]
fn configure_override_queries(config: &mut Config) {
    config.override_queries = Some(|_, providers| {
        providers.extern_queries.used_crate_source = |tcx, cnum| {
            let mut providers = Providers::default();
            rustc_metadata::provide(&mut providers);
            let mut crate_source = (providers.extern_queries.used_crate_source)(tcx, cnum);
            // HACK: rustc will emit "crate ... required to be available in rlib format, but
            // was not found in this form" errors once we use `tcx.dependency_formats()` if
            // there's no rlib provided, so setting a dummy path here to workaround those errors.
            Lrc::make_mut(&mut crate_source).rlib = Some((PathBuf::new(), PathKind::All));
            crate_source
        };
    });
}

#[rustversion::since(1.96)]
fn configure_override_queries(_config: &mut Config) {}

impl RapCallback {
    pub fn enable_rcanary(&mut self) {
        self.rcanary = true;
    }

    pub fn is_rcanary_enabled(&self) -> bool {
        self.rcanary
    }

    pub fn enable_mop(&mut self) {
        self.mop = true;
    }

    pub fn is_mop_enabled(&self) -> bool {
        self.mop
    }

    pub fn enable_safedrop(&mut self) {
        self.safedrop = true;
    }

    pub fn is_safedrop_enabled(&self) -> bool {
        self.safedrop
    }

    pub fn enable_unsafety_isolation(&mut self, x: usize) {
        self.unsafety_isolation = x;
    }

    pub fn is_unsafety_isolation_enabled(&self) -> usize {
        self.unsafety_isolation
    }

    pub fn enable_api_dep(&mut self) {
        self.api_dep = true;
    }

    pub fn is_api_dep_enabled(self) -> bool {
        self.api_dep
    }

    pub fn enable_annotation(&mut self) {
        self.annotation = true;
    }

    pub fn is_annotation_enabled(&self) -> bool {
        self.annotation
    }

    pub fn enable_callgraph(&mut self) {
        self.callgraph = true;
    }

    pub fn is_callgraph_enabled(&self) -> bool {
        self.callgraph
    }

    pub fn enable_show_mir(&mut self) {
        self.show_mir = true;
    }

    pub fn is_show_mir_enabled(&self) -> bool {
        self.show_mir
    }

    pub fn enable_dataflow(&mut self, x: usize) {
        self.dataflow = x;
    }

    pub fn is_dataflow_enabled(self) -> usize {
        self.dataflow
    }

    pub fn enable_opt(&mut self) {
        self.opt = true;
    }

    pub fn is_opt_enabled(self) -> bool {
        self.opt
    }

    pub fn enable_unsoundaudit(&mut self) {
        self.unsoundaudit = true;
    }

    pub fn is_lwz_enabled(self) -> bool {
        self.unsoundaudit
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum RapPhase {
    Cleanup,
    Cargo,
    Rustc,
    LLVM, // unimplemented yet
}

/// Returns the "default sysroot" that RAP will use if no `--sysroot` flag is set.
/// Should be a compile-time constant.
pub fn compile_time_sysroot() -> Option<String> {
    // Optionally inspects an environment variable at compile time.
    if option_env!("RUSTC_STAGE").is_some() {
        // This is being built as part of rustc, and gets shipped with rustup.
        // We can rely on the sysroot computation in rustc.
        return None;
    }
    if let Some(sysroot) = option_env!("RUST_SYSROOT") {
        return Some(sysroot.to_string());
    }
    // For builds outside rustc, we need to ensure that we got a sysroot
    // that gets used as a default.  The sysroot computation in librustc_session would
    // end up somewhere in the build dir (see `get_or_default_sysroot`).
    // Taken from PR <https://github.com/Manishearth/rust-clippy/pull/911>.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    let env = if home.is_some() && toolchain.is_some() {
        format!("{}/toolchains/{}", home.unwrap(), toolchain.unwrap())
    } else {
        option_env!("RUST_SYSROOT")
            .expect("To build RAPx without rustup, set the `RUST_SYSROOT` env var at build time")
            .to_string()
    };
    Some(env)
}

#[rustversion::before(1.83)]
fn install_source_map_compat(tcx: TyCtxt<'_>) {
    crate::utils::log::install_source_map(tcx.sess.parse_sess.clone_source_map());
}

#[rustversion::since(1.83)]
fn install_source_map_compat(_tcx: TyCtxt<'_>) {}

pub fn start_analyzer(tcx: TyCtxt, callback: RapCallback) {
    install_source_map_compat(tcx);

    run_optional_analyses(tcx, callback);

    if callback.is_lwz_enabled() {
        println!("UnsoundAudit is enabled");
        analysis::unsoundaudit::LwzCheck::new(tcx).start();
    }
}

#[cfg(feature = "full_analyses")]
fn run_optional_analyses(tcx: TyCtxt, callback: RapCallback) {
    let _rcanary: Option<rCanary> = if callback.is_rcanary_enabled() {
        let mut rcx = rCanary::new(tcx);
        rcx.start();
        Some(rcx)
    } else {
        None
    };

    if callback.is_mop_enabled() {
        MopAlias::new(tcx).start();
    }

    if callback.is_safedrop_enabled() {
        SafeDrop::new(tcx).start();
    }

    let x = callback.is_unsafety_isolation_enabled();
    match x {
        1 => UnsafetyIsolationCheck::new(tcx).start(UigInstruction::StdSp),
        2 => UnsafetyIsolationCheck::new(tcx).start(UigInstruction::Doc),
        3 => UnsafetyIsolationCheck::new(tcx).start(UigInstruction::Upg),
        4 => UnsafetyIsolationCheck::new(tcx).start(UigInstruction::Ucons),
        _ => {}
    }

    if callback.is_annotation_enabled() {
        SenryxCheck::new(tcx, 2).start();
    }

    if callback.is_show_mir_enabled() {
        ShowMir::new(tcx).start();
    }

    if callback.is_api_dep_enabled() {
        ApiDep::new(tcx).start();
    }

    match callback.is_dataflow_enabled() {
        1 => DataFlow::new(tcx, false).start(),
        2 => DataFlow::new(tcx, true).start(),
        _ => {}
    }

    if callback.is_callgraph_enabled() {
        CallGraph::new(tcx).start();
    }

    if callback.is_opt_enabled() {
        Opt::new(tcx).start();
    }
}

#[cfg(not(feature = "full_analyses"))]
fn run_optional_analyses(_tcx: TyCtxt, _callback: RapCallback) {}
