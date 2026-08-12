#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_session;

use rapx::{
    compile_time_sysroot, rap_info, rap_trace,
    utils::log::{init_log, rap_error_and_exit},
    RapCallback, RAP_DEFAULT_ARGS,
};
use rustc_session::config::ErrorOutputType;
#[rustversion::before(1.83)]
use rustc_session::EarlyErrorHandler as EarlyDiagCtxt;
#[rustversion::since(1.83)]
use rustc_session::EarlyDiagCtxt;
use std::env;

#[path = "shared/rap_arg_boundary.rs"]
mod rap_arg_boundary;
use rap_arg_boundary::{plan_rap_invocation, RapAnalysis};

fn prepare_invocation() -> Result<(Vec<String>, RapCallback), String> {
    let serialized_options = match env::var("RAP_ARGS") {
        Ok(serialized) => Some(
            serde_json::from_str::<Vec<String>>(&serialized)
                .map_err(|err| format!("Failed to deserialize RAP_ARGS: {err}")),
        ),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => {
            Some(Err("RAP_ARGS must be valid Unicode JSON".to_owned()))
        }
    };
    let plan = plan_rap_invocation(env::args().collect(), serialized_options)?;
    let mut compiler = RapCallback::default();
    for analysis in plan.analyses {
        match analysis {
            RapAnalysis::SafeDrop => compiler.enable_safedrop(),
            RapAnalysis::RCanary => compiler.enable_rcanary(),
            RapAnalysis::Mop => compiler.enable_mop(),
            RapAnalysis::Dataflow(level) => compiler.enable_dataflow(level),
            RapAnalysis::UnsafetyIsolation(level) => {
                compiler.enable_unsafety_isolation(level)
            }
            RapAnalysis::Annotation => compiler.enable_annotation(),
            RapAnalysis::CallGraph => compiler.enable_callgraph(),
            RapAnalysis::Opt => compiler.enable_opt(),
            RapAnalysis::ShowMir => compiler.enable_show_mir(),
            RapAnalysis::ApiDep => compiler.enable_api_dep(),
            RapAnalysis::UnsoundAudit => compiler.enable_unsoundaudit(),
        }
    }
    Ok((plan.compiler_argv, compiler))
}

#[rustversion::before(1.96)]
fn run_complier(args: &mut Vec<String>, callback: &mut RapCallback) -> i32 {
    if let Some(sysroot) = compile_time_sysroot() {
        let sysroot_flag = "--sysroot";
        if !args.iter().any(|e| e == sysroot_flag) {
            // We need to overwrite the default that librustc_session would compute.
            args.push(sysroot_flag.to_owned());
            args.push(sysroot);
        }
    }
    // Finally, add the default flags all the way in the beginning, but after the binary name.
    args.splice(1..1, RAP_DEFAULT_ARGS.iter().map(ToString::to_string));

    let handler = EarlyDiagCtxt::new(ErrorOutputType::default());
    rustc_driver::init_rustc_env_logger(&handler);
    rustc_driver::install_ice_hook("bug_report_url", |_| ());

    let run_compiler = rustc_driver::RunCompiler::new(&args, callback);
    let exit_code = rustc_driver::catch_with_exit_code(move || run_compiler.run());
    rap_trace!("The arg for compilation is {:?}", args);

    exit_code
}

#[rustversion::since(1.96)]
fn run_complier(args: &mut Vec<String>, callback: &mut RapCallback) -> std::process::ExitCode {
    if let Some(sysroot) = compile_time_sysroot() {
        let sysroot_flag = "--sysroot";
        if !args.iter().any(|e| e == sysroot_flag) {
            args.push(sysroot_flag.to_owned());
            args.push(sysroot);
        }
    }
    args.splice(1..1, RAP_DEFAULT_ARGS.iter().map(ToString::to_string));

    let handler = EarlyDiagCtxt::new(ErrorOutputType::default());
    rustc_driver::init_rustc_env_logger(&handler);
    rustc_driver::install_ice_hook("bug_report_url", |_| ());

    let exit_code = rustc_driver::catch_with_exit_code(|| {
        rustc_driver::run_compiler(args, callback);
    });
    rap_trace!("The arg for compilation is {:?}", args);

    exit_code
}

#[rustversion::before(1.96)]
fn main() {
    let (mut args, mut compiler) =
        prepare_invocation().unwrap_or_else(|err| rap_error_and_exit(err));
    if let Err(err) = init_log() {
        eprintln!("Failed to init log: {err}");
    }
    rap_info!("Start analysis with RAP.");
    rap_trace!("rap received arguments{:#?}", env::args());
    rap_trace!("arguments to rustc: {:?}", &args);

    let exit_code = run_complier(&mut args, &mut compiler);
    std::process::exit(exit_code)
}

#[rustversion::since(1.96)]
fn main() -> std::process::ExitCode {
    let (mut args, mut compiler) =
        prepare_invocation().unwrap_or_else(|err| rap_error_and_exit(err));
    if let Err(err) = init_log() {
        eprintln!("Failed to init log: {err}");
    }
    rap_info!("Start analysis with RAP.");
    rap_trace!("rap received arguments{:#?}", env::args());
    rap_trace!("arguments to rustc: {:?}", &args);

    run_complier(&mut args, &mut compiler)
}
