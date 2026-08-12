/*
    This is a cargo program to start RAP.
    The file references the cargo file for Miri: https://github.com/rust-lang/miri/blob/master/cargo-miri/src/main.rs
*/
#![feature(rustc_private)]

#[macro_use]
extern crate rapx;

use rapx::utils::log::{init_log, rap_error_and_exit};
use rapx::utils::unit_output::{
    dispatch_phase_from_argv, nested_workspace_wrapper_reason, DispatchPhase, UnitRoute,
};
use std::env;

mod args;
mod help;

mod utils;
use crate::utils::*;

mod cargo_check;

fn phase_cargo_rap() {
    rap_trace!("Start cargo-rapx.");

    // here we skip two args: cargo rapx
    let Some(arg) = args::get_arg(2) else {
        rap_error!("Expect command: e.g., `cargo rapx -help`.");
        return;
    };
    match arg {
        "-V" | "-version" => {
            rap_info!("{}", help::RAPX_VERSION);
            return;
        }
        "-H" | "-help" | "--help" => {
            rap_info!("{}", help::RAPX_HELP);
            return;
        }
        _ => {}
    }

    cargo_check::run();
}

fn phase_rustc_wrapper() {
    rap_trace!("Launch cargo-rapx again triggered by cargo check.");

    match args::unit_route().unwrap_or_else(|err| rap_error_and_exit(err)) {
        UnitRoute::PassThrough => run_rustc(),
        UnitRoute::Analyze(identity) => {
            let nested_wrapper = env::var_os("RUSTC_WORKSPACE_WRAPPER");
            let nested_wrapper_reason = nested_wrapper.as_ref().and_then(|value| {
                (!value.is_empty())
                    .then(|| nested_workspace_wrapper_reason(Some("configured")))
                    .flatten()
            });
            if let Some(reason) = nested_wrapper_reason {
                rap_error_and_exit(format!(
                    "{{\"reason\":\"{reason}\",\"scope\":\"local_analysis_unit\"}}"
                ));
            }
            run_rap(identity);
        }
        UnitRoute::Skip { identity, reason } => {
            write_skip_receipt(&identity, reason);
            run_rustc();
        }
    }
}

fn main() {
    /* This function will be enteredd twice:
       1. When we run `cargo rapx ...`, cargo dispatches the execution to cargo-rapx.
      In this step, we set RUSTC_WRAPPER to cargo-rapx, and execute `cargo check ...` command;
       2. Cargo check actually triggers `path/cargo-rapx path/rustc` according to RUSTC_WRAPPER.
          Because RUSTC_WRAPPER is defined, Cargo calls the command: `$RUSTC_WRAPPER path/rustc ...`
    */

    // Init the log_system
    init_log().expect("Failed to init log.");

    match dispatch_phase_from_argv(args::all_args()) {
        Ok(DispatchPhase::CargoRapx) => phase_cargo_rap(),
        Ok(DispatchPhase::RustcWrapper) => phase_rustc_wrapper(),
        Err(err) => rap_error_and_exit(err),
    }
}
