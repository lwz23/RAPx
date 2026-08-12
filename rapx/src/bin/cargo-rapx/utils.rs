use crate::args;
use rapx::utils::unit_output::{
    json_output_path, rustc_commit_from_runner, rustc_path_from_wrapper_args,
    validate_exact_tool_path, verify_expected_commit, write_skip_receipt as write_skip_receipt_file,
    UnitIdentity,
};
use std::{
    env, fs,
    path::PathBuf,
    process::{self, Command},
};

#[path = "../shared/rap_arg_boundary.rs"]
mod rap_arg_boundary;
use rap_arg_boundary::plan_wrapper_invocation;

fn find_rap() -> PathBuf {
    let mut path = args::current_exe_path().to_owned();
    path.set_file_name("rapx");
    path
}

pub fn run_cmd(mut cmd: Command) {
    rap_trace!("Command is: {:?}.", cmd);
    match cmd.status() {
        Ok(status) => {
            if !status.success() {
                process::exit(status.code().unwrap());
            }
        }
        Err(err) => panic!("Error in running {:?} {}.", cmd, err),
    }
}

pub fn run_rustc() {
    let rustc_path = rustc_path_from_wrapper_args(args::all_args())
        .expect("Cargo wrapper invocation must include the rustc path at argv[1]");
    let mut cmd = Command::new(rustc_path);
    cmd.args(args::skip2());
    run_cmd(cmd);
}

pub fn run_rap(identity: UnitIdentity) {
    let mut cmd = Command::new(find_rap());
    let serialized_rap_args = env::var("RAP_ARGS").expect("Missing RAP_ARGS.");
    let invocation =
        plan_wrapper_invocation(args::skip2().to_vec(), serialized_rap_args);
    cmd.args(invocation.argv);
    for (key, value) in invocation.environment {
        cmd.env(key, value);
    }
    let cargo_supplied_rustc = rustc_path_from_wrapper_args(args::all_args())
        .expect("Cargo wrapper invocation must include the rustc path at argv[1]");
    let expected_rustc_path_value = env::var("UNSOUND_SCANNER_EXPECTED_RUSTC_PATH").ok();
    let expected_rustc_path = validate_exact_tool_path(
        expected_rustc_path_value.as_deref(),
        "expected exact rustc",
    )
    .unwrap_or_else(|err| rapx::utils::log::rap_error_and_exit(err));
    let cargo_supplied_rustc_path =
        validate_exact_tool_path(Some(cargo_supplied_rustc), "Cargo-supplied rustc")
            .unwrap_or_else(|err| rapx::utils::log::rap_error_and_exit(err));
    if cargo_supplied_rustc_path != expected_rustc_path {
        rapx::utils::log::rap_error_and_exit(format!(
            "Cargo-supplied rustc path mismatch: expected {}, got {}",
            expected_rustc_path.display(),
            cargo_supplied_rustc_path.display()
        ));
    }
    if PathBuf::from(&identity.rustc_path) != cargo_supplied_rustc_path {
        rapx::utils::log::rap_error_and_exit(
            "routed unit rustc path differs from Cargo-supplied rustc path",
        );
    }
    let actual_rustc_commit = rustc_commit_from_runner(&identity.rustc_path, |rustc_path| {
        let output = Command::new(rustc_path)
            .arg("-vV")
            .output()
            .map_err(|err| format!("failed to run '{rustc_path} -vV': {err}"))?;
        if !output.status.success() {
            return Err(format!("'{rustc_path} -vV' exited with {}", output.status));
        }
        String::from_utf8(output.stdout)
            .map_err(|err| format!("'{rustc_path} -vV' did not emit UTF-8: {err}"))
    })
    .unwrap_or_else(|err| rapx::utils::log::rap_error_and_exit(err));
    let expected_rustc_commit = env::var("UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT").ok();
    let rustc_commit = verify_expected_commit(
        &actual_rustc_commit,
        expected_rustc_commit.as_deref(),
        "Cargo-supplied rustc",
    )
    .unwrap_or_else(|err| rapx::utils::log::rap_error_and_exit(err));
    for (key, value) in identity.child_environment() {
        cmd.env(key, value);
    }
    cmd.env("UNSOUND_SCANNER_CARGO_SUPPLIED_RUSTC_COMMIT", rustc_commit);

    let json_dir = env::var_os("UNSOUND_SCANNER_RAP_JSON_DIR").map(PathBuf::from);
    let legacy_output = env::var_os("UNSOUND_SCANNER_RAP_JSON_OUT").map(PathBuf::from);
    if let Some(output_path) = json_output_path(
        json_dir.as_deref(),
        legacy_output.as_deref(),
        &identity.unit_id,
    ) {
        if let Some(json_dir) = json_dir {
            fs::create_dir_all(&json_dir).expect("Failed to create RAP JSON output directory");
        }
        cmd.env("UNSOUND_SCANNER_RAP_JSON_OUT", output_path);
    }
    run_cmd(cmd);
}

pub fn write_skip_receipt(identity: &UnitIdentity, reason: &str) {
    let json_dir = env::var_os("UNSOUND_SCANNER_RAP_JSON_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            rapx::utils::log::rap_error_and_exit(
                "UNSOUND_SCANNER_RAP_JSON_DIR is required for per-unit skip receipts",
            )
        });
    fs::create_dir_all(&json_dir).expect("Failed to create RAP JSON output directory");
    let output_path = json_dir.join(format!("{}.json", identity.unit_id));
    let project_root = env::var("UNSOUND_SCANNER_PROJECT_ROOT")
        .unwrap_or_else(|_| "<missing>".to_string());
    write_skip_receipt_file(&output_path, identity, reason, &project_root).unwrap_or_else(|err| {
        rapx::utils::log::rap_error_and_exit(format!(
            "Failed to atomically write RAP skip receipt '{}': {err}",
            output_path.display()
        ))
    });
}
