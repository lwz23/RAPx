use crate::args;
use cargo_metadata::camino::Utf8Path;
use rapx::utils::log::rap_error_and_exit;
use rapx::utils::unit_output::{
    parse_inner_timeout_seconds, parse_rustc_commit, validate_exact_tool_path,
    verify_expected_commit,
};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
    time::Duration,
};
use wait_timeout::ChildExt;

mod workspace;

static EXACT_CARGO_PATH: OnceLock<PathBuf> = OnceLock::new();
static EXACT_RUSTC_PATH: OnceLock<PathBuf> = OnceLock::new();

fn tool_commit(path: &Path, field: &str) -> Result<String, String> {
    let output = Command::new(path)
        .arg("-Vv")
        .output()
        .map_err(|err| format!("failed to run '{} -Vv': {err}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "'{} -Vv' exited with {} while verifying {field}",
            path.display(),
            output.status
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| format!("'{} -Vv' did not emit UTF-8: {err}", path.display()))?;
    parse_rustc_commit(&stdout)
}

fn verified_tool_from_environment(
    path_variable: &str,
    commit_variable: &str,
    field: &str,
) -> Result<PathBuf, String> {
    let configured_path = env::var(path_variable).ok();
    let path = validate_exact_tool_path(configured_path.as_deref(), field)?;
    let actual_commit = tool_commit(&path, field)?;
    let expected_commit = env::var(commit_variable).ok();
    verify_expected_commit(&actual_commit, expected_commit.as_deref(), field)?;
    Ok(path)
}

pub(super) fn exact_cargo_path() -> &'static Path {
    EXACT_CARGO_PATH.get_or_init(|| {
        verified_tool_from_environment(
            "UNSOUND_SCANNER_EXACT_CARGO_PATH",
            "UNSOUND_SCANNER_EXACT_CARGO_COMMIT",
            "exact Cargo",
        )
        .unwrap_or_else(|err| rap_error_and_exit(err))
    })
}

fn exact_rustc_path() -> &'static Path {
    EXACT_RUSTC_PATH.get_or_init(|| {
        let expected_path_value = env::var("UNSOUND_SCANNER_EXPECTED_RUSTC_PATH").ok();
        let expected_path = validate_exact_tool_path(
            expected_path_value.as_deref(),
            "expected exact rustc",
        )
        .unwrap_or_else(|err| rap_error_and_exit(err));
        let configured_path_value = env::var("RUSTC").ok();
        let configured_path =
            validate_exact_tool_path(configured_path_value.as_deref(), "configured rustc")
                .unwrap_or_else(|err| rap_error_and_exit(err));
        if configured_path != expected_path {
            rap_error_and_exit(format!(
                "configured rustc path mismatch: expected {}, got {}",
                expected_path.display(),
                configured_path.display()
            ));
        }
        let actual_commit = tool_commit(&configured_path, "exact rustc")
            .unwrap_or_else(|err| rap_error_and_exit(err));
        let expected_commit = env::var("UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT").ok();
        verify_expected_commit(
            &actual_commit,
            expected_commit.as_deref(),
            "exact rustc",
        )
        .unwrap_or_else(|err| rap_error_and_exit(err));
        configured_path
    })
}

pub fn run() {
    match env::var("RAP_RECURSIVE")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("none") | None => default_run(),
        Some("deep") => workspace::deep_run(),
        Some("shallow") => workspace::shallow_run(),
        _ => rap_error_and_exit(
            "`recursive` should only accept one the values: none, shallow or deep.",
        ),
    }
}

fn cargo_check(dir: &Utf8Path) {
    // always clean before check due to outdated except `RAP_CLEAN` is false
    rap_trace!("cargo clean in package folder {dir}");
    cargo_clean(dir, args::rap_clean());

    rap_trace!("cargo check in package folder {dir}");
    let [rap_args, cargo_args] = args::rap_and_cargo_args();
    rap_trace!("rap_args={rap_args:?}\tcargo_args={cargo_args:?}");

    /*Here we prepare the cargo command as cargo check, which is similar to build, but much faster*/
    let mut cmd = Command::new(exact_cargo_path());
    cmd.current_dir(dir);
    cmd.arg("check");
    cmd.env("RUSTC", exact_rustc_path());
    cmd.env_remove("RUSTC_WORKSPACE_WRAPPER");
    if env::var_os("UNSOUND_SCANNER_PROJECT_ROOT").is_none() {
        let project_root = Path::new(dir.as_str()).canonicalize().unwrap_or_else(|err| {
            rap_error_and_exit(format!(
                "failed to resolve RAP project root '{}': {err}",
                dir
            ))
        });
        cmd.env("UNSOUND_SCANNER_PROJECT_ROOT", project_root);
    }

    /* set the target as a filter for phase_rustc_rap */
    cmd.args(cargo_args);

    // Serialize the remaining args into a special environment variable.
    // This will be read by `phase_rustc_rap` when we go to invoke
    // our actual target crate (the binary or the test we are running).

    cmd.env(
        "RAP_ARGS",
        serde_json::to_string(rap_args).expect("Failed to serialize args."),
    );

    // Invoke actual cargo for the job, but with different flags.
    let cargo_rap_path = args::current_exe_path();
    cmd.env("RUSTC_WRAPPER", cargo_rap_path);

    rap_trace!("Command is: {:?}.", cmd);

    let timeout = match parse_inner_timeout_seconds(
        env::var("UNSOUND_SCANNER_RAP_INNER_TIMEOUT_SECONDS")
            .ok()
            .as_deref(),
    ) {
        Ok(timeout) => timeout,
        Err(err) => rap_error_and_exit(err),
    };
    let mut child = cmd.spawn().expect("Could not run cargo check.");
    let status = match timeout {
        None => child.wait().expect("Failed to wait for subprocess."),
        Some(seconds) => match child
            .wait_timeout(Duration::from_secs(seconds))
            .expect("Failed to wait for subprocess.")
        {
            Some(status) => status,
            None => {
                child.kill().expect("Failed to kill subprocess.");
                child.wait().expect("Failed to wait for subprocess.");
                rap_error_and_exit("Process killed due to timeout.");
            }
        },
    };
    if !status.success() {
        rap_error_and_exit("Finished with non-zero exit code.");
    }
}

fn cargo_clean(dir: &Utf8Path, really: bool) {
    if really {
        if let Err(err) = Command::new(exact_cargo_path())
            .arg("clean")
            .current_dir(dir)
            .output()
        {
            rap_error_and_exit(format!("`cargo clean` exits unexpectedly:\n{err}"));
        }
    }
}

/// Just like running a cargo check in a folder.
fn default_run() {
    cargo_check(".".into());
}
