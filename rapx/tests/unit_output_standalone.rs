#[path = "../src/utils/unit_output.rs"]
mod unit_output;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Mutex;
use unit_output::{
    atomic_write, atomic_write_with_rename, atomic_write_with_writer, crate_skip_reason,
    build_verification_receipt_from_environment_with_current_exe, dispatch_phase_from_argv,
    embedded_rap_compiler_commit, fnv1a64_hex, json_output_path,
    nested_workspace_wrapper_reason, parse_inner_timeout_seconds, parse_rustc_commit,
    parse_build_verification_challenge, parse_build_verification_mode,
    route_rustc_unit, rustc_commit_from_runner, rustc_path_from_wrapper_args,
    sanitize_file_component, validate_commit_hash, validate_exact_tool_path,
    verify_expected_commit, write_skip_receipt, BuildVerificationActual, DispatchPhase,
    IdentityEnvironment, UnitIdentity, UnitRoute,
};

const VERIFICATION_HASH: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const VERIFICATION_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
static VERIFICATION_ENV_MUTEX: Mutex<()> = Mutex::new(());
const VERIFICATION_ENV_KEYS: &[&str] = &[
    "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_MODE",
    "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT",
    "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT_SHA256",
    "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_CHALLENGE",
    "UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH",
    "UNSOUND_SCANNER_CARGO_SUPPLIED_RUSTC_COMMIT",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
];
#[cfg(target_os = "linux")]
const VERIFICATION_LOADER_VARIABLE: &str = "LD_LIBRARY_PATH";
#[cfg(target_os = "macos")]
const VERIFICATION_LOADER_VARIABLE: &str = "DYLD_LIBRARY_PATH";
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
const VERIFICATION_LOADER_VARIABLE: &str = "LD_LIBRARY_PATH";

struct EnvironmentRestore {
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvironmentRestore {
    fn capture(keys: &[&'static str]) -> Self {
        Self {
            values: keys
                .iter()
                .map(|key| (*key, env::var_os(key)))
                .collect(),
        }
    }

    fn set(&self, key: &str, value: impl AsRef<std::ffi::OsStr>) {
        env::set_var(key, value);
    }

    fn remove(&self, key: &str) {
        env::remove_var(key);
    }
}

impl Drop for EnvironmentRestore {
    fn drop(&mut self) {
        for (key, value) in &self.values {
            match value {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
    }
}

fn verification_actual() -> BuildVerificationActual {
    BuildVerificationActual {
        expected_challenge: VERIFICATION_HASH.to_string(),
        rapx_path: "/verify/bin/rapx".to_string(),
        cargo_rapx_path: "/verify/bin/cargo-rapx".to_string(),
        cargo_supplied_rustc_path: "/toolchain/bin/rustc".to_string(),
        cargo_supplied_rustc_commit: VERIFICATION_COMMIT.to_string(),
        loader_environment_variable: VERIFICATION_LOADER_VARIABLE.to_string(),
        loader_path: "/toolchain/lib".to_string(),
    }
}

fn valid_verification_challenge() -> String {
    format!(
        concat!(
            "{{",
            "\"schema\":\"rap-build-verification-challenge-v1\",",
            "\"challenge\":\"{hash}\",",
            "\"attempt_id\":\"attempt-1\",",
            "\"exact_toolchain\":\"nightly-2026-08-03\",",
            "\"cargo\":{{\"path\":\"/toolchain/bin/cargo\",\"sha256\":\"{hash}\",\"identity\":{{\"component\":\"cargo\",\"release\":\"1.0\",\"commit_hash\":\"{commit}\",\"commit_date\":\"2026-08-03\",\"host\":\"x86_64-unknown-linux-gnu\",\"first_line\":\"cargo 1.0\"}}}},",
            "\"rustc\":{{\"path\":\"/toolchain/bin/rustc\",\"sha256\":\"{hash}\",\"identity\":{{\"component\":\"rustc\",\"release\":\"1.0\",\"commit_hash\":\"{commit}\",\"commit_date\":\"2026-08-03\",\"host\":\"x86_64-unknown-linux-gnu\",\"first_line\":\"rustc 1.0\"}}}},",
            "\"cargo_rapx\":{{\"path\":\"/verify/bin/cargo-rapx\",\"sha256\":\"{hash}\"}},",
            "\"rapx\":{{\"path\":\"/verify/bin/rapx\",\"sha256\":\"{hash}\"}},",
            "\"rap_source\":{{\"path\":\"/source/rap\",\"sha256\":\"{hash}\"}},",
            "\"builder_sidecar\":{{\"path\":\"/verify/input/builder_sidecar.json\",\"sha256\":\"{hash}\"}},",
            "\"loader\":{{\"environment_variable\":\"{loader_variable}\",\"path\":\"/toolchain/lib\",\"contents_sha256\":\"{hash}\"}},",
            "\"fixture\":{{\"project_root\":\"/verify/work/fixture\",\"manifest_sha256\":\"{hash}\",\"source_sha256\":\"{hash}\",\"lock_sha256\":\"{hash}\"}}",
            "}}"
        ),
        hash = VERIFICATION_HASH,
        commit = VERIFICATION_COMMIT,
        loader_variable = VERIFICATION_LOADER_VARIABLE,
    )
}

fn verification_challenge_for(actual: &BuildVerificationActual) -> String {
    valid_verification_challenge()
        .replace("/verify/bin/cargo-rapx", &actual.cargo_rapx_path)
        .replace("/verify/bin/rapx", &actual.rapx_path)
        .replace(
            "/toolchain/bin/rustc",
            &actual.cargo_supplied_rustc_path,
        )
        .replace("/toolchain/lib", &actual.loader_path)
}

#[test]
fn build_verification_mode_is_absent_only_when_unset() {
    assert_eq!(parse_build_verification_mode(None), Ok(None));
    assert!(parse_build_verification_mode(Some("wrong-mode")).is_err());
}

#[test]
fn build_verification_rejects_duplicate_json_keys() {
    let json = valid_verification_challenge().replacen(
        "\"schema\":\"rap-build-verification-challenge-v1\",",
        "\"schema\":\"rap-build-verification-challenge-v1\",\"schema\":\"rap-build-verification-challenge-v1\",",
        1,
    );
    assert!(parse_build_verification_challenge(&json, VERIFICATION_HASH, &verification_actual())
        .expect_err("duplicate keys must be rejected")
        .contains("duplicate"));
}

#[test]
fn build_verification_rejects_missing_and_unknown_challenge_fields() {
    let missing = valid_verification_challenge().replacen("\"attempt_id\":\"attempt-1\",", "", 1);
    assert!(parse_build_verification_challenge(&missing, VERIFICATION_HASH, &verification_actual()).is_err());

    let unknown = valid_verification_challenge().replacen(
        "\"attempt_id\":\"attempt-1\",",
        "\"attempt_id\":\"attempt-1\",\"unexpected\":\"field\",",
        1,
    );
    assert!(parse_build_verification_challenge(&unknown, VERIFICATION_HASH, &verification_actual()).is_err());
}

#[test]
fn build_verification_attempt_id_uses_conservative_shared_grammar() {
    let too_long = "a".repeat(257);
    for invalid in ["", "under:colon", "-leading", too_long.as_str()] {
        let json = valid_verification_challenge().replacen("attempt-1", invalid, 1);
        assert!(parse_build_verification_challenge(
            &json,
            VERIFICATION_HASH,
            &verification_actual(),
        )
        .is_err());
    }

    let valid = valid_verification_challenge().replacen("attempt-1", "A-z_9.ok", 1);
    parse_build_verification_challenge(&valid, VERIFICATION_HASH, &verification_actual())
        .expect("shared conservative attempt id must parse");
}

#[test]
fn build_verification_rejects_bad_challenge_input_hash_and_path() {
    assert!(parse_build_verification_challenge(
        &valid_verification_challenge(),
        "not-a-sha256",
        &verification_actual(),
    )
    .is_err());

    let bad_path = valid_verification_challenge().replacen(
        "\"path\":\"/verify/bin/rapx\"",
        "\"path\":\"relative/rapx\"",
        1,
    );
    assert!(parse_build_verification_challenge(&bad_path, VERIFICATION_HASH, &verification_actual()).is_err());
}

#[test]
fn build_verification_rejects_escaped_nul_in_material_path() {
    let nul_path = valid_verification_challenge().replacen(
        "/source/rap",
        "/source/ra\\u0000p",
        1,
    );
    assert!(parse_build_verification_challenge(
        &nul_path,
        VERIFICATION_HASH,
        &verification_actual(),
    )
    .expect_err("escaped NUL path must be rejected")
    .contains("canonical path"));
}

#[test]
fn build_verification_json_combines_surrogate_pairs_and_rejects_bad_pairs() {
    let emoji = valid_verification_challenge().replacen(
        "\"first_line\":\"cargo 1.0\"",
        "\"first_line\":\"cargo \\uD83D\\uDE00\"",
        1,
    );
    let receipt = parse_build_verification_challenge(
        &emoji,
        VERIFICATION_HASH,
        &verification_actual(),
    )
    .expect("valid UTF-16 surrogate pair");
    assert_eq!(receipt.cargo_identity.first_line, "cargo 😀");

    for bad_escape in ["\\uD83D", "\\uDE00\\uD83D", "\\uD83D\\uD83D"] {
        let invalid = valid_verification_challenge().replacen(
            "cargo 1.0",
            bad_escape,
            1,
        );
        assert!(parse_build_verification_challenge(
            &invalid,
            VERIFICATION_HASH,
            &verification_actual(),
        )
        .is_err());
    }
}

#[test]
fn build_verification_json_has_size_and_nesting_limits() {
    let oversized = " ".repeat(64 * 1024 + 1);
    assert!(parse_build_verification_challenge(
        &oversized,
        VERIFICATION_HASH,
        &verification_actual(),
    )
    .expect_err("oversized challenge must fail")
    .contains("exceeds"));

    let deeply_nested = format!("{}null{}", "[".repeat(40), "]".repeat(40));
    assert!(parse_build_verification_challenge(
        &deeply_nested,
        VERIFICATION_HASH,
        &verification_actual(),
    )
    .expect_err("deeply nested challenge must fail")
    .contains("nesting depth"));
}

#[test]
fn build_verification_rejects_executable_sibling_and_rustc_mismatches() {
    let mut actual = verification_actual();
    actual.expected_challenge = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
    assert!(parse_build_verification_challenge(&valid_verification_challenge(), VERIFICATION_HASH, &actual).is_err());

    let mut actual = verification_actual();
    actual.rapx_path = "/other/rapx".to_string();
    assert!(parse_build_verification_challenge(&valid_verification_challenge(), VERIFICATION_HASH, &actual).is_err());

    let mut actual = verification_actual();
    actual.cargo_rapx_path = "/other/cargo-rapx".to_string();
    assert!(parse_build_verification_challenge(&valid_verification_challenge(), VERIFICATION_HASH, &actual).is_err());

    let mut actual = verification_actual();
    actual.cargo_supplied_rustc_path = "/other/rustc".to_string();
    assert!(parse_build_verification_challenge(&valid_verification_challenge(), VERIFICATION_HASH, &actual).is_err());

    let mut actual = verification_actual();
    actual.cargo_supplied_rustc_commit = "fedcba9876543210fedcba9876543210fedcba98".to_string();
    assert!(parse_build_verification_challenge(&valid_verification_challenge(), VERIFICATION_HASH, &actual).is_err());

    let mut actual = verification_actual();
    actual.loader_path = "/caller/bogus-loader".to_string();
    assert!(parse_build_verification_challenge(&valid_verification_challenge(), VERIFICATION_HASH, &actual).is_err());

    let mut actual = verification_actual();
    actual.loader_environment_variable = "BOGUS_LIBRARY_PATH".to_string();
    assert!(parse_build_verification_challenge(&valid_verification_challenge(), VERIFICATION_HASH, &actual).is_err());
}

#[test]
fn build_verification_accepts_only_cargo_target_deps_loader_prepend() {
    let mut actual = verification_actual();
    actual.loader_path = env::join_paths([
        PathBuf::from("/verify/work/target/debug/deps"),
        PathBuf::from("/toolchain/lib"),
    ])
    .expect("loader paths join")
    .into_string()
    .expect("loader paths are Unicode");
    parse_build_verification_challenge(
        &valid_verification_challenge(),
        VERIFICATION_HASH,
        &actual,
    )
    .expect("Cargo's deterministic target/debug/deps prepend must be accepted");

    actual.loader_path = env::join_paths([
        PathBuf::from("/caller/ambient"),
        PathBuf::from("/verify/work/target/debug/deps"),
        PathBuf::from("/toolchain/lib"),
    ])
    .expect("loader paths join")
    .into_string()
    .expect("loader paths are Unicode");
    assert!(parse_build_verification_challenge(
        &valid_verification_challenge(),
        VERIFICATION_HASH,
        &actual,
    )
    .expect_err("ambient loader entries must remain forbidden")
    .contains("loader path"));
}

#[test]
fn build_verification_accepts_legacy_cargo_deterministic_loader_prefixes() {
    let mut actual = verification_actual();
    actual.loader_path = env::join_paths([
        PathBuf::from("/verify/work/target/debug/deps"),
        PathBuf::from("/toolchain/lib"),
        PathBuf::from("/toolchain/lib"),
    ])
    .expect("loader paths join")
    .into_string()
    .expect("loader paths are Unicode");
    parse_build_verification_challenge(
        &valid_verification_challenge(),
        VERIFICATION_HASH,
        &actual,
    )
    .expect("legacy Cargo's deterministic loader prefixes must be accepted");

    actual.loader_path = env::join_paths([
        PathBuf::from("/caller/ambient"),
        PathBuf::from("/verify/work/target/debug/deps"),
        PathBuf::from("/toolchain/lib"),
        PathBuf::from("/toolchain/lib"),
    ])
    .expect("loader paths join")
    .into_string()
    .expect("loader paths are Unicode");
    assert!(parse_build_verification_challenge(
        &valid_verification_challenge(),
        VERIFICATION_HASH,
        &actual,
    )
    .expect_err("ambient loader entries must remain forbidden")
    .contains("loader path"));
}

#[test]
fn valid_build_verification_envelope_has_stable_receipt_fields() {
    let envelope = parse_build_verification_challenge(
        &valid_verification_challenge(),
        VERIFICATION_HASH,
        &verification_actual(),
    )
    .expect("valid challenge envelope");
    assert_eq!(envelope.challenge, VERIFICATION_HASH);
    assert_eq!(envelope.challenge_input_sha256, VERIFICATION_HASH);
    assert_eq!(envelope.cargo_rapx_path, "/verify/bin/cargo-rapx");
    assert_eq!(envelope.rapx_path, "/verify/bin/rapx");
    assert_eq!(envelope.rustc_path, "/toolchain/bin/rustc");
    assert_eq!(envelope.loader_environment_variable, VERIFICATION_LOADER_VARIABLE);
    assert_eq!(envelope.loader_path, "/toolchain/lib");
    assert_eq!(envelope.loader_contents_sha256, VERIFICATION_HASH);
    assert_eq!(envelope.fixture_project_root, "/verify/work/fixture");
}

#[test]
fn build_verification_environment_entry_reads_file_and_binds_rustc_path() {
    let _lock = VERIFICATION_ENV_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let environment = EnvironmentRestore::capture(VERIFICATION_ENV_KEYS);
    environment.remove("LD_LIBRARY_PATH");
    environment.remove("DYLD_LIBRARY_PATH");
    let directory = env::temp_dir().join(format!(
        "rap-build-verification-environment-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let bin = directory.join("bin");
    let toolchain = directory.join("toolchain");
    let loader = toolchain.join("lib");
    fs::create_dir_all(&bin).expect("verification bin directory");
    fs::create_dir_all(&toolchain).expect("verification toolchain directory");
    fs::create_dir_all(&loader).expect("verification loader directory");
    let rapx = bin.join("rapx");
    let cargo_rapx = bin.join("cargo-rapx");
    let rustc = toolchain.join("rustc");
    let other_rustc = toolchain.join("other-rustc");
    fs::write(&rapx, b"rapx fixture").expect("rapx fixture");
    fs::write(&cargo_rapx, b"cargo-rapx fixture").expect("cargo-rapx fixture");
    fs::write(&rustc, b"rustc fixture").expect("rustc fixture");
    fs::write(&other_rustc, b"other rustc fixture").expect("other rustc fixture");
    for executable in [&rapx, &cargo_rapx, &rustc, &other_rustc] {
        fs::set_permissions(executable, fs::Permissions::from_mode(0o700))
            .expect("fake executable permissions");
    }
    let rapx = rapx.canonicalize().expect("canonical rapx fixture");
    let cargo_rapx = cargo_rapx
        .canonicalize()
        .expect("canonical cargo-rapx fixture");
    let rustc = rustc.canonicalize().expect("canonical rustc fixture");
    let other_rustc = other_rustc
        .canonicalize()
        .expect("canonical other rustc fixture");
    let actual = BuildVerificationActual {
        expected_challenge: VERIFICATION_HASH.to_string(),
        rapx_path: rapx.display().to_string(),
        cargo_rapx_path: cargo_rapx.display().to_string(),
        cargo_supplied_rustc_path: rustc.display().to_string(),
        cargo_supplied_rustc_commit: VERIFICATION_COMMIT.to_string(),
        loader_environment_variable: VERIFICATION_LOADER_VARIABLE.to_string(),
        loader_path: loader
            .canonicalize()
            .expect("canonical loader directory")
            .display()
            .to_string(),
    };
    let challenge_path = directory.join("challenge.json");

    environment.set(
        "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_MODE",
        "rap-build-verification-v1",
    );
    environment.set(
        "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT",
        &challenge_path,
    );
    environment.set(
        "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT_SHA256",
        VERIFICATION_HASH,
    );
    environment.set(
        "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_CHALLENGE",
        VERIFICATION_HASH,
    );
    environment.set(
        "UNSOUND_SCANNER_CARGO_SUPPLIED_RUSTC_COMMIT",
        VERIFICATION_COMMIT,
    );
    environment.set(VERIFICATION_LOADER_VARIABLE, "/caller/bogus-loader");

    fs::write(&challenge_path, vec![b' '; 64 * 1024 + 1])
        .expect("oversized challenge fixture");
    environment.set(
        "UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH",
        &rustc,
    );
    environment.set(VERIFICATION_LOADER_VARIABLE, &actual.loader_path);
    let error = build_verification_receipt_from_environment_with_current_exe(&rapx)
        .expect_err("oversized challenge file must fail");
    assert!(error.contains("exceeds"), "{error}");

    fs::write(&challenge_path, verification_challenge_for(&actual))
        .expect("valid challenge fixture");
    environment.set(
        "UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH",
        &other_rustc,
    );
    let error = build_verification_receipt_from_environment_with_current_exe(&rapx)
        .expect_err("Cargo-supplied rustc path mismatch must fail");
    assert!(error.contains("rustc path"), "{error}");

    environment.set(
        "UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH",
        &rustc,
    );
    environment.set(VERIFICATION_LOADER_VARIABLE, "/caller/bogus-loader");
    let error = build_verification_receipt_from_environment_with_current_exe(&rapx)
        .expect_err("ambient loader path mismatch must fail");
    assert!(error.contains("loader path"), "{error}");

    environment.set(VERIFICATION_LOADER_VARIABLE, &actual.loader_path);
    let receipt = build_verification_receipt_from_environment_with_current_exe(&rapx)
        .expect("valid environment entry")
        .expect("verification receipt present");
    assert_eq!(receipt.rapx_path, actual.rapx_path);
    assert_eq!(receipt.cargo_rapx_path, actual.cargo_rapx_path);
    assert_eq!(receipt.rustc_path, actual.cargo_supplied_rustc_path);

    drop(environment);
    fs::remove_dir_all(directory).expect("remove verification environment fixture");
}

fn identity_from(args: &[&str]) -> UnitIdentity {
    UnitIdentity::from_rustc_args_and_env(
        &args
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>(),
        &IdentityEnvironment {
            package_name: Some("demo-pkg".to_string()),
            package_version: Some("1.2.3".to_string()),
            manifest_path: Some("/workspace/Cargo.toml".to_string()),
            ..IdentityEnvironment::default()
        },
    )
    .expect("identity parses")
}

#[test]
fn parses_equals_flags_and_comma_crate_types() {
    let identity = identity_from(&[
        "/toolchain/rustc",
        "--crate-name=demo",
        "--crate-type=lib,rlib",
        "-Cextra-filename=-abc",
        "--target=x86_64-unknown-linux-gnu",
        "src/lib.rs",
    ]);

    assert_eq!(identity.crate_name, "demo");
    assert_eq!(identity.crate_types, vec!["lib", "rlib"]);
    assert_eq!(identity.extra_filename, "-abc");
    assert_eq!(
        identity.target_triple.as_deref(),
        Some("x86_64-unknown-linux-gnu")
    );
    assert_eq!(identity.source_path, "src/lib.rs");
}

#[test]
fn parses_split_flags() {
    let identity = identity_from(&[
        "/toolchain/rustc",
        "--crate-name",
        "demo",
        "--crate-type",
        "bin,cdylib",
        "-C",
        "extra-filename=-def",
        "--target",
        "aarch64-unknown-linux-gnu",
        "src/main.rs",
    ]);

    assert_eq!(identity.crate_types, vec!["bin", "cdylib"]);
    assert_eq!(identity.extra_filename, "-def");
    assert_eq!(
        identity.target_triple.as_deref(),
        Some("aarch64-unknown-linux-gnu")
    );
}

#[test]
fn combines_repeated_crate_type_flags_with_stable_deduplication() {
    let identity = identity_from(&[
        "/toolchain/rustc",
        "--crate-name=demo",
        "--crate-type=lib,rlib",
        "--crate-type",
        "cdylib,lib",
        "src/lib.rs",
    ]);

    assert_eq!(identity.crate_types, vec!["lib", "rlib", "cdylib"]);
}

#[test]
fn required_identity_inputs_fail_instead_of_using_unknown_placeholders() {
    let valid_args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=demo".to_string(),
        "--crate-type=lib".to_string(),
        "src/lib.rs".to_string(),
    ];
    let valid_environment = IdentityEnvironment {
        package_name: Some("demo-pkg".to_string()),
        package_version: Some("1.2.3".to_string()),
        manifest_path: Some("/workspace/Cargo.toml".to_string()),
        ..IdentityEnvironment::default()
    };

    for (args, field) in [
        (vec![], "rustc path"),
        (
            vec![
                "/toolchain/rustc".to_string(),
                "--crate-type=lib".to_string(),
                "src/lib.rs".to_string(),
            ],
            "crate name",
        ),
        (
            vec![
                "/toolchain/rustc".to_string(),
                "--crate-name=demo".to_string(),
                "src/lib.rs".to_string(),
            ],
            "crate types",
        ),
        (
            vec![
                "/toolchain/rustc".to_string(),
                "--crate-name=demo".to_string(),
                "--crate-type=lib".to_string(),
            ],
            "source path",
        ),
    ] {
        let error = UnitIdentity::from_rustc_args_and_env(&args, &valid_environment)
            .expect_err("missing argv identity must fail");
        assert!(error.contains(field), "{error}");
    }

    for (environment, field) in [
        (
            IdentityEnvironment {
                package_name: None,
                ..valid_environment.clone()
            },
            "package name",
        ),
        (
            IdentityEnvironment {
                package_version: None,
                ..valid_environment.clone()
            },
            "package version",
        ),
        (
            IdentityEnvironment {
                manifest_path: None,
                ..valid_environment.clone()
            },
            "manifest path",
        ),
    ] {
        let error = UnitIdentity::from_rustc_args_and_env(&valid_args, &environment)
            .expect_err("missing environment identity must fail");
        assert!(error.contains(field), "{error}");
    }
}

#[test]
fn stable_hash_has_a_literal_fnv1a_oracle() {
    assert_eq!(fnv1a64_hex(b"hello"), "a430d84680aabd0b");
}

#[test]
fn unit_ids_are_safe_stable_and_do_not_collide_for_unit_inputs() {
    let first = identity_from(&[
        "/toolchain/rustc",
        "--crate-name",
        "hello/world",
        "--crate-type=lib",
        "-Cextra-filename=-one",
        "src/lib.rs",
    ]);
    let same = identity_from(&[
        "/toolchain/rustc",
        "--crate-name",
        "hello/world",
        "--crate-type=lib",
        "-Cextra-filename=-one",
        "src/lib.rs",
    ]);
    let other = identity_from(&[
        "/toolchain/rustc",
        "--crate-name",
        "hello/world",
        "--crate-type=lib",
        "-Cextra-filename=-two",
        "src/lib.rs",
    ]);
    let other_crate = identity_from(&[
        "/toolchain/rustc",
        "--crate-name",
        "other",
        "--crate-type=lib",
        "-Cextra-filename=-one",
        "src/lib.rs",
    ]);
    let other_source = identity_from(&[
        "/toolchain/rustc",
        "--crate-name",
        "hello/world",
        "--crate-type=lib",
        "-Cextra-filename=-one",
        "src/other.rs",
    ]);

    assert_eq!(first.unit_id, same.unit_id);
    assert_ne!(first.unit_id, other.unit_id);
    assert_ne!(first.unit_id, other_crate.unit_id);
    assert_ne!(first.unit_id, other_source.unit_id);
    assert!(first
        .unit_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'));
    assert_eq!(sanitize_file_component("a/b:c ü"), "a_b_c___");
}

#[test]
fn invocation_ids_include_package_manifest_target_and_complete_rustc_arguments() {
    let base_args = [
        "/toolchain/rustc",
        "--crate-name",
        "demo",
        "--crate-type=lib",
        "--target",
        "x86_64-unknown-linux-gnu",
        "src/lib.rs",
    ];
    let base = identity_from(&base_args);
    let different_argument = identity_from(&[
        "/toolchain/rustc",
        "--crate-name",
        "demo",
        "--crate-type=lib",
        "--target",
        "x86_64-unknown-linux-gnu",
        "--cfg",
        "feature=one",
        "src/lib.rs",
    ]);
    let other_manifest = UnitIdentity::from_rustc_args_and_env(
        &base_args
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>(),
        &IdentityEnvironment {
            package_name: Some("demo-pkg".to_string()),
            package_version: Some("1.2.3".to_string()),
            manifest_path: Some("/other/Cargo.toml".to_string()),
            ..IdentityEnvironment::default()
        },
    )
    .expect("identity parses");

    assert!(base.package_id.contains("/workspace/Cargo.toml"));
    assert_ne!(
        base.rustc_invocation_id,
        different_argument.rustc_invocation_id
    );
    assert_ne!(base.rustc_invocation_id, other_manifest.rustc_invocation_id);
}

#[test]
fn atomic_write_replaces_content_without_leaving_temporary_files() {
    let directory = std::env::temp_dir().join(format!(
        "rap-unit-output-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir(&directory).expect("temporary directory");
    let output = directory.join("result.json");

    atomic_write(&output, b"first").expect("first write");
    atomic_write(&output, b"second").expect("replacement write");

    assert_eq!(fs::read(&output).expect("final content"), b"second");
    let entries = fs::read_dir(&directory)
        .expect("read directory")
        .map(|entry| entry.expect("entry").file_name())
        .collect::<Vec<_>>();
    assert_eq!(
        entries,
        vec![PathBuf::from("result.json").file_name().unwrap()]
    );
    fs::remove_file(output).expect("remove output");
    fs::remove_dir(directory).expect("remove directory");
}

#[test]
fn atomic_write_cleans_temporary_file_when_rename_fails() {
    let directory = std::env::temp_dir().join(format!(
        "rap-unit-output-rename-failure-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir(&directory).expect("temporary directory");
    let output = directory.join("result.json");

    let error = atomic_write_with_rename(&output, b"data", |_, _| {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "rename denied",
        ))
    })
    .expect_err("rename failure propagates");

    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(fs::read_dir(&directory)
        .expect("read directory")
        .next()
        .is_none());
    fs::remove_dir(directory).expect("remove directory");
}

#[test]
fn atomic_write_cleans_temporary_file_when_write_fails() {
    let directory = std::env::temp_dir().join(format!(
        "rap-unit-output-write-failure-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir(&directory).expect("temporary directory");
    let output = directory.join("result.json");

    let error = atomic_write_with_writer(&output, b"data", |_, _| {
        Err(std::io::Error::new(
            std::io::ErrorKind::WriteZero,
            "write denied",
        ))
    })
    .expect_err("write failure propagates");

    assert_eq!(error.kind(), std::io::ErrorKind::WriteZero);
    assert!(fs::read_dir(&directory)
        .expect("read directory")
        .next()
        .is_none());
    fs::remove_dir(directory).expect("remove directory");
}

#[test]
fn build_scripts_and_proc_macros_have_explicit_skip_reasons() {
    assert_eq!(
        crate_skip_reason(&["bin".to_string()], "build_script_build"),
        Some("build_script")
    );
    assert_eq!(
        crate_skip_reason(&["proc-macro".to_string()], "macro_impl"),
        Some("proc_macro")
    );
    assert_eq!(crate_skip_reason(&["lib".to_string()], "normal"), None);
}

fn routing_environment(
    manifest_path: &str,
    project_root: &str,
    primary_package: bool,
) -> IdentityEnvironment {
    IdentityEnvironment {
        package_name: Some("demo-pkg".to_string()),
        package_version: Some("1.2.3".to_string()),
        manifest_path: Some(manifest_path.to_string()),
        target_kind: Some("lib".to_string()),
        primary_package,
        project_root: Some(project_root.to_string()),
        working_directory: Some(project_root.to_string()),
    }
}

fn assert_skip_receipt(route: UnitRoute, expected_reason: &str) {
    let UnitRoute::Skip { identity, reason } = route else {
        panic!("expected an explicit skip route")
    };
    assert_eq!(reason, expected_reason);
    let directory = std::env::temp_dir().join(format!(
        "rap-skip-receipt-{}-{expected_reason}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir(&directory).expect("skip receipt directory");
    let output = directory.join(format!("{}.json", identity.unit_id));
    write_skip_receipt(&output, &identity, reason, "/workspace")
        .expect("skip receipt write succeeds");
    let json = fs::read_to_string(&output).expect("skip receipt is UTF-8 JSON");
    assert!(json.contains("\"schema_version\":\"rap-unit-skip-v1\""));
    assert!(json.contains("\"status\":\"skipped\""));
    assert!(json.contains("\"success\":false"));
    assert!(json.contains(&format!("\"skip_reason\":\"{expected_reason}\"")));
    assert!(json.contains(&format!("\"unit_id\":\"{}\"", identity.unit_id)));
    fs::remove_dir_all(directory).expect("remove skip receipt directory");
}

#[test]
fn cargo_primary_package_with_bound_manifest_routes_absolute_source_to_analysis() {
    let project = std::env::temp_dir().join(format!(
        "rap-primary-routing-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(project.join("src")).expect("project source directory");
    let manifest = project.join("Cargo.toml");
    let source = project.join("src/lib.rs");
    fs::write(&manifest, b"[package]\nname='demo'\nversion='0.1.0'\n")
        .expect("manifest fixture");
    fs::write(&source, b"pub fn demo() {}\n").expect("source fixture");
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=demo".to_string(),
        "--crate-type=lib".to_string(),
        source.display().to_string(),
    ];
    let environment = routing_environment(
        manifest.to_str().expect("manifest UTF-8"),
        project.to_str().expect("project UTF-8"),
        true,
    );

    let route = route_rustc_unit(&args, &environment).expect("route resolves");
    let UnitRoute::Analyze(identity) = route else {
        panic!("Cargo primary package must be analyzed")
    };
    assert_eq!(identity.manifest_path, manifest.display().to_string());
    assert_eq!(identity.source_path, source.display().to_string());

    fs::remove_dir_all(project).expect("remove routing fixture");
}

#[test]
fn repository_local_non_primary_package_routes_to_analysis() {
    let project = std::env::temp_dir().join(format!(
        "rap-local-dependency-routing-{}",
        std::process::id()
    ));
    let member = project.join("member");
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(member.join("src")).expect("member source directory");
    let manifest = member.join("Cargo.toml");
    let source = member.join("src/lib.rs");
    fs::write(&manifest, b"[package]\nname='member'\nversion='0.1.0'\n")
        .expect("member manifest");
    fs::write(&source, b"pub fn member() {}\n").expect("member source");
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=member".to_string(),
        "--crate-type=lib".to_string(),
        "member/src/lib.rs".to_string(),
    ];
    let environment = routing_environment(
        manifest.to_str().expect("manifest UTF-8"),
        project.to_str().expect("project UTF-8"),
        false,
    );

    let route = route_rustc_unit(&args, &environment).expect("route resolves");
    let UnitRoute::Analyze(identity) = route else {
        panic!("repository-local member must be analyzed")
    };
    assert_eq!(identity.source_path, source.display().to_string());

    fs::remove_dir_all(project).expect("remove routing fixture");
}

#[test]
fn non_primary_manifest_outside_project_is_an_explicit_skip() {
    let outside = std::env::temp_dir().join(format!(
        "rap-non-primary-outside-{}",
        std::process::id()
    ));
    let project = std::env::temp_dir().join(format!(
        "rap-non-primary-project-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&outside);
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(outside.join("src")).expect("outside source directory");
    fs::create_dir_all(&project).expect("project directory");
    let manifest = outside.join("Cargo.toml");
    let source = outside.join("src/lib.rs");
    fs::write(&manifest, b"[package]\nname='third-party'\nversion='1.2.3'\n")
        .expect("outside manifest");
    fs::write(&source, b"pub fn third_party() {}\n").expect("outside source");
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=third_party".to_string(),
        "--crate-type=lib".to_string(),
        source.display().to_string(),
    ];
    let environment = routing_environment(
        manifest.to_str().expect("manifest UTF-8"),
        project.to_str().expect("project UTF-8"),
        false,
    );

    let route = route_rustc_unit(&args, &environment).expect("dependency route resolves");
    let UnitRoute::Skip { reason, .. } = route else {
        panic!("outside dependency must be skipped")
    };
    assert_eq!(reason, "manifest_outside_project");

    fs::remove_dir_all(outside).expect("remove outside fixture");
    fs::remove_dir_all(project).expect("remove project fixture");
}

#[test]
fn rustc_non_unit_probe_is_passed_through_without_fabricating_a_skip_receipt() {
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--print".to_string(),
        "cfg".to_string(),
    ];

    assert_eq!(
        route_rustc_unit(&args, &IdentityEnvironment::default())
            .expect("probe route resolves"),
        UnitRoute::PassThrough,
    );
}

#[test]
fn build_script_route_produces_an_explicit_skip_receipt() {
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=build_script_build".to_string(),
        "--crate-type=bin".to_string(),
        "build.rs".to_string(),
    ];
    let environment = routing_environment("/workspace/Cargo.toml", "/workspace", true);

    assert_skip_receipt(
        route_rustc_unit(&args, &environment).expect("build script route resolves"),
        "build_script",
    );
}

#[test]
fn proc_macro_route_produces_an_explicit_skip_receipt() {
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=macro_impl".to_string(),
        "--crate-type=proc-macro".to_string(),
        "src/lib.rs".to_string(),
    ];
    let environment = routing_environment("/workspace/Cargo.toml", "/workspace", true);

    assert_skip_receipt(
        route_rustc_unit(&args, &environment).expect("proc macro route resolves"),
        "proc_macro",
    );
}

#[test]
fn primary_manifest_outside_project_root_is_an_explicit_skip() {
    let outside = std::env::temp_dir().join(format!(
        "rap-outside-manifest-{}",
        std::process::id()
    ));
    let project = std::env::temp_dir().join(format!(
        "rap-bound-project-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&outside);
    let _ = fs::remove_dir_all(&project);
    fs::create_dir_all(outside.join("src")).expect("outside source directory");
    fs::create_dir_all(&project).expect("bound project directory");
    let manifest = outside.join("Cargo.toml");
    let source = outside.join("src/lib.rs");
    fs::write(&manifest, b"[package]\nname='outside'\nversion='0.1.0'\n")
        .expect("outside manifest");
    fs::write(&source, b"pub fn outside() {}\n").expect("outside source");
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=outside".to_string(),
        "--crate-type=lib".to_string(),
        source.display().to_string(),
    ];
    let environment = routing_environment(
        manifest.to_str().expect("manifest UTF-8"),
        project.to_str().expect("project UTF-8"),
        true,
    );

    assert_skip_receipt(
        route_rustc_unit(&args, &environment).expect("manifest route resolves"),
        "manifest_outside_project",
    );

    fs::remove_dir_all(outside).expect("remove outside fixture");
    fs::remove_dir_all(project).expect("remove project fixture");
}

#[test]
fn primary_source_outside_bound_manifest_is_an_explicit_skip() {
    let project = std::env::temp_dir().join(format!(
        "rap-source-bound-project-{}",
        std::process::id()
    ));
    let outside = std::env::temp_dir().join(format!(
        "rap-source-outside-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&project);
    let _ = fs::remove_dir_all(&outside);
    fs::create_dir_all(&project).expect("project directory");
    fs::create_dir_all(&outside).expect("outside directory");
    let manifest = project.join("Cargo.toml");
    let source = outside.join("lib.rs");
    fs::write(&manifest, b"[package]\nname='demo'\nversion='0.1.0'\n")
        .expect("manifest fixture");
    fs::write(&source, b"pub fn outside() {}\n").expect("outside source");
    let args = vec![
        "/toolchain/rustc".to_string(),
        "--crate-name=demo".to_string(),
        "--crate-type=lib".to_string(),
        source.display().to_string(),
    ];
    let environment = routing_environment(
        manifest.to_str().expect("manifest UTF-8"),
        project.to_str().expect("project UTF-8"),
        true,
    );

    assert_skip_receipt(
        route_rustc_unit(&args, &environment).expect("source route resolves"),
        "source_outside_manifest",
    );

    fs::remove_dir_all(project).expect("remove project fixture");
    fs::remove_dir_all(outside).expect("remove outside fixture");
}

#[test]
fn timeout_parser_distinguishes_unset_disabled_and_invalid_values() {
    assert_eq!(parse_inner_timeout_seconds(None), Ok(None));
    assert_eq!(parse_inner_timeout_seconds(Some("0")), Ok(None));
    assert_eq!(parse_inner_timeout_seconds(Some("15")), Ok(Some(15)));
    assert!(parse_inner_timeout_seconds(Some("-1")).is_err());
    assert!(parse_inner_timeout_seconds(Some("nonsense")).is_err());
}

#[test]
fn wrapper_uses_cargo_supplied_rustc_path() {
    let argv = vec![
        "cargo-rapx".to_string(),
        "/custom/toolchain/bin/rustc".to_string(),
        "--crate-name".to_string(),
        "demo".to_string(),
    ];
    assert_eq!(
        rustc_path_from_wrapper_args(&argv).expect("rustc path"),
        "/custom/toolchain/bin/rustc"
    );
    assert!(rustc_path_from_wrapper_args(&["cargo-rapx".to_string()]).is_err());
}

#[test]
fn dispatches_custom_compiler_paths_to_wrapper_phase() {
    assert_eq!(
        dispatch_phase_from_argv(&["cargo-rapx".to_string(), "rapx".to_string()])
            .expect("cargo phase"),
        DispatchPhase::CargoRapx
    );
    assert_eq!(
        dispatch_phase_from_argv(&[
            "cargo-rapx".to_string(),
            "/opt/toolchain/compiler".to_string(),
        ])
        .expect("wrapper phase"),
        DispatchPhase::RustcWrapper
    );
    assert!(dispatch_phase_from_argv(&["cargo-rapx".to_string()]).is_err());
}

#[test]
fn commit_parser_requires_exact_hex_commit_and_propagates_runner_failures() {
    let commit = "0123456789abcdef0123456789abcdef01234567";
    assert_eq!(
        parse_rustc_commit(&format!("rustc 1.0\ncommit-hash: {commit}\n")),
        Ok(commit.to_string())
    );
    assert!(parse_rustc_commit("rustc 1.0\n").is_err());
    assert!(parse_rustc_commit("commit-hash: not-a-commit\n").is_err());
    assert!(
        rustc_commit_from_runner("/toolchain/compiler", |_| Err("launch failed".to_string()))
            .is_err()
    );
}

#[test]
fn embedded_rap_compiler_commit_is_strict_and_missing_without_build_environment() {
    let commit = "0123456789abcdef0123456789abcdef01234567";
    assert_eq!(
        validate_commit_hash(Some(commit), "RAP compiler commit"),
        Ok(commit.to_string())
    );
    assert!(validate_commit_hash(None, "RAP compiler commit").is_err());
    assert!(validate_commit_hash(Some("short"), "RAP compiler commit").is_err());
    assert!(embedded_rap_compiler_commit().is_err());
}

#[test]
fn nested_workspace_wrapper_is_rejected_only_when_nonempty() {
    assert_eq!(nested_workspace_wrapper_reason(None), None);
    assert_eq!(nested_workspace_wrapper_reason(Some("")), None);
    assert_eq!(nested_workspace_wrapper_reason(Some("   ")), None);
    assert_eq!(
        nested_workspace_wrapper_reason(Some("/toolchain/sccache")),
        Some("nested_rustc_workspace_wrapper_unsupported")
    );
}

#[test]
fn exact_tool_path_must_be_absolute_existing_file_and_canonical() {
    let directory = std::env::temp_dir().join(format!(
        "rap-exact-tool-path-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir(&directory).expect("temporary directory");
    let tool = directory.join("cargo");
    fs::write(&tool, b"tool").expect("tool placeholder");

    assert_eq!(
        validate_exact_tool_path(Some(tool.to_str().expect("utf8 path")), "exact Cargo")
            .expect("valid exact path"),
        tool
    );
    assert!(validate_exact_tool_path(Some("relative/cargo"), "exact Cargo").is_err());
    assert!(validate_exact_tool_path(None, "exact Cargo").is_err());
    assert!(validate_exact_tool_path(
        Some(directory.join("missing").to_str().expect("utf8 path")),
        "exact Cargo"
    )
    .is_err());

    fs::remove_file(tool).expect("remove tool");
    fs::remove_dir(directory).expect("remove directory");
}

#[test]
fn expected_commit_requires_valid_equal_receipt() {
    let expected = "0123456789abcdef0123456789abcdef01234567";
    assert_eq!(
        verify_expected_commit(expected, Some(expected), "exact rustc"),
        Ok(expected.to_string())
    );
    assert!(verify_expected_commit("f123456789abcdef0123456789abcdef01234567", Some(expected), "exact rustc").is_err());
    assert!(verify_expected_commit(expected, None, "exact rustc").is_err());
}

#[test]
fn json_directory_output_takes_priority_over_legacy_output() {
    let directory = PathBuf::from("/results/per-attempt");
    let legacy = PathBuf::from("/results/legacy.json");
    assert_eq!(
        json_output_path(Some(&directory), Some(&legacy), "unit-123"),
        Some(directory.join("unit-123.json"))
    );
    assert_eq!(
        json_output_path(None, Some(&legacy), "unit-123"),
        Some(legacy)
    );
}
