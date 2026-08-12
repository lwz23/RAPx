use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
const BUILD_VERIFICATION_CHALLENGE_MAX_BYTES: usize = 64 * 1024;
const BUILD_VERIFICATION_JSON_MAX_DEPTH: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchPhase {
    CargoRapx,
    RustcWrapper,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IdentityEnvironment {
    pub package_name: Option<String>,
    pub package_version: Option<String>,
    pub manifest_path: Option<String>,
    pub target_kind: Option<String>,
    pub primary_package: bool,
    pub project_root: Option<String>,
    pub working_directory: Option<String>,
}

impl IdentityEnvironment {
    pub fn from_process() -> Self {
        Self {
            package_name: env::var("CARGO_PKG_NAME").ok(),
            package_version: env::var("CARGO_PKG_VERSION").ok(),
            manifest_path: env::var("CARGO_MANIFEST_PATH").ok().or_else(|| {
                env::var("CARGO_MANIFEST_DIR")
                    .ok()
                    .map(|directory| format!("{directory}/Cargo.toml"))
            }),
            target_kind: env::var("CARGO_TARGET_KIND").ok(),
            primary_package: env::var_os("CARGO_PRIMARY_PACKAGE").is_some(),
            project_root: env::var("UNSOUND_SCANNER_PROJECT_ROOT").ok(),
            working_directory: env::current_dir()
                .ok()
                .map(|path| path.display().to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitIdentity {
    pub unit_id: String,
    pub rustc_invocation_id: String,
    pub package_id: String,
    pub package_name: String,
    pub package_version: String,
    pub crate_name: String,
    pub crate_types: Vec<String>,
    pub target_kind: String,
    pub target_triple: Option<String>,
    pub manifest_path: String,
    pub source_path: String,
    pub extra_filename: String,
    pub rustc_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitRoute {
    PassThrough,
    Analyze(UnitIdentity),
    Skip {
        identity: UnitIdentity,
        reason: &'static str,
    },
}

impl UnitIdentity {
    pub fn from_rustc_args_and_env(
        args: &[String],
        environment: &IdentityEnvironment,
    ) -> Result<Self, String> {
        let rustc_path = args
            .first()
            .filter(|value| !value.trim().is_empty())
            .cloned()
            .ok_or_else(|| "missing rustc path".to_string())?;
        let crate_name = find_flag_value(args, "--crate-name")
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "missing crate name".to_string())?;
        let crate_types = crate_types_from_rustc_args(args);
        if crate_types.is_empty() {
            return Err("missing crate types".to_string());
        }
        let source_path = args
            .iter()
            .find(|arg| arg.ends_with(".rs") && !arg.starts_with('-'))
            .cloned()
            .ok_or_else(|| "missing source path".to_string())?;
        let extra_filename =
            find_codegen_value(args, "extra-filename").unwrap_or_else(|| "none".to_string());
        let target_triple = find_flag_value(args, "--target").filter(|value| !value.is_empty());
        let package_name =
            required_environment_value(environment.package_name.clone(), "package name")?;
        let package_version =
            required_environment_value(environment.package_version.clone(), "package version")?;
        let manifest_path =
            required_environment_value(environment.manifest_path.clone(), "manifest path")?;
        let target_kind = environment
            .target_kind
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| crate_types.join(","));
        let package_id = format!("{}@{}#{}", package_name, package_version, manifest_path);
        let fingerprint = stable_fingerprint(&[
            &package_id,
            &crate_name,
            &crate_types.join(","),
            &target_kind,
            target_triple.as_deref().unwrap_or("host"),
            &manifest_path,
            &source_path,
            &extra_filename,
        ]);
        let mut invocation_parts = Vec::with_capacity(args.len() + 4);
        invocation_parts.push(package_id.as_str());
        invocation_parts.push(manifest_path.as_str());
        invocation_parts.push(target_kind.as_str());
        invocation_parts.push(target_triple.as_deref().unwrap_or("host"));
        invocation_parts.extend(args.iter().map(String::as_str));
        let invocation_fingerprint = stable_fingerprint(&invocation_parts);

        Ok(Self {
            unit_id: format!("{}-{}", sanitize_file_component(&crate_name), fingerprint),
            rustc_invocation_id: format!("rustc-{}", invocation_fingerprint),
            package_id,
            package_name,
            package_version,
            crate_name,
            crate_types,
            target_kind,
            target_triple,
            manifest_path,
            source_path,
            extra_filename,
            rustc_path,
        })
    }

    pub fn child_environment(&self) -> [(&'static str, String); 14] {
        [
            ("UNSOUND_SCANNER_RAP_UNIT_ID", self.unit_id.clone()),
            (
                "UNSOUND_SCANNER_RAP_RUSTC_INVOCATION_ID",
                self.rustc_invocation_id.clone(),
            ),
            ("UNSOUND_SCANNER_RAP_PACKAGE_ID", self.package_id.clone()),
            (
                "UNSOUND_SCANNER_RAP_PACKAGE_NAME",
                self.package_name.clone(),
            ),
            (
                "UNSOUND_SCANNER_RAP_PACKAGE_VERSION",
                self.package_version.clone(),
            ),
            ("UNSOUND_SCANNER_RAP_CRATE_NAME", self.crate_name.clone()),
            (
                "UNSOUND_SCANNER_RAP_CRATE_TYPES",
                self.crate_types.join(","),
            ),
            ("UNSOUND_SCANNER_RAP_TARGET_KIND", self.target_kind.clone()),
            (
                "UNSOUND_SCANNER_RAP_TARGET_TRIPLE",
                self.target_triple.clone().unwrap_or_default(),
            ),
            (
                "UNSOUND_SCANNER_RAP_MANIFEST_PATH",
                self.manifest_path.clone(),
            ),
            ("UNSOUND_SCANNER_RAP_SOURCE_PATH", self.source_path.clone()),
            (
                "UNSOUND_SCANNER_RAP_EXTRA_FILENAME",
                self.extra_filename.clone(),
            ),
            (
                "UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH",
                self.rustc_path.clone(),
            ),
            ("UNSOUND_SCANNER_RAP_RUSTC_PATH", self.rustc_path.clone()),
        ]
    }
}

pub fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub fn sanitize_file_component(value: &str) -> String {
    let sanitized: String = value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' {
                char::from(byte)
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

pub fn crate_skip_reason(crate_types: &[String], crate_name: &str) -> Option<&'static str> {
    if crate_types
        .iter()
        .any(|crate_type| crate_type == "proc-macro" || crate_type == "proc_macro")
    {
        Some("proc_macro")
    } else if crate_name == "build_script_build"
        && crate_types.iter().any(|crate_type| crate_type == "bin")
    {
        Some("build_script")
    } else {
        None
    }
}

pub fn route_rustc_unit(
    args: &[String],
    environment: &IdentityEnvironment,
) -> Result<UnitRoute, String> {
    if !args
        .iter()
        .any(|argument| argument.ends_with(".rs") && !argument.starts_with('-'))
    {
        return Ok(UnitRoute::PassThrough);
    }
    let mut identity = UnitIdentity::from_rustc_args_and_env(args, environment)?;
    if let Some(reason) = crate_skip_reason(&identity.crate_types, &identity.crate_name) {
        return Ok(UnitRoute::Skip { identity, reason });
    }
    let project_root = required_environment_value(
        environment.project_root.clone(),
        "scanner project root",
    )?;
    let project_root = PathBuf::from(&project_root)
        .canonicalize()
        .map_err(|err| format!("failed to resolve scanner project root '{project_root}': {err}"))?;
    let manifest = PathBuf::from(&identity.manifest_path)
        .canonicalize()
        .map_err(|err| {
            format!(
                "failed to resolve Cargo manifest '{}': {err}",
                identity.manifest_path
            )
        })?;
    if !manifest.starts_with(&project_root) {
        return Ok(UnitRoute::Skip {
            identity,
            reason: "manifest_outside_project",
        });
    }
    let manifest_dir = manifest
        .parent()
        .ok_or_else(|| format!("Cargo manifest has no parent: {}", manifest.display()))?;
    let source = PathBuf::from(&identity.source_path);
    let source = if source.is_absolute() {
        source
    } else {
        let working_directory = required_environment_value(
            environment.working_directory.clone(),
            "rustc working directory",
        )?;
        PathBuf::from(working_directory).join(source)
    };
    let source = source.canonicalize().map_err(|err| {
        format!(
            "failed to resolve rustc source '{}': {err}",
            identity.source_path
        )
    })?;
    if !source.starts_with(manifest_dir) {
        return Ok(UnitRoute::Skip {
            identity,
            reason: "source_outside_manifest",
        });
    }
    identity.source_path = source.display().to_string();
    Ok(UnitRoute::Analyze(identity))
}

pub fn skip_receipt_bytes(
    identity: &UnitIdentity,
    reason: &str,
    project_root: &str,
) -> Vec<u8> {
    let crate_types = identity
        .crate_types
        .iter()
        .map(|value| json_string(value))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{",
            "\"schema_version\":\"rap-unit-skip-v1\",",
            "\"source\":\"cargo-rapx\",",
            "\"success\":false,",
            "\"status\":\"skipped\",",
            "\"skip_reason\":{},",
            "\"project_root\":{},",
            "\"unit_id\":{},",
            "\"rustc_invocation_id\":{},",
            "\"package_id\":{},",
            "\"package_name\":{},",
            "\"package_version\":{},",
            "\"crate_name\":{},",
            "\"crate_types\":[{}],",
            "\"target_kind\":{},",
            "\"target_triple\":{},",
            "\"manifest_path\":{},",
            "\"source_path\":{},",
            "\"extra_filename\":{},",
            "\"rustc_path\":{}",
            "}}\n"
        ),
        json_string(reason),
        json_string(project_root),
        json_string(&identity.unit_id),
        json_string(&identity.rustc_invocation_id),
        json_string(&identity.package_id),
        json_string(&identity.package_name),
        json_string(&identity.package_version),
        json_string(&identity.crate_name),
        crate_types,
        json_string(&identity.target_kind),
        json_string(identity.target_triple.as_deref().unwrap_or("")),
        json_string(&identity.manifest_path),
        json_string(&identity.source_path),
        json_string(&identity.extra_filename),
        json_string(&identity.rustc_path),
    )
    .into_bytes()
}

pub fn write_skip_receipt(
    path: &Path,
    identity: &UnitIdentity,
    reason: &str,
    project_root: &str,
) -> io::Result<()> {
    atomic_write(path, &skip_receipt_bytes(identity, reason, project_root))
}

pub fn crate_types_from_rustc_args(args: &[String]) -> Vec<String> {
    find_flag_values(args, "--crate-type")
        .iter()
        .flat_map(|value| split_crate_types(value))
        .fold(Vec::new(), |mut types, crate_type| {
            if !types.contains(&crate_type) {
                types.push(crate_type);
            }
            types
        })
}

pub fn parse_inner_timeout_seconds(value: Option<&str>) -> Result<Option<u64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let seconds: u64 = value
        .trim()
        .parse()
        .map_err(|_| format!("UNSOUND_SCANNER_RAP_INNER_TIMEOUT_SECONDS must be a non-negative integer, got '{value}'"))?;
    Ok((seconds != 0).then_some(seconds))
}

pub fn rustc_path_from_wrapper_args(args: &[String]) -> Result<&str, String> {
    args.get(1)
        .filter(|value| !value.is_empty())
        .map(String::as_str)
        .ok_or_else(|| "missing Cargo-supplied rustc path at wrapper argv[1]".to_string())
}

pub fn dispatch_phase_from_argv(args: &[String]) -> Result<DispatchPhase, String> {
    let invocation = args
        .get(1)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "missing cargo-rapx invocation at argv[1]".to_string())?;
    if invocation == "rapx" {
        Ok(DispatchPhase::CargoRapx)
    } else {
        Ok(DispatchPhase::RustcWrapper)
    }
}

pub fn parse_rustc_commit(version_output: &str) -> Result<String, String> {
    let commit = version_output
        .lines()
        .find_map(|line| line.strip_prefix("commit-hash:").map(str::trim))
        .ok_or_else(|| "rustc -vV output is missing commit-hash".to_string())?;
    validate_commit_hash(Some(commit), "rustc -vV commit-hash")
}

pub fn validate_commit_hash(value: Option<&str>, field: &str) -> Result<String, String> {
    let value = value.ok_or_else(|| format!("missing {field}"))?;
    if value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_string())
    } else {
        Err(format!("{field} must be exactly 40 hexadecimal characters"))
    }
}

pub fn validate_exact_tool_path(value: Option<&str>, field: &str) -> Result<PathBuf, String> {
    let value = value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("missing {field} path"))?;
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{field} path must be absolute: {value}"));
    }
    let canonical = path
        .canonicalize()
        .map_err(|err| format!("failed to resolve {field} path '{value}': {err}"))?;
    if canonical != path {
        return Err(format!(
            "{field} path must already be canonical: {} -> {}",
            path.display(),
            canonical.display()
        ));
    }
    if !canonical.is_file() {
        return Err(format!("{field} path is not a file: {}", canonical.display()));
    }
    Ok(canonical)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildVerificationActual {
    pub expected_challenge: String,
    pub rapx_path: String,
    pub cargo_rapx_path: String,
    pub cargo_supplied_rustc_path: String,
    pub cargo_supplied_rustc_commit: String,
    pub loader_environment_variable: String,
    pub loader_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildVerificationReceipt {
    pub challenge: String,
    pub challenge_input_sha256: String,
    pub exact_toolchain: String,
    pub cargo_path: String,
    pub cargo_sha256: String,
    pub cargo_identity: BuildVerificationToolIdentity,
    pub rustc_path: String,
    pub rustc_sha256: String,
    pub rustc_identity: BuildVerificationToolIdentity,
    pub cargo_rapx_path: String,
    pub cargo_rapx_sha256: String,
    pub rapx_path: String,
    pub rapx_sha256: String,
    pub rap_source_path: String,
    pub rap_source_sha256: String,
    pub builder_sidecar_path: String,
    pub builder_sidecar_sha256: String,
    pub fixture_project_root: String,
    pub fixture_manifest_sha256: String,
    pub fixture_source_sha256: String,
    pub fixture_lock_sha256: String,
    pub loader_environment_variable: String,
    pub loader_path: String,
    pub loader_contents_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildVerificationToolIdentity {
    pub component: String,
    pub release: String,
    pub commit_hash: String,
    pub commit_date: String,
    pub host: String,
    pub first_line: String,
}

pub fn parse_build_verification_mode(value: Option<&str>) -> Result<Option<()>, String> {
    match value {
        None => Ok(None),
        Some("rap-build-verification-v1") => Ok(Some(())),
        Some(value) => Err(format!(
            "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_MODE must be rap-build-verification-v1, got '{value}'"
        )),
    }
}

pub fn parse_build_verification_challenge(
    input: &str,
    challenge_input_sha256: &str,
    actual: &BuildVerificationActual,
) -> Result<BuildVerificationReceipt, String> {
    if input.len() > BUILD_VERIFICATION_CHALLENGE_MAX_BYTES {
        return Err(format!(
            "build verification challenge exceeds {} bytes",
            BUILD_VERIFICATION_CHALLENGE_MAX_BYTES
        ));
    }
    validate_sha256(challenge_input_sha256, "challenge input SHA-256")?;
    let value = StrictJsonParser::new(input).parse()?;
    let root = value.object("build verification challenge")?;
    require_exact_keys(
        root,
        &[
            "schema",
            "challenge",
            "attempt_id",
            "exact_toolchain",
            "cargo",
            "rustc",
            "cargo_rapx",
            "rapx",
            "rap_source",
            "builder_sidecar",
            "loader",
            "fixture",
        ],
        "build verification challenge",
    )?;
    if required_string(root, "schema", "build verification challenge")?
        != "rap-build-verification-challenge-v1"
    {
        return Err("unexpected build verification challenge schema".to_string());
    }

    let challenge = required_string(root, "challenge", "build verification challenge")?;
    validate_sha256(&challenge, "build verification challenge")?;
    validate_sha256(&actual.expected_challenge, "build verification expected challenge")?;
    if challenge != actual.expected_challenge {
        return Err("build verification challenge does not match environment".to_string());
    }
    let attempt_id = required_string(root, "attempt_id", "build verification challenge")?;
    validate_attempt_id(&attempt_id)?;
    let exact_toolchain = required_string(root, "exact_toolchain", "build verification challenge")?;
    validate_nightly_toolchain(&exact_toolchain)?;

    let cargo = parse_tool_material(required_object(root, "cargo", "build verification challenge")?, "cargo")?;
    let rustc = parse_tool_material(required_object(root, "rustc", "build verification challenge")?, "rustc")?;
    let cargo_rapx = parse_path_hash(
        required_object(root, "cargo_rapx", "build verification challenge")?,
        "cargo_rapx",
    )?;
    let rapx = parse_path_hash(required_object(root, "rapx", "build verification challenge")?, "rapx")?;
    let rap_source = parse_path_hash(
        required_object(root, "rap_source", "build verification challenge")?,
        "rap_source",
    )?;
    let builder_sidecar = parse_path_hash(
        required_object(root, "builder_sidecar", "build verification challenge")?,
        "builder_sidecar",
    )?;
    let loader = parse_loader_material(required_object(
        root,
        "loader",
        "build verification challenge",
    )?)?;
    let fixture = parse_fixture(required_object(root, "fixture", "build verification challenge")?)?;

    if rapx.path != actual.rapx_path {
        return Err("build verification rapx path does not match current executable".to_string());
    }
    if cargo_rapx.path != actual.cargo_rapx_path {
        return Err("build verification cargo-rapx path does not match rapx sibling".to_string());
    }
    if rustc.path != actual.cargo_supplied_rustc_path {
        return Err("build verification rustc path does not match Cargo-supplied rustc".to_string());
    }
    let actual_commit = validate_commit_hash(
        Some(&actual.cargo_supplied_rustc_commit),
        "Cargo-supplied rustc commit",
    )?
    .to_ascii_lowercase();
    if rustc.identity.commit_hash != actual_commit {
        return Err("build verification rustc commit does not match Cargo-supplied rustc".to_string());
    }
    if loader.environment_variable != actual.loader_environment_variable {
        return Err("build verification loader environment variable mismatch".to_string());
    }
    if !verification_loader_path_matches(
        &loader.path,
        &actual.loader_path,
        &fixture.project_root,
    ) {
        return Err("build verification loader path does not match process environment".to_string());
    }

    Ok(BuildVerificationReceipt {
        challenge,
        challenge_input_sha256: challenge_input_sha256.to_string(),
        exact_toolchain,
        cargo_path: cargo.path,
        cargo_sha256: cargo.sha256,
        cargo_identity: cargo.identity,
        rustc_path: rustc.path,
        rustc_sha256: rustc.sha256,
        rustc_identity: rustc.identity,
        cargo_rapx_path: cargo_rapx.path,
        cargo_rapx_sha256: cargo_rapx.sha256,
        rapx_path: rapx.path,
        rapx_sha256: rapx.sha256,
        rap_source_path: rap_source.path,
        rap_source_sha256: rap_source.sha256,
        builder_sidecar_path: builder_sidecar.path,
        builder_sidecar_sha256: builder_sidecar.sha256,
        fixture_project_root: fixture.project_root,
        fixture_manifest_sha256: fixture.manifest_sha256,
        fixture_source_sha256: fixture.source_sha256,
        fixture_lock_sha256: fixture.lock_sha256,
        loader_environment_variable: loader.environment_variable,
        loader_path: loader.path,
        loader_contents_sha256: loader.contents_sha256,
    })
}

fn verification_loader_path_matches(
    expected_loader_path: &str,
    actual_loader_path: &str,
    fixture_project_root: &str,
) -> bool {
    if actual_loader_path == expected_loader_path {
        return true;
    }

    let Some(work_dir) = Path::new(fixture_project_root).parent() else {
        return false;
    };
    let cargo_target_deps = work_dir.join("target").join("debug").join("deps");
    let actual_paths = env::split_paths(actual_loader_path).collect::<Vec<_>>();
    let expected_loader_path = Path::new(expected_loader_path);
    let current_cargo_paths = [cargo_target_deps.clone(), expected_loader_path.to_path_buf()];
    if actual_paths == current_cargo_paths {
        return true;
    }

    let legacy_cargo_paths = [
        cargo_target_deps,
        expected_loader_path.to_path_buf(),
        expected_loader_path.to_path_buf(),
    ];
    actual_paths == legacy_cargo_paths
}

pub fn build_verification_receipt_from_environment(
) -> Result<Option<BuildVerificationReceipt>, String> {
    if env::var_os("UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_MODE").is_none() {
        return Ok(None);
    }
    let current_exe = env::current_exe()
        .map_err(|err| format!("failed to resolve current rapx executable: {err}"))?;
    build_verification_receipt_from_environment_with_current_exe(&current_exe)
}

pub fn build_verification_receipt_from_environment_with_current_exe(
    current_exe: &Path,
) -> Result<Option<BuildVerificationReceipt>, String> {
    let mode = env::var_os("UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_MODE")
        .ok_or_else(|| "missing UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_MODE".to_string())?
        .into_string()
        .map_err(|_| {
            "UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_MODE must be valid Unicode".to_string()
        })?;
    parse_build_verification_mode(Some(&mode))?;

    let input_path = env::var("UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT")
        .map_err(|_| "missing UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT".to_string())?;
    let input_path = validate_exact_tool_path(Some(&input_path), "build verification input")?;
    let input = read_build_verification_challenge(&input_path)?;
    let challenge_input_sha256 = env::var("UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT_SHA256")
        .map_err(|_| "missing UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_INPUT_SHA256".to_string())?;
    let expected_challenge = env::var("UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_CHALLENGE")
        .map_err(|_| "missing UNSOUND_SCANNER_RAP_BUILD_VERIFICATION_CHALLENGE".to_string())?;

    let rapx_path = current_exe
        .canonicalize()
        .map_err(|err| format!("failed to canonicalize current rapx executable: {err}"))?;
    if !rapx_path.is_file() {
        return Err(format!(
            "current rapx executable is not a regular file: {}",
            rapx_path.display()
        ));
    }
    let cargo_rapx_path = rapx_path
        .parent()
        .ok_or_else(|| "current rapx executable has no parent directory".to_string())?
        .join("cargo-rapx")
        .canonicalize()
        .map_err(|err| format!("failed to canonicalize cargo-rapx sibling: {err}"))?;
    if !cargo_rapx_path.is_file() {
        return Err(format!(
            "cargo-rapx sibling is not a regular file: {}",
            cargo_rapx_path.display()
        ));
    }
    let cargo_supplied_rustc_path = validate_exact_tool_path(
        env::var("UNSOUND_SCANNER_RAP_CARGO_SUPPLIED_RUSTC_PATH")
            .ok()
            .as_deref(),
        "Cargo-supplied rustc",
    )?;
    let cargo_supplied_rustc_commit = env::var("UNSOUND_SCANNER_CARGO_SUPPLIED_RUSTC_COMMIT")
        .map_err(|_| "missing Cargo-supplied rustc commit".to_string())?;
    let loader_environment_variable = verification_loader_environment_variable()?;
    let forbidden_loader_environment_variable = if loader_environment_variable == "LD_LIBRARY_PATH" {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    if env::var_os(forbidden_loader_environment_variable).is_some() {
        return Err(format!(
            "unexpected ambient loader variable {forbidden_loader_environment_variable}"
        ));
    }
    let loader_path = env::var(loader_environment_variable)
        .map_err(|_| format!("missing {loader_environment_variable}"))?;
    let actual = BuildVerificationActual {
        expected_challenge,
        rapx_path: rapx_path.display().to_string(),
        cargo_rapx_path: cargo_rapx_path.display().to_string(),
        cargo_supplied_rustc_path: cargo_supplied_rustc_path.display().to_string(),
        cargo_supplied_rustc_commit,
        loader_environment_variable: loader_environment_variable.to_string(),
        loader_path,
    };
    parse_build_verification_challenge(&input, &challenge_input_sha256, &actual).map(Some)
}

fn read_build_verification_challenge(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|err| {
        format!(
            "failed to open build verification input '{}': {err}",
            path.display()
        )
    })?;
    let length = file
        .metadata()
        .map_err(|err| {
            format!(
                "failed to stat build verification input '{}': {err}",
                path.display()
            )
        })?
        .len();
    if length > BUILD_VERIFICATION_CHALLENGE_MAX_BYTES as u64 {
        return Err(format!(
            "build verification challenge exceeds {} bytes",
            BUILD_VERIFICATION_CHALLENGE_MAX_BYTES
        ));
    }
    let mut input = String::new();
    Read::by_ref(&mut file)
        .take(BUILD_VERIFICATION_CHALLENGE_MAX_BYTES as u64 + 1)
        .read_to_string(&mut input)
        .map_err(|err| {
            format!(
                "failed to read build verification input '{}': {err}",
                path.display()
            )
        })?;
    if input.len() > BUILD_VERIFICATION_CHALLENGE_MAX_BYTES {
        return Err(format!(
            "build verification challenge exceeds {} bytes",
            BUILD_VERIFICATION_CHALLENGE_MAX_BYTES
        ));
    }
    Ok(input)
}

struct PathHash {
    path: String,
    sha256: String,
}

struct ToolMaterial {
    path: String,
    sha256: String,
    identity: BuildVerificationToolIdentity,
}

struct FixtureMaterial {
    project_root: String,
    manifest_sha256: String,
    source_sha256: String,
    lock_sha256: String,
}

struct LoaderMaterial {
    environment_variable: String,
    path: String,
    contents_sha256: String,
}

fn parse_tool_material(object: &[(String, JsonValue)], field: &str) -> Result<ToolMaterial, String> {
    require_exact_keys(object, &["path", "sha256", "identity"], field)?;
    let path = required_string(object, "path", field)?;
    validate_canonical_path(&path, &format!("{field} path"))?;
    let sha256 = required_string(object, "sha256", field)?;
    validate_sha256(&sha256, &format!("{field} SHA-256"))?;
    let identity = required_object(object, "identity", field)?;
    require_exact_keys(
        identity,
        &["component", "release", "commit_hash", "commit_date", "host", "first_line"],
        &format!("{field} identity"),
    )?;
    let component = required_string(identity, "component", &format!("{field} identity"))?;
    if component != field {
        return Err(format!("{field} identity component does not match"));
    }
    let release = required_nonempty_string(identity, "release", &format!("{field} identity"))?;
    let commit_hash = validate_commit_hash(
        Some(&required_string(identity, "commit_hash", &format!("{field} identity"))?),
        &format!("{field} identity commit_hash"),
    )?
    .to_ascii_lowercase();
    let commit_date = required_nonempty_string(identity, "commit_date", &format!("{field} identity"))?;
    let host = required_nonempty_string(identity, "host", &format!("{field} identity"))?;
    let first_line = required_nonempty_string(identity, "first_line", &format!("{field} identity"))?;
    Ok(ToolMaterial {
        path,
        sha256,
        identity: BuildVerificationToolIdentity {
            component,
            release,
            commit_hash,
            commit_date,
            host,
            first_line,
        },
    })
}

fn parse_path_hash(object: &[(String, JsonValue)], field: &str) -> Result<PathHash, String> {
    require_exact_keys(object, &["path", "sha256"], field)?;
    let path = required_string(object, "path", field)?;
    validate_canonical_path(&path, &format!("{field} path"))?;
    let sha256 = required_string(object, "sha256", field)?;
    validate_sha256(&sha256, &format!("{field} SHA-256"))?;
    Ok(PathHash { path, sha256 })
}

fn parse_fixture(object: &[(String, JsonValue)]) -> Result<FixtureMaterial, String> {
    require_exact_keys(
        object,
        &["project_root", "manifest_sha256", "source_sha256", "lock_sha256"],
        "fixture",
    )?;
    let project_root = required_string(object, "project_root", "fixture")?;
    validate_canonical_path(&project_root, "fixture project_root")?;
    let manifest_sha256 = required_string(object, "manifest_sha256", "fixture")?;
    validate_sha256(&manifest_sha256, "fixture manifest SHA-256")?;
    let source_sha256 = required_string(object, "source_sha256", "fixture")?;
    validate_sha256(&source_sha256, "fixture source SHA-256")?;
    let lock_sha256 = required_string(object, "lock_sha256", "fixture")?;
    validate_sha256(&lock_sha256, "fixture lock SHA-256")?;
    Ok(FixtureMaterial {
        project_root,
        manifest_sha256,
        source_sha256,
        lock_sha256,
    })
}

fn parse_loader_material(object: &[(String, JsonValue)]) -> Result<LoaderMaterial, String> {
    require_exact_keys(
        object,
        &["environment_variable", "path", "contents_sha256"],
        "loader",
    )?;
    let environment_variable = required_string(object, "environment_variable", "loader")?;
    if environment_variable != "LD_LIBRARY_PATH"
        && environment_variable != "DYLD_LIBRARY_PATH"
    {
        return Err("build verification loader environment variable is invalid".to_string());
    }
    let path = required_string(object, "path", "loader")?;
    validate_canonical_path(&path, "loader path")?;
    let contents_sha256 = required_string(object, "contents_sha256", "loader")?;
    validate_sha256(&contents_sha256, "loader contents SHA-256")?;
    Ok(LoaderMaterial {
        environment_variable,
        path,
        contents_sha256,
    })
}

#[cfg(target_os = "linux")]
fn verification_loader_environment_variable() -> Result<&'static str, String> {
    Ok("LD_LIBRARY_PATH")
}

#[cfg(target_os = "macos")]
fn verification_loader_environment_variable() -> Result<&'static str, String> {
    Ok("DYLD_LIBRARY_PATH")
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn verification_loader_environment_variable() -> Result<&'static str, String> {
    Err("unsupported build verification loader platform".to_string())
}

fn required_object<'a>(
    object: &'a [(String, JsonValue)],
    key: &str,
    context: &str,
) -> Result<&'a [(String, JsonValue)], String> {
    object
        .iter()
        .find(|(name, _)| name == key)
        .ok_or_else(|| format!("missing {key} in {context}"))?
        .1
        .object(context)
}

fn required_string(
    object: &[(String, JsonValue)],
    key: &str,
    context: &str,
) -> Result<String, String> {
    object
        .iter()
        .find(|(name, _)| name == key)
        .ok_or_else(|| format!("missing {key} in {context}"))?
        .1
        .string(context)
}

fn required_nonempty_string(
    object: &[(String, JsonValue)],
    key: &str,
    context: &str,
) -> Result<String, String> {
    let value = required_string(object, key, context)?;
    if value.trim().is_empty() {
        return Err(format!("{key} in {context} must not be empty"));
    }
    Ok(value)
}

fn require_exact_keys(
    object: &[(String, JsonValue)],
    expected: &[&str],
    context: &str,
) -> Result<(), String> {
    if object.len() != expected.len() {
        return Err(format!("{context} has missing or unknown fields"));
    }
    for key in expected {
        if !object.iter().any(|(actual, _)| actual == key) {
            return Err(format!("missing {key} in {context}"));
        }
    }
    if let Some((key, _)) = object
        .iter()
        .find(|(actual, _)| !expected.iter().any(|expected| actual == expected))
    {
        return Err(format!("unknown field {key} in {context}"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        Ok(())
    } else {
        Err(format!("{field} must be 64 lowercase hexadecimal characters"))
    }
}

fn validate_canonical_path(value: &str, field: &str) -> Result<(), String> {
    if value.contains('\0')
        || !value.starts_with('/')
        || value.len() == 1
        || value.ends_with('/')
        || value
            .split('/')
            .skip(1)
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!("{field} must be an absolute canonical path"));
    }
    Ok(())
}

fn validate_attempt_id(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 256
        || !value.as_bytes()[0].is_ascii_alphanumeric()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err("build verification attempt_id is invalid".to_string());
    }
    Ok(())
}

fn validate_nightly_toolchain(value: &str) -> Result<(), String> {
    let Some(date) = value.strip_prefix("nightly-") else {
        return Err("build verification exact_toolchain must be nightly-YYYY-MM-DD".to_string());
    };
    let valid = date.len() == 10
        && date.as_bytes()[4] == b'-'
        && date.as_bytes()[7] == b'-'
        && date.bytes().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7) || byte.is_ascii_digit()
        });
    if valid {
        Ok(())
    } else {
        Err("build verification exact_toolchain must be nightly-YYYY-MM-DD".to_string())
    }
}

enum JsonValue {
    Object(Vec<(String, JsonValue)>),
    String(String),
    Other,
}

impl JsonValue {
    fn object(&self, context: &str) -> Result<&[(String, JsonValue)], String> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err(format!("{context} must be a JSON object")),
        }
    }

    fn string(&self, context: &str) -> Result<String, String> {
        match self {
            Self::String(value) => Ok(value.clone()),
            _ => Err(format!("{context} field must be a JSON string")),
        }
    }
}

struct StrictJsonParser<'a> {
    input: &'a [u8],
    index: usize,
}

impl<'a> StrictJsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input: input.as_bytes(), index: 0 }
    }

    fn parse(mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        let value = self.value(0)?;
        self.skip_whitespace();
        if self.index != self.input.len() {
            return Err("unexpected trailing data in build verification challenge JSON".to_string());
        }
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => {
                if depth >= BUILD_VERIFICATION_JSON_MAX_DEPTH {
                    return Err(format!(
                        "build verification challenge JSON exceeds nesting depth {}",
                        BUILD_VERIFICATION_JSON_MAX_DEPTH
                    ));
                }
                self.object_value(depth + 1)
            }
            Some(b'\"') => self.string_value().map(JsonValue::String),
            Some(b'[') => {
                if depth >= BUILD_VERIFICATION_JSON_MAX_DEPTH {
                    return Err(format!(
                        "build verification challenge JSON exceeds nesting depth {}",
                        BUILD_VERIFICATION_JSON_MAX_DEPTH
                    ));
                }
                self.array_value(depth + 1)
            }
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'n') => self.literal(b"null"),
            Some(b'-' | b'0'..=b'9') => self.number_value(),
            _ => Err("invalid JSON value in build verification challenge".to_string()),
        }
    }

    fn object_value(&mut self, child_depth: usize) -> Result<JsonValue, String> {
        self.expect(b'{')?;
        self.skip_whitespace();
        let mut entries = Vec::new();
        if self.consume(b'}') {
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            let key = self.string_value()?;
            if entries.iter().any(|(existing, _)| existing == &key) {
                return Err(format!("duplicate JSON key '{key}' in build verification challenge"));
            }
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.value(child_depth)?;
            entries.push((key, value));
            self.skip_whitespace();
            if self.consume(b'}') {
                return Ok(JsonValue::Object(entries));
            }
            self.expect(b',')?;
        }
    }

    fn array_value(&mut self, child_depth: usize) -> Result<JsonValue, String> {
        self.expect(b'[')?;
        self.skip_whitespace();
        if self.consume(b']') {
            return Ok(JsonValue::Other);
        }
        loop {
            self.value(child_depth)?;
            self.skip_whitespace();
            if self.consume(b']') {
                return Ok(JsonValue::Other);
            }
            self.expect(b',')?;
        }
    }

    fn literal(&mut self, expected: &[u8]) -> Result<JsonValue, String> {
        if self.input.get(self.index..self.index + expected.len()) == Some(expected) {
            self.index += expected.len();
            Ok(JsonValue::Other)
        } else {
            Err("invalid JSON literal in build verification challenge".to_string())
        }
    }

    fn number_value(&mut self) -> Result<JsonValue, String> {
        let start = self.index;
        if self.consume(b'-') && !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err("invalid JSON number in build verification challenge".to_string());
        }
        if self.consume(b'0') {
        } else {
            self.consume_digits();
        }
        if self.consume(b'.') && !self.consume_digits() {
            return Err("invalid JSON number in build verification challenge".to_string());
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.index += 1;
            let _ = self.consume(b'+') || self.consume(b'-');
            if !self.consume_digits() {
                return Err("invalid JSON number in build verification challenge".to_string());
            }
        }
        if self.index == start {
            return Err("invalid JSON number in build verification challenge".to_string());
        }
        Ok(JsonValue::Other)
    }

    fn string_value(&mut self) -> Result<String, String> {
        self.expect(b'\"')?;
        let mut output = String::new();
        loop {
            let byte = self.next().ok_or_else(|| "unterminated JSON string in build verification challenge".to_string())?;
            match byte {
                b'\"' => return Ok(output),
                b'\\' => match self.next().ok_or_else(|| "unterminated JSON escape in build verification challenge".to_string())? {
                    b'\"' => output.push('\"'),
                    b'\\' => output.push('\\'),
                    b'/' => output.push('/'),
                    b'b' => output.push('\u{08}'),
                    b'f' => output.push('\u{0c}'),
                    b'n' => output.push('\n'),
                    b'r' => output.push('\r'),
                    b't' => output.push('\t'),
                    b'u' => {
                        let first = self.hex_code_unit()?;
                        let code = if (0xd800..=0xdbff).contains(&first) {
                            self.expect(b'\\')?;
                            self.expect(b'u')?;
                            let second = self.hex_code_unit()?;
                            if !(0xdc00..=0xdfff).contains(&second) {
                                return Err(
                                    "invalid JSON UTF-16 surrogate pair in build verification challenge"
                                        .to_string(),
                                );
                            }
                            0x10000
                                + ((u32::from(first) - 0xd800) << 10)
                                + (u32::from(second) - 0xdc00)
                        } else if (0xdc00..=0xdfff).contains(&first) {
                            return Err(
                                "isolated JSON UTF-16 low surrogate in build verification challenge"
                                    .to_string(),
                            );
                        } else {
                            u32::from(first)
                        };
                        let character = char::from_u32(code).ok_or_else(|| {
                            "invalid JSON unicode escape in build verification challenge".to_string()
                        })?;
                        output.push(character);
                    }
                    _ => return Err("invalid JSON escape in build verification challenge".to_string()),
                },
                0..=0x1f => return Err("control character in JSON string".to_string()),
                byte if byte.is_ascii() => output.push(char::from(byte)),
                _ => {
                    let start = self.index - 1;
                    let remaining = std::str::from_utf8(&self.input[start..])
                        .map_err(|_| "invalid UTF-8 in build verification challenge".to_string())?;
                    let character = remaining.chars().next().expect("non-empty UTF-8 slice");
                    output.push(character);
                    self.index = start + character.len_utf8();
                }
            }
        }
    }

    fn hex_code_unit(&mut self) -> Result<u16, String> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let byte = self.next().ok_or_else(|| "short JSON unicode escape".to_string())?;
            value = value * 16
                + match byte {
                    b'0'..=b'9' => u16::from(byte - b'0'),
                    b'a'..=b'f' => u16::from(byte - b'a' + 10),
                    b'A'..=b'F' => u16::from(byte - b'A' + 10),
                    _ => return Err("invalid JSON unicode escape".to_string()),
                };
        }
        Ok(value)
    }

    fn consume_digits(&mut self) -> bool {
        let start = self.index;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.index += 1;
        }
        self.index != start
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.index += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err("invalid JSON syntax in build verification challenge".to_string())
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.index).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.index += 1;
        Some(value)
    }
}

pub fn verify_expected_commit(
    actual: &str,
    expected: Option<&str>,
    field: &str,
) -> Result<String, String> {
    let actual = validate_commit_hash(Some(actual), &format!("actual {field} commit"))?
        .to_ascii_lowercase();
    let expected = validate_commit_hash(expected, &format!("expected {field} commit"))?
        .to_ascii_lowercase();
    if actual != expected {
        return Err(format!(
            "{field} commit mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(actual)
}

pub fn embedded_rap_compiler_commit() -> Result<String, String> {
    validate_commit_hash(
        option_env!("UNSOUND_SCANNER_RAP_BUILD_RUSTC_COMMIT"),
        "RAP compiler commit",
    )
}

pub fn nested_workspace_wrapper_reason(value: Option<&str>) -> Option<&'static str> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(|_| "nested_rustc_workspace_wrapper_unsupported")
}

pub fn rustc_commit_from_runner<F>(rustc_path: &str, runner: F) -> Result<String, String>
where
    F: FnOnce(&str) -> Result<String, String>,
{
    runner(rustc_path).and_then(|output| parse_rustc_commit(&output))
}

pub fn json_output_path(
    json_dir: Option<&Path>,
    legacy_output: Option<&Path>,
    unit_id: &str,
) -> Option<PathBuf> {
    json_dir
        .map(|directory| directory.join(format!("{unit_id}.json")))
        .or_else(|| legacy_output.map(Path::to_path_buf))
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    atomic_write_with_operations(path, bytes, write_and_sync, |from, to| fs::rename(from, to))
}

pub fn atomic_write_with_rename<F>(path: &Path, bytes: &[u8], rename: F) -> io::Result<()>
where
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    atomic_write_with_operations(path, bytes, write_and_sync, rename)
}

pub fn atomic_write_with_writer<F>(path: &Path, bytes: &[u8], writer: F) -> io::Result<()>
where
    F: FnOnce(&Path, &[u8]) -> io::Result<()>,
{
    atomic_write_with_operations(path, bytes, writer, |from, to| fs::rename(from, to))
}

fn atomic_write_with_operations<W, R>(
    path: &Path,
    bytes: &[u8],
    writer: W,
    rename: R,
) -> io::Result<()>
where
    W: FnOnce(&Path, &[u8]) -> io::Result<()>,
    R: FnOnce(&Path, &Path) -> io::Result<()>,
{
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    let temp_path = unique_temp_path(parent, file_name)?;
    let result = writer(&temp_path, bytes).and_then(|()| rename(&temp_path, path));
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn unique_temp_path(parent: &Path, file_name: &str) -> io::Result<PathBuf> {
    for _ in 0..128 {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{file_name}.tmp-{}-{counter}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate unique RAP JSON temporary file",
    ))
}

fn write_and_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = File::options().write(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()
}

fn required_environment_value(value: Option<String>, field: &str) -> Result<String, String> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("missing {field}"))
}

fn json_string(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len() + 2);
    encoded.push('"');
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\u{08}' => encoded.push_str("\\b"),
            '\u{0c}' => encoded.push_str("\\f"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                write!(encoded, "\\u{:04x}", character as u32)
                    .expect("writing JSON escape to String cannot fail");
            }
            character => encoded.push(character),
        }
    }
    encoded.push('"');
    encoded
}

fn split_crate_types(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn find_flag_value(args: &[String], flag: &str) -> Option<String> {
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == flag {
            return args.get(index + 1).cloned();
        }
        if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
            return Some(value.to_string());
        }
        index += 1;
    }
    None
}

fn find_flag_values(args: &[String], flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == flag {
            if let Some(value) = args.get(index + 1) {
                values.push(value.clone());
                index += 2;
                continue;
            }
        } else if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
            values.push(value.to_string());
        }
        index += 1;
    }
    values
}

fn find_codegen_value(args: &[String], key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if let Some(value) = arg.strip_prefix("-C") {
            if let Some(value) = value.strip_prefix(&prefix) {
                return Some(value.to_string());
            }
            if value.is_empty() {
                if let Some(value) = args
                    .get(index + 1)
                    .and_then(|next| next.strip_prefix(&prefix))
                {
                    return Some(value.to_string());
                }
            }
        }
        index += 1;
    }
    None
}

fn stable_fingerprint(values: &[&str]) -> String {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(value.len().to_string().as_bytes());
        bytes.push(b':');
        bytes.extend_from_slice(value.as_bytes());
        bytes.push(b'|');
    }
    fnv1a64_hex(&bytes)
}
