#!/usr/bin/env python3
"""Validated historical replay and real-project admission support for RAPx."""

from __future__ import annotations

import argparse
import importlib.util
import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable


SCHEMA = "rapx-fast-historical-replay-cases-v2"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CASE_KEYS = {
    "case_id",
    "repository",
    "repo_issue",
    "fix_commit",
    "pre_fix_commit",
    "manifest_path",
    "package",
    "target_kind",
    "target_name",
    "target_triple",
    "default_features",
    "features",
    "vulnerable_path",
    "vulnerable_snippet_sha256",
    "fixed_snippet_sha256",
    "fix_diff_sha256",
    "expected_rule_id",
    "capability_tier",
    "validation_status",
}
UNIT_STATUSES = {
    "not_started",
    "invalid_input",
    "control_failed",
    "analyzer_failed",
    "receipt_failed",
    "partial_timeout",
    "success",
}
PINNED_RUSTC_COMMIT = "1bc403daadbebb553ccc211a0a8eebb73989665f"
PINNED_CARGO_COMMIT = "15fbd2f607d4defc87053b8b76bf5038f2483cf4"
WALL_LIMIT_SECONDS = 15 * 60
PHYSICAL_LIMIT_BYTES = 8 * 1024**3


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"{field} must be a normalized repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized repository-relative path")
    return value


def _validated_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_case(case: Any) -> dict[str, Any]:
    if not isinstance(case, dict) or set(case) != CASE_KEYS:
        raise ValueError("historical case keys drifted")
    if not isinstance(case["case_id"], int) or isinstance(case["case_id"], bool):
        raise ValueError("historical case_id must be an integer")
    for field in ("repository", "repo_issue", "package", "target_name", "expected_rule_id"):
        if not isinstance(case[field], str) or not case[field]:
            raise ValueError(f"historical {field} must be a non-empty string")
    _relative_path(case["manifest_path"], "manifest_path")
    _relative_path(case["vulnerable_path"], "vulnerable_path")
    if case["target_kind"] not in {"lib", "bin"}:
        raise ValueError("historical target_kind must be lib or bin")
    target_triple = case["target_triple"]
    if target_triple is not None and (
        not isinstance(target_triple, str)
        or not re.fullmatch(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)+", target_triple)
    ):
        raise ValueError("historical target_triple must be null or an exact Rust target")
    features = case["features"]
    if not isinstance(case["default_features"], bool):
        raise ValueError("historical default_features must be a boolean")
    if (
        not isinstance(features, list)
        or features != sorted(set(features))
        or any(
            not isinstance(feature, str)
            or not feature
            or any(character.isspace() for character in feature)
            for feature in features
        )
    ):
        raise ValueError("historical features must be a sorted unique string list")
    if case["capability_tier"] not in {"existing_rule", "general_extension", "out_of_scope"}:
        raise ValueError("historical capability_tier is invalid")
    status = case["validation_status"]
    if status == "validated":
        for field in ("fix_commit", "pre_fix_commit"):
            if not isinstance(case[field], str) or not HEX40.fullmatch(case[field]):
                raise ValueError(f"validated historical {field} must be an exact commit")
        for field in (
            "vulnerable_snippet_sha256",
            "fixed_snippet_sha256",
            "fix_diff_sha256",
        ):
            _validated_hash(case[field], field)
    elif status == "invalid_input":
        for field in (
            "fix_commit",
            "pre_fix_commit",
            "vulnerable_snippet_sha256",
            "fixed_snippet_sha256",
            "fix_diff_sha256",
        ):
            if case[field] is not None:
                raise ValueError(f"invalid input {field} must be null")
    else:
        raise ValueError("historical validation_status is invalid")
    return dict(case)


def load_case_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema", "cases"}:
        raise ValueError("historical replay manifest keys drifted")
    if payload["schema"] != SCHEMA or not isinstance(payload["cases"], list):
        raise ValueError("historical replay manifest schema is invalid")
    cases = [_validate_case(case) for case in payload["cases"]]
    ids = [case["case_id"] for case in cases]
    if ids != list(range(1, len(cases) + 1)) or len(ids) != len(set(ids)):
        raise ValueError("historical replay cases must be uniquely and contiguously ordered")
    return {"schema": SCHEMA, "cases": cases}


def new_unit_result(
    *,
    case_id: int,
    package: str,
    target_kind: str,
    target_name: str,
    status: str,
    revision: str = "pre_fix",
    finding_count: int | None = None,
    pattern_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    if status not in UNIT_STATUSES:
        raise ValueError("unit result status is invalid")
    if revision not in {"pre_fix", "fixed"}:
        raise ValueError("unit result revision is invalid")
    if status == "success":
        if finding_count is None or pattern_counts is None:
            raise ValueError("successful result requires finding and pattern counts")
        expected = {f"pattern{number}" for number in range(1, 7)}
        if (
            not isinstance(finding_count, int)
            or isinstance(finding_count, bool)
            or finding_count < 0
            or set(pattern_counts) != expected
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in pattern_counts.values()
            )
            or sum(pattern_counts.values()) != finding_count
        ):
            raise ValueError("successful result counts are inconsistent")
    elif finding_count is not None or pattern_counts is not None:
        raise ValueError("non-success result counts must be null")
    return {
        "schema": "rapx-fast-historical-replay-unit-v2",
        "case_id": case_id,
        "package": package,
        "target_kind": target_kind,
        "target_name": target_name,
        "revision": revision,
        "status": status,
        "finding_count": finding_count,
        "pattern_counts": pattern_counts,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| case | package | target | status | findings | p1 | p2 | p3 | p4 | p5 | p6 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda value: (value["case_id"], value.get("revision", "pre_fix"), value["package"]),
    ):
        counts = row.get("pattern_counts")
        rendered = (
            [str(row["finding_count"])]
            + [str(counts[f"pattern{number}"]) for number in range(1, 7)]
            if row.get("status") == "success" and isinstance(counts, dict)
            else ["—"] * 7
        )
        lines.append(
            "| {case} | {package} | {kind}:{target} | {status} | {counts} |".format(
                case=f"{row['case_id']}/{row.get('revision', 'pre_fix')}",
                package=row["package"],
                kind=row["target_kind"],
                target=row["target_name"],
                status=row["status"],
                counts=" | ".join(rendered),
            )
        )
    return "\n".join(lines) + "\n"


def _target_record(
    package: dict[str, Any], target: dict[str, Any], repository: Path
) -> dict[str, str]:
    manifest = package.get("manifest_path")
    package_id = package.get("id")
    package_name = package.get("name")
    target_name = target.get("name")
    kinds = target.get("kind")
    if (
        not isinstance(manifest, str)
        or not isinstance(package_id, str)
        or not isinstance(package_name, str)
        or not isinstance(target_name, str)
        or not isinstance(kinds, list)
        or not all(isinstance(kind, str) for kind in kinds)
    ):
        raise ValueError("Cargo metadata target is malformed")
    try:
        relative_manifest = Path(manifest).resolve().relative_to(repository.resolve()).as_posix()
    except (OSError, ValueError) as error:
        raise ValueError("Cargo metadata manifest escapes repository") from error
    library_kinds = {"lib", "rlib", "dylib", "staticlib", "cdylib"}
    if library_kinds.intersection(kinds):
        kind = "lib"
    elif "bin" in kinds and "example" not in kinds:
        kind = "bin"
    else:
        raise ValueError("Cargo target is not an analyzable library or binary")
    return {
        "kind": kind,
        "manifest_path": relative_manifest,
        "package_id": package_id,
        "package": package_name,
        "name": target_name,
    }


def _metadata_packages(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    packages = metadata.get("packages")
    if not isinstance(packages, list) or not all(isinstance(package, dict) for package in packages):
        raise ValueError("Cargo metadata packages are invalid")
    return packages


def select_historical_target(
    metadata: dict[str, Any], repository: Path, case: dict[str, Any]
) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    for package in _metadata_packages(metadata):
        for raw_target in package.get("targets", []):
            if not isinstance(raw_target, dict):
                raise ValueError("Cargo metadata target is malformed")
            try:
                target = _target_record(package, raw_target, repository)
            except ValueError as error:
                if "not an analyzable" in str(error):
                    continue
                raise
            if (
                target["manifest_path"] == case["manifest_path"]
                and target["package"] == case["package"]
                and target["kind"] == case["target_kind"]
                and target["name"] == case["target_name"]
            ):
                matches.append(target)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest-bound historical target, found {len(matches)}")
    return matches[0]


def _preferred_target(package: dict[str, Any], repository: Path) -> dict[str, str] | None:
    libraries: list[dict[str, str]] = []
    binaries: list[dict[str, str]] = []
    targets = package.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Cargo metadata package targets are invalid")
    for raw_target in targets:
        if not isinstance(raw_target, dict):
            raise ValueError("Cargo metadata target is malformed")
        try:
            target = _target_record(package, raw_target, repository)
        except ValueError as error:
            if "not an analyzable" in str(error):
                continue
            raise
        (libraries if target["kind"] == "lib" else binaries).append(target)
    pool = libraries if libraries else binaries
    return sorted(pool, key=lambda item: (item["name"], item["manifest_path"]))[0] if pool else None


def select_default_targets(
    metadata: dict[str, Any], repository: Path, *, workspace_all: bool = False
) -> list[dict[str, str]]:
    packages = _metadata_packages(metadata)
    by_id = {package.get("id"): package for package in packages if isinstance(package.get("id"), str)}
    if workspace_all:
        selected_ids = metadata.get("workspace_members")
    else:
        resolve = metadata.get("resolve")
        root = resolve.get("root") if isinstance(resolve, dict) else None
        selected_ids = [root] if isinstance(root, str) else metadata.get("workspace_default_members")
        if not selected_ids:
            selected_ids = metadata.get("workspace_members")
    if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
        raise ValueError("Cargo metadata default member set is invalid")
    targets: list[dict[str, str]] = []
    for package_id in selected_ids:
        package = by_id.get(package_id)
        if package is None:
            raise ValueError("Cargo metadata member does not name a package")
        target = _preferred_target(package, repository)
        if target is not None:
            targets.append(target)
    return sorted(targets, key=lambda item: (item["manifest_path"], item["package"], item["name"]))


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and ".git" not in candidate.relative_to(root).parts
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lockfile_argv(cargo: Path, manifest_path: Path | str) -> list[str]:
    return [
        str(cargo),
        "-Z",
        "msrv-policy",
        "generate-lockfile",
        "--manifest-path",
        str(manifest_path),
    ]


def historical_lock_environment(environment: dict[str, str]) -> dict[str, str]:
    result = dict(environment)
    result["CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS"] = "fallback"
    return result


def prepare_locked_workspace(
    *,
    repository: Path,
    workspace_root: Path,
    manifest_path: str,
    copy_root: Path,
    cargo: Path,
    environment: dict[str, str],
    run_command: Callable[[list[str], Path, dict[str, str]], subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    workspace_root = workspace_root.resolve(strict=True)
    try:
        workspace_relative = workspace_root.relative_to(repository)
    except ValueError as error:
        raise ValueError("Cargo workspace root escapes repository") from error
    lockfile = workspace_root / "Cargo.lock"
    if lockfile.is_file():
        return {
            "repository": repository,
            "workspace_root": workspace_root,
            "lockfile": lockfile,
            "lock_sha256": _sha256_file(lockfile),
            "generated": False,
        }
    if copy_root.exists():
        raise ValueError("disposable lock workspace already exists")
    shutil.copytree(
        repository,
        copy_root,
        ignore=shutil.ignore_patterns(".git", "target", "__pycache__", "*.pyc"),
    )
    copied_workspace = copy_root / workspace_relative
    copied_manifest = copy_root / _relative_path(manifest_path, "manifest_path")
    result = run_command(
        lockfile_argv(cargo, copied_manifest),
        copied_workspace,
        historical_lock_environment(environment),
    )
    if result.returncode != 0:
        raise ValueError("pinned Cargo failed to generate a lockfile")
    copied_lock = copied_workspace / "Cargo.lock"
    if not copied_lock.is_file():
        raise ValueError("pinned Cargo succeeded without creating the workspace lockfile")
    return {
        "repository": copy_root.resolve(),
        "workspace_root": copied_workspace.resolve(),
        "lockfile": copied_lock.resolve(),
        "lock_sha256": _sha256_file(copied_lock),
        "generated": True,
    }


def collect_repository_receipts(
    receipt_dir: Path,
    *,
    repository_packages: set[str],
    selected_target: dict[str, str],
    require_selected: bool = True,
) -> dict[str, Any]:
    repository_receipts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for path in sorted(receipt_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("RAP receipt must be an object")
        schema = value.get("schema_version")
        if schema == "rap-unit-skip-v1":
            continue
        if schema != "rap-unit-v2" or value.get("success") is not True:
            raise ValueError("RAP receipt is malformed or unsuccessful")
        package = value.get("package_name")
        findings = value.get("findings")
        counts = value.get("pattern_counts")
        if (
            not isinstance(package, str)
            or not isinstance(findings, list)
            or value.get("finding_count") != len(findings)
            or not isinstance(counts, dict)
            or set(counts) != {f"pattern{number}" for number in range(1, 7)}
            or any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values())
            or sum(counts.values()) != len(findings)
        ):
            raise ValueError("RAP unit receipt counts are inconsistent")
        if package in repository_packages:
            repository_receipts.append(value)
        if package == selected_target["package"] and value.get("target_kind") == selected_target["kind"]:
            selected.append(value)
    if len(selected) > 1 or (require_selected and len(selected) != 1):
        raise ValueError(f"expected exactly one selected receipt, found {len(selected)}")
    return {
        "repository_receipts": repository_receipts,
        "selected_receipt": selected[0] if selected else None,
    }


def _append_features(argv: list[str], features: list[str]) -> None:
    if features:
        argv.extend(["--features", ",".join(features)])


def _append_target(argv: list[str], target_triple: str | None) -> None:
    if target_triple is not None:
        argv.extend(["--target", target_triple])


def scan_argv(
    cargo_rapx: Path,
    target: dict[str, str],
    features: list[str] | None = None,
    target_triple: str | None = None,
    default_features: bool = True,
) -> list[str]:
    package = target.get("package_id")
    if not isinstance(package, str) or not package:
        raise ValueError("selected target lacks an exact Cargo package ID")
    argv = [
        str(cargo_rapx),
        "rapx",
        "-unsoundaudit",
        "--",
        "--locked",
        "--jobs",
        "1",
        "--manifest-path",
        target["manifest_path"],
        "--package",
        package,
    ]
    argv.extend(["--lib"] if target["kind"] == "lib" else ["--bin", target["name"]])
    if not default_features:
        argv.append("--no-default-features")
    _append_features(argv, features or [])
    _append_target(argv, target_triple)
    return argv


def result_from_execution(
    *,
    case_id: int,
    target: dict[str, str],
    execution_status: str,
    receipts: dict[str, Any] | None,
    revision: str = "pre_fix",
) -> dict[str, Any]:
    if execution_status == "success" and receipts is not None:
        selected = receipts["selected_receipt"]
        result = new_unit_result(
            case_id=case_id,
            package=target["package"],
            target_kind=target["kind"],
            target_name=target["name"],
            status="success",
            revision=revision,
            finding_count=selected["finding_count"],
            pattern_counts=selected["pattern_counts"],
        )
    else:
        status = "partial_timeout" if execution_status in {"timeout", "memory_limit"} else "analyzer_failed"
        result = new_unit_result(
            case_id=case_id,
            package=target["package"],
            target_kind=target["kind"],
            target_name=target["name"],
            status=status,
            revision=revision,
        )
    result["partial_receipt_count"] = (
        len(receipts["repository_receipts"]) if receipts is not None else 0
    )
    return result


def attach_recall_outcome(
    row: dict[str, Any], case: dict[str, Any], revision: str
) -> dict[str, Any]:
    row["features"] = list(case["features"])
    row["default_features"] = case["default_features"]
    row["target_triple"] = case["target_triple"]
    expected_rule = case["expected_rule_id"]
    row["expected_rule_id"] = expected_rule
    if row.get("status") != "success":
        row["expected_rule_finding_count"] = None
        row["recall_outcome"] = None
        return row
    findings = row.get("findings")
    if not isinstance(findings, list):
        raise ValueError("successful replay result lacks complete findings")
    count = sum(
        1
        for finding in findings
        if isinstance(finding, dict)
        and finding.get("rule_id") == expected_rule
        and isinstance(finding.get("first_contract_failure"), dict)
        and isinstance(finding["first_contract_failure"].get("span"), dict)
        and finding["first_contract_failure"]["span"].get("path")
        == case["vulnerable_path"]
    )
    row["expected_rule_finding_count"] = count
    if revision == "pre_fix":
        if count:
            row["recall_outcome"] = "detected"
        elif case["capability_tier"] == "out_of_scope":
            row["recall_outcome"] = "validated_out_of_scope"
        else:
            row["recall_outcome"] = "successful_zero"
    else:
        row["recall_outcome"] = "fixed_clean" if count == 0 else "fixed_rule_remaining"
    return row


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"git query failed: {error}") from error


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"git content query failed: {error}") from error


def verify_case_locator(repository: Path, case: dict[str, Any]) -> None:
    if case["validation_status"] != "validated":
        raise ValueError("only validated locators may be verified")
    path = case["vulnerable_path"]
    vulnerable = _git_bytes(repository, "show", f"{case['pre_fix_commit']}:{path}")
    fixed = _git_bytes(repository, "show", f"{case['fix_commit']}:{path}")
    difference = _git_bytes(
        repository,
        "diff",
        "--no-ext-diff",
        "--binary",
        case["pre_fix_commit"],
        case["fix_commit"],
        "--",
        path,
    )
    observed = {
        "vulnerable_snippet_sha256": hashlib.sha256(vulnerable).hexdigest(),
        "fixed_snippet_sha256": hashlib.sha256(fixed).hexdigest(),
        "fix_diff_sha256": hashlib.sha256(difference).hexdigest(),
    }
    for field, digest in observed.items():
        if digest != case[field]:
            raise ValueError(f"historical locator {field} drifted")


def _load_resource_runner() -> Any:
    directory = Path(__file__).resolve().parent / "unsoundaudit-v2"
    path = directory / "run_stdlib_pilot.py"
    module_name = "rapx_historical_replay_v2_resource_runner"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load resource runner from {path}")
    sys.path.insert(0, str(directory))
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _run_limited(
    runner: Any,
    argv: list[str],
    cwd: Path,
    environment: dict[str, str],
    log: Path,
) -> Any:
    log.parent.mkdir(parents=True, exist_ok=True)
    return runner.run_limited(
        argv,
        cwd,
        environment,
        log,
        WALL_LIMIT_SECONDS,
        PHYSICAL_LIMIT_BYTES,
        memory_limit_metric="physical_footprint",
    )


def _metadata_from_log(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cargo metadata output is not one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("Cargo metadata output is not an object")
    return value


def _nearest_lock(repository: Path, manifest_path: str) -> Path | None:
    manifest = repository / _relative_path(manifest_path, "manifest_path")
    if not manifest.is_file():
        raise ValueError("historical manifest is missing")
    current = manifest.parent
    while True:
        candidate = current / "Cargo.lock"
        if candidate.is_file():
            return candidate
        if current == repository:
            return None
        if repository not in current.parents:
            raise ValueError("historical manifest escapes repository")
        current = current.parent


def _command_environment(
    *,
    cargo: Path,
    rustc: Path,
    cargo_home: Path,
    target_dir: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CARGO_HOME": str(cargo_home),
            "CARGO_TARGET_DIR": str(target_dir),
            "CARGO_BUILD_JOBS": "1",
            "RUSTC": str(rustc),
            "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(cargo),
            "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": PINNED_CARGO_COMMIT,
            "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(rustc),
            "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": PINNED_RUSTC_COMMIT,
        }
    )
    rustflags = environment.get("RUSTFLAGS", "").strip()
    environment["RUSTFLAGS"] = " ".join(
        component for component in (rustflags, "--cap-lints allow") if component
    )
    return environment


def _metadata_command(cargo: Path, manifest_path: str) -> list[str]:
    return [
        str(cargo),
        "metadata",
        "--quiet",
        "--locked",
        "--no-deps",
        "--format-version",
        "1",
        "--manifest-path",
        manifest_path,
    ]


def _prepare_case_workspace(
    *,
    runner: Any,
    repository: Path,
    case: dict[str, Any],
    case_root: Path,
    cargo: Path,
    environment: dict[str, str],
) -> tuple[Path, bool]:
    lockfile = _nearest_lock(repository, case["manifest_path"])
    if lockfile is not None:
        return repository, False
    copy = case_root / "work"
    shutil.copytree(
        repository,
        copy,
        ignore=shutil.ignore_patterns(".git", "target", "__pycache__", "*.pyc"),
    )
    result = _run_limited(
        runner,
        lockfile_argv(cargo, case["manifest_path"]),
        copy,
        historical_lock_environment(environment),
        case_root / "logs" / "generate-lockfile.log",
    )
    if result.classification != "success":
        raise ValueError(f"lock generation failed: {result.classification}")
    return copy, True


def _repository_packages(metadata: dict[str, Any], repository: Path) -> set[str]:
    names = set()
    for package in _metadata_packages(metadata):
        manifest = package.get("manifest_path")
        name = package.get("name")
        if not isinstance(manifest, str) or not isinstance(name, str):
            raise ValueError("Cargo metadata package is malformed")
        try:
            Path(manifest).resolve().relative_to(repository.resolve())
        except ValueError:
            continue
        names.add(name)
    return names


def _control_argv(
    cargo: Path,
    target: dict[str, str],
    features: list[str] | None = None,
    target_triple: str | None = None,
    default_features: bool = True,
) -> list[str]:
    argv = [
        str(cargo),
        "check",
        "--locked",
        "--jobs",
        "1",
        "--manifest-path",
        target["manifest_path"],
        "--package",
        target["package_id"],
    ]
    argv.extend(["--lib"] if target["kind"] == "lib" else ["--bin", target["name"]])
    if not default_features:
        argv.append("--no-default-features")
    _append_features(argv, features or [])
    _append_target(argv, target_triple)
    return argv


def _journal(result_root: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (row["case_id"], row.get("revision", "pre_fix"), row["package"]),
    )
    with (result_root / "results.jsonl").open("w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (result_root / "results.md").write_text(markdown_table(ordered), encoding="utf-8")


def run_replay(
    *,
    case_manifest: Path,
    cargo: Path,
    rustc: Path,
    cargo_rapx: Path,
    result_root: Path,
    selected_cases: set[int] | None = None,
    revision: str = "pre_fix",
) -> list[dict[str, Any]]:
    if revision not in {"pre_fix", "fixed"}:
        raise ValueError("replay revision must be pre_fix or fixed")
    commit_field = "pre_fix_commit" if revision == "pre_fix" else "fix_commit"
    if result_root.exists():
        raise ValueError("result root already exists; replay v2 is append-only")
    for executable, label in ((cargo, "Cargo"), (rustc, "rustc"), (cargo_rapx, "cargo-rapx")):
        if not executable.resolve(strict=True).is_file():
            raise ValueError(f"{label} executable is missing")
    if not cargo_rapx.with_name("rapx").is_file():
        raise ValueError("cargo-rapx sibling rapx is missing")
    manifest = load_case_manifest(case_manifest)
    result_root.mkdir(parents=True)
    runner = _load_resource_runner()
    rows: list[dict[str, Any]] = []
    shared_cargo_home = result_root / "cargo-home"
    shared_cargo_home.mkdir()
    for case in manifest["cases"]:
        if selected_cases is not None and case["case_id"] not in selected_cases:
            continue
        if case["validation_status"] == "invalid_input":
            row = new_unit_result(
                case_id=case["case_id"],
                package=case["package"],
                target_kind=case["target_kind"],
                target_name=case["target_name"],
                status="invalid_input",
                revision=revision,
            )
            row["partial_receipt_count"] = 0
            attach_recall_outcome(row, case, revision)
            rows.append(row)
            _journal(result_root, rows)
            continue
        case_root = result_root / "cases" / f"case-{case['case_id']:02d}"
        source = case_root / "source"
        logs = case_root / "logs"
        receipts_dir = case_root / "receipts"
        logs.mkdir(parents=True)
        receipts_dir.mkdir()
        environment = _command_environment(
            cargo=cargo,
            rustc=rustc,
            cargo_home=shared_cargo_home,
            target_dir=case_root / "control-target",
        )
        clone = _run_limited(
            runner,
            [
                "git",
                "clone",
                "--no-recurse-submodules",
                f"https://github.com/{case['repository']}.git",
                str(source),
            ],
            case_root,
            environment,
            logs / "clone.log",
        )
        if clone.classification != "success":
            row = new_unit_result(
                case_id=case["case_id"],
                package=case["package"],
                target_kind=case["target_kind"],
                target_name=case["target_name"],
                status="control_failed",
                revision=revision,
            )
            row.update({"failure": f"clone_{clone.classification}", "partial_receipt_count": 0})
            attach_recall_outcome(row, case, revision)
            rows.append(row)
            _journal(result_root, rows)
            continue
        try:
            verify_case_locator(source, case)
            checkout = _run_limited(
                runner,
                ["git", "checkout", "--detach", case[commit_field]],
                source,
                environment,
                logs / "checkout.log",
            )
            if checkout.classification != "success":
                raise ValueError(f"checkout failed: {checkout.classification}")
            source_tree = _git_output(source, "rev-parse", "HEAD^{tree}")
            if _git_output(source, "status", "--porcelain"):
                raise ValueError("historical source checkout is dirty")
            workspace, generated = _prepare_case_workspace(
                runner=runner,
                repository=source,
                case=case,
                case_root=case_root,
                cargo=cargo,
                environment=environment,
            )
            environment["CARGO_TARGET_DIR"] = str(case_root / "control-target")
            metadata_log = logs / "metadata.log"
            metadata_result = _run_limited(
                runner,
                _metadata_command(cargo, case["manifest_path"]),
                workspace,
                environment,
                metadata_log,
            )
            if metadata_result.classification != "success":
                raise ValueError(f"metadata failed: {metadata_result.classification}")
            metadata = _metadata_from_log(metadata_log)
            target = select_historical_target(metadata, workspace, case)
            workspace_root_value = metadata.get("workspace_root")
            if not isinstance(workspace_root_value, str):
                raise ValueError("Cargo metadata workspace root is invalid")
            workspace_root = Path(workspace_root_value).resolve(strict=True)
            workspace_root.relative_to(workspace.resolve())
            lockfile = workspace_root / "Cargo.lock"
            if not lockfile.is_file():
                raise ValueError("Cargo metadata workspace has no frozen lockfile")
            lock_sha = _sha256_file(lockfile)
            control = _run_limited(
                runner,
                _control_argv(
                    cargo,
                    target,
                    case["features"],
                    case["target_triple"],
                    case["default_features"],
                ),
                workspace,
                environment,
                logs / "control.log",
            )
            if control.classification != "success":
                row = new_unit_result(
                    case_id=case["case_id"],
                    package=case["package"],
                    target_kind=case["target_kind"],
                    target_name=case["target_name"],
                    status="control_failed",
                    revision=revision,
                )
                row.update(
                    {
                        "failure": control.classification,
                        "generated_lock": generated,
                        "lock_sha256": lock_sha,
                        "partial_receipt_count": 0,
                    }
                )
                attach_recall_outcome(row, case, revision)
                rows.append(row)
                _journal(result_root, rows)
                continue
            environment.update(
                {
                    "CARGO_TARGET_DIR": str(case_root / "scan-target"),
                    "UNSOUND_SCANNER_RAP_JSON_DIR": str(receipts_dir),
                    "UNSOUND_SCANNER_PROJECT_ROOT": str(workspace),
                }
            )
            scan = _run_limited(
                runner,
                scan_argv(
                    cargo_rapx,
                    target,
                    case["features"],
                    case["target_triple"],
                    case["default_features"],
                ),
                workspace,
                environment,
                logs / "scan.log",
            )
            repository_packages = _repository_packages(metadata, workspace)
            require_selected = scan.classification == "success"
            try:
                receipts = collect_repository_receipts(
                    receipts_dir,
                    repository_packages=repository_packages,
                    selected_target=target,
                    require_selected=require_selected,
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                row = new_unit_result(
                    case_id=case["case_id"],
                    package=case["package"],
                    target_kind=case["target_kind"],
                    target_name=case["target_name"],
                    status="receipt_failed",
                    revision=revision,
                )
                row.update({"failure": str(error), "partial_receipt_count": 0})
            else:
                if scan.classification == "success":
                    row = result_from_execution(
                        case_id=case["case_id"],
                        target=target,
                        execution_status="success",
                        receipts=receipts,
                        revision=revision,
                    )
                    selected = receipts["selected_receipt"]
                    if selected is None:
                        raise ValueError("successful scan lacks selected receipt")
                    row["findings"] = selected["findings"]
                else:
                    row = result_from_execution(
                        case_id=case["case_id"],
                        target=target,
                        execution_status=scan.classification,
                        receipts=receipts,
                        revision=revision,
                    )
                row.update(
                    {
                        "execution": runner.process_result_record(
                            "scan", 1, scan, "physical_footprint"
                        ),
                        "generated_lock": generated,
                        "lock_sha256": lock_sha,
                        "source_commit": case[commit_field],
                        "source_tree": source_tree,
                        "features": case["features"],
                        "target_triple": case["target_triple"],
                    }
                )
            if _git_output(source, "status", "--porcelain"):
                raise ValueError("replay mutated original historical checkout")
        except ValueError as error:
            row = new_unit_result(
                case_id=case["case_id"],
                package=case["package"],
                target_kind=case["target_kind"],
                target_name=case["target_name"],
                status="invalid_input",
                revision=revision,
            )
            row.update({"failure": str(error), "partial_receipt_count": 0})
        attach_recall_outcome(row, case, revision)
        rows.append(row)
        _journal(result_root, rows)
    _write_json(
        result_root / "environment.json",
        {
            "schema": "rapx-fast-historical-replay-environment-v2",
            "rustc_commit": PINNED_RUSTC_COMMIT,
            "cargo_commit": PINNED_CARGO_COMMIT,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "physical_footprint_limit_bytes": PHYSICAL_LIMIT_BYTES,
            "case_manifest_sha256": _sha256_file(case_manifest),
            "revision": revision,
            "historical_lint_cap": "allow",
        },
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--cargo-rapx", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append")
    parser.add_argument("--revision", choices=("pre_fix", "fixed"), default="pre_fix")
    arguments = parser.parse_args()
    run_replay(
        case_manifest=arguments.case_manifest,
        cargo=arguments.cargo,
        rustc=arguments.rustc,
        cargo_rapx=arguments.cargo_rapx,
        result_root=arguments.result_root,
        selected_cases=set(arguments.case) if arguments.case else None,
        revision=arguments.revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
