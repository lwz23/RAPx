#!/usr/bin/env python3
"""Fail-closed orchestration for the RAPx real-project candidate-quality pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


PROFILE_SCHEMA = "rapx-real-project-profile-contract-v1"
SELECTION_SCHEMA = "rapx-real-project-selection-manifest-v1"
SELECTION_SEED = "rapx-v2-real-project-pilot-v1"
STRATA = ("stars_top250", "downloads_top250")
LEGACY_SOURCE_COMMIT = "75ded1f25f3d5a1d81d899c2fe070e34d1f5f8ad"
LEGACY_RUSTC_COMMIT = "1bc403daadbebb553ccc211a0a8eebb73989665f"
LEGACY_CARGO_COMMIT = "15fbd2f607d4defc87053b8b76bf5038f2483cf4"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
RUN_LOCAL_FIELDS = {
    "generated_at",
    "started_at",
    "finished_at",
    "duration_ms",
    "absolute_log_path",
    "target_dir",
}
REVIEW_DISPOSITIONS = {
    "new_iue_candidate",
    "likely_false_positive",
    "unresolved",
    "duplicate",
    "tool_or_receipt_failure",
}
FALSE_POSITIVE_FAMILIES = {
    "source_misattribution",
    "missing_dominating_validation",
    "interprocedural_summary_imprecision",
    "sink_contract_imprecision",
    "primary_pattern_precedence",
    "deduplication",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error


def _closed_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys drifted")


def _relative_executable(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("profile executable must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or "\\" in value:
        raise ValueError("profile executable must be a normalized relative path")
    return value


def load_profile_contract(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("profile contract must be an object")
    profiles = value.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 3:
        raise ValueError("profile contract must contain exactly three profiles")
    _closed_keys(
        value,
        {"schema", "rap_source_commit", "rustc_commit", "cargo_commit", "profiles"},
        "profile contract",
    )
    if value["schema"] != PROFILE_SCHEMA:
        raise ValueError("profile contract schema drifted")
    if value["rap_source_commit"] != LEGACY_SOURCE_COMMIT:
        raise ValueError("legacy RAP source commit drifted")
    if value["rustc_commit"] != LEGACY_RUSTC_COMMIT:
        raise ValueError("legacy rustc commit drifted")
    if value["cargo_commit"] != LEGACY_CARGO_COMMIT:
        raise ValueError("legacy Cargo commit drifted")
    names: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("profile record must be an object")
        _closed_keys(
            profile,
            {"name", "executable", "sha256", "rapx_sha256", "argv"},
            "profile record",
        )
        name = profile["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("profile names must be unique and non-empty")
        names.add(name)
        _relative_executable(profile["executable"])
        if not isinstance(profile["sha256"], str) or not HEX64.fullmatch(profile["sha256"]):
            raise ValueError("profile executable hash must be lowercase SHA-256")
        if not isinstance(profile["rapx_sha256"], str) or not HEX64.fullmatch(
            profile["rapx_sha256"]
        ):
            raise ValueError("profile rapx sibling hash must be lowercase SHA-256")
        argv = profile["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("profile argv must be a non-empty string array")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def finding_key(finding: dict[str, Any]) -> str:
    if not isinstance(finding, dict):
        raise ValueError("finding must be an object")
    semantic = {key: value for key, value in finding.items() if key != "finding_id"}
    return hashlib.sha256(_canonical_bytes(semantic)).hexdigest()


def finding_root_key(finding: dict[str, Any]) -> str:
    causal = finding.get("causal_key")
    pattern = finding.get("primary_pattern")
    if not isinstance(causal, dict) or not isinstance(pattern, str):
        raise ValueError("finding lacks causal key or primary pattern")
    return hashlib.sha256(
        _canonical_bytes({"causal_key": causal, "primary_pattern": pattern})
    ).hexdigest()


def root_cause_key(repository: str, finding: dict[str, Any]) -> str:
    payload = {
        "repository": normalize_repository(repository),
        "rule_id": finding.get("rule_id"),
        "primary_pattern": finding.get("primary_pattern"),
        "source": finding.get("source"),
        "sink": finding.get("sink"),
        "first_contract_failure": finding.get("first_contract_failure"),
        "obligation": finding.get("obligation"),
        "validation_status": finding.get("validation_status"),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_review_row(
    row: dict[str, Any], known_roots: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("review row must be an object")
    if row.get("error_family") == "repository_allowlist":
        raise ValueError("repository_allowlist is forbidden")
    disposition = row.get("disposition")
    if disposition not in REVIEW_DISPOSITIONS:
        raise ValueError("review disposition is invalid")
    evidence = row.get("evidence_zh")
    rationale = row.get("rationale_zh")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("review evidence_zh must be non-empty")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("review rationale_zh must be non-empty")
    family = row.get("error_family")
    if disposition == "likely_false_positive":
        if family not in FALSE_POSITIVE_FAMILIES:
            raise ValueError("likely false positive requires a generic error family")
    elif family is not None:
        raise ValueError("error_family is only valid for likely false positives")
    duplicate = row.get("duplicate_of")
    if disposition == "duplicate":
        if not isinstance(duplicate, str) or not HEX64.fullmatch(duplicate):
            raise ValueError("duplicate review requires a root-key reference")
        if duplicate == row.get("root_key"):
            raise ValueError("duplicate review cannot reference itself")
        if known_roots is not None and duplicate not in known_roots:
            raise ValueError("duplicate review references an unknown root key")
    elif duplicate is not None:
        raise ValueError("duplicate_of is only valid for duplicate reviews")
    return row


def validate_review_ledger(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    roots = [row.get("root_key") for row in rows]
    if any(not isinstance(root, str) or not HEX64.fullmatch(root) for root in roots):
        raise ValueError("review root keys must be lowercase SHA-256 values")
    if len(roots) != len(set(roots)):
        raise ValueError("review ledger has duplicate root keys")
    known = set(roots)
    return [validate_review_row(row, known) for row in rows]


def normalize_receipt(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {
            key: normalize_receipt(item)
            for key, item in value.items()
            if key not in RUN_LOCAL_FIELDS
        }
        findings = normalized.get("findings")
        if isinstance(findings, list):
            if not all(isinstance(finding, dict) for finding in findings):
                raise ValueError("receipt findings must be objects")
            normalized["findings"] = sorted(findings, key=finding_key)
        return normalized
    if isinstance(value, list):
        return [normalize_receipt(item) for item in value]
    return value


def _root_pattern_set(receipt: dict[str, Any]) -> set[tuple[str, str]]:
    findings = receipt.get("findings")
    if not isinstance(findings, list) or not all(
        isinstance(finding, dict) for finding in findings
    ):
        raise ValueError("receipt findings must be an array of objects")
    result: set[tuple[str, str]] = set()
    for finding in findings:
        pattern = finding.get("primary_pattern")
        if not isinstance(pattern, str):
            raise ValueError("finding lacks primary_pattern")
        result.add((finding_root_key(finding), pattern))
    return result


def profile_stable_findings(
    canonical: dict[str, Any], debug: dict[str, Any], release: dict[str, Any]
) -> list[dict[str, Any]]:
    canonical_set = _root_pattern_set(canonical)
    if canonical_set != _root_pattern_set(debug) or canonical_set != _root_pattern_set(release):
        return []
    findings = canonical.get("findings")
    assert isinstance(findings, list)
    return sorted(findings, key=finding_key)


def normalize_repository(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("repository name must be a string")
    normalized = value.strip().removesuffix("/").removesuffix(".git").lower()
    if normalized.count("/") != 1 or any(not part for part in normalized.split("/")):
        raise ValueError(f"invalid owner/repository name: {value!r}")
    return normalized


def _selection_record(row: dict[str, Any], stratum: str) -> dict[str, Any]:
    strata = row.get("strata")
    detail = strata.get(stratum) if isinstance(strata, dict) else None
    if not isinstance(detail, dict):
        raise ValueError("eligible roster row has invalid stratum detail")
    repo_id = row.get("repo_id")
    if not isinstance(repo_id, int) or isinstance(repo_id, bool) or repo_id < 0:
        raise ValueError("eligible roster row has invalid numeric repo_id")
    repository = normalize_repository(row.get("full_name"))
    url = row.get("repository_url")
    branch = row.get("default_branch")
    if not isinstance(url, str) or not url or not isinstance(branch, str) or not branch:
        raise ValueError("eligible roster row lacks repository URL or default branch")
    rank = detail.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("eligible roster row has invalid stratum rank")
    return {
        "repo_id": repo_id,
        "repository": repository,
        "repository_url": url,
        "default_branch": branch,
        "stratum": stratum,
        "stratum_rank": rank,
    }


def build_selection_manifest(
    roster_rows: list[dict[str, Any]],
    known_repositories: set[str],
    development_repositories: set[str],
    selected_count: int = 5,
    reserve_per_stratum: int = 25,
) -> dict[str, Any]:
    if selected_count <= 0 or reserve_per_stratum < 0:
        raise ValueError("selection and reserve counts must be non-negative")
    excluded = {normalize_repository(value) for value in known_repositories}
    excluded.update(normalize_repository(value) for value in development_repositories)
    candidates: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    seen_ids: set[int] = set()
    for row in roster_rows:
        if not isinstance(row, dict) or row.get("eligible") is not True:
            continue
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, int) or isinstance(repo_id, bool) or repo_id in seen_ids:
            raise ValueError("eligible roster repo_id is invalid or duplicated")
        seen_ids.add(repo_id)
        repository = normalize_repository(row.get("full_name"))
        if repository in excluded:
            continue
        row_strata = row.get("strata")
        if not isinstance(row_strata, dict):
            raise ValueError("eligible roster row has no strata object")
        matched = [stratum for stratum in STRATA if stratum in row_strata]
        if len(matched) != 1:
            raise ValueError("eligible roster row must belong to exactly one pilot stratum")
        candidates[matched[0]].append(_selection_record(row, matched[0]))
    required = selected_count + reserve_per_stratum
    for stratum in STRATA:
        candidates[stratum].sort(
            key=lambda row: (
                hashlib.sha256(f"{SELECTION_SEED}:{row['repo_id']}".encode()).hexdigest(),
                row["repo_id"],
            )
        )
        if len(candidates[stratum]) < required:
            raise ValueError(f"insufficient eligible repositories in {stratum}")
    return {
        "schema": SELECTION_SCHEMA,
        "seed": SELECTION_SEED,
        "selected_count_per_stratum": selected_count,
        "reserve_count_per_stratum": reserve_per_stratum,
        "excluded_names": sorted(excluded),
        "selected": {
            stratum: candidates[stratum][:selected_count] for stratum in STRATA
        },
        "reserves": {
            stratum: candidates[stratum][selected_count:required] for stratum in STRATA
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                raise ValueError(f"blank JSONL row at {path}:{number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{number}")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL input {path}: {error}") from error
    return rows


def _development_exclusions(path: Path) -> set[str]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("development exclusions must be an object")
    _closed_keys(value, {"schema", "records"}, "development exclusions")
    if value["schema"] != "rapx-real-project-development-exclusions-v1":
        raise ValueError("development exclusions schema drifted")
    if not isinstance(value["records"], list):
        raise ValueError("development exclusions records must be an array")
    repositories: set[str] = set()
    for record in value["records"]:
        if not isinstance(record, dict):
            raise ValueError("development exclusion record must be an object")
        _closed_keys(record, {"repository", "reason", "source_document"}, "exclusion record")
        if not all(isinstance(record[key], str) and record[key] for key in record):
            raise ValueError("development exclusion fields must be non-empty strings")
        repository = normalize_repository(record["repository"])
        if repository in repositories:
            raise ValueError("duplicate development exclusion")
        repositories.add(repository)
    return repositories


def select_from_inputs(
    roster_path: Path,
    instances_path: Path,
    exclusions_path: Path,
    output: Path,
    reserve_per_stratum: int,
) -> dict[str, Any]:
    roster = _read_jsonl(roster_path)
    instances = _read_jsonl(instances_path)
    known = {
        normalize_repository(row.get("repository"))
        for row in instances
        if isinstance(row.get("repository"), str)
    }
    if len(known) == 0:
        raise ValueError("IUE input contains no canonical repositories")
    result = build_selection_manifest(
        roster, known, _development_exclusions(exclusions_path), 5, reserve_per_stratum
    )
    result["inputs"] = {
        "roster_sha256": sha256_file(roster_path),
        "instances_sha256": sha256_file(instances_path),
        "development_exclusions_sha256": sha256_file(exclusions_path),
        "roster_records": len(roster),
        "instance_records": len(instances),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def admit_candidates(
    manifest: dict[str, Any], runner: Any, required_per_stratum: int = 5
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    admitted: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    for stratum in STRATA:
        selected = manifest.get("selected", {}).get(stratum, [])
        reserves = manifest.get("reserves", {}).get(stratum, [])
        if not isinstance(selected, list) or not isinstance(reserves, list):
            raise ValueError(f"manifest candidates are invalid for {stratum}")
        open_replacements: list[str] = []
        for position, original in enumerate([*selected, *reserves]):
            if len(admitted[stratum]) >= required_per_stratum:
                break
            if not isinstance(original, dict) or original.get("stratum") != stratum:
                raise ValueError("manifest candidate has wrong stratum")
            candidate = dict(original)
            replacement_parent = None
            if position >= len(selected):
                if not open_replacements:
                    break
                replacement_parent = open_replacements[0]
                candidate["replacement_parent"] = replacement_parent
            outcome = runner(candidate)
            if not isinstance(outcome, dict) or not isinstance(outcome.get("status"), str):
                raise ValueError("admission runner returned an invalid outcome")
            attempt = {**candidate, **outcome}
            attempts.append(attempt)
            if outcome["status"] == "admitted":
                admitted[stratum].append(attempt)
                if replacement_parent is not None:
                    open_replacements.pop(0)
            elif position < len(selected):
                open_replacements.append(candidate["repository"])
        if len(admitted[stratum]) < required_per_stratum:
            attempts.append(
                {
                    "status": "reserve_exhausted",
                    "stratum": stratum,
                    "required": required_per_stratum,
                    "admitted": len(admitted[stratum]),
                }
            )
    return {
        "attempts": attempts,
        "admitted": admitted,
        "admitted_by_stratum": {
            stratum: len(admitted[stratum]) for stratum in STRATA
        },
    }


def _run_logged(
    argv: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
        output = result.stdout
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        output = partial + "\ncommand timed out after 900 seconds\n"
        result = subprocess.CompletedProcess(argv, 124, output, None)
    except OSError as error:
        output = f"command launch failed: {error}\n"
        result = subprocess.CompletedProcess(argv, 127, output, None)
    log_path.write_text(output, encoding="utf-8")
    return result


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
        raise ValueError(f"git query failed in admitted repository: {error}") from error


def _select_cargo_target(metadata: dict[str, Any], repository: Path) -> dict[str, str] | None:
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise ValueError("cargo metadata has no packages array")
    libraries: list[tuple[str, str, str, str]] = []
    binaries: list[tuple[str, str, str, str]] = []
    library_kinds = {"lib", "rlib", "dylib", "staticlib", "cdylib"}
    for package in packages:
        if not isinstance(package, dict):
            continue
        package_name = package.get("name")
        manifest = package.get("manifest_path")
        targets = package.get("targets")
        if not isinstance(package_name, str) or not isinstance(manifest, str) or not isinstance(targets, list):
            continue
        try:
            relative_manifest = Path(manifest).resolve().relative_to(repository.resolve()).as_posix()
        except (OSError, ValueError):
            raise ValueError("cargo metadata manifest escapes repository")
        for target in targets:
            if not isinstance(target, dict) or not isinstance(target.get("name"), str):
                continue
            kinds = target.get("kind")
            if not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds):
                continue
            record = (relative_manifest, package_name, target["name"], manifest)
            if library_kinds.intersection(kinds):
                libraries.append(record)
            elif "bin" in kinds and "example" not in kinds:
                binaries.append(record)
    using_library = bool(libraries)
    pool = sorted(libraries) if using_library else sorted(binaries)
    if not pool:
        return None
    relative_manifest, package_name, target_name, manifest = pool[0]
    return {
        "kind": "lib" if using_library else "bin",
        "manifest_path": relative_manifest,
        "manifest_sha256": sha256_file(Path(manifest)),
        "package": package_name,
        "name": target_name,
    }


class RealAdmissionRunner:
    def __init__(
        self, clone_root: Path, result_root: Path, cargo: Path, cutoff: str
    ) -> None:
        self.clone_root = clone_root.resolve()
        self.result_root = result_root.resolve()
        self.cargo = cargo.resolve(strict=True)
        self.cutoff = cutoff
        self.clone_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)

    def __call__(self, candidate: dict[str, Any]) -> dict[str, Any]:
        label = f"{candidate['stratum']}-{candidate['repo_id']}"
        repository = self.clone_root / label
        log_dir = self.result_root / "logs" / label
        clone_argv = [
            "git",
            "clone",
            "--no-recurse-submodules",
            candidate["repository_url"],
            str(repository),
        ]
        if repository.exists():
            return {"status": "clone_target_exists", "clone_argv": clone_argv}
        clone = _run_logged(clone_argv, self.clone_root, log_dir / "clone.log")
        base = {
            "clone_argv": clone_argv,
            "clone_exit": clone.returncode,
            "clone_log": f"logs/{label}/clone.log",
            "clone_path": str(repository),
        }
        if clone.returncode != 0:
            return {**base, "status": "clone_failed"}
        try:
            symbolic = _git_output(repository, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
            prefix = "origin/"
            if not symbolic.startswith(prefix):
                raise ValueError("origin HEAD is not a remote branch")
            default_branch = symbolic.removeprefix(prefix)
            revision = _git_output(
                repository,
                "rev-list",
                "-1",
                f"--before={self.cutoff}",
                f"refs/remotes/origin/{default_branch}",
            )
        except ValueError as error:
            return {**base, "status": "default_branch_failed", "error": str(error)}
        if not revision:
            return {**base, "status": "pre_cutoff_commit_missing", "default_branch": default_branch}
        checkout = _run_logged(
            ["git", "checkout", "--detach", revision], repository, log_dir / "checkout.log"
        )
        if checkout.returncode != 0:
            return {**base, "status": "checkout_failed", "source_commit": revision}
        lockfile = repository / "Cargo.lock"
        if not lockfile.is_file():
            return {**base, "status": "lockfile_missing", "source_commit": revision}
        metadata_argv = [
            str(self.cargo),
            "metadata",
            "--locked",
            "--no-deps",
            "--format-version",
            "1",
        ]
        metadata_result = _run_logged(
            metadata_argv, repository, log_dir / "metadata.log", dict(os.environ)
        )
        if metadata_result.returncode != 0:
            return {**base, "status": "metadata_failed", "source_commit": revision, "metadata_argv": metadata_argv}
        try:
            metadata = json.loads(metadata_result.stdout)
            target = _select_cargo_target(metadata, repository)
        except (json.JSONDecodeError, ValueError) as error:
            return {**base, "status": "metadata_invalid", "source_commit": revision, "error": str(error)}
        if target is None:
            return {**base, "status": "target_missing", "source_commit": revision}
        cargo_home = Path(tempfile.mkdtemp(prefix=f"cargo-home-{label}-", dir=self.result_root))
        cargo_target = Path(tempfile.mkdtemp(prefix=f"cargo-target-{label}-", dir=self.result_root))
        control_argv = [
            str(self.cargo),
            "check",
            "--locked",
            "--jobs",
            "1",
            "--package",
            target["package"],
        ]
        control_argv.extend(["--lib"] if target["kind"] == "lib" else ["--bin", target["name"]])
        environment = dict(os.environ)
        environment.update(
            {
                "CARGO_HOME": str(cargo_home),
                "CARGO_TARGET_DIR": str(cargo_target),
                "CARGO_BUILD_JOBS": "1",
            }
        )
        control = _run_logged(
            control_argv, repository, log_dir / "control.log", environment
        )
        record = {
            **base,
            "source_commit": revision,
            "source_tree": _git_output(repository, "rev-parse", "HEAD^{tree}"),
            "default_branch": default_branch,
            "cargo_lock_sha256": sha256_file(lockfile),
            "metadata_argv": metadata_argv,
            "control_argv": control_argv,
            "control_exit": control.returncode,
            "control_log": f"logs/{label}/control.log",
            "target": target,
            "build_scripts_or_proc_macros_may_execute": True,
        }
        if control.returncode != 0:
            return {**record, "status": "control_failed"}
        return {**record, "status": "admitted"}


def run_admission(
    manifest_path: Path,
    clone_root: Path,
    result_root: Path,
    cargo: Path,
    cutoff: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != SELECTION_SCHEMA:
        raise ValueError("selection manifest schema drifted")
    admissions_path = result_root / "admissions.jsonl"
    if admissions_path.exists():
        raise ValueError("admissions.jsonl already exists; admission is append-only")
    runner = RealAdmissionRunner(clone_root, result_root, cargo, cutoff)
    result = admit_candidates(manifest, runner)
    with admissions_path.open("x", encoding="utf-8") as output:
        for row in result["attempts"]:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = result_root / "admission_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "rapx-real-project-admission-summary-v1",
                "admitted_by_stratum": result["admitted_by_stratum"],
                "reserve_exhausted": any(
                    row["status"] == "reserve_exhausted" for row in result["attempts"]
                ),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _admitted_rows(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in _read_jsonl(path) if row.get("status") == "admitted"]
    if len(rows) != 10:
        raise ValueError(f"scan requires exactly ten admitted targets, found {len(rows)}")
    strata = {stratum: 0 for stratum in STRATA}
    repositories: set[str] = set()
    for row in rows:
        stratum = row.get("stratum")
        repository = normalize_repository(row.get("repository"))
        if stratum not in strata or repository in repositories:
            raise ValueError("admitted targets have invalid strata or duplicate repository")
        strata[stratum] += 1
        repositories.add(repository)
        target = row.get("target")
        if not isinstance(target, dict) or target.get("kind") not in {"lib", "bin"}:
            raise ValueError("admitted target record is invalid")
    if any(count != 5 for count in strata.values()):
        raise ValueError("scan requires five admitted targets per stratum")
    return rows


def _profile_map(contract: dict[str, Any], bin_dir: Path) -> dict[str, dict[str, Any]]:
    if {profile["name"] for profile in contract["profiles"]} != {
        "debug",
        "release",
        "rapx-perf",
    }:
        raise ValueError("profile contract names drifted")
    root = bin_dir.resolve(strict=True)
    profiles: dict[str, dict[str, Any]] = {}
    for profile in contract["profiles"]:
        executable = (root / profile["executable"]).resolve(strict=True)
        if root not in executable.parents or not executable.is_file():
            raise ValueError("profile executable escapes binary directory")
        if sha256_file(executable) != profile["sha256"]:
            raise ValueError(f"profile executable hash drifted: {profile['name']}")
        rapx = executable.with_name("rapx")
        if not rapx.is_file() or sha256_file(rapx) != profile["rapx_sha256"]:
            raise ValueError(f"profile rapx sibling is missing or drifted: {profile['name']}")
        profiles[profile["name"]] = {**profile, "path": executable}
    return profiles


def _validate_unit_receipt(
    value: dict[str, Any], admission: dict[str, Any]
) -> dict[str, Any]:
    if value.get("schema_version") != "rap-unit-v2" or value.get("success") is not True:
        raise ValueError("selected receipt is not a successful rap-unit-v2 receipt")
    for field in (
        "cargo_supplied_rustc_commit",
        "rap_compiler_commit",
        "rustc_commit",
    ):
        if value.get(field) != LEGACY_RUSTC_COMMIT:
            raise ValueError(f"selected receipt {field} drifted")
    target = admission["target"]
    if value.get("package_name") != target["package"]:
        raise ValueError("selected receipt package differs from admitted target")
    expected_kind = "lib" if target["kind"] == "lib" else "bin"
    if value.get("target_kind") != expected_kind:
        raise ValueError("selected receipt target kind differs from admitted target")
    findings = value.get("findings")
    counts = value.get("pattern_counts")
    if not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings):
        raise ValueError("selected receipt findings are invalid")
    if value.get("finding_count") != len(findings):
        raise ValueError("selected receipt finding_count is inconsistent")
    expected_patterns = {f"pattern{number}" for number in range(1, 7)}
    if not isinstance(counts, dict) or set(counts) != expected_patterns:
        raise ValueError("selected receipt pattern_counts are invalid")
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values()):
        raise ValueError("selected receipt pattern count is invalid")
    if sum(counts.values()) != len(findings):
        raise ValueError("selected receipt pattern counts do not sum to finding_count")
    for finding in findings:
        finding_id = finding.get("finding_id")
        if not isinstance(finding_id, str) or not HEX64.fullmatch(finding_id):
            raise ValueError("selected receipt finding ID is invalid")
        finding_key(finding)
        finding_root_key(finding)
    return value


def _select_project_receipt(
    receipt_dir: Path, admission: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(receipt_dir.glob("*.json")):
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError("RAP receipt is not an object")
        schema = value.get("schema_version")
        if schema == "rap-unit-v2" and value.get("package_name") == admission["target"]["package"]:
            matches.append((path, _validate_unit_receipt(value, admission)))
        elif schema not in {"rap-unit-v2", "rap-unit-skip-v1"}:
            raise ValueError(f"unrecognized RAP receipt schema: {path.name}")
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one admitted-package receipt, found {len(matches)}"
        )
    return matches[0]


def _scan_argv(profile: dict[str, Any], admission: dict[str, Any]) -> list[str]:
    target = admission["target"]
    argv = [
        str(profile["path"]),
        *profile["argv"],
        "--locked",
        "--jobs",
        "1",
        "--package",
        target["package"],
    ]
    argv.extend(["--lib"] if target["kind"] == "lib" else ["--bin", target["name"]])
    return argv


def _snippet_evidence(repository: Path, finding: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for role in ("source", "sink", "first_contract_failure"):
        item = finding.get(role)
        span = item.get("span") if isinstance(item, dict) else None
        if not isinstance(span, dict) or not isinstance(span.get("path"), str):
            raise ValueError(f"finding {role} lacks a stable span")
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise ValueError(f"finding {role} span lacks endpoints")
        start_line = start.get("line")
        end_line = end.get("line")
        if (
            not isinstance(start_line, int)
            or isinstance(start_line, bool)
            or not isinstance(end_line, int)
            or isinstance(end_line, bool)
            or start_line <= 0
            or end_line < start_line
        ):
            raise ValueError(f"finding {role} span has invalid lines")
        relative = PurePosixPath(span["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in span["path"]:
            raise ValueError(f"finding {role} span path is not repository-relative")
        source_path = (repository / relative.as_posix()).resolve(strict=True)
        if repository not in source_path.parents:
            raise ValueError(f"finding {role} span escapes repository")
        key = (relative.as_posix(), start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read finding source evidence: {error}") from error
        if end_line > len(lines):
            raise ValueError(f"finding {role} span exceeds source file")
        snippet = "".join(lines[start_line - 1 : end_line]).encode("utf-8")
        evidence.append(
            {
                "roles": [role],
                "path": relative.as_posix(),
                "start_line": start_line,
                "end_line": end_line,
                "file_sha256": sha256_file(source_path),
                "snippet_sha256": hashlib.sha256(snippet).hexdigest(),
            }
        )
    return evidence


class RealScanRunner:
    def __init__(
        self,
        result_root: Path,
        profiles: dict[str, dict[str, Any]],
        shared_cargo_home: Path,
    ) -> None:
        self.result_root = result_root.resolve()
        self.profiles = profiles
        self.shared_cargo_home = shared_cargo_home.resolve()
        self.shared_cargo_home.mkdir(parents=True, exist_ok=False)
        (self.result_root / "scan-logs").mkdir(parents=True, exist_ok=False)
        (self.result_root / "raw-receipts").mkdir(parents=True, exist_ok=False)
        (self.result_root / "scan-targets").mkdir(parents=True, exist_ok=False)

    def run(
        self, admission: dict[str, Any], profile_name: str, run_number: int
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        repository_name = normalize_repository(admission["repository"])
        label = f"{admission['stratum']}-{admission['repo_id']}"
        repository = Path(admission["clone_path"]).resolve(strict=True)
        if _git_output(repository, "rev-parse", "HEAD") != admission["source_commit"]:
            raise ValueError(f"admitted source commit drifted: {repository_name}")
        if _git_output(repository, "rev-parse", "HEAD^{tree}") != admission["source_tree"]:
            raise ValueError(f"admitted source tree drifted: {repository_name}")
        if _git_output(repository, "status", "--porcelain"):
            raise ValueError(f"admitted source worktree is dirty: {repository_name}")
        lockfile = repository / "Cargo.lock"
        if sha256_file(lockfile) != admission["cargo_lock_sha256"]:
            raise ValueError(f"admitted Cargo.lock drifted: {repository_name}")

        run_label = f"{label}-{profile_name}-{run_number}"
        receipt_dir = self.result_root / "raw-receipts" / run_label
        target_dir = self.result_root / "scan-targets" / run_label
        receipt_dir.mkdir()
        target_dir.mkdir()
        profile = self.profiles[profile_name]
        argv = _scan_argv(profile, admission)
        environment = dict(os.environ)
        environment.update(
            {
                "CARGO_HOME": str(self.shared_cargo_home),
                "CARGO_TARGET_DIR": str(target_dir),
                "CARGO_BUILD_JOBS": "1",
                "UNSOUND_SCANNER_RAP_JSON_DIR": str(receipt_dir),
                "UNSOUND_SCANNER_PROJECT_ROOT": str(repository),
            }
        )
        started = time.monotonic()
        result = _run_logged(
            argv,
            repository,
            self.result_root / "scan-logs" / f"{run_label}.log",
            environment,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        row: dict[str, Any] = {
            "schema": "rapx-real-project-scan-attempt-v1",
            "repository": repository_name,
            "repo_id": admission["repo_id"],
            "stratum": admission["stratum"],
            "source_commit": admission["source_commit"],
            "source_tree": admission["source_tree"],
            "target": admission["target"],
            "profile": profile_name,
            "run_number": run_number,
            "profile_executable_sha256": profile["sha256"],
            "profile_rapx_sha256": profile["rapx_sha256"],
            "argv": argv,
            "exit_code": result.returncode,
            "duration_ms": elapsed_ms,
            "log": f"scan-logs/{run_label}.log",
            "receipt_directory": f"raw-receipts/{run_label}",
        }
        if result.returncode != 0:
            row["status"] = "analyzer_failed"
            return row, None
        try:
            receipt_path, receipt = _select_project_receipt(receipt_dir, admission)
        except ValueError as error:
            row.update({"status": "receipt_failed", "error": str(error)})
            return row, None
        if _git_output(repository, "status", "--porcelain"):
            raise ValueError(f"scan mutated admitted source worktree: {repository_name}")
        row.update(
            {
                "status": "success",
                "receipt": f"raw-receipts/{run_label}/{receipt_path.name}",
                "receipt_sha256": sha256_file(receipt_path),
                "finding_count": receipt["finding_count"],
                "pattern_counts": receipt["pattern_counts"],
                "normalized_sha256": hashlib.sha256(
                    _canonical_bytes(normalize_receipt(receipt))
                ).hexdigest(),
            }
        )
        return row, receipt


def run_scans(
    admissions_path: Path,
    contract_path: Path,
    bin_dir: Path,
    result_root: Path,
) -> dict[str, Any]:
    _required_environment()
    admissions = _admitted_rows(admissions_path)
    profiles = _profile_map(load_profile_contract(contract_path), bin_dir)
    result_root = result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    scans_path = result_root / "scans.jsonl"
    findings_path = result_root / "findings.jsonl"
    summary_path = result_root / "scan_summary.json"
    if any(path.exists() for path in (scans_path, findings_path, summary_path)):
        raise ValueError("scan outputs already exist; scans are append-only")
    runner = RealScanRunner(result_root, profiles, result_root / "scan-cargo-home")
    scan_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    deterministic = 0
    profile_stable_targets = 0
    failed_targets = 0
    for admission in admissions:
        canonical_runs: list[dict[str, Any]] = []
        canonical_rows: list[dict[str, Any]] = []
        for run_number in (1, 2):
            row, receipt = runner.run(admission, "rapx-perf", run_number)
            scan_rows.append(row)
            canonical_rows.append(row)
            if receipt is not None:
                canonical_runs.append(receipt)
        if len(canonical_runs) != 2:
            failed_targets += 1
            continue
        left = _canonical_bytes(normalize_receipt(canonical_runs[0]))
        right = _canonical_bytes(normalize_receipt(canonical_runs[1]))
        if left != right:
            for row in canonical_rows:
                row["deterministic"] = False
            failed_targets += 1
            continue
        for row in canonical_rows:
            row["deterministic"] = True
        deterministic += 1
        if canonical_runs[0]["finding_count"] == 0:
            continue
        diagnostic: dict[str, dict[str, Any]] = {}
        for profile_name in ("debug", "release"):
            row, receipt = runner.run(admission, profile_name, 1)
            scan_rows.append(row)
            if receipt is not None:
                diagnostic[profile_name] = receipt
        if set(diagnostic) != {"debug", "release"}:
            failed_targets += 1
            continue
        stable = profile_stable_findings(
            canonical_runs[0], diagnostic["debug"], diagnostic["release"]
        )
        if not stable:
            continue
        profile_stable_targets += 1
        for finding in stable:
            finding_rows.append(
                {
                    "schema": "rapx-real-project-stable-finding-v1",
                    "repository": normalize_repository(admission["repository"]),
                    "repository_url": admission["repository_url"],
                    "repo_id": admission["repo_id"],
                    "stratum": admission["stratum"],
                    "source_commit": admission["source_commit"],
                    "source_tree": admission["source_tree"],
                    "target": admission["target"],
                    "finding_key": finding_key(finding),
                    "finding_root_key": finding_root_key(finding),
                    "root_cause_key": root_cause_key(admission["repository"], finding),
                    "canonical_receipt": canonical_rows[0]["receipt"],
                    "source_evidence": _snippet_evidence(
                        Path(admission["clone_path"]).resolve(strict=True), finding
                    ),
                    "finding": finding,
                }
            )
    with scans_path.open("x", encoding="utf-8") as output:
        for row in scan_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finding_rows.sort(key=lambda row: (row["repository"], row["root_cause_key"], row["finding_key"]))
    with findings_path.open("x", encoding="utf-8") as output:
        for row in finding_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "rapx-real-project-scan-summary-v1",
        "selected_targets": len(admissions),
        "deterministic_targets": deterministic,
        "profile_stable_nonzero_targets": profile_stable_targets,
        "failed_or_nondeterministic_targets": failed_targets,
        "profile_stable_finding_count": len(finding_rows),
        "unique_root_cause_count": len({row["root_cause_key"] for row in finding_rows}),
        "scans_sha256": sha256_file(scans_path),
        "findings_sha256": sha256_file(findings_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def verify_scans(result_root: Path) -> dict[str, Any]:
    result_root = result_root.resolve(strict=True)
    scans_path = result_root / "scans.jsonl"
    findings_path = result_root / "findings.jsonl"
    summary = _read_json(result_root / "scan_summary.json")
    if not isinstance(summary, dict) or summary.get("schema") != "rapx-real-project-scan-summary-v1":
        raise ValueError("scan summary schema drifted")
    if summary.get("scans_sha256") != sha256_file(scans_path):
        raise ValueError("scans.jsonl hash differs from scan summary")
    if summary.get("findings_sha256") != sha256_file(findings_path):
        raise ValueError("findings.jsonl hash differs from scan summary")
    scans = _read_jsonl(scans_path)
    findings = _read_jsonl(findings_path)
    if summary.get("profile_stable_finding_count") != len(findings):
        raise ValueError("scan summary stable finding count drifted")
    roots: set[str] = set()
    for row in findings:
        finding = row.get("finding")
        if not isinstance(finding, dict):
            raise ValueError("stable finding row lacks finding object")
        if row.get("finding_key") != finding_key(finding):
            raise ValueError("stable finding key drifted")
        if row.get("finding_root_key") != finding_root_key(finding):
            raise ValueError("stable finding root key drifted")
        if row.get("root_cause_key") != root_cause_key(row.get("repository"), finding):
            raise ValueError("stable finding root-cause key drifted")
        receipt = result_root / row.get("canonical_receipt", "")
        if not receipt.is_file():
            raise ValueError("stable finding canonical receipt is missing")
        roots.add(row["root_cause_key"])
    if summary.get("unique_root_cause_count") != len(roots):
        raise ValueError("scan summary unique root count drifted")
    if not scans:
        raise ValueError("scan ledger is empty")
    return summary


def _review_template(group: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(group, key=lambda row: (row["finding_key"], row["finding"]["finding_id"]))
    representative = ordered[0]
    finding = representative["finding"]
    return {
        "schema": "rapx-real-project-review-row-v1",
        "root_key": representative["root_cause_key"],
        "repository": representative["repository"],
        "repository_url": representative["repository_url"],
        "source_commit": representative["source_commit"],
        "source_tree": representative["source_tree"],
        "target": representative["target"],
        "rule_id": finding.get("rule_id"),
        "primary_pattern": finding.get("primary_pattern"),
        "source": finding.get("source"),
        "sink": finding.get("sink"),
        "first_contract_failure": finding.get("first_contract_failure"),
        "obligation": finding.get("obligation"),
        "validation_status": finding.get("validation_status"),
        "propagation": finding.get("propagation"),
        "finding_ids": sorted(row["finding"]["finding_id"] for row in ordered),
        "source_evidence": representative["source_evidence"],
        "raw_receipt": representative["canonical_receipt"],
        "disposition": "unresolved",
        "error_family": None,
        "duplicate_of": None,
        "evidence_zh": "待依据固定源码、调用路径和契约失败点完成逐条审计。",
        "rationale_zh": "当前仅确认其为跨构建稳定的静态候选，尚未作真实性结论。",
    }


def export_review(result_root: Path) -> dict[str, Any]:
    scan_summary = verify_scans(result_root)
    result_root = result_root.resolve(strict=True)
    ledger_path = result_root / "review_ledger.jsonl"
    human_path = result_root / "human_review.md"
    summary_path = result_root / "summary.md"
    if any(path.exists() for path in (ledger_path, human_path, summary_path)):
        raise ValueError("review outputs already exist; review export is append-only")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(result_root / "findings.jsonl"):
        root = row.get("root_cause_key")
        if not isinstance(root, str) or not HEX64.fullmatch(root):
            raise ValueError("stable finding lacks a valid root-cause key")
        groups.setdefault(root, []).append(row)
    review_rows = [_review_template(groups[root]) for root in sorted(groups)]
    with ledger_path.open("x", encoding="utf-8") as output:
        for row in review_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    validate_review_ledger(ledger_path)

    markdown = [
        "# RAPx 真实项目候选质量 pilot：人工审计工作表",
        "",
        "本表中的记录均为静态候选，不代表已确认缺陷。审计必须绑定固定 revision、同一 source/sink 与完整调用见证。",
        "",
    ]
    for index, row in enumerate(review_rows, 1):
        finding_ids = ", ".join(f"`{value}`" for value in row["finding_ids"])
        markdown.extend(
            [
                f"## {index}. `{row['repository']}` / `{row['rule_id']}`",
                "",
                f"- 根因键：`{row['root_key']}`",
                f"- 仓库：{row['repository_url']} @ `{row['source_commit']}`",
                f"- 目标：`{row['target']['package']}` / `{row['target']['kind']}` / `{row['target']['name']}`",
                f"- source：`{json.dumps(row['source'], ensure_ascii=False, sort_keys=True)}`",
                f"- sink：`{json.dumps(row['sink'], ensure_ascii=False, sort_keys=True)}`",
                f"- first failure：`{json.dumps(row['first_contract_failure'], ensure_ascii=False, sort_keys=True)}`",
                f"- obligation：`{json.dumps(row['obligation'], ensure_ascii=False, sort_keys=True)}`",
                f"- finding IDs：{finding_ids}",
                f"- 原始 receipt：`{row['raw_receipt']}`",
                "- 审计结论：`unresolved`",
                "- 证据：待审计",
                "- 理由：待审计",
                "",
            ]
        )
    human_path.write_text("\n".join(markdown), encoding="utf-8")

    admissions_path = result_root / "admissions.jsonl"
    admissions = _read_jsonl(admissions_path) if admissions_path.is_file() else []
    admitted = sum(row.get("status") == "admitted" for row in admissions)
    control_failures = sum(row.get("status") == "control_failed" for row in admissions)
    disposition_counts = {name: 0 for name in sorted(REVIEW_DISPOSITIONS)}
    for row in review_rows:
        disposition_counts[row["disposition"]] += 1
    summary = [
        "# RAPx 真实项目候选质量 pilot 摘要",
        "",
        "本结果只描述静态候选生成与人工审计状态，不构成真实缺陷率或生态覆盖结论。",
        "",
        f"- 预定目标：{scan_summary['selected_targets']}",
        f"- 已准入目标：{admitted}",
        f"- control failure：{control_failures}",
        f"- analyzer/nondeterminism failure：{scan_summary['failed_or_nondeterministic_targets']}",
        f"- canonical 双跑确定性目标：{scan_summary['deterministic_targets']}",
        f"- 三 profile 稳定非零目标：{scan_summary['profile_stable_nonzero_targets']}",
        f"- 三 profile 稳定 findings：{scan_summary['profile_stable_finding_count']}",
        f"- 唯一 source–sink 根因：{scan_summary['unique_root_cause_count']}",
        "",
        "## 审计 disposition",
        "",
    ]
    summary.extend(f"- {name}: {count}" for name, count in disposition_counts.items())
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {
        "review_rows": len(review_rows),
        "review_ledger_sha256": sha256_file(ledger_path),
        "human_review_sha256": sha256_file(human_path),
        "summary_sha256": sha256_file(summary_path),
    }


def _replace_runtime_root(value: Any, runtime_root: Path) -> Any:
    marker = str(runtime_root)
    if isinstance(value, dict):
        return {
            key: _replace_runtime_root(item, runtime_root)
            for key, item in value.items()
            if key not in {"clone_path", "duration_ms"}
        }
    if isinstance(value, list):
        return [_replace_runtime_root(item, runtime_root) for item in value]
    if isinstance(value, str):
        return value.replace(marker, "<runtime-root>")
    return value


def _sanitized_admission(row: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    value = _replace_runtime_root(row, runtime_root)
    label = f"{row.get('stratum')}-{row.get('repo_id')}"
    clone_argv = value.get("clone_argv")
    if isinstance(clone_argv, list) and clone_argv:
        clone_argv[-1] = f"clones/{label}"
    for key in ("metadata_argv", "control_argv"):
        argv = value.get(key)
        if isinstance(argv, list) and argv:
            argv[0] = "cargo"
    return value


def _sanitized_scan(row: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    value = _replace_runtime_root(row, runtime_root)
    argv = value.get("argv")
    profile = value.get("profile")
    if isinstance(argv, list) and argv and isinstance(profile, str):
        argv[0] = f"{profile}/cargo-rapx"
    return value


def _reject_private_or_secret_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/private/tmp/",
        "/tmp/",
        "ghp_",
        "github_pat_",
        "Authorization:",
        "-----BEGIN PRIVATE KEY-----",
    )
    matches = [token for token in forbidden if token in text]
    if matches:
        raise ValueError(f"sanitized evidence contains private path or secret marker: {matches}")


def freeze_evidence(
    result_root: Path,
    output_dir: Path,
    receipt_manifest: Path,
    selection_manifest: Path,
    profile_contract: Path,
) -> dict[str, Any]:
    verify_scans(result_root)
    validate_review_ledger(result_root / "review_ledger.jsonl")
    result_root = result_root.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    files: dict[str, str] = {}

    sanitized_admissions = output_dir / "admissions.jsonl"
    with sanitized_admissions.open("x", encoding="utf-8") as output:
        for row in _read_jsonl(result_root / "admissions.jsonl"):
            output.write(
                json.dumps(
                    _sanitized_admission(row, result_root),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    sanitized_scans = output_dir / "scans.jsonl"
    with sanitized_scans.open("x", encoding="utf-8") as output:
        for row in _read_jsonl(result_root / "scans.jsonl"):
            output.write(
                json.dumps(
                    _sanitized_scan(row, result_root),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    for name in (
        "findings.jsonl",
        "review_ledger.jsonl",
        "human_review.md",
        "summary.md",
    ):
        source = result_root / name
        target = output_dir / name
        target.write_bytes(source.read_bytes())
    raw_summary = _read_json(result_root / "scan_summary.json")
    if not isinstance(raw_summary, dict):
        raise ValueError("scan summary is not an object")
    raw_summary["scans_sha256"] = sha256_file(sanitized_scans)
    raw_summary["findings_sha256"] = sha256_file(output_dir / "findings.jsonl")
    sanitized_summary = output_dir / "scan_summary.json"
    sanitized_summary.write_text(
        json.dumps(raw_summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            _reject_private_or_secret_text(path)
            files[path.name] = sha256_file(path)
    manifest = {
        "schema": "rapx-real-project-pilot-receipt-manifest-v1",
        "rap_source_commit": LEGACY_SOURCE_COMMIT,
        "rustc_commit": LEGACY_RUSTC_COMMIT,
        "cargo_commit": LEGACY_CARGO_COMMIT,
        "selection_manifest_sha256": sha256_file(selection_manifest),
        "profile_contract_sha256": sha256_file(profile_contract),
        "evidence_files": files,
        "raw_logs_committed": False,
        "binaries_or_caches_committed": False,
    }
    receipt_manifest.parent.mkdir(parents=True, exist_ok=True)
    receipt_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reject_private_or_secret_text(receipt_manifest)
    return manifest


def _command_commit(executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "-Vv"], check=True, text=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect exact toolchain executable {executable}: {error}") from error
    for line in result.stdout.splitlines():
        if line.startswith("commit-hash: "):
            return line.removeprefix("commit-hash: ")
    raise ValueError(f"toolchain executable did not report commit-hash: {executable}")


def _required_environment() -> tuple[Path, Path, Path]:
    required = (
        "UNSOUND_SCANNER_EXACT_CARGO_PATH",
        "UNSOUND_SCANNER_EXACT_CARGO_COMMIT",
        "RUSTC",
        "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH",
        "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT",
        "RUST_SYSROOT",
        "Z3_SYS_Z3_HEADER",
        "RUSTFLAGS",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"missing exact legacy environment: {', '.join(missing)}")
    if os.environ.get("CARGO_ENCODED_RUSTFLAGS"):
        raise ValueError("CARGO_ENCODED_RUSTFLAGS must be unset")
    cargo = Path(os.environ["UNSOUND_SCANNER_EXACT_CARGO_PATH"]).resolve()
    rustc = Path(os.environ["RUSTC"]).resolve()
    expected_rustc = Path(os.environ["UNSOUND_SCANNER_EXPECTED_RUSTC_PATH"]).resolve()
    if rustc != expected_rustc:
        raise ValueError("RUSTC path differs from expected rustc path")
    if os.environ["UNSOUND_SCANNER_EXACT_CARGO_COMMIT"] != LEGACY_CARGO_COMMIT:
        raise ValueError("exact Cargo environment commit drifted")
    if os.environ["UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT"] != LEGACY_RUSTC_COMMIT:
        raise ValueError("expected rustc environment commit drifted")
    if _command_commit(cargo) != LEGACY_CARGO_COMMIT:
        raise ValueError("actual Cargo commit drifted")
    if _command_commit(rustc) != LEGACY_RUSTC_COMMIT:
        raise ValueError("actual rustc commit drifted")
    sysroot = Path(os.environ["RUST_SYSROOT"]).resolve()
    if not sysroot.is_dir() or not Path(os.environ["Z3_SYS_Z3_HEADER"]).is_file():
        raise ValueError("legacy sysroot or Z3 header is missing")
    return cargo, rustc, sysroot


def verify_profiles(contract_path: Path, bin_dir: Path, output: Path) -> str:
    status = "blocked_legacy_preflight"
    try:
        contract = load_profile_contract(contract_path)
        _required_environment()
        root = bin_dir.resolve(strict=True)
        for profile in contract["profiles"]:
            executable = (root / profile["executable"]).resolve(strict=True)
            if not executable.is_file() or root not in executable.parents:
                raise ValueError("profile executable escapes binary directory")
            if sha256_file(executable) != profile["sha256"]:
                raise ValueError(f"profile executable hash drifted: {profile['name']}")
            sibling = executable.with_name("rapx")
            if not sibling.is_file() or sha256_file(sibling) != profile["rapx_sha256"]:
                raise ValueError(f"profile rapx sibling is missing or drifted: {profile['name']}")
        status = "ok"
    except (OSError, ValueError):
        status = "blocked_legacy_preflight"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(status + "\n", encoding="utf-8")
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-profiles")
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--bin-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    select = commands.add_parser("select")
    select.add_argument("--roster", type=Path, required=True)
    select.add_argument("--instances", type=Path, required=True)
    select.add_argument("--development-exclusions", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--reserve-per-stratum", type=int, default=25)
    admit = commands.add_parser("admit")
    admit.add_argument("--manifest", type=Path, required=True)
    admit.add_argument("--clone-root", type=Path, required=True)
    admit.add_argument("--result-root", type=Path, required=True)
    admit.add_argument("--cargo", type=Path, required=True)
    admit.add_argument("--cutoff", default="2024-10-12T00:00:00Z")
    scan = commands.add_parser("scan")
    scan.add_argument("--admissions", type=Path, required=True)
    scan.add_argument("--profile-contract", type=Path, required=True)
    scan.add_argument("--bin-dir", type=Path, required=True)
    scan.add_argument("--result-root", type=Path, required=True)
    verify_scans_parser = commands.add_parser("verify-scans")
    verify_scans_parser.add_argument("--result-root", type=Path, required=True)
    export = commands.add_parser("export-review")
    export.add_argument("--result-root", type=Path, required=True)
    freeze = commands.add_parser("freeze-evidence")
    freeze.add_argument("--result-root", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--receipt-manifest", type=Path, required=True)
    freeze.add_argument("--selection-manifest", type=Path, required=True)
    freeze.add_argument("--profile-contract", type=Path, required=True)
    review = commands.add_parser("validate-review")
    review.add_argument("--ledger", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-profiles":
        return 0 if verify_profiles(args.contract, args.bin_dir, args.output) == "ok" else 1
    if args.command == "select":
        select_from_inputs(
            args.roster,
            args.instances,
            args.development_exclusions,
            args.output,
            args.reserve_per_stratum,
        )
        return 0
    if args.command == "admit":
        result = run_admission(
            args.manifest, args.clone_root, args.result_root, args.cargo, args.cutoff
        )
        return 0 if all(result["admitted_by_stratum"][key] == 5 for key in STRATA) else 1
    if args.command == "scan":
        run_scans(
            args.admissions,
            args.profile_contract,
            args.bin_dir,
            args.result_root,
        )
        return 0
    if args.command == "verify-scans":
        verify_scans(args.result_root)
        return 0
    if args.command == "export-review":
        export_review(args.result_root)
        return 0
    if args.command == "freeze-evidence":
        freeze_evidence(
            args.result_root,
            args.output_dir,
            args.receipt_manifest,
            args.selection_manifest,
            args.profile_contract,
        )
        return 0
    if args.command == "validate-review":
        validate_review_ledger(args.ledger)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
