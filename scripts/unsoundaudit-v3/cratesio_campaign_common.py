#!/usr/bin/env python3
"""Common closed contracts, canonical JSON, hashing, and loading for crates.io campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


POLICY_KEYS = {
    "schema_version",
    "campaign_id",
    "selection_date",
    "toolchain",
    "candidate_pool",
    "wall_time_seconds",
    "cargo_jobs",
    "features",
    "target_kind",
    "suppressed_quota_per_crate",
    "unestablished_quota_per_crate",
    "review_seed",
}

TERMINAL_STATES = {
    "control_failed",
    "scan_timeout",
    "scan_failed",
    "partial_coverage",
    "complete_exact",
}

ENVIRONMENT_EXACT_REMOVALS = {
    "CARGO_BUILD_JOBS",
    "CARGO_BUILD_TARGET",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_HOME",
    "CARGO_INCREMENTAL",
    "CARGO_TARGET_DIR",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "MACOSX_DEPLOYMENT_TARGET",
    "RAP_UNSOUNDAUDIT_OUTPUT",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
    "RUSTUP_TOOLCHAIN",
}


class CampaignError(ValueError):
    """Raised when campaign execution or policy violates closed invariants."""


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_keys(data: dict[str, Any], expected: set[str] | list[str], label: str = "object") -> None:
    expected_set = set(expected)
    if not isinstance(data, dict) or set(data) != expected_set:
        raise CampaignError(f"{label} keys drifted")


def exact_hex(value: str, expected_len: int = 64, label: str = "value") -> None:
    if not isinstance(value, str) or len(value) != expected_len or not all(c in "0123456789abcdef" for c in value):
        raise CampaignError(f"{label} is not a valid lowercase hex string of length {expected_len}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_policy(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != POLICY_KEYS:
        raise CampaignError("campaign policy keys drifted")
    if value["schema_version"] != "rap-unsoundaudit-cratesio-campaign-policy-v1":
        raise CampaignError("campaign policy schema drifted")
    if len(value["candidate_pool"]) != 10:
        raise CampaignError("campaign requires exactly ten domains")
    if value["wall_time_seconds"] != 900 or value["cargo_jobs"] != 1:
        raise CampaignError("campaign execution limits drifted")
    if not re.fullmatch(r"[0-9a-f]{64}", value["review_seed"]):
        raise CampaignError("campaign review seed is invalid")
    return value


def stable_rank(seed: str, identity: str) -> str:
    return hashlib.sha256((seed + "\0" + identity).encode()).hexdigest()


def tool_commit(executable: Path | str, environment: dict[str, str] | None = None, label: str = "tool") -> str:
    env = environment if environment is not None else os.environ.copy()
    try:
        proc = subprocess.run(
            [str(executable), "-Vv"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as e:
        raise CampaignError(f"failed to run {label} ({executable}): {e}") from e
    if proc.returncode != 0:
        raise CampaignError(f"failed to inspect {label}: {proc.stderr.strip()}")
    for line in proc.stdout.splitlines():
        if line.startswith("commit-hash: "):
            val = line.removeprefix("commit-hash: ").strip()
            if len(val) == 40 and all(c in "0123456789abcdef" for c in val):
                return val
    raise CampaignError(f"{label} did not report a full commit hash")


def sanitized_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    inherited = (base if base is not None else os.environ).copy()
    for variable in tuple(inherited):
        if (
            variable in ENVIRONMENT_EXACT_REMOVALS
            or variable.startswith("CARGO_BUILD_")
            or variable.startswith("CARGO_PROFILE_")
            or variable.startswith("CARGO_TARGET_")
            or variable.startswith("DYLD_")
            or variable.startswith("RAP_")
            or variable.startswith("UNSOUND_SCANNER_")
        ):
            inherited.pop(variable, None)
    return inherited
