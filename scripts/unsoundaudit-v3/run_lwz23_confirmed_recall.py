#!/usr/bin/env python3
"""Scan frozen lwz23 maintainer-confirmed bugs with the current RAP scanner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cratesio_campaign_common import sanitized_environment, tool_commit  # noqa: E402
from run_stdlib_gate import run_limited  # noqa: E402


RUSTC_COMMIT = "ba28ff76f353a722f31c4f3dd2ac4e437d36411b"
CARGO_COMMIT = "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0"
WALL_TIME_SECONDS = 900

# Frozen before any analyzer rebuild. Vulnerable revision is the parent of the
# first commit that changed the cited public API.
CASES: list[dict[str, Any]] = [
    {
        "id": "asm-lsp-187",
        "issue": "https://github.com/bergercookie/asm-lsp/issues/187",
        "repo": "https://github.com/bergercookie/asm-lsp.git",
        "rev": "4b8bc91aedf7498980aea4667a5c70c5d9f81f83",
        "package": "asm-lsp",
        "manifest": "asm-lsp/Cargo.toml",
        "group": "A-existing",
        "expected_pattern": "P1",
        "expected_roots": ["get_str", "get_string"],
        "expected_callees": ["from_utf8_unchecked"],
    },
    {
        "id": "amdgpu_top-104",
        "issue": "https://github.com/Umio-Yasuno/amdgpu_top/issues/104",
        "repo": "https://github.com/Umio-Yasuno/amdgpu_top.git",
        "rev": "3c9673895194883c3f5d47dafbb43491d8d64b2a",
        "package": "libamdgpu_top",
        "manifest": "crates/libamdgpu_top/Cargo.toml",
        "group": "A-existing",
        "expected_pattern": "P1",
        "expected_roots": ["PCAcc::get", "get"],
        "expected_callees": ["get_unchecked"],
    },
    {
        "id": "pleco-163",
        "issue": "https://github.com/pleco-rs/Pleco/issues/163",
        "repo": "https://github.com/pleco-rs/Pleco.git",
        "rev": "83f8953452b87e812366f0c5881826ff20831896",
        "package": "pleco_engine",
        "manifest": "pleco_engine/Cargo.toml",
        "group": "A-existing",
        "expected_pattern": "P1",
        "expected_roots": ["insert_score", "insert_score_depth"],
        "expected_callees": ["get_unchecked_mut"],
    },
    {
        "id": "revm-1902",
        "issue": "https://github.com/bluealloy/revm/issues/1902",
        "repo": "https://github.com/bluealloy/revm.git",
        "rev": "64f38434101be5f131b621124c122b5676b92b28",
        "package": "revm-interpreter",
        "manifest": "crates/interpreter/Cargo.toml",
        "group": "A-existing",
        "expected_pattern": "P2",
        "expected_roots": ["opcode", "ExtBytecode"],
        "expected_callees": [],
        "expected_site_kind": "raw_pointer_dereference",
    },
    {
        "id": "blitz-169",
        "issue": "https://github.com/DioxusLabs/blitz/issues/169",
        "repo": "https://github.com/DioxusLabs/blitz.git",
        "rev": "be8894f0b481757f856434f43722911f290bb597",
        "package": "blitz-dom",
        "manifest": "packages/blitz-dom/Cargo.toml",
        "group": "A-existing",
        "expected_pattern": "P2",
        "expected_roots": ["tree", "Node"],
        "expected_callees": [],
        "expected_site_kind": "raw_pointer_dereference",
    },
    {
        "id": "kcl-1776",
        "issue": "https://github.com/kcl-lang/kcl/issues/1776",
        "repo": "https://github.com/kcl-lang/kcl.git",
        "rev": "bc74f86d4af33d0e3591c7dcda748535ab436469",
        "package": None,
        "manifest": "kclvm/runtime/Cargo.toml",
        "group": "A-existing",
        "expected_pattern": "P1",
        "expected_roots": ["c2str_vec"],
        "expected_callees": ["from_ptr"],
        "expected_site_kind": "raw_pointer_dereference",
    },
    {
        "id": "fugue-core-18",
        "issue": "https://github.com/fugue-re/fugue-core/issues/18",
        "repo": "https://github.com/fugue-re/fugue-core.git",
        "rev": "f4e1cd5fde7b82bb176bd37de5ae0ed8e61b2de7",
        "package": "fugue-ir",
        "manifest": "fugue-ir/Cargo.toml",
        "group": "A-new-contract",
        "expected_pattern": "P1",
        "expected_roots": ["unchecked_symbol"],
        "expected_callees": ["get_unchecked"],
        "skip_if_already_unsafe": True,
    },
    {
        "id": "ckb-vm-444",
        "issue": "https://github.com/nervosnetwork/ckb-vm/issues/444",
        "repo": "https://github.com/nervosnetwork/ckb-vm.git",
        "rev": "c9351f44c4e664e974eaef88b4e8bb1c142c8478",
        "package": None,
        "manifest": "Cargo.toml",
        "group": "A-new-contract",
        "expected_pattern": "P1",
        "expected_roots": ["cast_ptr_to_slice"],
        "expected_callees": ["from_raw_parts"],
    },
    {
        "id": "libsql-1859",
        "issue": "https://github.com/tursodatabase/libsql/issues/1859",
        "repo": "https://github.com/tursodatabase/libsql.git",
        "rev": "8cd7c7e82e7c3ce349e6e204ecb6d19da4972685^",
        "package": None,
        "manifest": "vendored/sqlite3-parser/Cargo.toml",
        "group": "A-new-contract",
        "expected_pattern": "P1",
        "expected_roots": ["keyword_token"],
        "expected_callees": ["from_utf8_unchecked"],
    },
    {
        "id": "may-112",
        "issue": "https://github.com/Xudong-Huang/may/issues/112",
        "repo": "https://github.com/Xudong-Huang/may.git",
        "rev": "2dae05fbcaaa009c2cbb70478d67e1d3337bc819",
        "package": "may",
        "manifest": "Cargo.toml",
        "group": "B",
        "expected_pattern": "P1",
        "expected_roots": ["wakeup"],
        "expected_callees": ["get_unchecked"],
        "visibility_may_be_private": True,
    },
    {
        "id": "aya-1112",
        "issue": "https://github.com/aya-rs/aya/issues/1112",
        "repo": "https://github.com/aya-rs/aya.git",
        "rev": "5a43bedc0180ba41854e6c23f0476c2cbeb1e2bd",
        "package": "aya-ebpf",
        "manifest": "ebpf/aya-ebpf/Cargo.toml",
        "group": "B",
        "expected_pattern": "P2",
        "expected_roots": ["SkBuff", "len"],
        "expected_callees": [],
        "expected_site_kind": "raw_pointer_dereference",
        "special_target": True,
    },
    {
        "id": "retrowin32-72",
        "issue": "https://github.com/evmar/retrowin32/issues/72",
        "repo": "https://github.com/evmar/retrowin32.git",
        "rev": "e6c045f28bd956745ae4ee12dba616b0cc68a248",
        "package": None,
        "manifest": "memory/Cargo.toml",
        "group": "B",
        "expected_pattern": "P1",
        "expected_roots": ["copy"],
        "expected_callees": ["copy"],
        "special_target": True,
    },
    {
        "id": "eduos-46",
        "issue": "https://github.com/RWTH-OS/eduOS-rs/issues/46",
        "repo": "https://github.com/RWTH-OS/eduOS-rs.git",
        "rev": "55b540664b19fdb2240beed265c21e3e9d04828d",
        "package": None,
        "manifest": "Cargo.toml",
        "group": "B",
        "expected_pattern": "P1",
        "expected_roots": ["RomHandle::new", "new"],
        "expected_callees": ["from_raw_parts"],
        "special_target": True,
    },
    {
        "id": "pua-lang-76",
        "issue": "https://github.com/flaneur2020/pua-lang/issues/76",
        "repo": "https://github.com/flaneur2020/pua-lang.git",
        "rev": "cbaf8e2ca1abc3797c332afcc9103c567ee16002",
        "package": None,
        "manifest": "Cargo.toml",
        "bin_name": "pua-lang",
        "group": "B",
        "expected_pattern": "P1",
        "expected_roots": ["dealloc"],
        "expected_callees": ["from_raw_parts"],
        "not_lib": True,
    },
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_mvp(mvp: Path) -> dict[str, Any]:
    candidates = load_jsonl(mvp / "candidates.jsonl")
    families = load_jsonl(mvp / "families.jsonl")
    unestablished = load_jsonl(mvp / "unestablished.jsonl")
    ledger = load_jsonl(mvp / "suppression-ledger.jsonl")
    coverage = {}
    coverage_path = mvp / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
    active = []
    for row in candidates:
        active.append(
            {
                "pattern": row.get("primary_pattern"),
                "root": ((row.get("public_safe_root") or {}).get("owner")),
                "callee": ((row.get("unsafe_site") or {}).get("callee")),
                "kind": ((row.get("unsafe_site") or {}).get("kind")),
                "span": ((row.get("unsafe_site") or {}).get("span")),
                "model": ((row.get("contract") or {}).get("model_id")),
            }
        )
    unest = []
    for row in unestablished:
        unest.append(
            {
                "pattern": row.get("attempted_pattern"),
                "root": row.get("public_root"),
                "reason": row.get("primary_reason"),
                "model": row.get("requirement"),
            }
        )
    return {
        "n_candidates": len(candidates),
        "n_families": len(families),
        "n_unestablished": len(unestablished),
        "n_suppressed": len(ledger),
        "accounting": coverage.get("candidate_accounting"),
        "complete": coverage.get("complete"),
        "semantics": coverage.get("semantics"),
        "active": active,
        "unestablished": unest[:40],
    }


def matches_expected(summary: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    roots = [str(r).lower() for r in case.get("expected_roots") or []]
    callees = [str(c).lower() for c in case.get("expected_callees") or []]
    pattern = case.get("expected_pattern")
    hits = []
    for row in summary.get("active") or []:
        root = str(row.get("root") or "").lower()
        callee = str(row.get("callee") or "").lower()
        kind = str(row.get("kind") or "").lower()
        root_ok = not roots or any(token in root for token in roots)
        callee_ok = not callees or any(token in callee for token in callees)
        pattern_ok = not pattern or row.get("pattern") == pattern
        kind_ok = not case.get("expected_site_kind") or kind == case.get("expected_site_kind")
        if root_ok and callee_ok and pattern_ok and kind_ok:
            hits.append(row)
    return {"hit": bool(hits), "hits": hits}


def ensure_repo(case: dict[str, Any], src_root: Path) -> Path:
    dest = src_root / case["id"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        if dest.exists():
            shutil.rmtree(dest)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-tags", case["repo"], str(dest)],
            check=True,
        )
    fetch = subprocess.run(
        ["git", "-C", str(dest), "fetch", "--filter=blob:none", "origin", case["rev"]],
        check=False,
        capture_output=True,
        text=True,
    )
    checkout = subprocess.run(
        ["git", "-C", str(dest), "checkout", "--force", case["rev"]],
        check=False,
        capture_output=True,
        text=True,
    )
    if checkout.returncode != 0:
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--filter=blob:none", "origin", case["rev"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", "--force", "FETCH_HEAD"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", "--force", case["rev"]],
            check=False,
        )
    return dest


def cargo_args(case: dict[str, Any], locked: bool) -> list[str]:
    args = []
    if case.get("package"):
        args.extend(["-p", case["package"]])
    if case.get("not_lib"):
        if case.get("bin_name"):
            args.extend(["--bin", case["bin_name"]])
    else:
        args.append("--lib")
    if locked:
        args.append("--locked")
    args.extend(["--jobs", "1"])
    return args


def maybe_generate_lock(src: Path, cargo: Path, rustc: Path, env: dict[str, str], log: Path) -> bool:
    if (src / "Cargo.lock").exists():
        return True
    with log.open("wb") as stream:
        result = subprocess.run(
            [str(cargo), "generate-lockfile"],
            cwd=src,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode == 0


def scan_case(
    case: dict[str, Any],
    *,
    cargo: Path,
    rustc: Path,
    cargo_rapx: Path,
    rust_sysroot: Path,
    cargo_commit: str,
    rustc_commit: str,
    work_root: Path,
    cargo_home: Path,
    groups: set[str],
) -> dict[str, Any]:
    if case["group"] not in groups and ("A" not in groups or not case["group"].startswith("A")):
        return {"id": case["id"], "status": "skipped_group"}
    case_dir = work_root / case["id"]
    logs = case_dir / "logs"
    mvp = case_dir / "mvp"
    receipts = case_dir / "receipts"
    control_target = case_dir / "control-target"
    scan_target = case_dir / "scan-target"
    for path in (logs, mvp, receipts):
        if path.exists() and path == mvp:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in (control_target, scan_target):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "id": case["id"],
        "issue": case["issue"],
        "rev": case["rev"],
        "group": case["group"],
        "package": case.get("package"),
        "expected_pattern": case.get("expected_pattern"),
        "expected_roots": case.get("expected_roots"),
    }
    try:
        src = ensure_repo(case, work_root / "src")
    except Exception as error:
        record["status"] = "clone_failed"
        record["error"] = str(error)
        return record
    record["src"] = str(src)
    manifest = src / case["manifest"]
    if not manifest.exists():
        record["status"] = "manifest_missing"
        record["manifest"] = str(manifest)
        return record
    pkg_cwd = manifest.parent

    base = sanitized_environment()
    base.update(
        {
            "RUSTC": str(rustc),
            "RUST_SYSROOT": str(rust_sysroot),
            "CARGO_HOME": str(cargo_home),
            "CARGO_BUILD_JOBS": "1",
            "RAP_CLEAN": "false",
            "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(cargo),
            "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": cargo_commit,
            "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(rustc),
            "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": rustc_commit,
        }
    )
    if sys.platform == "darwin":
        base["DYLD_LIBRARY_PATH"] = str(rust_sysroot / "lib")

    locked = maybe_generate_lock(pkg_cwd, cargo, rustc, base, logs / "lock.log")
    control_env = base.copy()
    control_env["CARGO_TARGET_DIR"] = str(control_target)
    control_cmd = [str(cargo), "check", *cargo_args(case, locked)]
    control = run_limited(
        control_cmd,
        cwd=pkg_cwd,
        environment=control_env,
        wall_time_seconds=WALL_TIME_SECONDS,
        physical_footprint_limit_bytes=None,
        log_path=logs / "control.log",
    )
    record["control"] = {
        "returncode": control.returncode,
        "elapsed_seconds": control.elapsed_seconds,
        "classification": control.classification,
    }
    if control.classification == "timeout" or control.returncode != 0:
        record["status"] = "control_failed"
        return record

    scan_env = base.copy()
    scan_env.update(
        {
            "CARGO_TARGET_DIR": str(scan_target),
            "UNSOUND_SCANNER_PROJECT_ROOT": str(pkg_cwd),
            "RAP_UNSOUNDAUDIT_OUTPUT": str(mvp),
            "UNSOUND_SCANNER_RAP_JSON_DIR": str(receipts),
            "UNSOUND_SCANNER_PERF_TRACE": str(logs / "scan_perf_trace.jsonl"),
        }
    )
    scan_cmd = [str(cargo_rapx), "rapx", "-unsoundaudit", "--", *cargo_args(case, locked)]
    scan = run_limited(
        scan_cmd,
        cwd=pkg_cwd,
        environment=scan_env,
        wall_time_seconds=WALL_TIME_SECONDS,
        physical_footprint_limit_bytes=None,
        log_path=logs / "scan.log",
    )
    record["scan"] = {
        "returncode": scan.returncode,
        "elapsed_seconds": scan.elapsed_seconds,
        "classification": scan.classification,
        "max_physical_footprint_bytes": scan.max_physical_footprint_bytes,
        "max_rss_bytes": scan.max_rss_bytes,
    }
    if scan.classification == "timeout":
        record["status"] = "scan_timeout"
        return record
    summary = summarize_mvp(mvp)
    if scan.returncode != 0 and not (
        summary.get("complete") and (mvp / "candidates.jsonl").exists()
    ):
        record["status"] = "scan_failed"
        record["mvp"] = summary
        return record
    record["mvp"] = summary
    record["match"] = matches_expected(summary, case)
    record["status"] = "hit" if record["match"]["hit"] else "miss"
    shutil.rmtree(control_target, ignore_errors=True)
    shutil.rmtree(scan_target, ignore_errors=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--cargo-rapx", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    rustc = args.rustc.resolve(strict=True)
    cargo = args.cargo.resolve(strict=True)
    cargo_rapx = args.cargo_rapx.resolve(strict=True)
    rust_sysroot = Path(
        subprocess.check_output([str(rustc), "--print", "sysroot"], text=True).strip()
    )
    preflight = sanitized_environment()
    if sys.platform == "darwin":
        preflight["DYLD_LIBRARY_PATH"] = str(rust_sysroot / "lib")
    rustc_commit = tool_commit(rustc, preflight, "rustc")
    cargo_commit = tool_commit(cargo, preflight, "Cargo")
    if rustc_commit != RUSTC_COMMIT:
        raise SystemExit(f"rustc commit mismatch: {rustc_commit}")
    if cargo_commit != CARGO_COMMIT:
        raise SystemExit(f"cargo commit mismatch: {cargo_commit}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cargo_home = output_root / "cargo-home"
    cargo_home.mkdir(exist_ok=True)
    groups = set(args.group) if args.group else {"A-existing", "A-new-contract"}
    selected = [case for case in CASES if not args.case or case["id"] in args.case]
    results = []
    started = time.time()
    for case in selected:
        print(f"SCAN {case['id']} group={case['group']}", flush=True)
        record = scan_case(
            case,
            cargo=cargo,
            rustc=rustc,
            cargo_rapx=cargo_rapx,
            rust_sysroot=rust_sysroot,
            cargo_commit=cargo_commit,
            rustc_commit=rustc_commit,
            work_root=output_root,
            cargo_home=cargo_home,
            groups=groups,
        )
        results.append(record)
        write_json(output_root / "results.json", {"elapsed_seconds": time.time() - started, "results": results})
        print(
            f"  status={record.get('status')} control={record.get('control')} match={record.get('match', {}).get('hit')}",
            flush=True,
        )
    write_json(output_root / "roster.json", CASES)
    write_json(output_root / "results.json", {"elapsed_seconds": time.time() - started, "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
