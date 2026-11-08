#!/usr/bin/env python3
"""Freeze the 148-issue miss ledger (phase 0) from the v3 journal and sources."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TRUE_HITS = {
    ("Andy-Python-Programmer/aero", "120"),
    ("Umio-Yasuno/amdgpu_top", "104"),
    ("brson/rust-sdl", "168"),
    ("danielpclark/rutie", "190"),
    ("evmar/n2", "122"),
    ("evmar/n2", "123"),
    ("hermit-os/uhyve", "834"),
    ("hermit-os/uhyve", "855"),
    ("jeaye/ncurses-rs", "224"),
    ("ninja-quant/ninjabook", "3"),
    ("zackradisic/rust-vs-zig", "2"),
}

PARTIAL_HITS = {
    ("evmar/n2", "121"): "Scanner::slice not an active root; sibling peek/peek_newline were",
}

FALSE_JOURNAL_HITS = {
    ("Starlight-JS/starlight", "137"): "token crates/stack matched wtf-rs stack_bounds; environment.rs not compiled",
    ("douchuan/jvm", "13"): "matched another from_utf8_unchecked (Tag::from), not find_symbol",
    ("evmar/retrowin32", "74"): "token pointer matched memory crate; Arg<T> is in x86",
}

RUSTSEC_MAP = {
    "2140": "NVSL/Corundum",
    "2141": "rust-ngrammatic/ngrammatic (not in scan roster)",
    "2146": "pleco-rs/Pleco",
    "2149": "perpetual-ml/perpetual",
    "2150": "ninja-quant/ninjabook",
    "2151": "chinedufn/swift-bridge",
}

SKIP_DIR = {".git", "target", ".github", "node_modules"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def api_needles(title: str, obligation: str) -> list[str]:
    text = f"{title} {obligation}"
    found: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        item = item.strip()
        if len(item) < 3:
            return
        key = item.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(item)

    for match in re.finditer(r"[\w./-]+\.rs", text):
        add(match.group(0))
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+", text):
        add(match.group(0))
        add(match.group(0).split("::")[-1])
    for match in re.finditer(r"`([^`]+)`", text):
        add(match.group(1).split("(")[0].split("<")[0])
    for match in re.finditer(r'"([^"]+)"', text):
        inner = match.group(1).split("(")[0]
        if re.match(r"^[A-Za-z_./]", inner):
            add(inner)
    return found


def locate_files(src: Path, needles: list[str]) -> list[str]:
    if not src.exists():
        return []
    hits: list[Path] = []
    for needle in needles:
        if ".rs" in needle or "/" in needle:
            candidate = src / needle
            if candidate.exists() and candidate.is_file():
                hits.append(candidate)
    names = []
    for needle in needles:
        part = needle.split("::")[-1]
        if part.isidentifier() and len(part) >= 3 and part.lower() not in {
            "new", "get", "raw", "name", "path", "from", "into", "drop"
        }:
            names.append(part)
    names = list(dict.fromkeys(names))[:5]
    if names:
        for path in src.rglob("*.rs"):
            if any(part in SKIP_DIR for part in path.parts):
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for name in names:
                if re.search(r"\b(?:pub\s+)?(?:unsafe\s+)?fn\s+" + re.escape(name) + r"\b", text):
                    hits.append(path)
                    break
            if len(hits) >= 8:
                break
    unique: list[str] = []
    seen: set[str] = set()
    for path in hits:
        rel = str(path.relative_to(src))
        if rel not in seen:
            seen.add(rel)
            unique.append(rel)
    return unique[:8]


def nearest_manifest(src: Path, rel: str) -> str:
    current = (src / rel).parent
    while True:
        manifest = current / "Cargo.toml"
        if manifest.exists():
            try:
                return str(manifest.relative_to(src))
            except ValueError:
                return str(manifest)
        if current == src or current.parent == current:
            return ""
        current = current.parent


def load_mvp(work: Path, repo: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = work / repo.replace("/", "__") / "mvp"
    active: list[dict[str, Any]] = []
    unest: list[dict[str, Any]] = []
    supp: list[dict[str, Any]] = []
    if not root.exists():
        return active, unest, supp
    for pkg in root.iterdir():
        if not pkg.is_dir():
            continue
        for line in load_jsonl(pkg / "candidates.jsonl"):
            site = line.get("unsafe_site") or {}
            span = site.get("span") or {}
            active.append(
                {
                    "root": ((line.get("public_safe_root") or {}).get("owner")),
                    "callee": site.get("callee"),
                    "path": span.get("path"),
                    "pattern": line.get("primary_pattern"),
                }
            )
        for line in load_jsonl(pkg / "unestablished.jsonl"):
            unest.append(
                {
                    "root": line.get("public_root"),
                    "reason": line.get("primary_reason"),
                    "pattern": line.get("attempted_pattern"),
                }
            )
        for line in load_jsonl(pkg / "suppression-ledger.jsonl"):
            cand = line.get("candidate") or {}
            site = cand.get("unsafe_site") or {}
            supp.append(
                {
                    "root": ((cand.get("public_safe_root") or {}).get("owner")),
                    "reason": line.get("reason"),
                    "pattern": cand.get("primary_pattern"),
                    "callee": site.get("callee"),
                }
            )
    return active, unest, supp


def blob_hit(blob: str, needles: list[str]) -> list[str]:
    blob_l = blob.lower()
    hits = []
    weak = {"slice", "pointer", "stack", "crates", "length", "offset", "table", "free"}
    for needle in needles:
        key = needle.lower()
        if len(key) < 4:
            continue
        if key in weak:
            continue
        if key in blob_l:
            hits.append(needle)
    return hits


def classify_fail_text(text: str) -> str:
    low = text.lower()
    rules = [
        ("S6_windows", r"winapi|windows-sys|cannot find `shared` in `winapi`"),
        ("S6_x86_asm", r"invalid register `r[a-z]x`|unknown register"),
        ("S5_libclang", r"libclang|bindgen"),
        ("S5_git_dep", r"revspec|failed to fetch|failed to get `"),
        ("S5_locked", r"--locked was passed|cannot update the lock file"),
        ("S5_missing_crate", r"no matching package named"),
        ("S5_nightly_feature", r"unstable library feature|unknown feature"),
        ("S6_json_target", r"json. target spec|-zjson-target-spec"),
        ("S5_generated_span", r"repository-relative span"),
        ("S5_native_lib", r"pkg-config|failed to run custom build"),
        ("S5_ring_asm", r"ring-0\.|aesv8"),
        ("S5_type_error", r"mismatched types|could not compile"),
    ]
    for name, pattern in rules:
        if re.search(pattern, low):
            return name
    return "S5_compile_other"


def repo_logs(work: Path, repo: str) -> str:
    logs = work / repo.replace("/", "__") / "logs"
    if not logs.exists():
        return ""
    parts = []
    for path in sorted(logs.glob("*.log")):
        try:
            parts.append(path.read_text(errors="replace")[-6000:])
        except OSError:
            pass
    return "\n".join(parts)


def main() -> int:
    workspace = Path("/Users/liaowenzhao/Documents/RAP重构")
    rap = workspace / "RAPx-unsoundaudit-v2"
    src = workspace / "lwz23-valid-issue-src"
    scan = workspace / "lwz23-valid-issue-scan"
    roster = rap / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1"
    issues = list(csv.DictReader((roster / "issues.csv").open()))
    journal_rows = load_jsonl(scan / "scan_journal.jsonl")
    latest = {row["repository"]: row for row in journal_rows if row.get("repository")}
    obligations: dict[str, str] = {}
    for instance in load_jsonl(
        rap / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl"
    ):
        url = str(instance.get("issue_url") or "")
        text = instance.get("repair_obligation_zh") or ""
        if url and text:
            obligations[url] = (obligations.get(url, "") + " " + text).strip()

    out_dir = scan / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []

    for issue in issues:
        repo = issue["repository"]
        number = str(issue["issue_number"])
        url = issue["issue_url"]
        title = issue["title"]
        rec = latest.get(repo)
        status = rec.get("status") if rec else "no_repo"
        error = (rec.get("error") if rec else "") or ""
        obligation = obligations.get(url, "")
        needles = api_needles(title, obligation)
        files = locate_files(src / repo, needles)
        manifests = [nearest_manifest(src / repo, rel) for rel in files]
        manifests = [item for item in dict.fromkeys(manifests) if item]
        active, unest, supp = load_mvp(scan / "work", repo) if rec else ([], [], [])
        act_blob = " ".join(json.dumps(item, ensure_ascii=False) for item in active)
        un_blob = " ".join(json.dumps(item, ensure_ascii=False) for item in unest)
        su_blob = " ".join(json.dumps(item, ensure_ascii=False) for item in supp)
        in_active = blob_hit(act_blob, needles)
        in_unest = blob_hit(un_blob, needles)
        in_supp = blob_hit(su_blob, needles)
        journal_hit = False
        if rec:
            for match in rec.get("issue_matches") or []:
                if (match.get("issue_url") == url or match.get("title") == title) and match.get("hit"):
                    journal_hit = True
        key = (repo, number)
        if key in TRUE_HITS:
            verdict = "true_hit"
            bucket = "HIT"
            recover = "keep"
            note = "adjudicated: active finding is the reported API"
        elif key in PARTIAL_HITS:
            verdict = "partial"
            bucket = "S2_matcher"
            recover = "phase1_matcher_or_same_crate_already"
            note = PARTIAL_HITS[key]
        elif key in FALSE_JOURNAL_HITS:
            verdict = "false_journal_hit"
            bucket = "S2_matcher"
            recover = "phase1_right_package"
            note = FALSE_JOURNAL_HITS[key]
        elif repo == "rustsec/advisory-db":
            verdict = "miss"
            bucket = "S7_meta"
            recover = "retarget:" + RUSTSEC_MAP.get(number, "unknown")
            note = "advisory-db issue; scan the mapped crate"
        elif repo == "BrowserWorks/waterfox":
            verdict = "miss"
            bucket = "S7_excluded"
            recover = "out_of_host_or_skip"
            note = "clone timeout; excluded from 125-repo scan"
        elif "Operation not permitted" in error:
            verdict = "miss"
            bucket = "S0_eperm"
            recover = "phase1_retry_no_sandbox"
            note = "Python OSError during scan; source is present"
        elif status == "no_package":
            verdict = "miss"
            bucket = "S1_nested_manifest"
            recover = "phase1_nested_manifest"
            note = "root Cargo.toml missing; nearest manifests: " + ",".join(manifests[:4])
        elif status in {"control_failed", "scan_failed", "error"}:
            logs = repo_logs(scan / "work", repo)
            fail = classify_fail_text(logs + " " + error)
            if fail.startswith("S6"):
                bucket = fail
                recover = "phase4_platform"
            elif fail == "S5_generated_span":
                bucket = fail
                recover = "phase3_span_policy"
            elif fail == "S5_locked":
                bucket = fail
                recover = "phase1_unlocked_retry"
            else:
                bucket = fail
                recover = "phase3_compile"
            verdict = "miss"
            note = fail
        elif in_unest and not in_active:
            verdict = "miss"
            bucket = "S3_gated_unestablished"
            recover = "phase2_heuristic_p1"
            note = "reported tokens in unestablished.jsonl"
        elif in_supp and not in_active:
            verdict = "miss"
            bucket = "S3_gated_suppressed"
            recover = "phase2_review_suppression"
            note = "reported tokens only in suppression ledger"
        elif status in {"miss", "hit"} and (len(active) + len(unest) + len(supp) == 0):
            verdict = "miss"
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "scan completed with empty four-state; likely wrong package"
        elif status in {"miss", "hit"} and not in_active:
            cfg_hint = ""
            joined = " ".join(files).lower() + title.lower()
            if "wasm" in joined or "as_bytes" in title.lower():
                bucket = "S4_cfg_wasm"
                recover = "phase3_wasm32"
                cfg_hint = "wasm/cfg suspected"
            elif "winapi" in joined or "d3d12" in title.lower() or "memcpy" in title.lower():
                bucket = "S4_cfg_windows"
                recover = "phase4_windows_or_cfg"
                cfg_hint = "windows/cfg suspected"
            else:
                bucket = "S1_wrong_or_empty_package"
                recover = "phase1_target_package"
                cfg_hint = "API not in four-state of scanned packages"
            verdict = "miss"
            note = cfg_hint
        else:
            verdict = "miss"
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "default miss"

        # special cases that token search misses
        if repo == "Amanieu/atomic-rs":
            bucket = "S3_gated_unestablished"
            recover = "phase2_heuristic_p1"
            note = "1157 P1 unmodeled_unsafe_call on Atomic::fetch_*; old RAP Pattern1"
        elif repo == "async-rs/async-std":
            bucket = "S3_gated_unestablished"
            recover = "phase2_heuristic_p1"
            note = "44 P1 unmodeled on public wrappers"
        elif repo == "smol-rs/async-task":
            bucket = "S3_gated_unestablished"
            recover = "phase2_heuristic_p1"
            note = "spawn/waker in unestablished/suppressed"
        elif repo == "bluss/arrayvec" and verdict != "true_hit":
            bucket = "S3_gated_unestablished"
            recover = "phase2_heuristic_p1"
            note = "some exact P1 active; many old P1s suppressed/unestablished"
        elif repo == "Ralith/hecs":
            bucket = "S3_gated_unestablished"
            recover = "phase2_heuristic_p1"
            note = "View::get_mut is P1 unmodeled_unsafe_call"
        elif repo == "CosmWasm/cosmwasm":
            bucket = "S4_cfg_wasm"
            recover = "phase3_wasm32"
            note = "Region::as_bytes behind cfg(target_arch=wasm32)"
        elif repo == "Eugeny/russh":
            bucket = "S4_cfg_windows"
            recover = "phase3_wasm_or_windows_cfg"
            note = "memcpy only in platform/{windows,wasm}.rs; CryptoVec P1s are related not the report"
        elif repo == "Traverse-Research/gpu-allocator":
            bucket = "S4_cfg_windows"
            recover = "phase4_windows"
            note = "from_winapi_d3d12_resource_desc is cfg(all(windows, feature=d3d12))"
        elif repo == "bluealloy/revm" or repo == "bnb-chain/revm":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "ExtBytecode/current_opcode live in crates/interpreter, not first-4 libs"
        elif repo == "eclipse-zenoh/zenoh":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "OwnedDescriptor/watchdog in commons/zenoh-shm"
        elif repo == "Andy-Python-Programmer/aero" and number == "121":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "RSDP as_ptr is in aero_kernel, not aero_syscall"
        elif repo == "Xudong-Huang/may":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "wakeup+get_unchecked in may (kqueue.rs); may control_failed, only may_queue scanned"
        elif repo == "Pagefind/pagefind":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "request_filter_indexes in pagefind_web"
        elif repo == "nuta/kerla":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "parse_str_field in kernel/fs/initramfs.rs"
        elif repo == "Starlight-JS/starlight" and number == "137":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "starlight crate control_failed; environment.rs from_raw_parts not scanned"
        elif repo == "douchuan/jvm" and number == "13":
            bucket = "S1_wrong_or_empty_package"
            recover = "phase1_target_package"
            note = "find_symbol is crates/vm; scanned classfile Tag::from instead"

        rows_out.append(
            {
                "repository": repo,
                "issue_number": number,
                "issue_url": url,
                "title": title,
                "journal_status": status,
                "journal_hit": str(journal_hit).lower(),
                "verdict": verdict,
                "bucket": bucket,
                "recover_in": recover,
                "n_active": rec.get("n_active") if rec else "",
                "n_unestablished": len(unest),
                "n_suppressed": len(supp),
                "packages_scanned": ",".join(rec.get("packages") or []) if rec else "",
                "api_needles": " | ".join(needles[:8]),
                "source_files": " | ".join(files[:6]),
                "nearest_manifests": " | ".join(manifests[:6]),
                "in_active_needles": " | ".join(in_active[:6]),
                "in_unestablished_needles": " | ".join(in_unest[:6]),
                "in_suppressed_needles": " | ".join(in_supp[:6]),
                "error": error[:180],
                "note": note,
            }
        )

    csv_path = out_dir / "miss_ledger.csv"
    fields = list(rows_out[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    verdicts = Counter(row["verdict"] for row in rows_out)
    buckets = Counter(row["bucket"] for row in rows_out)
    recover = Counter(row["recover_in"].split(":")[0] for row in rows_out)
    md = out_dir / "MISS_LEDGER.md"
    lines = [
        "# lwz23 148-issue miss ledger (phase 0)",
        "",
        "Frozen from RAP v3 journal + source trees. `true_hit` uses the adjudicated reported-API rule, not the title-token journal flag.",
        "",
        f"- issues: {len(rows_out)}",
        f"- true_hit: {verdicts.get('true_hit', 0)}",
        f"- partial: {verdicts.get('partial', 0)}",
        f"- false_journal_hit: {verdicts.get('false_journal_hit', 0)}",
        f"- miss: {verdicts.get('miss', 0)}",
        "",
        "## Verdicts",
        "",
    ]
    for key, value in verdicts.most_common():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Buckets", ""]
    for key, value in buckets.most_common():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Recover-in", ""]
    for key, value in recover.most_common():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Phase 1 retry repositories", ""]
    phase1 = sorted(
        {
            row["repository"]
            for row in rows_out
            if row["recover_in"].startswith("phase1") and row["verdict"] != "true_hit"
        }
    )
    for repo in phase1:
        n = sum(1 for row in rows_out if row["repository"] == repo)
        lines.append(f"- `{repo}` ({n} issues)")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    retry_path = out_dir / "phase1_retry_repos.txt"
    retry_path.write_text("\n".join(phase1) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md}")
    print(f"wrote {retry_path} n={len(phase1)}")
    print("verdicts", dict(verdicts))
    print("buckets", dict(buckets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
