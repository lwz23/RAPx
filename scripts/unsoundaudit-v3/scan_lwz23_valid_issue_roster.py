#!/usr/bin/env python3
"""Serial RAP v3 scan of frozen lwz23 valid-issue sources, matching reported bugs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cratesio_campaign_common import sanitized_environment, tool_commit  # noqa: E402


class ProcessResult(NamedTuple):
    classification: str
    returncode: int | None
    elapsed_seconds: float
    max_physical_footprint_bytes: int
    max_rss_bytes: int


def terminate_group(pid: int) -> None:
    def send(sig: int) -> bool:
        for target in ((os.killpg, pid), (os.kill, pid)):
            try:
                target[0](target[1], sig)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                continue
        return False

    send(signal.SIGTERM)
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return
        time.sleep(0.05)
    send(signal.SIGKILL)


MEMCG_HELPER = Path(__file__).with_name("run_in_user_memcg.py")
MEMORY_MAX_DEFAULT = "18G"
OOM_RETURNCODES = {137, 9, -9, 247}


def user_systemd_ready() -> bool:
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    return (runtime / "bus").exists()


def memory_cgroup_command(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> tuple[list[str], str | None, Path | None]:
    """Wrap command in a user systemd unit with MemoryMax. Scanner stays outside."""
    require = os.environ.get("RAP_REQUIRE_MEMORY_CGROUP", "1").strip().lower()
    if require in {"0", "false", "no"}:
        return command, None, None
    if not user_systemd_ready():
        raise SystemExit(
            "user systemd bus missing; refuse to run RAP in an uncapped session"
        )
    memory_max = os.environ.get("RAP_MEMORY_MAX") or MEMORY_MAX_DEFAULT
    unit = f"rapx-mem-{os.getpid()}-{time.time_ns()}"
    env_path = log_path.with_suffix(log_path.suffix + ".memcg-env.json")
    env_path.write_text(json.dumps(environment, ensure_ascii=False), encoding="utf-8")
    wrapped = [
        "systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--pipe",
        "--quiet",
        f"--unit={unit}",
        f"--working-directory={cwd}",
        "-p",
        f"MemoryMax={memory_max}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=kill",
        "-p",
        "ManagedOOMMemoryPressure=kill",
        "-p",
        "KillMode=control-group",
        "--",
        sys.executable,
        str(MEMCG_HELPER),
        str(env_path),
        str(cwd),
        *command,
    ]
    return wrapped, unit, env_path


def classify_exit(returncode: int | None, log_path: Path) -> str:
    if returncode == 0:
        return "success"
    blob = ""
    try:
        blob = log_path.read_text(errors="replace")[-4000:].lower()
    except OSError:
        blob = ""
    if returncode in OOM_RETURNCODES or "oom-kill" in blob or "oom_kill" in blob:
        return "oom_killed"
    return "compiler_failure"


def stop_user_unit(unit: str | None) -> None:
    if not unit:
        return
    subprocess.run(
        ["systemctl", "--user", "stop", unit],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def run_limited(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    wall_time_seconds: float,
    physical_footprint_limit_bytes: int | None,
) -> ProcessResult:
    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped, unit, env_path = memory_cgroup_command(command, cwd, environment, log_path)
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    run_env = os.environ.copy()
    run_env["XDG_RUNTIME_DIR"] = runtime
    run_env["DBUS_SESSION_BUS_ADDRESS"] = os.environ.get(
        "DBUS_SESSION_BUS_ADDRESS"
    ) or f"unix:path={runtime}/bus"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            wrapped,
            cwd=cwd,
            env=run_env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        classification = "compiler_failure"
        returncode: int | None = None
        try:
            while True:
                returncode = process.poll()
                elapsed = time.monotonic() - start
                if returncode is not None:
                    classification = classify_exit(returncode, log_path)
                    break
                if elapsed > wall_time_seconds:
                    classification = "timeout"
                    stop_user_unit(unit)
                    terminate_group(process.pid)
                    returncode = process.wait()
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                stop_user_unit(unit)
                terminate_group(process.pid)
                process.wait()
            if env_path is not None:
                env_path.unlink(missing_ok=True)
    return ProcessResult(classification, returncode, time.monotonic() - start, 0, 0)


RUSTC_COMMIT = "ba28ff76f353a722f31c4f3dd2ac4e437d36411b"
CARGO_COMMIT = "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0"
RUSTC_COMMIT_2024 = "1bc403daadbebb553ccc211a0a8eebb73989665f"
CARGO_COMMIT_2024 = "15fbd2f607d4defc87053b8b76bf5038f2483cf4"
ALLOWED_TOOLCHAINS = {
    (RUSTC_COMMIT, CARGO_COMMIT),
    (RUSTC_COMMIT_2024, CARGO_COMMIT_2024),
}
CONTROL_SECONDS = 900
KNOB_SECONDS = 300
SCAN_SECONDS = 1800
REPO_WALL_SECONDS = 3600
PHYSICAL_LIMIT = 10 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024
MAX_UNTARGETED_PACKAGES = 12
SKIP_DIR_NAMES = {".git", "target", "node_modules", ".github"}
WEAK_MATCH_TOKENS = {
    "slice",
    "crates",
    "stack",
    "pointer",
    "possible",
    "report",
    "access",
    "starlight",
    "length",
    "offset",
    "table",
    "free",
    "name",
    "path",
    "from",
}
STOPWORDS = {
    "unsound",
    "unsoundness",
    "undefined",
    "behavior",
    "maybe",
    "function",
    "functions",
    "problem",
    "problems",
    "public",
    "issue",
    "crate",
    "rust",
    "safe",
    "unsafe",
    "multiple",
    "potential",
    "soundness",
    "causes",
    "leading",
    "memory",
    "corruption",
    "allows",
    "implementation",
    "missing",
    "check",
    "checks",
    "lack",
    "bounds",
    "invalid",
    "input",
    "handling",
    "method",
    "other",
}


def host_library_path(rust_sysroot: Path) -> dict[str, str]:
    lib = str(rust_sysroot / "lib")
    env = {"LD_LIBRARY_PATH": lib}
    if platform.system() == "Darwin":
        env["DYLD_LIBRARY_PATH"] = lib
    return env


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def host_scan_environment(
    rustc: Path,
    rust_sysroot: Path,
    cargo: Path,
    cargo_home: Path,
    rustc_commit: str,
    cargo_commit: str,
) -> dict[str, str]:
    """Generic host native env. Linux uses distro z3/libclang; macOS keeps brew."""
    env: dict[str, str] = {
        "RUSTC": str(rustc),
        "RUST_SYSROOT": str(rust_sysroot),
        "CARGO_HOME": str(cargo_home),
        "CARGO_BUILD_JOBS": "1",
        "CARGO_NET_GIT_FETCH_WITH_CLI": "true",
        "RAP_CLEAN": "false",
        "UNSOUND_SCANNER_EXACT_CARGO_PATH": str(cargo),
        "UNSOUND_SCANNER_EXACT_CARGO_COMMIT": cargo_commit,
        "UNSOUND_SCANNER_EXPECTED_RUSTC_PATH": str(rustc),
        "UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT": rustc_commit,
    }
    env.update(host_library_path(rust_sysroot))
    if platform.system() == "Darwin":
        llvm = Path(subprocess.check_output(["brew", "--prefix", "llvm"], text=True).strip())
        z3 = Path(subprocess.check_output(["brew", "--prefix", "z3"], text=True).strip())
        old_path = os.environ.get("PATH", "")
        env.update(
            {
                "PATH": f"{llvm / 'bin'}:{old_path}" if old_path else str(llvm / "bin"),
                "LIBCLANG_PATH": str(llvm / "lib"),
                "Z3_SYS_Z3_HEADER": str(z3 / "include" / "z3.h"),
                "RUSTFLAGS": f"-Lnative={z3 / 'lib'}",
            }
        )
        return env

    llvm_lib = _first_existing(
        [
            Path("/usr/lib/llvm-15/lib"),
            Path("/usr/lib/llvm-18/lib"),
            Path("/usr/lib/llvm-17/lib"),
            Path("/usr/lib/llvm-16/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]
    )
    z3_header = _first_existing(
        [
            Path("/usr/include/z3.h"),
            Path("/usr/include/z3/z3.h"),
        ]
    )
    z3_lib = _first_existing(
        [
            Path("/usr/lib/x86_64-linux-gnu/libz3.so"),
            Path("/usr/lib/libz3.so"),
        ]
    )
    if llvm_lib is not None:
        env["LIBCLANG_PATH"] = str(llvm_lib)
    if z3_header is not None:
        env["Z3_SYS_Z3_HEADER"] = str(z3_header)
    if z3_lib is not None:
        env["RUSTFLAGS"] = f"-Lnative={z3_lib.parent}"
    return env


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_journal(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = row.get("repository")
        if key:
            latest[str(key)] = row
    return latest


def tokens_from_text(*texts: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(token: str, *, allow_short: bool = False) -> None:
        item = token.strip().lower()
        min_len = 2 if allow_short else 4
        if len(item) < min_len or item in STOPWORDS or item in seen:
            return
        seen.add(item)
        found.append(item)

    blob = "\n".join(t for t in texts if t)
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+", blob):
        add(match.group(0), allow_short=True)
        for part in match.group(0).split("::"):
            add(part, allow_short=True)
    for match in re.finditer(r"`([^`]+)`", blob):
        inner = match.group(1).split("(")[0].split("<")[0]
        add(inner, allow_short=True)
        if "::" in inner:
            add(inner.split("::")[-1], allow_short=True)
    for match in re.finditer(r"'([A-Za-z_][A-Za-z0-9_:]*)'", blob):
        add(match.group(1), allow_short=True)
        add(match.group(1).split("::")[-1], allow_short=True)
    for match in re.finditer(r"[\w./-]+\.rs", blob):
        add(match.group(0), allow_short=True)
    for match in re.finditer(r"\b(?:fn|pub fn)\s+([A-Za-z_][A-Za-z0-9_]*)\b", blob):
        add(match.group(1), allow_short=True)
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", blob):
        add(match.group(0))
    return found


def finding_hay(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "").lower()
        for key in ("root", "callee", "kind", "path")
    )


def token_matches_finding(token: str, row: dict[str, Any]) -> bool:
    hay = finding_hay(row)
    root = str(row.get("root") or "").lower()
    path = str(row.get("path") or "").lower()
    if token not in hay:
        return False
    if token in WEAK_MATCH_TOKENS:
        last = root.split("::")[-1]
        stem = Path(path).stem.lower() if path else ""
        return token == last or token == stem or token in root.split("::")
    if token == "get":
        return root.endswith("::get") or root.endswith("::get_mut") or "::get(" in root
    if token == "slice" and "core::slice" in str(row.get("callee") or "").lower():
        return "slice" in root.split("::")
    return True


def match_rows(issue: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = tokens_from_text(
        issue.get("body") or "",
        issue.get("repair_obligation_zh") or "",
        issue.get("title") or "",
    )
    hits = []
    for row in rows:
        matched = [token for token in tokens if token_matches_finding(token, row)]
        hay = finding_hay(row)
        if "starlight" in hay or "find_symbol" in hay or ("retrowin32" in hay and "arg" in hay):
            continue
        if matched:
            hits.append({"finding": row, "matched_tokens": matched})
    return {"hit": bool(hits), "tokens": tokens, "hits": hits[:8]}


def match_issue(issue: dict[str, Any], active: list[dict[str, Any]]) -> dict[str, Any]:
    return match_rows(issue, active)


def summarize_mvp(mvp: Path) -> dict[str, Any]:
    candidates = load_jsonl(mvp / "candidates.jsonl")
    families = load_jsonl(mvp / "families.jsonl")
    unestablished = load_jsonl(mvp / "unestablished.jsonl")
    ledger = load_jsonl(mvp / "suppression-ledger.jsonl")
    coverage: dict[str, Any] = {}
    if (mvp / "coverage.json").exists():
        coverage = json.loads((mvp / "coverage.json").read_text())
    active = []
    for row in candidates:
        span = (row.get("unsafe_site") or {}).get("span") or {}
        active.append(
            {
                "pattern": row.get("primary_pattern"),
                "root": ((row.get("public_safe_root") or {}).get("owner")),
                "callee": ((row.get("unsafe_site") or {}).get("callee")),
                "kind": ((row.get("unsafe_site") or {}).get("kind")),
                "path": span.get("path"),
                "line": ((span.get("start") or {}).get("line")),
                "model": ((row.get("contract") or {}).get("model_id")),
            }
        )
    unestablished_rows = []
    for row in unestablished:
        unestablished_rows.append(
            {
                "pattern": row.get("attempted_pattern"),
                "root": row.get("public_root"),
                "callee": "",
                "kind": "unestablished",
                "path": "",
                "line": None,
                "model": row.get("primary_reason"),
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
        "active_preview": active[:20],
        "unestablished": unestablished_rows,
    }


def nearest_manifest(path: Path, src: Path) -> Path | None:
    current = path.parent if path.is_file() else path
    while True:
        manifest = current / "Cargo.toml"
        if manifest.exists():
            return manifest
        if current == src or current.parent == current:
            return None
        current = current.parent


def find_issue_source_files(src: Path, issues: list[dict[str, Any]]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen or not path.exists():
            return
        seen.add(key)
        found.append(path)

    src = src.resolve()
    github_blob = re.compile(
        r"github\.com/[^/\s]+/[^/\s]+/(?:blob|tree)/[^/\s]+/([\w./-]+\.rs)",
        re.I,
    )
    rel_rs = re.compile(
        r"(?:(?<=[\s`\"'(])|^)((?:src|crates|packages|lib|core|pagefind(?:_[a-z0-9]+)?|wasm|runtime)[\w./-]*\.rs)",
        re.I,
    )
    nested_rs = re.compile(
        r"(?:(?<=[\s`\"'(])|^)([\w.-]+(?:/[\w.-]+){1,8}\.rs)",
    )

    for issue in issues:
        before = len(found)
        blob = "\n".join(
            str(issue.get(key) or "")
            for key in ("body", "repair_obligation_zh", "title")
        )
        blob_norm = blob.replace("\\", "/")
        for match in github_blob.finditer(blob_norm):
            add(src / match.group(1))
        for match in rel_rs.finditer(blob_norm):
            add(src / match.group(1))
        for match in nested_rs.finditer(blob_norm):
            rel = match.group(1)
            if rel.startswith("http"):
                continue
            add(src / rel)
        methods: list[str] = []
        for match in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", blob):
            methods.append(match.group(1))
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*::([A-Za-z_][A-Za-z0-9_]*)", blob):
            methods.append(match.group(1))
        for match in re.finditer(r"`([^`]+)`", blob):
            methods.append(match.group(1).split("(")[0].split("::")[-1].split("<")[0])
        methods = [name for name in methods if name.isidentifier() and len(name) >= 3]
        methods = list(dict.fromkeys(methods))[:8]
        type_names: list[str] = []
        for match in re.finditer(r"\b(?:struct|enum|trait)\s+([A-Z][A-Za-z0-9_]*)", blob):
            type_names.append(match.group(1))
        for match in re.finditer(
            r"\bimpl(?:<[^>]+>)?\s+(?:for\s+)?([A-Z][A-Za-z0-9_]*)", blob
        ):
            type_names.append(match.group(1))
        for match in re.finditer(r"`([A-Z][A-Za-z0-9_]*)`", blob):
            type_names.append(match.group(1))
        type_names = list(dict.fromkeys(type_names))[:8]
        generic_fns = {
            "new", "main", "drop", "clone", "default", "fmt", "from", "into",
            "from_ptr", "as_ptr", "as_mut", "as_ref", "from_raw", "into_raw",
            "offset", "push", "to_string", "unwrap", "expect", "len",
            "is_null", "as_str", "to_str", "from_utf8", "from_raw_parts",
        }
        if not type_names:
            methods = [name for name in methods if name not in generic_fns]
        if len(found) > before:
            continue
        if not methods or len(found) >= 12:
            continue
        for rust in src.rglob("*.rs"):
            if any(part in SKIP_DIR_NAMES for part in rust.parts):
                continue
            try:
                rust_text = rust.read_text(errors="replace")
            except OSError:
                continue
            if type_names and not any(name in rust_text for name in type_names):
                continue
            for name in methods:
                if re.search(r"\bfn\s+" + re.escape(name) + r"\b", rust_text):
                    add(rust)
                    break
            if len(found) >= 12:
                break
        distinctive = [
            name
            for name in methods
            if name not in generic_fns and len(name) >= 7
        ]
        if distinctive and len(found) < 12:
            for rust in src.rglob("*.rs"):
                if any(part in SKIP_DIR_NAMES for part in rust.parts):
                    continue
                try:
                    rust_text = rust.read_text(errors="replace")
                except OSError:
                    continue
                for name in distinctive:
                    if re.search(r"\bfn\s+" + re.escape(name) + r"\b", rust_text):
                        add(rust)
                        break
                if len(found) >= 12:
                    break
    return found


def ensure_git_submodules(src: Path, log: Path) -> None:
    """Best-effort checkout of path deps declared in .gitmodules. No repo allowlists."""
    if not (src / ".gitmodules").exists():
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        stream.write(b"\n# git submodule update --init --recursive\n")
        subprocess.run(
            ["git", "-C", str(src), "submodule", "update", "--init", "--recursive"],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )



def _strip_workspace_table(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("[") and stripped.rstrip().endswith("]"):
            key = stripped.strip()[1:-1].strip()
            skipping = key == "workspace" or key.startswith("workspace.")
        if skipping:
            continue
        out.append(line)
    return "".join(out)


def _materialize_workspace_keys(text: str) -> str:
    text = re.sub(
        r"version\s*=\s*\{\s*workspace\s*=\s*true\s*\}",
        'version = "0.1.0"',
        text,
    )
    text = re.sub(
        r"edition\s*=\s*\{\s*workspace\s*=\s*true\s*\}",
        'edition = "2021"',
        text,
    )
    text = re.sub(r"^version\.workspace\s*=\s*true\s*$", 'version = "0.1.0"', text, flags=re.M)
    text = re.sub(r"^edition\.workspace\s*=\s*true\s*$", 'edition = "2021"', text, flags=re.M)
    text = re.sub(
        r"^[A-Za-z0-9_.-]+\.workspace\s*=\s*true\s*$",
        "",
        text,
        flags=re.M,
    )
    text = re.sub(r",\s*workspace\s*=\s*true", "", text)
    text = re.sub(r"workspace\s*=\s*true\s*,", "", text)
    text = re.sub(r"workspace\s*=\s*true", "", text)
    text = re.sub(r"\{\s*,", "{", text)
    text = re.sub(r",\s*\}", "}", text)
    return text


def _ensure_stub_crate(work: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "stub"
    stub = work / "stubs" / safe
    src_dir = stub / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (stub / "Cargo.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2021"\n'
        f'[lib]\npath = "src/lib.rs"\n',
        encoding="utf-8",
    )
    lib = src_dir / "lib.rs"
    if not lib.exists():
        lib.write_text("", encoding="utf-8")
    return stub


def _rewrite_path_deps(text: str, orig_dir: Path, work: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix, rel, suffix = match.group(1), match.group(2), match.group(3)
        if rel.endswith(".rs") or rel.startswith("src/"):
            return match.group(0)
        target = (orig_dir / rel).resolve()
        if (target / "Cargo.toml").is_file():
            return f'{prefix}{target}{suffix}'
        name_match = re.search(r"([A-Za-z0-9_-]+)\s*=\s*\{[^}]*$", prefix)
        dep_name = name_match.group(1) if name_match else target.name
        stub = _ensure_stub_crate(work, dep_name)
        return f'{prefix}{stub.resolve()}{suffix}'

    return re.sub(
        r'([A-Za-z0-9_-]+\s*=\s*\{[^}]*?path\s*=\s*")([^"]+)(")',
        repl,
        text,
    )


def _absolute_lib_path(orig_dir: Path, text: str) -> str:
    block = re.search(r"(?ms)^\[lib\](.*?)(?=^\[|\Z)", text)
    if block:
        path_m = re.search(r'path\s*=\s*"([^"]+)"', block.group(1))
        if path_m:
            return str((orig_dir / path_m.group(1)).resolve())
    for rel in ("src/lib.rs", "src/main.rs", "lib.rs", "main.rs"):
        candidate = orig_dir / rel
        if candidate.exists():
            return str(candidate.resolve())
    return str((orig_dir / "src" / "lib.rs").resolve())


def _set_lib_path(text: str, abs_path: str) -> str:
    if re.search(r"(?m)^\[lib\]", text):
        def repl_block(match: re.Match[str]) -> str:
            block = match.group(0)
            if re.search(r'path\s*=\s*"', block):
                block = re.sub(r'path\s*=\s*"[^"]*"', f'path = "{abs_path}"', block, count=1)
            else:
                block = block.rstrip() + f'\npath = "{abs_path}"\n'
            return block

        return re.sub(r"(?ms)^\[lib\](.*?)(?=^\[|\Z)", repl_block, text, count=1)
    if not text.endswith("\n"):
        text += "\n"
    return text + f'\n[lib]\npath = "{abs_path}"\n'


def isolate_package_for_metadata(src: Path, manifest: Path, work: Path) -> Path:
    """Write a standalone Cargo.toml for one package; do not edit the checkout."""
    src = src.resolve()
    manifest = manifest.resolve()
    orig_dir = manifest.parent
    work = work.resolve()
    dest = work / orig_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    text = manifest.read_text(encoding="utf-8")
    text = _strip_workspace_table(text)
    text = _materialize_workspace_keys(text)
    text = _rewrite_path_deps(text, orig_dir, work)
    text = _set_lib_path(text, _absolute_lib_path(orig_dir, manifest.read_text(encoding="utf-8")))
    isolated = dest / "Cargo.toml"
    isolated.write_text(text, encoding="utf-8")
    return isolated


def metadata_packages(src: Path, cargo: Path, env: dict[str, str], manifest: Path, log: Path) -> list[dict[str, str]]:
    log.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest.resolve()
    src = src.resolve()
    with log.open("ab") as stream:
        stream.write(f"\n# metadata {manifest}\n".encode())
        result = subprocess.run(
            [str(cargo), "metadata", "--format-version", "1", "--no-deps", "--manifest-path", str(manifest)],
            cwd=str(manifest.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stream,
            check=False,
        )
    if result.returncode != 0:
        return []
    try:
        metadata = json.loads(result.stdout.decode())
    except json.JSONDecodeError:
        return []
    packages = []
    for pkg in metadata.get("packages") or []:
        manifest_path = Path(pkg.get("manifest_path") or "")
        try:
            in_tree = True
            manifest_path.resolve().relative_to(src)
        except ValueError:
            in_tree = manifest_path.resolve() == manifest.resolve()
        if not in_tree:
            continue
        lib = False
        bin_name = None
        for target in pkg.get("targets") or []:
            kind = set(target.get("kind") or [])
            if kind & {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}:
                lib = True
            if "bin" in kind and not bin_name:
                bin_name = target.get("name")
        if not lib and not bin_name:
            continue
        packages.append(
            {
                "name": pkg.get("name") or manifest_path.parent.name,
                "manifest": str(manifest_path),
                "cwd": str(manifest_path.parent),
                "lib": lib,
                "bin_name": bin_name or "",
            }
        )
    return packages


def select_packages(
    src: Path,
    cargo: Path,
    env: dict[str, str],
    log: Path,
    issues: list[dict[str, Any]] | None = None,
    isolate_root: Path | None = None,
) -> list[dict[str, str]]:
    issue_files = find_issue_source_files(src, issues or [])
    manifests: list[Path] = []
    for file in issue_files:
        manifest = nearest_manifest(file, src)
        if manifest and manifest not in manifests:
            manifests.append(manifest)
    if not manifests:
        root = src / "Cargo.toml"
        if root.exists():
            manifests = [root]
        else:
            for path in sorted(src.rglob("Cargo.toml")):
                if any(part in SKIP_DIR_NAMES for part in path.parts):
                    continue
                rel = path.relative_to(src).as_posix()
                if rel.count("/") <= 2:
                    manifests.append(path)
            manifests = manifests[:8]
    if not manifests:
        return []

    packages: list[dict[str, str]] = []
    seen: set[str] = set()
    work = isolate_root if isolate_root is not None else log.parent / "isolate"
    for manifest in manifests:
        found: list[dict[str, str]] = []
        try:
            isolated = isolate_package_for_metadata(src, manifest, work)
            found = metadata_packages(src, cargo, env, isolated, log)
        except OSError:
            found = []
        if not found:
            found = metadata_packages(src, cargo, env, manifest, log)
        for package in found:
            key = package["manifest"]
            if key in seen:
                continue
            seen.add(key)
            packages.append(package)

    if issue_files:
        targeted = []
        for package in packages:
            pkg_dir = Path(package["cwd"])
            for file in issue_files:
                try:
                    file.resolve().relative_to(pkg_dir.resolve())
                except (ValueError, OSError):
                    continue
                targeted.append(package)
                break
        if targeted:
            return targeted
    libs = [package for package in packages if package["lib"]]
    chosen = libs or packages
    if issue_files:
        return chosen
    return chosen[:MAX_UNTARGETED_PACKAGES]


def cargo_args(
    package: dict[str, str],
    locked: bool,
    extra: list[str] | None = None,
) -> list[str]:
    args = ["-p", package["name"]]
    if package["lib"]:
        args.append("--lib")
    elif package["bin_name"]:
        args.extend(["--bin", package["bin_name"]])
    if locked:
        args.append("--locked")
    args.extend(["--jobs", "1", "--ignore-rust-version"])
    if extra:
        args.extend(extra)
    return args


def rustflags_with(*attrs: str) -> str:
    flags = [f"-Zcrate-attr=feature({attr})" for attr in attrs]
    return " ".join(flags)


MSRV_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_-]+)@(?P<ver>\d+\.\d+\.\d+)\s+requires rustc"
)


def parse_semver(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2] if len(parts) > 2 else 0)


def crates_io_older_versions(name: str, current: str, limit: int = 8) -> list[str]:
    url = f"https://crates.io/api/v1/crates/{name}/versions"
    request = urllib.request.Request(url, headers={"User-Agent": "rapx-unsoundaudit-msrv/1"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        major, minor, patch = parse_semver(current)
        fallback = []
        if patch > 0:
            fallback.append(f"{major}.{minor}.0")
        if minor > 0:
            fallback.append(f"{major}.{minor - 1}.0")
        return fallback[:limit]
    current_tuple = parse_semver(current)
    older: list[str] = []
    for item in payload.get("versions") or []:
        if item.get("yanked"):
            continue
        ver = str(item.get("num") or "")
        try:
            parsed = parse_semver(ver)
        except (TypeError, ValueError):
            continue
        if parsed < current_tuple:
            older.append(ver)
        if len(older) >= limit:
            break
    return older


def cargo_update_precise(
    cargo: Path,
    cwd: Path,
    env: dict[str, str],
    crate: str,
    version: str,
    log: Path,
) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        stream.write(f"\n# cargo update -p {crate} --precise {version}\n".encode())
        result = subprocess.run(
            [str(cargo), "update", "-p", crate, "--precise", version],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode == 0


def msrv_downgrade_plan(
    log_text: str,
    older_versions_fn=None,
) -> list[tuple[str, str]]:
    """Map a compiler log to generic (crate, older-version) pins. No repo names."""
    lookup = older_versions_fn or crates_io_older_versions
    plan: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in MSRV_RE.finditer(log_text):
        crate = match.group("name")
        version = match.group("ver")
        key = f"{crate}@{version}"
        if key in seen:
            continue
        seen.add(key)
        older_list = lookup(crate, version)
        if older_list:
            plan.append((crate, older_list[0]))
    return plan


def try_msrv_downgrade(
    cargo: Path,
    cwd: Path,
    env: dict[str, str],
    log_text: str,
    logs: Path,
    package_name: str,
) -> list[str]:
    applied: list[str] = []
    for crate, older in msrv_downgrade_plan(log_text):
        ok = cargo_update_precise(
            cargo,
            cwd,
            env,
            crate,
            older,
            logs / f"{package_name}.msrv-{crate}-{older}.log",
        )
        if ok:
            applied.append(f"{crate}->{older}")
    return applied


HARD_FAIL_REASONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"404 Not Found|failed to authenticate", re.I), "git_dep"),
    (re.compile(r"failed to get `[^`]+` as a dependency[\s\S]{0,800}(?:git2|github\\.com|gitlab\\.com|revspec)", re.I), "git_dep"),
    (re.compile(r"can't find crate for `std`|target may not be installed", re.I), "missing_std_or_target"),
    (re.compile(r"linking with `cc` failed|library not found|pkg-config.*not found|could not find native static library", re.I), "native_lib"),
    (re.compile(r"can only be built for the `windows`|on windows only|requires windows", re.I), "windows_only"),
    (re.compile(r"error: failed to parse manifest", re.I), "bad_manifest"),
    (re.compile(r"feature `edition2024` is required", re.I), "edition2024"),
    (re.compile(r"failed to read `.+Cargo\.toml`.+No such file or directory", re.S | re.I), "missing_path_dep"),
    (re.compile(r"couldn't read `.+`: No such file or directory", re.I), "missing_source_file"),
)

REPO_LEVEL_HARD_FAIL = {
    "git_dep",
    "missing_std_or_target",
    "native_lib",
    "windows_only",
    "bad_manifest",
    "edition2024",
    "missing_path_dep",
    "repo_wall",
    "oom_killed",
}



STALE_FEATURE_TO_CRATE = {
    "proc_macro_span_shrink": "proc-macro2",
    "stdsimd": "ahash",
}


def try_stale_feature_update(
    cargo: Path,
    cwd: Path,
    env: dict[str, str],
    log_text: str,
    logs: Path,
    package_name: str,
) -> list[str]:
    """Unlock crates that enable rustc features removed from the pinned nightly."""
    if not log_text:
        return []
    features = set(re.findall(r"unknown feature `([^`]+)`", log_text))
    crates: list[str] = []
    for feat in features:
        crate = STALE_FEATURE_TO_CRATE.get(feat)
        if crate and crate not in crates:
            crates.append(crate)
    applied: list[str] = []
    for crate in crates:
        log_path = logs / f"{package_name}.update-{crate}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as stream:
            stream.write(f"\n# cargo update -p {crate}\n".encode())
            result = subprocess.run(
                [str(cargo), "update", "-p", crate],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            applied.append(crate)
    return applied


def hard_control_failure(log_text: str) -> str | None:
    """Errors that MSRV/lock/feature knobs cannot repair. No repository slugs."""
    if not log_text:
        return None
    for pattern, reason in HARD_FAIL_REASONS:
        if pattern.search(log_text):
            return reason
    return None


def compile_knob_ladder() -> list[tuple[str, bool, list[str], dict[str, str]]]:
    """Shared RAP rustc compile unblocks. Must not mention repository slugs."""
    return [
        ("unlocked", False, [], {}),
        ("no-default-features", False, ["--no-default-features"], {}),
        (
            "crate-attr-freeze-defaults",
            False,
            [],
            {"RUSTFLAGS": rustflags_with("freeze")},
        ),
        (
            "crate-attr-freeze",
            False,
            ["--no-default-features"],
            {"RUSTFLAGS": rustflags_with("freeze")},
        ),
        (
            "crate-attr-common",
            False,
            ["--no-default-features"],
            {
                "RUSTFLAGS": rustflags_with(
                    "freeze",
                    "generic_const_exprs",
                ),
                "RUSTC_BOOTSTRAP": "1",
            },
        ),
        (
            "allow-invalid-ref-cast",
            False,
            ["--no-default-features"],
            {
                "RUSTFLAGS": (
                    "-A invalid_reference_casting -A dangerous_implicit_autorefs"
                )
            },
        ),
        (
            "allow-implicit-autoref",
            False,
            [],
            {
                "RUSTFLAGS": (
                    "-A invalid_reference_casting -A dangerous_implicit_autorefs"
                )
            },
        ),
    ]


def failed_control_packages(previous: dict[str, Any] | None) -> list[str]:
    if not previous:
        return []
    return [
        str(pr.get("package"))
        for pr in previous.get("package_results") or []
        if pr.get("status") == "control_failed" and pr.get("package")
    ]


def classify_resume(
    previous: dict[str, Any] | None,
    *,
    retry_timeouts: bool,
    retry_empty_miss: bool,
    retry_partial: bool,
) -> tuple[str, list[str] | None]:
    """Return ('skip'|'scan', only_packages). Pure journal policy."""
    if not previous or not already_terminal(previous.get("status")):
        return "scan", None
    if previous.get("status") == "oom_killed":
        return "skip", None
    retryable = {"scan_timeout", "control_failed", "error", "scan_failed", "no_package"}
    failed_pkgs = failed_control_packages(previous)
    empty_miss = (
        retry_empty_miss
        and previous.get("status") == "miss"
        and not (previous.get("n_active") or 0)
    )
    timeout_retry = retry_timeouts and previous.get("status") in retryable
    hard_failed = [
        pr
        for pr in previous.get("package_results") or []
        if pr.get("status") == "control_failed"
        and pr.get("control_hard_fail")
        and pr.get("control_hard_fail") != "repo_wall"
    ]
    all_hard = bool(failed_pkgs) and len(hard_failed) == len(failed_pkgs)
    partial_retry = (
        retry_partial
        and previous.get("status") in {"miss", "hit"}
        and bool(failed_pkgs)
        and not all_hard
    )
    if all_hard and previous.get("status") == "control_failed":
        return "skip", None
    if timeout_retry or empty_miss:
        return "scan", None
    if partial_retry:
        return "scan", failed_pkgs
    return "skip", None



def select_control_knobs(first_text: str) -> list[tuple[str, bool, list[str], dict[str, str]]]:
    """Pick compile knobs for a failed control log. Keep crate-attr-* on rustc errors."""
    knobs = compile_knob_ladder()
    rustc_errored = bool(
        first_text
        and ("could not compile" in first_text or "error[E" in first_text)
        and not MSRV_RE.search(first_text)
    )
    if rustc_errored:
        keep = {
            "unlocked",
            "no-default-features",
            "allow-invalid-ref-cast",
            "allow-implicit-autoref",
            "crate-attr-freeze-defaults",
            "crate-attr-freeze",
            "crate-attr-common",
        }
        knobs = [row for row in knobs if row[0] in keep]
    return knobs


def try_control_knobs(
    cargo: Path,
    package: dict[str, str],
    pkg_cwd: Path,
    control_env: dict[str, str],
    logs: Path,
    first_log: Path | None = None,
) -> tuple[ProcessResult | None, bool, list[str], dict[str, str]]:
    """Retry cargo check with RAP-compatible compile unblocks (no AST)."""
    first_text = ""
    if first_log and first_log.exists():
        try:
            first_text = first_log.read_text(errors="replace")
        except OSError:
            first_text = ""
    hard = hard_control_failure(first_text)
    if hard:
        return None, False, [], {"control_hard_fail": hard}
    stale = try_stale_feature_update(
        cargo, pkg_cwd, control_env, first_text, logs, package["name"]
    )
    if stale:
        log_path = logs / f"{package['name']}.control-stale-feature.log"
        result = run_limited(
            [str(cargo), "check", *cargo_args(package, False, ["--no-default-features"])],
            cwd=pkg_cwd,
            environment=control_env,
            wall_time_seconds=KNOB_SECONDS,
            physical_footprint_limit_bytes=PHYSICAL_LIMIT,
            log_path=log_path,
        )
        if result.classification == "success" and result.returncode == 0:
            return result, False, ["--no-default-features"], {}
    if first_text and MSRV_RE.search(first_text):
        applied = try_msrv_downgrade(
            cargo, pkg_cwd, control_env, first_text, logs, package["name"]
        )
        if applied:
            log_path = logs / f"{package['name']}.control-msrv-first.log"
            result = run_limited(
                [str(cargo), "check", *cargo_args(package, False, ["--no-default-features"])],
                cwd=pkg_cwd,
                environment=control_env,
                wall_time_seconds=KNOB_SECONDS,
                physical_footprint_limit_bytes=PHYSICAL_LIMIT,
                log_path=log_path,
            )
            if result.classification == "success" and result.returncode == 0:
                return result, False, ["--no-default-features"], {}
    knobs = select_control_knobs(first_text)
    last: ProcessResult | None = None
    log_blobs: list[str] = []
    for name, locked, extra, env_extra in knobs:
        env = control_env.copy()
        env.update(env_extra)
        log_path = logs / f"{package['name']}.control-{name}.log"
        result = run_limited(
            [str(cargo), "check", *cargo_args(package, locked, extra)],
            cwd=pkg_cwd,
            environment=env,
            wall_time_seconds=KNOB_SECONDS,
            physical_footprint_limit_bytes=PHYSICAL_LIMIT,
            log_path=log_path,
        )
        last = result
        try:
            blob = log_path.read_text(errors="replace")
            log_blobs.append(blob)
        except OSError:
            blob = ""
        if result.classification == "success" and result.returncode == 0:
            return result, locked, extra, env_extra
        hard = hard_control_failure(blob)
        if hard:
            return last, False, [], {"control_hard_fail": hard}

    applied = try_msrv_downgrade(
        cargo,
        pkg_cwd,
        control_env,
        "\n".join(log_blobs),
        logs,
        package["name"],
    )
    if applied:
        env = control_env.copy()
        log_path = logs / f"{package['name']}.control-msrv.log"
        result = run_limited(
            [str(cargo), "check", *cargo_args(package, False, ["--no-default-features"])],
            cwd=pkg_cwd,
            environment=env,
            wall_time_seconds=KNOB_SECONDS,
            physical_footprint_limit_bytes=PHYSICAL_LIMIT,
            log_path=log_path,
        )
        last = result
        if result.classification == "success" and result.returncode == 0:
            return result, False, ["--no-default-features"], {}
    return last, False, [], {}


def already_terminal(status: str | None) -> bool:
    return status in {
        "hit",
        "miss",
        "control_failed",
        "scan_timeout",
        "scan_failed",
        "manifest_missing",
        "revision_mismatch",
        "no_package",
        "skipped_waterfox",
        "oom_killed",
    }


def scan_repo(
    *,
    entry: dict[str, Any],
    issues: list[dict[str, Any]],
    src_root: Path,
    work_root: Path,
    cargo: Path,
    rustc: Path,
    cargo_rapx: Path,
    rust_sysroot: Path,
    cargo_home: Path,
    cargo_commit: str,
    rustc_commit: str,
    only_packages: list[str] | None = None,
) -> dict[str, Any]:
    repo = entry["repository"]
    src = src_root / repo
    record: dict[str, Any] = {
        "repository": repo,
        "expected_sha": entry.get("sha"),
        "issue_urls": [i["issue_url"] for i in issues],
        "src": str(src),
    }
    if not src.exists():
        record["status"] = "missing_source"
        return record
    head = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    actual = (head.stdout or "").strip()
    record["actual_sha"] = actual
    if entry.get("sha") and actual and actual != entry["sha"]:
        record["status"] = "revision_mismatch"
        return record

    case_dir = work_root / repo.replace("/", "__")
    logs = case_dir / "logs"
    mvp_root = case_dir / "mvp"
    receipts = case_dir / "receipts"
    for path in (logs, mvp_root, receipts):
        path.mkdir(parents=True, exist_ok=True)

    base = sanitized_environment()
    base.update(host_scan_environment(rustc, rust_sysroot, cargo, cargo_home, rustc_commit, cargo_commit))

    ensure_git_submodules(src, logs / "metadata.log")
    packages = select_packages(src, cargo, base, logs / "metadata.log", issues=issues)
    if only_packages:
        wanted = set(only_packages)
        filtered = [p for p in packages if p["name"] in wanted]
        if filtered:
            packages = filtered
    if not packages:
        record["status"] = "no_package"
        return record
    record["packages"] = [p["name"] for p in packages]

    issue_matches = []
    package_rows = []
    any_timeout = False
    any_complete = False
    combined_active: list[dict[str, Any]] = []
    combined_unestablished: list[dict[str, Any]] = []
    repo_started = time.monotonic()
    for package in packages:
        if time.monotonic() - repo_started > REPO_WALL_SECONDS:
            package_rows.append(
                {
                    "package": package["name"],
                    "status": "control_failed",
                    "control_hard_fail": "repo_wall",
                }
            )
            continue
        pkg_cwd = Path(package["cwd"])
        locked = (pkg_cwd / "Cargo.lock").exists() or (src / "Cargo.lock").exists()
        if not locked:
            lock_log = logs / f"{package['name']}.lock.log"
            with lock_log.open("wb") as stream:
                try:
                    subprocess.run(
                        [str(cargo), "generate-lockfile", "--manifest-path", package["manifest"]],
                        cwd=pkg_cwd,
                        env=base,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=KNOB_SECONDS,
                    )
                except subprocess.TimeoutExpired:
                    stream.write(b"generate-lockfile timeout\n")
            locked = (pkg_cwd / "Cargo.lock").exists() or (src / "Cargo.lock").exists()

        pkg_target = case_dir / f"target-{package['name']}"
        mvp = mvp_root / package["name"]
        shutil.rmtree(pkg_target, ignore_errors=True)
        shutil.rmtree(mvp, ignore_errors=True)
        pkg_target.mkdir(parents=True, exist_ok=True)
        mvp.mkdir(parents=True, exist_ok=True)

        control_env = base.copy()
        control_env["CARGO_TARGET_DIR"] = str(pkg_target)
        control = run_limited(
            [str(cargo), "check", *cargo_args(package, locked)],
            cwd=pkg_cwd,
            environment=control_env,
            wall_time_seconds=CONTROL_SECONDS,
            physical_footprint_limit_bytes=PHYSICAL_LIMIT,
            log_path=logs / f"{package['name']}.control.log",
        )
        pkg_row: dict[str, Any] = {
            "package": package["name"],
            "control": {
                "returncode": control.returncode,
                "elapsed_seconds": round(control.elapsed_seconds, 3),
                "classification": control.classification,
            },
        }
        extra_cargo: list[str] = []
        extra_env: dict[str, str] = {}
        lock_backups: list[tuple[Path, Path]] = []
        for lock in (pkg_cwd / "Cargo.lock", src / "Cargo.lock"):
            if lock.exists() and lock not in {item[0] for item in lock_backups}:
                bak = lock.with_suffix(lock.suffix + ".rapxbak")
                shutil.copy2(lock, bak)
                lock_backups.append((lock, bak))
        if control.classification == "oom_killed":
            pkg_row["status"] = "oom_killed"
            pkg_row["control_hard_fail"] = "oom_killed"
            package_rows.append(pkg_row)
            shutil.rmtree(pkg_target, ignore_errors=True)
            for lock, bak in lock_backups:
                try:
                    shutil.move(str(bak), str(lock))
                except OSError:
                    pass
            break
        if control.classification != "success" or control.returncode != 0:
            recovered, locked, extra_cargo, extra_env = try_control_knobs(
                cargo,
                package,
                pkg_cwd,
                control_env,
                logs,
                first_log=logs / f"{package['name']}.control.log",
            )
            hard_fail = extra_env.pop("control_hard_fail", None) if extra_env else None
            pkg_row["control_knobs"] = {
                "extra_cargo": extra_cargo,
                "extra_env": extra_env,
                "classification": None if recovered is None else recovered.classification,
                "returncode": None if recovered is None else recovered.returncode,
            }
            if hard_fail:
                pkg_row["control_hard_fail"] = hard_fail
            if recovered is not None and recovered.classification == "oom_killed":
                pkg_row["status"] = "oom_killed"
                pkg_row["control_hard_fail"] = "oom_killed"
                package_rows.append(pkg_row)
                shutil.rmtree(pkg_target, ignore_errors=True)
                for lock, bak in lock_backups:
                    try:
                        shutil.move(str(bak), str(lock))
                    except OSError:
                        pass
                break
            if recovered is not None and recovered.classification == "success" and recovered.returncode == 0:
                control = recovered
            else:
                pkg_row["status"] = "control_failed"
                package_rows.append(pkg_row)
                shutil.rmtree(pkg_target, ignore_errors=True)
                for lock, bak in lock_backups:
                    try:
                        shutil.move(str(bak), str(lock))
                    except OSError:
                        pass
                if hard_fail in REPO_LEVEL_HARD_FAIL:
                    break
                continue

        scan_env = base.copy()
        scan_env.update(extra_env)
        scan_env.update(
            {
                "CARGO_TARGET_DIR": str(pkg_target),
                "UNSOUND_SCANNER_PROJECT_ROOT": str(src),
                "RAP_UNSOUNDAUDIT_OUTPUT": str(mvp),
                "UNSOUND_SCANNER_RAP_JSON_DIR": str(receipts / package["name"]),
            }
        )
        # Keep dep artifacts from control; dirty only this package so RAP actually runs.
        subprocess.run(
            [str(cargo), "clean", "-p", package["name"], "--manifest-path", package["manifest"]],
            cwd=pkg_cwd,
            env=scan_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        (receipts / package["name"]).mkdir(parents=True, exist_ok=True)
        scan = run_limited(
            [
                str(cargo_rapx),
                "rapx",
                "-unsoundaudit",
                "--",
                *cargo_args(package, locked, extra_cargo),
            ],
            cwd=pkg_cwd,
            environment=scan_env,
            wall_time_seconds=SCAN_SECONDS,
            physical_footprint_limit_bytes=PHYSICAL_LIMIT,
            log_path=logs / f"{package['name']}.scan.log",
        )
        pkg_row["scan"] = {
            "returncode": scan.returncode,
            "elapsed_seconds": round(scan.elapsed_seconds, 3),
            "classification": scan.classification,
            "max_physical_footprint_bytes": scan.max_physical_footprint_bytes,
            "max_rss_bytes": scan.max_rss_bytes,
        }
        if scan.classification == "oom_killed":
            pkg_row["status"] = "oom_killed"
            package_rows.append(pkg_row)
            shutil.rmtree(pkg_target, ignore_errors=True)
            for lock, bak in lock_backups:
                try:
                    shutil.move(str(bak), str(lock))
                except OSError:
                    pass
            break
        if scan.classification == "timeout":
            pkg_row["status"] = "scan_timeout"
            any_timeout = True
        else:
            summary = summarize_mvp(mvp)
            pkg_row["mvp"] = {
                "n_candidates": summary["n_candidates"],
                "n_families": summary["n_families"],
                "n_unestablished": summary["n_unestablished"],
                "n_suppressed": summary["n_suppressed"],
                "complete": summary["complete"],
                "semantics": summary["semantics"],
                "accounting": summary["accounting"],
                "active_preview": summary["active_preview"],
            }
            if scan.returncode != 0 and not (
                summary.get("complete") and (mvp / "candidates.jsonl").exists()
            ):
                pkg_row["status"] = "scan_failed"
            else:
                pkg_row["status"] = "scanned"
                any_complete = True
                combined_active.extend(summary["active"])
                combined_unestablished.extend(summary.get("unestablished") or [])
        package_rows.append(pkg_row)
        shutil.rmtree(pkg_target, ignore_errors=True)
        for lock, bak in lock_backups:
            try:
                shutil.move(str(bak), str(lock))
            except OSError:
                pass

    for issue in issues:
        matched = match_issue(issue, combined_active)
        unest_matched = match_rows(issue, combined_unestablished)
        issue_matches.append(
            {
                "issue_url": issue["issue_url"],
                "title": issue.get("title"),
                "hit": matched["hit"],
                "unestablished_hit": unest_matched["hit"],
                "tokens": matched["tokens"][:12],
                "hits": matched["hits"],
                "unestablished_hits": unest_matched["hits"][:4],
            }
        )

    record["package_results"] = package_rows
    record["issue_matches"] = issue_matches
    record["n_active"] = len(combined_active)
    record["n_issue_hits"] = sum(1 for row in issue_matches if row["hit"])
    if any(p.get("status") == "oom_killed" for p in package_rows):
        record["status"] = "oom_killed"
    elif any(row["hit"] for row in issue_matches):
        record["status"] = "hit"
    elif any_complete:
        record["status"] = "miss"
    elif any_timeout:
        record["status"] = "scan_timeout"
    elif package_rows and all(p.get("status") == "control_failed" for p in package_rows):
        record["status"] = "control_failed"
    else:
        record["status"] = "scan_failed"
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--cargo-rapx", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-repos", type=Path, help="File with one repository full name per line")
    parser.add_argument("--cargo-home", type=Path)
    parser.add_argument(
        "--resume-journal",
        type=Path,
        help="Skip repositories already terminal in this journal (default: output-root/scan_journal.jsonl)",
    )
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="Re-scan repos whose last status is scan_timeout, control_failed, error, or scan_failed",
    )
    parser.add_argument(
        "--retry-empty-miss",
        action="store_true",
        help="Re-scan miss repos that produced zero active findings (RAP likely skipped rebuild)",
    )
    parser.add_argument(
        "--retry-partial",
        action="store_true",
        help="Re-scan repos that already have some findings but still have control_failed packages",
    )
    args = parser.parse_args()

    rustc = args.rustc.resolve(strict=True)
    cargo = args.cargo.resolve(strict=True)
    cargo_rapx = args.cargo_rapx.resolve(strict=True)
    args.repo_root = args.repo_root.resolve(strict=True)
    args.src_root = args.src_root.resolve(strict=True)
    rust_sysroot = Path(subprocess.check_output([str(rustc), "--print", "sysroot"], text=True).strip())
    preflight = sanitized_environment()
    preflight.update(host_library_path(rust_sysroot))
    rustc_commit = tool_commit(rustc, preflight, "rustc")
    cargo_commit = tool_commit(cargo, preflight, "Cargo")
    if (rustc_commit, cargo_commit) not in ALLOWED_TOOLCHAINS:
        raise SystemExit(
            f"toolchain mismatch: rustc={rustc_commit} cargo={cargo_commit}"
        )

    roster_dir = args.repo_root / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1"
    plan = load_json(roster_dir / "clone_plan.json")
    revisions = {row["repository"]: row for row in load_json(args.src_root / "checked_out_revisions.json")}
    obligations: dict[str, list[str]] = {}
    instances_path = args.repo_root / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl"
    for instance in load_jsonl(instances_path):
        url = instance.get("issue_url")
        text = instance.get("repair_obligation_zh") or ""
        if url and text:
            obligations.setdefault(str(url), []).append(str(text))
    bodies: dict[str, str] = {}
    for row in load_jsonl(roster_dir / "issue_bodies.jsonl"):
        url = str(row.get("issue_url") or "")
        if url:
            bodies[url] = str(row.get("body") or "")
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for repo in plan["repositories"]:
        if repo["repository"] not in revisions:
            continue
        issues = []
        for issue in repo["issues"]:
            extra = dict(issue)
            extra["repair_obligation_zh"] = " ".join(obligations.get(issue["issue_url"], []))
            extra["body"] = bodies.get(issue["issue_url"], "")
            issues.append(extra)
        by_repo[repo["repository"]] = issues

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cargo_home = (args.cargo_home or (output_root / "cargo-home")).resolve()
    cargo_home.mkdir(parents=True, exist_ok=True)
    journal = output_root / "scan_journal.jsonl"
    resume_journal = (args.resume_journal or journal).resolve()
    done = latest_journal(resume_journal)

    repos = [name for name in sorted(by_repo) if name != "BrowserWorks/waterfox"]
    if args.only_repos:
        wanted = {
            line.strip()
            for line in args.only_repos.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        repos = [name for name in repos if name in wanted]
        missing = sorted(wanted - set(repos) - {"BrowserWorks/waterfox", "rustsec/advisory-db"})
        if missing:
            print("only-repos not in clone plan:", missing, flush=True)
    if args.limit:
        repos = repos[: args.limit]

    counts: dict[str, int] = {}
    for repo in repos:
        previous = done.get(repo)
        only_packages = None
        action, only_packages = classify_resume(
            previous,
            retry_timeouts=args.retry_timeouts,
            retry_empty_miss=args.retry_empty_miss,
            retry_partial=args.retry_partial,
        )
        if action == "skip":
            counts["already_done"] = counts.get("already_done", 0) + 1
            continue
        free = shutil.disk_usage(output_root).free
        if free < MIN_FREE_BYTES:
            append_jsonl(journal, {"repository": repo, "status": "stopped_low_disk", "free_bytes": free})
            counts["stopped_low_disk"] = counts.get("stopped_low_disk", 0) + 1
            break
        print(f"SCAN {repo}", flush=True)
        (output_root / "heartbeat.txt").write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} repo={repo}\n",
            encoding="utf-8",
        )
        started = time.time()
        try:
            record = scan_repo(
                entry=revisions[repo],
                issues=by_repo[repo],
                src_root=args.src_root,
                work_root=output_root / "work",
                cargo=cargo,
                rustc=rustc,
                cargo_rapx=cargo_rapx,
                rust_sysroot=rust_sysroot,
                cargo_home=cargo_home,
                cargo_commit=cargo_commit,
                rustc_commit=rustc_commit,
                only_packages=only_packages,
            )
        except Exception as error:  # noqa: BLE001
            record = {"repository": repo, "status": "error", "error": str(error)}
        record["elapsed_seconds"] = round(time.time() - started, 3)
        append_jsonl(journal, record)
        counts[record.get("status") or "unknown"] = counts.get(record.get("status") or "unknown", 0) + 1
        print(
            f"  {record.get('status')} hits={record.get('n_issue_hits', 0)} "
            f"active={record.get('n_active', 0)} {record['elapsed_seconds']}s",
            flush=True,
        )

    latest = latest_journal(journal)
    issue_hit = 0
    issue_total = 0
    covered = []
    missed = []
    for repo, row in latest.items():
        for item in row.get("issue_matches") or []:
            issue_total += 1
            if item.get("hit"):
                issue_hit += 1
                covered.append(item["issue_url"])
            elif row.get("status") in {"hit", "miss"}:
                missed.append(item["issue_url"])
    summary = {
        "counts": counts,
        "repos_terminal": len(latest),
        "issue_total_in_journal": issue_total,
        "issue_hits": issue_hit,
        "covered_issue_urls": covered,
        "missed_complete_scan_issue_urls": missed,
        "free_bytes": shutil.disk_usage(output_root).free,
    }
    write_json(output_root / "scan_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
