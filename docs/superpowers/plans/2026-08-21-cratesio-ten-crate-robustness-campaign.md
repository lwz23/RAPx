# crates.io Ten-Crate Robustness Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze ten recent high-download crates.io library packages with package-local unsafe Rust, run the current RAP P1–P6 analyzer under a closed serial protocol, review every active family, audit deterministic suppressed/unestablished samples, and produce reproducible robustness and precision evidence.

**Architecture:** Add a Python standard-library campaign toolchain with four focused boundaries: common closed contracts, official crates.io roster freezing, process-controlled RAP execution, and deterministic family review/reporting. Reuse the existing v3 receipt/MVP validators and macOS process-group resource monitor rather than modifying analyzer semantics. Keep large package, target, and raw scan artifacts outside Git; bind them with tracked hashes, manifests, reviews, and reports.

**Tech Stack:** Python 3 standard library (`argparse`, `hashlib`, `json`, `tarfile`, `tomllib`, `urllib`), Cargo/rustup, `nightly-2026-08-14`, existing `rapx-perf`, `rap-unit-v3`, v3 MVP JSONL, macOS `proc_pid_rusage`, crates.io HTTP API, JSON/CSV/Markdown.

---

## File map

| Path | Responsibility |
|---|---|
| `scripts/unsoundaudit-v3/cratesio_campaign_common.py` | Closed keys, canonical JSON, hashing, environment sanitization, stable ranking, and strict manifest loading |
| `scripts/unsoundaudit-v3/select_cratesio_campaign.py` | crates.io metadata/archive retrieval, safe extraction, latest-stable validation, unsafe-source admission, and immutable roster generation |
| `scripts/unsoundaudit-v3/run_cratesio_campaign.py` | Package preparation, lock/fetch, control build, resource-limited RAP scan, receipt/MVP validation, terminal state journal, and deterministic reruns |
| `scripts/unsoundaudit-v3/review_cratesio_campaign.py` | Active census, suppressed/unestablished sampling, review validation, metrics, CSV, JSON, and Markdown report generation |
| `scripts/unsoundaudit-v3/test_cratesio_campaign.py` | No-network/no-build contract tests for selection, execution, sampling, metrics, and reporting |
| `tests/unsoundaudit-v3/cratesio_campaign_policy.json` | Frozen ten-domain candidate pool, current toolchain commits, execution limits, quotas, and review seed |
| `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json` | Official metadata, package checksums, admission evidence, rejected candidates, and selected roster |
| `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-protocol.json` | Host, toolchain, source-tree, binary, mechanism-pack, command, and evidence-root contract |
| `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-evidence-manifest.json` | Hashes and terminal states for external raw evidence |
| `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.json` | All active-family labels plus deterministic suppressed/unestablished audit labels |
| `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-summary.json` | Machine-readable robustness, coverage, precision, and model-gap results |
| `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.csv` | Flat family review table |
| `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-report.md` | Paper-oriented narrative with explicit inference limits |
| `tests/unsoundaudit-v3/cratesio-bug-repros/` | Created only for active families that survive static review as confirmed or likely bugs |
| `TASK_CONTEXT.md` | Decisions, evidence paths/hashes, failures, counts, and remaining limitations |

Analyzer Rust source is outside this file map. A detector defect preserves the original result and requires a separately approved campaign revision.

### Task 1: Freeze the closed campaign policy and shared contracts

**Files:**
- Create: `tests/unsoundaudit-v3/cratesio_campaign_policy.json`
- Create: `scripts/unsoundaudit-v3/cratesio_campaign_common.py`
- Create: `scripts/unsoundaudit-v3/test_cratesio_campaign.py`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Write RED tests for exact policy keys and domain membership**

Start `test_cratesio_campaign.py` with the same dynamic import convention as the other v3 tests, then add:

```python
EXPECTED_DOMAINS = {
    "byte_buffers", "general_collections", "concurrency_reclamation",
    "synchronization", "zero_copy_serialization", "pod_layout_casting",
    "memory_mapping", "ffi_dynamic_loading", "compact_containers",
    "hashing_runtime",
}

def test_policy_is_closed_and_has_ten_unique_domains(self) -> None:
    policy = self.common.load_policy(
        self.repo / "tests/unsoundaudit-v3/cratesio_campaign_policy.json"
    )
    self.assertEqual(set(policy["candidate_pool"]), EXPECTED_DOMAINS)
    self.assertTrue(all(len(names) == 5 and len(set(names)) == 5
                        for names in policy["candidate_pool"].values()))
    self.assertEqual(policy["wall_time_seconds"], 900)
    self.assertEqual(policy["cargo_jobs"], 1)
    self.assertEqual(policy["suppressed_quota_per_crate"], 3)
    self.assertEqual(policy["unestablished_quota_per_crate"], 5)
```

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
```

Expected: FAIL because the policy and loader do not exist.

- [x] **Step 2: Add the exact policy**

Create `cratesio_campaign_policy.json`:

```json
{
  "schema_version": "rap-unsoundaudit-cratesio-campaign-policy-v1",
  "campaign_id": "cratesio-2026-08-21",
  "selection_date": "2026-08-21",
  "toolchain": {
    "channel": "nightly-2026-08-14",
    "rustc_commit": "ba28ff76f353a722f31c4f3dd2ac4e437d36411b",
    "cargo_commit": "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0",
    "target_triple": "aarch64-apple-darwin"
  },
  "candidate_pool": {
    "byte_buffers": ["bytes", "byteorder", "bstr", "bytesize", "byte-unit"],
    "general_collections": ["hashbrown", "indexmap", "dashmap", "slab", "slotmap"],
    "concurrency_reclamation": ["crossbeam-epoch", "crossbeam-deque", "arc-swap", "flurry", "seize"],
    "synchronization": ["parking_lot_core", "lock_api", "parking_lot", "spin", "atomic-waker"],
    "zero_copy_serialization": ["zerocopy", "rkyv", "postcard", "flatbuffers", "scroll"],
    "pod_layout_casting": ["bytemuck", "safe-transmute", "plain", "ref-cast", "pod"],
    "memory_mapping": ["memmap2", "memmap", "mmap-rs", "fmmap", "region"],
    "ffi_dynamic_loading": ["libloading", "dlopen2", "abi_stable", "libffi", "shared_library"],
    "compact_containers": ["smallvec", "arrayvec", "tinyvec", "heapless", "smallstr"],
    "hashing_runtime": ["ahash", "rustc-hash", "fxhash", "twox-hash", "seahash"]
  },
  "wall_time_seconds": 900,
  "cargo_jobs": 1,
  "features": "default",
  "target_kind": "lib",
  "suppressed_quota_per_crate": 3,
  "unestablished_quota_per_crate": 5,
  "review_seed": "dacd31d8f446692177be93f19314e578f01119d0ef637e80c071ad0cdc3638ec"
}
```

- [x] **Step 3: Implement canonical primitives and strict loading**

Create `cratesio_campaign_common.py` with this public contract:

```python
POLICY_KEYS = {
    "schema_version", "campaign_id", "selection_date", "toolchain",
    "candidate_pool", "wall_time_seconds", "cargo_jobs", "features",
    "target_kind", "suppressed_quota_per_crate",
    "unestablished_quota_per_crate", "review_seed",
}
TERMINAL_STATES = {
    "control_failed", "scan_timeout", "scan_failed",
    "partial_coverage", "complete_exact",
}

class CampaignError(ValueError):
    pass

def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

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
```

Also implement `read_json`, `read_jsonl`, `write_json`, `exact_keys`, `exact_hex`, `tool_commit`, and `sanitized_environment`. Remove every inherited `RAP_*` and `UNSOUND_SCANNER_*` variable plus Cargo/Rust wrapper, target/profile, Rustup, `DYLD_*`, `LD_*`, and preload variables.

- [x] **Step 4: Make the focused and full Python gates pass; commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v \
  scripts/unsoundaudit-v3 'test_*.py'
git diff --check
git add tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  scripts/unsoundaudit-v3/cratesio_campaign_common.py \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
git add -p TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v3): freeze crates.io campaign policy"
```

Expected: all tests pass and no bytecode/cache file is added.

### Task 2: Implement official roster freezing and source admission

**Files:**
- Create: `scripts/unsoundaudit-v3/select_cratesio_campaign.py`
- Modify: `scripts/unsoundaudit-v3/test_cratesio_campaign.py`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Add RED selection tests with injected HTTP responses**

Use in-memory API records and tar.gz archives:

```python
def test_selector_uses_latest_stable_non_yanked_and_highest_download_eligible(self) -> None:
    api = FakeCratesIo({
        "popular-safe": crate_payload("2.0.0", downloads=200, unsafe=False),
        "unsafe-lib": crate_payload("1.4.0", downloads=100, unsafe=True),
    })
    selected = self.selector.select_domain(
        "byte_buffers", ["popular-safe", "unsafe-lib"], api.metadata, api.archive
    )
    self.assertEqual(selected["name"], "unsafe-lib")
    self.assertEqual(selected["version"], "1.4.0")
    self.assertEqual(selected["selection_rank"], 2)

def test_archive_admission_rejects_traversal_symlink_and_test_only_unsafe(self) -> None:
    for archive in (traversal_archive(), symlink_archive(), test_only_unsafe_archive()):
        with self.assertRaises(self.common.CampaignError):
            self.selector.inspect_archive(archive, "sample", "1.0.0")
```

Expected: RED because selector functions are absent.

- [x] **Step 2: Implement fetch, safe extraction, and admission**

Use official endpoints and injectable fetchers:

```python
API = "https://crates.io/api/v1/crates/{name}"
DOWNLOAD = "https://crates.io/api/v1/crates/{name}/{version}/download"

def latest_stable(metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    version = metadata["crate"]["max_stable_version"]
    rows = [row for row in metadata["versions"] if row["num"] == version]
    if len(rows) != 1 or rows[0]["yanked"] is not False or "-" in version:
        raise CampaignError("latest stable release is missing, yanked, or prerelease")
    return version, rows[0]

def safe_members(archive: tarfile.TarFile, expected_root: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        parts = PurePosixPath(member.name).parts
        if not parts or parts[0] != expected_root or ".." in parts:
            raise CampaignError("crate archive escapes its expected root")
        if member.issym() or member.islnk() or member.isdev():
            raise CampaignError("crate archive contains a link or device")
    return members
```

`inspect_archive` verifies `Cargo.toml`, exact package name/version, one library target (explicit `[lib].path` or `src/lib.rs`), and at least one lexical `unsafe` keyword in package-local `.rs` files under the library source root, excluding `tests`, `examples`, `benches`, and `target`. Implement a finite Rust lexical scan that skips nested block comments, line comments, byte/normal/raw strings, and character literals before recognizing identifier tokens; add comment-only and string-only negative tests. Record sorted evidence paths, token count, Rust file count, and physical Rust source lines.

- [x] **Step 3: Implement immutable domain selection**

For every domain fetch all five candidate records, sort by `(-total_downloads, crate_name)`, inspect in order, select the first admitted package, and retain every candidate's metadata/admission disposition. Fail before writing a roster if any domain lacks an admitted package. Require ten unique domains and package names, published/independent checksum equality, stable non-yanked versions, and absence of RAP result fields.

CLI:

```bash
python3 scripts/unsoundaudit-v3/select_cratesio_campaign.py \
  --policy tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  --download-root "$SELECTION_ROOT" \
  --output "$SELECTION_ROOT/roster.json"
```

- [x] **Step 4: Pass no-network tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
git diff --check
git add scripts/unsoundaudit-v3/select_cratesio_campaign.py \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
git add -p TASK_CONTEXT.md
git commit -m "feat(unsoundaudit-v3): add crates.io roster freezer"
```

### Task 3: Implement the fail-closed serial campaign runner

**Files:**
- Create: `scripts/unsoundaudit-v3/run_cratesio_campaign.py`
- Modify: `scripts/unsoundaudit-v3/test_cratesio_campaign.py`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Add RED tests for all terminal states and exact root selection**

Use fake Cargo/RAP executables and temporary package trees:

```python
def test_terminal_state_partition(self) -> None:
    self.assertEqual(run_case(control_exit=1).state, "control_failed")
    self.assertEqual(run_case(scan_classification="timeout").state, "scan_timeout")
    self.assertEqual(run_case(scan_exit=1).state, "scan_failed")
    self.assertEqual(run_case(coverage="partial").state, "partial_coverage")
    self.assertEqual(run_case(coverage="exact").state, "complete_exact")

def test_root_receipt_requires_exact_name_version_lib_target(self) -> None:
    selected = self.runner.select_root_receipt(
        receipts, schema, package_name="sample", package_version="1.2.3",
        crate_name="sample", rustc_commit="a" * 40,
    )
    self.assertEqual(selected["package_id"], expected_package_id)
    with self.assertRaises(self.common.CampaignError):
        self.runner.select_root_receipt(duplicate_receipts, schema, **identity)
```

Also prove environment sanitization, offline locked commands, fresh targets, process-group timeout classification, memory observation without a limit, and continuation after package failure.

- [x] **Step 2: Reuse existing validators and monitor**

Import rather than copy:

```python
from run_stdlib_gate import (
    GateError, process_record, run_limited, validate_development_mvp,
    verify_binary_launch,
)
from validate_receipt import ContractError, read_json, validate_receipt
```

Use `run_limited(..., wall_time_seconds=900, physical_footprint_limit_bytes=None)` for RAP. Record peak physical footprint and RSS. Apply the same 900-second operational timeout to control, mapping timeout/nonzero to `control_failed`.

- [x] **Step 3: Implement package preparation and exact commands**

Each package receives separate `archive`, `source`, `cargo-home`, `control-target`, `scan-target`, `receipts`, `mvp`, and `logs` directories. Verify/re-extract the `.crate`, confirm Cargo metadata name/version/lib identity, generate and hash one lockfile, fetch it, then run:

```python
control_command = [
    str(cargo), "check", "--lib", "--locked", "--offline", "--jobs", "1",
]
scan_command = [
    str(cargo_rapx), "rapx", "-unsoundaudit", "--",
    "--lib", "--locked", "--offline", "--jobs", "1",
]
```

The scan environment sets only exact toolchain paths/commits, `CARGO_HOME`, fresh `CARGO_TARGET_DIR`, `CARGO_BUILD_JOBS=1`, `RAP_CLEAN=false`, package source as `UNSOUND_SCANNER_PROJECT_ROOT`, one receipt directory, one MVP directory, and one performance trace. On macOS reconstruct `DYLD_LIBRARY_PATH` only as the exact current sysroot `lib` after binary launch preflight.

- [x] **Step 4: Validate and cross-check successful scans**

Validate every formal receipt, select exactly one root name/version/crate/lib receipt, run `validate_development_mvp`, cross-check formal/MVP census and active pattern counts, and require four-state closure. A valid lower-bound output becomes `partial_coverage`; malformed/missing/ambiguous output becomes `scan_failed` with retained reason. Atomically journal `campaign.json` after every package and continue through all ten. The output root must not exist at startup.

- [x] **Step 5: Implement deterministic semantic normalization and reruns**

Normalize only declared operational fields: absolute temporary roots, timestamps, durations, sampled memory, and process identifiers. Include receipt semantics plus candidates, families, suppression ledger, and unestablished decisions. After ten first runs, rerun the smallest and largest `complete_exact` package by roster source lines with fresh targets and require byte-identical normalized semantics. If fewer than two qualify, record `not_run_fewer_than_two_complete`.

- [x] **Step 6: Pass focused/full no-build gates and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v \
  scripts/unsoundaudit-v3 'test_*.py'
git diff --check
git add scripts/unsoundaudit-v3/run_cratesio_campaign.py \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
git add -p TASK_CONTEXT.md
git commit -m "feat(unsoundaudit-v3): add crates.io campaign runner"
```

### Task 4: Implement deterministic review sampling and reporting

**Files:**
- Create: `scripts/unsoundaudit-v3/review_cratesio_campaign.py`
- Modify: `scripts/unsoundaudit-v3/test_cratesio_campaign.py`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Add RED tests for the review population**

Create synthetic exact package outputs:

```python
def test_review_template_contains_every_active_family_and_bounded_samples(self) -> None:
    review = self.review.build_template(policy, campaign)
    active = [row for row in review["rows"] if row["population"] == "active"]
    suppressed = [row for row in review["rows"] if row["population"] == "suppressed"]
    unestablished = [row for row in review["rows"] if row["population"] == "unestablished"]
    self.assertEqual({row["family_id"] for row in active}, all_active_ids)
    self.assertLessEqual(len(suppressed), 3 * complete_crate_count)
    self.assertLessEqual(len(unestablished), 5 * complete_crate_count)
    self.assertEqual(review, self.review.build_template(policy, campaign))

def test_inconclusive_is_excluded_from_decidable_precision(self) -> None:
    metrics = self.review.metrics(
        ["confirmed_bug", "likely_bug", "false_positive", "inconclusive"]
    )
    self.assertEqual(metrics["decidable"], 3)
    self.assertEqual(
        metrics["false_positive_fraction"], {"numerator": 1, "denominator": 3}
    )
```

Expected: RED before implementation.

- [x] **Step 2: Implement family grouping and stable sampling**

Consume active `families.jsonl` directly. For suppressed rows, call `family_id_for_candidate` from `run_mechanism_pack.py`, group by structural family ID, and retain disposition/reason evidence. For unestablished rows, hash:

```python
(
    attempted_pattern, public_root, unsafe_site_id, operand_role,
    source.kind, source.identity, primary_reason,
)
```

Within each package stratify by pattern, contract precision, terminal reason, and propagation depth. Order ties with `stable_rank(review_seed, stable_identity)`. Select at most three suppressed and five unestablished groups without transferring unused quota across packages.

- [x] **Step 3: Implement the closed review contract**

Every template row has immutable evidence and mutable review fields:

```json
{
  "population": "active",
  "package": "sample@1.2.3",
  "stable_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "family_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "pattern": "P1",
  "contract_precision": "generic",
  "reason": null,
  "propagation_depth": "inter_procedural",
  "evidence": {},
  "label": null,
  "reviewer": null,
  "rationale": null,
  "reproduction": null,
  "independent_confirmation": "not_requested"
}
```

The displayed IDs are test-fixture values; production rows use the actual validated 64-character lowercase hex identities. Template generation permits null review fields. Final validation requires a reviewer and nonempty rationale for every row. Active labels are exactly `confirmed_bug`, `likely_bug`, `false_positive`, or `inconclusive`; sampled labels are exactly `classification_supported`, `missed_active_witness`, `invalid_safety_proof`, or `inconclusive`.

- [x] **Step 4: Implement metrics and report rendering**

Compute exact counts/fractions and Wilson intervals without dependencies:

```python
def wilson(successes: int, total: int,
           z: float = 1.959963984540054) -> dict[str, float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        (p * (1 - p) + z * z / (4 * total)) / total
    ) / denominator
    return {"low": max(0.0, center - margin),
            "high": min(1.0, center + margin)}
```

Render summary JSON, family CSV, and Markdown from validated review data. State that the active census is purposive and Wilson intervals are exploratory, not crates.io population confidence intervals.

- [x] **Step 5: Pass tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
git diff --check
git add scripts/unsoundaudit-v3/review_cratesio_campaign.py \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
git add -p TASK_CONTEXT.md
git commit -m "feat(unsoundaudit-v3): add crates.io family review tooling"
```

### Task 5: Freeze the real roster, build the current analyzer, and freeze the protocol

**Files:**
- Create: `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json`
- Create: `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-protocol.json`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Resolve and freeze the official roster before any RAP package scan**

```bash
REPO=/Users/liaowenzhao/Documents/RAP重构/RAPx-unsoundaudit-v2
SELECTION_ROOT=$(mktemp -d /private/tmp/rapx-cratesio-selection.XXXXXX)
cd "$REPO"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v3/select_cratesio_campaign.py \
  --policy tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  --download-root "$SELECTION_ROOT" \
  --output "$SELECTION_ROOT/roster.json"
cp "$SELECTION_ROOT/roster.json" \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json
```

Inspect only selection metadata and unsafe-source admission evidence. Confirm ten unique domains/packages, latest stable non-yanked releases, checksum equality, and no RAP result fields.

- [x] **Step 2: Commit the roster before analyzer execution**

```bash
git add docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json
git add -p TASK_CONTEXT.md
git commit -m "docs: freeze crates.io robustness roster"
```

- [x] **Step 3: Build a fresh current `rapx-perf` pair**

```bash
TOOLCHAIN=nightly-2026-08-14
CARGO=$(rustup which --toolchain "$TOOLCHAIN" cargo)
RUSTC=$(rustup which --toolchain "$TOOLCHAIN" rustc)
SYSROOT=$($RUSTC --print sysroot)
BUILD_ROOT=$(mktemp -d /private/tmp/rapx-cratesio-current-build.XXXXXX)
env -u DYLD_LIBRARY_PATH \
  RUSTC="$RUSTC" RUST_SYSROOT="$SYSROOT" CARGO_TARGET_DIR="$BUILD_ROOT/target" \
  "$CARGO" build --manifest-path rapx/Cargo.toml --locked \
  --profile rapx-perf --bins --jobs 1 2>&1 | tee "$BUILD_ROOT/build.log"
BIN_DIR="$BUILD_ROOT/target/rapx-perf"
"$BIN_DIR/rapx" -Vv
shasum -a 256 "$BIN_DIR/cargo-rapx" "$BIN_DIR/rapx" "$BUILD_ROOT/build.log"
```

Require embedded rustc commit `ba28ff76f353a722f31c4f3dd2ac4e437d36411b`. Compute Git HEAD, porcelain status, `git diff --binary -- rapx/src rapx/Cargo.toml rapx/Cargo.lock` SHA-256, and a canonical sorted `(relative path, file SHA-256)` manifest over `rapx/src`, `rapx/Cargo.toml`, and `rapx/Cargo.lock`. Write `$BIN_DIR/rapx-build-profile.json` using the existing `unsoundaudit-v3-rap-binary-profile-v1` exact fields: `profile=rapx-perf`, `opt_level=3`, `debug=1`, `debug_assertions=true`, `overflow_checks=true`, `incremental=false`, `target_cargo_profile=dev`, current Git HEAD, current rustc commit, and both binary hashes. The campaign protocol additionally binds the dirty-diff and source-manifest hashes, so a dirty analyzer tree is not misrepresented as the clean HEAD.

- [x] **Step 4: Admit binaries through the current mechanism pack**

```bash
MECHANISM_ROOT=$(mktemp -d /private/tmp/rapx-cratesio-mechanisms.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v3/run_mechanism_pack.py \
  --repo-root "$REPO" --cargo "$CARGO" --rustc "$RUSTC" \
  --cargo-rapx "$BIN_DIR/cargo-rapx" --output-root "$MECHANISM_ROOT/result"
shasum -a 256 "$MECHANISM_ROOT/result/capability_matrix.json"
```

Expected: all P1–P6 oracles pass with exact coverage; totals remain 50 active, 49 families, 38 pure suppressions, 12 reclassifications, 28 unestablished, and 128 considered. Drift stops before any crates.io scan.

- [x] **Step 5: Write, validate, and commit the protocol**

Bind the roster/policy hashes, source/binary/build/mechanism hashes, host facts, Cargo/rustc versions, native target, default features, library target, one Cargo job, 900-second wall time, no memory limit, serial order, fresh targets, exact root-receipt rule, and external evidence root.

```bash
git add docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-protocol.json
git add -p TASK_CONTEXT.md
git commit -m "docs: freeze crates.io robustness protocol"
```

### Task 6: Execute all ten control builds and RAP scans

**Files:**
- Create externally: `/private/tmp/rapx-cratesio-campaign-2026-08-21-*/`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Start the serial campaign in a fresh evidence root**

```bash
EVIDENCE_ROOT=$(mktemp -d /private/tmp/rapx-cratesio-campaign-2026-08-21.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v3/run_cratesio_campaign.py \
  --repo-root "$REPO" \
  --policy tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  --roster docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json \
  --protocol docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-protocol.json \
  --schema tests/unsoundaudit-v3/rap-unit-v3.schema.json \
  --cargo "$CARGO" --rustc "$RUSTC" --rap-bin-dir "$BIN_DIR" \
  --output-root "$EVIDENCE_ROOT"
```

The runner journals after every package and continues through the roster. Do not replace failures or rerun ordinary cases selectively.

- [x] **Step 2: Verify terminal state and exactness accounting**

Require ten package rows. For each `complete_exact` row verify:

```text
bound_exact + optimized_away_exact == discovered
unbound == ambiguous == elided == coverage_failures == 0
active + pure_suppressed + reclassified + unestablished == considered
```

For failures, verify logs, return codes/signals, elapsed time, and retained partial hashes. Missing/partial output is never zero findings.

- [x] **Step 3: Verify deterministic reruns and evidence identity**

Require one additional fresh scan for smallest/largest complete packages and byte-identical normalized semantics. If fewer than two complete, require `not_run_fewer_than_two_complete`. Hash the campaign journal, per-package resource records, receipts, MVP artifacts, logs, traces, lockfiles, and normalized reruns. Record evidence root, states, performance, coverage, and failures in `TASK_CONTEXT.md`; do not commit large raw data.

### Task 7: Freeze the review population and complete family audits

**Files:**
- Create: `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.json`
- Conditionally create: `tests/unsoundaudit-v3/cratesio-bug-repros/`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Generate and hash-freeze the unlabeled review template**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v3/review_cratesio_campaign.py template \
  --policy tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  --roster docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json \
  --campaign "$EVIDENCE_ROOT/campaign.json" \
  --output docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.json
```

Require every active family, no incomplete-scan family, per-crate quotas, and stable sample hash before reading selected source evidence.

- [x] **Step 2: Review every active family**

Inspect exact published source, representative, public safe root, source, complete witness, sink contract, safety documentation, type/relationship obligation, feature/target conditions, and code dictionary. Assign one active label with falsifiable rationale. Package reputation and expected paper value are not evidence.

- [x] **Step 3: Review suppressed and unestablished samples**

Decide `classification_supported`, `missed_active_witness`, `invalid_safety_proof`, or `inconclusive`. Record recurring gaps by pattern, reason, precision, and depth; do not calculate a false-negative rate.

- [x] **Step 4: Validate surviving bug candidates**

For every confirmed/likely family create:

```text
tests/unsoundaudit-v3/cratesio-bug-repros/{package_name}-{family_id[:12]}/Cargo.toml
tests/unsoundaudit-v3/cratesio-bug-repros/{package_name}-{family_id[:12]}/Cargo.lock
tests/unsoundaudit-v3/cratesio-bug-repros/{package_name}-{family_id[:12]}/src/main.rs
tests/unsoundaudit-v3/cratesio-bug-repros/{package_name}-{family_id[:12]}/README.md
```

Use only a safe caller, exact crate version, explicit contract argument, and control/Miri/sanitizer status. If none survive, record an empty candidate-validation array.

- [x] **Step 5: Validate completed review**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v3/review_cratesio_campaign.py validate \
  --policy tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  --campaign "$EVIDENCE_ROOT/campaign.json" \
  --review docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.json
```

Expected: each active/sample row labeled exactly once and evidence hashes unchanged.

### Task 8: Generate the final evidence manifest and report

**Files:**
- Create: `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-evidence-manifest.json`
- Create: `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-summary.json`
- Create: `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.csv`
- Create: `docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-report.md`
- Modify: `TASK_CONTEXT.md`

- [x] **Step 1: Render machine-readable and narrative outputs**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/unsoundaudit-v3/review_cratesio_campaign.py report \
  --policy tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  --roster docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json \
  --protocol docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-protocol.json \
  --campaign "$EVIDENCE_ROOT/campaign.json" \
  --review docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.json \
  --evidence-manifest docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-evidence-manifest.json \
  --summary docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-summary.json \
  --csv docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.csv \
  --markdown docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-report.md
```

Require per-package/aggregate states, performance, unsafe-site/contract coverage, P1–P6 four-state counts, reason distributions, rerun status, all active labels, sample outcomes, candidate status, exact denominators, Wilson intervals, and limitations.

- [x] **Step 2: Independently recompute totals from raw evidence**

Use a separate short Python command reading only `campaign.json`, JSONL artifacts, and review rows. Match terminal-state, active-family, label, and four-state totals with the generated summary. Any mismatch blocks the report.

- [x] **Step 3: Run final gates**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v \
  scripts/unsoundaudit-v3 'test_*.py'
git diff --check
git status --short
```

Rerun the current mechanism pack with exact campaign binaries and require the protocol's capability-matrix hash. Do not modify analyzer expectations or add allowlists.

- [x] **Step 4: Update context and commit tracked campaign evidence**

Record measured facts, inferences, provisional labels, independent-review status, evidence custody, and unresolved gaps.

```bash
git add scripts/unsoundaudit-v3/cratesio_campaign_common.py \
  scripts/unsoundaudit-v3/select_cratesio_campaign.py \
  scripts/unsoundaudit-v3/run_cratesio_campaign.py \
  scripts/unsoundaudit-v3/review_cratesio_campaign.py \
  scripts/unsoundaudit-v3/test_cratesio_campaign.py \
  tests/unsoundaudit-v3/cratesio_campaign_policy.json \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-roster.json \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-protocol.json \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-evidence-manifest.json \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.json \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-summary.json \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-review.csv \
  docs/unsoundaudit-v3/cratesio-campaign-2026-08-21-report.md
git add -p TASK_CONTEXT.md
git commit -m "docs: report crates.io robustness campaign"
```

If bug reproductions exist, add their exact directories separately. Never add `.crate` archives, extracted packages, Cargo homes, targets, raw code dictionaries, or `/private/tmp` evidence.

## Stop and interpretation rules

- A roster/policy/protocol hash mismatch stops before output creation.
- A mechanism-pack regression stops before the first crates.io RAP scan.
- One package failure does not stop later packages; it remains in the robustness denominator.
- Malformed successful-process output is `scan_failed`, not zero findings.
- Partial coverage is excluded from precision review and remains visible robustness evidence.
- No package is replaced after roster freeze.
- No analyzer source or contract is changed during the frozen campaign.
- All active families in complete scans are reviewed; active sampling is not permitted.
- Suppressed/unestablished audits identify defects and model gaps but do not estimate false-negative prevalence.
- Confirmed/likely labels remain provisional for paper claims until a qualified second reviewer or maintainer confirms them.
- The report describes ten default-feature macOS arm64 library builds only; it makes no crates.io population claim.
