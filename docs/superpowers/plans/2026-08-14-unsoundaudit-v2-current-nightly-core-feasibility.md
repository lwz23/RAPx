# RAPx Current-Nightly Core Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine at bounded cost whether RAPx can compile on the frozen 2026-08-13 nightly and produce useful new or changed P1–P6 families from one current `core` scan.

**Architecture:** Make only the known rustc-private API adaptations, preserve analyzer semantics, and validate them against the existing synthetic suites. Run one development `core` scan and compare its families with the historical population using existing family keys plus a conservative exact-source-tree diff. Stop before `alloc`, `std`, a formal protocol, or a generalized dependency-diff framework unless the current `core` result contains a credible high-value signal.

**Tech Stack:** Rust nightly `2026-08-13`, rustc-private MIR APIs, Cargo `-Z build-std`, existing Python no-build tools, Git, JSON/SHA-256, `rapx-perf`, macOS physical-footprint monitoring.

---

### Task 1: Freeze the bounded current-nightly target

**Files:**
- Create: `tests/unsoundaudit-v2/latest_stdlib_target.json`
- Create: `scripts/unsoundaudit-v2/test_current_nightly_core.py`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Write the failing identity test**

```python
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "tests/unsoundaudit-v2/latest_stdlib_target.json"


class CurrentNightlyTargetTests(unittest.TestCase):
    def test_current_target_is_exact(self) -> None:
        value = json.loads(TARGET.read_text(encoding="utf-8"))
        self.assertEqual(set(value), {
            "schema_version", "nightly", "stable", "historical_baseline",
            "target_triple", "rust_src_tree_sha256",
        })
        self.assertEqual(value["schema_version"],
                         "unsoundaudit-v2-latest-stdlib-target-v1")
        self.assertEqual(value["nightly"]["rustc_commit"],
                         "ba28ff76f353a722f31c4f3dd2ac4e437d36411b")
        self.assertEqual(value["nightly"]["cargo_commit"],
                         "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0")
        self.assertEqual(set(value["nightly"]), {
            "cargo_commit", "channel", "miri_version", "rustc_commit",
        })
        self.assertEqual(value["nightly"]["channel"], "nightly-2026-08-13")
        self.assertEqual(value["nightly"]["miri_version"],
                         "miri 0.1.0 (ba28ff76f3 2026-08-13)")
        self.assertEqual(value["stable"], {
            "cargo_commit": "c980f4866141969fab6254a680546a277789d6f0",
            "channel": "1.97.1",
            "rustc_commit": "8bab26f4f68e0e26f0bb7960be334d5b520ea452",
        })
        self.assertEqual(value["historical_baseline"], {
            "channel": "nightly-2024-10-12",
            "rustc_commit": "1bc403daadbebb553ccc211a0a8eebb73989665f",
        })
        self.assertEqual(value["target_triple"], "aarch64-apple-darwin")
        self.assertEqual(value["rust_src_tree_sha256"], {
            "alloc": "3d40f86c6a4fbfa21b58b89a0d8e8b9029e94afb315597109031dbe58b0cdbcf",
            "core": "c0a1e218b985dbcff02409dcef163588799dd79ccec21b2d8361f0c8f9370707",
            "std": "d758f147c839f3a2f5d672490864f83603d8d1b720ee284b4d493968a9d9945f",
        })


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_current_nightly_core.py
```

Expected: FAIL only because `latest_stdlib_target.json` is missing.

- [ ] **Step 3: Add the exact identity manifest**

```json
{
  "historical_baseline": {
    "channel": "nightly-2024-10-12",
    "rustc_commit": "1bc403daadbebb553ccc211a0a8eebb73989665f"
  },
  "nightly": {
    "cargo_commit": "eb98b54bc9f3c74519f43d066cb3fd02ebc88df0",
    "channel": "nightly-2026-08-13",
    "miri_version": "miri 0.1.0 (ba28ff76f3 2026-08-13)",
    "rustc_commit": "ba28ff76f353a722f31c4f3dd2ac4e437d36411b"
  },
  "rust_src_tree_sha256": {
    "alloc": "3d40f86c6a4fbfa21b58b89a0d8e8b9029e94afb315597109031dbe58b0cdbcf",
    "core": "c0a1e218b985dbcff02409dcef163588799dd79ccec21b2d8361f0c8f9370707",
    "std": "d758f147c839f3a2f5d672490864f83603d8d1b720ee284b4d493968a9d9945f"
  },
  "schema_version": "unsoundaudit-v2-latest-stdlib-target-v1",
  "stable": {
    "cargo_commit": "c980f4866141969fab6254a680546a277789d6f0",
    "channel": "1.97.1",
    "rustc_commit": "8bab26f4f68e0e26f0bb7960be334d5b520ea452"
  },
  "target_triple": "aarch64-apple-darwin"
}
```

- [ ] **Step 4: Verify GREEN and commit**

Run the focused test and full no-build suite, record hashes in `TASK_CONTEXT.md`, then commit. The three source-tree digests are frozen observations of the installed target and are not silently regenerated by a different tree-hash algorithm:

```bash
git add tests/unsoundaudit-v2/latest_stdlib_target.json \
  scripts/unsoundaudit-v2/test_current_nightly_core.py TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v2): freeze current nightly target"
```

### Task 2: Make only the known rustc-private API adaptations

**Files:**
- Modify: `rust-toolchain.toml`
- Modify: `rapx/src/lib.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Preserve the genuine compile RED**

The current probe has 39 errors and log SHA-256 `e0aaff14ed86d9f39e34e108e66961bc7fb8724cc1c89d29b449e131da1453b1`. Reproduce it with a fresh target and `DYLD_LIBRARY_PATH` unset. Expected classes are `E0599=20`, `E0603=7`, `E0308=4`, `E0432=2`, `E0425=2`, `E0023=2`, `E0433=1`, `E0609=1`.

- [ ] **Step 2: Pin the current toolchain**

First require the dated alias to resolve to the frozen commit. Installing the alias is allowed only if it resolves exactly; otherwise stop:

```bash
rustup toolchain install nightly-2026-08-13 --profile minimal \
  --component rustc-dev,rust-src,llvm-tools-preview,rustfmt,miri
rustc +nightly-2026-08-13 -Vv | \
  grep 'commit-hash: ba28ff76f353a722f31c4f3dd2ac4e437d36411b'
```

```toml
[toolchain]
channel = "nightly-2026-08-13"
components = ["rustc-dev", "rust-src", "llvm-tools-preview", "rustfmt", "miri"]
```

Add `extern crate rustc_attr_ir;` to `rapx/src/lib.rs`.

- [ ] **Step 3: Apply mechanical API replacements**

Use `rustc_attr_ir::lang_items::LangItem`, `rustc_abi::ExternAbi`, `rustc_span::{Span, Spanned, Symbol}`, and `rustc_middle::mir::PlaceTy`. Replace:

- `Rvalue::Use(operand)` with `Rvalue::Use(operand, _)`;
- deleted `Rvalue::Len` with the already-present `UnaryOp(UnOp::PtrMetadata, ...)` path only;
- `Const::try_eval_target_usize(...ParamEnv...)` for array lengths with `try_to_target_usize(tcx)`;
- `GenericArg::unpack()` with `GenericArg::as_type()`;
- `sym::mem_size_of` with `Symbol::intern("mem_size_of")`;
- `FileNameDisplayPreference` with `prefer_local_unconditionally()` and `prefer_remapped_unconditionally()`;
- `Abi::C` with `ExternAbi::C`;
- legacy `ParamEnvAnd` query inputs with `TypingEnv::post_analysis(tcx, caller).as_query_input(value)`;
- `Instance::try_resolve(...ParamEnv...)` with the same post-analysis `TypingEnv`.

Replace associated-item queries with exact current metadata, not def-path strings:

```rust
fn associated_impl(tcx: TyCtxt<'_>, did: DefId) -> Option<DefId> {
    tcx.opt_associated_item(did)?.impl_container(tcx)
}

fn associated_trait(tcx: TyCtxt<'_>, did: DefId) -> Option<DefId> {
    let item = tcx.opt_associated_item(did)?;
    match item.container {
        ty::AssocContainer::Trait => Some(tcx.parent(did)),
        ty::AssocContainer::TraitImpl(Ok(trait_item)) => Some(tcx.parent(trait_item)),
        ty::AssocContainer::TraitImpl(Err(_)) | ty::AssocContainer::InherentImpl => None,
    }
}
```

Use `AssocItem::is_method()` for the self-parameter test.

- [ ] **Step 4: Run the stop gate**

Run `cargo check --lib --locked --jobs 1` with a fresh target. If errors remain only because of a missed mechanical occurrence from the listed API families, correct that occurrence. If a new MIR representation requires a new analyzer abstraction or changes an unsupported operation from `Unknown` to proved/safe, stop and revert this task; do not continue to `core`.

- [ ] **Step 5: Run current-nightly unit tests and commit**

Run the complete exact Rust library suite, rustfmt check, and `git diff --check`. The 64-hop and direct/mutual recursion tests must remain present. Commit only if all pass:

```bash
git add rust-toolchain.toml rapx/src/lib.rs \
  rapx/src/analysis/unsoundaudit/mir.rs TASK_CONTEXT.md
git commit -m "build(unsoundaudit-v2): port analyzer to current nightly"
```

### Task 3: Run bounded semantic regression gates

**Files:**
- Modify: `scripts/unsoundaudit-v2/run_fixture_suite.py`
- Modify: `scripts/unsoundaudit-v2/run_challenge_suite.py`
- Modify: `scripts/unsoundaudit-v2/test_contract_hardening.py`
- Modify: `scripts/unsoundaudit-v2/test_challenge_contract.py`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Add an explicit toolchain-contract RED**

Add `--toolchain-contract` to both runners. Its loader accepts only the exact key set from Task 1 and returns the current Cargo/rustc commits. Omission must continue to use `baseline_manifest_v1.json`; this preserves historical runners.

```python
def toolchain_contract(repo_root: Path, path: Path | None) -> tuple[str, str]:
    if path is None:
        return baseline_toolchain(repo_root)
    value = read_json(canonical_existing(path, "toolchain contract"))
    if set(value) != {
        "schema_version", "nightly", "stable", "historical_baseline",
        "target_triple", "rust_src_tree_sha256",
    }:
        fail("toolchain contract drifted")
    return value["nightly"]["cargo_commit"], value["nightly"]["rustc_commit"]
```

Pass that returned tuple into `verify_environment`; do not let it reread the historical manifest:

```python
def verify_environment(
    repo_root: Path,
    cargo: Path,
    rap_bin_dir: Path,
    expected_toolchain: tuple[str, str] | None = None,
) -> tuple[Path, Path]:
    expected_cargo_commit, expected_rustc_commit = (
        expected_toolchain
        if expected_toolchain is not None
        else baseline_toolchain(repo_root)
    )
    # The remaining path, environment, Z3, binary-sibling, and commit checks
    # stay byte-for-byte identical.
```

Both `main()` functions compute `expected_toolchain` once, pass it to
`verify_environment`, and pass `expected_toolchain[1]` to
`verify_receipt_toolchain`. Historical invocations omit the new argument and
therefore retain the frozen 2024 behavior.

Expected RED: missing loader/argument. Tests also reject wrong commits and extra keys.

- [ ] **Step 2: Run one current challenge and fixture pass**

Build metadata-correct `rapx-perf` binaries in a fresh target. Run challenge 27 once and original 27 once with the explicit current target. Require every fixture oracle to pass and every receipt to name `ba28ff...11b`. This feasibility stage does not yet require double-run formal evidence.

- [ ] **Step 3: Compare analyzer profiles on a fixed sentinel set**

Do not repeat all 54 cases for both diagnostic profiles. Run only this fixed six-case sentinel set with debug and stock release analyzers:

```text
challenge: call_wrappers_5_hop_positive
challenge: cfg_diamond_all_paths_guard_negative
challenge: recursion_mutual_positive
original: legacy_p1_param_direct_positive
original: p4_internal_bounds_checked_negative
original: p6_ffi_out_param_positive
```

Require debug/release/rapx-perf equality after excluding only `finding_id` and source-root path prefixes. Rule, source, sink, obligation, validation, and propagation depth must remain equal. Any unexplained missing positive stops the feasibility stage.

- [ ] **Step 4: Commit runner support and evidence**

Run all no-build tests and `git diff --check`, record exact counts/hashes in `TASK_CONTEXT.md`, then commit the four runner/test files and context.

### Task 4: Run one development current-core scan

**Files:**
- Create: `scripts/unsoundaudit-v2/run_current_core_feasibility.py`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/current-core-feasibility/core.normalized.json`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/current-core-feasibility/core.summary.json`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/current-core-feasibility/core.execution.json`
- Modify: `scripts/unsoundaudit-v2/test_current_nightly_core.py`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Add a single-run development runner RED**

The test imports the new runner and requires this exact schedule and unit set:

```python
class CurrentCoreRunnerTests(unittest.TestCase):
    def test_runner_is_one_control_plus_one_core_scan(self) -> None:
        import run_current_core_feasibility as runner

        self.assertEqual(runner.UNIT, "core")
        self.assertEqual(runner.STEPS, (("control", 0), ("scan", 1)))
        self.assertEqual(runner.WALL_LIMIT_SECONDS, 900)
        self.assertEqual(runner.PHYSICAL_LIMIT_BYTES, 8 * 1024**3)

    def test_runner_never_names_alloc_or_std_outputs(self) -> None:
        import run_current_core_feasibility as runner

        self.assertNotIn("alloc", runner.output_names())
        self.assertNotIn("std", runner.output_names())
```

Expected RED: `ModuleNotFoundError` for the absent runner.

- [ ] **Step 2: Implement the minimal non-formal runner**

`run_current_core_feasibility.py` reuses, rather than copies, these existing functions:

```python
from noise_audit import family_id, family_key
from run_fixture_suite import canonical_existing, toolchain_contract, verify_environment
from run_stdlib_pilot import (
    cargo_arguments,
    run_limited,
    scan_roots,
    select_unit_receipt,
    stdlib_source_root,
    write_driver,
)

UNIT = "core"
STEPS = (("control", 0), ("scan", 1))
WALL_LIMIT_SECONDS = 900
PHYSICAL_LIMIT_BYTES = 8 * 1024**3


def output_names() -> tuple[str, ...]:
    return ("core.normalized.json", "core.summary.json", "core.execution.json")
```

The runner accepts the repository, current target manifest, schema, Cargo,
`rapx-perf` binary directory, rust sysroot, and a nonexistent output root. It
uses a temporary core-only build-std contract, calls `verify_environment` with
the manifest's exact current toolchain tuple, performs exactly one fresh
control and one fresh scan through `run_limited`, and rejects timeout, physical
footprint excess, zero/multiple selected receipts, or a receipt not bound to
`ba28ff...11b`. It writes this closed normalized shape directly from the
validated manifest and selected receipt:

```python
normalized = {
    "schema_version": "unsoundaudit-v2-current-core-feasibility-v1",
    "unit": "core",
    "target_manifest_sha256": sha256(target_manifest_path),
    "toolchain": {
        "cargo_commit": expected_toolchain[0],
        "rustc_commit": expected_toolchain[1],
    },
    "pattern_counts": receipt["pattern_counts"],
    "finding_count": receipt["finding_count"],
    "findings": receipt["findings"],
}
```

It does not call `run_stdlib_pilot.main`, does not accept a formal protocol,
does not run twice, and does not create a suppression ledger or family report.

- [ ] **Step 3: Run current core once**

Use fresh control/scan targets, trace off, target development `cargo check`, and the current `rapx-perf` binaries. Require exactly one valid selected-unit receipt. Timeout, memory excess, compiler mismatch, or zero receipts is failure rather than zero findings.

- [ ] **Step 4: Cluster with the existing family algorithm**

Group the current receipt by the existing imported `family_key("core", finding)`
and `family_id(key)` functions. Do not add another family-key definition and do
not create a generalized source dependency engine.

- [ ] **Step 5: Produce the simple conservative differential**

Compare current family keys with historical v4 core families. Mark for review:

- every new current family;
- every family whose sink, first failure, obligation, source kind, or propagation depth differs;
- every matched family whose source, root, sink, first-failure, or witness span names a relative `library/core` file whose bytes differ between the installed exact `nightly-2024-10-12` and `nightly-2026-08-13` rust-src trees.

Compute the changed-file set by relative path and SHA-256 over the two installed trees; do not clone the full `rust-lang/rust` repository and do not attempt function-level dependency reconstruction. Rank the review queue deterministically by: new family first; then P1/raw-read families touching changed files; then other changed-file families; finally `(rule_id, family_id)`. Close at most the first 20 families in this feasibility phase. Record the remainder as `not_reviewed_feasibility_budget` and unchanged matches as `not_reaudited`; neither status is a safety classification or suppression decision. This human-review budget does not truncate RAPx propagation or remove findings.

- [ ] **Step 6: Apply the value stop gate**

Continue only if at least one of the at-most-20 reviewed families is new or its changed source contract plausibly allows a safe caller to violate the reported sink precondition. If the bounded review contains no such signal, record `current_core_no_high_value_signal_in_bounded_review` and stop without scanning `alloc/std` or building additional tooling. Do not generalize that result into a claim that current `core` contains no bug.

- [ ] **Step 7: Commit the bounded result**

Commit the runner, normalized receipt, family/difference summary, test, and context. Exclude raw logs, private paths, targets, binaries, and caches.

### Task 5: Validate only credible core candidates

**Files:**
- Create conditionally: `tests/unsoundaudit-v2/stdlib-current-validation/<candidate>/Cargo.toml`
- Create conditionally: `tests/unsoundaudit-v2/stdlib-current-validation/<candidate>/Cargo.lock`
- Create conditionally: `tests/unsoundaudit-v2/stdlib-current-validation/<candidate>/src/lib.rs`
- Create conditionally: `tests/unsoundaudit-v2/stdlib-current-validation/<candidate>/src/main.rs`
- Create conditionally: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/current-core-feasibility/<candidate>.json`
- Create conditionally: `docs/unsoundaudit-v2/disclosure/<candidate>.md`
- Modify: `scripts/unsoundaudit-v2/test_current_nightly_core.py`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Close obvious safe cases by source proof**

For each reviewed family, reconstruct the exact safe call chain, sink precondition, guards, same-object relation, mutation, callback, and unwind behavior. If the proof closes, classify it `confirmed_safe_current` for that exact family and stop; do not write a PoC just to generate activity.

- [ ] **Step 2: Create a safe-only PoC only for a falsifiable remaining hypothesis**

The crate has zero dependencies and no application-boundary `unsafe`. Tests execute controls only. The candidate binary requires exact `UNSOUNDAUDIT_STDLIB_POC_ACK=1`.

- [ ] **Step 3: Run current Miri without native candidate execution**

Run debug and release Miri with fresh targets and `-Zmiri-env-forward=UNSOUNDAUDIT_STDLIB_POC_ACK`. An ordinary panic is not UB; clean Miri is negative evidence, not proof.

- [ ] **Step 4: Check stable and upstream duplicates**

Inspect current stable rust-src, then search official Rust issues, PRs, commits, releases, advisories, and security policy for the exact mechanism. Prepare a disclosure draft only for a current safe-public-API UB with no exact duplicate.

- [ ] **Step 5: Stop for disclosure approval**

Set disclosure status to `awaiting_user_approval`. Do not open an issue or send a security report without a new explicit user instruction.

## Acceptance and stopping boundary

- [ ] The known 39-error port either compiles and passes tests, or is reverted without architectural expansion.
- [ ] Existing P1–P6 positives and recursion/no-cap tests remain intact.
- [ ] Only one development `core` scan and at most 20 prioritized family reviews are performed before the value decision.
- [ ] No `alloc/std` scan, formal protocol, or generalized dependency-diff framework is built without a credible core signal and renewed scope review.
- [ ] No current bug claim lacks a safe public boundary and exact source or Miri evidence.
- [ ] No external report is sent without explicit approval.
