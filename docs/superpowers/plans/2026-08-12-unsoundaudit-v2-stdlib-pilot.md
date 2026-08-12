# UnsoundAudit v2 Standard-Library Pilot Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a deterministic, resource-bounded `core`/`alloc`/`std` stress census and pre-registered precision review after the adversarial robustness plan is fully green.

**Architecture:** A standard-library-only pilot runner creates untracked `-Z build-std` drivers under the exact pinned toolchain, routes exactly one selected standard-library crate to RAP, and fails closed on resource, routing, schema, or determinism errors. A separate deterministic sampler freezes findings before two-reviewer labeling and computes bounded per-unit and per-rule metrics.

**Tech Stack:** Rust nightly `2024-10-12`, Cargo `-Z build-std`, Python 3 standard library, JSON, SHA-256, macOS process controls.

---

## Entry Gate and File Map

This plan may start only after Task 11 of `2026-08-12-unsoundaudit-v2-robustness-implementation.md` passes and its GREEN evidence is committed.

| Path | Responsibility |
|---|---|
| `tests/unsoundaudit-v2/stdlib_pilot_protocol.json` | Exact commits, unit order, sample seed, thresholds, and resource limits |
| `scripts/unsoundaudit-v2/run_stdlib_pilot.py` | Build-std driver, routing, limits, receipt normalization |
| `scripts/unsoundaudit-v2/review_sample.py` | Deterministic stratification, label validation, and metrics |
| `scripts/unsoundaudit-v2/test_stdlib_pilot.py` | No-build tests for pilot tooling |
| `artifacts/unsoundaudit-v2/mac/stdlib-pilot/` | Small normalized evidence and reviews |

### Task 1: Freeze the pilot protocol before scanning

**Files:**
- Create: `tests/unsoundaudit-v2/stdlib_pilot_protocol.json`
- Create: `scripts/unsoundaudit-v2/test_stdlib_pilot.py`

- [ ] **Step 1: Write protocol validation tests**

Require exact RAP commit, rustc/Cargo commits, unit order `core,alloc,std`, a non-secret fixed sample seed, jobs=1, two runs, fresh targets, wall-time/RSS limits per unit, count/KLoC and review thresholds, and `build_std_components` for each unit. Reject extra keys and absolute paths.

- [ ] **Step 2: Run tests and capture RED**

Expected: FAIL because the protocol is absent.

- [ ] **Step 3: Add the closed protocol**

Use schema `unsoundaudit-v2-stdlib-pilot-protocol-v1`. Record limits before scanning: 60 minutes/12 GiB for `core`, 90 minutes/16 GiB for `alloc`, and 180 minutes/24 GiB for `std`. Raising a limit after observing failure requires a new protocol version and records the original gate as failed.

- [ ] **Step 4: Run protocol tests and commit**

Expected: all no-build tests PASS and canonical JSON SHA is printed.

```bash
git add tests/unsoundaudit-v2/stdlib_pilot_protocol.json scripts/unsoundaudit-v2/test_stdlib_pilot.py
git commit -m "test(unsoundaudit-v2): freeze standard-library pilot protocol"
```

### Task 2: Implement fail-closed build-std routing

**Files:**
- Create: `scripts/unsoundaudit-v2/run_stdlib_pilot.py`
- Modify: `scripts/unsoundaudit-v2/test_stdlib_pilot.py`

- [ ] **Step 1: Add RED command-construction tests**

Assert untracked driver manifests, no build script, `-Z build-std=<components>`, `--locked`, `--jobs 1`, and selected unit root. Reject unverified rust-src, non-empty output, missing rust-src, encoded rustflags, or mismatched commits.

- [ ] **Step 2: Run tests and capture RED**

Expected: FAIL because routing helpers do not exist.

- [ ] **Step 3: Implement temporary drivers outside the repository**

Use a minimal library manifest. The `core` source is `#![no_std]`; `alloc` uses `#![no_std]` plus `extern crate alloc`; `std` uses an empty ordinary library. Control and scan use identical Cargo arguments; only scan enables RAP and selects the standard-library project root.

- [ ] **Step 4: Add resource enforcement**

Use a process group, sample child RSS, terminate the group on limit breach, and classify `timeout`, `memory_limit`, `compiler_failure`, `route_failure`, or `success`. Partial receipts never normalize as success.

- [ ] **Step 5: Run no-build tests and commit**

```bash
git add scripts/unsoundaudit-v2/run_stdlib_pilot.py scripts/unsoundaudit-v2/test_stdlib_pilot.py
git commit -m "test(unsoundaudit-v2): add fail-closed build-std pilot runner"
```

### Task 3: Implement deterministic sampling and metrics

**Files:**
- Create: `scripts/unsoundaudit-v2/review_sample.py`
- Modify: `scripts/unsoundaudit-v2/test_stdlib_pilot.py`

- [ ] **Step 1: Add RED sampling tests**

Test full review for `N<=250`; `rule_id × depth` strata; at least 20 per non-empty stratum; SHA seed order; all recursive findings; up to 150/all interprocedural findings; input-permutation stability; and labels `A,B,C,D,E,U` only.

- [ ] **Step 2: Add RED metric tests**

Verify actionable, structurally real, noise, egregious, unknown-conservative lower bounds, Wilson intervals, per-rule failure, kappa, and early-stop calculations on fixed examples.

- [ ] **Step 3: Implement standard-library-only sampling and metrics**

Emit canonical JSON with population/protocol SHA, seed, inclusion rule, causal keys, strata, inclusion probabilities, and formulas. Do not depend on random iteration.

- [ ] **Step 4: Run tests and commit**

```bash
git add scripts/unsoundaudit-v2/review_sample.py scripts/unsoundaudit-v2/test_stdlib_pilot.py
git commit -m "test(unsoundaudit-v2): add deterministic precision review sampling"
```

### Task 4: Run the `core` gate

**Files:**
- Create one of: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/core.normalized.json`, `core.failure.json`
- Create on success: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/core.sample.json`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Re-run exact Rust tests, frozen 27 twice, and challenge suite twice**

Expected: every synthetic gate PASS and each normalized pair is byte-identical.

- [ ] **Step 2: Verify exact rust-src and library lockfile, then populate only the isolated cache with pinned dependencies**

Do not modify source or lockfiles.

- [ ] **Step 3: Run the control build and two fresh `core` scans**

Require exactly one analyzed core receipt, limits respected, and identical normalized bytes. On failure write only `core.failure.json` and stop before `alloc`.

- [ ] **Step 4: Apply schema, ID, aggregate, path, truncation, and count/KLoC gates**

- [ ] **Step 5: Freeze the deterministic sample before viewing findings and commit the gate result**

```bash
git add artifacts/unsoundaudit-v2/mac/stdlib-pilot TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v2): record core robustness pilot"
```

### Task 5: Review `core` and decide continuation

**Files:**
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/core.labels.json`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/core.review.json`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Obtain two independent A–U labels for the frozen sample**

Each interprocedural label records root, every local edge, mappings, sink provenance, CFG/validation, and cycle evidence.

- [ ] **Step 2: Adjudicate, require kappa at least 0.70, and compute pre-registered metrics**

- [ ] **Step 3: Apply early-stop, egregious, per-rule, and usability thresholds**

Any fabricated edge or hard threshold failure stops. After findings are viewed, `core` becomes development data for any subsequent fix.

- [ ] **Step 4: Commit sanitized labels and review**

### Task 6: Run and review `alloc`

**Files:**
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/alloc.*.json`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Require a passing `core` continuation decision**

- [ ] **Step 2: Repeat the exact two-run gate, frozen sample, independent labels, adjudication, and thresholds**

Set analysis root to `library/alloc`; core remains opaque. Stop before std on any hard failure.

- [ ] **Step 3: Commit alloc evidence or failure receipt**

### Task 7: Run and review `std`

**Files:**
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/std.*.json`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Require passing continuation decisions for core and alloc**

- [ ] **Step 2: Repeat the exact two-run gate, frozen sample, independent labels, adjudication, and thresholds**

Set analysis root to `library/std`; dependencies and build units remain opaque. Do not average away a failed unit or rule.

- [ ] **Step 3: Commit std evidence or failure receipt**

### Task 8: Publish the bounded report

**Files:**
- Create: `docs/unsoundaudit-v2/STDLIB_PILOT_REVIEW.md`
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-pilot/sha256_manifest.json`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Re-run all exact synthetic gates**

- [ ] **Step 2: Recompute every committed source, protocol, result, sample, label, and review hash**

Reject absolute paths, raw logs, targets, caches, drivers, binaries, or rust-src copies.

- [ ] **Step 3: Report each unit separately**

Include route, resources, counts, deterministic SHA, A–U population, intervals, per-rule and interprocedural results, failures, and unsupported cross-crate boundaries. Use `inconclusive` where evidence is insufficient.

- [ ] **Step 4: Obtain independent read-only final review and commit**

```bash
git add docs/unsoundaudit-v2/STDLIB_PILOT_REVIEW.md artifacts/unsoundaudit-v2/mac/stdlib-pilot/sha256_manifest.json TASK_CONTEXT.md
git commit -m "docs(unsoundaudit-v2): report standard-library robustness pilot"
```

- [ ] **Step 5: Guarded push**

Fetch the fixed feature branch, require remote tip to be a local ancestor, and push normally. Stop without rebase or force on divergence.

## Plan Self-Review

- The entry gate prevents real-project scanning before synthetic correctness is green.
- Tasks 1–3 cover protocol, routing, limits, sampling, labels, and statistics.
- Tasks 4–7 preserve unit order and stop after any hard failure.
- Task 8 provides hashes, independent review, bounded claims, and guarded publication.
- No task adds registry forms, precedence changes, cross-crate propagation, target execution, Linux access, or third-party holdout downloads.
