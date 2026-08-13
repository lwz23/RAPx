# UnsoundAudit v2 Real-Project Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve complete P1–P6 finding propagation while meeting 15/30/60-minute and 8/10/12-GiB `core`/`alloc`/`std` release gates.

**Architecture:** Build the analyzer with a safe optimized profile while keeping target MIR in the development profile. Remove redundant clones, converge CFGs without statement snapshots, replay final states into compact observations, and solve local outputs before one full interprocedural summary fixed point. No internal propagation cap is permitted.

**Tech Stack:** Rust nightly `2024-10-12`, rustc private MIR APIs, Cargo profiles, Python 3 standard library, macOS `proc_pid_rusage`, JSONL, deterministic JSON/SHA-256.

---

## File Map

| Path | Responsibility |
|---|---|
| `rapx/Cargo.toml` | `rapx-perf` analyzer build contract |
| `rapx/src/analysis/unsoundaudit/dataflow.rs` | CFG convergence, replay, and clone counters |
| `rapx/src/analysis/unsoundaudit/mir.rs` | Value joins, snapshots, output phases, trace events |
| `scripts/unsoundaudit-v2/run_stdlib_pilot.py` | Physical-footprint enforcement and profile verification |
| `scripts/unsoundaudit-v2/test_stdlib_pilot.py` | No-build runner/profile/monitor tests |
| `artifacts/unsoundaudit-v2/mac/stdlib-pilot/core.v3.failure.json` | Immutable sanitized v3 failure |

### Task 1: Freeze v3 and the analyzer profile

- [ ] Add a RED no-build test requiring the closed `rapx-perf` profile values and proving target scan commands contain no `--release`/`--profile`.
- [ ] Add the `rapx-perf` profile to `rapx/Cargo.toml`; make runner binary metadata record and validate the analyzer profile separately from target Cargo arguments.
- [ ] Add `core.v3.failure.json` with exact protocol/source/binary/log hashes, timeout classification, zero unit receipts, one skip receipt, and all later stages `not_started`.
- [ ] Update `TASK_CONTEXT.md`, run `git diff --check`, run the focused Python tests, and commit the immutable failure/profile block.

Expected command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_stdlib_pilot.py
```

Expected result: all no-build tests pass and no bytecode/cache appears in the repository.

### Task 2: Add fail-closed performance telemetry

- [ ] Add RED tests for trace-disabled/no-file behavior, the exact JSONL key set, balanced coarse phases, deterministic integer counts, and valid timeout prefixes.
- [ ] Implement an optional diagnostic trace writer with phases for collection, normalization/extraction, out-must, output waves/rebuilds, full summary solving, finding construction, and receipt serialization.
- [ ] Add RED Python tests for physical-footprint sampling, disappearing children, unsupported APIs, permission/parse failures, process-group aggregation, and threshold termination.
- [ ] Implement macOS footprint sampling with a small stdlib/ctypes adapter; retain RSS as a recorded secondary metric and make sampling failure fail closed.
- [ ] Run focused Rust/Python tests and commit telemetry independently from solver optimization.

### Task 3: Remove clone-heavy joins and CFG propagation

- [ ] Add RED tests for every `ValueFacts` field, including same-key atom version/invalidation growth with unchanged set length; assert first join changes and the second is idempotent.
- [ ] Replace whole-value clone/equality with per-field union/OR helpers that report exact change.
- [ ] Add a test-only CFG counter showing unchanged edges do not recompute successor entries and predecessor inputs are not cloned into a temporary vector.
- [ ] Change CFG propagation to skip unchanged edges and join borrowed predecessor states through one accumulator.
- [ ] Add a write-invalidation counter proving unrelated values are not cloned, then implement the read-only relevance precheck.
- [ ] Run full exact-nightly Rust tests and both fixture suites twice; compare canonical finding-set projections and commit only if equal.

### Task 4: Separate CFG convergence from final replay

- [ ] Retain a test-only reference solver that stores every `before` state.
- [ ] Add table-driven diamond, loop, normal/unwind, pending-validation, and relation-invalidation tests comparing every reference `before[(bb,index)]` with replay from final entry states.
- [ ] Change production convergence to retain only entry/exit/edge states and add deterministic final block replay.
- [ ] Move materialization and body aggregate folding into replay; assert replay cannot enqueue CFG work or alter convergence state.
- [ ] Add structural counters requiring zero persistent full statement states during convergence.
- [ ] Run the exact Rust gate and double fixture gates, then commit this block separately.

### Task 5: Replace persistent states with validation snapshots

- [ ] Define `ValidationSnapshot` with current bindings and exact unchanged entry-formal mappings.
- [ ] Add RED equivalence tests comparing snapshot queries with `has_current_validation` and `entry_formal_slot` at resolved calls, all local predicate sinks, writes, ambiguous identities, and havoced formals.
- [ ] Precompute the exact observation-point set from normalized calls/sinks/contracts; replay captures snapshots only at those points.
- [ ] Replace all three persistent-state consumers with snapshot queries. A missing required snapshot returns Hard/unvalidated rather than using a whole-body fallback.
- [ ] Project persistent body aggregate values to origins/value-flow only and add tests proving candidate/source/alias results are unchanged.
- [ ] Run full semantic gates and a trace-enabled 15-minute `core` development diagnostic. If it fails, preserve the trace as local diagnostic data and continue to Task 6.

### Task 6: Separate local-output and full-summary fixed points

- [ ] Add a test-only reference implementation of the current outer loop.
- [ ] Add mixed DAG/self/mutual-recursion tests comparing final local outputs, every rebuilt body seed, complete summaries, and findings between reference and new algorithms.
- [ ] Implement an output-only projection over `return_value`, `out_dependencies`, call mappings, and frozen definite outs; it must not invoke the full requirement solver.
- [ ] Make local-output join return the exact changed-owner set. Re-extract only changed owners from immutable body programs using synchronous waves.
- [ ] Invoke the full summary solver once after output equality, then run the existing unresolved-route and witness pipeline.
- [ ] Add counters asserting unchanged owners are not rebuilt and the full solver runs exactly once.
- [ ] Add at least a 64-hop local-call propagation test plus self/mutual recursion to prove no hidden depth cap.
- [ ] Run exact Rust and double fixture gates; commit only on finding-set equality.

### Task 7: Build optimized analyzers and run development gates

- [ ] Build fresh dev, stock release, and `rapx-perf` analyzer binaries with the exact toolchain and embedded RAP commit. Do not change the target crate profile.
- [ ] Run challenge and original fixtures twice with each analyzer profile. Require canonical finding-set equality across profiles and byte-identical normalized output within each profile.
- [ ] Run a trace-enabled `rapx-perf` `core` scan with a 15-minute/8-GiB physical-footprint gate. Stop immediately on failure and inspect only phase/resource diagnostics, not findings.
- [ ] If `core` passes, run single `alloc` and `std` development scans at 30/10 and 60/12 gates. Do not run a later unit after an earlier failure.
- [ ] If all pass, skip Task 8 and freeze a new formal protocol. If a gate fails, use the trace to select only the necessary conditional optimization.

### Task 8: Conditional deep optimization

- [ ] If state key/string operations dominate, add body-local deterministic `PlaceId` interning and exact boundary restoration; compare all program-point query results before/after.
- [ ] If write invalidation dominates, add storage-to-relation watcher indexes and prove indexed invalidation equals the reference full scan.
- [ ] If output derivation/rebuild dominates, add a stable dirty set containing changed owners and reverse resolved-local callers; preserve synchronous batch updates.
- [ ] Repeat semantic and development performance gates after each selected change. Stop once all hard gates pass; do not implement unused optimizations.

### Task 9: Freeze and run the formal performance campaign

- [ ] Commit the final analyzer source and exact synthetic evidence before writing the next protocol.
- [ ] Freeze a new protocol that pins source/toolchain/binary hashes, `rapx-perf`, target dev/check, 15/30/60 minutes, physical-footprint 8/10/12 GiB, RSS diagnostics, two runs, and unchanged review thresholds.
- [ ] Run `core`, `alloc`, then `std`; each unit requires two fresh trace-disabled scans, exactly one unit receipt per scan, canonical validation, and byte-identical normalized output.
- [ ] Freeze the deterministic review sample only after all three units pass. Dispatch two independent reviewers, adjudicate conflicts, and apply existing precision thresholds.
- [ ] Update the bounded pilot report and SHA manifest, run independent final review, commit, guarded-fetch, and push.

## Stop Conditions

- Any canonical finding addition/removal stops performance work as a semantic regression.
- Any hidden truncation, fabricated witness, or incorrect validation discharge is an immediate blocker.
- Any development or formal resource failure stops later units.
- Reaching all time/memory gates ends optimization; do not implement speculative deeper changes.

