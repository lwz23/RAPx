# P2–P5 High-Confidence Witness Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Require exact, realizable finite violation witnesses for active P2–P5 findings while preserving every unresolved candidate as unestablished and retaining all existing positive mechanism oracles.

**Architecture:** Extend the existing exact influence/context solvers and four-state reporting rather than adding a universal symbolic executor. P2 reuses influence tokens; P3 adds finite validity/initialization outcomes; P4 evaluates finite derivation obligations against site snapshots; P5 completes callsite substitution and equality handling.

**Tech Stack:** Rust/rustc private APIs, finite MIR abstract domains, deterministic JSONL reports, Python 3 validators, pinned nightly-2024-10-12 and current nightly.

---

## File map

- Modify `rapx/src/analysis/unsoundaudit/candidate.rs`: exact P2 gate, P3 validity outcome, P4 derivation outcome, P5 unknown recording, generalized unestablished identities, focused tests.
- Modify `rapx/src/analysis/unsoundaudit/mir.rs`: export site-current initialization and relation evidence required by P3/P4; strengthen compiler-derived generic substitutions where necessary.
- Modify `rapx/src/analysis/unsoundaudit/influence.rs`: only finite requirement/reason additions required by generalized gates.
- Modify `rapx/src/analysis/unsoundaudit/v3_output.rs`: admit P2–P5 unestablished decisions without changing active finding/family schema.
- Modify `scripts/unsoundaudit-v3/run_mechanism_pack.py` and `run_stdlib_gate.py`: validate all P1–P6 unestablished rows.
- Modify mechanism-pack P2–P5 sources/oracles only for general semantic matrix cases.
- Modify `TASK_CONTEXT.md`: decisions, gates, performance, artifacts, and remaining limitations.

### Task 1: Generalize exact decision accounting

- [x] Add failing Rust/Python tests proving P2–P5 decisions are accepted, deterministically ordered, disjoint from active/suppressed identities, and preserve stable reasons.
- [x] Generalize the candidate constructor and validators without changing existing P1/P6 rows.
- [x] Run focused reporting and no-build Python tests; require PASS.

### Task 2: Gate P2 through exact influence

- [x] Add tests for direct pointer/boolean fields, overwritten routes, helper transport, capability mismatch, and opaque sinks.
- [x] Admit `PersistentPublicField` in exact influence origin matching and invoke the solver for P2.
- [x] Mark only exact P2 candidates `ClassificationConfidence::Exact`; send missing/model-opaque facts to unestablished.
- [x] Run candidate/influence tests and the P2 mechanism case; preserve all current positive roots.

### Task 3: Gate P3 through validity and initialization outcomes

- [x] Add tests for invalid bool/enum, validity-weakening transmute, complete typed writes/copies, unknown initialization, and public-out exposure.
- [x] Capture the site-current typed initialization state for relevant unsafe operands.
- [x] Implement `UnsafeResultSafety::{ViolationWitness, ProvenSafe, Unknown}` from static type implication, contract model, initialization state, and exact exposure.
- [x] Move `Unknown` to unestablished before active selection; keep dynamic/static proven-safe candidates in the suppression ledger.
- [x] Run focused tests and the P3 mechanism case; all existing positives remain active.

### Task 4: Gate P4 through finite derivation obligations

- [x] Add tests for equality-to-length violations, dominated guards, stale/diamond guards, boolean false witnesses, layout widening, and unsupported relations.
- [x] Evaluate bounds/range, boolean, layout/alignment, and typed-initialization negations using the site snapshot and compiler-derived facts.
- [x] Require an exact source-to-sink corridor and `ViolationWitness` for active P4; send incomplete domains to unestablished.
- [x] Run focused relation/candidate tests and the P4 mechanism case; all existing positives remain active.

### Task 5: Complete P5 callsite witness transport

- [x] Add tests for same direct type, equal projections, const arrays, representation-preserving views, concrete mismatch, and failed normalization.
- [x] Union compiler equality classes through every exact internal call and retain concrete size/alignment/const facts.
- [x] Keep only legal parametric counterexamples active; suppress concrete satisfaction, reclassify concrete violation to P4, and record incomplete normalization as unestablished.
- [x] Run focused context-P5 tests and the P5 mechanism case; preserve all five current positive roots.

### Task 6: Full verification and audit

- [x] Run scoped rustfmt and `git diff --check`.
- [x] Run the complete pinned Rust library gate and all no-build Python tests.
- [x] Build metadata-correct pinned/current analyzers and run P2–P5 mechanism packs.
- [x] Run exact core with a 900-second wall limit and record elapsed time and peak physical footprint.
- [x] Compare the frozen 48-family sample and deterministically select 48 new families stratified by pattern, state, model, and depth.
- [x] Write counts, mappings, reason distributions, positive recall, sampled residual false positives, and limitations to `TASK_CONTEXT.md`.

All commits must stage only owned files or reviewed hunks because the working tree contains pre-existing shared edits.
