# UnsoundAudit v3 P1 Root-Cause Classification Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan inline with TDD; do not delegate because this shared worktree already contains the v3 MVP WIP.

**Goal:** Classify a public formal that transparently violates an immediate-validity unsafe contract as P1 even when the unsafe result is later exposed, while retaining the exposure metadata and preserving true internally-originated P3 findings.

**Architecture:** Keep census, binding, endpoint flow, contracts, and finding cardinality unchanged. Refine only the structural candidate classifier: before creating a P3 candidate for `UnsafeResult -> public exposure`, look for a root-matching public formal that reaches a relevant unsafe operand without an internal derivation; if present, emit the same candidate as P1 with the original exposure attached. Otherwise retain the existing P3 path.

**Tech Stack:** Rust, existing `EndpointFlowGraph`, pinned nightly-2024 toolchain, production `cargo rapx -unsoundaudit` Gitoxide smoke.

---

### Task 1: Add the root-cause regression

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/candidate.rs`

- [x] Add a graph-level test with `Formal -> UnsafeOperand`, `UnsafeResult -> Return`, an immediate-validity contract, and a public safe owner. Assert one P1 candidate, `PublicParameter` source, the relevant operand role, and retained exposure.
- [x] Keep the existing no-input `UnsafeResult -> Return` test and assert it remains P3.
- [x] Run the two focused tests and confirm the new test is RED because the current early P3 path ignores operand provenance.

### Task 2: Implement the minimal classifier change

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/candidate.rs`

- [x] In the immediate-validity exposure path, inspect only `ContractDescriptor::relevant_roles` and exact `UnsafeOperand` nodes for the same site.
- [x] Select a public formal only when its owner matches the public exposure root and its canonical route contains no `InternalDerivation` node.
- [x] Emit P1 using `p1_public_parameter_flow_v1`, preserve `CandidateExposure`, use the formal-to-operand endpoint route, and retain the real call route to the unsafe-site owner.
- [x] If no qualifying formal exists, execute the existing P3 construction unchanged.
- [x] Run the focused candidate tests, then the P1 and P3 production mechanism crates.

### Task 3: Verify the real historical differential

**Files:**
- Modify: `TASK_CONTEXT.md`

- [x] Rebuild the pinned-2024 `rapx-perf` binaries.
- [x] Scan Gitoxide #1460 pre-fix commit `a807dd1ffb05efd177700d065095249e6c4b3c68`; assert both target candidates are P1, retain public-return exposure, and have complete coverage.
- [x] Scan the fixed revision; assert the same unsafe-site causal target is absent.
- [x] Run the full pinned-2024 Rust library gate and v3 Python gate, then `cargo fmt --check` and `git diff --check`.
- [x] Record the corrected classification and evidence paths in `TASK_CONTEXT.md`; do not change old replay-v1/v2 evidence.
