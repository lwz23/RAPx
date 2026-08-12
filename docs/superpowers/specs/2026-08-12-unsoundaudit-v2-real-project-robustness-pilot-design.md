# UnsoundAudit v2 Real-Project Robustness Pilot Design

## 1. Purpose

This work package tests whether the fixture-closed UnsoundAudit v2 implementation remains structurally correct and reviewable on real Rust code. It does not redefine P1–P6 and does not treat the Rust standard library as a zero-finding oracle.

The existing validated claim remains unchanged: source commit `f5988cb156f6f3d8cda59b03ac085607156e6acc` implements the approved heuristic candidate generators on 27 frozen fixtures. The pilot adds evidence about crashes, scale, determinism, causal-chain correctness, and false-positive severity.

## 2. Authorized Change Boundary

The pilot may:

- fix CFG fact propagation, validation propagation, actual/formal mapping, return mapping, and local out-parameter mapping when a failing test demonstrates a structural error;
- recognize semantically equivalent spellings of an already frozen predicate;
- add adversarial fixtures and real-project evaluation tooling;
- improve performance without changing finding semantics;
- suppress a candidate when the existing frozen obligation is demonstrably discharged.

The pilot may not, without separate approval:

- add a source, sink, origin operation, predicate, obligation, or primary pattern;
- change P1–P6 precedence or causal-key identity;
- add cross-crate propagation, whole-program analysis, speculative unresolved edges, or dynamic-dispatch recovery;
- change the original 27 fixture oracles or their aggregate counts;
- describe a heuristic candidate as a confirmed vulnerability without source-level proof.

Any proposed change that crosses these boundaries stops at a written finding and design question.

## 3. Evidence Model

The work uses three evidence sets with different purposes.

| Evidence set | Purpose | Permitted conclusion |
|---|---|---|
| Frozen 27 fixtures | Preserve the approved P1–P6 contract | Regression compatibility only |
| Adversarial challenge pack | Test data-flow and interprocedural properties | Structural correctness for tested transformations |
| Real-project corpora | Measure observed precision and scale | Corpus-bounded empirical results |

Results from one set do not substitute for another. In particular, passing the 27 fixtures does not estimate precision, and a zero count in a real project does not establish recall.

## 4. Phase A: Adversarial Challenge Pack

### 4.1 Required cases

Each positive continues to require exactly one primary finding. Each discharged or unsupported case requires zero findings.

| Area | Required challenge |
|---|---|
| CFG join | Diamond branches where only one path validates, and where only one path writes after validation |
| Loops | Loop-carried reassignment before a sink and a loop with an invariant dominating guard |
| Validation propagation | Guard in the public root, first helper, and sink helper; wrong-value guard; non-dominating guard; write after guard |
| Call mapping | 0-, 1-, 3-, and 5-hop wrappers; constant-return helper; return-to-destination; local projected out-parameter |
| Recursion | Direct and mutual recursion in actual MIR, with a stable cycle witness and no duplicate finding |
| Opaque boundaries | Function pointer, closure, `dyn` call, unresolved trait call, and ordinary external Rust call do not fabricate local edges |
| Equivalent predicates | Existing bounds, non-empty, non-null, and range obligations expressed through approved equivalent comparisons or assertions |
| Spans | Declarative macro and `include!` call sites remain repository-relative, stable, and non-colliding |

The challenge pack is frozen before production fixes begin. A failing case is RED evidence, not permission to expand the registry.

### 4.2 Required engine properties

The implementation must replace block-order-dependent value accumulation with a finite, monotone CFG data-flow solution. Facts are keyed by program point or block state, joined at merges, invalidated by relevant writes, and iterated to equality without an arbitrary round cap.

Function summaries must carry only finite contract facts. A callee-side validation may discharge an obligation only when it binds the same subject and collection, dominates the sink on every relevant path, and remains valid after writes. Return and local out mappings must be structural; unrelated arguments cannot taint a destination merely because they cross a call boundary.

Witness selection remains deterministic, but finding validity cannot depend on a single preferred shortest path. If any feasible unvalidated path reaches the sink, the obligation remains undischarged. A witness is selected only after the path property has been decided.

## 5. Phase B: Rust Standard Library Stress Census

### 5.1 Frozen inputs

- RAP branch and source commit are recorded before the first scan.
- Rust source must match rustc commit `1bc403daadbebb553ccc211a0a8eebb73989665f`.
- The channel remains `nightly-2024-10-12`; Cargo remains commit `15fbd2f607d4defc87053b8b76bf5038f2483cf4`.
- `core`, `alloc`, and `std` are three independent analysis units.
- Cargo jobs and project concurrency remain 1; every control build and scan uses a fresh target.
- No standard-library source or lockfile is modified.

### 5.2 Build route

A temporary, untracked driver uses pinned Cargo `-Z build-std` so the selected standard-library crate is compiled under the RAP wrapper. Merely scanning a crate that depends on `std` is invalid because the analyzer intentionally visits only local MIR.

The order is `core`, then `alloc`, then `std`. The analysis project root is set to the selected standard-library crate so other build units are skipped or passed through. This route validates only intra-crate calls; `std -> alloc -> core` propagation remains unsupported.

Before each scan, the control build must pass with the same source, Cargo arguments, target, feature selection, and exact toolchain. Fetching lockfile-pinned registry dependencies into an isolated Cargo home is allowed; changing dependency versions or lockfiles is not.

### 5.3 Resource and failure gates

Each unit has an explicit wall-time and resident-memory limit recorded before execution. A timeout, out-of-memory event, compiler crash, schema failure, missing unit receipt, hidden truncation, or non-deterministic normalized output is a failed stress gate. It is reported separately from precision and is not converted into a partial success.

Two successful fresh scans of each unit must produce byte-identical normalized receipts. Counts, finding IDs, causal keys, spans, witnesses, and aggregate totals are checked before human review.

## 6. Human Review Protocol

### 6.1 Review population

- If a corpus has at most 250 unique findings, review all findings.
- If it has more than 250, stratify by `rule_id` and propagation depth.
- Within each non-empty stratum, sort by `SHA256(seed || causal_key)` and select at least 20, allocating the remainder proportionally to reach at least 250 total reviews.
- Review every recursive finding and every finding with an interprocedural witness when there are at most 150; otherwise review at least 150 using the same deterministic stratification.

The seed, strata, and sample size are committed before labels are assigned.

### 6.2 Labels

| Label | Meaning |
|---|---|
| A | Safe public root proves a real unsoundness chain under documented Rust contracts |
| B | Complete causal chain is real and actionable, but additional contract evidence is needed |
| C | Chain is structurally real but a project or type invariant proves it safe |
| D | Egregious structural false positive: wrong root, source, sink, validation, flow, or impossible path |
| E | Duplicate, fabricated witness, schema, span-identity, or determinism defect |
| U | Evidence is insufficient to decide |

Reports distinguish confirmed bugs `A`, actionable precision `(A+B)`, structurally real chains `(A+B+C)`, explainable noise `C`, and egregious false positives `(D+E)`. `U` is included as failure in conservative lower bounds.

Two reviewers label independently, with at least one reviewer experienced in Rust unsafe contracts. Conflicts receive third-reviewer adjudication. Cohen's kappa below 0.70 pauses precision claims until the taxonomy or evidence is improved.

### 6.3 Interprocedural chain audit

For each reviewed interprocedural finding, reviewers reconstruct:

1. the public safe root and source origin;
2. every local statically resolved call edge;
3. actual-to-formal and return/out-to-destination mappings;
4. sink operand provenance and CFG reachability;
5. validation dominance, value binding, and write invalidation;
6. any SCC cycle token against an actual recursive cycle.

The report includes edge precision, full-chain precision, summary-contamination rate, and semantic duplicate rate.

## 7. Pre-Registered Acceptance and Stop Lines

The following thresholds are frozen before viewing real-project results:

- zero non-deterministic receipts, schema violations, duplicate finding IDs, fabricated local edges, or hidden truncation;
- egregious false-positive point estimate at most 3%, with Wilson 95% upper bound at most 5%;
- any `rule_id` with at least 20 reviewed findings fails independently if its egregious rate exceeds 10%;
- conservative actionable-precision lower bound at least 60%, and at least 40% for each sufficiently sampled high-frequency `rule_id`;
- more than 500 unique findings across `core + alloc + std`, or more than 0.5 findings/KLoC, pauses the census for noise-root analysis before continuing review;
- three `D/E` labels among the first 20 hash-selected findings, or one fabricated interprocedural edge, triggers early stop and repair;
- fewer than 50 reviewed findings, fewer than 20 interprocedural findings, or zero findings for a rule yields `inconclusive` for that dimension, never `pass`.

After any rule change informed by the standard-library results, the standard library becomes development data. A precision claim then requires an unseen holdout.

## 8. Phase C: Blinded Multi-Project Holdout

A separate corpus of 12–20 pinned library crates is selected without inspecting RAP output. Selection is stratified by size, domain, unsafe density, and dependency complexity. Crate source commits and lockfiles are frozen before scanning.

The corpus is split by crate into development and holdout sets. Rule changes may use only development results. The holdout is opened once after the implementation, 27 fixtures, challenge pack, and standard-library stress gates pass. The same deterministic scan, sampling, labeling, and acceptance protocol applies.

The project list is a separate approval item because downloading and scanning third-party projects expands external inputs beyond the currently authorized standard-library pilot.

## 9. Test and Execution Safety

- Only compilation and static scanning are permitted.
- No target function, test body, example, benchmark, Miri run, or UB path is executed.
- No Linux campaign, coverage ledger, or historical baseline is touched.
- Targets, Cargo homes, standard-library build products, raw logs, and temporary drivers remain untracked.
- Only source, fixtures, schemas, small normalized receipts, SHA manifests, label data, and review summaries may be committed.
- Raw logs remain in local custody with committed SHA-256 digests and no absolute machine paths.

## 10. Deliverables

1. Frozen adversarial fixture manifest and RED receipt.
2. CFG and interprocedural correctness implementation with pure tests and challenge-pack receipts.
3. Per-unit `core`, `alloc`, and `std` build/scan receipts, normalized determinism evidence, resource measurements, and failure classification.
4. Deterministic review sample, labels, adjudication record, and precision report.
5. Updated limitation statement separating fixture, stress-corpus, and holdout evidence.
6. If separately approved, blinded multi-project holdout manifest and final empirical report.

Completion of Phase B permits only a standard-library-bounded stress and precision statement. A broader robustness or ecosystem-precision statement requires Phase C.
