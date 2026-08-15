# UnsoundAudit v3 Compiler-Aligned P1–P6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace registry-owned unsafe-sink discovery with a compiler-aligned, two-toolchain THIR census while retaining RAPx's crate-local MIR dataflow, interprocedural summaries, recursion handling, validation, and deterministic witness construction.

**Architecture:** A version-bound query wrapper collects every rustc `check_unsafety` site into an owned IR before THIR is stolen. A deterministic binder attaches each site to built MIR and solver MIR, after which a sparse endpoint-flow lane creates P1–P6 structural candidates independently of the contract catalog; precise and generic contract models may refine or discharge those candidates, while opaque contracts remain visible. The new analyzer writes only `rap-unit-v3`; v1/v2 validators and evidence remain immutable readers.

**Tech Stack:** Rust/rustc-private THIR and MIR APIs, nightly-2024-10-12, nightly-2026-08-14, existing finite CFG/SCC summary engine, JSON Schema, Python standard-library validators/runners, `rapx-perf`, macOS process-group physical-footprint monitoring.

---

### Task 0: Preserve the verified v2 baseline

**Files:**
- Preserve: `TASK_CONTEXT.md`
- Preserve: `rapx/src/analysis/unsoundaudit/`
- Preserve: `docs/unsoundaudit-v2/fast_historical_replay_v2_evidence/`

- [x] Run the exact pinned Rust library gate and require 168/168.
- [x] Run all no-build Python contracts and require 132/132.
- [x] Compile the library with current pinned rustc `ba28ff76…`.
- [x] Commit the pre-existing replay-v2/analyzer work as `d406a9a5ff7948e52c9aa5453dbfe4ef76f9e27f`.
- [x] Create annotated tag `unsoundaudit-v2-pre-v3-preservation-2026-08-16`.
- [x] Keep `tmp/` and all build/log/cache artifacts outside the commit.

### Task 1: Freeze v3 public IR and schema

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/unsafe_site/mod.rs`
- Create: `tests/unsoundaudit-v3/rap-unit-v3.schema.json`
- Create: `scripts/unsoundaudit-v3/validate_receipt.py`
- Test: `scripts/unsoundaudit-v3/test_contract.py`

- [ ] Write schema RED tests for closed keys, site-ID uniqueness, census reconciliation, partial lower-bound counts, coverage failures, opaque contracts, and deduplicated code blocks.
- [ ] Define owned `UnsafeSiteKind`, `UnsafeSiteSeed`, `OperandAnchor`, `ResultAnchor`, `OwnerCensus`, binding status, coverage failure, contract precision, and v3 finding DTOs.
- [ ] Define `site_id = SHA-256(adapter_id, owner def-path hash, owner-local traversal ordinal, kind)` and reject duplicate IDs.
- [ ] Define `analysis_status = complete | partial`; any unbound, ambiguous, elided-without-equivalent-event, or collector fault makes the receipt partial.
- [ ] Define `contract_precision = precise | generic | opaque` independently from binding and analysis status.
- [ ] Preserve observed partial findings but mark observed counts as lower bounds; validators must reject presentation of partial zero as complete zero.
- [ ] Keep v1/v2 schemas and validators byte-for-byte unchanged after Task 0.
- [ ] Run focused Rust/Python tests, `git diff --check`, and commit.

### Task 2: Add exact toolchain adapter manifests and census store

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/unsafe_site/store.rs`
- Create: `rapx/src/analysis/unsoundaudit/unsafe_site/legacy_2024.rs`
- Create: `rapx/src/analysis/unsoundaudit/unsafe_site/current_2026.rs`
- Modify: `rapx/src/lib.rs`

- [ ] Write RED tests for the two accepted rustc commits, checker source digests, emitted kind sets, a third unsupported nightly, stable owner-local ordering, and concurrent owner insertion.
- [ ] Freeze 2024 adapter manifest: rustc `1bc403daadbebb553ccc211a0a8eebb73989665f`, checker SHA-256 `bde4d27635e54d8ca0582ef8decd204c3089591e5c540322891d7aaf1a182023`.
- [ ] Freeze current adapter manifest: rustc `ba28ff76f353a722f31c4f3dd2ac4e437d36411b`, checker SHA-256 `25b4b8e22ff7d561dda2a9abf429ab5bef035d246ec0200b9b887f7713fca346`.
- [ ] Implement an invocation-generation-scoped, thread-safe owned census store; write an `OwnerCensus` even when an owner contains zero sites.
- [ ] Reject unsupported commits before analysis output creation.
- [ ] Preserve the existing 2024 extern-rlib query override when installing the census wrapper.
- [ ] Run adapter unit tests under both toolchains and commit.

### Task 3: Mirror rustc `check_unsafety` in THIR

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/unsafe_site/legacy_2024.rs`
- Modify: `rapx/src/analysis/unsoundaudit/unsafe_site/current_2026.rs`
- Test: `tests/unsoundaudit-v3/census-fixtures/`

- [ ] Add compile fixtures for every emitted 2024 and current unsafe category, direct/FnPtr calls, closures, inline consts, macros, includes, nested blocks, and exemptions.
- [ ] Capture RED census output before implementing visitors.
- [ ] Implement per-version THIR visitors using the same decision points and recursive body ownership as the matching rustc checker; do not traverse AST or use an AST fallback.
- [ ] Permit only type-level matching of the THIR-embedded asm macro enum to exclude current `naked_asm!`.
- [ ] Store only owned spans, type fingerprints, roles, callee IDs, and traversal ordinals; release every THIR borrow before calling rustc's original provider exactly once.
- [ ] Add after-analysis completeness sweep: recorded owner → no-op; unstolen missing owner → collect; stolen missing owner → coverage fault.
- [ ] Require registry-empty and registry-full runs to have identical site-ID sets.
- [ ] Require two-run byte identity for each toolchain's census pack and commit.

### Task 4: Bind THIR sites to built and solver MIR

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/unsafe_site/binding.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`

- [ ] Write RED tests for call actuals/destination, function pointers, raw read/write/borrow, union fields, static accesses, inline asm, same-span multiple sites, macros, zero candidates, and multiple perfect matchings.
- [ ] Enumerate owner-local built-MIR events and solver-MIR events without consulting the contract catalog.
- [ ] Build a bipartite match using owner, raw hygienic span, source scope/lint root, site kind, callee or FnPtr signature, operand count/type/role, result type, and MIR shape.
- [ ] Accept only a unique complete matching; never choose by nearest span or insertion order.
- [ ] Attach `Vec<BoundUnsafeSite>` to `ProgramOp`, preserving existing Assign/Call/InlineAsm transfer semantics.
- [ ] Emit deterministic `unbound`, `ambiguous`, and `elided` coverage records and mark the unit partial.
- [ ] Allow a THIR-local low-confidence candidate only when the source-to-site relation is directly established without inventing an interprocedural edge.
- [ ] Run focused binding tests under both toolchains and commit.

### Task 5: Build the sparse endpoint-flow lane

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/endpoint_flow.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs`

- [ ] Add RED tests for formal/public-field/external-output/internal-derivation origins, transparent flow, return/out propagation, normal/unwind, 64 hops, self recursion, and mutual recursion.
- [ ] Define `EndpointFlowGraph { nodes: BTreeSet<EndpointNode>, edges: BTreeSet<EndpointEdge> }`; nodes and edges carry only stable finite identities and one-hop provenance.
- [ ] Define finite endpoint nodes for formal, public field, external behavior result/out, internal derivation, unsafe operand/result, return/out, and exposure.
- [ ] Reuse existing exact local value-flow, actual↔formal, return/out, write invalidation, and SCC summaries to emit only one-hop ordered edges.
- [ ] Compute reachability over SCC condensation without path/fact/depth/round caps; construct a canonical witness lazily only for emitted findings.
- [ ] Persist sparse snapshots only at bound unsafe sites and exposure points; do not reintroduce full per-statement `BlockState` storage.
- [ ] Add structural counters proving census-only sites do not enter the predicate lattice and commit.

### Task 6: Convert the registry into a ContractCatalog

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/contract_catalog.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs`

- [ ] Write RED tests proving the catalog cannot create/delete census sites or pre-discharge structural candidates.
- [ ] Move current `get_unchecked`, raw-read, UTF-8, `assume_init`, NonNull, range, and FFI models behind `ContractDescriptor`.
- [ ] Implement `precision = precise | generic | opaque`, failure phase, relevant roles, obligation, and validation model.
- [ ] Add generic language-site contracts for raw deref/access, restricted validity, union result validity, and layout-constrained access.
- [ ] For opaque unsafe calls, select only deterministic memory/validity roles: memory-bearing values, scalars flowing into address/index/length/size/alignment/layout/discriminant, generic capabilities affecting representation/validity, writable out values, and validity-sensitive returns.
- [ ] Leave pure unrelated scalar operands census-only; do not use callee names, paths, projects, or line numbers.
- [ ] Permit discharge only for precise/generic models with exact subject, collection, access width, versions, and dominating CFG validation. Opaque candidates are never heuristically discharged.
- [ ] Remove registry discovery authority only after shadow comparisons pass; commit.

### Task 7: Implement structural P1–P6 classification

**Files:**
- Create: `rapx/src/analysis/unsoundaudit/candidate.rs`
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Test: `tests/unsoundaudit-v3/pattern-fixtures/`

- [ ] Add P1 fixtures for direct, wrapper, 64-hop, FnPtr, self-recursive, and mutual-recursive public-formal flow.
- [ ] Add P2 fixtures for exact public-field state, private/wrong field, and write invalidation.
- [ ] Add P3 fixtures for illegal reference/slice/enum/initialization construction followed by return, store, or exposure.
- [ ] Add P4 fixtures for internal bounds, address/offset, access width, alignment, layout, and arithmetic re-break.
- [ ] Add P5 fixtures for nonempty, all-bytes-initialized, size/alignment compatibility, representation, and validity capability gaps.
- [ ] Add P6 fixtures for Iterator, open trait, callback/FnPtr, FFI return, and FFI out.
- [ ] Implement deterministic precedence: P3 exposure; P6 external behavior; P5 capability gap; P2 public state; P4 proven internal derivation/re-break; otherwise P1 public formal.
- [ ] Treat copy/cast/projection/call/return as transparent and arithmetic/index/length/address/layout transformations as derivations.
- [ ] If P1→P4 re-break cannot be established, retain P1 with structural classification confidence.
- [ ] Remove `RuleId` from internal requirements. Keep `primary_pattern=P1..P6`, a separate mechanism ID, and contract precision.
- [ ] Run wrong-subject/collection, diamond, loop, normal/unwind, write/havoc, and insertion-order tests; commit.

### Task 8: Emit `rap-unit-v3` and complete code bundles

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit.rs`
- Modify: `rapx/src/utils/unit_output.rs`
- Modify: `scripts/unsoundaudit-v3/validate_receipt.py`

- [ ] Write serialization RED tests for complete/partial receipts, deterministic code dictionaries, normalized paths, missing source/docs, and finding-ID stability.
- [ ] Emit root, source, first failure, unsafe site, optional exposure, contract precision/model/obligation, binding confidence, exact witness, and code references.
- [ ] Build a deduplicated dictionary containing complete local function bodies for the safe root, every causal function, unsafe-site owner, and exposure owner.
- [ ] Include local/external unsafe signatures and `# Safety` docs when available; record a deterministic unavailable reason otherwise.
- [ ] Compute v3 finding IDs from pattern, site, source, first failure, root, and exposure; exclude witness choice, code text, profile, and contract-model version.
- [ ] Keep the CLI `-unsoundaudit`; write only v3. Retain v1/v2 read-only validation tools.
- [ ] Start v3 with all v2 stdlib suppression disabled and commit.

### Task 9: Cut over and remove legacy discovery authority

**Files:**
- Modify: `rapx/src/analysis/unsoundaudit/mir.rs`
- Modify: `rapx/src/analysis/unsoundaudit/summary.rs`

- [ ] Run shadow mode on every legacy known operation and reconcile old sites against census/binding.
- [ ] Switch finding seeds to structural candidates from bound unsafe sites.
- [ ] Delete or demote `is_modeled_registry_call`, API-name sink discovery, and `adjusted_rule` so they cannot decide site existence or primary pattern.
- [ ] Keep compatibility projection tests for original/challenge semantics.
- [ ] Assert no site is silently dropped and no v2 writer remains reachable; commit.

### Task 10: Full regression, determinism, and performance gates

**Files:**
- Create: `scripts/unsoundaudit-v3/run_fixture_suite.py`
- Create: `scripts/unsoundaudit-v3/run_stdlib_gate.py`
- Modify: `TASK_CONTEXT.md`

- [ ] Run all Rust/Python tests with 0 failed and 0 ignored under both supported adapters.
- [ ] Run original 27 twice and require semantic aggregate `3/2/2/2/1/2`.
- [ ] Run challenge 27 twice and require P1=13, P2–P6=0.
- [ ] Compare debug, stock-release, and `rapx-perf` semantic projections.
- [ ] Verify v1/v2 frozen hashes and preservation tag remain unchanged.
- [ ] Run 2024 core/alloc/std twice with 900/1800/3600-second and 8/10/12-GiB physical-footprint gates.
- [ ] Run current census/binding/pattern/real-project sentinel and core twice with 900 seconds / 8 GiB.
- [ ] Require one selected receipt, trace off, fresh targets, jobs=1, byte-identical normalized v3 receipts, no partial analysis, and no hidden truncation.
- [ ] Record exact commits, binary/schema hashes, resource receipts, limitations, and the mechanism-fixture-only recall boundary in `TASK_CONTEXT.md`; commit and tag the first complete v3 protocol source.

## Fixed acceptance boundary

- The PDF distribution P1/P2/P3/P4/P5/P6 = 9/1/24/5/5/5 is a design-scope statement, not a claim that 49 original repositories were replayed.
- The remaining 33 concurrency/lifetime/Drop/reentrancy mechanisms are out of scope for findings, though compiler unsafe sites remain visible in the census.
- No SMT, AST discovery/fallback, repository whitelist, source-line rule, stdlib suppression migration, or propagation/fixed-point cap is permitted.
- Binding gaps are visible partial coverage, never successful zero findings.
