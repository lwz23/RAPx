# RAPx P1–P6 Lightweight Relational Refinement Design

## Status and decision

This design is approved for direct implementation. It adds no SMT dependency and no approximation cap. The existing scanner remains the high-recall candidate generator; a deterministic lightweight proof layer may discharge a candidate only when finite MIR facts prove the exact sink obligation on every reaching path. Unknown, ambiguous, stale, opaque, or resource-failed cases remain findings.

The standard-library development population motivates the work but does not define name-based rules. The target regression set is the 22 audited safe families (35 findings) in `STDLIB_ACTIVE_FINDING_AUDIT.md`; the historical `alloc::str::{join, concat}` family must remain reported. The 839 version-bound suppressions remain separate and are not automatically reclassified by this work.

## Problem

The v5 population contains 876 raw findings grouped into 77 root-cause families. Public-root fanout and multiple real call routes inflate the raw count, but family-level precision is still inadequate. The post-v5 audit closed 22 active families by source proof. Their proofs repeatedly use the same relations:

- a loop or search result is below an exact extent;
- a normalized range satisfies `start <= end <= len`;
- a returned prefix or suffix is contained in an initialized region;
- normal return from reserve establishes capacity for a requested length;
- pointer offsets remain within the exact base object;
- state updates and unwind ordering preserve an initialization invariant.

The analyzer already has point-sensitive length, checked-add, range, UTF-8, storage-version, validation, and local-output facts. The missing capability is a reusable finite relational vocabulary and a way to carry those relations through local assignments and resolved-local calls.

## Goals

1. Discharge the audited 22 safe families through generic MIR relations rather than standard-library names or suppression entries.
2. Preserve the historical join/concat UB finding and every existing positive fixture.
3. Preserve exact least-fixed-point behavior, normal/unwind separation, storage-version invalidation, and deterministic `BTree*` ordering.
4. Add no path-depth, fact-count, version-count, recursion, or round cap.
5. Keep the analysis practical by limiting the new domain to finite static places, constants, program points, call mappings, and predicates and by reusing the existing sparse final-entry replay.
6. Keep LLM review downstream. LLM output may explain or propose a relation but cannot suppress a finding without acceptance by the deterministic proof layer.

## Non-goals

- General theorem proving, arbitrary nonlinear arithmetic, or an SMT-backed verifier.
- Inferring every private representation invariant represented by the 839 v5 hard suppressions.
- Function-name, def-path-prefix, line-range, or standard-library-only sanitizers.
- Changing the P1–P6 registry, receipt schema, finding ID, causal key, or witness validity rules.
- Treating family aggregation as semantic suppression. Raw active findings remain available.

## Architecture

The pipeline becomes:

```text
MIR normalization
  -> existing high-recall origin/validation dataflow
  -> finite relational facts and normal-return summaries
  -> exact local sink proof
  -> existing unresolved requirement propagation
  -> findings, family presentation, optional LLM review
```

Relational facts participate in the same CFG fixed point as existing must validations. The dataflow predicate type becomes `ProofBinding::{Validation, Relation}`; `BoundValidation` continues to seal each binding to the complete current storage-version sets, intersect it at predecessor joins, and kill it on writes. This avoids treating a relation seen on only one reaching path as a proof. Interprocedural summaries are inferred only for resolved-local MIR bodies and solved synchronously over the existing call-graph SCCs. The proof layer runs before a local requirement is exported: if the sink snapshot proves the requirement, it is removed; otherwise the unchanged requirement enters the existing Task 6/8 propagation and witness machinery.

## Finite relational domain

### Terms

`RelationTerm` identifies only finite normalized entities:

- `Place(PlaceKey)` for a current value;
- `Length(PlaceKey)` and `Capacity(PlaceKey)` for an exact base object;
- `Const(u64)`;
- `Return` and `Formal(u32)` in a summary.

Every place-backed term is bound through `BoundValidation` to `canonical_storage(place)` and the complete current set of `StaticWriteToken`s for that storage. A fact is current only when all sealed sets exactly match the state. Exact writes replace the storage version and remove bindings that mention it; may-writes grow the version set and also remove those bindings. At CFG joins a relation survives only when the same semantic relation occurs on every reachable predecessor; its version sets union, preserving loop monotonicity.

### Relations

The first implementation supports only relations needed by the audited proof groups:

- `Eq(left, right)`;
- `Le(left, right)` and `Lt(left, right)`;
- `AddConst(result, base, k)` where checked or branch-proved arithmetic establishes no overflow;
- `Sub(result, upper, lower)` with a proven `lower <= upper`;
- `SumEq(total, left, right)`;
- `Within { base, start, end }`, meaning `start <= end <= Length(base)`;
- `InitializedWithin { base, start, end }`;
- `CapacityFor { base, end }`, meaning `end <= Capacity(base)`;
- `SameObject(left, right)` for exact collection/pointer identity.

Facts contain only terms drawn from finite MIR places, constants and boundary slots, so the lattice is finite. A relation's semantic identity intentionally excludes its evidence location, allowing the same relation established at distinct predecessor points to survive must intersection. `BoundValidation` separately seals the reaching-definition versions. `ProofBinding::Relation` facts intersect at CFG joins using the existing must lattice. An unsupported operand, opaque mutation, mixed object identity, unknown projection, or mismatched version prevents proof.

### Arithmetic closure

Closure is deliberately small and deterministic. It computes only:

- reflexive equality and equality substitution;
- transitivity of `Le`/`Lt`;
- `Lt(x, y) => Le(x + 1, y)` only when `x + 1` is represented by a current `AddConst` fact;
- `SumEq(t, a, b)` plus nonnegative component facts needed to prove each component is at most `t`;
- `Sub(r, u, l)` plus `l <= u` to prove `r <= u` and the corresponding contained range;
- `Within` projection to the exact bounds required by a sink;
- `CapacityFor` projection to a pointer or spare-capacity access on the same base.

There is no search over arbitrary expressions and no dynamic path enumeration.

## Normalization and transfer

MIR binary operations are no longer all represented as comparisons. Normalization freezes `Add`, `Sub`, comparison, checked arithmetic, and range construction as distinct DTO variants. Transfer creates a relation only when operand identity is exact and arithmetic safety is established by a normal edge, an assert/switch polarity, or a checked operation. Raw, wrapping, saturating, ambiguous, and overflow-capable arithmetic does not silently become an exact relation.

Existing assignment, call-return, mutable-actual, FFI-out, and Task 7 out-write hooks already kill every `BoundValidation` tied to affected storage; relation bindings use the same mechanism. Normal and unwind edges remain distinct; return facts and postconditions are installed only on the normal edge.

## Interprocedural summaries

`FunctionSummary` gains a finite `relational_postconditions` set. A postcondition uses only formals, return, out slots, constants, and normal-return predicates. It never contains a path, depth, local place, preferred witness, or solver state.

Summary inference projects a relation from each reachable normal return only when:

1. every reachable normal return proves the same canonical relation;
2. every referenced local maps exactly to a formal, return, or out slot;
3. storage identity is current and non-havoced;
4. unwind paths contribute no postcondition;
5. opaque and foreign callees contribute no fabricated relation.

At a resolved-local call, actual/formal and return/destination mappings instantiate the callee postcondition onto caller places. The existing synchronous SCC fixed point unions only canonical finite summaries. Recursive bodies without a grounded relation remain unknown; no iteration cap is introduced.

Variant-conditioned results are represented by a `ConditionalPostcondition` tied to the exact returned discriminant and edge polarity. This supports facts such as `Some(index) => index < Length(exact_receiver)` without assuming them on the `None` path or after a mixed join.

## Sink proof

Each P1/P2/P4 bounds-like sink constructs a canonical proof obligation from its existing subject, collection, pointer base/offset, access width, and program point. A sink is discharged only when its sparse `ValidationSnapshot` contains current relations that prove every predicate against the exact same object.

The proof decision is binary:

- `Proved`: remove the local requirement and record an internal test-only proof reason;
- `Unknown`: keep the requirement unchanged;
- internal invariant failure or resource failure: fail closed and produce no valid receipt.

There is no `AssumedSafe` state. The proof reason does not enter finding IDs or receipts in this work package.

## Coverage of the audited families

The regression corpus is grouped by proof capability, not API name:

- loop bounds and fixed extents: `get_many_mut`, SipHash tail reads;
- contained prefix/suffix and live initialized ranges: array `IntoIter`;
- split/search return postconditions: `Split`/`RSplit`;
- range normalization and exact object identity: `OsStr::slice_encoded_bytes`;
- sum/subtraction and state-before-drop ordering: `VecDeque::truncate`;
- exact pointer base plus offset range: slice rotation;
- normal-return capacity and initialized spare range: `Cursor<Vec<u8>>`.

Production logic may recognize compiler/lang/diagnostic identities already used to normalize primitive operations, but it may not match these enclosing function names or source spans. Each group receives production-shaped positive and adversarial negative fixtures: wrong subject, wrong collection/base, stale write, one-branch-only proof, loop reuse, normal/unwind difference, and opaque-call havoc.

The join/concat historical UB remains because its second borrow can produce a copy length not related to the first-pass reserved length. The domain must see either mismatched definition/version identity or absence of a universal relation and return `Unknown`.

## Efficiency

No full-program symbolic executor is added. Relations use sorted finite sets in the existing must-binding state only while CFG solving/replay needs them. Persistent body snapshots store only current proof facts at call and sink observations. Summary postconditions are small projections of normal-return states.

Performance is accepted only if `rapx-perf` retains the existing development gates:

- `core`: at most 15 minutes and 8 GiB physical footprint;
- `alloc`: at most 30 minutes and 10 GiB;
- `std`: at most 60 minutes and 12 GiB.

If the relational layer threatens these gates, optimization must use indexing, interning, sparse observation, or demand-driven closure. It may not use truncation or silently skip proofs. A missed proof retains a finding.

## Verification

Implementation follows TDD in independently reviewable blocks:

1. domain C/A/I, version invalidation, branch joins, loops, and normal/unwind;
2. normalization and intraprocedural range/loop/capacity proofs;
3. exact resolved-local summary instantiation, including 64-hop, self-recursive, and mutual-recursive cases;
4. production-shaped fixtures for all eight audited proof groups plus the historical UB negative-control family;
5. exact-nightly full Rust tests and no-build Python tests;
6. challenge 27 twice and original 27 twice, preserving canonical finding sets;
7. debug, stock release, and `rapx-perf` canonical finding-set equality;
8. one development standard-library `core` scan, followed only on success by `alloc` and `std`.

Success requires the 22 audited safe families to disappear because of deterministic generic proofs, the historical UB family to remain, no unexplained existing positive to disappear, no negative fixture to gain a finding, and all performance gates to remain satisfied. If a proof group cannot be expressed safely in this finite domain, its finding remains; the implementation must not add a case-specific suppression to meet the count target.

## Reporting boundary

Raw candidates, deterministic proof discharges, hard suppressions, active findings, and LLM classifications are reported as distinct stages. Family aggregation is a presentation layer and never changes the underlying set. Results on the standard library remain development evidence, not an unseen-holdout precision estimate, and do not establish P3/P5/P6 real-project precision or cross-crate precision.
