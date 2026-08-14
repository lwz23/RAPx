# RAPx Latest Standard-Library Differential Soundness Audit Design

## Status and decision

This design is approved. The primary research target is the standard library from a frozen current Rust nightly/main commit. Current stable is a confirmation target, not a substitute for the primary scan. The work seeks safe-public-API soundness defects that remain in current source; historical-only or already-fixed defects remain validation evidence and are not reportable as current bugs.

The selected strategy is a differential audit with a full-scan guardrail. RAPx scans the complete current `core`, `alloc`, and `std` populations, because that machine pass is inexpensive and catches findings exposed by compiler, MIR, or analyzer changes. Human review is restricted to new or semantically impacted families rather than repeating the 2024 population audit.

## Motivation

RAPx previously scanned the standard library corresponding to `nightly-2024-10-12`. That scan found one real historical soundness family, but current Rust has already fixed it. Repeating the entire historical manual review on current source would waste effort. Conversely, searching only source lines containing a changed `unsafe` token would be incomplete: a memory-safety precondition can change when a safe caller, validation helper, trait implementation, private representation invariant, or compiler lowering changes while the final unsafe operation remains textually unchanged.

The useful compromise is therefore:

1. freeze and analyze one current Rust source/toolchain identity;
2. retain a complete current raw population as a cheap guardrail;
3. deterministically identify the families affected since the historical baseline;
4. spend source-proof and PoC effort only on that differential population.

Finding a current bug is an objective, not an acceptance condition. A negative result remains valid research evidence. Publication value must not lower the proof threshold or bias candidate classification.

## Goals

1. Port RAPx to a frozen current nightly without changing the P1–P6 registry, causal identity, proof semantics, or no-cap fixed points.
2. Scan the exact corresponding current `core`, `alloc`, and `std` sources with `rapx-perf` while the target remains ordinary development `cargo check`.
3. Build a deterministic differential inventory from the 2024 baseline to the current source.
4. Audit every new or semantically impacted active family for a safe public call chain and the exact unsafe precondition.
5. Validate plausible current defects with a dependency-free safe-only PoC and Miri where Miri supports the operation.
6. Search upstream source, issues, pull requests, and security records before preparing any disclosure.
7. Keep disclosure as a separate explicit approval and never open an issue automatically.

## Non-goals

- Re-auditing every unchanged historical family.
- Treating the absence of a changed `unsafe` keyword as proof that a sink is unaffected.
- Carrying the 2024 rustc-bound hard-suppression table onto current Rust.
- Expanding P1–P6, adding SMT, or introducing path, fact, version, recursion, or fixed-point caps.
- Claiming that a clean Miri run proves an API safe.
- Native execution of a suspected-UB candidate before its behavior and evidence boundary are understood.
- Reporting a historical duplicate, a fixed defect, or a candidate that requires an unsafe caller-contract violation as a current Rust bug.

## Frozen identities

Each campaign freezes and records:

- the current nightly rustc commit and release date;
- the exact `rust-lang/rust` source commit and rust-src tree hashes;
- Cargo and Miri commits;
- the RAPx source commit and binary hashes for `cargo-rapx` and `rapx`;
- the historical baseline rustc/source identity;
- the source-diff manifest and its SHA-256;
- all scan, normalization, differential, PoC, and review artifacts.

Moving aliases such as `nightly`, `stable`, or `main` are resolved before any measurement. A later upstream commit starts a new campaign rather than silently changing the target.

## Pipeline

```text
freeze current Rust identity
  -> port RAPx and pass semantic regression gates
  -> full current core/alloc/std machine scan
  -> canonical current-family population
  -> historical/current source and finding reconciliation
  -> impacted-family differential
  -> source-contract audit
  -> safe-only PoC and Miri where justified
  -> stable/main confirmation and upstream duplicate search
  -> disclosure draft waiting for explicit approval
```

The complete current scan is not a complete current manual review. Its purposes are to prevent a source-diff filter from hiding a new finding and to detect changes caused by the newer compiler/MIR or by the newer relational analyzer.

## Differential inventory

### Directly changed sites

A family is impacted when its exact sink, enclosing function, registry source, validation branch, or any selected witness call edge is added, deleted, or changed between the historical and current source identities.

### Contract-dependency changes

A family is also impacted when a change touches a safety fact used to justify its sink, including:

- a safe caller or forwarding wrapper on the selected route;
- a guard, checked-arithmetic helper, range normalizer, or returned-index contract;
- a trait implementation or safe callback contract used by the path;
- a private type representation, constructor, mutation method, or drop/unwind invariant;
- an allocator, capacity, initialization, provenance, or aliasing condition used by the proof.

This dependency relation is conservative. An unresolved mapping or unavailable source identity marks the family impacted rather than unchanged.

### Analyzer and compiler deltas

Every current family with no historical semantic counterpart is audited even when its source text is unchanged. This captures candidates exposed by current MIR lowering, rustc-private API changes, or the relational refinement. A historical family whose sink identity matches but whose canonical obligation, source kind, propagation depth class, or route structure changes is also impacted.

### Unchanged families

A family may be classified `not_reaudited_unchanged` only when its source and dependency fingerprints match exactly and it has a historical family counterpart. This label means only that it was outside the differential review scope. It is not a new safety proof, current-version suppression, or precision claim.

## Current-source audit

Review order prioritizes:

1. new current families with no historical counterpart;
2. changed safe-public roots or safety guards around an existing unsafe sink;
3. user-implementable safe traits, callbacks, interior mutability, reentrancy, panic/unwind, allocation failure, integer overflow, and alias/provenance boundaries;
4. new or changed interprocedural and recursive routes;
5. changed private representation invariants;
6. remaining impacted families ordered by stable family ID.

Each audited family records the exact sink precondition, complete safe call chain, guards and invariants, mutation and unwind points, same-object binding, version status, and classification. The allowed classifications are:

- `confirmed_safe_current` for a closed proof of the exact reported obligation and enumerated safe roots;
- `poc_required_current` when a safe caller may plausibly violate the precondition;
- `confirmed_ub_current` only with a safe boundary and source or dynamic evidence of UB;
- `historical_or_fixed` when current source closes the defect;
- `duplicate_current` when an upstream report or fix already covers the exact mechanism;
- `inconclusive_current` when evidence is insufficient.

## PoC and Miri boundary

A PoC must be dependency-free, deterministic, minimal, and contain no application-boundary `unsafe`. It must reach the same unsafe sink and violate the same precondition as the finding, with a non-triggering control. Toolchain, command, profile, source hashes, exit status, and raw-log SHA-256 are recorded.

Miri UB through the safe public call chain can confirm the executed case. An ordinary panic is not UB, and a clean Miri result is not a proof. If Miri cannot model the relevant operation, the classification remains source-based or inconclusive. Suspected-UB cases are not run natively as a substitute for Miri.

## Upstream and disclosure boundary

For each `confirmed_ub_current` candidate:

1. reproduce or prove it on the frozen current nightly source;
2. determine whether current stable contains the same defect;
3. search official Rust issues, pull requests, commits, release notes, advisories, and security guidance for the exact mechanism;
4. prepare a sanitized technical package with source identity, safe PoC, Miri/source evidence, affected versions, suggested invariant or repair, and duplicate-search record;
5. stop for explicit user approval before contacting the Rust project.

The final routing decision between a public issue and the Rust security process follows the current official policy and the defect's demonstrated impact. This design authorizes research and draft preparation, not external submission.

## Port and regression gates

The current-nightly port may adapt rustc-private APIs and MIR normalization, but it may not silently change analysis semantics. Before scanning current std, it must pass:

- the complete exact Rust library suite;
- all no-build Python contracts;
- challenge 27 twice and original 27 twice with canonical reconciliation against the approved baseline;
- debug, stock release, and `rapx-perf` canonical finding-set equivalence;
- the 64-hop, direct-recursion, mutual-recursion, normal/unwind, local-output, relation-invalidation, and route-witness gates;
- deterministic serialization and schema validation.

Compiler-driven MIR changes may legitimately alter full witnesses or candidate details. Every canonical difference is reconciled and explained; no unexpected loss of an existing positive is accepted as a porting convenience.

## Differential-tool verification

The source/finding reconciliation receives no-build tests for:

- comments and whitespace not creating a semantic source change;
- function moves and line shifts not being mistaken for new unsafe mechanisms;
- an unchanged unsafe operation with a changed guard being marked impacted;
- a changed safe caller, trait implementation, constructor, representation field, and unwind path being marked impacted;
- a new finding on textually unchanged source being marked impacted;
- missing baseline/source data failing closed;
- duplicate and ambiguous mappings failing closed;
- input reversal producing byte-identical manifests and family inventories.

The tool reports raw current families, impacted audited families, unchanged-not-reaudited families, historical/fixed families, duplicates, and unresolved mappings separately.

## Performance and stopping rules

The existing release gates remain:

- `core`: 15 minutes and 8 GiB physical footprint;
- `alloc`: 30 minutes and 10 GiB;
- `std`: 60 minutes and 12 GiB.

Resource exhaustion is fail-closed and produces no valid unit receipt. `core` is run first; `alloc` and `std` follow only after it passes. The full machine scan is retained because its measured cost is small relative to source audit and PoC work.

The research stops for redesign if the current-nightly port changes candidate semantics without a defensible MIR explanation, if the differential mapping cannot fail closed, or if a plausible current defect cannot be investigated without violating a caller unsafe contract. It does not broaden rules merely to force discovery of a publishable bug.

## Reporting boundary

The report distinguishes:

- historical raw and active populations;
- complete current raw families;
- impacted families actually reviewed;
- unchanged families not re-audited;
- current confirmed-safe, PoC-required, confirmed-UB, fixed, duplicate, and inconclusive families;
- Miri-supported evidence versus source-only proof;
- current stable impact versus nightly/main-only impact;
- disclosure drafts versus externally submitted or accepted reports.

The work may support a claim that RAPx differentially audited current Rust standard-library changes for the fixed P1–P6 classes. It does not establish absence of all stdlib soundness bugs, precision for unmodeled unsafe patterns, or a guaranteed bug discovery.

## Rejected alternatives

- **Re-audit every current family:** unnecessary human cost; the full machine population remains available as a guardrail.
- **Search only lines containing changed `unsafe`:** misses changed callers, contracts, invariants, traits, and compiler-exposed findings.
- **Reuse the 2024 suppression table on current Rust:** suppressions are exact rustc/source-bound and cannot cross versions.
- **Skip the full current scan:** saves little runtime and removes the only guardrail for compiler/analyzer-induced new families outside the textual diff.
- **Assume current bugs must exist:** creates confirmation bias and weakens the evidence standard.
- **Automatically submit a GitHub issue:** duplicate/security routing and explicit authorization must be resolved first.
