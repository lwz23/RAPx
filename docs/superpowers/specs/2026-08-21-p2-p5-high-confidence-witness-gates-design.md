# P2–P5 High-Confidence Witness Gates Design

**Approved:** 2026-08-21

## Goal

Reduce obvious P2–P5 false positives without API/path allowlists and without deleting unresolved audit candidates. An active finding must carry a realizable symbolic contract-violation witness in the exact type, path, and call context. A complete safety proof is suppressed; incomplete analysis is unestablished.

## Classification contract

- `active`: exact corridor plus a satisfiable finite violation witness.
- `suppressed`: compiler/type, must-validation, relation, initialization, or concrete-layout evidence proves the obligation.
- `unestablished`: an exact candidate exists, but its contract, normalization, controller, relation, validity, or initialization state is incomplete.
- Existing P2–P5 positive mechanism oracles remain active. Test expectations may not be weakened to manufacture lower counts.
- Family discovery remains separate from classification. Unestablished decisions retain stable identity and a reason.

## Pattern gates

### P2

Route `PersistentPublicField` through the existing exact influence solver. Require an exact field-to-role corridor, a modeled role-scoped sink requirement, compatible compiler-derived capabilities, and an externally writable controller. Overwritten values, exact validation, incompatible roles, and opaque sinks must not remain active.

### P3

Classify unsafe results as `ViolationWitness`, `ProvenSafe`, or `Unknown`. Use the existing type-validity implication, site-current `MaybeUninit`/typed-write state, exact initialization witnesses, and exposure corridor. Public return and public out/FFI exposure use the same rule.

### P4

Classify internal derivations through finite bounds/range, boolean, layout/alignment, and typed-initialization obligations. Combine the exact endpoint corridor with the site-local must snapshot and compiler-derived derivation kind. A relevant derivation without a satisfiable negated obligation is unknown, not active.

### P5

Re-evaluate a source-local generic witness at each exact callsite. Apply compiler-derived type equality, legal projection certificates, const values, concrete size/alignment facts, and representation-preserving views. A legal remaining parametric counterexample stays P5; concrete violations reclassify to P4; failed normalization is unestablished.

## Compatibility and evidence

Keep current CLI and finding/family fields stable. Extend the existing `unestablished.jsonl` population to P2–P5 without removing P1/P6 support. Every classification records a stable reason, requirement, source identity, and exact route or proof scope. No function names, source paths, or line numbers participate in semantics.

## Acceptance

- Existing P2–P5 positive mechanism roots remain active and negative roots remain non-active.
- Add general matrix cases for exact/overwritten/opaque P2, valid/invalid/unknown P3, guarded/violating/unknown P4, and equal/mismatched/incomplete P5.
- Pinned and current toolchain gates pass.
- Reclassify the frozen 48-family audit sample and review a fresh deterministic 48-family sample.
- The exact core development scan finishes within 900 seconds; peak memory is recorded but not gated.
- No target finding-count reduction is used as an acceptance criterion.
