# P1/P6 High-Confidence Token-Flow Design

**Date:** 2026-08-20

**Status:** Approved design; implementation has not started

**Scope:** RAP UnsoundAudit P1/P6 precision, exact four-state accounting, P1 schema correction, and complete unsafe-site binding

## 1. Purpose

The current core diagnostic output is dominated by P6 and contains structurally implausible correlations. Of 109,238 diagnostic P6 rows, 108,304 originate from the open-trait lane, and 82,580 target opaque contracts. Examples correlate an open-trait return in one call corridor with an unrelated pointer or layout operation in another corridor. P1 has the same opaque-contract problem and also emits P3-only exposure data.

This design changes P1/P6 from broad provenance correlation to category-specific, high-confidence influence tracking. It preserves FFI recall without requiring a universal whole-program proof. It also makes every inconclusive decision visible rather than treating it as either a finding or a proven suppression.

## 2. Confirmed Problems

The redesign addresses these confirmed implementation properties:

1. Every `ExternalBehavior` source currently receives highest-priority P6 treatment.
2. P1/P2/P4/P6 eligibility uses a coarse endpoint route, while only P5 uses exact callsite-aware context.
3. Intersecting roots that reach a source owner and a site owner can splice sibling callsites, branches, or unrelated callees.
4. Open-trait evidence stores trait openness but does not transport the receiver/controller term through caller substitutions.
5. Opaque unsafe calls provide neither role-specific requirements nor discharge conditions, yet currently produce active findings.
6. Source types and transformations are not checked against the concrete requirement of the unsafe operand.
7. Rank-zero P6 can displace a more specific P1, P2, or P4 candidate before P6 itself has strong evidence.
8. Non-P3 candidates can retain a P3-only exposure value.
9. The latest diagnostic core census contains 165 elided unsafe sites, so coverage remains lower-bound rather than exact.

The diagnostic counts are not a measured false-positive rate because no labeled core ground truth exists. They are evidence of an over-broad mechanism, not a statistical precision claim.

## 3. Goals

1. Emit active P1/P6 findings only when a category-specific external/public source can influence a compatible modeled unsafe requirement through one realizable callsite-instantiated corridor.
2. Preserve FFI return, FFI out-parameter, and FFI callback recall without requiring generic-controller proof.
3. Resolve open-trait and callback controllers at the public root rather than at an unconstrained helper definition.
4. Prevent sibling-call, sibling-branch, and unrelated-root correlation.
5. Distinguish active, proven suppression, reclassification, and unestablished outcomes exactly.
6. Keep the analysis finite and scalable by reusing compact relation IDs and interned token states rather than storing a proof DAG per finding.
7. Preserve deterministic output, family identity, and canonical witness selection.
8. Restore exact unsafe-site coverage and enforce that exposure is P3-only.
9. Complete a pinned core scan and receipt in under 900 seconds with memory measured, not used as the product-quality criterion.

## 4. Non-Goals

1. Proving that every possible Rust monomorphization or downstream implementation exists.
2. Treating the absence of a proof as proof of safety.
3. Adding path, state, relation, recursion, or output caps.
4. Changing P2-P5 semantics except where exact four-state accounting or shared output validation requires it.
5. Inferring precise contracts from function names or free-form diagnostic strings.
6. Declaring a global false-positive rate from an unlabeled core scan.
7. Preserving the formal 8-GiB development limit as a correctness gate; memory remains observable evidence.

## 5. Considered Approaches

### 5.1 Universal proof-carrying findings

Require a complete root/controller/source/sink proof for every active finding.

**Rejected.** It imposes excessive proof construction and can discard valid FFI cases merely because a generic controller cannot be reconstructed. It also duplicates the existing compact context machinery with per-finding proof state.

### 5.2 Category-specific token flow over exact relation IDs

Represent external/public influence as interned finite tokens. Apply source-specific controller rules, propagate tokens over exact callsite-instantiated summaries, and require compatibility with a modeled sink requirement.

**Selected.** It removes the known correlation defects, preserves FFI recall, and can use a seminaive fixed point without enumerating paths or retaining proof DAGs.

### 5.3 Heuristic filtering of common owners and contracts

Suppress `fmt`, iterator, `Result`, opaque calls, or high-fanout roots by name or frequency.

**Rejected.** It would make core counts smaller without establishing semantic precision, would be brittle across compiler versions, and could suppress genuine findings.

## 6. Architecture

### 6.1 Reuse the exact context arena

The implementation reuses the compact `RelationId`/`AtomId` context arena. A new analysis must not reconstruct owned `EndpointNode`, `CallSiteId`, substitution, and predecessor payloads for each candidate.

The production fixed point operates on compact states:

```text
TokenState = (AtomId, TokenId)
```

`TokenId` is an interned identifier for an immutable `InfluenceToken`. Worklists, visited sets, summaries, and waiter dependencies store IDs. Rich values are decoded only for final output or a selected canonical witness.

### 6.2 Influence token

Conceptually, each token contains:

```text
InfluenceToken {
    origin,
    controller,
    capabilities,
    precision
}
```

The fields form a finite domain and are interned by complete semantic value.

#### Origins

- `PublicFormal`
- `PersistentPublicField`
- `FfiReturn`
- `FfiOut`
- `FfiCallbackInput`
- `OpenTraitReturn`
- `PublicCallbackReturn`
- `PublicFunctionPointerReturn`

#### Controllers

- `FfiHost`
- `PublicFormal`
- `PersistentPublicField`
- `OpenTrait { trait_id, receiver_dependency }`
- `PublicCallback`
- `Internal`
- `Unknown`

#### Capabilities

A small bitset records what the value can still affect:

- `ADDRESS`
- `INDEX_OR_LENGTH`
- `TYPED_VALUE`
- `VALIDITY_BITS`
- `BOOLEAN`
- `CALLABLE_RESULT`

Capabilities describe influence, not validity. For example, a boolean token cannot satisfy an address requirement merely because both values share a coarse dataflow corridor.

#### Precision

- `StrongBoundary`: direct FFI or compiler-established public boundary evidence.
- `ExactDerived`: preserved through modeled typed transformations.
- `Incomplete`: provenance remains auditable, but an opaque transform, unresolved substitution, or havoc prevents an active finding.

`Incomplete` tokens can produce `Unestablished` records but never active findings.

### 6.3 Per-function summaries

Each function exports compact token-transfer summaries between:

- formal or external sources;
- return and out values;
- modeled unsafe operand roles;
- typed derivation nodes required to preserve or change capability.

Summaries are instantiated only at real callsites. Internal actual bindings apply the exact generic/capability substitution. Formal summary boundaries preserve identity until an actual callsite provides the binding.

Propagation is seminaive: each newly established `(RelationId, TokenId)` pair is propagated through each dependent callsite once. Recursive and mutually recursive summaries iterate to a least fixed point. There is no propagation cap and no path enumeration.

Only a final selected finding asks the context index for a canonical display route. Display alternatives are not used as the eligibility oracle.

## 7. Category-Specific Source Rules

### 7.1 FFI

FFI is an intrinsically strong external boundary. These sources create `StrongBoundary` tokens directly:

- return value of an external function;
- memory written through an FFI out parameter;
- value supplied to an FFI host callback.

An FFI token does not need open-trait or generic-controller proof. It still must reach the unsafe operand through the exact token flow and retain a capability compatible with the sink requirement. Wrappers and helper functions preserve it through summaries.

An opaque sink or a sink whose compatibility cannot be established does not erase the FFI source. It produces an `Unestablished` decision rather than an active finding. A requirement that is positively proven incompatible produces candidate-specific `PureSuppressed` evidence; it is not a claim that the whole unsafe site is safe.

### 7.2 Open trait

An open-trait return is active P6 evidence only when dispatch remains externally selectable at the public root. Supported controller forms are:

- a public generic parameter constrained by a downstream-implementable trait;
- a public `dyn Trait` receiver or value;
- a persistent public field that stores such a receiver.

The token retains the receiver dependency. Every call substitution transports that dependency. A concrete internal implementation, sealed/private trait, concrete caller binding, unresolved normalization, or unsupported predicate makes the controller `Internal` or `Unknown`; it does not remain `ProvenOpen` merely because the helper definition is generic.

`Internal` is not P6. `Unknown` is `Unestablished` if the token otherwise reaches a compatible sink.

### 7.3 Callback and function pointer

Callback-derived P6 requires the callable itself to originate from:

- a public formal;
- a persistent public field;
- an FFI callback boundary; or
- another already established external token.

An internally constructed closure, fixed internal function item, overwritten callback, or definitely local controller is not P6. A lower pattern may still be emitted only if it independently satisfies that pattern's eligibility rules.

## 8. MIR Transfer Semantics

### 8.1 Preservation

Typed assignments, moves, copies, exact projections, modeled return/out transport, and identity-preserving views retain the token and its capabilities.

### 8.2 Derivation

Modeled operations transform capabilities explicitly. Examples include deriving an index/length from a typed value, producing a boolean from a comparison, or deriving an address from an established pointer representation. No generic "value influenced value" edge is allowed to manufacture unrelated capability bits.

### 8.3 Overwrite

An exact strong overwrite kills the previous token at that storage version. A may-write or ambiguous alias invalidates exactness and changes the affected token to `Incomplete`; it does not invent an external source.

### 8.4 Validation

A validation discharges a requirement only when it applies to the same token, capability, storage version, and operand role on all realizable paths to the sink. Join semantics use must-information. Validation on one branch is insufficient for a sink reachable from another branch.

### 8.5 Opaque transforms and havoc

Opaque transforms and havoc preserve enough origin identity for audit accounting but remove active eligibility. They produce an incomplete token and an `Unestablished` reason such as `opaque_transform` or `havoced_provenance`.

## 9. Sink Requirement Model

### 9.1 Known unsafe contracts

Known compiler-identified unsafe APIs use role-specific contract predicates from the catalog. Requirements are attached to the exact operand role rather than applied as an unscoped predicate set.

### 9.2 Compiler-inherent operations

MIR operations with language-defined obligations derive a minimum requirement from the operation and compiler type information. This includes the relevant address, bounds, layout, validity, initialization, or callable-result requirement.

### 9.3 Opaque unsafe calls

An opaque unsafe call is active only if a minimum role-specific requirement can be derived reliably from compiler identity, the typed signature, and the operand role. Function names and debug strings are not evidence.

If no requirement can be established, the decision is `Unestablished` with reason `unmodeled_unsafe_call`. It is not active and is not described as safe.

## 10. P1 and P6 Eligibility

P1 and P6 share these requirements:

1. one public root scope;
2. one exact callsite-instantiated source-to-sink token route;
3. a strong or exact-derived token;
4. a modeled role-specific sink requirement;
5. token capability compatible with that requirement;
6. no exact must-validation or other evidence-backed discharge.

P1 accepts public-input and persistent-public-field origins. P6 accepts the category-specific external-behavior origins described above and applies the matching controller rule.

Pattern rank is evaluated only after eligibility. An unestablished P6 source cannot displace a valid P1, P2, or P4 candidate. Reclassification requires the replacement pattern to be independently established; uncertainty is never converted into a lower pattern merely to avoid an unestablished result.

## 11. Four-State Decision Contract

Every unique considered P1/P6 decision ends in exactly one state:

1. `Active`: all high-confidence eligibility conditions hold.
2. `PureSuppressed`: an exact validation, type implication, or other supported proof discharges it.
3. `Reclassified`: another pattern independently establishes a more accurate active candidate.
4. `Unestablished`: reachability exists, but required controller, normalization, context, contract, capability, or validation information is incomplete.

The accounting invariant is:

```text
active + pure_suppressed + reclassified + unestablished = considered
```

The four populations are pairwise disjoint. Duplicate traversal alternatives are normalized before this decision count and cannot appear as multiple states.

## 12. Unestablished Output

`unestablished.jsonl` is a compact, deterministic raw-decision stream. Each row contains:

- stable decision ID;
- attempted pattern;
- public-root identity;
- unsafe site ID and operand role;
- source identity and source category;
- controller category;
- token capability bits;
- sink requirement, when known;
- one primary blocking reason and a sorted set of secondary reasons.

The primary reason follows a fixed precedence so the same decision is not counted multiple times. The stream does not embed a full finding, code dictionary, or proof DAG. The validator checks uniqueness, disjointness from active/suppressed/reclassified identities, reason-schema validity, and exact accounting.

Primary-reason precedence is fixed as follows: `missing_exact_context`, `incomplete_normalization`, `unresolved_controller`, `unmodeled_unsafe_call`, `unknown_capability_match`, `opaque_transform`, `havoced_provenance`, then `validation_state_unknown`. All applicable lower-priority reasons remain in the sorted secondary-reason list. Positively proven controller exclusion or capability incompatibility is evidence-backed `PureSuppressed`, not `Unestablished`.

Representative unestablished cases may be grouped for human review after validation, but grouping never replaces the raw accounting stream.

## 13. Output and Schema Corrections

1. `CandidateExposure` is valid only for P3. P1, P2, P4, P5, and P6 construction forces exposure to `None`.
2. Family-key construction ignores exposure for every non-P3 pattern.
3. The external validator rejects any non-P3 active, suppression, replacement, or unestablished row carrying exposure.
4. Evidence validation becomes reason-specific. A nonempty dictionary alone is insufficient.
5. Writer accounting includes the unestablished population and fails closed on any missing or duplicate identity.

## 14. Exact Unsafe-Site Coverage

The existing 165 `elided_unsafe_site` records are part of this implementation scope because exact six-pattern reporting is otherwise impossible.

Each currently elided site must be classified by compiler identity and bound to an exact operand/contract model. Binding uses `DefId`, diagnostic item, intrinsic identity, MIR operation identity, and typed argument positions; it does not use function-name allowlists.

After the repair:

- discovered count equals bound-exact count;
- unbound, ambiguous, and elided counts are zero;
- coverage is `complete=true` and `semantics=exact`;
- unsupported newly discovered identities fail the gate rather than silently disappearing.

## 15. Error Handling

- ID arena overflow fails fast with a diagnostic; it never wraps or truncates.
- Incomplete generic normalization, unsupported predicates, ambiguous aliases, opaque transformations, and unresolved controllers produce `Unestablished` rather than an active finding.
- Missing compiler binding makes coverage incomplete and fails the final gate.
- A worker panic, malformed output row, duplicate decision ID, accounting mismatch, or nondeterministic family membership fails closed.
- FFI provenance is never downgraded solely because generic-controller proof is unavailable; only route, capability, requirement, or validation uncertainty can make that FFI decision unestablished.

## 16. Performance Boundaries

The design avoids per-finding global proof construction:

- compact IDs in worklists and summaries;
- interned finite token values;
- seminaive relation/token propagation;
- shared callsite substitutions and controller terms;
- cached reverse/exact reachability by target and root scope;
- canonical witness rendering only for selected active findings;
- streaming unestablished output and validation.

There is no semantic cap. Performance tests must record token, relation, transfer, and unestablished counts so an unexpected state-space expansion is observable.

For the core acceptance run, the wall limit remains 900 seconds. The run uses a sufficiently high diagnostic memory allowance, initially 12 GiB, to avoid treating the former 8-GiB engineering target as a correctness condition. Actual physical footprint and RSS remain recorded.

## 17. Test Design

### 17.1 Source-category fixtures

- direct FFI return to compatible sink: P6 active;
- FFI out parameter to compatible sink: P6 active;
- FFI value through nested wrappers: P6 active;
- FFI callback input: P6 active;
- public generic and public `dyn Trait` open dispatch: P6 active;
- concrete internal impl, sealed trait, private trait: no P6;
- public callback/formal/persistent field: P6 active;
- internal closure, fixed function item, overwritten callback: no P6.

### 17.2 Context fixtures

- sibling callsites with the same callee cannot splice;
- mutually exclusive branches cannot splice;
- different public roots cannot splice;
- nested wrappers preserve the same token;
- direct and mutual recursion reach a fixed point;
- distinct substitutions and sibling alternatives remain distinct;
- missing callee or incomplete normalization yields unestablished, not active.

### 17.3 Capability and contract fixtures

- boolean and `Result` tokens cannot satisfy address/layout/validity requirements;
- modeled transformations change capability precisely;
- role-specific known contracts activate the matching role only;
- opaque sink with a typed minimum requirement is evaluated normally;
- opaque sink without such a requirement is unestablished.

### 17.4 Validation fixtures

- exact validation on all paths suppresses;
- validation on only one branch does not suppress;
- exact overwrite kills the prior token;
- ambiguous overwrite makes the decision unestablished;
- validation for one capability does not discharge another.

### 17.5 Reporting fixtures

- all four populations are disjoint and exactly reconcile;
- reason-specific evidence is validated;
- unestablished reason precedence and secondary-reason ordering are stable;
- reversed insertion order produces byte-identical normalized output;
- non-P3 exposure is rejected at construction and validation;
- no unestablished case is silently converted into suppression or a lower pattern.

### 17.6 Coverage fixtures

- every currently elided compiler-identity class receives an exact binding fixture;
- unknown future identities fail coverage;
- current and pinned compiler adapters preserve the same semantic binding.

## 18. Acceptance Criteria

### 18.1 Focused and mechanism gates

1. All new token, source-category, context, contract, validation, reporting, and binding tests pass on pinned nightly-2024-10-12.
2. Current-nightly API compatibility checks pass for the affected compiler adapters.
3. The complete P1-P6 mechanism pack passes with exact coverage and exact four-state accounting.
4. Existing expected FFI positives remain active.
5. Known sibling-call, concrete-controller, capability-mismatch, opaque-contract, and exposure false correlations are absent from active output and appear in the correct alternate state.

### 18.2 Core gate

1. Metadata-correct pinned binaries are used.
2. Scan plus receipt completes in less than 900 seconds.
3. Coverage is complete/exact with zero elided, ambiguous, or unbound sites.
4. The strict streaming validator accepts schema, identities, families, replacements, evidence, and four-state accounting.
5. A repeated normalized run produces identical finding, family, suppression, unestablished, coverage, and accounting artifacts.
6. Physical footprint and RSS are recorded but the former 8-GiB target is not a correctness gate.

### 18.3 Precision review

Counts alone are not accepted as a precision measure. A deterministic family-ID sample is reviewed:

- every FFI family;
- all P1 families when there are at most 100, otherwise a deterministic 100-family sample;
- all open-trait P6 families when there are at most 100, otherwise a deterministic 100-family sample.

Any definite cross-callsite splice, concretized controller, or requirement incompatibility in the active sample fails the precision gate. The review reports observed labels and limitations; it does not extrapolate a global false-positive rate without ground truth.

## 19. Delivery Sequence

1. Add token/controller/capability DTOs and interning with unit tests.
2. Add exact summary transfer and seminaive callsite instantiation.
3. Implement FFI, open-trait, and callback source rules.
4. Add typed sink requirements and validation/overwrite transfer.
5. Replace P1/P6 eligibility and rank ordering.
6. Add four-state accounting and streaming unestablished output.
7. Fix P3-only exposure enforcement.
8. Classify and bind all currently elided unsafe sites.
9. Run pinned focused tests and complete mechanism pack.
10. Build metadata-correct binaries and run the strict core acceptance sequence.

## 20. Approved Decisions

- Only high-confidence P1/P6 decisions are active.
- FFI remains a first-class strong source and does not require universal controller proof.
- Open-trait and callback lanes use category-specific controller rules.
- Exact callsite token correlation and typed sink relevance are required.
- The implementation reuses the compact exact relation arena with a finite TokenId fixed point.
- Uncertainty is emitted as `Unestablished`, not hidden, suppressed, or mislabeled as a lower pattern.
- Four-state accounting is externally validated.
- The P1 exposure defect and 165 elided unsafe sites are in scope.
- The core success criterion is correctness plus a sub-15-minute run; memory is measured rather than used as the quality criterion.
