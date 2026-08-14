# UnsoundAudit v2 Standard-Library Active-Finding Validation Design

## 1. Purpose

This work determines whether any of the 37 active v5 standard-library findings represents a real Rust standard-library soundness bug. An active finding is an unsuppressed static-analysis candidate, not a confirmed bug. The 37 findings form 23 exact families and nine source-level root-cause shapes.

The work prioritizes evidence over finding count or publication value. A finding becomes a confirmed standard-library soundness bug only when a safe public call chain can violate Rust's memory-safety rules without relying on an unsafe caller contract violation. Miri evidence may confirm an executed undefined-behavior path, but a clean Miri run does not prove a family safe.

## 2. Evidence and Claim Boundary

- The immutable v4 population and v5 active population remain unchanged.
- The previous one-person review inspected all 23 active families and labeled them B: their causal chains were structurally plausible, but the available contract evidence was incomplete.
- No active family has yet been labeled A or reported upstream.
- This phase may read pinned and current rust-src, construct isolated PoCs, install and run the matching Miri component, and search current upstream source and issue history.
- This phase does not alter P1-P6, the analyzer, v5 suppression tables, receipts, or pilot conclusions.
- No external Rust issue or security report is submitted until the evidence package receives a separate user approval.

The exact historical target is rustc commit `1bc403daadbebb553ccc211a0a8eebb73989665f` from `nightly-2024-10-12`. A reportable current bug must also reproduce on a current supported toolchain or be shown to remain in current upstream source. A historical-only bug is recorded as such and is not presented as an unfixed vulnerability.

## 3. Active-Family Inventory

| Root-cause shape | Families | Findings | Preliminary status |
|---|---:|---:|---|
| `IntoIter::advance_by/advance_back_by` range helpers | 4 | 6 | Likely analyzer summary loss; `take_prefix/take_suffix` clamps the range |
| Slice `Split/RSplit` using `position/rposition` | 8 | 8 | Likely analyzer return-contract loss; returned indices are in the searched slice |
| `VecDeque::truncate` branch arithmetic | 4 | 8 | Likely missing guard recognizer |
| Slice `rotate_left/rotate_right` pointer rotation | 1 | 2 | Likely cross-call loss; public methods assert the count is at most the slice length |
| `get_many_mut` checked-to-unchecked handoff | 2 | 2 | Likely validation-summary loss; indices are checked for bounds and uniqueness |
| SipHash tail read | 1 | 4 | Likely arithmetic/caller-contract loss; requires a complete start/length proof |
| String `join/concat` spare-capacity range | 1 | 2 | High-priority PoC candidate because safe user-defined `Borrow` may vary across calls |
| `OsStr::slice_encoded_bytes` range | 1 | 1 | Likely return-contract loss from `slice::range` plus boundary checks |
| `Cursor<Vec<u8>>` reserve/spare-capacity range | 1 | 4 | Likely checked-capacity summary loss; overflow and allocation behavior require proof |
| **Total** | **23** | **37** | No confirmed bug yet |

## 4. Validation Strategy

### 4.1 Contract-first triage

Each family receives a dossier containing:

1. the exact active finding IDs and public roots;
2. the unsafe operation and its documented preconditions;
3. the complete safe call chain that reaches it;
4. every guard, arithmetic relation, type invariant, and mutation between guard and sink;
5. whether a safe caller can violate the precondition;
6. the result: `confirmed_safe`, `likely_safe`, `poc_required`, `confirmed_ub`, or `inconclusive`.

`confirmed_safe` requires a source-level proof covering all public inputs and callbacks. `likely_safe` is not enough to suppress or close the family. Families that depend on user-defined safe traits, callbacks, allocation failure, integer boundaries, panic/unwind behavior, or interior mutability default to `poc_required` unless a complete proof closes them.

### 4.2 PoC requirements

A reportable PoC must:

- use only public safe Rust at the application boundary;
- avoid forging invalid references, raw-pointer misuse, invalid `unsafe impl`, FFI contract violations, or unsupported compiler flags;
- minimize unrelated allocation and concurrency behavior;
- exercise the same unsafe sink and contract failure identified by the finding;
- have a control variant that does not trigger the suspected condition;
- record exact source, toolchain, command, exit status, and Miri diagnostic.

Property tests may search inputs, but the final evidence must be a deterministic minimal case. Native execution is unnecessary for confirming UB and is avoided until the Miri behavior is understood.

### 4.3 Version matrix

Each PoC is evaluated in this order:

1. `nightly-2024-10-12` with its matching Miri component, reproducing the scanned source;
2. current stable with a compatible current Miri toolchain where possible;
3. current nightly and current upstream rust-src;
4. upstream issue and pull-request search for an existing report or fix.

The historical and current results are reported separately. A failure to reproduce under Miri is recorded as negative dynamic evidence, not a proof of safety. If Miri cannot model a required operation, the family remains source-audited and inconclusive unless another sound dynamic method is available.

## 5. First Candidate: String `join/concat`

The first PoC targets `library/alloc/src/str.rs:130-184` in the pinned source. `join_generic_copy` calculates `reserved_len` by calling `Borrow::borrow`, then calls `borrow` again while extending and copying. `Borrow` is a safe trait, so library memory safety cannot depend on an undocumented unsafe implementation obligation.

The initial hypotheses are:

- a safe `Borrow<str>` implementation can use interior state to return a short string during length calculation and a longer string during copying;
- the first element can make `result.len()` exceed `reserved_len`, reaching `reserved_len - pos` before the unchecked spare-capacity range;
- later elements may instead be stopped by safe `split_at_mut`, so first-element and later-element cases must be separated;
- debug, optimized, pinned, and current library builds may differ because integer-underflow behavior and the standard-library implementation may have changed.

The PoC matrix covers `concat`, `join`, first-element variation, later-element variation, shorter-to-longer and longer-to-shorter transitions, empty separators, and a stable-borrow control. The goal is not random fuzzing; it is to decide whether safe trait behavior can reach the exact unchecked range with an invalid bound.

## 6. Remaining-Family Order

After the string candidate, review proceeds by potential for safe adversarial input:

1. `Cursor<Vec<u8>>` capacity and integer-boundary arithmetic;
2. SipHash tail arithmetic and state transitions;
3. callback-driven `Split/RSplit` behavior;
4. panic/unwind behavior in `VecDeque::truncate` and array `IntoIter` dropping;
5. `rotate`, `get_many_mut`, and `OsStr` range contracts.

Clearly proven false positives stop at a proof dossier. Miri suites are written only where a safe caller, callback, trait implementation, integer boundary, or unwind path could plausibly invalidate the proof.

## 7. Disclosure Gate

A candidate is eligible for an upstream draft only if all of the following hold:

- the triggering application contains no unsafe block or unsafe implementation relevant to the failure;
- Miri reports UB, or an equivalent source-level proof demonstrates unavoidable UB;
- the path reaches the identified standard-library unsafe operation;
- the issue is not solely a documented panic, abort, OOM, or caller-contract violation;
- current upstream remains affected or the historical impact is accurately bounded;
- an issue/PR search finds no duplicate report.

The draft includes the minimal PoC, exact Miri output, affected versions, root-cause analysis, impact, suggested fix direction, and a statement that the discovery originated from UnsoundAudit. Security-sensitive details are routed through the current official Rust security process rather than posted publicly. Submission is a separate external action requiring user approval.

## 8. Artifacts and Isolation

- Design and final audit tables are tracked under `docs/unsoundaudit-v2/` or `docs/superpowers/`.
- PoC source may be tracked in a dedicated small validation directory after its purpose is approved in the implementation plan.
- Cargo homes, targets, Miri sysroots, raw logs, core dumps, and machine paths remain outside the repository.
- Sanitized Miri diagnostics receive SHA-256 digests; raw local logs are not committed.
- `TASK_CONTEXT.md` records decisions, candidate status, and evidence boundaries.

## 9. Success Criteria

This phase succeeds when:

- all 23 active families have an explicit source-level disposition;
- every `poc_required` family has a deterministic pinned-version result;
- every pinned-version UB has a current-version result and duplicate-report search;
- confirmed-safe and inconclusive families are not described as bugs;
- every confirmed current UB has a reviewable disclosure package;
- no upstream submission occurs without the separate approval gate.

Finding no true bug is a valid result. The paper may report the number of candidates closed by proof, rejected by Miri, left inconclusive, and confirmed, but it must not optimize labels for novelty.

## 10. Rejected Alternatives

- Treating all 37 active findings as bugs.
- Writing 23 broad fuzz harnesses before understanding their contracts.
- Using a clean Miri run as proof of safety.
- Triggering UB by violating an unsafe API's documented caller obligations.
- Testing only the historical nightly and claiming a current vulnerability.
- Publishing a public issue before checking current source, duplicates, and the official disclosure route.
