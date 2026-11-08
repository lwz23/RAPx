# Missing-sanitizer warning v1

Locked 2026-09-15 after the teacher discussion: lifetime / `Sync` / Drop
cases are not folded into P1-P6. RAP instead emits a residual warning when a
public safe API can reach a pointer or index unsafe sink with no dominating
runtime sanitizer.

This is a missing-sanitizer audit, not an unsoundness proof.
`NonNull` means non-null, not "still allocated".

## Claim

Warn iff all of the following hold:

1. A `public_safe` function (existing `is_public_at_level(Reachable)` +
   `Safety::Safe`, not a closure) can reach the sink along the existing
   resolved-local call graph, including the zero-hop case where the sink
   lives in that public function.
2. The sink is bound to MIR (`bound_exact`). Unbound / elided / ambiguous
   sites are skipped.
3. The sink is in exactly one of the two disjoint v1 classes below.
4. The class sanitizer does not dominate the sink: there is no current
   `must_validation` of that predicate at the pre-op snapshot for the sink
   statement (CFG must-intersection, write-kill included).
5. No active P1-P6 finding already claims the same
   `(public_root_owner, unsafe_site_id)`.

## What counts as a check

Only RAP matchers that already exist:

- pointer: `ptr.is_null()` polarity that establishes `Predicate::NonNull`
- index: comparison with `len` (and the existing assert plumbing) that
  establishes `Predicate::InBounds`

`slice::get`, `as_ref`, `NonNull::new`, `unwrap`, SAFETY comments, lifetimes,
`T: Sync`, and Drop protocols do not count.

## Interprocedural

Use the sink body's local snapshot plus whatever formal `NonNull` /
`InBounds` contracts the existing summary exporter already installs.

Do **not** transport `self.field` projection validations across calls in v1.
A check on `self.base` in a wrapper, with the deref in a helper, may still
warn.

## Sink classes (disjoint)

Pointer sinks — obligation `nonnull` on the base pointer operand
(`Address`, else `Place`, else `Argument(0)`, else `Receiver`):

- `RawPointerDereference`
- `ptr::read` / `ptr::read_unaligned` (`ptr_read_v1`, `ptr_read_unaligned_v1`)
- `ptr::offset` (`ptr_offset_v1`)
- `ptr::copy` (`ptr_copy_v1`)
- `slice::from_raw_parts` (`slice_from_raw_parts_v1`)
- `CStr::from_ptr` (`cstr_from_ptr_v1`)

An in-bounds check on an offset must **not** discharge a pointer warning.

Index sinks — obligation `in_bounds` on the index operand
(`Index`, else `Argument(1)`):

- `slice::get_unchecked` / `get_unchecked_mut` (`slice_get_unchecked_v1`)
- `Vec::get_unchecked` / `get_unchecked_mut` (`vec_get_unchecked_v1`)

`get_unchecked(0)` / `NonEmpty` is out of v1 (P5 territory).

`ValidForRead`, alignment, and initialization are out of v1.

## Output

Not a finding, not P7, not a rustc diagnostic, not part of `observed_counts`.

`rap-unit-v3` optional array `missing_sanitizer_warnings` (not required, so
old receipts still validate). Same rows also go to MVP
`missing_sanitizer_warnings.jsonl`.

Each row:

- `public_root_owner`: function display path
- `unsafe_site_id`: existing site SHA-256
- `predicate`: `nonnull` | `in_bounds`
- `sink_model`: `raw_pointer_dereference` or the `KnownContractModel` id

At most one row per `(public_root_owner, unsafe_site_id)`. Two sites in one
function (offset + `ptr::read`) may yield two rows. `finding_count` is
unchanged.

## Acceptance

Positive on buggy historical crates, at least one warning each:

- Bumpalo `IntoIter::next` (commit `6be3953ed3ff81003d87ec67ab112975ffc04bbc`)
- Wasmer `write_u8` (commit `49782633c3d581338ca08c45b7b19c05fddc948b` / 3.1.0)
- event-listener `notify` (commit `531c106f0ee27bf3fa7cf1bb0a621e46a1a4c9ed`)

parking_lot is not a positive for this warning. Patches that fix lifetime /
`Sync` / Drop are **not** required to silence the warning.

Index v1 is fixture-only:

- private `self.i` into `get_unchecked` with no `i < len`, no active P1-P6
- negative: `i < len` dominates, no warning
- negative: public `i` is P1 only, no warning

Pointer fixture negatives: dominating `is_null` then deref; public pointer
parameter that is already an active P1.

## v1 fixture gap

Same-function `is_null` / `idx < len` on field-derived locals currently do **not** dominate `ptr::read` / `get_unchecked` in RAP snapshots (`current_validations` stays empty; field index is not in `proven_contract_sites`). v1 oracle therefore checks:

- positives: internal unchecked `ptr::read` and `get_unchecked`
- P1 dedup: public parameter sinks stay findings, no warning

Do not treat `read_null_checked` / `get_private_index_checked` as v1 negatives until snapshot NonNull / proven InBounds covers them.
