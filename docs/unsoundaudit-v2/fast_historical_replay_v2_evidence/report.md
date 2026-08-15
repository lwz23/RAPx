# Fast Historical Replay v2

Replay v2 repairs the v1 admission failure: validated historical revisions are scanned at their exact vulnerable package/target, lockless libraries receive a deterministic lock only inside a disposable copy, all repository-local receipts are retained, and failures never render as zero findings.

## Outcome

| classification | cases |
|---|---:|
| `input_invalid` | 3 |
| `detected` | 3 |
| `validated_out_of_scope` | 4 |
| `successful_zero` | 0 |
| `partial` | 0 |
| `not_run` | 0 |

The validated, successful, in-scope denominator is three cases, and all three were detected. This is a closed historical-case result, not an estimate of ecosystem recall.

| case | expected rule | pre-fix target findings | fixed target findings | classification |
|---|---|---:|---:|---|
| Gitoxide #1460 | `P3.unchecked_utf8` | 2 | 0 | detected |
| lru-rs #73 | `P3.assume_init_uninitialized` | 6 | 0 | detected |
| rand #1158 | `P4.bounds` | 2 | 0 | detected |
| libc #3560 | `P2.union_field_validity` | 0 | 0 | validated out of scope |
| rand #779 | `P1.raw_read` | 0 | 0 | validated out of scope |
| nix #1819 | `P6.ffi_out_invalid_value` | 0 | 0 | validated out of scope |
| Diesel #4223 | `P6.ffi_out_param` | 0 | 0 | validated out of scope |

Smallvec #343, http #639, and bumpalo #164 lack a verifiable vulnerable/fixed revision pair in the supplied historical input and are excluded as `invalid_input` rather than counted as misses.

## Implemented general repairs

- `P3.unchecked_utf8` now binds the obligation to the input bytes and treats `from_utf8_unchecked` as an immediate validity failure. Losing return provenance through a wrapper no longer erases the finding; a current-version validation of the same bytes still discharges it.
- `P3.assume_init_uninitialized` generalizes the old bool-only detector to definitely uninitialized `MaybeUninit<T>::assume_init` for arbitrary `T`, retaining the existing merge, write, opaque-call, unwind, and tombstone semantics.
- `P6.ffi_out_invalid_value` recognizes only a direct FFI write into definitely uninitialized storage followed by exposure as bool, char, or enum. It deliberately does not infer arbitrary callback, helper-owner, closure-capture, or cross-crate semantics.
- Optional candidate-lifecycle JSONL traces record seed, local discharge, return-exposure filtering, route/root classification, dedup, and emission without entering receipts or finding IDs.

## Explicit capability boundaries

- libc #3560 requires a sound union active-field model. A broad union rule was not promoted because the real fix population contains many representation/debug cases and a may-active approximation would add structural noise.
- rand #779 needs stronger initialized-region and raw-read ownership reasoning than the present P1 rule provides.
- nix #1819 transfers FFI-written storage through a helper/owner relation not represented by the finite direct-output model.
- Diesel #4223 captures an FFI output pointer in a closure before `NonNull::new_unchecked`. The one P6 finding seen in both revisions is in `raw.rs`, not the vulnerable `stmt.rs`, and is therefore not counted as a hit.

No project, function, path, or line whitelist was added. There is no SMT solver and no path, fact, version, recursion, SCC, or fixed-point cap. Replay v1 remains byte-for-byte unchanged.

## Evidence boundary

All seven valid cases completed under the 900-second and 8-GiB physical-footprint gates. Generated locks are bound by SHA-256 and were created only in disposable workspaces. `summary.json` contains the exact source revisions, target-path finding counts, generated/committed lock hashes, elapsed time, peak physical footprint, rule IDs, and target finding IDs used for this classification.

The current evidence is one measured pre-fix/fixed pass per valid case plus deterministic unit/contract gates. The final challenge and original 27-case suites were each run twice and were byte-identical. It does not claim the stronger two-run identity gate for every historical package or newly rerun debug/release/`rapx-perf` equivalence for this uncommitted source snapshot.

The initial historical rows are bound to the `historical_scan_*` binary and source hashes in `summary.json`. A final generic compatibility repair made `is_null() == false` validate an exact FFI-output pointer even though its value provenance is intentionally unknown; it does not treat an unchecked havoc value as safe. The final source/binaries are recorded separately as `final_validation_*`. Gitoxide was rerun with that final binary and retained the exact 2→0 target differential; its pre-fix/fixed result hashes are recorded in `verification`. The same snapshot passed 168/168 Rust library tests, a current-nightly compile check, the complete challenge 27/27 gate at normalized SHA-256 `33ca7788...b483f6`, and the original 27/27 gate at `3fa14caf...edc0aa`. The other historical package rows were not silently relabelled as executions of the later binary.
