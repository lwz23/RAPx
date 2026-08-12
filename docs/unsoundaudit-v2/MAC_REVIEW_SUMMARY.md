# UnsoundAudit v2 Mac Review Summary

## Scope and Verdict

Mac validation passed for source commit `f5988cb156f6f3d8cda59b03ac085607156e6acc` on branch `feature/unsoundaudit-v2-p1-p6`.

The bounded completion claim is: P1–P6 heuristic candidate generators are implemented and verified against the 27 frozen fixture oracles. This does not claim real-project precision, recall, ecosystem coverage, or Linux validation.

No material scope deviation occurred. The implementation preserves the existing `-unsoundaudit` CLI and `LwzCheck::new(tcx).start()` entry, emits the closed `rap-unit-v2` contract directly, and does not modify the Cargo wrapper, root lockfile, nightly, dependency versions, or Linux evidence.

## Baseline and Toolchain Gates

The baseline and preflight evidence remains authoritative in:

- `artifacts/unsoundaudit-v2/mac/exact-nightly-2024-10-12-arm64/environment_receipt.json`
- `artifacts/unsoundaudit-v2/mac/exact-nightly-2024-10-12-arm64/baseline_receipt.json`

The frozen baseline tag/commit, restricted source diff, RAP source-tree hash, original `unsoundaudit.rs` hash, clean worktree, and exact baseline build passed before implementation. The Mac preflight verified native arm64 Xcode Command Line Tools, Homebrew LLVM/libclang, Z3, CMake, pkg-config, and architecture consistency.

The final validation used:

| Component | Pinned identity |
|---|---|
| Rust channel | `nightly-2024-10-12` |
| rustc commit | `1bc403daadbebb553ccc211a0a8eebb73989665f` |
| Cargo commit | `15fbd2f607d4defc87053b8b76bf5038f2483cf4` |
| Implementation commit | `f5988cb156f6f3d8cda59b03ac085607156e6acc` |
| RAP source-tree SHA-256 | `607544d36ba88f92507125b958ae64e3e280103e7467f985b7e5c185cce4deec` |

## Implemented Contract

The implementation is crate-local and follows only statically resolved local MIR calls. It uses ordered summary facts, Tarjan SCCs, equality-based fixed points without a hidden round cap, stable recursive cycle tokens, causal-key deduplication, and deterministic witnesses.

The registry is deliberately fixture-closed:

| Pattern | Implemented forms |
|---|---|
| P1 | Public safe parameters reaching raw reads or `get_unchecked`, including local helper chains |
| P2 | Literal public fields reaching the same frozen sinks |
| P3 | Lifetime-extension transmute to a public `&'static` return, known-uninitialized `MaybeUninit<bool>::assume_init`, and unchecked UTF-8 exposure |
| P4 | Internal-only bounds and derived pointer offset/access-width candidates |
| P5.1 | Generic or associated slice element-zero non-empty obligation, with `family_support: "partial"` |
| P6 | Exact `extern "C"` `*mut *mut T` output to `NonNull::new_unchecked`, and selected open `IndexSource::index -> usize` output to `get_unchecked` |

Unsupported/future forms include FFI returns, Iterator outputs, callbacks, general dynamic dispatch, raw reference/slice origins, `Box::from_raw`, and `set_len`. Ordinary unresolved or external Rust calls remain opaque and do not create speculative local edges.

## Verification Results

All validation used project concurrency 1, Cargo jobs 1, an isolated temporary Cargo home, and fresh targets.

| Gate | Result |
|---|---|
| Exact fresh binary build | Pass; `cargo-rapx` and `rapx` present |
| Pure Rust library tests | 26 passed, 0 failed, 0 ignored |
| Contract hardening tests | 14 passed, 0 failed |
| Frozen fixture suite | 27/27 passed |
| Positive fixtures | 12, each with exactly one primary finding |
| Negative and noise fixtures | 15, all with zero findings |
| Independent read-only review | `ship`, zero blocking findings |

The aggregate primary counts are:

| Pattern | Count |
|---|---:|
| P1 | 3 |
| P2 | 2 |
| P3 | 2 |
| P4 | 2 |
| P5 | 1 |
| P6 | 2 |

Each fixture run first performed `cargo check --lib --locked --jobs 1` in a fresh control target and then the static `cargo-rapx rapx -unsoundaudit -- --lib --locked --jobs 1` scan in a separate fresh target. No fixture target function was executed.

Two complete serial runs produced byte-identical normalized receipts. Both SHA-256 values, and the committed normalized result, are `3fa14cafd01a58b17a57094ae4c785d4dda306835adcb433219245e102edc0aa`. The normalized result contains no Mac user, target, cache, or temporary absolute path.

Machine-readable details are in:

- `artifacts/unsoundaudit-v2/mac/exact-nightly-2024-10-12-arm64/fixture_results.normalized.json`
- `artifacts/unsoundaudit-v2/mac/exact-nightly-2024-10-12-arm64/unit_test_summary.json`
- `artifacts/unsoundaudit-v2/mac/exact-nightly-2024-10-12-arm64/sha256_manifest.json`

The SHA manifest references the canonical fixture contract at `tests/unsoundaudit-v2/fixture_manifest.json`; it does not duplicate that manifest, so its frozen relative provenance paths remain valid.

## Safety Boundaries and Custody

No Miri run, benchmark, fixture execution, real-project scan, Linux RAP campaign access, coverage-ledger change, or reinterpretation of the 2,848 historical successes occurred. No Mac binary, target directory, Cargo/rustup cache, dylib, or raw machine-path log is committed.

Raw unit, build, contract, control, and scan logs remain in local temporary custody only. Their SHA-256 digests are recorded in the committed validation summary with `committed: false`.

## Limitations and Next Gate

These results prove only the frozen candidate-generator contract on the 27 dependency-free library fixtures. The rules remain heuristic and fixture-closed; unregistered shapes may be missed, and no real-project false-positive rate was measured.

Linux final validation remains pending. After the Linux campaign is confirmed stopped, an independent checkout must verify this branch commit, source hashes, exact toolchain, fresh build, 27 fixtures, determinism, and source review before any separately authorized real-project pilot.
