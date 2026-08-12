# Task Context

## Goal

- Implement fixture-closed UnsoundAudit v2 on `feature/unsoundaudit-v2-p1-p6`.
- Replace the v1 string-based detectors with one crate-local MIR summary engine using statically resolved calls and SCC fixed points.
- Validate only the frozen 27 dependency-free library fixtures and emit deterministic `rap-unit-v2` receipts.

## Status

- Formal checkout is on `feature/unsoundaudit-v2-p1-p6` at handoff HEAD `a8ac7640d8778ac047bc9e0b6a1a3afecd144c34`.
- Baseline tag and commit, restricted source diff, source-tree hash, and `unsoundaudit.rs` hash all passed.
- Native arm64 Xcode Command Line Tools, Homebrew LLVM/libclang, Z3, CMake, and pkg-config passed architecture and availability checks.
- `nightly-2024-10-12` is installed with the required components. Rustc and Cargo commits exactly match the manifest.
- The exact frozen baseline built successfully with a fresh target, isolated Cargo home, `--locked`, and one Cargo job.
- Next gate: freeze the 27 v2 fixture oracles, schema, runner, normalizer, and observe the specified first RED result before detector implementation.

## Decisions

- The user approved a fixture-closed minimum contract. New source, sink, predicate, P3-origin, and P6-boundary forms outside the 27 fixtures are unsupported/future work.
- P3 is limited to lifetime-extension transmute, `MaybeUninit<bool>::assume_init`, and unchecked UTF-8 validity.
- P6 is limited to an `extern "C"` out-pointer flowing to `NonNull::new_unchecked` and an open trait `usize` result flowing to `get_unchecked`.
- Keep the existing `-unsoundaudit` CLI and `LwzCheck::new(tcx).start()` entry point; replace its internal implementation and output schema directly.
- Keep project concurrency and Cargo jobs at one. Every build, test, control check, and scan uses a fresh target.
- Existing shell `RUSTFLAGS` are explicitly cleared. The build shell then sets only the handoff-mandated Z3 native search path.
- GitHub SSH uses `ssh.github.com:443` for transport while `origin` remains `git@github.com:lwz23/RAPx.git`.

## Rejected Alternatives

- No nightly substitution, dependency upgrade, RAP lockfile edit, or source workaround for Mac compatibility.
- No cross-crate whole-program analysis, unresolved-call speculation, fixed iteration limit, name-based sanitizer, or keyword-only candidate rule.
- No FFI-return, Iterator, callback, general dynamic-dispatch, raw-reference/slice, `Box::from_raw`, or `set_len` extension in this work package.
- No Miri, benchmarks, fixture execution, real-project scan, Linux campaign access, coverage-ledger update, or historical-baseline reinterpretation.

## Verification Invariants

- Each positive fixture has exactly one primary finding; every negative and noise fixture has zero findings.
- Aggregate primary counts are P1=3, P2=2, P3=2, P4=2, P5=1, and P6=2.
- `pattern_counts` contains exactly `pattern1` through `pattern6`.
- Two complete normalized fixture receipts from fresh serial runs must be byte-identical.
- Only source, fixtures, tracked fixture lockfiles, small JSON/text receipts, and review summaries may be committed.
