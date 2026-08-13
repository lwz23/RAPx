# UnsoundAudit v2 Standard-Library Performance v4

## Outcome

The exact P1–P6 analyzer now completes the crate-local `core`, `alloc`, and `std` scans inside the release performance gates. Every formal unit used one control build followed by two fresh, trace-disabled scans. The two normalized outputs for each unit were byte-identical.

This is a performance and determinism result, not a precision result. The three units produced 876 findings over 394.765 KLoC, exceeding the pre-registered pause lines of 500 findings and 0.5 findings/KLoC. Deterministic samples are frozen, but independent human labels and adjudication have not started.

## What Changed

The implemented performance work preserves exact fixed points and does not impose a path, recursion, fact, write-version, or iteration cap:

- `rapx-perf` builds the analyzer with optimization while retaining debug assertions and overflow checks. The analyzed standard library still uses ordinary development `cargo check`.
- summary SCC rounds stage component-local updates instead of cloning the complete crate summary map;
- CFG propagation skips unchanged edges, borrows predecessor states, and reports joins field by field;
- CFG convergence retains block and edge states, then performs one deterministic final-entry replay instead of retaining a full state at every statement;
- persistent body observations are compact validation snapshots plus aggregate origins and value-flow;
- local return/out propagation reaches equality before one full requirement/validation/route summary solve, and only changed owners are rebuilt.

The conditional `PlaceId`, relation-watcher index, and dirty-caller-cone changes were not implemented because the measured hard gates already passed. This follows the approved stop condition and avoids additional semantic risk.

## Formal Resource Results

| Unit | Scan 1 | Scan 2 | Peak physical 1 | Peak physical 2 | Hard gate |
|---|---:|---:|---:|---:|---:|
| `core` | 141.705 s | 134.516 s | 2,108,051,200 B | 2,066,796,240 B | 900 s / 8 GiB |
| `alloc` | 75.283 s | 69.354 s | 816,695,936 B | 810,879,568 B | 1,800 s / 10 GiB |
| `std` | 125.173 s | 122.410 s | 810,158,696 B | 813,075,048 B | 3,600 s / 12 GiB |

Physical footprint was sampled for every live process-group member through macOS `proc_pid_rusage(RUSAGE_INFO_V4)` at 100 ms intervals. RSS was recorded separately and was not the success metric. Sampling errors other than an exited-process race fail closed.

## Correctness and Compatibility Gates

- Exact nightly Rust gate: 136 passed, zero failed, ignored, or filtered.
- Challenge suite: two byte-identical 27/27 runs, normalized SHA-256 `33ca7788b5f52aa264285929395fecfb69afa5ac6c8e56f478163b1cd9b483f6`.
- Original suite: two byte-identical 27/27 runs, normalized SHA-256 `3fa14cafd01a58b17a57094ae4c785d4dda306835adcb433219245e102edc0aa`.
- Debug, stock release, and `rapx-perf` produced the same canonical finding sets. Stock release remains diagnostic only.
- A 64-call propagation test and the existing self/mutual-recursion tests guard against implicit depth limits.
- `rap-unit-v2`, the P1–P6 registry, causal keys, and finding-ID construction are unchanged.

## Formal Populations

| Unit | KLoC | Findings | Findings/KLoC | Dominant rules | Frozen sample |
|---|---:|---:|---:|---|---:|
| `core` | 169.027 | 269 | 1.591462 | P2=197, P4=62, P1=10 | 250 |
| `alloc` | 68.839 | 501 | 7.277851 | P4=493, P1=8 | 250 |
| `std` | 156.899 | 106 | 0.675594 | P4=106 | 106 |

The combined population is 876 findings over 394.765 KLoC, or 2.219042 findings/KLoC. These counts do not establish that the findings are false positives. They establish that the pre-registered census-size/noise gate requires investigation before a bounded precision claim can proceed.

## Evidence and Boundaries

The machine-readable evidence is under `artifacts/unsoundaudit-v2/mac/stdlib-pilot/performance-v4/`:

- each `*.execution.json` records the analyzer profile, binary hashes, protocol hash, time, physical footprint, RSS, and outcome;
- each `*.normalized.json` is the path/resource-free, schema-validated population produced identically by both scans;
- each `*.sample.json` is selected from that population by the frozen v4 seed before any labels exist;
- `summary.json` records the exact unit gates, source/toolchain identity, synthetic equivalence, and the precision pause;
- `sha256_manifest.json` binds the committed evidence and this report.

Raw logs, raw build receipts, Cargo homes, targets, analyzer binaries, temporary drivers, and rust-src copies remain outside the repository. No cross-crate propagation was added: `core`, `alloc`, and `std` remain separate crate-local populations.

## Remaining Gate

Independent human review is intentionally not claimed as complete. The next permitted step is a noise-root analysis of the frozen samples, followed by two independent A–U reviews with at least one Rust unsafe-contract reviewer. Interprocedural findings require reconstruction of every local edge and mapping; disagreements require an independent adjudicator. Any result-informed analyzer change makes these standard-library findings development data and requires a newly frozen protocol before another formal campaign.
