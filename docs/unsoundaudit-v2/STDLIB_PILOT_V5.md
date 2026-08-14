# UnsoundAudit v2 standard-library pilot v5

## Result

The v5 campaign passed its performance, determinism, schema, route, and adjusted usability gates. It does **not** establish formal standard-library precision.

The analyzer scanned the pinned `nightly-2024-10-12` `core`, `alloc`, and `std` sources twice each with the assertion-preserving `rapx-perf` analyzer. The analyzed crates still used ordinary development `cargo check`. Every scan stayed well below its wall-time and macOS physical-footprint limit, emitted exactly one selected unit receipt, and reproduced byte-identical normalized receipts, suppression ledgers, and family reports.

| Unit | Raw | Hard-suppressed | Active | Scan times | Peak physical footprint |
|---|---:|---:|---:|---:|---:|
| `core` | 269 | 247 | 22 | 151.741 / 149.040 s | 2.046 / 2.006 GB |
| `alloc` | 501 | 491 | 10 | 72.159 / 78.031 s | 0.813 / 0.810 GB |
| `std` | 106 | 101 | 5 | 136.849 / 134.380 s | 0.817 / 0.810 GB |
| **Total** | **876** | **839** | **37** | — | — |

The active density is 0.093727 findings/KLoC across 394.765 KLoC. This is below the v5 adjusted usability lines of 500 active findings and 0.5 active findings/KLoC. The unadjusted raw density remains 2.219042 findings/KLoC and must not be replaced by the active density in reporting.

## What changed from v4

The v4 population was kept immutable. A deterministic single-review diagnostic grouped all 876 findings into 77 exact root-cause families and selected 124 review rows. The reviewer assigned 91 sampled rows to class C and 33 to class B, with no A, D, E, U, or fabricated edge. Population weighting identified 839 findings in 54 private type/constructor-invariant C families and retained 37 findings in 23 B families that still need general guard, summary, alias, range, or propagation evidence.

Only the 54 audited C families entered the built-in suppression table. Every matcher binds the exact rustc commit, crate, rule, sink span identity, first-failure span identity, canonical obligation, source kind, family ID, evidence digest, and rationale. There are no glob, prefix, regex, line-range, or MIR-transfer shortcuts. Suppression occurs after finding construction and causal deduplication; it does not alter summaries, routes, causal keys, or finding IDs. A rustc commit mismatch suppresses nothing.

Every removed finding is present in a separate complete suppression ledger. A set-level reconciliation proved, per unit, that:

```text
v5 active finding IDs ∪ v5 ledger finding IDs = v4 raw finding IDs
v5 active finding IDs ∩ v5 ledger finding IDs = ∅
```

All 23 B families remain active and all 54 C families are represented in the ledger. There were no unexplained A/B disappearances.

## Compatibility evidence

- Fresh exact-nightly Rust gate: 141/141.
- No-build noise-audit and pilot contracts: 50/50.
- Challenge pack: two `rapx-perf` runs, 27/27, normalized SHA-256 `33ca7788b5f52aa264285929395fecfb69afa5ac6c8e56f478163b1cd9b483f6`.
- Original fixtures: two `rapx-perf` runs, 27/27, normalized SHA-256 `3fa14cafd01a58b17a57094ae4c785d4dda306835adcb433219245e102edc0aa`.
- Debug, stock release, and `rapx-perf` produced the same two canonical synthetic finding sets.
- The 64-hop test and direct/mutual-recursion tests remain green; no path, fact, version, recursive-round, or fixed-point cap was introduced.

## Evidence layout

Committed v5 evidence is under `artifacts/unsoundaudit-v2/mac/stdlib-pilot/suppression-v5/`:

- `*.normalized.json`: complete active findings using the unchanged `rap-unit-v2` finding shape;
- `*.suppression-ledger.json`: complete hard-suppressed findings and exact allowlist entries;
- `*.family-report.json`: raw/active/suppressed family counts, public-root fanout, and representative shortest/longest witnesses;
- `*.execution.json`: control/scan wall time and resource measurements;
- `summary.json` and `sha256_manifest.json`: campaign-level gates and artifact hashes.

Raw logs, raw receipts, build targets, Cargo caches, and binaries are not committed.

## Reporting boundary

This was a single-person development diagnostic, not the original two-reviewer precision protocol. No Cohen's kappa, adjudication, or Wilson precision interval was computed, and no formal precision pass is claimed. The result means that exact, rustc-pinned suppression makes this standard-library population operationally reviewable while preserving the remaining uncertain families; it does not mean the generic analyzer became more precise on those suppressed invariants.

The standard-library run exercised only P1, P2, and P4. It does not measure real-project performance or precision for P3, P5, or P6, does not validate cross-crate propagation, and does not support third-party ecosystem claims. A future formal real-project precision result requires a separately frozen, previously unseen third-party holdout; these standard-library findings cannot be reused as that holdout.
