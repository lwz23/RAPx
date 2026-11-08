# crates.io 10-Crate Robustness Campaign Report (cratesio-2026-08-21)

## Executive Summary

- **Campaign ID**: `cratesio-2026-08-21`
- **Selection Date**: `2026-08-21`
- **Evaluated Toolchain**: `nightly-2026-08-14` (rustc commit `ba28ff76f353a722f31c4f3dd2ac4e437d36411b`)
- **Evaluated Profile**: `rapx-perf` on `aarch64-apple-darwin`
- **Wall-clock timeout**: 900s per process group

## Robustness & Terminal States

| Crate | Version | Domain | Rust LOC | Terminal State | Control Wall (s) | Scan Wall (s) | Peak Mem (MB) |
|---|---|---|---|---|---|---|---|
| `bytes` | `1.12.1` | `byte_buffers` | 9881 | `complete_exact` | 1.325 | 9.951 | 291.42 |
| `smallvec` | `1.15.2` | `compact_containers` | 3822 | `complete_exact` | 0.433 | 6.77 | 215.44 |
| `crossbeam-epoch` | `0.9.20` | `concurrency_reclamation` | 4925 | `partial_coverage` | 3.615 | 5.493 | 120.16 |
| `libloading` | `0.9.0` | `ffi_dynamic_loading` | 2833 | `complete_exact` | 0.494 | 1.583 | 81.31 |
| `hashbrown` | `0.17.1` | `general_collections` | 23720 | `complete_exact` | 1.473 | 24.017 | 712.02 |
| `ahash` | `0.8.12` | `hashing_runtime` | 3793 | `complete_exact` | 11.844 | 13.548 | 340.74 |
| `memmap2` | `0.9.11` | `memory_mapping` | 3976 | `complete_exact` | 2.411 | 3.473 | 86.74 |
| `bytemuck` | `1.25.2` | `pod_layout_casting` | 4162 | `complete_exact` | 0.612 | 1.568 | 82.8 |
| `parking_lot_core` | `0.9.12` | `synchronization` | 3687 | `complete_exact` | 3.801 | 51.26 | 208.6 |
| `zerocopy` | `0.8.56` | `zero_copy_serialization` | 28280 | `partial_coverage` | 8.707 | 35.399 | 964.55 |

## Terminal States Aggregate

- **Complete & Exact (100% Exact Coverage)**: 8/10
- **Partial Coverage**: 2/10
- **Scan Failed**: 0/10
- **Scan Timeout**: 0/10
- **Control Failed**: 0/10

## Active Family Precision Review

- **Total Active Families**: 2
- **Decidable Families**: 2
- **Confirmed Bugs**: 0
- **Likely Bugs**: 0
- **False Positives**: 2
- **Inconclusive**: 0
- **False Positive Rate**: 100.00% (2/2)
- **Exploratory 95% Wilson Interval**: [34.24%, 100.00%]

## Sampled Audits (Suppressed & Unestablished)

- **Audited Suppressed Groups**: 15
- **Audited Unestablished Groups**: 27
- **Sample Decisions Breakdown**: {
  "classification_supported": 42
}

## Limitations and Scope

1. **Scope**: This robustness campaign evaluates default-feature library targets on macOS `aarch64-apple-darwin` under clean serial execution.
2. **Purposive Sample**: The 10 crates represent high-download strata with package-local `unsafe` code. The active census is purposive, and Wilson intervals are exploratory rather than ecosystem-wide population estimates.
3. **Independent Confirmation**: Confirmed/likely bug claims require safe callers, minimal reproductions, and independent maintainer or expert confirmation before formal CVE disclosure.
