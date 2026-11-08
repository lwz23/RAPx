# Pattern numbering

Commit `tosem-v3-scan-snapshot-20261108` is the analyzer exactly as used for the
crates.io / GitHub scans, with the original tool numbering. The following commit
renumbers the patterns to match the paper. Analysis logic is unchanged.

| Paper / current code | Pattern                                   | Snapshot (old) |
|----------------------|-------------------------------------------|----------------|
| P1                   | public parameter flow                     | P1             |
| P2                   | public state (field) flow                 | P2             |
| P3                   | external behavior (FFI, callback, trait)  | P6             |
| P4                   | generic capability gap                    | P5             |
| P5                   | internal unsafe exposure (invalid value)  | P3             |
| P6                   | internal derivation / concrete capability | P4             |

Renamed: Rust sources, receipt schema v3, fixture and mechanism-pack oracles,
and the v3 runner/validator scripts. Fixture directory names, design documents
under `docs/`, and historical receipts keep the old numbering.

Check: the snapshot and renamed binaries (nightly-2026-08-14) were run on the 27
fixtures, 27 challenges, 7 mechanism packs and 9 crates.io crates. All 141 receipts
are identical after mapping the names (134 findings).
