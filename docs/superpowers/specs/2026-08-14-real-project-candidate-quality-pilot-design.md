# Legacy-2024 Real-project Candidate-quality Pilot

## Purpose and claim boundary

This is a ten-repository, post-development pilot for inspecting the quality of
UnsoundAudit P1--P6 findings on previously unseen real projects.  It is not a
known-positive regression suite: the P1--P6 fixtures and historical examples
were already used during implementation.  It is also not a coverage, recall,
or ecosystem estimate.

A non-zero finding is a **candidate**, not a false positive by default.  Review
may classify it as a new IUE candidate, a likely false positive, unresolved,
or a duplicate.  A zero-finding result means only that this fixed analysis did
not emit a candidate for the selected target.

## Isolation and frozen analyzer

All work runs on the Mac in a separate `legacy-2024` directory.  It must not
read, write, rebuild, or revalidate the active Linux RAP source tree, registry,
coverage ledger, or 2,848 historical-success baseline.

The analyzer input is commit
`75ded1f25f3d5a1d81d899c2fe070e34d1f5f8ad`, built only with
`nightly-2024-10-12` (rustc commit
`1bc403daadbebb553ccc211a0a8eebb73989665f`).  The pilot first records the
exact source-tree and executable hashes.  The rules remain frozen for all ten
first-pass scans.  Current HEAD and its nightly-2026-08-14-only source are out
of scope.

Use fresh per-project `CARGO_HOME` and `CARGO_TARGET_DIR`, one project at a
time, and Cargo jobs equal to one.  Do not run application tests, examples, or
binaries.  `cargo check` and the RAP invocation can nevertheless execute Cargo
build scripts and procedural macros; that fact is explicitly recorded rather
than hidden.  Mac binaries and `target` directories never return to Linux;
only source commits, manifests, and sanitized receipts may be shared.

## Sampling universe and exclusions

The authoritative universe is the frozen hybrid-500 roster at
`results/tse_iue_external_validation/run_20260806_github_discovery_v3/roster.jsonl`.
It contains the project framework selected as 250 eligible repositories by
GitHub stars plus 250 eligible repositories by crates.io download impact.

Before sampling, create and hash an exclusion set containing every canonical
repository in the 82 atomic GitHub IUE instances and every repository whose
historical source or reduced fixture was used to develop P1--P6.  The pilot is
therefore not allowed to select a known IUE source or a detector-development
input.  Exclude language/toolchain repositories and repositories with no
ordinary Cargo software target under the roster contract.

Select five candidates from `stars_top250` and five from `downloads_top250`.
Within each stratum, sort eligible non-excluded rows by:

```
sha256("rapx-v2-real-project-pilot-v1:" + repo_id)
```

then break a hash tie by numeric `repo_id`.  The selector produces a
machine-readable ordered reserve list, not merely the final ten.  This avoids
choosing projects after seeing their code or RAP findings.

## Repository revision and target admission

For each selected repository, clone its public Git history on the Mac and
resolve the default-branch commit immediately preceding:

```
2024-10-12T00:00:00Z
```

This is a build-compatibility snapshot for the frozen legacy toolchain, not a
replacement for the GitHub-IUE study's 2025-01-01 disclosure cutoff.  Record
the default branch, commit OID, commit timestamp, repository tree hash, and
the SHA-256 of every checked Cargo manifest and lockfile.

Admission is fail-closed:

1. the exact commit must be reachable and contain the original `Cargo.lock`;
2. `cargo metadata --locked --no-deps` must succeed without changing files;
3. select one target deterministically: the lexicographically first library
   target ordered by `(package manifest path relative to repo root, package
   name, target name)`; if no library exists, use the lexicographically first
   non-example binary by the same order;
4. the ordinary control command for that exact target must pass with
   `cargo check --locked`.

Every admission failure stays in the ledger with its command, exit status,
and digest.  Replace it only with the next precommitted reserve repository in
the same stratum.  A replacement is never selected after inspecting RAP
output.  The completed pilot reports both attempted and admitted counts.

## Scan and receipt protocol

For an admitted target, run the same frozen checkout and exact target twice:

1. ordinary control `cargo check --locked`;
2. RAP P1--P6 analysis with the canonical `rapx-perf` profile;
3. a second identical RAP invocation in a fresh target directory.

The normalized finding receipts must be byte-identical.  A wrapper/analyzer
failure, a non-deterministic receipt, or a profile-independent infrastructure
failure is reported as such and is not converted into either a zero finding or
a code-quality judgment.  For every non-zero result, repeat the scan under the
other two validated analyzer profiles.  A candidate whose primary pattern,
source--sink path, or count changes across profiles is marked `profile_unstable`
and is excluded from quantitative candidate counts pending diagnosis.

Raw logs remain local to the Mac.  The shareable receipt includes the analyzer
hashes, toolchain identities, command-template digest, source/lock hashes,
target identity, finding JSON hashes, and redacted failure summary, but no
absolute personal paths or credentials.

## Finding review and rule-correction discipline

Review every unique `(repository, revision, target, primary pattern,
source--sink root cause)` once.  Preserve duplicate emitted locations as links
to that root cause.  Each reviewed candidate receives exactly one disposition:

| Disposition | Meaning |
| --- | --- |
| `new_iue_candidate` | Safe reachability, internal unsafe sink/invariant, and a plausible missing Rust-defined-UB contract are present; this is not yet a confirmed disclosure. |
| `likely_false_positive` | Code establishes the exact required predicate, the flow is not the same source--sink path, or the selected API cannot be safely reached as reported. |
| `unresolved` | The local source and available contract evidence cannot decide the UB chain. |
| `duplicate` | The report has the same root cause and safety obligation as an already reviewed candidate in this target. |
| `tool_or_receipt_failure` | The analyzer/output is not trustworthy enough for source review. |

For `likely_false_positive`, record one causal error family rather than a
repository-specific suppression: `source_misattribution`,
`missing_dominating_validation`, `interprocedural_summary_imprecision`,
`sink_contract_imprecision`, `primary_pattern_precedence`, or
`deduplication`.  A rule change is considered only after collection finishes
and only when the same family recurs or a systemic soundness issue is shown.
Repository-name allowlists are prohibited.

Any proposed correction must add a minimal paired regression fixture, preserve
all existing 27 fixture oracles, and replay the ten pilot targets.  The pilot
then becomes development data; an independent later hold-out sample is needed
before claiming improved candidate quality.

## Deliverables and success criteria

The Mac handoff produces an append-only `real-project-pilot-v1` bundle with:

- frozen selector input hashes, exclusion list, ordered candidates, reserves,
  and all replacement reasons;
- per-repository source, lockfile, target, control, RAP, and determinism
  receipts;
- raw RAP finding JSON retained locally and sanitized reproducibility metadata
  suitable for sharing;
- a Chinese review ledger with source links/spans, candidate disposition,
  error family, and rationale;
- a summary that separates selection/admission failures, tool failures,
  findings, unique root causes, dispositions, and profile stability.

The pilot is complete when ten admitted targets have two deterministic RAP
receipts, every non-zero finding has a disposition, and no Linux active RAP
state was changed.  If fewer than ten are admitted because the reserve list is
exhausted, publish the shortfall and its admission evidence rather than
broadening the universe or changing dependency/lockfile semantics.
