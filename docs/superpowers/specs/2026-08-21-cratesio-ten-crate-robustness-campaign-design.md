# crates.io Ten-Crate Robustness Campaign Design

**Approved:** 2026-08-21

## Goal

Evaluate the current RAP UnsoundAudit P1–P6 implementation on ten recent, widely used crates.io library releases that contain package-local unsafe Rust. The campaign measures build and scan robustness, exact unsafe-site coverage, semantic-model coverage, active-family precision, and visible model gaps. It may also produce reproducible candidate bugs suitable for later paper evidence.

This is a purposive, domain-stratified campaign rather than a random crates.io population sample. Its results describe these ten frozen packages on the recorded host, target, features, compiler, and analyzer revision. They do not estimate the ecosystem-wide false-positive or false-negative rate.

## Scope and fixed interpretation

- “Latest” means the highest stable SemVer release that is not yanked at the roster-freeze time. Pre-releases and repository HEAD revisions are excluded.
- Each selected package must expose a Rust library target and contain direct unsafe Rust in package-local library source. Unsafe code found only in dependencies, tests, examples, or benchmarks does not satisfy admission.
- The campaign scans the default-feature library target on the native macOS arm64 host. It does not claim coverage of all features, binaries, examples, tests, benchmarks, platforms, or conditional compilation branches.
- The supported current adapter and `nightly-2026-08-14` toolchain are used. A package that requires a newer compiler or an unsupported target is recorded as a robustness failure; the compiler adapter is not changed during the frozen campaign.
- The current high-confidence four-state policy remains unchanged: active, evidence-backed pure suppression, independently established reclassification, and explicit unestablished. No finding-count target is used.
- The analyzer, contract catalog, classifications, and test oracles are frozen before the first RAP package scan. A detector change creates a new campaign revision and never overwrites the original evidence.

## Roster selection and freeze

The roster contains one package from each of ten domains:

1. byte buffers and byte-oriented data structures;
2. general collections;
3. concurrency and memory reclamation;
4. synchronization primitives;
5. zero-copy parsing or serialization;
6. POD, casting, or layout conversion;
7. memory mapping or operating-system memory;
8. FFI or dynamic loading;
9. compact vectors, arrays, or small containers; and
10. hashing or runtime primitives.

For each domain, selection uses official crates.io metadata sorted by total downloads at the selection timestamp. The five highest-download plausible candidates are inspected in descending order. The first candidate that satisfies the library-target and package-local-unsafe admission rules is selected. A package may occupy only one domain; its primary documented purpose determines the domain, and the recorded selection evidence explains ambiguous classifications.

The selection step may inspect crates.io metadata, package manifests, and source text, but it may not inspect RAP findings or use finding volume as a selection criterion. After all ten packages are selected, the roster is immutable. Build failures, scan failures, timeouts, partial coverage, or zero findings do not trigger replacement.

The frozen roster records, for every package:

- package name, exact version, domain, description, total downloads, and selection timestamp;
- crates.io page, repository URL, license expression, and yanked status;
- crates.io-published package checksum and independently computed `.crate` SHA-256;
- library target name and package-local unsafe-source admission evidence;
- generated `Cargo.lock` SHA-256 and the checksums of resolved registry packages; and
- a short reason for selecting it over higher-ranked rejected candidates in the same domain.

## Frozen execution environment

The campaign records the host identifier, macOS version, architecture, logical CPU count, physical memory, filesystem volume, timezone, and working directory. It also records:

- rustup toolchain name plus complete `rustc -Vv` and `cargo -Vv` output;
- RAP Git HEAD, porcelain status, binary diff hash, relevant analyzer-source manifest hash, and build profile;
- SHA-256 for `cargo-rapx` and `rapx`;
- Cargo configuration and environment variables that can affect resolution or compilation; and
- exact argv, timeout, process-group policy, and output directory for every command.

A fresh metadata-correct `rapx-perf` build is produced from the frozen current source tree. The existing current-toolchain mechanism pack must pass before its binaries are admitted to the campaign. The campaign does not depend on an unrecorded binary in `/private/tmp`; any external binary path is bound by its hash and copied or reproducibly rebuilt as part of the frozen environment.

## Package preparation

Each package uses a separate source, Cargo home, control target, scan target, and result directory. Preparation proceeds serially:

1. Download the exact `.crate` release from crates.io and verify both the published checksum and the independently computed SHA-256.
2. Extract it into a fresh directory and reject path traversal, duplicate roots, or manifest identity/version mismatches.
3. Generate `Cargo.lock` once with the frozen Cargo version when the release does not contain one, record its SHA-256, and retain it as campaign evidence.
4. Fetch the locked dependency graph, then perform control and RAP executions with `--locked`; use offline mode after the fetch completes.
5. Confirm from Cargo metadata that the selected root package has the expected library target and default feature set.

Package source, registry archives, and Cargo target directories are evidence inputs rather than repository source. Large artifacts remain outside Git; the tracked manifest and report bind them through hashes and immutable paths.

## Control and RAP executions

For each package, the control build and RAP scan use the same root package, default features, library target, locked graph, host target, Cargo version, job count, and compilation environment. The only intentional difference is the RAP compiler wrapper and analysis flag.

The sequence is:

1. run the ordinary locked library check in its fresh control target;
2. if control succeeds, run RAP in a different fresh scan target;
3. enforce a 900-second wall timeout on the complete RAP process group;
4. sample and record wall time, CPU time where available, peak RSS, and peak physical footprint, without applying a memory eviction threshold;
5. retain stdout, stderr, exit status, termination signal, command receipt, coverage receipt, candidates, families, suppression ledger, unestablished ledger, and performance trace; and
6. validate all emitted structured artifacts before admitting any counts.

Runs are serial so that elapsed time and memory observations are not distorted by concurrent package scans. Dependency unit outputs are retained for diagnosis but are not mixed into root-package counts. The root receipt is selected by exact Cargo package identity, manifest containment, version, and library target kind. Missing or ambiguous root receipts fail closed rather than selecting the nearest path or the largest output.

Every package ends in exactly one terminal execution state:

- `control_failed`;
- `scan_timeout`;
- `scan_failed`;
- `partial_coverage`; or
- `complete_exact`.

Only `complete_exact` root-package scans enter finding-precision calculations. All five states remain in the robustness denominator. A partial receipt is evidence of a coverage gap, not evidence of zero findings.

## Exactness and coverage interpretation

For a `complete_exact` scan, the unsafe-site census must reconcile every discovered site into an accepted terminal binding state, with no unbound, ambiguous, elided-without-proof, or coverage-failure record. The four-state decision accounting must also close exactly for every considered P1–P6 candidate.

Exact census and MIR binding do not imply complete semantic modeling. The report therefore separates:

- compiler-discovered unsafe sites;
- exactly bound and proven optimized-away sites;
- sites with precise, generic, opaque, or census-only contracts;
- active, suppressed, reclassified, and unestablished decisions; and
- reason-specific unestablished model gaps.

Opaque and census-only sites remain visible coverage facts and are never presented as proved safe.

## Determinism check

After the first ten scans, identify the smallest and largest complete package by package-local Rust source lines. The size metric counts physical lines in package-local `.rs` files under the library source root and excludes `tests`, `examples`, `benches`, and Cargo target directories.

Each of those two packages receives one additional fresh RAP scan with the same frozen inputs. Normalize away only declared operational fields: absolute temporary roots, wall-clock timestamps, durations, sampled memory, and process identifiers. The normalized semantic receipts, candidates, families, suppression ledger, and unestablished ledger must be byte-identical. Any semantic mismatch is reported as a determinism failure and investigated without rewriting the original runs.

## Family-level manual review

The review unit is a family. Raw findings and root fanout remain associated evidence but do not multiply the precision denominator.

Every active family from every `complete_exact` package is reviewed; there is no active-family sampling cap. Each review record binds the exact campaign, package, family ID, pattern, root, unsafe site, role, source, contract, witness, relevant source blocks, and reviewer rationale. It receives one of four labels:

- `confirmed_bug`: a safe public use can violate the applicable safety contract and the violation is reproduced or independently established;
- `likely_bug`: a realizable safe public witness violates the contract, but dynamic reproduction is unavailable or inappropriate;
- `false_positive`: the reported witness is impossible or the applicable contract is satisfied by evidence the analyzer should be able to model; or
- `inconclusive`: the available evidence does not decide the claim.

The report uses the decidable active denominator `confirmed_bug + likely_bug + false_positive`. It presents confirmed precision, confirmed-plus-likely precision, false-positive fraction, and the inconclusive fraction separately. Wilson intervals are reported as exploratory uncertainty summaries only; they are not ecosystem confidence intervals.

## Suppressed and unestablished audit

Before reading the selected source evidence, freeze a deterministic seed. Within each complete package, sample up to three suppressed families and up to five unestablished decision families. Selection is stratified by pattern, contract precision, terminal reason, and propagation depth, then ordered by SHA-256 over the seed and stable family or decision key. Quotas are not transferred between packages, preventing a large package from dominating the audit.

An unestablished decision-family key is the stable tuple of pattern, root identity, site identity, operand role, source identity, and terminal reason. This review looks for incorrect safety proofs, missed active witnesses, and recurring model gaps. Because neither population has complete ground truth, the sample is not used to estimate a false-negative rate. Any discovered missed bug or invalid suppression is reported directly and triggers a detector-defect record.

## Candidate bug validation

A RAP active family alone is insufficient for a paper bug claim. Each `confirmed_bug` or `likely_bug` candidate undergoes the following validation:

1. verify the exact published source and its public safety contract;
2. construct a minimal safe-caller witness with frozen package and dependency versions;
3. prove reachability and discharge all documented caller preconditions;
4. use an isolated test, Miri, sanitizer, or another appropriate runtime oracle when the behavior can be exercised without replacing the static contract argument;
5. distinguish a contract violation, a possible undefined behavior path, and an actually reproduced undefined behavior symptom; and
6. record feature, target, optimizer, platform, and version limitations.

Until a second qualified reviewer or the package maintainer confirms the analysis, paper-facing labels remain provisional. The campaign may produce strong candidates, but it must not upgrade a detector result into a confirmed ecosystem bug solely for narrative value.

## Metrics and reporting

The final report includes:

- roster, environment, artifact hashes, exact commands, and terminal execution states;
- control success, complete-scan success, timeout, crash, partial-coverage, and deterministic-rerun counts;
- per-package and aggregate elapsed time, peak memory observations, unsafe-site census, binding exactness, and contract-model coverage;
- P1–P6 raw and family counts by active, suppressed, reclassified, and unestablished state;
- unestablished and suppression reason distributions;
- every active-family review label with supporting evidence;
- suppressed/unestablished sample results and identified model gaps;
- reproducible candidate-bug packages and their validation status; and
- explicit limitations for selection, host, target, default features, compiler support, semantic coverage, and reviewer independence.

Machine-readable JSON and CSV accompany a Markdown narrative. Counts are derived from validated structured artifacts rather than scraped logs. The report preserves zero-result, failure, and timeout cases and never uses package replacement or detector tuning to improve the presentation.

## Acceptance criteria

The campaign is complete when:

- the ten-package roster and all selection evidence are frozen before RAP result inspection;
- the admitted RAP binaries pass the current mechanism pack and are bound to source and binary hashes;
- every selected package reaches one recorded terminal execution state under the exact protocol;
- every `complete_exact` scan passes census, binding, four-state accounting, and root-package-selection validation;
- all active families from complete scans are reviewed;
- the deterministic suppressed/unestablished samples are reviewed;
- the smallest/largest complete-package reruns are compared, or the report states that fewer than two complete packages prevented the check;
- every claimed candidate bug has a reproducible static witness and an explicit independent-confirmation status; and
- the final report distinguishes measured facts from inference and does not generalize the ten-package results to crates.io as a whole.

The campaign may still conclude that RAP is not robust on this sample. Tool failure is a valid result and must not be converted into a successful precision claim.

## Rejected alternatives

- **All features and all targets:** rejected for this first campaign because platform-specific dependencies and feature matrices would conflate analyzer robustness with a much larger build-compatibility study. A later revision can add a frozen feature matrix.
- **Repository HEAD:** rejected because it is mutable and does not represent the published crate consumed by users.
- **Historical three-version comparison:** rejected because this campaign evaluates the current high-confidence implementation; Fudan and November implementations require a separate compatibility and differential-study protocol.
- **Replacing failed or zero-finding packages:** rejected because it introduces outcome-dependent selection bias.
- **Reviewing only a convenient active sample:** rejected because the selected ten-crate active population is the primary precision evidence.
- **Treating unestablished as safe or active:** rejected because missing semantic evidence supports neither conclusion.
- **Tuning contracts after seeing campaign results:** rejected for the frozen run; any fix requires a separately identified campaign revision.
