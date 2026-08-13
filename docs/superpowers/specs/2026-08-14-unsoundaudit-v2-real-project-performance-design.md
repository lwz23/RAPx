# UnsoundAudit v2 Real-Project Performance Design

## 1. Purpose

UnsoundAudit v2 must preserve complete crate-local P1–P6 propagation while becoming practical on real Rust projects. The analyzer may fail closed when an external time or memory budget is exhausted, but it must not silently truncate paths, facts, versions, aliases, recursion, or fixed-point rounds.

The release performance gates on the approved 16 GiB Apple M4 host are:

| Unit | Wall time per fresh scan | macOS physical footprint |
|---|---:|---:|
| `core` | 15 minutes | 8 GiB |
| `alloc` | 30 minutes | 10 GiB |
| `std` | 60 minutes | 12 GiB |

## 2. Evidence and Compatibility Boundary

- The v1, v2, and v3 pilot campaigns are immutable development evidence. A source or protocol change starts a new campaign.
- The v3 `core` control passed, but scan 1 timed out at 3,600 seconds with zero `rap-unit-v2` receipts and one build-script skip receipt. Scan 2, `alloc`, `std`, sampling, and review did not start.
- `rap-unit-v2`, the P1–P6 registry, causal keys, and finding IDs do not change in this work.
- The canonical finding-set projection must remain equal across the current analyzer and every optimized analyzer profile:

```text
finding_id, rule_id, primary, source, sink,
obligation, validation, propagation_depth
```

- Witness selection may change only to another deterministic, validator-accepted route through real unresolved edges. Each optimized profile must be byte-deterministic across two fresh runs.

## 3. Build and Measurement Model

The analyzer and analyzed target use separate profiles. `cargo-rapx` and `rapx` use a dedicated optimized profile:

```toml
[profile.rapx-perf]
inherits = "release"
opt-level = 3
debug = 1
debug-assertions = true
overflow-checks = true
incremental = false
```

The analyzed crate continues to use ordinary development-profile `cargo check`; the pilot must not pass `--release` or another profile to the target build because that would change MIR input.

Optional performance tracing is a diagnostic sidecar, never receipt data. `UNSOUND_SCANNER_PERF_TRACE` points to a fresh JSONL file containing only a schema, phase/event, monotonic elapsed microseconds, and integer counts. It must not contain source paths, findings, random IDs, or wall-clock timestamps. With tracing disabled, no sidecar is created.

On macOS, the runner samples every process in the scan process group through `proc_pid_rusage(RUSAGE_INFO_V4)`. The sum of current `ri_phys_footprint` values is the primary memory metric. A process disappearing between enumeration and sampling is ignored; unsupported APIs, permission errors, parse errors, or sampling failures are gate failures. Process-group RSS remains diagnostic.

## 4. Intraprocedural Optimization

### 4.1 Clone-free joins and propagation

- `ValueFacts::join_may` reports change per field rather than cloning and comparing the complete value.
- Relation atoms retain their existing semantic keys; union of captured versions and sticky invalidation must report change even when the atom-container length is unchanged.
- An unchanged CFG edge does not recompute its successor entry.
- Predecessor join consumes borrowed states and clones only the accumulator.
- Relation invalidation performs a read-only relevance test before cloning/updating a value.

These are representation and scheduling changes only. BTree ordering, finite may-union, must-intersection, reaching versions, and sticky invalidation remain unchanged.

### 4.2 Convergence followed by replay

CFG convergence stores only block entry, exit, and edge states. It does not retain a full state before every MIR operation.

After equality, each reachable block is replayed exactly once from its final entry state using the same deterministic transfer functions. Replay performs materialization and captures only approved observation points. It does not feed facts back into the lattice.

Persistent `BodyFacts` uses a compact `ValidationSnapshot` at resolved-local calls, P1–P6 sinks, and validation-contract sinks:

```text
ValidationSnapshot
- current_validations: bindings valid at the captured reaching versions
- entry_formals: exact semantic place -> unique unchanged Formal(n)
```

The snapshot must answer the existing `has_current_validation` and `entry_formal_slot` queries exactly. Missing required snapshots fail closed. Body-wide aggregate values retain only downstream origins and value-flow data; length, UTF-8, and range relations remain transient replay state.

## 5. Interprocedural Optimization

Definite out writes remain an independent finite must fixed point and are frozen before may-origin propagation.

Local return/out propagation uses only `return_value`, `out_dependencies`, call mappings, and frozen definite-out facts. It reaches equality without invoking the full requirement/validation/route summary solver. Each synchronous wave derives from one old snapshot, joins outputs, and re-extracts only owners whose local-output map grew.

After outputs stabilize, the analyzer runs the complete summary fixed point exactly once. It then performs unresolved-route selection and finding construction from those final summaries.

The first implementation retains global synchronous output derivation. A stable dirty-owner plus reverse-resolved-caller cone is authorized only if profiling after the approved stages still misses a performance gate. Per-wave updates must remain batch/Jacobi style.

## 6. Conditional Deep Optimization

Only if the approved stages miss a gate may a follow-up implement:

- body-local `PlaceId(u32)` interning with `PlaceKey` restoration at public boundaries;
- storage-to-relation watcher indexes for targeted write invalidation;
- stable dirty-caller output scheduling.

No path/depth, alias, origin, fact, version-token, recursion, or fixed-point cap is authorized. Failure after these changes triggers architecture review rather than hidden truncation.

## 7. Verification and Release Gates

Deterministic tests must prove:

- per-field joins preserve C/A/I and detect same-key payload growth;
- final-entry replay equals the prior full before-state at every reachable program point;
- compact snapshots produce the same validation and entry-formal query results;
- old and new output loops produce equal outputs, body seeds, and complete summaries on DAG, self-recursive, mutual-recursive, normal/unwind, definite-out, conditional-out, and nested-call cases;
- at least a 64-hop wrapper and recursive propagation terminate without a hidden depth limit;
- convergence stores zero full statement states, unchanged owners are not rebuilt, and full summary solving occurs once after output stability.

Every implementation stage runs exact-nightly Rust tests plus two challenge and two original-fixture scans. Canonical finding-set projections must match the current baseline. Any finding addition or removal is a semantic regression and stops performance work.

Development `core` diagnostics stop at 15 minutes. `alloc` and `std` are tested only after the preceding unit meets its gate. Once all three units meet their single-run development gates, a newly committed source and protocol run two fresh trace-disabled scans per unit under the same time and physical-footprint limits. Only then may sampling and independent precision review begin.

