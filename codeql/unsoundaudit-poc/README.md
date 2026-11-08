# UnsoundAudit CodeQL PoC (P1–P6)

Custom QL approximating RAP UnsoundAudit patterns. This is rust-analyzer
HIR / taint tracking, not RAP MIR contract solving.

## Layout

- `Unsound.qll` — shared sources, sinks, barriers
- `P1PublicParam.ql` … `P6ExternalBehavior.ql`
- Runner: `scripts/unsoundaudit-v3/run_codeql_fixture_poc.py`
- Extraction probe: `scripts/unsoundaudit-v3/probe_codeql_no_artifact.py`

CLI used for the first run: CodeQL 2.26.4 at
`lwz23-valid-issue-scan/codeql-poc/codeql/codeql` (not in this git tree).

## First fixture run (27 cases)

Presence scoring (expected primary found, not RAP exact-count):

- positive 8/12
- negative 10/14
- noise 0/1

Hits: all P1 positives, both P2 positives, both P3 positives, P6 open-trait.
Misses: P4 (0), P5 (0), P6 FFI out-param.
False positives: P2 name-matching across the combined workspace DB; utf8
negative reported as P1 because the `from_utf8` barrier is weak.

Full table:
`lwz23-valid-issue-scan/codeql-poc/fixture-results/fixture_poc_report.md`

## no_artifact extraction probe (6 RAP-failed repos)

| repo | result | notes |
|---|---|---|
| NVSL/Corundum | EXTRACT_OK ~208s | RAP fails on std `Pattern` lifetime drift |
| asg017/sqlite-loadable-rs | EXTRACT_OK ~224s | |
| archseer/enigma | EXTRACT_OK ~191s | timed out at 180s wall; ok at ~3 min |
| Indra-db/Flecs-Rust | EXTRACT_OK ~212s | |
| DioxusLabs/blitz | TIMEOUT 300s | stuck after seeing `Cargo.toml` (git/edition) |
| Pagefind/pagefind | TIMEOUT 300s | extract finished; killed during TRAP import |

## Run

```bash
python3 scripts/unsoundaudit-v3/run_codeql_fixture_poc.py \
  --codeql /path/to/codeql --reuse-db
```
