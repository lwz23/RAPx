# Remote handoff: lwz23 148/125 recall

This file is for a Codex/agent whose working directory is the remote
`unsound_scanner` tree. **Treat that directory as local cwd.** Do not rely
on Mac chat history. Do **not** start the crates.io 5000+ campaign.

Owner intent (2026-09-07): Mac disk and USB cargo cannot finish this
experiment. Move it to the workstation. First make the **125 GitHub repos /
148 valid issues** compile and be re-detected under the **new P1–P6**.

Confirmed 2026-09-07:

1. Universe = this frozen **148 issues / 125 scan repos**.
2. Patterns = **new P1–P6** (v2 semantics).
3. A hit must match **issue-body source / function names / paths**, not only
   the title.
4. Copy the Mac working tree `RAPx-unsoundaudit-v2/` (it already contains
   UnsoundAudit **v3**; there is no separate v3 folder). See §3.
5. “100+ projects” = the **125 scan repos**. Engine may be RAP v3 **or**
   custom CodeQL P1–P6. Forbidden: a name-only sink allowlist as the
   discovery engine (old RAP “if callee is one of these 10 strings, report”).

---

## 0. Success criteria

1. Every in-scope scan repo (125, minus honest `skip_host`) can be analyzed
   by **RAP UnsoundAudit v3** or **CodeQL custom P1–P6 queries**.
2. **Most of the 148 reported issues** produce a finding whose public root,
   callee, or span matches the **issue body** (fenced Rust, `pub fn`,
   `src/….rs`, API names). Title-only match is last resort and is **not**
   enough when the body names a specific function.
3. Detection implements the six patterns with dataflow / compiler unsafe
   sites. A **contract catalog** of real Rust unsafe operations
   (`get_unchecked`, `assume_init`, raw deref, `from_utf8_unchecked`, …)
   used *after* discovering unsafe sites is allowed. A hardcoded-only sink
   list as the *only* way a site can appear is not.
4. Stop when (2) is honestly true, or document each remaining miss with a
   **failure class** (not a per-repo hack).

Not success: 95% KPI theater; crates.io 5000+ coverage; “compiled” without a
finding that maps to the reported API; matching an unrelated finding in the
same crate.

Mac last ledger (strict **title** scorer, RAP-only, Mac host) — **not** the
remote target:

`true_hit 32 / no_artifact 76 / miss 26 / skip_host 11 / gated 3` over 148.

`no_artifact` = never compiled/scanned. `miss` = had RAP JSON but no title
match. Shrink both. After upgrading the matcher to issue bodies, some of
the 32 may drop; that is expected — re-score honestly.

---

## 1. Universe (frozen)

| Name | Count | Meaning |
|---|---:|---|
| Valid issues | **148** | Recall denominator. Author `lwz23`, window `[2024-11-01, 2026-01-01)`, not explicitly rebutted/withdrawn. |
| Scan repos | **125** | Host GitHub repos of those issues (`repositories.csv`). |
| “100+ projects” | **125 repos** | Same set. |
| Software-repo issues | 142 | 148 minus 6 `rustsec/advisory-db` issues. Those 6 stay in the 148; scan their **mapped software crates** (Corundum, Pleco, perpetual, ninjabook, swift-bridge, …), do **not** clone advisory-db. |
| Accepted-and-fixed | 27 | Subset; still in the 148. |
| Out of scope | — | crates.io 5000+ campaign; IUE-83 paper subset; 14-case fixture subset as the universe. |

**Authoritative files** (inside the copied tree):

- `docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/ROSTER.md`
- `docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/issues.csv` (148 data rows)
- `docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/repositories.csv` (125)
- `docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/clone_plan.json`
- `docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/exclusions.json`
- `docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/bug_instances.jsonl`
- `docs/unsoundaudit-v3/lwz23-valid-issue-roster-inputs/issue_outcomes.jsonl`

`BrowserWorks/waterfox` **is** in the 125. It is huge; mark `skip_host` with
class “browser-monorepo / not a practical Cargo crate scan” if it cannot
be built. Do not silently drop it from the 148.

Fragile revision / package / lockfile is **not** frozen. Default branch at
report time is acceptable if documented. Prefer a historical SHA if the
roster later gains one.

---

## 2. Patterns (v2 semantics, v3 engine)

Six **UnsoundAudit v2** pattern meanings (`docs/unsoundaudit-v2/design.md`):

| ID | Meaning |
|---|---|
| P1 | Public function parameter (0..N local helpers) reaches an unsafe sink without a dominating proof. |
| P2 | Literal `pub` field a safe caller can write reaches an unsafe sink. |
| P3 | After a legal entry, internal unsafe constructs/exposes an illegal value (`assume_init` on uninit, `transmute` lifetime, unchecked UTF-8, …). |
| P4 | Internally derived index/len/offset fails a memory precondition. |
| P5 | Currently P5.1: public generic/associated type does not guarantee non-empty, sink uses element 0. |
| P6 | FFI return/out-param or open trait/callback/iterator result reaches an unsafe sink unchecked. |

Oracle: 27 fixtures under `tests/unsoundaudit-v2/fixtures/` (12 positive,
14 negative, 1 noise). Aggregate expected `P1=3, P2=2, P3=2, P4=2, P5=1, P6=2`.

These are **not** the old paper P1–P4 numbers (old P3 became new P1 helper
flow; old P4 became new P3).

---

## 3. v2 vs v3 (folder name is misleading)

**There is no `RAPx-unsoundaudit-v3` directory.** All later work happened
**on this Mac**, inside `RAPx-unsoundaudit-v2/`, on branch
`feature/unsoundaudit-v2-p1-p6`. Copy that working tree.

| Layer | What it is |
|---|---|
| **Repo folder** | `RAPx-unsoundaudit-v2/` — copy **this dirty working tree**, not `git clone origin`. |
| **UnsoundAudit v2** | First Mac P1–P6 implementation. Frozen tag `unsoundaudit-v2-pre-v3-preservation-2026-08-16` (`d406a9a`). Crate-local MIR summaries, modeled contract/sink registry, output `rap-unit-v2`, `rap-unit-v2` receipts, 27 fixtures. |
| **UnsoundAudit v3 (committed)** | Rewrite **after that tag** on the same branch: rustc `check_unsafety` THIR census → bind to MIR → sparse endpoint flow → structural P1–P6. Output schema `rap-unit-v3` only. Production `-unsoundaudit` writes v3. Two nightlies: `nightly-2024-10-12` (rustc `1bc403da…`) and `nightly-2026-08-14` (rustc `ba28ff76…`). |
| **Mac extra (mostly uncommitted)** | Branch is **83 commits ahead of `origin/feature/unsoundaudit-v2-p1-p6`**, plus a large dirty tree (~32k insertions): heuristic unmodeled-unsafe P1/P2, recovery knobs, `v3_output` expansion, lwz23 roster, `scan_lwz23_valid_issue_roster.py`, `rescore_lwz23_coverage.py`, CodeQL PoC QL. **If you clone origin you will miss all of this.** |

v3 vs v2 in one sentence: **v2 discovers sinks from a modeled registry +
MIR summaries; v3 discovers unsafe sites from rustc, then classifies
P1–P6 on endpoint flow.** Contract models still exist for discharge; they
must not be the only way a site can appear.

Heuristic already in dirty v3 `candidate.rs`: public parameter/field + sole
reason `UnmodeledUnsafeCall` may become an active finding (recall-oriented,
higher FP). Keep it.

Do **not** modify upstream `safer-rust/RAPx`. Work in this fork.

---

## 4. Tooling you may use

Treat `unsound_scanner/` as cwd. After copy, analyzer lives at e.g.
`unsound_scanner/RAPx-unsoundaudit-v2/`.

- **Primary:** RAP `cargo-rapx` / `rapx` from this tree, profile `rapx-perf`.
  Only the two nightlies above pass v3 preflight.
- **Allowed fallback:** CodeQL Rust **custom** P1–P6 queries in
  `codeql/unsoundaudit-poc/` (not stock OWASP). Mac fixture: P1/P2/P3 can
  hit; P4/P5/P6.FFI weak. Mac extraction died on USB `CARGO_HOME` + full
  disk — Linux SSD should retry. Put cargo/target on **local SSD**, not a
  slow external volume.
- **Forbidden:** syn/AST Pattern1 as a RAP replacement; per-repo
  allowlists; issue-id special cases; reopening Mac
  `phase0/msrv_retry.txt` leftover 58; crates.io 5000+.

Generic recovery (not per repo): MSRV `--precise`, `--no-default-features`,
`feature(freeze)`, unlocked lockfile, retry only failed packages. Hard host
impossibility (Windows PE, json-target bare metal, mandatory missing C lib)
→ `skip_host` with a **class** reason.

---

## 5. Matching reported issues (upgrade the Mac scorer)

Mac `scripts/unsoundaudit-v3/rescore_lwz23_coverage.py` currently needles
**title + `repair_obligation_zh`**. That is **too weak**. `issues.csv` has
**no issue body column**. `bug_instances.jsonl` often copies the title into
`repair_obligation_zh`.

Required match sources, in order:

1. GitHub issue **body**: fenced Rust, `` `pub fn` ``, paths `src/….rs`,
   method names, types. Fetch with `gh issue view --json body,title` (or
   the GitHub API) for all 148 URLs and cache locally. Do not use the
   anonymized `docs/remote-vulnerability-assets/github_issues_report_anonymous.md`
   as the oracle (links stripped; not 148-complete).
2. `bug_instances.jsonl` / `issue_outcomes.jsonl` obligations (secondary).
3. Title (last resort).

A hit = a RAP/CodeQL finding whose public root, callee, or span path
matches those needles. Do not count an unrelated finding in the same crate.

Known false “hits” to exclude: starlight; JVM `find_symbol`; retrowin32 Arg.

---

## 6. Suggested execution order

1. **Copy** the Mac dirty tree into this cwd (see §7). Verify
   `wc -l docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/issues.csv`
   → 149 lines (header + 148).
2. **Smoke:** build `rapx-perf` on both nightlies; run 27 fixtures; expect
   `3/2/2/2/1/2` and no extra patterns on negatives.
3. **Fetch** 148 issue bodies and build a needle cache (fn names, paths,
   code-fence identifiers).
4. **Clone** 125 repos from `clone_plan.json` (skip cloning advisory-db;
   waterfox may `skip_host`). Optional: Mac already has checkouts under
   `lwz23-valid-issue-src/` if network clone is slow — copy if available.
5. **Scan** with RAP v3 + generic recovery. Wall-clock per repo **must**
   have a timeout; never hang one crate for hours.
6. **Rescore** against issue **bodies**. Publish a 148-row ledger.
7. For remaining `no_artifact` / `miss`: CodeQL custom queries, or recovery
   knobs, or honest `skip_host`.
8. Iterate by **failure class** (git 404, edition, missing native lib,
   query miss) until most of 148 hit or remaining are skip_host.

Do not start crates.io 5000+. Do not wait for 100% compile of Windows-only
trees.

---

## 7. Copy checklist from the Mac

From `/Users/liaowenzhao/Documents/RAP重构/RAPx-unsoundaudit-v2/`
rsync/scp the **working tree including uncommitted and untracked files**.

Must include:

- `rapx/src/analysis/unsoundaudit/` (v3 analyzer, dirty)
- `rapx/src/bin/`, `rapx/src/lib.rs`, `rapx/src/utils/`
- `tests/unsoundaudit-v2/fixtures/` and `tests/unsoundaudit-v3/`
- `scripts/unsoundaudit-v3/` (`scan_lwz23_valid_issue_roster.py`,
  `rescore_lwz23_coverage.py`, recovery tests, CodeQL census scripts)
- `docs/unsoundaudit-v2/design.md`, `docs/unsoundaudit-v3/` (roster + this file)
- `codeql/unsoundaudit-poc/`
- `rust-toolchain.toml` (nightly-2026-08-14)
- `.git` is useful (history + tag) but **not sufficient alone**

Optional: Mac checkout `/Users/liaowenzhao/Documents/RAP重构/lwz23-valid-issue-src/`

Do **not** copy:

- `rapx/target/`, `liblib.rlib`, `tmp/`, `__pycache__/`
- Mac `lwz23-valid-issue-scan/codeql-poc/` CLI binaries (install CodeQL on Linux)
- USB `/Volumes/RAPDATA` cargo-home
- 241MB Codex rollouts
- `docs/remote-vulnerability-assets/` (not needed for this recall)

---

## 8. Constraints (do not “fix” by violating these)

- No AST/syn Pattern1 detector as the reported tool.
- No per-repository special cases / allowlists.
- No hardcoded-only sink list as the discovery engine.
- False positives may rise (heuristic P1 is allowed).
- Linux still cannot run Windows PE / some json-target / missing system C
  libraries — mark `skip_host`, do not fake compile.
- Do not edit upstream safer-rust/RAPx.
- Do not reopen leftover `msrv_retry.txt`.

---

## 9. First commands after copy

```text
# cwd = unsound_scanner/RAPx-unsoundaudit-v2  (or wherever you copied it)
wc -l docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/issues.csv
# expect 149 including header → 148 issues
wc -l docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/repositories.csv
# expect 126 including header → 125 repos
git status -sb
# expect dirty tree; if clean and 83 commits missing, you copied origin, not Mac
```

Then fixture smoke → fetch issue bodies → clone → scan. Log PID, journal,
timeout, and next action after exit.
