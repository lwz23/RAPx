# UnsoundAudit v2 Standard-Library Active-Finding Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine which of the 37 active v5 standard-library findings are proven false positives, inconclusive candidates, historical UB, or current reportable soundness bugs, beginning with the varying-`Borrow<str>` `join/concat` candidate.

**Architecture:** Keep validation outside the analyzer and immutable v4/v5 evidence. A dependency-free safe-only PoC exercises the exact public APIs under pinned and current Miri; a closed audit maps every active family to source proof and dynamic evidence. External disclosure remains a separate approval gate.

**Tech Stack:** Rust `nightly-2024-10-12`, current Rust nightly, Miri, Cargo, Python 3 no-build validation, SHA-256 evidence, pinned rust-src.

---

## File Map

- Create `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.toml`: isolated PoC crate.
- Create `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.lock`: pinned dependency-free resolution.
- Create `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/src/lib.rs`: safe changing-`Borrow<str>` cases.
- Create `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/src/main.rs`: acknowledged case runner.
- Create `scripts/unsoundaudit-v2/test_stdlib_bug_validation.py`: no-build contract tests.
- Create `artifacts/unsoundaudit-v2/mac/stdlib-bug-validation/join-varying-borrow/results.json`: sanitized version matrix.
- Create `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md`: 23-family dispositions.
- Modify `TASK_CONTEXT.md`: decisions, evidence hashes, and disclosure state.

No analyzer source, suppression table, receipt, causal key, v4 artifact, or v5 artifact changes.

### Task 1: Freeze the validation contract

**Files:**
- Create: `scripts/unsoundaudit-v2/test_stdlib_bug_validation.py`
- Create: `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Write the failing no-build contract test**
```python
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V5 = REPO / "artifacts/unsoundaudit-v2/mac/stdlib-pilot/suppression-v5"
AUDIT = REPO / "docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md"
DISPOSITIONS = {"confirmed_safe", "likely_safe", "poc_required", "confirmed_ub", "inconclusive"}


class ActiveFindingValidationTests(unittest.TestCase):
    def test_committed_active_population_is_37(self):
        findings = []
        for unit in ("core", "alloc", "std"):
            payload = json.loads((V5 / f"{unit}.normalized.json").read_text())
            findings.extend(payload["findings"])
        self.assertEqual(len(findings), 37)
        self.assertEqual(len({item["finding_id"] for item in findings}), 37)

    def test_audit_has_one_row_per_active_family(self):
        rows = [
            line for line in AUDIT.read_text().splitlines()
            if line.startswith("| `") and "family_id" not in line
        ]
        family_ids = [line.split("`")[1] for line in rows]
        self.assertEqual(len(family_ids), 23)
        self.assertEqual(len(set(family_ids)), 23)
        for line in rows:
            self.assertTrue(any(f"| `{status}` |" in line for status in DISPOSITIONS), line)

    def test_no_family_is_called_a_bug_without_confirmed_ub(self):
        for line in AUDIT.read_text().splitlines():
            if line.startswith("| `") and "confirmed bug" in line.lower():
                self.assertIn("| `confirmed_ub` |", line)
```

- [ ] **Step 2: Run the test to verify RED**
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_stdlib_bug_validation.py
```

Expected: FAIL because `STDLIB_ACTIVE_FINDING_AUDIT.md` does not exist.

- [ ] **Step 3: Create the closed audit table**
Create one row per family from the committed `*.family-report.json` files:

```markdown
| family_id | unit | rule | sink | active findings | public roots | disposition | source proof | PoC evidence |
```

Set the `join/concat` family to `poc_required`. Set the other families to `likely_safe` using the committed `family_reviews.json` rationale; do not use `confirmed_safe` until a complete proof exists.

- [ ] **Step 4: Run the test to verify GREEN**
Run Step 2. Expected: 3 tests PASS and no Python cache.

- [ ] **Step 5: Record the boundary and commit**
```bash
git add scripts/unsoundaudit-v2/test_stdlib_bug_validation.py \
  docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v2): freeze active finding validation"
```

### Task 2: Build the safe-only `join/concat` PoC

**Files:**
- Create: `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.toml`
- Create: `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.lock`
- Create: `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/src/lib.rs`
- Create: `tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/src/main.rs`
- Test: `scripts/unsoundaudit-v2/test_stdlib_bug_validation.py`

- [ ] **Step 1: Extend the no-build test with safe-only checks**
```python
    def test_join_poc_is_dependency_free_and_contains_no_unsafe(self):
        root = REPO / "tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow"
        manifest = (root / "Cargo.toml").read_text()
        source = "\n".join(path.read_text() for path in sorted((root / "src").glob("*.rs")))
        self.assertIn("[dependencies]\n", manifest)
        self.assertNotIn("unsafe", source)
        self.assertIn("UNSOUNDAUDIT_STDLIB_POC_ACK", source)
```

- [ ] **Step 2: Run the focused test to verify RED**
Run the full no-build command from Task 1. Expected: ERROR because the PoC crate does not exist.

- [ ] **Step 3: Create the dependency-free manifest**
```toml
[package]
name = "join-varying-borrow-poc"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
```

- [ ] **Step 4: Generate the lockfile and implement the safe changing-borrow cases**
```bash
cargo +nightly-2024-10-12 generate-lockfile --manifest-path \
  tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.toml
```

Create `src/lib.rs`:

```rust
use std::borrow::Borrow;
use std::cell::Cell;
use std::hint::black_box;

pub struct ChangingBorrow {
    calls: Cell<usize>,
    first: &'static str,
    later: &'static str,
}

impl ChangingBorrow {
    pub const fn new(first: &'static str, later: &'static str) -> Self {
        Self { calls: Cell::new(0), first, later }
    }
}

impl Borrow<str> for ChangingBorrow {
    fn borrow(&self) -> &str {
        let calls = self.calls.get();
        self.calls.set(calls + 1);
        if calls == 0 { self.first } else { self.later }
    }
}

pub fn join_growing_first() {
    let values = [ChangingBorrow::new("", "0123456789abcdef")];
    black_box(values.join(""));
}

pub fn concat_growing_first() {
    let values = [ChangingBorrow::new("", "0123456789abcdef")];
    black_box(values.concat());
}

pub fn join_growing_later() {
    let values = [
        ChangingBorrow::new("a", "a"),
        ChangingBorrow::new("", "0123456789abcdef"),
    ];
    black_box(values.join(""));
}

pub fn join_growing_first_nonempty_separator() {
    let values = [
        ChangingBorrow::new("", "0123456789abcdef"),
        ChangingBorrow::new("x", "x"),
    ];
    black_box(values.join("-"));
}

pub fn join_shrinking_first() {
    let values = [ChangingBorrow::new("0123456789abcdef", "")];
    black_box(values.join(""));
}

pub fn stable_control() {
    let values = [ChangingBorrow::new("abc", "abc")];
    assert_eq!(values.join(""), "abc");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_borrow_is_short_then_long() {
        let value = ChangingBorrow::new("", "long");
        assert_eq!(Borrow::<str>::borrow(&value), "");
        assert_eq!(Borrow::<str>::borrow(&value), "long");
    }

    #[test]
    fn stable_control_is_safe_to_execute() {
        stable_control();
    }
}
```

- [ ] **Step 5: Implement the acknowledged runner**
Create `src/main.rs`:

```rust
use join_varying_borrow_poc::{
    concat_growing_first, join_growing_first, join_growing_first_nonempty_separator,
    join_growing_later, join_shrinking_first, stable_control,
};

fn main() {
    assert_eq!(
        std::env::var("UNSOUNDAUDIT_STDLIB_POC_ACK").as_deref(),
        Ok("1"),
        "refusing to run a potential UB PoC without explicit acknowledgement",
    );
    match std::env::args().nth(1).as_deref() {
        Some("stable-control") => stable_control(),
        Some("join-growing-first") => join_growing_first(),
        Some("concat-growing-first") => concat_growing_first(),
        Some("join-growing-later") => join_growing_later(),
        Some("join-growing-first-nonempty-separator") => join_growing_first_nonempty_separator(),
        Some("join-shrinking-first") => join_shrinking_first(),
        other => panic!("unknown validation case: {other:?}"),
    }
}
```

- [ ] **Step 6: Verify compile-only and safe controls**
Do not run candidate cases natively.

```bash
PINNED_CARGO="$HOME/.rustup/toolchains/nightly-2024-10-12-aarch64-apple-darwin/bin/cargo"
POC_TARGET="$(mktemp -d /tmp/rapx-join-poc-check.XXXXXX)"
CARGO_TARGET_DIR="$POC_TARGET" "$PINNED_CARGO" test --locked --jobs 1 \
  --manifest-path tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.toml
UNSOUNDAUDIT_STDLIB_POC_ACK=1 CARGO_TARGET_DIR="$POC_TARGET" \
  "$PINNED_CARGO" run --quiet --manifest-path \
  tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.toml \
  -- stable-control
```

Expected: 2 tests PASS and control exits 0.

- [ ] **Step 7: Run contracts and commit**
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_stdlib_bug_validation.py
git add tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow \
  scripts/unsoundaudit-v2/test_stdlib_bug_validation.py
git commit -m "test(unsoundaudit-v2): add safe varying borrow poc"
```

### Task 3: Run the pinned Miri matrix

**Files:**
- Create: `artifacts/unsoundaudit-v2/mac/stdlib-bug-validation/join-varying-borrow/results.json`
- Modify: `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Install the matching historical Miri**
```bash
rustup component add miri --toolchain nightly-2024-10-12-aarch64-apple-darwin
rustup run nightly-2024-10-12-aarch64-apple-darwin miri --version
```

Expected: a Miri version tied to the pinned nightly. If unavailable, record `inconclusive/component_unavailable`; do not substitute another historical version.

- [ ] **Step 2: Run the pinned control**
```bash
POC=tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.toml
PINNED_TARGET="$(mktemp -d /tmp/rapx-join-poc-miri-pinned.XXXXXX)"
UNSOUNDAUDIT_STDLIB_POC_ACK=1 CARGO_TARGET_DIR="$PINNED_TARGET" \
  cargo +nightly-2024-10-12 miri run --manifest-path "$POC" -- stable-control
```

Expected: exit 0 with no Miri diagnostic.

Run `join-shrinking-first` under the same Miri target as a varying-borrow control. Expected: exit 0 because the reserved length is an overestimate rather than an underestimate.

- [ ] **Step 3: Run candidates separately and retain exit status**
```bash
for validation_case in \
  join-growing-first concat-growing-first join-growing-later \
  join-growing-first-nonempty-separator; do
  log="/tmp/rapx-${validation_case}-pinned-miri.log"
  set +e
  UNSOUNDAUDIT_STDLIB_POC_ACK=1 CARGO_TARGET_DIR="$PINNED_TARGET" \
    cargo +nightly-2024-10-12 miri run --manifest-path "$POC" -- "$validation_case" \
    >"$log" 2>&1
  status=$?
  set -e
  printf '%s exit=%s sha256=%s\n' "$validation_case" "$status" \
    "$(shasum -a 256 "$log" | awk '{print $1}')"
done
```

Interpret Miri UB as `confirmed_ub` for the pinned source only after its stack reaches `library/alloc/src/str.rs:164`. An ordinary panic is not UB; exit 0 is negative dynamic evidence; unsupported execution remains inconclusive.

- [ ] **Step 4: Create sanitized evidence from measured values**
Write `results.json` only after all commands finish. Its exact top-level keys are `schema_version`, `candidate`, `application_boundary_contains_unsafe`, `historical`, `current`, `upstream_duplicate_search`, and `disclosure_status`. Set the schema to `unsoundaudit-v2-stdlib-bug-validation-v1`, candidate to `alloc-str-join-varying-borrow`, the unsafe-boundary flag to `false`, and disclosure status to `not_authorized`.

`historical` records the pinned channel and rustc commit plus exactly six measured case objects: stable and shrinking controls plus four candidate cases. Each object has `name`, `outcome`, `exit_code`, `log_sha256`, `sink_reached`, and a sanitized one-line `diagnostic`. Set `current.status` and `upstream_duplicate_search.status` to `not_started`. Commit no raw logs, invented values, or absolute paths.

- [ ] **Step 5: Update the family row and commit**
Keep `poc_required` for negative or unsupported results; use `confirmed_ub` only for UB at the exact sink through the safe public call.

```bash
git add artifacts/unsoundaudit-v2/mac/stdlib-bug-validation/join-varying-borrow/results.json \
  docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md TASK_CONTEXT.md
git commit -m "test(unsoundaudit-v2): record pinned join poc evidence"
```

### Task 4: Check current Rust and upstream history

**Files:**
- Modify: `artifacts/unsoundaudit-v2/mac/stdlib-bug-validation/join-varying-borrow/results.json`
- Modify: `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Resolve current stable without executing the suspected path**
```bash
rustup component add rust-src --toolchain stable
rustc +stable -Vv
cargo +stable check --locked --jobs 1 --manifest-path \
  tests/unsoundaudit-v2/stdlib-bug-validation/join-varying-borrow/Cargo.toml
```

Inspect stable rust-src for the same capacity/copy pattern. Do not execute candidate cases natively; this compile and source comparison are supporting evidence, not dynamic UB validation.

- [ ] **Step 2: Install and resolve current Miri nightly**
```bash
rustup toolchain install nightly --profile minimal --component miri,rust-src
rustc +nightly -Vv
cargo +nightly miri --version
```

Record the resolved date and commit, never only the moving alias.

- [ ] **Step 3: Run the same six cases under current Miri**
Use a fresh target and Task 3's commands with `+nightly`. Control must exit 0. Record candidate outcomes and hashes independently of historical results.

- [ ] **Step 4: Inspect current official source**
Compare current `library/alloc/src/str.rs` with pinned lines 130-184. Record whether capacity is still calculated from one set of `Borrow` calls and copying uses another, and whether current checked bounds close the path.

- [ ] **Step 5: Search official upstream history**
Search Rust's official repository, issues, pull requests, release notes, and security policy for:

```text
join concat Borrow varying borrow join_generic_copy reserved_len spare_capacity unsound Miri
```

Record direct issue/PR/commit URLs and states. Search-result pages are not evidence.

- [ ] **Step 6: Classify and commit current impact**
- Current Miri UB, no duplicate: `confirmed_ub/current_unreported_candidate`.
- Current Miri UB, duplicate: `confirmed_ub/already_reported`.
- Historical UB, current fixed: `confirmed_ub/historical_fixed`.
- No UB plus complete source proof: `confirmed_safe`.
- Otherwise: `inconclusive`.

```bash
python3 -m json.tool \
  artifacts/unsoundaudit-v2/mac/stdlib-bug-validation/join-varying-borrow/results.json >/dev/null
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_stdlib_bug_validation.py
git diff --check
git add artifacts/unsoundaudit-v2/mac/stdlib-bug-validation/join-varying-borrow/results.json \
  docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md TASK_CONTEXT.md
git commit -m "audit(unsoundaudit-v2): classify current join soundness"
```

### Task 5: Prepare but do not submit disclosure

**Files:**
- Create conditionally: `docs/unsoundaudit-v2/RUST_STDLIB_JOIN_BUG_DRAFT.md`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Apply the eligibility gate**
Create the draft only when current UB is confirmed through safe public code and remains unfixed. Otherwise record why no draft is appropriate and skip the remaining task steps.

- [ ] **Step 2: Write the complete draft**
```markdown
# Safe varying `Borrow<str>` can violate `join/concat` memory safety

## Summary
## Minimal safe-only reproduction
## Exact toolchains and Miri output
## Root cause
## Affected current source
## Impact and reachability
## Suggested fix direction
## Existing issue/PR search
## Discovery provenance
## Disclosure status
```

Include the minimal PoC, current versions, diagnostic excerpt, and log hashes. State that UnsoundAudit identified the unchecked range. Include no machine path.

- [ ] **Step 3: Commit without external action**
```bash
git add docs/unsoundaudit-v2/RUST_STDLIB_JOIN_BUG_DRAFT.md TASK_CONTEXT.md
git commit -m "docs(unsoundaudit-v2): draft Rust join soundness report"
```

Stop and ask the user to approve, revise, or reject disclosure. Opening an issue or sending a report is not automatic.

### Task 6: Close the remaining 22 families by proof first

**Files:**
- Modify: `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md`
- Modify: `scripts/unsoundaudit-v2/test_stdlib_bug_validation.py`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Audit clamped and checked ranges**
For array `IntoIter`, `VecDeque::truncate`, `get_many_mut`, and `OsStr`, write the complete inequality or uniqueness proof. Include panic/unwind ordering where destructors occur. Promote only fully covered families to `confirmed_safe`.

- [ ] **Step 2: Audit returned-index contracts**
For `Split/RSplit`, prove `position/rposition` gives `idx < len`, `idx + 1 <= len` cannot overflow, and the same slice reaches the sink.

- [ ] **Step 3: Audit pointer and capacity summaries**
For `rotate_left/right`, SipHash, and `Cursor<Vec<u8>>`, expand checked arithmetic and allocation preconditions through the exact sink. If a safe callback, allocator, boundary, or unwind path remains unproved, retain `poc_required` and add a targeted case rather than assuming safety.

- [ ] **Step 4: Gate every remaining PoC-required family**
Do not advance to the final report while a family is still `poc_required` without pinned dynamic evidence. For each such family, write a candidate-specific follow-up spec and implementation plan with the same safe-boundary, control-case, pinned-Miri, current-source, and no-native-UB requirements used for `join/concat`. If a safe public reproducer cannot be constructed, classify it `inconclusive` and document the missing reachability fact rather than claiming safety.

- [ ] **Step 5: Add finding-ID reconciliation**
Extend the Python test to load active family membership from committed family reports and require each of the 37 IDs in exactly one audit row, with no suppressed ID present.

- [ ] **Step 6: Verify and commit**
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_stdlib_bug_validation.py
git diff --check
git add docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md \
  scripts/unsoundaudit-v2/test_stdlib_bug_validation.py TASK_CONTEXT.md
git commit -m "audit(unsoundaudit-v2): classify active stdlib families"
```

### Task 7: Publish the bounded local result

**Files:**
- Modify: `docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md`
- Modify: `TASK_CONTEXT.md`

- [ ] **Step 1: Summarize family and finding outcomes separately**
Report `confirmed_safe`, `likely_safe`, `poc_required`, `confirmed_ub`, and `inconclusive` by family and finding. Never report active count as bug count.

- [ ] **Step 2: State evidence limits**
For negative Miri runs, say only that tested executions exposed no UB. Preserve exclusions for P3/P5/P6 real-project precision, cross-crate precision, and unreviewed callbacks.

- [ ] **Step 3: Run final hygiene**
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  scripts/unsoundaudit-v2/test_stdlib_bug_validation.py
git diff --check
test -z "$(find . -type d -name __pycache__ -o -type f -name '*.pyc')"
test -z "$(find . -type d -name target)"
git status --short
```

Expected: tests PASS; no cache, target, raw log, binary, or machine path is tracked.

- [ ] **Step 4: Commit the bounded report**
```bash
git add docs/unsoundaudit-v2/STDLIB_ACTIVE_FINDING_AUDIT.md TASK_CONTEXT.md
git commit -m "docs(unsoundaudit-v2): report active finding validation"
```

The task ends with a clean worktree and, if current UB is confirmed, a disclosure draft waiting for explicit user approval.
