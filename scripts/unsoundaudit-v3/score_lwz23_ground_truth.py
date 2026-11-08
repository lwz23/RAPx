#!/usr/bin/env python3
"""Score RAP MVP findings against issue-body ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lwz23_ground_truth as gt  # noqa: E402
import rescore_lwz23_coverage as rescore  # noqa: E402


def load_gt(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in gt.load_jsonl(path) if hasattr(gt, "load_jsonl") else _jsonl(path):
        out[(str(row["repository"]), str(row["issue_number"]))] = row
    return out


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scan-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    roster = args.repo_root / "docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1"
    issues = list(csv.DictReader((roster / "issues.csv").open()))
    catalog = { (r["repository"], str(r["issue_number"])): r for r in _jsonl(args.ground_truth) }
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for issue in issues:
        repo = issue["repository"]
        number = str(issue["issue_number"])
        rec = catalog.get((repo, number), {})
        actives: list[dict[str, Any]] = []
        sources: list[str] = []
        for scan in args.scan_roots:
            for mvp in rescore.collect_mvp_dirs(scan, repo):
                try:
                    sources.append(str(mvp.relative_to(scan)))
                except ValueError:
                    sources.append(str(mvp))
                actives.extend(rescore.load_active(mvp))
        hits = gt.score_actives(rec, actives) if rec else []
        verdict = "true_hit" if hits else ("no_artifact" if not actives else "miss")
        if rec.get("src_exists") is False:
            skip = "no_checkout"
        else:
            skip = ""
        preview = hits[0] if hits else {}
        rows.append(
            {
                "repository": repo,
                "issue_number": number,
                "issue_url": issue["issue_url"],
                "title": issue["title"],
                "verdict": verdict,
                "n_active": len(actives),
                "n_hits": len(hits),
                "hit_root": (preview.get("root") or "")[:120],
                "hit_callee": (preview.get("callee") or "")[:80],
                "hit_path": preview.get("path") or "",
                "hit_why": " | ".join(preview.get("why") or []),
                "gt_fns": ",".join(rec.get("fns") or []),
                "gt_paths": ",".join(rec.get("paths") or []),
                "gt_callees": ",".join(rec.get("callees") or []),
                "gt_located": rec.get("located_n") or 0,
                "gt_matchable": bool(rec.get("matchable") or rec.get("located_n")),
                "mvp_sources": " | ".join(sources),
                "skip": skip,
            }
        )
    csv_path = out_dir / "coverage_ground_truth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["verdict"] for r in rows)
    md = ["# Ground-truth RAP score", "", f"- issues: {len(rows)}"]
    for key, value in counts.most_common():
        md.append(f"- `{key}`: {value}")
    md += ["", "## true_hit", ""]
    for row in rows:
        if row["verdict"] == "true_hit":
            md.append(
                f"- {row['repository']}#{row['issue_number']} `{row['hit_root']}` → `{row['hit_callee']}` ({row['hit_why']})"
            )
    (out_dir / "coverage_ground_truth.md").write_text("\n".join(md) + "\n")
    print(dict(counts))
    print("wrote", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
