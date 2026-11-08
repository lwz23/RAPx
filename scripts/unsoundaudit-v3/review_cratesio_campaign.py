#!/usr/bin/env python3
"""Deterministic review sampling, validation, metrics, and report rendering for crates.io robustness campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cratesio_campaign_common import (
    CampaignError,
    canonical_json,
    load_policy,
    read_json,
    read_jsonl,
    sha256,
    stable_rank,
    write_json,
    exact_keys,
    exact_hex,
)


ACTIVE_LABELS = {"confirmed_bug", "likely_bug", "false_positive", "inconclusive"}
SAMPLED_LABELS = {
    "classification_supported",
    "missed_active_witness",
    "invalid_safety_proof",
    "inconclusive",
}
CONFIRMATION_CHOICES = {"not_requested", "confirmed", "disputed", "inconclusive"}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def precision_metrics(labels: list[str]) -> dict[str, Any]:
    decidable_labels = [l for l in labels if l != "inconclusive"]
    decidable = len(decidable_labels)
    confirmed = sum(1 for l in labels if l == "confirmed_bug")
    likely = sum(1 for l in labels if l == "likely_bug")
    false_positive = sum(1 for l in labels if l == "false_positive")
    inconclusive = sum(1 for l in labels if l == "inconclusive")
    fp_fraction = {"numerator": false_positive, "denominator": decidable}
    fp_rate = (false_positive / decidable) if decidable > 0 else None
    return {
        "total": len(labels),
        "decidable": decidable,
        "confirmed_bug": confirmed,
        "likely_bug": likely,
        "false_positive": false_positive,
        "inconclusive": inconclusive,
        "false_positive_fraction": fp_fraction,
        "false_positive_rate": fp_rate,
        "false_positive_wilson": wilson(false_positive, decidable) if decidable > 0 else None,
    }


def build_review_template(
    policy: dict[str, Any],
    campaign: dict[str, Any],
    campaign_dir: Path | None = None,
) -> dict[str, Any]:
    review_seed = policy["review_seed"]
    suppressed_quota = policy.get("suppressed_quota_per_crate", 3)
    unestablished_quota = policy.get("unestablished_quota_per_crate", 5)

    rows: list[dict[str, Any]] = []

    for pkg_record in campaign.get("packages", []):
        if pkg_record.get("terminal_state") != "complete_exact":
            continue

        pkg_name = pkg_record["name"]
        pkg_version = pkg_record["version"]
        pkg_id = f"{pkg_name}@{pkg_version}"

        if campaign_dir is None:
            continue

        mvp_dir = campaign_dir / "packages" / pkg_name / "mvp"

        # 1. Active families: 100% census
        families_file = mvp_dir / "families.jsonl"
        active_families = read_jsonl(families_file) if families_file.exists() else []
        for fam in active_families:
            fam_id = fam.get("family_id") or hashlib.sha256(canonical_json(fam)).hexdigest()
            fam_pattern = (
                fam.get("pattern")
                or fam.get("key", {}).get("primary_pattern")
                or fam.get("representatives", {}).get("longest", {}).get("primary_pattern")
                or "P1"
            )
            contract_precision = (
                fam.get("contract_precision")
                or fam.get("representatives", {}).get("longest", {}).get("contract", {}).get("precision")
                or "generic"
            )
            propagation_depth = (
                fam.get("propagation_depth")
                or fam.get("representatives", {}).get("longest", {}).get("propagation", {}).get("depth")
                or "inter_procedural"
            )
            row = {
                "population": "active",
                "package": pkg_id,
                "stable_id": fam_id,
                "family_id": fam_id,
                "pattern": fam_pattern,
                "contract_precision": contract_precision,
                "reason": None,
                "propagation_depth": propagation_depth,
                "evidence": fam,
                "label": None,
                "reviewer": None,
                "rationale": None,
                "reproduction": None,
                "independent_confirmation": "not_requested",
            }
            rows.append(row)

        # 2. Suppressed candidates: up to suppressed_quota per crate
        suppressed_file = mvp_dir / "suppression-ledger.jsonl"
        if not suppressed_file.exists():
            suppressed_file = mvp_dir / "suppression_ledger.jsonl"
        raw_suppressed = read_jsonl(suppressed_file) if suppressed_file.exists() else []
        # Group suppressed by structural ID
        suppressed_groups: dict[str, list[dict[str, Any]]] = {}
        for sup in raw_suppressed:
            cand = sup.get("candidate", sup)
            sup_sig = (
                cand.get("primary_pattern") or sup.get("pattern"),
                cand.get("public_safe_root", {}).get("owner") or sup.get("public_root"),
                cand.get("unsafe_site", {}).get("site_id") or sup.get("unsafe_site_id"),
                sup.get("reason"),
            )
            grp_id = hashlib.sha256(canonical_json(sup_sig)).hexdigest()
            suppressed_groups.setdefault(grp_id, []).append(sup)

        sorted_sup_group_ids = sorted(
            suppressed_groups.keys(),
            key=lambda gid: stable_rank(review_seed, f"{pkg_id}:suppressed:{gid}"),
        )
        selected_sup_ids = sorted_sup_group_ids[:suppressed_quota]

        for gid in selected_sup_ids:
            rep = suppressed_groups[gid][0]
            cand = rep.get("candidate", rep)
            row = {
                "population": "suppressed",
                "package": pkg_id,
                "stable_id": gid,
                "family_id": gid,
                "pattern": cand.get("primary_pattern") or rep.get("pattern", "P1"),
                "contract_precision": cand.get("contract", {}).get("precision") or rep.get("contract_precision", "generic"),
                "reason": rep.get("reason"),
                "propagation_depth": cand.get("propagation", {}).get("depth") or rep.get("propagation_depth", "intra_procedural"),
                "evidence": rep,
                "label": None,
                "reviewer": None,
                "rationale": None,
                "reproduction": None,
                "independent_confirmation": "not_requested",
            }
            rows.append(row)

        # 3. Unestablished decisions: up to unestablished_quota per crate
        unest_file = mvp_dir / "unestablished.jsonl"
        if not unest_file.exists():
            unest_file = mvp_dir / "unestablished_decisions.jsonl"
        raw_unest = read_jsonl(unest_file) if unest_file.exists() else []
        unest_groups: dict[str, list[dict[str, Any]]] = {}
        for un in raw_unest:
            src = un.get("source", {})
            un_sig = (
                un.get("attempted_pattern"),
                un.get("public_root"),
                un.get("unsafe_site_id"),
                un.get("operand_role"),
                src.get("kind") if isinstance(src, dict) else str(src),
                src.get("identity") if isinstance(src, dict) else "",
                un.get("primary_reason"),
            )
            grp_id = hashlib.sha256(canonical_json(un_sig)).hexdigest()
            unest_groups.setdefault(grp_id, []).append(un)

        sorted_unest_group_ids = sorted(
            unest_groups.keys(),
            key=lambda gid: stable_rank(review_seed, f"{pkg_id}:unestablished:{gid}"),
        )
        selected_unest_ids = sorted_unest_group_ids[:unestablished_quota]

        for gid in selected_unest_ids:
            rep = unest_groups[gid][0]
            row = {
                "population": "unestablished",
                "package": pkg_id,
                "stable_id": gid,
                "family_id": gid,
                "pattern": rep.get("attempted_pattern", rep.get("pattern", "P1")),
                "contract_precision": rep.get("contract_precision", "generic"),
                "reason": rep.get("primary_reason", rep.get("reason")),
                "propagation_depth": rep.get("propagation_depth", "intra_procedural"),
                "evidence": rep,
                "label": None,
                "reviewer": None,
                "rationale": None,
                "reproduction": None,
                "independent_confirmation": "not_requested",
            }
            rows.append(row)

    return {
        "schema_version": "rap-unsoundaudit-cratesio-campaign-review-v1",
        "campaign_id": policy["campaign_id"],
        "review_seed": review_seed,
        "rows": rows,
    }


def validate_review(review: dict[str, Any], campaign: dict[str, Any]) -> None:
    rows = review.get("rows", [])
    if not isinstance(rows, list):
        raise CampaignError("review rows must be a list")

    for i, row in enumerate(rows):
        pop = row.get("population")
        if pop not in {"active", "suppressed", "unestablished"}:
            raise CampaignError(f"row {i}: invalid population {pop!r}")

        reviewer = row.get("reviewer")
        if not reviewer or not isinstance(reviewer, str) or not reviewer.strip():
            raise CampaignError(f"row {i}: missing or empty reviewer")

        rationale = row.get("rationale")
        if not rationale or not isinstance(rationale, str) or not rationale.strip():
            raise CampaignError(f"row {i}: missing or empty rationale")

        label = row.get("label")
        if pop == "active":
            if label not in ACTIVE_LABELS:
                raise CampaignError(f"row {i} (active): invalid label {label!r}")
        else:
            if label not in SAMPLED_LABELS:
                raise CampaignError(f"row {i} ({pop}): invalid sample label {label!r}")

        ind_conf = row.get("independent_confirmation")
        if ind_conf not in CONFIRMATION_CHOICES:
            raise CampaignError(f"row {i}: invalid independent_confirmation {ind_conf!r}")


def render_markdown_report(
    summary: dict[str, Any],
    review: dict[str, Any],
    roster: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# crates.io 10-Crate Robustness Campaign Report ({summary['campaign_id']})\n")
    lines.append("## Executive Summary\n")
    lines.append(f"- **Campaign ID**: `{summary['campaign_id']}`")
    lines.append(f"- **Selection Date**: `{summary['selection_date']}`")
    lines.append(f"- **Evaluated Toolchain**: `{protocol.get('toolchain', {}).get('channel', 'nightly-2026-08-14')}` (rustc commit `{protocol.get('toolchain', {}).get('rustc_commit', 'unknown')}`)")
    lines.append(f"- **Evaluated Profile**: `rapx-perf` on `aarch64-apple-darwin`")
    lines.append(f"- **Wall-clock timeout**: {protocol.get('execution_limits', {}).get('wall_time_seconds', 900)}s per process group")
    lines.append("")

    lines.append("## Robustness & Terminal States\n")
    lines.append("| Crate | Version | Domain | Rust LOC | Terminal State | Control Wall (s) | Scan Wall (s) | Peak Mem (MB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pkg in summary["package_results"]:
        lines.append(
            f"| `{pkg['name']}` | `{pkg['version']}` | `{pkg.get('domain', '-')}` | {pkg.get('physical_rust_lines', '-')} | `{pkg['terminal_state']}` | {pkg.get('control_time_seconds', '-')} | {pkg.get('scan_time_seconds', '-')} | {pkg.get('peak_memory_mb', '-')} |"
        )
    lines.append("")

    lines.append("## Terminal States Aggregate\n")
    lines.append(f"- **Complete & Exact (100% Exact Coverage)**: {summary['terminal_states']['complete_exact']}/10")
    lines.append(f"- **Partial Coverage**: {summary['terminal_states']['partial_coverage']}/10")
    lines.append(f"- **Scan Failed**: {summary['terminal_states']['scan_failed']}/10")
    lines.append(f"- **Scan Timeout**: {summary['terminal_states']['scan_timeout']}/10")
    lines.append(f"- **Control Failed**: {summary['terminal_states']['control_failed']}/10")
    lines.append("")

    lines.append("## Active Family Precision Review\n")
    p_met = summary["active_precision"]
    lines.append(f"- **Total Active Families**: {p_met['total']}")
    lines.append(f"- **Decidable Families**: {p_met['decidable']}")
    lines.append(f"- **Confirmed Bugs**: {p_met['confirmed_bug']}")
    lines.append(f"- **Likely Bugs**: {p_met['likely_bug']}")
    lines.append(f"- **False Positives**: {p_met['false_positive']}")
    lines.append(f"- **Inconclusive**: {p_met['inconclusive']}")
    if p_met["false_positive_rate"] is not None:
        fp_pct = p_met["false_positive_rate"] * 100.0
        lines.append(f"- **False Positive Rate**: {fp_pct:.2f}% ({p_met['false_positive_fraction']['numerator']}/{p_met['false_positive_fraction']['denominator']})")
    else:
        lines.append("- **False Positive Rate**: N/A (0 decidable active families)")
    if p_met.get("false_positive_wilson"):
        w = p_met["false_positive_wilson"]
        lines.append(f"- **Exploratory 95% Wilson Interval**: [{w['low'] * 100.0:.2f}%, {w['high'] * 100.0:.2f}%]")
    lines.append("")

    lines.append("## Sampled Audits (Suppressed & Unestablished)\n")
    lines.append(f"- **Audited Suppressed Groups**: {summary['sampled_audits']['suppressed_count']}")
    lines.append(f"- **Audited Unestablished Groups**: {summary['sampled_audits']['unestablished_count']}")
    lines.append(f"- **Sample Decisions Breakdown**: {json.dumps(summary['sampled_audits']['decisions_breakdown'], indent=2)}")
    lines.append("")

    lines.append("## Limitations and Scope\n")
    lines.append("1. **Scope**: This robustness campaign evaluates default-feature library targets on macOS `aarch64-apple-darwin` under clean serial execution.")
    lines.append("2. **Purposive Sample**: The 10 crates represent high-download strata with package-local `unsafe` code. The active census is purposive, and Wilson intervals are exploratory rather than ecosystem-wide population estimates.")
    lines.append("3. **Independent Confirmation**: Confirmed/likely bug claims require safe callers, minimal reproductions, and independent maintainer or expert confirmation before formal CVE disclosure.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and report crates.io robustness campaign.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    template_parser = subparsers.add_parser("template", help="Generate review template")
    template_parser.add_argument("--policy", type=Path, required=True)
    template_parser.add_argument("--roster", type=Path, required=True)
    template_parser.add_argument("--campaign", type=Path, required=True)
    template_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate filled review template")
    validate_parser.add_argument("--policy", type=Path, required=True)
    validate_parser.add_argument("--campaign", type=Path, required=True)
    validate_parser.add_argument("--review", type=Path, required=True)

    report_parser = subparsers.add_parser("report", help="Generate summary, CSV, and markdown report")
    report_parser.add_argument("--policy", type=Path, required=True)
    report_parser.add_argument("--roster", type=Path, required=True)
    report_parser.add_argument("--protocol", type=Path, required=True)
    report_parser.add_argument("--campaign", type=Path, required=True)
    report_parser.add_argument("--review", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, default=None)
    report_parser.add_argument("--evidence-manifest", type=Path, default=None)
    report_parser.add_argument("--summary", type=Path, default=None)
    report_parser.add_argument("--csv", type=Path, default=None)
    report_parser.add_argument("--markdown", type=Path, default=None)

    args = parser.parse_args()

    if args.action == "template":
        policy = load_policy(args.policy)
        campaign = read_json(args.campaign)
        template = build_review_template(policy, campaign, args.campaign.parent)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, template)
        print(f"Review template generated at {args.output} ({len(template['rows'])} rows).")

    elif args.action == "validate":
        campaign = read_json(args.campaign)
        review = read_json(args.review)
        validate_review(review, campaign)
        print(f"Review at {args.review} successfully validated.")

    elif args.action == "report":
        policy = load_policy(args.policy)
        roster = read_json(args.roster)
        protocol = read_json(args.protocol)
        campaign = read_json(args.campaign)
        review = read_json(args.review)
        validate_review(review, campaign)

        out_dir = args.output_dir or Path("docs/unsoundaudit-v3")
        summary_path = args.summary or (out_dir / "cratesio-campaign-2026-08-21-summary.json")
        csv_path = args.csv or (out_dir / "cratesio-campaign-2026-08-21-review.csv")
        markdown_path = args.markdown or (out_dir / "cratesio-campaign-2026-08-21-report.md")
        manifest_path = args.evidence_manifest or (out_dir / "cratesio-campaign-2026-08-21-evidence-manifest.json")

        # Compute summary
        terminal_states = {
            "complete_exact": sum(1 for p in campaign["packages"] if p["terminal_state"] == "complete_exact"),
            "partial_coverage": sum(1 for p in campaign["packages"] if p["terminal_state"] == "partial_coverage"),
            "scan_failed": sum(1 for p in campaign["packages"] if p["terminal_state"] == "scan_failed"),
            "scan_timeout": sum(1 for p in campaign["packages"] if p["terminal_state"] == "scan_timeout"),
            "control_failed": sum(1 for p in campaign["packages"] if p["terminal_state"] == "control_failed"),
        }

        active_labels = [r["label"] for r in review["rows"] if r["population"] == "active"]
        active_prec = precision_metrics(active_labels)

        sampled_rows = [r for r in review["rows"] if r["population"] in {"suppressed", "unestablished"}]
        sampled_decisions = {}
        for r in sampled_rows:
            sampled_decisions[r["label"]] = sampled_decisions.get(r["label"], 0) + 1

        pkg_results = []
        for p in campaign["packages"]:
            roster_pkg = next((rp for rp in roster["packages"] if rp["name"] == p["name"]), {})
            scan_data = p.get("scan") or {}
            ctrl_data = p.get("control") or {}
            peak_bytes = scan_data.get("max_physical_footprint_bytes") or scan_data.get("peak_physical_footprint_bytes") or 0
            ctrl_time = ctrl_data.get("elapsed_seconds") or ctrl_data.get("wall_time_seconds")
            scan_time = scan_data.get("elapsed_seconds") or scan_data.get("wall_time_seconds")
            pkg_results.append(
                {
                    "name": p["name"],
                    "version": p["version"],
                    "domain": roster_pkg.get("domain"),
                    "physical_rust_lines": roster_pkg.get("admission", {}).get("physical_rust_lines"),
                    "terminal_state": p["terminal_state"],
                    "control_time_seconds": round(ctrl_time, 3) if ctrl_time else None,
                    "scan_time_seconds": round(scan_time, 3) if scan_time else None,
                    "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2) if peak_bytes else None,
                }
            )

        summary = {
            "schema_version": "rap-unsoundaudit-cratesio-campaign-summary-v1",
            "campaign_id": policy["campaign_id"],
            "selection_date": policy["selection_date"],
            "policy_sha256": roster.get("policy_sha256"),
            "protocol_sha256": protocol.get("protocol_sha256"),
            "terminal_states": terminal_states,
            "active_precision": active_prec,
            "sampled_audits": {
                "suppressed_count": sum(1 for r in review["rows"] if r["population"] == "suppressed"),
                "unestablished_count": sum(1 for r in review["rows"] if r["population"] == "unestablished"),
                "decisions_breakdown": sampled_decisions,
            },
            "package_results": pkg_results,
            "rerun_verification": campaign.get("reruns"),
        }

        # Write summary JSON
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, summary)

        # Write CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "population",
                "package",
                "stable_id",
                "pattern",
                "contract_precision",
                "reason",
                "propagation_depth",
                "label",
                "reviewer",
                "rationale",
                "independent_confirmation",
            ])
            for r in review["rows"]:
                writer.writerow([
                    r["population"],
                    r["package"],
                    r["stable_id"],
                    r["pattern"],
                    r["contract_precision"],
                    r.get("reason") or "",
                    r["propagation_depth"],
                    r["label"],
                    r["reviewer"],
                    r["rationale"],
                    r["independent_confirmation"],
                ])

        # Write Markdown Report
        md_text = render_markdown_report(summary, review, roster, protocol)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(md_text, encoding="utf-8")

        # Write Evidence Manifest
        manifest = {
            "schema_version": "rap-unsoundaudit-cratesio-campaign-evidence-manifest-v1",
            "campaign_id": policy["campaign_id"],
            "selection_date": policy["selection_date"],
            "files": {
                "policy": str(args.policy),
                "policy_sha256": sha256(args.policy),
                "roster": str(args.roster),
                "roster_sha256": sha256(args.roster),
                "protocol": str(args.protocol),
                "protocol_sha256": sha256(args.protocol),
                "campaign": str(args.campaign),
                "campaign_sha256": sha256(args.campaign),
                "review": str(args.review),
                "review_sha256": sha256(args.review),
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "csv": str(csv_path),
                "csv_sha256": sha256(csv_path),
                "report": str(markdown_path),
                "report_sha256": sha256(markdown_path),
            },
        }
        write_json(manifest_path, manifest)
        print(f"Campaign reporting and evidence manifest generation complete at {manifest_path}.")


if __name__ == "__main__":
    main()
