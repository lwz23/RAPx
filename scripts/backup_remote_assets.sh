#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/docs/remote-vulnerability-assets"
mkdir -p "$DEST_DIR"
mkdir -p "$DEST_DIR/plans"
mkdir -p "$DEST_DIR/tse_iue_study_v2"

REMOTE_HOST="phase14-ubuntu"
REMOTE_DIR="/home/lwz/unsound_scanner"

echo "=== Syncing remote vulnerability assets from ${REMOTE_HOST}:${REMOTE_DIR} ==="

FILES=(
  "github_issues_report_anonymous.md"
  "fixed_github_issues_report_anonymous.md"
  "Historical Vulnerability Cases.xlsx"
  "LLM_cases.md"
  "llm_stage_real_examples.md"
  "llm_stage_output_and_case.md"
  "summary.csv"
  "TASK_CONTEXT.md"
  "TSE_IUE_83_AUDIT_HANDOFF_MAC.md"
  "TSE_LEGACY_P1_P4_REAUDIT_MAC_HANDOFF.md"
)

for file in "${FILES[@]}"; do
  echo "Downloading: $file..."
  scp -o ConnectTimeout=10 "${REMOTE_HOST}:${REMOTE_DIR}/${file}" "${DEST_DIR}/" || echo "Warning: failed to download $file"
done

echo "Downloading superpower plans..."
scp -r -o ConnectTimeout=10 "${REMOTE_HOST}:${REMOTE_DIR}/docs/superpowers/plans/"* "${DEST_DIR}/plans/" || echo "Warning: failed to download plans"

echo "Downloading TSE study materials..."
scp -r -o ConnectTimeout=10 "${REMOTE_HOST}:${REMOTE_DIR}/docs/tse_iue_study_v2/"* "${DEST_DIR}/tse_iue_study_v2/" || echo "Warning: failed to download tse_iue_study_v2"

echo "=== Backup completed to $DEST_DIR ==="
ls -la "$DEST_DIR"
