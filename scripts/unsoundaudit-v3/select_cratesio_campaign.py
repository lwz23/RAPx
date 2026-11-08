#!/usr/bin/env python3
"""Official crates.io roster selection, safe archive inspection, and unsafe source admission."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tomllib
from typing import Any, Callable
import urllib.error
import urllib.request

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cratesio_campaign_common import (
    CampaignError,
    canonical_json,
    load_policy,
    sha256,
    write_json,
)


CRATES_IO_API = "https://crates.io/api/v1/crates/{name}"
CRATES_IO_DOWNLOAD = "https://crates.io/api/v1/crates/{name}/{version}/download"
USER_AGENT = "RAP-UnsoundAudit-CratesIoCampaign/1.0 (research@example.org)"


def fetch_crate_metadata_http(name: str) -> dict[str, Any]:
    url = CRATES_IO_API.format(name=name)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            import time
            time.sleep(0.15 * (attempt + 1))
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except Exception as e:
            last_err = e
            continue
    raise CampaignError(f"failed to fetch crates.io metadata for {name}: {last_err}") from last_err


def fetch_crate_archive_http(name: str, version: str) -> bytes:
    url = CRATES_IO_DOWNLOAD.format(name=name, version=version)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            import time
            time.sleep(0.2 * (attempt + 1))
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            continue
    raise CampaignError(f"failed to download archive for {name}@{version}: {last_err}") from last_err


def latest_stable(metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    crate_info = metadata.get("crate", {})
    max_stable = crate_info.get("max_stable_version")
    if not max_stable or "-" in max_stable:
        raise CampaignError(
            f"latest stable release for {crate_info.get('name')} is missing or prerelease: {max_stable!r}"
        )
    matching = [row for row in metadata.get("versions", []) if row.get("num") == max_stable]
    if len(matching) != 1:
        raise CampaignError(f"version {max_stable} not found uniquely in versions list")
    ver_obj = matching[0]
    if ver_obj.get("yanked") is not False:
        raise CampaignError(f"latest stable version {max_stable} is yanked")
    return max_stable, ver_obj


def safe_members(archive: tarfile.TarFile, expected_root: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        parts = PurePosixPath(member.name).parts
        if not parts or parts[0] != expected_root or ".." in parts:
            raise CampaignError(f"crate archive member {member.name} escapes expected root {expected_root}")
        if member.issym() or member.islnk() or member.isdev():
            raise CampaignError(f"crate archive member {member.name} contains forbidden link or device")
    return members


def has_lexical_unsafe(source_code: str) -> tuple[bool, int]:
    i = 0
    n = len(source_code)
    count = 0
    while i < n:
        # 1. Line comment //
        if source_code[i : i + 2] == "//":
            i = source_code.find("\n", i + 2)
            if i == -1:
                break
            i += 1
            continue
        # 2. Block comment /* ... */ with nesting
        if source_code[i : i + 2] == "/*":
            i += 2
            depth = 1
            while i < n and depth > 0:
                if source_code[i : i + 2] == "/*":
                    depth += 1
                    i += 2
                elif source_code[i : i + 2] == "*/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            continue
        # 3. Raw string literal r"..." or r#"..."# or br#"..."#
        if (
            source_code[i : i + 2] == 'r"'
            or source_code[i : i + 2] == "r#"
            or source_code[i : i + 3] == 'br"'
            or source_code[i : i + 3] == "br#"
        ):
            is_byte = source_code[i] == "b"
            start_idx = i + 1 if is_byte else i
            num_hashes = 0
            cur = start_idx + 1  # after 'r'
            while cur < n and source_code[cur] == "#":
                num_hashes += 1
                cur += 1
            if cur < n and source_code[cur] == '"':
                closing = '"' + ("#" * num_hashes)
                end_pos = source_code.find(closing, cur + 1)
                if end_pos == -1:
                    break
                i = end_pos + len(closing)
                continue
        # 4. Normal string "..." or byte string b"..."
        if source_code[i] == '"' or source_code[i : i + 2] == 'b"':
            i = i + 2 if source_code[i] == "b" else i + 1
            while i < n:
                if source_code[i] == "\\":
                    i += 2
                elif source_code[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            continue
        # 5. Character literal '...' or b'...'
        if source_code[i] == "'" or source_code[i : i + 2] == "b'":
            i = i + 2 if source_code[i] == "b" else i + 1
            while i < n:
                if source_code[i] == "\\":
                    i += 2
                elif source_code[i] == "'":
                    i += 1
                    break
                else:
                    i += 1
            continue
        # 6. Identifier or Keyword
        if source_code[i].isalpha() or source_code[i] == "_":
            start = i
            while i < n and (source_code[i].isalnum() or source_code[i] == "_"):
                i += 1
            token = source_code[start:i]
            if token == "unsafe":
                count += 1
            continue
        i += 1
    return (count > 0, count)


def inspect_archive(archive: tarfile.TarFile, package_name: str, package_version: str) -> dict[str, Any]:
    expected_root = f"{package_name}-{package_version}"
    members = safe_members(archive, expected_root)

    manifest_name = f"{expected_root}/Cargo.toml"
    manifest_member = next((m for m in members if m.name == manifest_name), None)
    if manifest_member is None:
        raise CampaignError(f"archive {expected_root} missing root Cargo.toml")

    manifest_file = archive.extractfile(manifest_member)
    if manifest_file is None:
        raise CampaignError(f"failed to extract {manifest_name}")
    manifest = tomllib.loads(manifest_file.read().decode("utf-8"))

    pkg_section = manifest.get("package", {})
    if pkg_section.get("name") != package_name:
        raise CampaignError(f"manifest package name {pkg_section.get('name')} != {package_name}")
    if pkg_section.get("version") != package_version:
        raise CampaignError(f"manifest package version {pkg_section.get('version')} != {package_version}")

    lib_path = manifest.get("lib", {}).get("path", "src/lib.rs")
    expected_lib_member = f"{expected_root}/{lib_path}"
    if not any(m.name == expected_lib_member for m in members):
        raise CampaignError(f"library target {lib_path} not found in archive {expected_root}")

    # Inspect .rs files under package, excluding tests, examples, benches, target
    excluded_dirs = {"tests", "examples", "benches", "target"}
    evidence_paths: list[str] = []
    total_unsafe_tokens = 0
    scanned_rust_files: list[str] = []
    total_physical_lines = 0

    for member in members:
        if not member.isfile():
            continue
        rel = member.name[len(expected_root) + 1 :]
        parts = PurePosixPath(rel).parts
        if not parts or not rel.endswith(".rs"):
            continue
        if any(part in excluded_dirs for part in parts):
            continue
        if parts == ("build.rs",):
            continue

        scanned_rust_files.append(rel)
        extracted = archive.extractfile(member)
        if extracted is None:
            continue
        code_bytes = extracted.read()
        try:
            code_text = code_bytes.decode("utf-8")
        except UnicodeDecodeError:
            code_text = code_bytes.decode("latin-1")
        total_physical_lines += len(code_text.splitlines())

        has_u, u_count = has_lexical_unsafe(code_text)
        if has_u:
            evidence_paths.append(rel)
            total_unsafe_tokens += u_count

    if total_unsafe_tokens == 0:
        raise CampaignError(
            f"package {package_name}@{package_version} has no package-local unsafe code in library source"
        )

    return {
        "has_unsafe": True,
        "unsafe_token_count": total_unsafe_tokens,
        "rust_file_count": len(scanned_rust_files),
        "physical_rust_lines": total_physical_lines,
        "evidence_paths": sorted(evidence_paths),
        "lib_target_path": lib_path,
    }


def select_domain(
    domain: str,
    candidates: list[str],
    fetch_metadata: Callable[[str], dict[str, Any]],
    fetch_archive: Callable[[str, str], bytes],
) -> dict[str, Any]:
    candidate_records: list[dict[str, Any]] = []
    for cand in candidates:
        meta = fetch_metadata(cand)
        ver, ver_obj = latest_stable(meta)
        crate_info = meta.get("crate", {})
        total_downloads = crate_info.get("downloads", ver_obj.get("downloads", 0))
        candidate_records.append(
            {
                "name": cand,
                "version": ver,
                "ver_obj": ver_obj,
                "total_downloads": total_downloads,
                "metadata": meta,
            }
        )

    # Sort candidates by (-total_downloads, name)
    candidate_records.sort(key=lambda item: (-item["total_downloads"], item["name"]))

    rejected_candidates: list[dict[str, Any]] = []
    for rank, cand in enumerate(candidate_records, start=1):
        name = cand["name"]
        ver = cand["version"]
        ver_obj = cand["ver_obj"]
        downloads = cand["total_downloads"]
        try:
            archive_bytes = fetch_archive(name, ver)
            computed_sha = hashlib.sha256(archive_bytes).hexdigest()
            pub_checksum = ver_obj.get("checksum")
            if pub_checksum and computed_sha != pub_checksum:
                raise CampaignError(
                    f"downloaded checksum {computed_sha} != published {pub_checksum}"
                )
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
                admission = inspect_archive(tar, name, ver)
            return {
                "domain": domain,
                "name": name,
                "version": ver,
                "selection_rank": rank,
                "total_downloads": downloads,
                "published_checksum": pub_checksum or computed_sha,
                "archive_sha256": computed_sha,
                "admission": admission,
                "rejected_candidates": rejected_candidates,
                "raw_archive_bytes": archive_bytes,
            }
        except Exception as e:
            rejected_candidates.append(
                {
                    "name": name,
                    "version": ver,
                    "selection_rank": rank,
                    "total_downloads": downloads,
                    "reason": str(e),
                }
            )

    raise CampaignError(f"no candidate admitted for domain {domain}")


def select_campaign(
    policy: dict[str, Any],
    download_root: Path,
    fetch_metadata: Callable[[str], dict[str, Any]] = fetch_crate_metadata_http,
    fetch_archive: Callable[[str, str], bytes] = fetch_crate_archive_http,
) -> dict[str, Any]:
    pol_sha = hashlib.sha256(canonical_json(policy)).hexdigest()
    download_root.mkdir(parents=True, exist_ok=True)

    domains_pool = policy["candidate_pool"]
    selected_packages: list[dict[str, Any]] = []

    for domain in sorted(domains_pool):
        candidates = domains_pool[domain]
        selected = select_domain(domain, candidates, fetch_metadata, fetch_archive)
        # Save archive locally
        archive_bytes = selected.pop("raw_archive_bytes")
        archive_filename = f"{selected['name']}-{selected['version']}.crate"
        archive_file = download_root / archive_filename
        archive_file.write_bytes(archive_bytes)
        selected["archive_filename"] = archive_filename
        selected["archive_path"] = str(archive_file)
        selected_packages.append(selected)

    if len(selected_packages) != 10:
        raise CampaignError("campaign requires exactly ten selected packages")
    unique_names = {p["name"] for p in selected_packages}
    if len(unique_names) != 10:
        raise CampaignError("duplicate package names among selected packages")

    roster = {
        "schema_version": "rap-unsoundaudit-cratesio-campaign-roster-v1",
        "campaign_id": policy["campaign_id"],
        "selection_date": policy["selection_date"],
        "policy_sha256": pol_sha,
        "packages": selected_packages,
    }
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze crates.io campaign roster.")
    parser.add_argument("--policy", type=Path, required=True, help="Path to campaign policy JSON")
    parser.add_argument("--download-root", type=Path, required=True, help="Directory to save downloaded archives")
    parser.add_argument("--output", type=Path, required=True, help="Path to output roster JSON")
    args = parser.parse_args()

    policy = load_policy(args.policy)
    roster = select_campaign(policy, args.download_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, roster)
    print(f"Roster frozen successfully to {args.output} with {len(roster['packages'])} packages.")


if __name__ == "__main__":
    main()
