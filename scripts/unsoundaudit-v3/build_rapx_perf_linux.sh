#!/usr/bin/env bash
# Build rapx-perf on Ubuntu for one of the two v3 nightlies.
# Usage: build_rapx_perf_linux.sh nightly-2026-08-14|nightly-2024-10-12
set -euo pipefail

TOOLCHAIN="${1:?toolchain}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/lwz/unsound_scanner/lwz23-recall-work}"
LOG_DIR="$WORK/logs"
TARGET_DIR="$WORK/rapx-build/target-$TOOLCHAIN"
BIN_DIR="$WORK/rapx-build/bin-$TOOLCHAIN"
mkdir -p "$LOG_DIR" "$TARGET_DIR" "$BIN_DIR"

# crates.io and static.rust-lang.org TLS often fail from this host.
# Prefer USTC sparse index; do not send that traffic through 1082/7890.
unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="localhost,127.0.0.1,.ustc.edu.cn,mirrors.ustc.edu.cn"
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export RUSTUP_TOOLCHAIN="$TOOLCHAIN"
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

CARGO="$(rustup which --toolchain "$TOOLCHAIN" cargo)"
RUSTC="$(rustup which --toolchain "$TOOLCHAIN" rustc)"
SYSROOT="$("$RUSTC" --print sysroot)"
export RUSTC RUST_SYSROOT="$SYSROOT"
export LD_LIBRARY_PATH="$SYSROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-15/lib}"
if [[ -f /usr/include/z3.h ]]; then
  export Z3_SYS_Z3_HEADER=/usr/include/z3.h
fi
if [[ -f /usr/lib/x86_64-linux-gnu/libz3.so ]]; then
  export RUSTFLAGS="${RUSTFLAGS:-} -Lnative=/usr/lib/x86_64-linux-gnu"
fi
export CARGO_HOME="${CARGO_HOME:-$WORK/cargo-home}"
export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
mkdir -p "$CARGO_HOME"
CFG="$CARGO_HOME/config.toml"
if [[ ! -f "$CFG" ]]; then
  cat >"$CFG" <<'TOML'
[source.crates-io]
replace-with = "ustc"

[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"

[net]
git-fetch-with-cli = true
retry = 3
TOML
fi

COMMIT=$("$RUSTC" -vV | awk "/^commit-hash:/{print \$2; exit}")
if [[ -z "$COMMIT" || ${#COMMIT} -ne 40 ]]; then
  echo "could not parse rustc commit-hash" >&2
  exit 1
fi
export UNSOUND_SCANNER_RAP_BUILD_RUSTC_COMMIT="$COMMIT"
echo "RAP_BUILD_RUSTC_COMMIT=$COMMIT"

LOG="$LOG_DIR/build-rapx-perf-$TOOLCHAIN.log"
{
  echo "start $(date -Iseconds) toolchain=$TOOLCHAIN commit=$COMMIT"
  "$RUSTC" -vV
  "$CARGO" -vV
  "$CARGO" build --manifest-path "$ROOT/rapx/Cargo.toml" --locked \
    --profile rapx-perf --bins --jobs "$CARGO_BUILD_JOBS"
  cp -f "$TARGET_DIR/rapx-perf/cargo-rapx" "$TARGET_DIR/rapx-perf/rapx" "$BIN_DIR/"
  "$BIN_DIR/rapx" -Vv || true
  echo "done $(date -Iseconds)"
} 2>&1 | tee "$LOG"

echo "BIN_DIR=$BIN_DIR"
echo "LOG=$LOG"
