# RAP `-unsoundaudit` v2：Mac 实现与轻量验证交接说明

本文交给另一台 Mac 上的**新 Codex 线程**执行。它不是“建议清单”，而是实现边界、阶段门和验收口径。先读完本文，再修改代码。

## 0. 任务目标与硬边界

本分支要完成三件事：

1. 用统一的、crate 内过程间 MIR 摘要层替换现有 `unsoundaudit` 的局部/字符串式传播逻辑；
2. 将旧 P1 与旧 P3 合并为新 P1，并让 P1–P6 共用过程间传播基础；
3. 实现新 P4、受限版 P5.1 和 P6 的**静态候选规则**，用 27 个无依赖微型 library fixtures 做轻量验证。

本阶段的硬边界：

- 只在 fork `lwz23/RAPx` 的 `feature/unsoundaudit-v2-p1-p6` 分支工作；不得直接改上游 `safer-rust/RAPx` 或 Linux 现役目录。
- 不接触 Linux 上现役或刚结束、尚未显式冻结的 RAP campaign、coverage ledger 或 2,848 个历史成功基线。
- 不在真实第三方项目上扫描；不运行 Miri；不执行 fixture 中可能触发 UB 的函数。
- fixture 只允许 `cargo check --lib` 和 RAP 静态扫描。RAP 自身可以编译，并可运行不执行目标程序的纯 Rust unit tests。
- 项目并发为 1，Cargo jobs 为 1；每次验证使用 fresh target。
- 不通过更换 nightly、升级依赖、改写 lockfile 语义或扩大/缩小研究定义来“让测试通过”。
- Mac 产生的二进制、`target/`、Cargo 缓存和本机 dylib 不得提交或复制回 Linux；只返回源码、小型文本/JSON receipt 和审查摘要。
- 所有规则都称为 `heuristic candidate generator`。27 个构造 fixture 通过不等于真实项目检出率、precision、recall 或生态 coverage。

## 1. 固定的 Git 与源码基线

固定信息如下：

| 项目 | 固定值 |
|---|---|
| 远端 | `git@github.com:lwz23/RAPx.git` |
| 工作分支 | `feature/unsoundaudit-v2-p1-p6` |
| Linux v1 基线 tag | `unsoundaudit-v1-linux-baseline-20260812-9d36ece7` |
| tag 对应 commit | `9f04dbb377afa5ca0557a51ecb08c5f78ac4fc19` |
| `rapx/` 源码树 SHA-256 | `9d36ece76ddacc8ab052ecaf4f64f412f7d648a93c1acc102d418c499c9d84c1` |
| `unsoundaudit.rs` SHA-256 | `47112d602470fe5afd236ef164e5798a7c2fdc628352c5912a705c25ba2f01e0` |
| Rust toolchain | `nightly-2024-10-12` |
| rustc commit hash | `1bc403daadbebb553ccc211a0a8eebb73989665f` |
| Cargo commit hash | `15fbd2f607d4defc87053b8b76bf5038f2483cf4` |

完整机器可读信息见 `docs/unsoundaudit-v2/baseline_manifest_v1.json`。基线 hash 工具为 `scripts/unsoundaudit_v2_source_tree_sha256.py`。

### 1.1 Clone 与基线验证

选择一个新的、原先不存在的目录，然后执行：

```bash
set -euo pipefail
git -c core.autocrlf=false clone \
  --branch feature/unsoundaudit-v2-p1-p6 --single-branch \
  git@github.com:lwz23/RAPx.git RAPx-unsoundaudit-v2
cd RAPx-unsoundaudit-v2
git config core.autocrlf false
git fetch origin tag unsoundaudit-v1-linux-baseline-20260812-9d36ece7
git status --short --branch
```

必须满足以下检查；任意一项不满足都先停止，不要自行“修复”基线：

```bash
set -euo pipefail
test "$(git rev-parse 'unsoundaudit-v1-linux-baseline-20260812-9d36ece7^{commit}')" \
  = "9f04dbb377afa5ca0557a51ecb08c5f78ac4fc19"
git merge-base --is-ancestor \
  9f04dbb377afa5ca0557a51ecb08c5f78ac4fc19 HEAD
git diff --exit-code \
  unsoundaudit-v1-linux-baseline-20260812-9d36ece7..HEAD -- \
  rapx rust-toolchain.toml README.md
test "$(python3 scripts/unsoundaudit_v2_source_tree_sha256.py rapx)" \
  = "9d36ece76ddacc8ab052ecaf4f64f412f7d648a93c1acc102d418c499c9d84c1"
test "$(shasum -a 256 rapx/src/analysis/unsoundaudit.rs | awk '{print $1}')" \
  = "47112d602470fe5afd236ef164e5798a7c2fdc628352c5912a705c25ba2f01e0"
test -z "$(git status --porcelain)"
```

这里允许分支 HEAD 比基线 tag 多出交接文档、manifest 和纯 Python hash 工具；在首次实现修改前，`rapx/`、`rust-toolchain.toml` 与 `README.md` 必须仍与 tag 相同。

## 2. Mac 环境预检

### 2.1 不混用 CPU 架构

先记录：

```bash
uname -m
arch
sw_vers
xcode-select -p
xcrun --find clang
```

正常的原生组合是：

- Apple Silicon：`arm64` / Rust host `aarch64-apple-darwin` / ARM Homebrew；
- Intel Mac：`x86_64` / Rust host `x86_64-apple-darwin` / Intel Homebrew。

不要在同一次构建中混用 Rosetta x86_64 Rust、ARM Homebrew，或反过来。不要硬编码 `/opt/homebrew` 或 `/usr/local`，统一从 `brew --prefix` 派生路径。若必须接受 Xcode license 或安装 Command Line Tools，先告知用户，由用户完成需要管理员权限的操作。

### 2.2 必需依赖

需要：

- Xcode Command Line Tools；
- rustup；
- Homebrew；
- LLVM/libclang；
- Z3；
- CMake；
- pkg-config。

若尚未安装 Homebrew 包，可执行：

```bash
brew install llvm z3 cmake pkg-config
```

不要在这一步升级仓库依赖或更改 `Cargo.lock`。安装并核验精确 nightly：

```bash
rustup toolchain install nightly-2024-10-12 \
  --profile minimal \
  --component rustc-dev,rust-src,llvm-tools-preview
rustup component list --toolchain nightly-2024-10-12 --installed
rustc +nightly-2024-10-12 -Vv
cargo +nightly-2024-10-12 -Vv
```

`rustc -Vv` 中的 `commit-hash` 必须严格为：

```text
1bc403daadbebb553ccc211a0a8eebb73989665f
```

为当前 shell 设置本机构建环境。变量名不要覆盖系统的 `HOME`：

```bash
set -euo pipefail
RAPV2_BREW_PREFIX="$(brew --prefix)"
RAPV2_LLVM_PREFIX="$(brew --prefix llvm)"
RAPV2_Z3_PREFIX="$(brew --prefix z3)"
RAPV2_SYSROOT="$(rustc +nightly-2024-10-12 --print sysroot)"
RAPV2_RUSTC="$(rustup which --toolchain nightly-2024-10-12 rustc)"
RAPV2_CARGO="$(rustup which --toolchain nightly-2024-10-12 cargo)"
RAPV2_RUSTC_COMMIT="$(rustc +nightly-2024-10-12 -Vv | awk -F': ' '$1 == "commit-hash" {print $2}')"
RAPV2_CARGO_COMMIT="$(cargo +nightly-2024-10-12 -Vv | awk -F': ' '$1 == "commit-hash" {print $2}')"

test "$RAPV2_RUSTC_COMMIT" = "1bc403daadbebb553ccc211a0a8eebb73989665f"
test "$RAPV2_CARGO_COMMIT" = "15fbd2f607d4defc87053b8b76bf5038f2483cf4"
test -z "${RUSTFLAGS:-}" || {
  echo "Refusing to inherit pre-existing RUSTFLAGS for the exact baseline" >&2
  exit 1
}
test -z "${CARGO_ENCODED_RUSTFLAGS:-}" || {
  echo "Refusing to inherit CARGO_ENCODED_RUSTFLAGS for the exact baseline" >&2
  exit 1
}

export PATH="$RAPV2_LLVM_PREFIX/bin:$PATH"
export LIBCLANG_PATH="$RAPV2_LLVM_PREFIX/lib"
export PKG_CONFIG_PATH="$RAPV2_Z3_PREFIX/lib/pkgconfig:$RAPV2_LLVM_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export DYLD_LIBRARY_PATH="$RAPV2_SYSROOT/lib:$RAPV2_Z3_PREFIX/lib:$RAPV2_LLVM_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export Z3_SYS_Z3_HEADER="$RAPV2_Z3_PREFIX/include/z3.h"
export RUSTFLAGS="-Lnative=$RAPV2_Z3_PREFIX/lib"
export RUST_SYSROOT="$RAPV2_SYSROOT"

export UNSOUND_SCANNER_EXACT_CARGO_PATH="$RAPV2_CARGO"
export UNSOUND_SCANNER_EXACT_CARGO_COMMIT="$RAPV2_CARGO_COMMIT"
export RUSTC="$RAPV2_RUSTC"
export UNSOUND_SCANNER_EXPECTED_RUSTC_PATH="$RAPV2_RUSTC"
export UNSOUND_SCANNER_EXPECTED_RUSTC_COMMIT="$RAPV2_RUSTC_COMMIT"
```

这里的 Cargo commit 必须从 pinned Cargo 自己的 `-Vv` 单独提取并精确匹配，不能拿 rustc commit 代替。`Z3_SYS_Z3_HEADER` 供冻结的 `z3-sys 0.8.1` 选择 Homebrew 头文件；`RUSTFLAGS=-Lnative=...` 供链接器找到 Z3，`DYLD_LIBRARY_PATH` 只解决运行时动态库加载，三者不能互相替代。`RUST_SYSROOT` 必须在编译 RAP 时存在，因为本分支会把 sysroot 编入 binary；直接调用真实 toolchain Cargo 绕过了 rustup proxy，不能依赖 proxy 自动提供该值。为保持 exact baseline，不继承用户已有的额外 `RUSTFLAGS`。

随后记录并检查实际架构：

```bash
brew --prefix
brew list --versions llvm z3 cmake pkg-config
file "$RAPV2_RUSTC"
file "$RAPV2_CARGO"
file "$RAPV2_LLVM_PREFIX/bin/clang"
file "$RAPV2_Z3_PREFIX/lib/"*z3*.dylib
pkg-config --modversion z3
cmake --version
```

`file` 对 glob 可能输出多行，这是正常的；关键是 Rust/LLVM/Z3 不得是互不兼容的架构。

### 2.3 基线可构建门

在修改源码前，用 fresh target 和单 Cargo job 编译现有基线：

```bash
set -euo pipefail
RAPV2_BASELINE_TARGET="$(mktemp -d "${TMPDIR:-/tmp}/rapx-v1-baseline-target.XXXXXX")"
UNSOUND_SCANNER_RAP_BUILD_RUSTC_COMMIT="$RAPV2_RUSTC_COMMIT" \
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$RAPV2_BASELINE_TARGET" \
  "$RAPV2_CARGO" build \
  --manifest-path rapx/Cargo.toml --locked --bins --jobs 1
test -x "$RAPV2_BASELINE_TARGET/debug/cargo-rapx"
test -x "$RAPV2_BASELINE_TARGET/debug/rapx"
```

这一步的目标只是证明 Mac 环境能构建冻结的 Linux 源码，不是证明两个平台产出相同二进制。保留命令、退出码、`rustc -Vv`、源码 hash 和错误摘要，生成 preflight receipt。

若在依赖齐全且架构一致后仍失败：

1. 记录 `blocked_mac_exact_baseline` receipt；
2. 保留完整编译错误和环境信息；
3. 停止实现，不换 nightly、不升级 crate、不删 lockfile、不改 RAP 源码绕过问题；
4. 把 receipt 推回分支或交给用户，等待 Linux campaign 完全停止后转到 Linux 隔离验证。

## 3. v2 规则的冻结含义

`-unsoundaudit` 在此分支上直接升级为 v2，不保留第二套 v1 CLI。输出 schema 为 `rap-unit-v2`，主计数必须恰好为 `pattern1` 到 `pattern6`。Linux 现役 v1 二进制、旧 ledger 和四类 receipt 不重解释。

### P1：普通函数参数进入 unsafe sink

合并旧 P1 与旧 P3。public safe entry 的普通参数及仍 data-dependent 于它的派生值，经过 0..N 个 crate 内、静态可解析调用后，未经能支配 sink 的有效检查进入 unsafe operation。过程内/过程间只是 `propagation_depth` 属性，不再是两个 Pattern。普通参数即使经过 `index + 1` 等算术仍默认归 P1；不得仅因发生内部算术而改归 P4。

### P2：literal public field 污染 unsafe sink

严格限于 safe caller 能直接写入的 literal `pub` field，随后该字段值进入 unsafe sink。通过 safe setter 修改 private backing state、私有缓存或普通局部变量不属于本轮 P2；它们只能作为未覆盖 family 记录，不能为了增加数量扩大 P2。

### P3：内部 unsafe 构造并暴露非法值或状态

对应旧 P4 的严格版本：合法入口后，项目内部 unsafe-backed origin 位于 public safe API 的 return/store/exposure causal slice 上，并构造、保留或暴露非法运行时 value/state。Rust validity UB 可能在 invalid typed value 被构造时已经发生；return flow 用来证明 safe-only 可达 causal slice，不得声称 UB 一定等到返回后才发生。只使用冻结的有限 origin registry；同函数出现危险 API 或危险返回类型但不在同一 bug path 上，不算 P3。

### P4：内部派生内存操作数违反 unsafe 合同

source 不是调用者直接传入，而是由 private/internal state 在项目内部计算得到的 index、length、offset 或 access width；该派生量在进入 unsafe sink 前没有被证明满足对应 bounds/pointer-offset obligation。只有 private/internal state 是唯一 root，或外部值原有义务已成立、项目 transform 随后重新破坏该义务时才归 P4；`self` 仅作为 private state 容器不算 P1。

第一版只做两类：

1. bounds：如内部 index/access width 对目标长度无有效上界；
2. pointer offset：内部 offset 进入 pointer arithmetic/read/write 且无同源范围约束。

alignment/provenance 只有存在明确、可绑定的静态 facts 时才做；不得用 API 名猜测后宣称覆盖。

### P5.1：泛型/关联 slice 的非空义务不足

只实现有限子规则：公开泛型 API 对泛型或关联 slice 做固定位置（典型为 0）unchecked access，但公开 bounds 或支配性检查不能保证非空。

这不是通用的 `B(T) => R(T)` 证明器，不覆盖任意布局相等、padding、validity、provenance 或复杂 const-generic 关系。输出中必须标记 `family_support = partial`。

### P6：外部行为/宿主输出未经验证进入 unsafe sink

实现两条可区分子规则：

1. FFI：`extern` 返回值或 out-parameter 写入结果未经支配性验证，进入 enum/NonNull/长度/索引/transmute/unreachable 等 unsafe-sensitive sink；
2. Rust open behavior：trait、`Iterator`、callback 或动态 dispatch 的运行时返回结果未经验证进入上述 sink。

无法解析的普通外部调用不能一律报警；只有其返回/写出事实属于冻结的 P6 source 形态，并且存在同一路径 unsafe sink，才生成候选。

## 4. 统一过程间分析的实现合同

不要继续给旧的字符串 carrier 打补丁。先建立统一 MIR summary IR，再让 P1–P6 消费它。建议最小类型包括：

- `SourceKind`：parameter、public state、internal derived、unsafe origin、FFI output、open-behavior output；
- `SinkKind`：unchecked index、pointer offset/read/write/copy、slice construction、transmute、invalid-value construction、unreachable 等有限 registry；
- `Predicate`：non-empty、bounds、offset range、nonnull、valid discriminant 等有限义务；
- `FunctionSummary`：formal inputs、field reads、external outputs、derived values、validations、unsafe sinks、return/out flows 和 side effects；
- `Finding`：entry、source、传播边、sink、未满足义务、primary pattern、secondary source、rule id、证据 span、限制说明。

过程间范围固定为：

- 同一 local crate；
- 静态可解析调用，优先使用 rustc 的 `Instance::try_resolve` 或等价精确解析；
- actual→formal、callee return→caller destination，以及确有事实时的 out-parameter 映射；
- call graph 上按 SCC 做单调、有限高度的 fixed point；递归不能靠固定“10 轮”或静默截断；summary equality 不含无限展开的 call path/depth，使用稳定 SCC-cycle token、确定性最短 witness 和饱和的 `recursive` depth，facts 来自有限 registry；若加入会增长的抽象必须显式 widening 并测试；
- dyn/external/unresolved call 不做臆测式跨体内联，只有符合 P6 source 定义时才产生 external-output fact；
- 不做跨 crate whole-program analysis。

检查/净化事实必须满足：

- 与同一个 place/origin 及同一个 predicate 绑定；
- 在 CFG 上支配相关 sink；
- 检查后若相关 place、backing state 或别名可能被写入，事实失效；
- 仅凭函数名含 `check`、`validate` 或类似关键词，不得消除 finding。

Finding 分类与去重：

- 每条 source→sink bug path 只有一个互斥 primary pattern；
- 可保留 secondary source/mechanism，但不计入另一主 Pattern；
- 同一 entry/source/path/sink/obligation 不得因多个 fixed-point 轮次或多个表示重复输出；
- 输出顺序必须确定化，不能依赖 hash-map 迭代、临时目录、绝对路径或 wall-clock 时间。

## 5. TDD 与 27 个微型 fixtures

先写 oracle 和 RED tests，再写实现。建议新路径：

```text
tests/unsoundaudit-v2/
  fixtures/<fixture-name>/Cargo.toml
  fixtures/<fixture-name>/src/lib.rs
  expected/<fixture-name>.json
scripts/unsoundaudit-v2/
  run_fixture_suite.py
  normalize_receipts.py
```

每个 fixture 必须：

- 是无第三方依赖的 library crate；
- 没有 `build.rs`、proc macro、网络、测试运行时或环境依赖；
- vulnerable/fixed 的无关代码尽量逐字节相同；
- 把危险调用放在不会由验证命令执行的函数中；
- 注明它验证的必要事实和不验证的更广结论。

### 5.1 迁移现有 10 个 fixture

从 Linux 旧测试素材复制语义，不要改 Linux 原文件：

| 旧 fixture | v2 预期 |
|---|---|
| `pattern1_positive` | 新 P1 恰好 1 条 |
| `pattern1_negative_local_pointer` | 0 条 |
| `pattern2_positive` | 新 P2 恰好 1 条 |
| `pattern2_negative_private_field` | 0 条 |
| `pattern2_display_negative` | 0 条 |
| `pattern2_write_negative` | 0 条 |
| `pattern3_positive` | 新 P1 恰好 1 条，`propagation_depth=inter_procedural` |
| `pattern3_negative_private_chain` | 0 条 |
| `pattern4_positive` | 新 P3 恰好 1 条 |
| `pattern4_negative_checked_utf8` | 0 条 |

Linux 原始位置是 `/home/lwz/unsound_scanner/tests/fixtures/unsoundaudit/`，但 Mac 不应读取或依赖这个 Linux 绝对路径。交接分支已经在 `docs/unsoundaudit-v2/frozen-v1-fixtures/` 携带逐字副本和 `legacy_fixture_manifest.json`；迁移前必须重算清单中的 31 个 SHA-256，再从该冻结目录复制到正式 v2 fixture root。冻结目录里的原始 `README.md` 也只是 provenance bytes，其中 v1 checker/receipt 命令不适用于 v2，不得执行。若冻结目录、清单或任一哈希不匹配，停止并请 Linux 线程修复交接，不要凭名字重新编造“等价”案例。

### 5.2 新增 8 对 vulnerable/fixed fixture，共 16 个

每对分别验证：

1. P1：参数经过两层 local helper 到 sink；fixed 版有同源、支配 sink 的检查；
2. P2：literal `pub` field 经 helper 到 sink；fixed 版保留同一 public field 与 sink，只在 sink 前加入同值支配性验证；
3. P3：helper 内 unsafe origin 通过 return flow 暴露；fixed 版返回前恢复 validity；
4. P4-bounds：Bumpalo 风格 private backing state 派生 index/width，safe method 没有 scalar/slice public source；fixed 版保持同一 private source、slice、helper 与 sink，只增加支配性的同源 bounds 证明；
5. P4-offset：private backing state 内部派生 pointer offset 且无范围保证；fixed 版保持同源与 sink并建立有效 offset 范围；
6. P5.1-nonempty：泛型/关联 slice 上固定位置 unchecked access；fixed 版有支配性的 `is_empty`/length guard；
7. P6-FFI：最小 `extern "C"` 声明的返回值或 out value 进入 unsafe-sensitive sink；fixed 版验证值域/空指针/长度；只需 `cargo check`，不链接或调用外部符号；
8. P6-trait：open trait/Iterator/callback 结果进入 sink；fixed 版在 sink 前验证返回结果。

### 5.3 1 个全零 noise fixture

构造同时包含泛型、trait、FFI 声明、helper、public field、普通 slice 和 safe indexing 的 library，但不形成任何 source→unsafe-sink 违约路径。预期 P1–P6 全为 0，用于防止“看见关键词就报警”。

总数必须严格为：

```text
10 migrated + 8 vulnerable/fixed pairs (16) + 1 noise = 27 fixtures
```

### 5.4 每个 fixture 的 oracle

- 每个 positive：恰好一个预期 primary finding，其他五类为 0；
- 每个 fixed/negative/noise：六类均为 0；
- 不允许同一 source→sink 重复；
- P1 的 two-hop 和迁移的旧 P3 positive 必须明确显示过程间传播；
- P5 finding 必须显示 `P5.1` 和 partial 限制；
- P6 finding 必须区分 `ffi_output` 与 `open_behavior_output`；
- receipt 中的 span 使用仓库相对路径和稳定行列，不写临时目录绝对路径。

## 6. 推荐实现顺序与阶段门

严格按以下顺序推进；每阶段只提交与该阶段直接相关的代码和测试。

1. **测试合同与 schema**：落盘 27 fixture 清单、expected oracle、`rap-unit-v2` validator；此时 fixture scan 应为 RED 或明确 unsupported。
2. **核心 IR**：加入 source/sink/predicate/summary/finding 类型及纯 Rust unit tests。
3. **单函数事实提取**：从 MIR 产生 input、field、derived、validation、sink、return/out facts。
4. **调用解析与映射**：实现 local static call resolution、actual/formal/return/out 映射。
5. **SCC/fixed point**：递归 call graph 上收敛；测试顺序独立、递归、mutual recursion 和 stable dedup。
6. **验证事实**：实现 dominance、place/origin/predicate 绑定和 write invalidation；不要保留旧名字关键词 sanitizer 作为正确性依据。
7. **P1–P3 迁移**：先让 10 个旧 fixture 在新 summary 层上满足 v2 oracle。
8. **P4**：分别实现 bounds 与 pointer-offset 两个 rule id。
9. **P5.1**：只实现 non-empty obligation；在代码、schema 和文档中固定 partial scope。
10. **P6**：分别实现 FFI output 与 open Rust behavior output。
11. **分类、去重、稳定输出**：primary pattern 互斥，输出 `rap-unit-v2`，计数恰好 P1–P6。
12. **全套串行验证与独立复核**：27/27、unit tests、两次 normalized receipts byte-identical。

阶段 1 还必须为迁移后的 27 个 v2 crate 各生成并冻结 `Cargo.lock`。冻结输入目录故意没有携带 Ubuntu lockfile；把源码复制到正式 v2 fixture root 后，逐 crate 使用 pinned `$RAPV2_CARGO generate-lockfile --manifest-path <fixture>/Cargo.toml`，再用 `.gitignore` 精确 exception 或 `git add -f` 跟踪这些 lockfile。验收运行前 27/27 都必须已有版本控制中的 lockfile，之后才使用 `--locked`；不得修改 `docs/unsoundaudit-v2/frozen-v1-fixtures/` 的 provenance bytes。

若某条规则只能观察必要条件，保留候选生成器语义和明确 `limitations`；不要为了达到 fixture 数而把启发式输出写成 soundness detector。

## 7. 运行方式与资源限制

每次 RAP 自身构建或测试都使用全新的 target：

```bash
set -euo pipefail
RAPV2_BUILD_TARGET="$(mktemp -d "${TMPDIR:-/tmp}/rapx-v2-build-target.XXXXXX")"
UNSOUND_SCANNER_RAP_BUILD_RUSTC_COMMIT="$RAPV2_RUSTC_COMMIT" \
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$RAPV2_BUILD_TARGET" \
  "$RAPV2_CARGO" build \
  --manifest-path rapx/Cargo.toml --locked --bins --jobs 1
UNSOUND_SCANNER_RAP_BUILD_RUSTC_COMMIT="$RAPV2_RUSTC_COMMIT" \
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$RAPV2_BUILD_TARGET" \
  "$RAPV2_CARGO" test \
  --manifest-path rapx/Cargo.toml --locked --lib --jobs 1
RAPV2_BIN_DIR="$RAPV2_BUILD_TARGET/debug"
test -x "$RAPV2_BIN_DIR/cargo-rapx"
test -x "$RAPV2_BIN_DIR/rapx"
```

这里只运行 RAP 自身的纯 unit tests；不得把会触发 UB 的 fixture 函数写成并执行 `#[test]`。

fixture runner 必须逐个串行执行，且对每个 fixture：

1. 为 control check 和 RAP scan 分别建一个 fresh target，并为本 case 建一个 fresh receipt directory；
2. 在 fixture 目录运行 `CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$RAPV2_CONTROL_TARGET" "$RAPV2_CARGO" check --lib --locked --jobs 1`，证明 fixture 可编译；
3. 设置 `UNSOUND_SCANNER_RAP_JSON_DIR` 为该 case 的绝对 receipt directory、`UNSOUND_SCANNER_PROJECT_ROOT` 为 fixture 的 canonical absolute root，并保留第 2.2 节的 exact Cargo/rustc 环境；随后在 fixture 目录运行 `CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$RAPV2_SCAN_TARGET" "$RAPV2_BIN_DIR/cargo-rapx" rapx -unsoundaudit -- --lib --locked --jobs 1`。直接执行 `cargo-rapx -unsoundaudit` 是错误调用，因为当前 dispatcher 要求 argv[1] 为字面量 `rapx`；
4. 禁止 `cargo run`、fixture `cargo test`、Miri、benchmark 或真实项目扫描；
5. 解析 v2 receipt 并立即核对 oracle，失败即停止该轮且保留证据。

Runner 中每个 case 的调用应等价于下面的结构；变量都必须是 canonical absolute path：

```bash
set -euo pipefail
RAPV2_CONTROL_TARGET="$(mktemp -d "${TMPDIR:-/tmp}/rapx-v2-control-target.XXXXXX")"
RAPV2_SCAN_TARGET="$(mktemp -d "${TMPDIR:-/tmp}/rapx-v2-scan-target.XXXXXX")"
RAPV2_FIXTURE_ROOT="$(cd "$RAPV2_FIXTURE_DIR" && pwd -P)"
mkdir -p "$RAPV2_RECEIPT_DIR"
RAPV2_RECEIPT_ROOT="$(cd "$RAPV2_RECEIPT_DIR" && pwd -P)"

(
  cd "$RAPV2_FIXTURE_ROOT"
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$RAPV2_CONTROL_TARGET" \
    "$RAPV2_CARGO" check --lib --locked --jobs 1
  UNSOUND_SCANNER_RAP_JSON_DIR="$RAPV2_RECEIPT_ROOT" \
    UNSOUND_SCANNER_PROJECT_ROOT="$RAPV2_FIXTURE_ROOT" \
    CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR="$RAPV2_SCAN_TARGET" \
    "$RAPV2_BIN_DIR/cargo-rapx" rapx -unsoundaudit -- \
    --lib --locked --jobs 1
)
```

上例依赖第 2.2 节已经 export 的 exact Cargo/rustc 环境。Runner 必须先检查这些变量非空、两个 bin 为同一 build target 下的 sibling、fixture lockfile 存在且受 Git 跟踪、receipt directory 在调用前为空。

不要并行启动多个 Cargo。runner 本身要 fail closed：发现 schema 不对、缺 receipt、重复 finding、unexpected pattern、fixture 数不是 27，均返回非零。

## 8. Receipt、确定性与提交规范

建议把小型证据放在：

```text
artifacts/unsoundaudit-v2/mac/<run-id>/
```

至少包含：

- `environment_receipt.json`：macOS、arch、Xcode、Homebrew LLVM/Z3/CMake/pkg-config、nightly 和 rustc commit；
- `baseline_receipt.json`：tag/commit、修改前两个 source SHA、baseline build 命令和退出码；
- `fixture_manifest.json`：27 个 fixture 名称、类型、源码 SHA、预期 primary pattern；
- `fixture_results.normalized.json`：实际的结构化候选结果；
- `unit_test_summary.json`；
- `sha256_manifest.json`；
- 若失败，提交 `blocked_receipt.json`、脱敏错误摘要和 Mac 本地原始日志的 SHA-256；原始日志路径只记录在不提交的本地 custody note 中。

normalized receipt 不得包含 wall-clock 时间、随机 run id、绝对用户路径、target 路径、进程 ID、HashMap 非确定顺序或秘密。环境 receipt 可以另存非决定性元数据，但不得参与 byte-identical 比较。

原始完整构建/扫描日志只保存在 Mac 本地且不提交；提交前生成脱敏摘要，删除用户名、绝对 home/target/cache/tool 路径、环境变量值中的凭据及任何 token。`environment_receipt.json` 的提交副本同样不得保留绝对工具路径，只保留工具 identity、版本、commit、架构和路径角色。Git 中只保存研究结论所需的小型 JSON/text receipt 与 SHA，不保存指向另一台机器私有路径却不可复核的“完整日志路径”。若 `.cargo/config*` 或环境中存在 target-specific `rustflags`，也应视为 exact-environment 污染并停止，不能与本文的 Z3 单一 link flag 混用。

最终确定性门：从相同源码和相同 oracle 使用两个 fresh target 串行运行两次，两个 `fixture_results.normalized.json` 必须 byte-identical。还要重算所有源码/receipt SHA。

Git 规范：

- 修改文件使用可审查的小步提交；不夹带格式化整个仓库或无关重构；
- 不提交 `target/`、二进制、dylib、Cargo registry/cache、编辑器文件、token 或绝对 Mac 用户路径；
- 不 force-push；推送前执行 `git fetch origin feature/unsoundaudit-v2-p1-p6`，并要求 `git merge-base --is-ancestor origin/feature/unsoundaudit-v2-p1-p6 HEAD` 成功。失败表示远端有本地未包含的提交，必须停止协调，不能覆盖或静默 rebase；
- 每次 push 都保持分支可构建；若多人同时工作，先停下来协调，不用 rebase/覆盖他人提交静默解决；
- 最终 push 到 `origin/feature/unsoundaudit-v2-p1-p6`，不要合并到 `main`；
- 最终总结明确区分：已实现规则、fixture 结构验证、未做真实项目验证、已知漏检/误报边界。

推荐提交序列：

```text
test(unsoundaudit-v2): freeze 27 fixture oracles
refactor(unsoundaudit): add MIR function summary engine
feat(unsoundaudit): migrate P1-P3 to interprocedural summaries
feat(unsoundaudit): add P4 derived-operand candidates
feat(unsoundaudit): add bounded P5.1 candidates
feat(unsoundaudit): add P6 external-output candidates
test(unsoundaudit-v2): add deterministic fixture receipts
docs(unsoundaudit-v2): record Mac validation and limitations
```

实际提交可合并，但不要把无关改动混进去。

## 9. Mac 完成后仍需 Linux 最终门

Mac 通过只说明：当前源代码能在该 Mac 的精确 nightly 上构建，27 个构造 oracle 与静态输出一致。它不是论文实验结果。

待 Linux 现役 campaign 完全停止后：

1. 在 Linux 独立 checkout 拉取同一分支 commit；
2. 复核 commit、源码树和 `unsoundaudit.rs` 的新 SHA；
3. 使用 Linux exact `nightly-2024-10-12`、fresh Cargo home/target、jobs=1 构建；
4. 重跑 27 fixtures 和确定性检查；
5. 完成独立源码审查；
6. 只有这些通过后，才讨论极小规模真实项目 pilot；真实项目 pilot 必须另行授权和设计。

旧 Linux v1 基线应保留为 tag 和只读归档。即使未来把 v2 checkout 放回原 `RAPx/` 路径，也不得删除或重解释旧 v1 evidence。

## 10. 可直接复制给 Mac 新 Codex 线程的启动 prompt

```text
请在这台 Mac 上实现 RAPx 的 unsoundaudit v2。仓库是
git@github.com:lwz23/RAPx.git，工作分支固定为
feature/unsoundaudit-v2-p1-p6。

开始前请完整阅读：
docs/unsoundaudit-v2/MAC_HANDOFF.md
docs/unsoundaudit-v2/design.md
docs/unsoundaudit-v2/fixture_matrix.md
docs/unsoundaudit-v2/baseline_manifest_v1.json

必须先按 MAC_HANDOFF.md 验证 baseline tag/commit、rapx 源码树 hash、
unsoundaudit.rs hash、nightly-2024-10-12 及 rustc commit，并完成
Xcode/LLVM/libclang/Z3/CMake/pkg-config 和 CPU 架构预检。基线在依赖齐全后
仍不能构建时，请生成 blocked_mac_exact_baseline receipt 并停止；不要换 nightly、
升级依赖、删改 lockfile 或放宽规则范围。

实现范围固定：
1. 用 crate 内、静态可解析调用、SCC/fixed-point 的 MIR FunctionSummary 层统一
   P1-P6；不做跨 crate whole-program analysis。
2. 新 P1 合并旧 P1/P3；P2 严格限于 safe caller 可直接写入的 literal public field；P3 是旧 P4 的严格内部非法
   value/state 暴露；P4 先做 internal-derived bounds 和 pointer-offset；P5 只做
   P5.1 泛型/关联 slice 非空义务；P6 分 FFI output 和 trait/Iterator/callback
   output。
3. `-unsoundaudit` 在本分支直接输出 `rap-unit-v2`，primary pattern 恰好 P1-P6，
   互斥分类并稳定去重。所有新增规则只能称 heuristic candidate generator。
4. 先冻结 27 个无依赖 library fixture oracle，再 TDD 实现：迁移旧 10 个，新增
   8 对 vulnerable/fixed，另加 1 个全零 noise。每个 positive 恰好一个 primary，
   negatives/noise 全零；两次 normalized receipt byte-identical。

资源与安全边界：项目并发=1、Cargo jobs=1、fresh target；fixture 只 cargo check
和静态扫描，不运行 UB 路径、不跑 Miri/benchmark/真实项目。不触碰任何 Linux RAP
campaign、coverage ledger 或 2,848 历史成功基线。不要提交或回传 Mac 二进制、
target 或缓存，只提交源码、小型 JSON/text receipts 和审查摘要。

请先做只读 orientation，并用中文给我报告：基线验证结果、Mac preflight 结果、
拟修改的最小文件集合、27 fixture 清单与第一个 RED test。确认没有出现本文定义之外
的实质 scope 变化后，按文档阶段门持续实现、验证、提交并推送到同一 feature 分支。
有会改变 Pattern 语义、主计数、过程间范围或验收条件的不确定之处，停下来问我，
不要自行决定。
```
