# Codex handoff: lwz23 148/125 召回（CodeQL 优先）

给 **cwd = `/home/lwz/unsound_scanner`** 的 Codex。不要依赖 Mac 聊天。
**不要开 crates.io 5000+。**

2026-09-07 主人指示：核心目标 + 关键资产 + 两条实现思路。
**先看 CodeQL 能不能把召回率提上来**；RAP 是并列引擎，不是必须先编完 RAP 才能扫。

更长的 RAP 细节（脏树、nightly、v2/v3）见同目录
`REMOTE_LWZ23_RECALL_HANDOFF.md`。本文件覆盖执行优先级。

---

## 1. 核心目标

冻结宇宙：**148 条有效 issue / 125 个扫描仓**（作者 lwz23）。
Pattern = **新 P1–P6**（v2 语义，不是旧论文 P1–P4）。

一条 **true_hit** 必须满足：

- 分析器（**自定义 CodeQL P1–P6** 或 **RAP UnsoundAudit v3**）产出 finding；
- finding 的 public root / callee / span **对上 issue 正文**里的函数名、`src/….rs`、代码围栏，而不是只对标题；
- 同一 crate 里一条无关 finding **不算**命中。

Mac 旧账本（**只扫 RAP、只对标题**，不是远程目标）：

`true_hit 32 / no_artifact 76 / miss 26 / skip_host 11 / gated 3` / 148。

远程要做的事：把 `no_artifact` 和 `miss` 压下去，用正文重新诚实打分。
32 里一部分在正文规则下会掉，预期之内。

成功：大多数 148 条有正文对齐的 finding，或剩下的每条都有 **failure class**
（windows-pe、browser-monorepo、git 404、query miss、…），不是 per-repo hack。

不是成功：95% 演戏；「编过了」但没对上上报 API；标题碰运气。

---

## 2. 关键资产（已在本机，不要重拷）

| 东西 | 路径 |
|---|---|
| 分析器脏树（含 v3 + CodeQL PoC） | `RAPx-unsoundaudit-v2/` |
| 148 issue 表 | `RAPx-unsoundaudit-v2/docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/issues.csv`（149 行=头+148） |
| 125 仓表 | 同上 `repositories.csv`（126 行） |
| clone 计划 / 排除 | `clone_plan.json` `exclusions.json` |
| 义务语料 | `…/lwz23-valid-issue-roster-inputs/bug_instances.jsonl` |
| **148 条 GitHub 正文** | `…/lwz23-valid-issue-roster-v1/issue_bodies.jsonl` |
| 正文 needle | `…/issue_needles.json` |
| 125 仓源码 | `lwz23-valid-issue-src/`（~4.0G；`checked_out_revisions.json` 124 条） |
| CodeQL P1–P6 | `RAPx-unsoundaudit-v2/codeql/unsoundaudit-poc/` |
| 正文 scorer | `RAPx-unsoundaudit-v2/scripts/unsoundaudit-v3/rescore_lwz23_coverage.py` |
| CodeQL 普查脚本 | `scripts/unsoundaudit-v3/census_codeql_no_artifact.py` `run_codeql_fixture_poc.py` |
| SSD 工作根 | `lwz23-recall-work/`（cargo/target/db **只写这里**，不要 `/media/lwz/RAP_WORK`，99% 满） |

P1–P6 含义：`RAPx-unsoundaudit-v2/docs/unsoundaudit-v2/design.md`

- P1 公参 → unsafe sink（可经局部 helper）
- P2 可写 `pub` 字段 → sink
- P3 合法入口后内部造出非法值（`assume_init` / transmute / unchecked UTF-8…）
- P4 内部算出的 index/len/offset 破坏内存前置
- P5 泛型/关联类型不保证非空，sink 用元素 0
- P6 FFI 返回/out-param 或开放 trait/callback 结果未经检查进 sink

6 条 `rustsec/advisory-db` issue **留在 148 分母**；扫映射软件仓
（Corundum / Pleco / perpetual / ninjabook / swift-bridge / …），
**不要 clone advisory-db**。

`BrowserWorks/waterfox` 在 125 里。编不动就 `skip_host`，class =
`browser-monorepo / not a practical Cargo crate scan`。不要从 148 里默删。

Ubuntu `gh` 已登录 `lwz23`。CodeQL **CLI 还没装**（QL 包在树上）。

工具链（2026-09-07 已就绪，不必重装）：

- `nightly-2026-08-14`：rustc `ba28ff76` + **rustc-dev / rust-src / llvm-tools / rustfmt**
- `nightly-2024-10-12`：rustc `1bc403da` + rustc-dev
- RAP `rapx-perf` **2024 二进制已有**：`lwz23-recall-work/rapx-build/bin-nightly-2024-10-12/{cargo-rapx,rapx}`
- 2026 `rapx-perf` 还没编。 **先做 CodeQL 召回，不必先编 2026 RAP。**

---

## 3. 两条路（先 CodeQL）

发现引擎必须是 **数据流 / 编译器 unsafe 点**，不能是
「callee 是否属于这 10 个字符串」当唯一发现方式。
合同目录（`get_unchecked`、`assume_init`、raw deref、`from_utf8_unchecked`…）
可以在 **已经找到 unsafe 点之后** 用来分类或放电。

### A. CodeQL（先做，用来抬召回）

自定义 QL，不是库存 OWASP。

Mac 已验证：

- 27 fixture：正例 8/12，负例 10/14。P1/P2/P3 能打中；P4、P5、P6.FFI 弱。
- 6 个 RAP `no_artifact` 仓：Corundum / sqlite-loadable-rs / enigma / Flecs-Rust
  在 **SSD** 上 ~200s 抽出；blitz / pagefind 300s 不够。
- 24 仓普查在 Mac **失败**（300s 砍在 `Cargo.toml` 之后 + 磁盘 99%），
  **不能当成 CodeQL 没能力**。Linux SSD 应加长抽取墙钟（建议 ≥900s）。

建议顺序：

1. 把 CodeQL CLI 装到 SSD（例如 `lwz23-recall-work/codeql/`），不要拷 Mac USB 上的 CLI。
2. Fixture 冒烟：`python3 RAPx-unsoundaudit-v2/scripts/unsoundaudit-v3/run_codeql_fixture_poc.py`
3. 对 **RAP 编不过的仓**（Mac `no_artifact` 那批，约 76 issue / 对应仓）做 extract + P1–P6。
   `CARGO_HOME` / 数据库 / 日志全部在 `lwz23-recall-work/`。
4. 用 `issue_bodies.jsonl` + `rescore_lwz23_coverage.py` 打 148 行 ledger。
   命中 = finding 的 root/callee/path 对上正文 needle。
5. 已知假命中不要算：starlight；JVM `find_symbol`；retrowin32 Arg。
6. P4/P5/P6 弱就按 **failure class** 加强 QL（仍走 taint/HIR，不要 per-issue 特判）。

抽取超时、缺系统 C 库、Windows PE、json-target 裸机 → `skip_host` + class。
不要为了抽库去改每个仓的源码。

### B. RAP UnsoundAudit v3（第二条，补 CodeQL 打不中的）

树在 `RAPx-unsoundaudit-v2/`（**不要 `git clone origin`**，origin 落后 83 commit，
脏 v3 会丢）。编 `rapx-perf`：

```text
bash RAPx-unsoundaudit-v2/scripts/unsoundaudit-v3/build_rapx_perf_linux.sh nightly-2026-08-14
# 旧仓再用 nightly-2024-10-12
```

crates.io 直连 TLS 常挂，用中科大 sparse
（`lwz23-recall-work/cargo-home/config.toml` 已配 `ustc`）。

扫描：`scripts/unsoundaudit-v3/scan_lwz23_valid_issue_roster.py`
（每仓 wall timeout，Linux 已走 `LD_LIBRARY_PATH` + 系统 z3/libclang）。
27 fixture 期望聚合 `P1=3 P2=2 P3=2 P4=2 P5=1 P6=2`。

RAP 适合：能 `cargo check` 的仓、要 rustc unsafe 普查的点。
编不过的仓交给 CodeQL 或诚实 `skip_host`，不要 AST/syn Pattern1 顶替。

---

## 4. 约束

- 禁止 syn/AST Pattern1 当上报工具。
- 禁止 per-repo allowlist / issue-id 特判。
- 禁止只靠写死 sink 名单当发现引擎。
- 禁止改上游 `safer-rust/RAPx`。
- 禁止重开 Mac `msrv_retry.txt` leftover 58。
- 禁止 crates.io 5000+。
- 假阳性可以升（heuristic P1 允许）。
- 通用恢复（对所有仓）：MSRV `--precise`、`--no-default-features`、
  `feature(freeze)`、unlocked lockfile、只重试失败 package。

---

## 5. 建议的第一批命令

```text
cd /home/lwz/unsound_scanner
wc -l RAPx-unsoundaudit-v2/docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/issues.csv
# 149
wc -l RAPx-unsoundaudit-v2/docs/unsoundaudit-v3/lwz23-valid-issue-roster-v1/issue_bodies.jsonl
# 148

# 装 CodeQL CLI 到 SSD，然后 fixture 冒烟，再扫 RAP no_artifact 仓
# 输出：lwz23-recall-work/codeql-recall/coverage_rescored.csv
# 打完 148 行 ledger 再决定要不要动 RAP / 加强 QL
```

日志、PID、journal、超时写进 `lwz23-recall-work/`。
退出后读 ledger，按 failure class 迭代，不要等主人问「跑完了吗」。
