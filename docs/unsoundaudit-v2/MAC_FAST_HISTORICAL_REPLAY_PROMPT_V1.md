# 可直接交给 Mac 线程的 Prompt：快速历史源码回放 v1

请立即执行一次快速历史源码回放。目的不是开发或验证简化 fixture，也不是
估计误报率、precision、recall 或生态 coverage；目的是诊断：新 P1--P6 RAP
对十个 unseen 真实项目得到 0 finding，究竟是规则在完整项目中触发面很窄，
还是对已确认历史漏洞也存在明显漏检。

## 固定边界

1. 使用已经成功扫描十个项目的同一套 Mac legacy-2024 RAP 环境、同一 RAP
   binary/profile 和同一单项目扫描命令；不要切到 nightly-2026，不要重建或修改 RAP。
2. 不运行项目 test、example、binary、Miri 或 PoC；只运行已有扫描所需的
   control check 与 RAP 静态扫描。
3. 不修改项目源码、Cargo.toml、Cargo.lock 或依赖版本；有 Cargo.lock 就沿用
   --locked。无 lock 或 control build 失败，只记录并继续下一项，不要排障。
4. 不要把此结果写成“实际检出率”或论文结论；它只是完整历史源码上的快速诊断。
5. 每个案例 clone 一次、checkout 一次、扫描一次。不要因为零 finding 改 revision、
   改规则或重新挑项目。

## 使用的十个案例

| # | 预期模式 | repo / Issue | checkout |
|---|---|---|---|
| 1 | P1 | servo/rust-smallvec #343 | 0f3aacb99ccfe0cbbf0bb83ecf736f67339ce8ae^ |
| 2 | P1 | GitoxideLabs/gitoxide #1460 | a807dd1ffb05efd177700d065095249e6c4b3c68 |
| 3 | P2 | rust-lang/libc #3560 | 6faa521f32fc11db9fc43a248a64463ce288b48d^ |
| 4 | P3 | jeromefroe/lru-rs #73 | a96f71a84d108f9b7506908b8e1c5da9524bc4da^ |
| 5 | P3 | hyperium/http #639 | bbe2a8f34f0f0aadcd124f5f2841bdae454bc837^ |
| 6 | P4 | fitzgen/bumpalo #164 | 71d99b89835a36a497e1f3ff54b625de5012e3ed^ |
| 7 | P4 | rust-random/rand #779 internal-derived | 9828cdf0201f816e4e4bff07834d849a10be80e9 |
| 8 | P5.1 | rust-random/rand #1158 | 90b89cdbb2655ac278cee43b5c343102a89616b1^ |
| 9 | P6 | nix-rust/nix #1819 | 8e91b28b64bdd18fc6fa61af8e89947ad7fb97bf^ |
| 10 | P6 | diesel-rs/diesel #4223 | 49aa163f1a9c793da6d591e10481fbfb6c044b51^ |

带 ^ 的 revision 是快速测试使用的 pre-fix 候选 parent；如果 Git 无法解析，
记录 source_unavailable 并继续，不要切到 main。Rand #779 和 Gitoxide #1460
使用冻结审计中的直接 source revision。

## 每项的最小动作

1. clone 对应仓库（不要递归 submodule），checkout 表中的 detached revision；
2. 使用你刚完成的 real-project pilot 的相同 control scan；记录 control 成功/失败；
3. 仅在 control 成功时，用同一 RAP P1--P6 命令扫描同一 Cargo target；
4. 保存 RAP 原始 receipt 和 stdout/stderr 的本地路径；
5. 不做人为源码解释，不改规则，不增加额外验证。

若一个仓库是 workspace，直接复用你刚才 pilot 的 deterministic target-selection
规则；不要为追逐某个 Issue 临时换 package/feature。

## 最终只需返回一张表

case | expected_pattern | repo_issue | detached_revision | control | rap | p1 | p2 | p3 | p4 | p5 | p6 | result_dir

其中：control 为 ok / fail；rap 为 ok / fail / skipped_control_fail；正常完成且
p1--p6 全为 0 时写 no_finding；任何非零 finding 记录其 primary pattern、source
span 和 receipt 路径。先不要判断 finding 是否误报，也不要把 0 finding 写成“0 误报”。

十项结束后停止，把表、各项 receipt 路径、以及使用的 RAP commit/profile/hash 发回。
