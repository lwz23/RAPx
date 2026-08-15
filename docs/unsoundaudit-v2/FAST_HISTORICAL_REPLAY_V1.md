# 快速历史回放 v1：10 个完整项目源码案例

目的：快速判断新 P1--P6 在完整历史项目源码上是否能触发已知根因。
这不是新的盲测、recall、coverage 或论文统计；它是定位“十个 unseen 项目均为
0 finding”究竟更像规则过窄、实现漏检，还是候选表面很少的诊断实验。

直接复用 Mac 刚完成十项目扫描所使用的 legacy-2024 RAP 二进制、profile、环境和
单项目命令。不要改 RAP、不要重建 RAP、不要运行项目测试/二进制。每个条目 clone
一次、checkout 一次、按原扫描方式扫描一次即可。若仓库在该 revision 有 Cargo.lock，
继续使用 --locked；无 lock 或控制构建失败只记录失败，先不要花时间修复依赖。

## 执行顺序

每项均检出 source revision、scan exit code、finding JSON/文本的保存位置以及 P1--P6
计数。不要因为零 finding 再换 revision、改特征、改依赖或改规则。

| # | 预期模式 | 项目与 Issue | 快速 checkout | 预期根因 |
|---|---|---|---|---|
| 1 | P1 | servo/rust-smallvec #343 | 0f3aacb99ccfe0cbbf0bb83ecf736f67339ce8ae^ | 参数 index 在 bounds 检查前进入 ptr.add |
| 2 | P1 | GitoxideLabs/gitoxide #1460 | a807dd1ffb05efd177700d065095249e6c4b3c68 | 外部 bytes 进入 from_utf8_unchecked |
| 3 | P2 | rust-lang/libc #3560 | 6faa521f32fc11db9fc43a248a64463ce288b48d^ | public union active state 被 unsafe Debug 枚举 |
| 4 | P3 | jeromefroe/lru-rs #73 | a96f71a84d108f9b7506908b8e1c5da9524bc4da^ | unsafe 构造含未初始化字段的 LruEntry |
| 5 | P3 | hyperium/http #639 | bbe2a8f34f0f0aadcd124f5f2841bdae454bc837^ | Iter 在共享路径构造 &mut Bucket |
| 6 | P4 | fitzgen/bumpalo #164 | 71d99b89835a36a497e1f3ff54b625de5012e3ed^ | private len 递减后 get_unchecked(len) |
| 7 | P4 | rust-random/rand #779 internal-derived instance | 9828cdf0201f816e4e4bff07834d849a10be80e9 | 内部 u32 元素地址增宽为对齐 u64 pointer read |
| 8 | P5.1 | rust-random/rand #1158 | 90b89cdbb2655ac278cee43b5c343102a89616b1^ | generic Results 可为空但 get_unchecked(0) |
| 9 | P6 | nix-rust/nix #1819 | 8e91b28b64bdd18fc6fa61af8e89947ad7fb97bf^ | FFI out value未经 validity 验证 assume_init 为 enum |
| 10 | P6 | diesel-rs/diesel #4223 | 49aa163f1a9c793da6d591e10481fbfb6c044b51^ | SQLite out pointer 未验证即 NonNull::new_unchecked |

带 ^ 的值表示 source/fix locator 的父提交；这是本快速回放的修复前候选。
它们不是新的 affected-version 结论。若 Git 拒绝 parent 或 source 没有相应 crate，
记录 source-unavailable 并继续下一项，不要替换为 main。

## 最小输出

对每项只写一行：

```text
case | revision | control={ok|fail} | rap={ok|fail} | p1,p2,p3,p4,p5,p6 | result_dir
```

RAP 正常完成且六类均为零时写 `no_finding`；出现任意 finding 时保留原始
receipt，并标注其 pattern 和源码位置。先不要判真伪、不要压制误报、不要改规则。
十项完成后统一比较“预期根因”与“实际 finding”，再决定深入诊断。
