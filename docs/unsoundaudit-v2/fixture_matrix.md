# UnsoundAudit v2 轻量验证矩阵

本文档冻结本工作包的 fixture 范围与 oracle。它是开发测试，不是生态级准确率实验：通过这些用例只能说明规则实现与预定语义一致、没有出现明显的交叉误报，不能据此声称实际 precision、recall 或 coverage。

## 通用约束

- 总数必须恰好为 27 个无第三方依赖的 library crate。
- 只允许 `cargo check --lib` 和 RAP 静态扫描；不得执行任何可能触发 UB 的入口。
- 所有 positive 都必须保留 safe entry、unsafe sink 和完整 source-to-sink 路径。
- 每个 positive 只允许一个 primary pattern；其他五类计数必须为零。
- 每个 matched negative 保留与 positive 相同的 unsafe sink，只增加缺失的合同证明或改变 source 身份；不得通过删除 `unsafe` 使测试失去约束力。
- 所有 negative 与 noise fixture 的六类计数都必须为零。
- 同一 source-to-sink 根因只产生一条 finding；跨过程摘要不得按调用层重复报告。
- 每个 fixture 的 `fixture.json` 明确记录 expected primary、expected count、禁止的额外 pattern、source、sink、obligation 和修复事实。

## A. 迁移现有 10 个 v1 fixture

这些用例来自现有 UnsoundAudit 测试，但其 pattern 编号必须按 v2 语义重新解释。历史 receipt 不得原地改写。

| v2 fixture ID | 旧用例 | v2 oracle | 目的 |
|---|---|---:|---|
| `legacy_p1_param_direct_positive` | `pattern1_positive` | P1=1 | 普通公开参数直接进入 unsafe sink |
| `legacy_p1_local_pointer_negative` | `pattern1_negative_local_pointer` | 全零 | 本地安全派生指针不是公开参数 source |
| `legacy_p2_public_field_positive` | `pattern2_positive` | P2=1 | literal public field 可被安全调用者污染 |
| `legacy_p2_private_field_negative` | `pattern2_negative_private_field` | 全零 | private field 不构成 strict P2 source |
| `legacy_p2_display_negative` | `pattern2_display_negative` | 全零 | 常见 trait/格式化代码不因字段访问被误报 |
| `legacy_p2_write_negative` | `pattern2_write_negative` | 全零 | 常见写入路径不因字段访问被误报 |
| `legacy_p1_param_helper_positive` | `pattern3_positive` | P1=1 | 旧 P3 变为新 P1 的跨过程正例 |
| `legacy_p1_private_chain_negative` | `pattern3_negative_private_chain` | 全零 | 不可由公开参数建立的 private chain 不报 P1 |
| `legacy_p3_invalid_output_positive` | `pattern4_positive` | P3=1 | 旧 P4 变为新 P3：内部 unsafe 非法输出 |
| `legacy_p3_checked_utf8_negative` | `pattern4_negative_checked_utf8` | 全零 | 有效性检查建立后不得仅因危险 API 名报警 |

## B. 新增 8 组成对 fixture

### B1. P1：两层参数传播

- `p1_two_hop_positive`：`pub fn entry(slice, index)` 经两个 private helper 将 `index` 传给 `get_unchecked`，无 `index < len` 证明；预期 P1=1。
- `p1_two_hop_checked_negative`：同一 sink 与调用链，但 public caller 的范围检查支配调用；预期全零。该用例强制验证谓词能随 actual-to-formal 映射跨过程传播。

### B2. P2：公开字段经 helper 传播

- `p2_helper_positive`：结构体包含 `pub index: usize` 与 private buffer，安全方法经 helper 使用该字段执行 unchecked access；预期 P2=1，P1=0。
- `p2_helper_checked_negative`：同一路径，但方法先证明 `self.index < self.data.len()`；预期全零。`self` 不能被笼统归成 P1 参数。

### B3. P3：内部非法值经 helper 返回

- `p3_helper_return_positive`：private producer 对未初始化 `bool` 调用 `assume_init`，中间 helper 返回，public safe API 暴露该值；预期 P3=1。
- `p3_helper_return_initialized_negative`：保留 `assume_init`，但来源改为 `MaybeUninit::new(false)`；预期全零。该用例拒绝“只按危险 API 名报警”。

### B4. P4.1：内部派生 bounds 义务

- `p4_internal_bounds_positive`：合法入口后，private `len` 先减一，再对已缩短到该长度的 slice 使用 `get_unchecked(len)`；index 恰等于新长度；预期 P4=1。
- `p4_internal_bounds_checked_negative`：保持同一内部派生和 sink，但在原 slice 上读取 `len - 1`，或显式证明 index 小于缩短后长度；预期全零。

### B5. P4.2：内部派生 pointer-offset/access-width 义务

- `p4_internal_offset_positive`：private buffer 的内部偏移由长度算术派生，随后执行已知宽度的裸指针读取/复制；剩余空间小于访问宽度且没有证明；预期 P4=1。
- `p4_internal_offset_checked_negative`：同一 sink，但以支配性检查证明 `offset + width <= allocation_len`，并使用无溢出的规范化形式；预期全零。

第一版不将任意 alignment 或 provenance 猜测塞入 P4.2；只有分析中存在明确可比较事实时才能建立这些义务。

### B6. P5.1：关联类型 slice 的非空义务

- `p5_associated_slice_nonempty_positive`：公开泛型 API 只要求关联类型 `Results: AsRef<[u64]> + Default`，随后对 `as_ref()` 结果执行 `get_unchecked(0)`；一个合法 witness 使用 `[u64; 0]`；预期 P5=1。
- `p5_associated_slice_nonempty_negative`：同一泛型约束与 sink，但 `is_empty()` 早退支配 sink；预期全零。

该 oracle 只验证有限模板“generic/associated slice 必须非空”。它不要求 RAP 自动综合 witness，也不验证任意 `B(T) => R(T)`、布局相等、padding、provenance 或任意 trait 语义。

### B7. P6.1：FFI out-parameter

- `p6_ffi_out_param_positive`：`extern "C"` 函数通过 `*mut *mut u8` 写回指针；wrapper 只检查状态码，未检查指针非空就调用 `NonNull::new_unchecked`；预期 P6=1。
- `p6_ffi_out_param_nonnull_negative`：同一路径与 sink，但成功状态后另有 `out.is_null()` 检查；预期全零。只做 library check，不链接或调用外部函数。

### B8. P6.2：开放 trait/Iterator 返回值

- `p6_open_trait_result_positive`：public safe generic API 调用调用者可实现的 trait method，返回运行时 discriminant/index，再未经范围或 validity 验证进入有限的 unsafe sink；预期 P6=1。
- `p6_open_trait_result_checked_negative`：保留 trait call 与 sink，但对同一返回值做支配性范围/validity 检查；预期全零。

若缺失义务来自关联类型可选择的静态形状，优先归 P5；若缺失义务来自开放调用在运行时返回的值，归 P6。

## C. 全零噪声 fixture

`all_patterns_checked_noise` 在一个 crate 内覆盖常见但安全的相似形状：

- public 参数经 helper 后使用 checked indexing；
- public field 在进入 sink 前有同值 bounds check；
- 已初始化的 `MaybeUninit` 值被返回；
- 内部长度算术后有无溢出、支配性的范围证明；
- generic slice 在访问 0 前检查非空；
- FFI out-pointer 与开放 trait 返回值在进入 sink 前分别经过同值检查。

预期六类全部为零。该 crate 主要防止规则变成“看到 unsafe API、helper、泛型或 extern 就报警”。

## D. 纯 Rust 单元测试

除 27 个 crate 外，分析实现还必须有不依赖项目构建的单元测试：

1. source、sink、predicate、validation fact 的相等、join 与稳定排序；
2. callsite actual-to-formal、callee return-to-caller destination、out-parameter 投影映射；
3. 递归调用图的 SCC 单调 fixed point，事实稳定后终止；
4. validation block 对 sink block 的 dominance，以及写入后的 validation invalidation；
5. 同一 source/place/predicate 绑定，拒绝“检查了另一个值”或“只检查状态码”的假证明；
6. primary-pattern precedence 与 secondary-source 保留；
7. source-to-sink 根因去重和确定性序列化；
8. 无法静态解析的本 crate调用不被臆测为已有证明；只有符合 P6 source 条件时才进入 P6。

## E. 完成门

- 27/27 fixture 均能 `cargo check --lib`；这是可编译性，不执行入口。
- 所有 positive 精确命中其唯一 primary oracle；所有 paired negative 和 noise 为零。
- 两次 clean、串行的 normalized `rap-unit-v2` receipt 生成结果逐字节一致。
- 输出 schema 恰有 `pattern1` 至 `pattern6`，且每条 finding 带 source、propagation path、sink、obligation、validation 状态、rule ID 和 candidate-generator 限制。
- 任何一项失败都必须保留为明确失败或 blocked receipt，不能通过更换 nightly、删除 sink、放宽 oracle 或修改 fixture 根因使其变绿。
