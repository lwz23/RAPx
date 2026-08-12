# RAP `unsoundaudit` v2：P1–P6 与局部过程间分析设计

## 1. 文档状态

本文是 `feature/unsoundaudit-v2-p1-p6` 分支的实现规范。它固定本轮已经批准的模式语义、分析边界、输出接口和验证门槛，供 Mac 上的独立 Codex 线程实现。若实现发现本文存在无法同时满足的要求，应先记录冲突并暂停，不得静默扩大 Pattern、改变工具链或降低验收门槛。

本轮的 Linux 基线为：

| 项目 | 固定值 |
|---|---|
| Git 分支 | `feature/unsoundaudit-v2-p1-p6` |
| 基线标签 | `unsoundaudit-v1-linux-baseline-20260812-9d36ece7` |
| RAP 源码树 SHA-256 | `9d36ece76ddacc8ab052ecaf4f64f412f7d648a93c1acc102d418c499c9d84c1` |
| `unsoundaudit.rs` SHA-256 | `47112d602470fe5afd236ef164e5798a7c2fdc628352c5912a705c25ba2f01e0` |
| Rust 工具链 | `nightly-2024-10-12` |
| rustc commit | `1bc403daadbebb553ccc211a0a8eebb73989665f` |
| Cargo commit | `15fbd2f607d4defc87053b8b76bf5038f2483cf4` |

上述哈希用于确认实现起点，不表示 v2 完成后的源码仍应具有相同哈希。

## 2. 目标与非目标

### 2.1 本轮目标

1. 将旧 P1 与旧 P3 合并为新的 P1，并用统一的局部 crate 内过程间分析自然覆盖零层或多层 helper 传播。
2. 让 P2、P3 以及新增 P4–P6 使用同一套 MIR function-summary 层，不再分别依赖字符串式调用链拼接。
3. 直接用 v2 语义替换此分支上的 `-unsoundaudit` 输出；新 schema 为 `rap-unit-v2`，恰好报告 `pattern1` 至 `pattern6`。
4. 用 27 个小型、无第三方依赖、只做 `cargo check --lib` 的 fixtures 和纯 Rust 单元测试验证实现行为。
5. 保持旧 v1 二进制、四 Pattern receipt、历史 coverage ledger 与 2,848 个已接受成功项目不变。

### 2.2 明确不做的事情

- 不实现跨 crate 或全程序 points-to/call-graph 分析。
- 不把所有泛型健全性义务都归入 P5；P5 本轮只实现非空义务 P5.1。
- 不实现一般性的并发/Happens-Before、生命周期/variance、Drop/unwind、panic safety、重入或 async cancellation 分析。
- 不因同一函数中出现危险 API 名、`set_len` 或特定返回类型而报告；必须存在同一 causal path 上的 source、flow、sink/invalid exposure 和未满足义务。
- 不在本工作包扫描真实项目，也不重新验证历史项目。
- 不把 fixture 结果称为生态覆盖率、recall、实际 precision 或已发现真实 bug。

## 3. Pattern 命名与兼容映射

| v2 Pattern | 与旧 Pattern 的关系 | v2 主特征 |
|---|---|---|
| P1 | 合并旧 P1 + 旧 P3 | 普通 public 函数参数经零层或多层本地调用流入 unsafe sink |
| P2 | 保留旧 P2 的严格含义 | safe caller 可直接修改的 literal public field/state 污染 unsafe sink |
| P3 | 旧 P4 重命名 | 合法入口之后，内部 unsafe-backed 操作创建、保留或暴露非法运行时值/状态 |
| P4 | 新增 | 内部派生的 index/len/offset/access-width 不满足 unsafe 内存操作前置条件 |
| P5 | 新增，但仅部分实现 | P5.1：公开泛型/关联类型没有保证 unsafe 访问所需的非空义务 |
| P6 | 新增 | FFI/宿主输出或 open trait/Iterator/callback 的运行时返回未经验证流入 unsafe sink |

这里的编号仅表示工具中的六类候选规则。它不是 Rust UB mechanism 分类，也不是对全部 IUE 的穷举。

## 4. 共同 eligibility 与互斥分类

### 4.1 报告一条候选所需的共同证据

每条 finding 至少必须绑定以下事实：

1. 一个外部可安全调用的入口，或该入口在本地 crate 内可解析的调用链；
2. 一个具体 source 或内部 unsafe origin；
3. source 到 sink/invalid exposure 的同一路径数据流；
4. sink 所需的具体 safety predicate；
5. 该 predicate 在 sink 前没有被有效建立，或内部 unsafe 结果以非法状态跨入 Safe Rust；
6. 可复现的源码 span、函数 DefId/稳定标识和规则编号。

仅有“安全函数中存在 `unsafe`”不构成候选。仅有函数名、类型名或同函数内不相关的危险调用也不构成候选。

### 4.2 以首次 contract failure 判定主 Pattern

P1–P6 必须互斥。分类器先确定同一 UB causal slice 上的**首次 contract failure**，再按以下决策树选择一个 primary pattern：

1. 若入口参数和 unsafe 前置条件均合法，错误发生在内部 unsafe-backed 操作创建、保留或暴露非法运行时值/状态时，主类为 **P3**。
2. 否则属于 unsafe sink 的前置条件建立失败，按真正控制该义务的 source provenance 分类：
   - FFI return/out-param 或 open trait/Iterator/callback 的运行时返回：**P6**；
   - 泛型/关联类型能力缺少本轮支持的非空保证：**P5.1**；
   - safe caller 可直接修改的 literal public field/state：**P2**；
   - 普通 public 函数参数及仍 data-dependent 于该参数的派生值：**P1**；
   - 不来自上述外部 source、由 private/internal state 经函数内部算术、布局或地址计算产生的 operand：**P4**。

这个顺序不是“编号较小优先”，而是在消除表面重叠。例如：

- callback 本身可能作为函数参数传入，但真正使 sink 失效的是 callback 的返回值，因此是 P6，`parameter` 只作为 secondary source metadata。
- generic length 最终也表现为内部 index，但首次缺失的是 public bounds 无法保证非空，因此是 P5.1，而不是 P4。
- 普通 public 参数即使经过 `index + 1`、`len - 1` 等算术，source 仍是参数，因此默认是 P1。只有 private/internal state 是唯一 source root，或外部值原有义务已经成立、随后项目内部 transform 重新破坏该义务时，才归 P4；`self` 仅作为 private state 的容器时不算普通 P1 参数。
- P4 在 unsafe 操作执行**之前**已有不满足的 bounds/offset 前置条件；P3 在合法入口之后由内部 unsafe 操作产生并暴露非法值/状态。
- 若同一实例确有两个独立 safety obligation，应生成两个具有不同 causal key 的 findings，而不是把同一 causal edge 同时报入多个 Pattern。

### 4.3 Secondary 信息

Finding 可以保存 `secondary_source_kinds`、传播边界和 mechanism tags，供解释和研究统计使用；secondary 信息不得增加主 Pattern 计数。

## 5. 六类 Pattern 的精确定义

### 5.1 P1：普通函数参数进入 unsafe sink

**必要条件：**

- source 是 public safe entry 的普通 value/reference 参数；
- 参数本身或其可追踪派生值，经 0..N 个本地、静态可解析调用后，到达枚举的 unsafe sink；
- sink 需要的 predicate 没有在所有到达 sink 的路径上由匹配该 source/origin 的验证建立。

**过程间要求：** 旧 P1 的同函数流是 call depth 0，旧 P3 的 private-helper 流是 call depth ≥1。二者共享同一 summary engine 和同一 `pattern1` 输出，不保留两个检测器。

**排除：** callback/Iterator 的返回值归 P6；泛型非空义务归 P5.1；public field 归 P2；与参数无关的内部 operand 归 P4。

### 5.2 P2：literal public field/state 污染 unsafe sink

**必要条件：**

- 一个 struct/union 的字段在源码中是 literal `pub`，safe caller 能直接写入；
- unsafe 操作读取该持久字段或其派生值作为 pointer、index、len、offset、tag 等安全关键 operand；
- 写入值所需 predicate 未在 sink 前建立。

P2 同样允许字段读取后经过本地 helper。普通 private field 配合 safe setter 不自动扩大为 P2；这类情形可以记录为未覆盖 external-state family，除非以后单独冻结新定义。

### 5.3 P3：内部 unsafe origin 暴露非法值或状态

**必要条件：**

- safe caller 的入口输入满足公开合同；
- 内部 unsafe-backed 操作在其 own causal path 上创建、保留或暴露 Rust validity/invariant 不允许的运行时值或状态；
- 该 internal unsafe origin 位于 public safe API 的 return、store 或 exposure causal slice 上，使安全调用者可以到达该非法值/状态。对于 Rust validity 规则，UB 可能在 invalid typed value 被构造时已经发生；不得错误声称一定要等到 return 后才发生。

P3 必须使用有限且可审计的 internal-origin registry，例如 `transmute`、unchecked UTF-8/value construction、`MaybeUninit::assume_init`、raw-backed reference/slice construction 等，并为每个 origin 指定其 validity obligation。不得恢复旧实现中“同一函数有 high-risk call + 看起来危险的返回类型即可命中”的宽松分支。

若 unsafe memory operation 的 index/offset 在执行前已经越界，则是 P4；若操作前置条件合法但它构造了非法 Safe Rust value/state，则是 P3。

### 5.4 P4：内部派生内存操作数违反 unsafe 合同

P4 只处理 source 为 internal-derived operand 的前置条件错误。本轮实现两个可执行子规则。

#### P4.bounds

- 内部计算得到 index、length、range endpoint 或 access width；
- 它控制 `get_unchecked(_mut)`、raw pointer read/write/copy、slice construction 等已枚举 sink；
- 分析找不到针对同一 origin 的支配性 `index < len`、`start <= end <= len`、`width <= len - offset` 等证明。

#### P4.offset

- 内部计算得到 pointer offset、byte offset 或 count；
- 它流入 `offset`/`add`/`sub`/`byte_add`/copy/dereference 一类操作；
- 分析找不到同一 allocation/range 下对 offset 与 access-width 的支配性证明。

本轮不能从任意算术自动证明 alignment 或 strict provenance。只有源码/类型中已有明确 alignment/provenance facts，且实现能结构化表达时，才可利用这些 facts；不得仅按调用名猜测后宣称已验证 alignment/provenance。

### 5.5 P5.1：泛型/关联长度缺少非空保证

P5 是明确标为 `partial` 的 Pattern family，本轮只实现 P5.1。

**必要条件：**

- public safe generic API 的类型参数、associated type/const 或其转换结果决定一个 slice/array-like value 的长度；
- unsafe sink 无条件访问元素 0，或以等价方式要求 `len > 0`；
- public bounds 和 sink 前的 runtime checks 都不能建立 `NonEmpty(value)`。

可识别的 sink 首版应保持有限，例如 `get_unchecked(0)`、对首元素 pointer 的 unchecked read/write、需要至少一个元素的 raw construction。可识别的验证包括支配 sink 的 `!is_empty()`、`len() > 0`、non-zero const/type witness，且必须对应同一 origin。

**明确排除：** P5.1 不声称实现一般的 `B(T) => R(T)` 证明，不覆盖任意布局相等、padding、validity、alignment、provenance、trait semantic law 或复杂 const-generic 约束。其他 P5 历史案例最多作为 future work，不得计为当前规则适用范围。

### 5.6 P6：外部运行时结果未经验证进入 unsafe

P6 处理“source 不是普通参数值，而是外部行为/宿主在运行时产生的结果”的前置条件错误。本轮实现两个子规则。

#### P6.FFI

- source 是 `extern`/foreign ABI 调用的 return value，或 foreign call 写入的 out-param；
- pointer、length、tag、status、discriminant 等结果流入枚举的 unsafe sink；
- 对应 predicate（如 non-null、成功状态、bounds、valid tag）没有支配 sink 的验证。

Out-param 必须按 memory effect 建模：foreign call 对实参指向位置的写入产生新的 source version，并使此前针对该位置的验证失效。

#### P6.behavior

- source 是 open trait、dynamic dispatch、`Iterator`、callback/closure/function pointer 等 caller-supplied behavior 的返回值；
- 返回结果控制 `transmute`、`unreachable_unchecked`、unchecked index/write、invalid enum/NonNull construction 等枚举 sink；
- sink 所需 predicate 没有被验证。

若具体 callee 在当前 crate 内可静态解析，应优先使用其真实 summary；只有确实 open/unresolved 的行为边界才产生 P6 source。普通 external Rust dependency call 不因“无法分析”就自动成为 P6；只有它属于已建模的 FFI/open-behavior source 类且实际流向 sink 时才报告。

## 6. 统一 MIR 事实模型

v2 应先建立统一事实层，再让六类规则消费事实。建议最小数据模型如下；具体 Rust 类型名可以调整，但语义不得退化为字符串搜索。

### 6.1 `Source`

至少区分：

- `PublicParameter { index, origin }`
- `PublicMutableField { def_id, field, origin }`
- `InternalDerived { expression_kind, origin }`
- `GenericNonEmptyCapability { type_or_assoc, origin }`
- `FfiReturn { callee, origin }`
- `FfiOutParam { callee, arg_index, origin }`
- `OpenBehaviorReturn { call_kind, trait_or_signature, origin }`
- `InternalUnsafeOrigin { operation, origin }`

`origin` 必须能在 copy/move、projection、field store/load 与 call actual/formal mapping 后保持身份；不得以局部变量显示名作为身份。

### 6.2 `Sink`

每个 sink model 至少声明：

- sink kind 和源码 span；
- 哪些 operands 是安全关键值；
- 每个 operand 需要哪些 predicates；
- sink 是“前置条件消费”还是“非法值/状态构造或 exposure”。

初版 registry 只收录已有明确 Rust unsafe contract、并能给出可执行 oracle 的操作。扩充 registry 必须同时增加正例与 matched negative。

### 6.3 `Predicate`

最小集合可包括：

- `NonZero(x)` / `NonEmpty(x)`
- `InBounds { index, len }`
- `RangeInBounds { offset, width, extent }`
- `NonNull(x)`
- `Initialized(x)`
- `ValidDiscriminant(x, ty)`
- `ValidValue(x, ty)`
- `Aligned { ptr, align }`
- `SameAllocation { ptr, base }`

Alignment/provenance facts只能在显式可得时使用，不要求首版求解一般性别名或 allocation provenance。

### 6.4 `ValidationFact`

一条验证事实必须包含：

- 被验证的稳定 origin/place；
- 已建立的 predicate；
- 建立位置、可用 CFG 区域和来源（branch/assert/type witness）；
- 会使其失效的 writes/effects。

### 6.5 `FunctionSummary`

每个函数的 summary 至少表达：

- formal parameter、public-field、internal/external sources；
- return place 和 out-state 对哪些输入/origin 依赖；
- 本函数内的 sink obligations 与 invalid exposure；
- source 到 sink/return/out-state 的 flow；
- 已建立/要求的 predicates；
- writes、mutable effects、foreign out-param effects；
- 调用边界、resolved callee 和 actual↔formal/return mapping；
- 能生成 deterministic witness 的最短或规范传播路径。

Summary join 必须是单调且有限高度的，并对等价事实 canonicalize/deduplicate，使递归 SCC 能达到稳定点。Summary equality 不得包含随递归不断展开的完整 call path 或无界 call depth：递归传播使用稳定的 SCC-cycle token 和确定性的最短代表 witness，depth 在递归处饱和为 `recursive`；source、sink、predicate 与 expression kind 均来自有限 registry。若实现加入仍可能增长的抽象，必须定义有单元测试的 widening，不能靠隐藏轮数截断。

## 7. 局部 crate 内过程间分析

### 7.1 可解析边界

过程间范围限定为当前 crate 内、编译器能够静态解析的调用。可使用 rustc 的 `Instance::try_resolve` 或该 pinned toolchain 上语义等价的 API，解析 concrete local `DefId`/monomorphized instance。

- direct local function、可解析 inherent/trait method：进入 call graph。
- dynamic dispatch、function pointer、open trait/callback：不伪造 callee，按 P6.behavior 边界建模。
- foreign ABI：按 P6.FFI 建模。
- 普通无法解析的 external dependency：记录 opaque boundary；只有匹配明确 P6 source model 且结果流入 sink时才报告。

### 7.2 Summary 求解流程

1. 对每个 MIR body 做 intraprocedural fact extraction。
2. 构建 local resolved call graph。
3. 用 Tarjan/Kosaraju 等确定性算法划分 SCC，并固定 DefId/稳定 key 排序。
4. 对 DAG 按依赖顺序处理；对递归 SCC 做单调 summary fixed point，直到没有新 canonical fact。
5. Callsite 上完成 actual→formal、callee return→destination、out-state→caller place 的映射。
6. 将 callee 的 requirements、effects、exposures 和 source/sink witness 合成到 caller。
7. 从 public safe roots 查询可达 causal paths，分类、去重并序列化。

不得继续使用固定“最多 10 轮”作为正确性条件。若为防御异常程序必须设置资源上限，应在 receipt 中显式写出 truncation/indeterminate，且该运行不得通过完整性门。

### 7.3 传播边界

首版至少支持：

- `Copy`/`Move`、简单 cast 与投影；
- field store/load 与 return assignment；
- local call 的参数、receiver、return place；
- foreign out-param 写入；
- P4 所需的基本 index/len/offset 算术关系。

遇到无法保持 origin 的复杂 alias 时应保守标记 unknown/indeterminate；不得为了提高计数而把不相关 values 合并成同一 source。

## 8. 支配性验证与写入失效

Sanitizer 不能再以函数名包含 `check`、`validate` 等关键词判定。一个 validation 只有同时满足以下条件才可抑制 finding：

1. 它建立的 predicate 能蕴含该 sink 的具体 obligation；
2. 它验证的是同一 origin/place 或有明确等价映射的值；
3. 它在 MIR CFG 上支配 sink，亦即所有到达 sink 的路径都经过有效分支；
4. validation 与 sink 之间没有可能改变该值、长度、base pointer、allocation 或相关字段的写入/effect。

Assignments、可变引用逃逸、未知 mutable call、foreign out-param write 和相关 field write 必须使匹配的 validation 失效。Alias 不确定时宁可保留候选并标为限制，也不能把 unrelated check 当作证明。

跨函数 validation 只能通过明确的 summary contract 传播。例如 callee summary 可以表达“正常返回保证 result 非空”或“调用要求 arg0 < arg1.len”；不能从函数名推断。

## 9. Finding 分类、去重与输出

### 9.1 直接替换接口

在 v2 分支上，现有 `-unsoundaudit` 直接输出 `rap-unit-v2`。不新增并存的 v1/v2 CLI 开关，也不让 v2 结果进入旧 `rap-unit-v1` ledger。

Top-level `pattern_counts` 必须恰好包含：

```text
pattern1, pattern2, pattern3, pattern4, pattern5, pattern6
```

每条 finding 至少保留 v1 的 crate/toolchain/source metadata，并新增结构化字段：

- `primary_pattern` 和 `rule_id`（如 `P4.bounds`、`P6.ffi_return`）；
- `source`、`propagation`、`sink`、`obligation`、`validation_status`；
- public safe root 与 source/sink spans；
- local call-depth/boundaries 和 canonical witness；
- `secondary_source_kinds`；
- `heuristic_candidate_generator: true`；
- rule-specific `limitations`；
- schema、RAP commit、rustc commit 和输入标识。

### 9.2 互斥与去重

Primary classifier 使用第 4.2 节决策树。建议 canonical causal key 包含：

```text
crate + public_root + source_origin + first_contract_failure + sink/exposure + obligation
```

相同 causal key 经不同 call paths 到达时仅保留一条主 finding，并选择 deterministic witness；独立 obligations 具有不同 key，可分别报告。任何一条 finding 只能增加一个 `pattern_counts` 项。

### 9.3 确定性

所有 facts、SCC、findings、paths 与 JSON keys/arrays 都需使用稳定 key 排序；不得写入 wall-clock time、临时 target path 或机器相关绝对路径到 normalized receipt。相同 source/toolchain/input 的两次 normalized 输出必须 byte-identical。

## 10. 27 个轻量 fixtures

所有 fixtures 都是无第三方依赖的 library crate，只允许 `cargo check --lib` 加静态 scan。代码不得执行 UB path，不增加 runtime test 作为判断依据。

### 10.1 迁移现有 10 个 fixtures

| 现有 fixture | v2 预期 |
|---|---|
| `pattern1_positive` | 新 P1，恰好 1 条 |
| `pattern1_negative_local_pointer` | 0 条 |
| `pattern3_positive` | 新 P1，恰好 1 条，验证旧 P3 的 helper flow 被合并 |
| `pattern3_negative_private_chain` | 0 条 |
| `pattern2_positive` | 新 P2，恰好 1 条 |
| `pattern2_negative_private_field` | 0 条 |
| `pattern2_display_negative` | 0 条 |
| `pattern2_write_negative` | 0 条 |
| `pattern4_positive` | 新 P3，恰好 1 条 |
| `pattern4_negative_checked_utf8` | 0 条 |

迁移是复制到 v2 fixture root 并更新 oracle；不得改写当前 Linux campaign 使用的原始 v1 fixtures/receipts。

### 10.2 新增 8 组 vulnerable/fixed pairs

每组各含一个 vulnerable candidate 与一个 matched fixed negative，共 16 个 fixtures：

1. P1 two-hop parameter flow；
2. P2 public-field through helper；
3. P3 internal invalid construction through helper return；
4. P4.bounds；
5. P4.offset/access-width；
6. P5.1 generic/associated non-empty obligation；
7. P6.FFI return/out-param；
8. P6.behavior trait/Iterator/callback return。

Fixed fixture 应只增加真正对应的 validation/bound/ownership construction，不得通过删除 unsafe sink 或把 safe API 改为 unsafe 来“修复”。

### 10.3 Noise fixture

再增加 1 个 all-zero fixture：它应同时包含多个表面高风险词/类型或无关 unsafe 操作，但不存在满足共同 eligibility 的 causal path；六个 Pattern 均须为 0。它用于阻止基于 API 名或同函数共现的过度激进实现。

### 10.4 Fixture oracle

- 每个 positive fixture 恰好一个预期 primary finding，其他五类均为 0。
- 每个 fixed/negative/noise fixture 六类均为 0。
- 同一 source→sink causal key 不得重复报告。
- 27/27 fixtures 必须形成完整 receipt，无 missing/timeout/truncation。
- 在同一 checkout、toolchain 和 input 上运行两遍；normalized receipts 必须 byte-identical。

Fixture suite 只能说明这些有限 oracle 上的行为符合设计，不构成真实项目的 recall/precision 或生态 coverage 证据。

## 11. 纯 Rust 单元测试

至少覆盖以下独立组件：

- source/origin 在 MIR place/copy/projection 上的规范化；
- actual↔formal、callee return→destination 和 out-param mapping；
- call graph SCC 与递归 fixed point；
- summary join 的单调性、幂等性和 deterministic ordering；
- predicate entailment 的支持子集；
- CFG dominance；
- assignment/mutable call/foreign write 导致的 validation invalidation；
- 第 4.2 节 primary classification 决策树；
- causal-key dedup 和 deterministic witness 选择；
- `rap-unit-v2` schema 校验及 pattern1..pattern6 完整性。

单元测试不得依赖真实第三方项目或网络。

## 12. Mac 与 Linux 的阶段隔离

### 12.1 当前 Ubuntu 阶段

该 Ubuntu 源码路径被现役/刚结束的 RAP campaign provenance 绑定；在 campaign 完成状态和证据冻结未经显式确认前，仍按不可变路径处理。本阶段只允许：冻结源、创建隔离 Git 分支、编写文档/fixtures 源码和推送；不得运行新的 Cargo/rustc/Miri/RAP build/scan，也不得修改 `/home/lwz/unsound_scanner/RAPx`。

### 12.2 Mac 实现阶段

Mac 线程应从固定 branch/tag 开始，先验证基线哈希，再安装/核验：

- `nightly-2024-10-12`；
- `rustc-dev`、`rust-src`、`llvm-tools-preview`；
- Xcode Command Line Tools；
- LLVM/libclang、Z3、CMake、`pkg-config`；
- 构建时所需的 `DYLD_LIBRARY_PATH`。

Mac 上保持 project concurrency=1、Cargo jobs=1，使用独立 `CARGO_HOME` 和 target/result roots。不得修改 nightly 来绕过失败。若 exact baseline 在依赖补齐后仍不能构建，输出 blocked receipt（命令、版本、错误、哈希）并停止；不得通过改工具链或降低 oracle 继续。

Mac 只推送源码、fixture、测试和小型文本/JSON receipts；不得把 `target/`、二进制或机器绝对路径提交回 Git。

### 12.3 campaign 结束后的 Linux 复核

只有确认活动 campaign 完全停止后，Linux 才可在 fresh `CARGO_HOME`/target、jobs=1 环境中构建 v2 并复跑单位测试和 27-fixture suite。此工作包仍不扫描真实项目。

经复核通过后，部署采用有收据的原子切换：

1. 将旧目录重命名为只读的 `RAPx-v1-frozen-9d36ece7/`；
2. 将 v2 checkout 放到原 `RAPx/` 路径；
3. 记录旧 v1 registry 此后不能再对原路径做 live source revalidation；
4. 保留 v1 tag、源码哈希、旧二进制和 ledger 的可恢复性。

这一切换不能重解释或覆盖旧 `rap-unit-v1` receipts。

## 13. 分阶段实现顺序

建议 Mac 线程按以下最小闭环实施，每一步通过后再继续：

1. 加入 v2 数据类型、schema validator 和失败的 classification/serialization 单元测试。
2. 实现 intraprocedural fact extraction、origin 与 validation/invalidation。
3. 实现 local call graph、callsite mapping 与 SCC fixed point。
4. 用统一事实重写新 P1、P2、P3；迁移 10 个 fixture oracle。
5. 实现 P4.bounds 与 P4.offset，加入两组 pairs。
6. 实现 P5.1，明确输出 `partial` limitation，加入一组 pair。
7. 实现 P6.FFI 与 P6.behavior，加入两组 pairs。
8. 加入 all-zero noise、全局 dedup 和 deterministic receipt 双跑。
9. 生成 Mac verification receipt 并推送；等待 Linux campaign 结束后的独立复核。

若为了让某一步通过必须扩大 source、sink 或 predicate registry，应先同时给出新增正例、matched negative、误报边界和文档修订，不能只添加模糊关键词规则。

## 14. 验收门槛

v2 只有同时满足以下条件才可称为“fixture-level implementation complete”：

- 基线起点的 branch/tag、源码树哈希、`unsoundaudit.rs` 哈希和 pinned rustc commit 可核验；
- 统一 function-summary 引擎覆盖 P1–P6，局部递归用 SCC fixed point，不使用固定轮数冒充收敛；
- P1/P2/P3 与 P4/P5.1/P6 的 primary 语义符合本文，P5 明确为 partial；
- validation 与具体 origin/predicate 绑定、支配 sink，相关写入会失效；
- `-unsoundaudit` 直接产出 `rap-unit-v2`，`pattern_counts` 恰好为 pattern1..pattern6；
- 27/27 fixtures 完成：所有 positive 恰好一条主 finding，所有 negative/fixed/noise 为零，无跨 Pattern 重复；
- 纯 Rust 单元测试全部通过；
- 两次 normalized receipts byte-identical；
- 所有运行保持 jobs=1、fresh target、独立结果根，无历史 ledger 写入；
- 当前 Linux campaign、2,848 个 historical accepted successes、v1 fixtures/receipts 和四 Pattern schema 未被修改或重验。

Mac 通过这些门槛后仍只能声称“六类静态 candidate rules 已在设计 fixtures 上实现并验证”。真实项目适用性、增量 warnings、人工 precision、运行成本和论文中的经验效果，必须在 campaign 停止后以另一个有冻结输入和独立负例的实验包评估。

## 15. 科学表述边界

允许的表述：

- “P1–P6 是静态候选模式（candidate generators）。”
- “P5 当前仅支持 non-empty obligation 子规则 P5.1。”
- “统一 summary engine 支持当前 crate 内静态可解析调用的传播。”
- “27 个 fixtures 验证了固定 oracle 上的预期候选行为。”

禁止仅凭本工作包使用的表述：

- “RAP 已覆盖这些历史 Issue”或“发现了 PDF 中所有案例”；
- “P1–P6 代表完整 Rust 生态健全性”；
- “fixture 结果等于 recall、precision、false-positive rate 或真实项目 coverage”；
- “P5 证明任意泛型 bounds 足以蕴含 unsafe contract”；
- “P4 已一般性解决 alignment/provenance”；
- “P6 能可靠理解所有 trait、FFI 或宿主语义”。

历史 GitHub IUE 实例用于提出和解释这些规则，不应同时被当作独立验证集。后续论文实验必须冻结规则后，另用独立 confirmation cases、matched negatives 和真实项目 warning review 评价。
