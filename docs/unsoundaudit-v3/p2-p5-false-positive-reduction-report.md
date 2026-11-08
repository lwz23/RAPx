# P2–P5 明显误报消减报告

## 结论

本轮采用通用类型、调用点、关系和初始化证明域，没有加入函数名、源码路径或行号白名单。最终 `core` 精确扫描的 P2–P5 active 为 **0 raw / 0 family**；这表示当前有限证明域没有找到满足 active 门槛的 core 违约 witness，不表示 core 已被证明安全，也不表示总体误报率为零。未完成的证明全部保留在 `unestablished.jsonl`。

机制包的 30 个 P2–P5 正例 root 全部保持 target pattern active（30/30）；负例不进入对应 target-pattern active。P1/P6 机制计数保持 2/9。

## Core 新旧状态（raw / family-state incidence）

family 含义：active/suppressed/reclassified 使用生产 candidate-family key；unestablished 因输出契约没有 family ID，使用报告专用、去 root 的 decision-family key。两类 family 数不能作为同一总体直接相加。

| Pattern | 旧 active | 新 active | 新 suppressed | 新 reclassified | 新 unestablished | 新 considered raw |
|---|---:|---:|---:|---:|---:|---:|
| P2 | 11059 / 1863 | 0 / 0 | 163667 / 14790 | 0 / 0 | 8007 / 1595 | 171674 |
| P3 | 2273 / 2273 | 0 / 0 | 1334 / 1334 | 0 / 0 | 2249 / 225 | 3583 |
| P4 | 511 / 126 | 0 / 0 | 0 / 0 | 0 / 0 | 1004 / 290 | 1004 |
| P5 | 280 / 74 | 0 / 0 | 148026 / 14672 | 0 / 0 | 54743 / 3770 | 202769 |

旧基线的其余状态为：

| Pattern | 旧 suppressed | 旧 reclassified | 旧 unestablished | 旧 considered raw |
|---|---:|---:|---:|---:|
| P2 | 118110 / 10322 | 50600 / 5568 | 0 / 0 | 179769 |
| P3 | 1311 / 1311 | 0 / 0 | 0 / 0 | 3584 |
| P4 | 0 / 0 | 0 / 0 | 0 / 0 | 511 |
| P5 | 123975 / 13243 | 24039 / 2090 | 0 / 0 | 148294 |

旧 active 基线与最终 active 变化：P2 `11,059/1,863 → 0/0`，P3 `2,273/2,273 → 0/0`，P4 `511/126 → 0/0`，P5 `280/74 → 0/0`。没有设置 finding 数量下降目标；这些变化由义务分类产生。

## 原因分布

### P2
- **suppressed**：`nonpersistent_public_field` 163417 raw / 14702 family；`source_cannot_satisfy_requirement` 250 raw / 88 family
- **unestablished**：`unmodeled_unsafe_call` 5824 raw / 1098 family；`missing_exact_context` 2170 raw / 485 family；`unknown_capability_match` 13 raw / 13 family

### P3
- **suppressed**：`contract_implied_by_type` 1332 raw / 1332 family；`contract_discharged` 2 raw / 2 family
- **unestablished**：`missing_exact_context` 2053 raw / 159 family；`validity_obligation_unknown` 194 raw / 194 family；`initialization_state_unknown` 2 raw / 2 family

### P4
- **unestablished**：`missing_exact_context` 542 raw / 184 family；`derivation_obligation_unknown` 462 raw / 106 family

### P5
- **suppressed**：`pointer_view_has_no_capability_gap` 146792 raw / 14608 family；`root_has_no_relevant_capability_surface` 1234 raw / 64 family
- **unestablished**：`unmodeled_unsafe_call` 30916 raw / 2179 family；`generic_witness_unknown` 16949 raw / 1273 family；`missing_exact_context` 5331 raw / 283 family；`unknown_capability_match` 1547 raw / 86 family

## 验证与性能

- 最终 core：850.620s（≤900s），peak physical 8,947,475,864 bytes，peak RSS 5,600,395,264 bytes；内存仅记录，不设淘汰线。
- 覆盖：exact/complete；discovered 10,912，bound-exact 10,747，optimized-away-exact 165，unbound/ambiguous/elided 均为 0。
- 四态总账：0 active + 320,856 pure suppressed + 0 reclassified + 182,776 unestablished = 503,632 considered。
- pinned Rust：428/428；current Rust：429/429；Python no-build：40/40。
- pinned/current 机制包一致：50 active、49 families、38 pure suppressed、12 reclassified、28 unestablished、128 considered。

## 分层审查

- 原 48-family 名单未持久化。本报告按原配额 P2=12、P3=12、P4=16、P5=8，以固定 seed 按 model/depth 确定性重建；该限制在 JSON 中显式标注。重建样本没有 family 留在 active。
- 新盲抽 48 按 pattern/state/model/depth 分层。最终 active population 为 0，所以不存在可重点超采样的 active family；实际抽到 18 suppressed、30 unestablished（P4 无 suppressed population）。前者均有通用证明/不适用证据，后者均明确保留证据缺口。
- “active 样本无明显误报”在本轮是空真条件；更强的事实来自完整 census：最终 active 文件为空。盲样本不用于宣称总体误报率。

## 剩余风险

- 182,776 个 unestablished decision 仍需后续模型扩展；它们不是安全证明，也不是高置信 finding。
- core 的零 active 不能外推到其他 crate/toolchain；正例召回由机制包而非 core population 证明。
- unestablished 输出没有生产 family_id、contract model 和 propagation depth；报告只能使用显式标注的 decision-family/requirement bucket，不能伪造精确 family 映射。

## 机器可读证据

- `docs/unsoundaudit-v3/p2-p5-state-comparison.json`
- `docs/unsoundaudit-v3/p2-p5-old48-audit.json`
- `docs/unsoundaudit-v3/p2-p5-blind48-audit.json`
