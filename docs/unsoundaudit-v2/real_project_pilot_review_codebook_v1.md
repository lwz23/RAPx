# RAPx P1–P6 真实项目候选质量 Pilot 审计代码本 v1

本代码本仅用于十个冻结第三方仓库的静态候选诊断。finding 是候选生成器输出，不等于真实缺陷、检出率、精确率、召回率或生态覆盖率。

## 审计单位

每一行对应一个稳定的 source–sink 根因键。相同根因通过不同 public root 到达时合并展示，但原始 finding 和 receipt 仍保留。审计必须核对冻结 revision、目标、source/sink span、调用路径、源码片段哈希和原始 receipt。

## disposition

- `new_iue_candidate`：存在可证伪的 safe-public-path UB 假设，值得另立 PoC/动态验证任务；本 pilot 不执行 PoC、不上报。
- `likely_false_positive`：源码证据支持候选来自通用分析缺口，并必须选择下列一个错误族。
- `unresolved`：在当前人工预算内无法闭合安全证明，也无法建立具体 UB 假设。
- `duplicate`：与同一 ledger 中另一个 root key 属于同一根因，必须填写 `duplicate_of`。
- `tool_or_receipt_failure`：源码、span、route 或 receipt 不足以进行有效审计。

## likely_false_positive 错误族

- `source_misattribution`：源值或外部可控性归因错误。
- `missing_dominating_validation`：存在同对象、同版本且支配 sink 的有效检查，但分析未恢复。
- `interprocedural_summary_imprecision`：局部调用返回/参数/条件摘要缺失或过宽。
- `sink_contract_imprecision`：sink 的真实 unsafe 前置条件比当前 registry 更窄。
- `primary_pattern_precedence`：候选存在，但主 Pattern 归类优先级不合适。
- `deduplication`：相同 source–sink 根因被重复实例化。

禁止 `repository_allowlist`、仓库名/函数名/路径白名单、空证据和空理由。任何可推广修复必须独立进入 correction package，以成对 fixture、全部冻结 27 fixtures、本 pilot replay 和新 hold-out 样本验证；不得在本 pilot 中修改 analyzer。
