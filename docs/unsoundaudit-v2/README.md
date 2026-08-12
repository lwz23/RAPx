# UnsoundAudit v2 handoff package

本目录是 `feature/unsoundaudit-v2-p1-p6` 的冻结交接包。建议按以下顺序阅读：

1. [`baseline_manifest_v1.json`](baseline_manifest_v1.json)：机器可读的 Linux v1 精确基线；
2. [`design.md`](design.md)：P1–P6、统一过程间摘要引擎、输出 schema 与阶段门；
3. [`fixture_matrix.md`](fixture_matrix.md)：27 个微型 crate 的逐项 oracle；
4. [`frozen-v1-fixtures/`](frozen-v1-fixtures/)：Ubuntu 旧 10 个 fixture 的逐字冻结输入与哈希清单；
5. [`MAC_HANDOFF.md`](MAC_HANDOFF.md)：Mac 新线程的环境预检、执行顺序、receipt 和可复制启动 prompt。

基线 tag 是 `unsoundaudit-v1-linux-baseline-20260812-9d36ece7`。开发只在 feature branch 上进行；不得重写 tag，不得把 v2 receipt 写入旧 `rap-unit-v1` ledger。

本包规定的是候选生成规则的实现与 fixture-level 验收，不是实际项目 recall、precision 或生态 coverage 实验。
