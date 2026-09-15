# DCDLP 第二阶段实现报告

## 阶段一决策

服务器机制审计使用 6 个严格配对运行（Cora/CiteSeer × seed 0/1/2），
运行配对缺失数为 0，最终决策为：

```text
A14_MECHANISM_NOT_SUPPORTED
```

关键依据：

- A14-w010 与 A5 的平均 MRR 差为 `+0.002345`，在预设 `0.01`
  性能容忍范围内，但方向仅有 3/6 个 seed 为正。
- 主分支 Routing Selectivity 配对差为 `-0.005642`，
  bootstrap 95% CI 为 `[-0.017058, 0.004720]`，Holm 校正后
  `p=1.0`，不支持路由选择性提高。
- 包含 interaction 后的 Routing Selectivity 配对差为
  `+0.000193`，95% CI 为 `[-0.009168, 0.009641]`，仍无改善证据。
- 目标分支响应配对差为 `-0.008471`，95% CI 为
  `[-0.009658, -0.007347]`：A14-w010 显著削弱目标响应，而不是
  让目标分支相对占优。
- 非目标分支响应差为 `-0.008601`，但 95% CI
  `[-0.020287, 0.002611]` 跨零；不能把它解释为稳定的交叉泄漏下降。
- 受约束干预下 A14 相对 A5 的路由优势为 `-0.005642`，
  random-rewire 下为 `-0.003671`，受约束干预并不优于随机重连。
- interaction share 平均下降 `-0.007455`，说明没有明显 interaction
  绕行，但这不足以弥补路由与目标响应失败。
- post-hoc leakage 仅 17/36 个配对方向改善，目标信息保持为 11/12，
  方向并不稳定。
- Cora 的 CN 剂量 Spearman 从 A5 的 `0.159259` 降至
  A14-w010 的 `0.011111`，跨数据集剂量响应不稳定。
- A14-w010 平均训练时间为 `124.13s`，A5 为 `28.61s`，
  约慢 `4.34×`，但没有对应的稳定机制收益。

因此停止继续搜索 A14 固定损失权重。A5 保持为主预测模型，结构干预
保留为冻结模型的评测与机制审计工具。

## 缺失与风险

- 6 个 checkpoint/prediction 运行配对全部可用。
- 生成评测干预时有 402 次显式 `MISSING`：
  CN two-switch 找不到合法候选，或降低 degree 会断开端点。
- `dose_response.csv` 有 72 行显式 `MISSING`。程序没有用替代干预、
  Uniform 候选或伪造结果填补。
- `summary.json` 的 `evidence_status=COMPLETE` 表示运行级 A5/A14
  配对完整，不表示每个节点对的每个干预幅度都成功；剂量响应应按
  “部分覆盖”解释。
- A14-w010 权重是在查看测试 MRR 后选出的，只能作为探索性诊断结果，
  不能作为独立确认性证据。
- 进一步代码核验发现，旧 `train.py` 即使记录
  `protocol_eval=heart`，验证/测试仍调用 Uniform 负采样，没有读取
  `GraphDataset.valid_neg/test_neg` 中的 HeaRT 官方逐正边候选。因此旧
  MRR 和逐边 rank 只能视为 Uniform-candidate 指标，不能写成正式
  HeaRT 指标。这个问题不改变冻结 checkpoint 的分支路由、交叉响应和
  probe 表示结论，但降低了“预测性能持平”证据的协议有效性。

## 第二阶段代码实现

按路线 B 只实现目标明确的 A5 改进，不实现新的 A14 权重搜索：

1. 新增 `cn_feature_mode`：
   - `raw`：完全保留旧 A5 行为；
   - `residual`：conditional CN residual + normalized CN；
   - `raw_plus_residual`：raw log-CN + conditional residual +
     normalized CN。
2. Conditional CN 期望模型只在 train positive 和固定 seed 的
   train negative 上拟合，不读取 valid/test；端点 degree 使用对称
   排序，保证无向边交换不改变结果。
3. 保留共同邻居节点表示的 sum/max 聚合，未用显式统计替代表示聚合。
4. residualizer 与数据来源元信息写入 checkpoint，评测时必须恢复；
   非 raw 模式缺失 residualizer 会直接报错，不会静默退回 raw。
5. 新增 `interaction_mode`：
   - `unrestricted`：旧行为；
   - `audited`：保留 interaction，并在逐边预测中显式记录；
   - `disabled`：前向计算强制将 interaction 分数置零。
6. 逐边预测新增 `score_interaction`、模型实际使用的 CN expected、
   residual、normalized CN、feature mode 和 interaction mode。
7. 修复 HeaRT 候选接入：`heart/ogb` 模式强制读取数据集中的官方
   grouped candidates；缺失或形状不合法时明确失败，禁止回退到
   Uniform。官方候选保持原样，不按其他 split 的边做二次过滤。

## 修改文件

- `src/dcdlp/data/pair_statistics.py`
- `src/dcdlp/models/branches.py`
- `src/dcdlp/models/dcdlp.py`
- `src/dcdlp/train.py`
- `src/dcdlp/evaluate.py`
- `src/dcdlp/cli.py`
- `scripts/analyze_mechanism_audit.py`
- `configs/model/dcdlp.yaml`
- `tests/test_branch_shapes.py`
- `tests/test_metric_formulas.py`

## 验证结果

```text
python -m pytest -q
.................................. [100%]
37 passed
```

旧 checkpoint 严格加载验证：

```text
missing_keys=[]
unexpected_keys=[]
cn_feature_mode=raw
interaction_mode=unrestricted
```

已使用 Cora HeaRT 候选的 2 个正边 × 5 个官方负例完成评测路径
集成检查，返回 2 行逐边结果，没有调用 Uniform 生成器。

尚未启动任何新训练。旧 MRR 不可作为正式 HeaRT baseline，所以下一
阶段必须先用修复后的官方候选重跑文档规定的最小训练：
Cora、2 个预先固定 seed，比较 A5-raw 与 A5-raw_plus-residual；
interaction 的 disabled 对照只能作为识别性消融，不能根据测试结果选参。

## 2026-09-07 本地实现增补

- `scripts/run_suite.py` 现在会把 suite YAML 中的训练轮数、网络结构、优化器、干预上限以及 `cn_feature_mode`/`interaction_mode` 原样传给 CLI；命令行显式参数优先。
- 新增三个隔离的 phase-2 suite：`phase2_a5_raw`、`phase2_a5_residual`、`phase2_a5_no_interaction`，每个 suite 为 Cora、seed 0/1 的最小可运行矩阵。
- 结果 JSON 和聚合结果显式记录机制模式；旧结果标记为 `legacy_unknown`，不会与新结果静默合并。phase-2 汇总只接受 manifest 明确标记的 phase-2 行。
- 离线部署新增 `--phase2` 和全新的远程目录保护；phase-2 补丁包在 runtime overrides 之后最后覆盖，避免旧文件覆盖新代码。
- 本地已完成一次独立 Smoke 的 A5 `raw_plus_residual + audited` 1+1 epoch 训练、结果落盘和 checkpoint 复载；这只是链路验证，不是正式论文结果。
- 正式 phase-2 服务器训练仍未启动，因此当前摘要状态仍为 `IMPLEMENTED_AND_TESTED_NO_TRAINING`。
