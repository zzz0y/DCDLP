# DCDLP 修正版实现审计（Step 1）

审计日期：2026-09-13  
审计依据：`DCDLP_修正版_Codex工程修改规范_2026-09-11.md`  
审计范围：当前 `DCDLP/` 目录内的实际源码、配置、脚本和测试。此目录没有 `.git` 元数据，因此当前只能将版本标识记为 `untracked`，不能用 `git status` 区分用户既有改动。

## 1. 总结

当前工程已经具备较好的旧路线基础：数据 split/负样本安全检查、目标边 message-passing mask、conditional-CN 回归器、三种 CN mode 的初步接口、三种 interaction mode 的初步接口、CN/degree 受约束干预及 validator、官方 grouped candidate 拒绝 Uniform fallback、基础逐边预测、独立线性 probe 工具和旧 A5/A14 机制审计框架。

但修正版主链路尚未完成。最关键缺口是：

- `audited` 与 `unrestricted` interaction 当前前向行为完全相同，没有 share/delta penalty；
- 训练期 routing 仍基于 latent representation 距离，不是 score-level branch response；
- `residual` CN mode 仍显式加入 degree-normalized raw CN，违反“raw CN 不得偷偷重新进入 residual 输入”的要求；
- 两阶段训练没有 `frozen` / `low_lr` encoder 策略；
- checkpoint、run metadata、逐边/干预输出和独立 post-hoc audit 的字段与可追溯性不完整；
- 旧机制审计脚本硬编码 `A5` 与 `A14-w010`，不能用于本轮；
- 新的隔离结果根目录、M0–M4 suite、Smoke 全链路和 Step 12 决策产物尚不存在。

## 2. 已经存在且可优先复用的功能

### 2.1 数据与评估安全边界

- `src/dcdlp/data/loaders.py`
  - `GraphDataset.validate()` 检查静态数据 train/valid/test 正边互斥、自环、节点范围和 `all_positive` 一致性；
  - `train_graph()` 只由 `train_pos` 构建；
  - HeaRT 小数据集 loader 要求官方正边、特征和 grouped candidates 全部存在，否则抛出 `FileNotFoundError`；
  - HeaRT valid/test 正边保留原始行顺序，以维持 grouped candidate 对齐。
- `src/dcdlp/models/node_encoder.py`
  - `mask_pair_edges()` 在编码前移除当前评分目标 pair。
- `src/dcdlp/data/negative_sampling.py`
  - Uniform/degree-corrected 负样本排除传入的全部真实正边并排除自环。
- `src/dcdlp/train.py`
  - `grouped_official_negatives()` 在 candidate 缺失、形状错误或包含对应正边时显式失败，不生成 Uniform 替代；
  - official candidate 保留行对齐和候选内容。
- `tests/test_split_leakage.py`、`tests/test_negative_sampling.py`、`tests/test_metric_formulas.py`
  - 已覆盖多项 split、负样本和官方 candidate no-fallback 行为。

### 2.2 受约束干预

- `src/dcdlp/interventions/cn_intervention.py`
  - 使用 degree-preserving 2-switch；支持 `+/-1` 以及迭代得到 `+/-2`；调用 validator。
- `src/dcdlp/interventions/degree_intervention.py`
  - 改变指定端点 degree，同时保持目标 pair 的完整 CN set；调用 validator。
- `src/dcdlp/interventions/validator.py`
  - 检查 simple graph、自环、目标边、held-out forbidden edge、编辑日志与 graph diff、端点隔离、完整 degree vector、完整 CN set 和编辑数。
- `src/dcdlp/interventions/cache.py`
  - 失败保留真实 failure code，不生成替代干预。
- `tests/test_cn_intervention.py`、`tests/test_degree_intervention.py`
  - 已覆盖基本 `+/-` 合法性。

### 2.3 Conditional CN 与三种 CN mode 的基础

- `src/dcdlp/data/pair_statistics.py`
  - `ConditionalCNRegressor` 用 `log1p(CN)` 拟合 degree 条件输入；支持端点 degree 对称化；
  - `residual()` 返回 expected 与 `log1p(CN)-expected`。
- `src/dcdlp/train.py`
  - 非 raw 模式使用 `train_pos + 固定 seed 的 train negatives` 拟合 residualizer；
  - checkpoint 已保存 residualizer 对象和一部分 metadata。
- `src/dcdlp/models/branches.py`
  - 接受 `raw`、`residual`、`raw_plus_residual` 三种字符串并保持相同 branch output dimension。

### 2.4 Interaction mode 的基础接口

- `src/dcdlp/models/dcdlp.py`
  - 接受 `disabled`、`audited`、`unrestricted`；
  - `disabled` 将 `score_interaction` 精确置零，并关闭 interaction matrix/scale 的梯度；
  - 输出已有 `score_interaction`。
- `tests/test_branch_shapes.py`
  - 已验证 disabled 的 score 为零及总分差异。

### 2.5 预测、群体指标与基础审计工具

- `src/dcdlp/train.py::evaluate_split()`
  - 输出 MRR、Hits@10/20/50/100、AUC、AP、Macro/Worst-Group MRR、Group Gap 及 HH/HL/LH/LL；
  - 正边逐边 CSV 已含总分和四类 branch score，以及部分 CN 统计。
- `src/dcdlp/evaluation/routing_audit.py`
  - `branch_response()` 已能从同一 pair 的 before/after score 计算各 branch delta、target delta、non-target delta、routing selectivity 和 interaction delta share；
  - 已有 bootstrap 摘要与 paired effect 基础。
- `src/dcdlp/evaluation/posthoc_probe.py`
  - probe 超参数只在 validation 选择，test 只用于最终评分；支持 degree-bin 分类和 residual 回归所需的通用工具。
- `src/dcdlp/evaluation/statistics.py`
  - 已有 paired bootstrap CI、Cliff's delta、Holm–Bonferroni 基础函数。

### 2.6 向后兼容基础

- `src/dcdlp/train.py::ablation_profile()` 仍保留 A0–A14 配置入口；
- `scripts/aggregate_results.py` 将缺少新字段的历史 JSON 标为 `legacy_unknown`，避免与现有 Phase-2 mode 静默混合；
- 现有 `results/`、外层 `server_audit_results/` 和旧 checkpoint 均未在 Step 1 中写入或修改。

## 3. 部分存在，但需修复或扩展

### 3.1 Interaction 三模式

- `disabled` 基本满足 score/gradient 要求，但模型仍注册 interaction 参数；其 `requires_grad=False`，需要新增测试证明 optimizer step 后参数不变。
- `audited` 和 `unrestricted` 当前前向、参数和损失完全相同；现有测试反而断言两者相同。
- 模型未直接输出规范定义的 prediction-level `interaction_share`。
- 没有 `share_cap`、`delta_cap`、`lambda_share`、`lambda_delta` 配置与 loss。
- 没有 cap violation rate、mean/median/p90 share 审计。

### 3.2 CN feature mode

- 三种 mode 都能构造合法 shape，但参数量并不完全一致：显式输入分别为 1/2/3 维，第一层参数量不同；需要 input projection 或固定宽度输入以控制预算。
- `residual` 当前输入为 `[CN_residual, normalized_CN]`；`normalized_CN` 仍由 raw CN 直接计算，因此不满足 residual 主模式禁止 raw CN 偷渡的要求。
- `raw_plus_residual` 当前输入为 `[raw_log_cn, residual, normalized_CN]`，可以保留但应明确各字段语义并控制参数量。
- 现有 `test_raw_plus_residual_features_are_explicit_and_symmetric()` 用 `train_pos + valid_pos` 拟合回归器；虽是模型 shape 测试，但与本轮防泄漏测试目标冲突，应改为只用 train 来源。
- residualizer metadata 只有 estimator、split 标记、seed、样本数等，缺 estimator 参数、数据来源标识、数据/config hash。
- evaluation 分组阶段会重新拟合另一个 residualizer，并在异常时退化为均值 residual；这与 checkpoint 中模型 residualizer 不是同一审计对象，需统一并明确 fallback 只用于非模型审计还是直接失败。

### 3.3 Routing 与训练

- `src/dcdlp/models/losses.py::intervention_losses()` 使用 `z_*` 的向量距离；不是规范要求的 `score_*_after-score_*_before`。
- loss 未纳入 `delta_score_interaction`，也没有 target-response floor，因此全零响应仍可能只由旧 margin 间接处理，语义不符合新规范。
- `TrainConfig` 只有旧 `lambda_inv/lambda_route/lambda_adv/lambda_orth/lambda_degrob`，没有新的 routing/interaction 配置字段。
- Stage A/Stage B 只有 epoch 边界；没有 encoder `frozen` 或 `low_lr` param group，也未记录策略。
- A14 profile 在 Stage B 同时启用 orthogonal、adversarial、intervention，与新正式路线“score routing 为主，旧 loss 默认关闭/极小”不一致。
- 训练干预是在线生成且只累计 failure code；没有 attempted/valid/missing/valid_rate 和稳定 cache id/hash。

### 3.4 Prediction 与 intervention 输出

- 正边逐边 CSV 已有 `score_total/degree/cn/residual/interaction`，但缺规范定义的 `interaction_share`、明确的 `cn_raw`、`degree_u`、`degree_v`；当前使用 `cn` 和 `degree_score`。
- 模型内部统计名为 `cn_residual_feature`，CSV 同时还有另一个审计 residual，容易混淆。
- 只写 test positive 行，负候选不在逐边输出中；是否需要纳入需在 Step 5 统一定义。
- `_intervention_evaluation()` 只输出聚合 latent routing/cross 指标；未保存 per-pair before/after/delta score，未报告 interaction share/cap violation/failure coverage。
- `_intervention_evaluation()` 的 counterfactual edge tensor 固定在 CPU；当前 checkpoint evaluation 本身是 CPU，但接口不具备通用 device 安全性。

### 3.5 Post-hoc leakage audit

- 通用 classification/regression probe 已存在，且与训练 adversary 是独立实现。
- 目前没有从冻结 checkpoint 提取固定 train/valid/test pair representations 并运行规范要求的五个 probe 的正式 CLI/流水线。
- 旧 `scripts/analyze_mechanism_audit.py` 内有 probe 路径，但与 A5/A14-w010 审计耦合，不能直接复用作本轮最终证据。
- probe metadata 尚未完整保存 architecture、候选超参数、seed、split/hash、均值与标准差。

### 3.6 Suite、结果隔离与 run metadata

- `scripts/run_suite.py` 能从 YAML 转发少数模型/训练字段，支持 seed/dataset 子集和 dry-run。
- runner 固定写 `results/manifest.csv`、`results/logs/`，train 默认写 `results/`；没有本轮专用且由 suite 控制的 `results/phase2_selective/` 根目录。
- manifest identity 不含 cn mode、interaction mode、routing config 或 config hash，多个新模型矩阵可能冲突/误跳过。
- run JSON 缺 split hash、candidate hash、intervention cache hash、参数量、完整 loss weights、routing enabled/strategy 等。
- 当前 Phase-2 YAML 的 residual suite 实际设置为 `raw_plus_residual`，并非规范 M1 的纯 `residual`；no-interaction suite 同样是 `raw_plus_residual`，与规范 M3 不一致。
- `configs/model/dcdlp.yaml` 默认仍是旧 unrestricted/raw 与旧强 loss 结构。

## 4. 与规范明确不一致或尚不存在的功能

1. 没有真正独立的 audited interaction 行为及 penalty。
2. 没有新 score-level routing loss、target floor、interaction route weight 和对应方向/zero-cheat 测试。
3. 没有 encoder frozen/low_lr 的 Stage B 策略。
4. 没有完整 checkpoint round-trip 审计（residualizer + 所有新 config）。
5. 缺 official candidate 缺失时 `evaluate_checkpoint()` 的失败语义：当前返回 `NOT_AVAILABLE`，而正式 HeaRT/OGB 要求直接失败；Step 3/8 回归测试需固定最终合同。
6. 没有本轮独立 post-hoc probe 执行入口和规范五类 probe 产物。
7. `scripts/analyze_mechanism_audit.py` 明确硬编码 `MODEL_A5="A5"`、`MODEL_A14="A14-w010"`，CLI 还强制模型集合恰好为这两者。
8. 没有支持 `--baseline-run/--candidate-run/--output-dir` 的新 Phase-2 selective audit。
9. 没有按 seed/聚合输出规范第 11 节全部指标的实现。
10. 没有本轮 M0/M1/M2/M3（可选 M4）suite 和三个 Smoke mode 的全链路编排。
11. 没有 Step 12 要求的六个 selective 统计/决策文件。
12. 没有冻结后的 GO/CONDITIONAL_GO/PIVOT/STOP 机器判定实现。

## 5. 预计修改/新增文件（按 Step 2–12）

### Step 2：interaction

- 修改 `src/dcdlp/models/dcdlp.py`
- 修改 `src/dcdlp/models/losses.py`
- 修改 `src/dcdlp/train.py`
- 修改 `src/dcdlp/cli.py`
- 修改 `configs/model/dcdlp.yaml`
- 新增/扩展 `tests/test_interaction_modes.py`

### Step 3：CN mode 与 checkpoint

- 修改 `src/dcdlp/data/pair_statistics.py`
- 修改 `src/dcdlp/models/branches.py`
- 修改 `src/dcdlp/train.py`
- 修改 `src/dcdlp/evaluate.py`
- 新增 `tests/test_cn_feature_modes.py`
- 新增 `tests/test_checkpoint_roundtrip.py`
- 调整 `tests/test_branch_shapes.py`

### Step 4：score-level routing

- 修改 `src/dcdlp/models/losses.py`
- 修改 `src/dcdlp/train.py`
- 新增 `tests/test_score_routing_loss.py`

### Step 5：prediction/intervention output

- 修改 `src/dcdlp/models/dcdlp.py`
- 修改 `src/dcdlp/train.py`
- 修改 `src/dcdlp/evaluate.py`
- 修改 `src/dcdlp/evaluation/intervention_metrics.py` 或复用/扩展 `routing_audit.py`
- 新增对应输出 schema 测试

### Step 6：独立 probe

- 扩展 `src/dcdlp/evaluation/posthoc_probe.py`
- 新增独立执行模块/脚本（预计 `scripts/run_posthoc_leakage_audit.py`）
- 扩展 `tests/test_posthoc_probe.py`

### Step 7：Phase-2 selective audit

- 新增 `scripts/analyze_phase2_selective_routing.py`
- 复用并按需扩展 `src/dcdlp/evaluation/routing_audit.py`
- 不修改旧 `scripts/analyze_mechanism_audit.py` 的历史用途
- 新增 audit pairing/statistics/decision 测试

### Step 8–12：suite、隔离输出、Smoke、报告和 Cora 5-seed

- 修改 `scripts/run_suite.py` 以支持独立 output root、新字段与安全 manifest identity
- 新增 `configs/suite/phase2_selective_smoke_*.yaml` 或单一显式矩阵配置
- 新增 M0–M3（可选 M4）Cora suite 配置
- 新增 `results/phase2_selective/` 目录结构（只写新路径）
- 新增 `PHASE2_SELECTIVE_ROUTING_IMPLEMENTATION_REPORT.md`
- Step 12 由新 audit 脚本生成指定 CSV/JSON/MD

## 6. 实施约束

- 不改写或删除 `results/raw/`、`results/phase2/`、`results/mechanism_audit/`、外层 `server_audit_results/` 以及已有 checkpoint。
- 保留 A0–A14 和旧 CLI 字段；新字段提供安全默认值，旧 checkpoint loader 使用向后兼容默认。
- 本轮所有新运行写入 `results/phase2_selective/`。
- Step 8 完整 pytest 未全部通过前不执行 Step 9；三个 Smoke 全链路未全部通过前不执行 Cora 5-seed。
- 不运行 1195 全量任务，不扩展 CiteSeer/PubMed/OGB，不用 test MRR 搜索 loss 权重。

