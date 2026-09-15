# Phase-2 Selective Routing 工程实现报告（Step 10）

日期：2026-09-13  
结论范围：仅说明工程实现、测试和 Smoke 链路完成；**不据此声称模型有效、机制成立或统计显著**。

## 1. 修改完成情况

| Step | 状态 | 结果 |
|---|---|---|
| Step 1 代码审计 | DONE | 已生成 `IMPLEMENTATION_AUDIT.md`。 |
| Step 2 三种 interaction mode | DONE | `disabled/audited/unrestricted` 已具有不同 loss 合同；disabled score/share 精确为 0 且参数无梯度更新。 |
| Step 3 三种 CN mode | DONE | `raw/residual/raw_plus_residual` 已采用等宽 selective 输入；纯 residual 不再显式输入 raw/normalized CN；旧 checkpoint 走 legacy schema。 |
| Step 4 score-level routing | DONE | 已实现 CN/degree 方向 loss、interaction cross weight 和 target-response floor，并接入 Stage B。 |
| Step 5 prediction/intervention output | DONE | 已补齐逐边 prediction 字段和干预 before/after/delta/coverage CSV。 |
| Step 6 独立 post-hoc probe | DONE | 已实现冻结 checkpoint 后独立训练五类 probe；validation 选参，test 只评分。 |
| Step 7 泛化 Phase-2 audit | DONE | 新脚本无 A5/A14 模型名硬编码，支持 baseline/candidate CLI、严格配对、统计与四态决策。 |
| Step 8 完整 pytest | DONE | 54 passed，0 failed，2 warnings。 |
| Step 9 Smoke | DONE | M0/M1/M3 各 1 seed 的 train/checkpoint/reload/evaluate/audit/probe/aggregate 全链路完成。 |
| Step 10 修改报告 | DONE | 本文件。 |
| Step 11 Cora 5-seed | NOT DONE | 必须在本报告之后执行；尚未启动。 |
| Step 12 机制审计与决策 | NOT DONE | 等待 Step 11 的真实产物。 |

## 2. 修改文件列表

### 核心源码

- `src/dcdlp/utils.py`
  - 新增完整 array dtype/shape/bytes hash，用于 split、candidate 和 residualizer 数据审计。
  - 只新增工具函数，不改变历史 hash 接口。
- `src/dcdlp/data/pair_statistics.py`
  - residualizer 记录类型、完整 estimator 参数、fit seed、样本数和调用方数据元信息。
  - 原有 `fit(stats)` 调用仍兼容。
- `src/dcdlp/models/branches.py`
  - 新增 `selective_v1` 固定两维显式 CN 输入：raw=`[raw,0]`，residual=`[residual,0]`，raw_plus_residual=`[raw,residual]`。
  - 保留 `legacy` schema 恢复旧 mode-dependent state-dict shape。
- `src/dcdlp/models/dcdlp.py`
  - 输出规范定义的 prediction-level `interaction_share`。
  - 保留旧 interaction 参数命名和 state dict；disabled 仍精确置零。
- `src/dcdlp/models/losses.py`
  - 新增 interaction share/delta penalty。
  - 新增直接基于四类 branch score delta 的 CN/degree routing loss 和 target floor。
  - 旧 latent `intervention_losses()` 保留供 A0–A14 对照。
- `src/dcdlp/train.py`
  - 新增 train-only residualizer 拟合入口及 held-out pair 守卫。
  - 新增 selective routing/interaction 配置与 Stage B `frozen/low_lr` encoder 策略。
  - `lambda_link` 被强制为正，避免关闭主任务作弊。
  - 新增 split/candidate/intervention/参数量/loss/device metadata。
  - 新增隔离 prediction 子目录选项；默认仍为旧 `raw` 路径，保持兼容。
- `src/dcdlp/evaluate.py`
  - 提取可复用 checkpoint loader，旧 checkpoint 缺 schema 时使用 `legacy`。
  - 干预审计输出合法 pair 的 before/after/delta、routing、interaction 和 coverage；失败保留 code。
  - official candidate 缺失时直接失败，不再返回可被误当结果的 Uniform/普通评分。
- `src/dcdlp/evaluation/posthoc_probe.py`
  - probe 结果增加 architecture、候选超参数和 seed metadata。
  - 原 probe API 保持兼容。
- `src/dcdlp/cli.py`
  - 增加全部新配置、布尔解析、输出目录和评估产物目录参数。

### 脚本与配置

- `scripts/run_posthoc_leakage_audit.py`
  - 新增冻结 checkpoint 的五类独立 probe 流水线及 per-seed/summary/prediction/metadata 产物。
- `scripts/analyze_phase2_selective_routing.py`
  - 新增通用 `--baseline-run/--candidate-run/--output-dir` 审计。
  - 校验 dataset/seed/protocol/split/test-positive/candidate hash；干预只在合法交集上比较并报告覆盖率。
  - 生成规范要求的六个统计/决策文件。
- `scripts/run_phase2_selective_pipeline.py`
  - 新增隔离 M0–M4 pipeline；当前 Smoke 仅执行 M0/M1/M3。
  - 发现同 config 结果已存在时默认拒绝覆盖；只有显式 `--resume` 才复用。
  - 每个失败保存 failure code、message 和 traceback。
- `scripts/run_suite.py`
  - 转发新配置和独立 output root；识别 M0–M4，同时保留旧 suite/model 路径。
- `scripts/aggregate_results.py`
  - 新增可指定结果根目录的聚合函数/CLI；默认仍为历史 `results/`。
- `configs/model/dcdlp.yaml`
  - 增加新 interaction/routing/CN schema 安全默认占位；未写入论文最终权重。
- `configs/suite/phase2_selective_smoke.yaml`
  - 新增隔离 Smoke 配置。

### 审计与测试

- `IMPLEMENTATION_AUDIT.md`
  - Step 1 实际代码审计及修改映射。
- 新增：
  - `tests/test_interaction_modes.py`
  - `tests/test_cn_feature_modes.py`
  - `tests/test_checkpoint_roundtrip.py`
  - `tests/test_score_routing_loss.py`
  - `tests/test_prediction_outputs.py`
  - `tests/test_independent_posthoc_audit.py`
  - `tests/test_selective_audit_script.py`
- 调整：
  - `tests/test_branch_shapes.py` 的 residualizer 校准数据改为 train-only。

所有修改均保留旧 A0–A14 profile、旧 loss、旧 `analyze_mechanism_audit.py` 和旧结果读取能力。未修改 `results/raw/`、旧 `results/phase2/`、旧 `results/mechanism_audit/` 或外层 `server_audit_results/`。

## 3. 新配置

新增/正式化的主要字段：

```text
cn_feature_mode
cn_input_schema
interaction_mode
interaction_share_cap
interaction_delta_cap
lambda_interaction_share
lambda_interaction_delta
lambda_link
score_routing_enabled
lambda_score_route
lambda_target_response
routing_margin_cn
routing_margin_degree
target_floor_cn
target_floor_degree
interaction_route_weight
routing_finetune_encoder
encoder_lr_multiplier
output_dir
prediction_subdir
```

Smoke 使用预先写入配置的工程验证权重，没有基于 test MRR 搜索或修改权重。

## 4. Pytest

执行命令：

```text
.venv/Scripts/python.exe -m pytest
```

结果：

```text
54 passed, 0 failed, 2 warnings in 10.85s
```

两个 warning 来自审计脚本的单 seed 人工测试中 SciPy 方差自由度不足；测试结果与实际 5-seed 统计不受该单元测试 warning 影响。

覆盖的新增关键合同：

- audited interaction penalty > 0；unrestricted 不施加该 penalty；
- disabled interaction score/share 为 0、参数无梯度且 optimizer 后不变；
- 三种 selective CN mode 的 CN branch 参数量一致；
- residual 显式输入不含 raw/normalized CN；
- residualizer 拒绝 valid/test pair 污染；
- checkpoint 恢复 residualizer、CN/interaction/routing config；
- 正确 routing loss 低于错误 routing；
- 全零响应在正 target floor 下 loss > 0；
- prediction 与 intervention schema 完整；
- 五类独立 probe 均运行且模型冻结；
- audit 拒绝 candidate hash 不一致配对；
- official candidate 缺失直接失败。

## 5. Smoke

隔离根目录：`results/phase2_selective/`

| 模式 | train | checkpoint | reload/evaluate | intervention audit | probe | aggregate |
|---|---|---|---|---|---|---|
| M0 raw + audited | 成功 | 成功 | 成功 | 成功 | 成功 | 成功 |
| M1 residual + audited | 成功 | 成功 | 成功 | 成功 | 成功 | 成功 |
| M3 residual + disabled | 成功 | 成功 | 成功 | 成功 | 成功 | 成功 |

一致性核查：

- 三个 run 的 test candidate hash 均为 `7fb769b17dc02b40dd7bda059c63a82cdb41f3351cf7e44bd579236d24cfd050`；
- 每个 run 输出 9 条 test-positive prediction；
- 每个 run 输出 15 条 probe 结果（5 mappings × 3 probe seeds）；
- 每个 run 的 degree 干预 9/9 有效；
- 每个 run 的 CN 干预 7/9 有效，另 2 个明确记录 `NO_CANDIDATE`，未作为零响应进入统计；
- `aggregate/all_runs.csv` 与 `aggregate/summary.csv` 已生成。

Smoke MRR 仅作为链路执行记录，不用于效果主张：M0=`0.1696689`，M1=`0.1659652`，M3=`0.1659652`。

首次 Smoke 启动在训练前因直接脚本导入路径失败；已修复双模式导入并保存 `logs/smoke_launcher_failure_20260913.log`。该失败没有生成或替代任何模型分数。

## 6. 未解决问题

- 当前目录没有 `.git`，run metadata 的 commit 只能记录为 `untracked`；需要在有 Git 元数据的正式运行副本中获得 commit hash。
- M4 unrestricted interaction 是资源允许时的探索对照，Smoke 最低要求未要求，本步未运行；因此 audited 与 unrestricted 的经验 share 比较尚无数据。
- Cora 5-seed 尚未执行，不能判断 prediction/routing/target/leakage 门槛。
- 现有环境是否具备完整 Cora HeaRT 官方文件需在 Step 11 启动前做只读 preflight；若缺失必须失败，不能 fallback Uniform。
- 5-seed 只能用于路线筛选，后续报告不得将单个 t-test p 值解释为确认性显著。

## 7. 工程结论

截至 Step 10，只能声明：修正版 selective-routing 的工程实现、单元/回归测试和三个最小 Smoke 全链路已经完成。是否继续到多数据集、转向 calibration 或停止，必须等待 Step 11–12 的真实 Cora 5-seed 配对审计。

