# DCDLP A5 / A14-w010 机制审计报告

- 证据状态：`INCOMPLETE_MISSING`
- 严格配对：AVAILABLE=0, MISSING=6
- 决策：`A14_MECHANISM_NOT_SUPPORTED`
- 干预缓存目录：`D:\桌面\研究生相关事项\CodeWork\DCDLP_work\DCDLP\cache\interventions`（不存在或不匹配时显式记录 GENERATED_EVALUATION/MISSING）

## 1. 可直接复用的功能
- DCDLP.forward 已输出 total/degree/CN/residual/interaction 分数及三类表示。
- 现有 degree/CN 受约束干预、编辑日志和 validator 可直接复用。
- ConditionalCNRegressor、HH/HL/LH/LL 分组、bootstrap 与 Holm 校正已有基础实现。
- 现有逐边预测 CSV 提供正边 rank 和三主分支分数。

## 2. 原先尚未实现或未接入的功能
- 原统一分析入口、score-level routing、interaction bypass、dose-response 未接入。
- 原 post-hoc probe 没有 validation 选参、标准化、Spearman 或严格模型配对。
- 原 cliffs_delta 是非配对笛卡尔比较，不适合本任务的逐边配对效应。
- A8/A9 matching/random-rewire 对照无可复用模型；本实现仅生成冻结模型评测对照。

## 3. 本阶段新增/修改文件
- `scripts/analyze_mechanism_audit.py`
- `src/dcdlp/evaluation/routing_audit.py`
- `src/dcdlp/evaluation/posthoc_probe.py`
- `src/dcdlp/evaluation/dose_response.py`
- `tests/test_routing_audit.py`
- `tests/test_posthoc_probe.py`

## 4. 数据与统计风险
- A14-w010 来源于查看测试 MRR 后的探索性权重选择，不能作为独立确认性结论。
- HeaRT 固定划分对不同 model seed 复用 seed-0 数据；seed 是训练随机性而非独立数据划分。
- 缓存 pair_id 若仅为位置索引，脱离 dataset/split/seed 后可能错配；审计同时核验 u/v/type/magnitude。
- 逐边比较只允许相同 dataset/seed/protocol/u/v/intervention/control；任何缺项均标 MISSING。
- 官方 HeaRT/OGB 候选不可用时不得用 uniform 候选替代；机制审计不会生成替代排名。

## 5. 运行配对

| dataset | seed | protocol_train | protocol_eval | status | reason | prediction_pair_status | a5_config_hash | a14_w010_config_hash | a5_checkpoint | a14_w010_checkpoint | a5_predictions | a14_w010_predictions | a5_mrr | a14_w010_mrr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| citeseer | 0 | uniform | heart | MISSING | A5 missing, A14-w010 missing | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| citeseer | 1 | uniform | heart | MISSING | A5 missing, A14-w010 missing | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| citeseer | 2 | uniform | heart | MISSING | A5 missing, A14-w010 missing | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| cora | 0 | uniform | heart | MISSING | A5 missing, A14-w010 missing | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| cora | 1 | uniform | heart | MISSING | A5 missing, A14-w010 missing | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| cora | 2 | uniform | heart | MISSING | A5 missing, A14-w010 missing | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

## 6. 验收问题

1. **Degree 路由选择性是否提高？**
   - `{"met": false, "status": "MISSING", "mean_deltas": {"degree": null, "cn": null}}`
2. **CN 路由选择性是否提高？**
   - `{"met": false, "status": "MISSING", "mean_deltas": {"degree": null, "cn": null}}`
3. **交叉分支敏感性是否降低？**
   - `{"met": false, "status": "MISSING", "mean_deltas": {"degree_to_cn": null, "degree_to_residual": null, "cn_to_degree": null, "cn_to_residual": null}}`
4. **post-hoc 泄漏是否降低？**
   - `{"met": false, "status": "MISSING", "improved_probe_count": 0, "paired_probe_count": 0}`
5. **目标机制信息是否保持？**
   - `"见 posthoc_probe.csv 的 target_retention；缺失则为 MISSING"`
6. **interaction 是否成为绕行通道？**
   - `{"met": false, "status": "MISSING", "max_interaction_share": "MISSING", "paired_mean_delta": null}`
7. **是否优于 random-rewire？**
   - `"见 statistical_tests.csv；缺失配对不推断"`
8. **是否跨 dataset/seed 稳定？**
   - `"可用严格运行配对=0，缺失=6"`
9. **运行成本是否有合理回报？**
   - `"见 efficiency.csv 与 Pareto 图；机制条件未满足则无合理回报"`
10. **最终建议**
   - `"A14_MECHANISM_NOT_SUPPORTED"`

## 7. 决策依据

```json
{
  "performance_parity": {
    "met": false,
    "status": "MISSING"
  },
  "routing": {
    "met": false,
    "status": "MISSING",
    "mean_deltas": {
      "degree": null,
      "cn": null
    }
  },
  "cross_sensitivity": {
    "met": false,
    "status": "MISSING",
    "mean_deltas": {
      "degree_to_cn": null,
      "degree_to_residual": null,
      "cn_to_degree": null,
      "cn_to_residual": null
    }
  },
  "posthoc_leakage": {
    "met": false,
    "status": "MISSING",
    "improved_probe_count": 0,
    "paired_probe_count": 0
  },
  "target_retention": {
    "met": false,
    "status": "MISSING",
    "retained_count": 0,
    "paired_probe_count": 0
  },
  "interaction_bypass": {
    "met": false,
    "status": "MISSING",
    "max_interaction_share": "MISSING",
    "paired_mean_delta": null
  },
  "random_rewire_control": {
    "met": false,
    "status": "MISSING",
    "a14_advantage_constrained": null,
    "a14_advantage_random_rewire": null
  },
  "cross_dataset_seed_stability": {
    "met": false,
    "status": "MISSING",
    "nonnegative_fraction": "MISSING"
  }
}
```

若证据状态为 `INCOMPLETE_MISSING`，报告不会用默认 A14、替代分数或未配对样本填补缺口。
