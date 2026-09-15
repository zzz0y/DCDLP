# Experiment execution status

Delivery date: 2026-09-07 (Asia/Shanghai)

## Completed locally

- Project code compiles under Python 3.11.
- All 37 deterministic/unit tests pass, including the phase-2 configuration and legacy-result isolation checks.
- The smoke dataset was prepared with disjoint train/validation/test positives.
- Native degree-corrected negatives were generated without true-edge leakage.
- Conditional-CN statistics and quadrant assignments were generated.
- 32 constrained intervention attempts were audited: 26 valid, 6 explicit infeasible cases.
- The two-stage DCDLP smoke run completed on CPU (one pretraining epoch and one disentanglement epoch).
- The run emitted a strict JSON result, checkpoint, per-positive prediction CSV, manifest entry, aggregate CSV, LaTeX table, and PDF/SVG figures.
- Checkpoint reload and Standard, degree-corrected, and controlled-intervention evaluation were exercised.
- The phase-2 conditional-CN route is implemented locally: suite settings reach the CLI, raw/residual/no-interaction suites are isolated, and official HeaRT evaluation cannot silently fall back to Uniform candidates.
- A separate 1+1 epoch Smoke run for A5 `raw_plus_residual + audited` completed with result and checkpoint reload validation. It is a chain test, not a paper result.

The smoke run is an execution/integrity test, not a paper-quality claim. Its numerical score is intentionally retained rather than selected or polished.

## Not executed locally

The five real-dataset, 5--10 seed paper matrix and the new phase-2 server pilot have not been launched. Official baseline adapters for SEAL/BUDDY/NCN/NCNC/Neo-GNN/LTLP are still not complete. The launcher records unavailable jobs as NOT_RUN or FAILED and never substitutes uniform candidates or synthetic numbers for an unavailable official protocol.

Use run_all_experiments.ps1 in the full environment. Inspect results/manifest.csv after every run; a paper table is complete only when every required row is COMPLETED and points to a real JSON result/checkpoint.

## Verified commands

    python -m pytest -q
    python -m dcdlp.cli prepare-data --dataset smoke --protocol standard --seed 0
    python -m dcdlp.cli build-negatives --dataset smoke --method degree_corrected --split test --per-positive 10 --seed 0
    python -m dcdlp.cli build-pair-stats --dataset smoke --protocol standard --seed 0
    python -m dcdlp.cli build-interventions --dataset smoke --protocol standard --split test --types cn_plus,cn_minus,degree_u_plus,degree_u_minus --max-pairs 8 --seed 0
    python scripts/run_suite.py --suite smoke
    python scripts/aggregate_results.py
    python scripts/make_paper_figures.py
