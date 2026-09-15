# DCDLP experiment implementation

This directory is the executable companion to DCDLP_work0_Codex.docx. It implements the leakage-safe graph split representation, uniform and native degree-corrected negatives, conditional-CN residuals and quadrants, constrained CN/degree interventions with edit logs and validators, the symmetric three-branch DCDLP model, two-stage training, ranking/group/intervention metrics, deterministic tests, suite manifests, aggregation, and PDF/SVG figure generation.

The code never writes made-up scores. Missing official repositories, failed jobs, and jobs not launched are preserved as FAILED or NOT_RUN in results/manifest.csv.

## Datasets and downloads

The repository tracks code, configuration, documentation, tests, and the existing
experiment snapshot in `results/`. Downloaded datasets, generated caches, Python
environments, and the external repositories under `third_party/` are intentionally
excluded because several individual files are hundreds of megabytes or larger.

The paper suites use these datasets:

| Dataset | Experiments | Official download/documentation |
| --- | --- | --- |
| Cora | `paper_main`, `paper_ablation`, and the phase-2 A5 pilot (standard and HeaRT protocols) | [PyG Planetoid documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html); HeaRT files via [HeaRT](https://github.com/Juanhui28/HeaRT) |
| CiteSeer | `paper_main`, `paper_ablation` (standard and HeaRT protocols) | [PyG Planetoid documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html); HeaRT files via [HeaRT](https://github.com/Juanhui28/HeaRT) |
| PubMed | `paper_main`, `paper_ablation` (standard and HeaRT protocols) | [PyG Planetoid documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html); HeaRT files via [HeaRT](https://github.com/Juanhui28/HeaRT) |
| `ogbl-collab` | `paper_main`, `paper_ablation` (OGB official temporal split) | [OGB link-property datasets](https://ogb.stanford.edu/docs/linkprop/) |
| `ogbl-ddi` | `paper_main`, `paper_ablation` (OGB official protein-target split) | [OGB link-property datasets](https://ogb.stanford.edu/docs/linkprop/) |
| `synthetic-*` | `paper_analysis` mechanism experiments | Built into `src/dcdlp/data/synthetic.py`; no download |

The repository also contains an `ogbl-ppa` dataset configuration for optional
experiments; it is documented and downloaded through the same [OGB link-property
page](https://ogb.stanford.edu/docs/linkprop/). `smoke` and all
`synthetic-*` experiments are generated locally and need no external download.

To fetch and prepare the data used by the full suites:

```powershell
# PyG Planetoid and OGB downloads (requires the dependencies in environment.yml)
python scripts/download_data.py --datasets cora citeseer pubmed ogbl-collab ogbl-ddi

# HeaRT's fixed Cora/CiteSeer/PubMed splits and negative candidates.
# Run from Git Bash or WSL on Windows; the files are several GB in total.
python scripts/download_data.py --third-party
bash third_party/HeaRT/download_data.sh
python -m dcdlp.cli prepare-data --dataset cora --protocol heart --seed 0
python -m dcdlp.cli prepare-data --dataset citeseer --protocol heart --seed 0
python -m dcdlp.cli prepare-data --dataset pubmed --protocol heart --seed 0
```

The standard Planetoid and OGB artifacts are downloaded automatically by
`prepare-data` if they are absent. Keep them under `data/`; `.gitignore` prevents
them from being committed. See `configs/dataset/` and `configs/suite/` for the
exact protocol and seed matrix.

## Quick smoke test (Windows PowerShell)

From this directory, create the environment and run:

    conda env create -f environment.yml
    conda activate dcdlp
    pip install -e ".[full,test]"
    .\run_smoke_test.ps1

For the lightweight local venv used during delivery:

    .\.venv\Scripts\Activate.ps1
    .\run_smoke_test.ps1

The smoke suite uses a deterministic built-in graph, executes all unit tests, performs one LP-pretraining epoch plus one intervention-disentanglement epoch, writes a checkpoint and per-pair predictions, aggregates real results, and builds vector figures.

## GPU verification (Windows PowerShell)

The GPU environment uses PyTorch 2.3.1 with the bundled CUDA 12.1 runtime and CUDA-enabled PyG extensions. A newer NVIDIA driver-reported CUDA version is compatible with this runtime; the standalone CUDA Toolkit and `nvcc` are not required for the prebuilt wheels.

Run the complete CUDA, PyG, regression-test, and DCDLP training verification with:

    .\run_gpu_smoke_test.ps1 -Gpu 0

This command fails immediately if a required CUDA component is unavailable. On success it prints `status: PASS`, runs all tests, performs a real two-stage DCDLP GPU training job, and writes its checkpoint and predictions under `results/`.

## Required command surface

    python -m dcdlp.cli prepare-data --dataset cora --protocol standard
    python -m dcdlp.cli prepare-data --dataset ogbl-collab --protocol ogb
    python -m dcdlp.cli build-negatives --dataset cora --method degree_corrected --split test --seed 0
    python -m dcdlp.cli build-pair-stats --dataset cora --protocol standard
    python -m dcdlp.cli build-interventions --dataset cora --split train --types cn_plus,cn_minus,degree_u_plus,degree_u_minus --max-pairs 100000 --seed 0
    python -m dcdlp.cli train --dataset cora --protocol standard --protocol-train uniform --seed 0
    python -m dcdlp.cli evaluate --checkpoint results/checkpoints/FILE.pt --protocols standard,heart,degree_corrected,intervention
    python scripts/run_suite.py --suite paper_main --gpus 0,1 --continue-on-error
    python scripts/run_suite.py --suite paper_ablation --gpus 0,1 --continue-on-error
    python scripts/run_suite.py --suite paper_analysis --gpus 0,1 --continue-on-error
    python scripts/aggregate_results.py
    python scripts/make_paper_figures.py

The equivalent Hydra-style train tokens are accepted, for example:

    python -m dcdlp.cli train dataset=cora protocol=heart seed=0

## Phase-2 pilot

The post-audit pilot is intentionally separate from the legacy `paper_main`
and `paper_ablation` suites. It compares A5 with raw CN features,
conditional-CN residual features, and an interaction-disabled identification
control on Cora seeds 0 and 1:

    python scripts/run_suite.py --suite phase2_a5_raw --gpus 0 --dry-run
    python scripts/run_suite.py --suite phase2_a5_residual --gpus 0 --dry-run
    python scripts/run_suite.py --suite phase2_a5_no_interaction --gpus 0 --dry-run

The three suites forward their YAML training settings to the CLI, including
the CN feature and interaction modes. Run them only in a fresh results
directory (or a fresh remote project root); do not resume a legacy manifest.
After a real run, aggregate results and write the phase-2 comparison with:

    python scripts/aggregate_results.py
    python scripts/summarize_phase2.py

For the offline server bundle, use `deploy.ps1 --phase2 --remote-root
/home/book/zyr/DCDLP_phase2_20260907`. The deployment refuses to use the
legacy default root in phase-2 mode and uses the isolated Docker container
`dcdlp-experiment-phase2-20260907`.

## Full paper run

The resumable one-click PowerShell launcher is `run_full_experiments.ps1`. It validates the GPU and dependencies, prepares missing official datasets, runs tests, launches the main, ablation, and analysis suites, aggregates JSON/CSV, and generates vector figures:

    .\run_full_experiments.ps1 -Gpu 0

Run only the non-mutating checks and job count first with:

    .\run_full_experiments.ps1 -Gpu 0 -PreflightOnly

Completed and explicitly unsupported jobs are skipped when the command is rerun. Use `-Resume:$false` only when every configured job should be launched again. A full five-dataset, multi-seed run needs substantial GPU time and storage and is intentionally distinct from smoke validation.

HeaRT must use its official split/candidate files. Cora/CiteSeer/PubMed with protocol=standard use an 85/5/10 split only as the supplementary Standard protocol. OGB datasets use their official splits. Validation/test positives are never inserted into the message-passing graph.

Official model sources remain under third_party and are never edited. Their runtime failures or absence are logged; this keeps official-environment reproduction auditable. The current suite manifest enumerates every requested baseline, while the native implementation covers DCDLP, GAE building blocks, and the heuristic baselines.

## Outputs

- results/raw: one JSON result and one per-pair CSV prediction file per completed run.
- results/checkpoints: model checkpoints.
- results/manifest.csv: completed, failed, and not-run jobs with logs.
- results/aggregate: run-level and mean/std/count CSV files.
- results/tables: generated LaTeX tables.
- results/figures: generated PDF and SVG figures.
- cache/interventions: Parquet caches when pyarrow is installed, otherwise explicit JSONL fallback.

## Integrity rules implemented

- Splits are disjoint; negatives exclude every known positive edge.
- Candidate target links are batch-masked from message passing.
- Undirected pair features and all three model branches are exchange invariant.
- CN edits preserve the full degree vector and change CN by the requested amount.
- Degree edits preserve the exact target common-neighbor set and endpoint delta.
- Edit logs are checked against graph diffs and held-out forbidden edges.
- All random generators are seeded; deterministic cuDNN flags are set.
- MRR uses the same average optimistic/pessimistic tie policy for every model.

See configs/suite for the smoke, main, ablation, and analysis experiment matrices.
