.PHONY: install smoke test paper-main paper-ablation paper-analysis aggregate figures

install:
	pip install -e ".[full,test]"

test:
	python -m pytest -q

smoke:
	python -m pytest -q
	python scripts/run_suite.py --suite smoke

paper-main:
	python scripts/run_suite.py --suite paper_main --continue-on-error

paper-ablation:
	python scripts/run_suite.py --suite paper_ablation --continue-on-error

paper-analysis:
	python scripts/run_suite.py --suite paper_analysis --continue-on-error

aggregate:
	python scripts/aggregate_results.py

figures:
	python scripts/make_paper_figures.py

