PY ?= python
GAL ?= gal
SEEDS ?= 8
CONF ?= 0,0.6,1.2

.PHONY: install lint typecheck test check generate build attribute validate control report demo clean \
	real-criteo real-criteo-data

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

generate:
	$(GAL) generate -c configs/default.yaml -o data/raw

build: generate
	$(GAL) build --raw data/raw --db data/warehouse.duckdb

attribute: build
	$(GAL) attribute --raw data/raw --db data/warehouse.duckdb -o results/attribution.csv

validate:
	$(GAL) validate -c configs/default.yaml --n-seeds $(SEEDS) --confounding $(CONF) -o results/validation

control:
	$(GAL) validate -c configs/positive_control.yaml --n-seeds 5 -o results/positive_control

report: attribute
	$(GAL) report --raw data/raw --db data/warehouse.duckdb --attribution results/attribution.csv \
		--validation results/validation --control results/positive_control -o reports

demo: validate control report

# Real data: Criteo Attribution Modeling for Bidding (CC BY-NC-SA 4.0, arXiv:1707.06409).
# 653 MB download into data/real/ (gitignored); the run takes about 25 minutes on 2 CPUs.
CRITEO_URL ?= https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/criteo_attribution_dataset.tsv.gz
CRITEO_TSV ?= data/real/criteo_attribution_dataset.tsv.gz
BOOT ?= 30

$(CRITEO_TSV):
	mkdir -p data/real
	curl -fL --retry 3 -o $@.part $(CRITEO_URL) && mv $@.part $@

real-criteo-data: $(CRITEO_TSV)

real-criteo: real-criteo-data
	$(PY) scripts/run_criteo.py --tsv $(CRITEO_TSV) --parquet data/real/criteo.parquet --boot $(BOOT)

clean:
	rm -rf data .pytest_cache .mypy_cache .ruff_cache .hypothesis
