PY ?= python
GAL ?= gal
SEEDS ?= 8
CONF ?= 0,0.6,1.2

.PHONY: install lint typecheck test check generate build attribute validate control report demo clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

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

clean:
	rm -rf data .pytest_cache .mypy_cache .ruff_cache .hypothesis
