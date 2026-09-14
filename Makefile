VENV_PYTHON := $(wildcard .venv/bin/python)
PYTHON ?= $(if $(VENV_PYTHON),$(VENV_PYTHON),python3)
RESULTS := results
VIEWER := $(RESULTS)/viewer.html
REPLAYS := $(RESULTS)/replays.json

.DEFAULT_GOAL := help
.PHONY: help setup test lint typecheck format viewer clean

help:
	@echo "mario-blj targets"
	@echo "  setup      clone and build third_party/libsm64, then check for the ROM"
	@echo "  test       run the pytest suite"
	@echo "  lint       run ruff check"
	@echo "  typecheck  run pyright"
	@echo "  format     rewrite sources with ruff format"
	@echo "  viewer     pack the recorded replays and render $(VIEWER)"
	@echo "  clean      delete caches, build products and the rendered viewer"
	@echo
	@echo "Interpreter in use: $(PYTHON)"
	@echo "Override it with PYTHON=/path/to/python."
	@echo "Dev tools come from the dev extra: $(PYTHON) -m pip install -e '.[dev]'"

setup:
	scripts/setup.sh

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check --output-format=concise .

typecheck:
	$(PYTHON) -m pyright --pythonpath $(PYTHON)

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix --exit-zero .

viewer:
	$(PYTHON) tools/pack_replays.py --out $(REPLAYS)
	$(PYTHON) tools/make_viewer.py --replays $(REPLAYS) --out $(VIEWER)

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	rm -f $(VIEWER)
	find . -path ./third_party -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
