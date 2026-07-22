.PHONY: test lint typecheck kill-test paper install

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check patchguard tests

typecheck:
	mypy patchguard

# The go/no-go gate (MASTER_PLAN). Wired once retrievers + attack v0 land.
kill-test:
	python -m experiments.kill_test

# Regenerate every table/figure from bucketed metrics.json (MASTER_PLAN S11).
paper:
	python -m experiments.make_tables
