ARGS ?= --help

install:
	uv sync

run:
	uv run python3 -m src $(ARGS)

debug:
	uv run python3 -m pdb $(ARGS)

clean:
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .vscode -exec rm -rf {} +
	find . -name .DS_Store -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-def

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict