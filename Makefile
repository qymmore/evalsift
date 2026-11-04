.PHONY: install test lint demo clean

install:
	pip install -e ".[dev]" && pre-commit install

test:
	pytest --cov

lint:
	ruff check src tests && ruff format --check src tests && mypy

# One command: build demo logs offline, then show what evalsift finds.
demo:
	python tests/logfactory.py demo-logs 3
	evalsift check demo-logs --fail-on never
	evalsift summary demo-logs --by model
	evalsift bundle demo-logs -o demo-review.zip

clean:
	rm -rf demo-logs demo-review.zip sift-out dist .coverage
