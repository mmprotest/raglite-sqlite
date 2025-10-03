# Contributing to raglite

Thank you for helping improve raglite! The project aims to provide a practical local-first RAG stack powered by SQLite.

## Getting started

1. Fork and clone the repository.
2. Install dependencies: `pip install -e .[dev,server]`.
3. Enable pre-commit hooks: `pre-commit install`.
4. Run the demo self-test: `raglite self-test`.

## Development workflow

- Make sure `pytest -q` passes before submitting a pull request.
- Run `ruff`, `mypy`, and `black` using the provided pre-commit configuration.
- Update documentation and add tests for new features.
- Follow the MIT license and keep dependencies minimal.

## Reporting issues

Open an issue with detailed reproduction steps, expected behaviour, and environment information. Attach logs if available.

## Code of conduct

Please review `CODE_OF_CONDUCT.md` for community guidelines.
