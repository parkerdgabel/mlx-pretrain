# MLX-Pretrain Test Suite

This directory contains a comprehensive set of unit tests for the MLX-Pretrain project using pytest.

## Test Structure

The tests are organized as follows:

- `tests/test_train.py`: Tests for the train.py module
- `tests/test_dataset.py`: Tests for the dataset.py module
- `tests/test_generate.py`: Tests for both generate.py and generate_lite.py modules
- `tests/test_sft.py`: Tests for the sft.py module
- `tests/test_rl.py`: Tests for the rl.py module
- `tests/test_examine.py`: Tests for the examine.py module
- `tests/test_train_tokenizer.py`: Tests for the train-tokenizer.py module
- `tests/conftest.py`: Shared pytest fixtures

## Setup

To set up the development environment for testing:

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

## Running Tests

To run all tests:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=.
```

To run a specific test file:

```bash
pytest tests/test_train.py
```

To run a specific test function:

```bash
pytest tests/test_train.py::test_config_from_yaml
```

## Test Categories

Tests are categorized using markers:

- `@pytest.mark.unit`: Unit tests (default)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow tests

To run only unit tests:

```bash
pytest -m "unit"
```

To skip slow tests:

```bash
pytest -m "not slow"
```

## Continuous Integration

These tests can be integrated into a CI/CD pipeline to ensure code quality and prevent regressions.

## Adding New Tests

When adding new functionality to the project, please also add corresponding tests. Follow these guidelines:

1. Create test functions with names starting with `test_`
2. Use appropriate fixtures from `conftest.py` when possible
3. Mock external dependencies to ensure tests are isolated
4. Add appropriate markers to categorize tests