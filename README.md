# pdum.hydra

[![CI](https://github.com/habemus-papadum/pdum_hydra/actions/workflows/ci.yml/badge.svg)](https://github.com/habemus-papadum/pdum_hydra/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/habemus-papadum/pdum_hydra/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_hydra/blob/python-coverage-comment-action-data/htmlcov/index.html)

[![PyPI](https://img.shields.io/pypi/v/habemus-papadum-hydra.svg)](https://pypi.org/project/habemus-papadum-hydra/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A streamlined library for managing Hydra configurations with first-class support for parameter sweeps. Built on top of the [Hydra](https://hydra.cc/) framework, `pdum.hydra` simplifies sweep generation and configuration management for machine learning experiments. Generate all combinations of hyperparameters with ease, iterate over configurations programmatically, and manage complex experimental setups without the overhead of Hydra's CLI and job launching features. Perfect for ML experimentation workflows that need structured configs with powerful sweep capabilities.

## Installation

Install using pip:

```bash
pip install habemus-papadum-hydra
```

Or using uv:

```bash
uv pip install habemus-papadum-hydra
```

## Usage

### Basic Parameter Sweeps

```python
from pdum.hydra import generate_sweep_configs

# Generate sweep from config directory with parameter sweeps
runs = generate_sweep_configs(
    overrides=["training.lr=0.001,0.01,0.1", "model.layers=50,101"],
    config_dir="path/to/config",
    config_name="config"
)

# Iterate over all run configurations
for run in runs.runs:
    print(f"Running with: {run.override_dict}")
    # Access the fully resolved config
    config = run.config
    # Your training code here
    # train_model(config)

# Inspect the sweep parameters
print(f"Total runs: {len(runs.runs)}")  # 6 runs (3 lr × 2 layers)
print(f"Sweep parameters: {runs.override_map}")
```

### Using Sweep Files

For complex sweeps, use sweep configuration files. Create a file `config/sweeps/experiment.yaml`:

```yaml
parameters:
  # Arrays create sweep dimensions (cartesian product)
  trial: [0, 1, 2, 3, 4]
  competition:
    - random-acts-of-pizza
    - new-york-city-taxi-fare-prediction
    - tabular-playground-series-may-2022
  agent.code.model:
    - o1
    - o3-mini
    - o1-mini

  # Scalar values are applied to all runs (no sweep)
  limits.steps: 500
  agent.search.max_debug_depth: 20
  agent.search.num_drafts: 5
  limits.code_execution_time: 32400
  agent.code.temp: 1.0
  agent.k_fold_validation: 5
  limits.total_time: 86400
```

Then use it in your code:

```python
# Load sweep from file
runs = generate_sweep_configs(
    overrides=["+sweeps=experiment"],
    config_dir="path/to/config"
)

# This generates 5 trials × 3 competitions × 3 models = 45 runs
print(f"Total runs: {len(runs.runs)}")  # 45

# All runs share the same scalar values
for run in runs.runs:
    assert run.config.limits.steps == 500
    assert run.config.agent.code.temp == 1.0
    # But have different sweep parameter values
    print(f"Trial {run.config.trial}, Model: {run.config.agent.code.model}")
```

### Combining Sweep Files with Overrides

You can combine sweep files with additional command-line overrides:

```python
runs = generate_sweep_configs(
    overrides=[
        "+sweeps=experiment",  # Load base sweep
        "optimizer=adam,sgd"    # Add another sweep dimension
    ],
    config_dir="path/to/config"
)

# Now: 45 runs × 2 optimizers = 90 runs
print(f"Total runs: {len(runs.runs)}")  # 90
```

## Development

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

### Setup

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/habemus-papadum/pdum_hydra.git
cd pdum_hydra

# Provision the entire toolchain (uv sync, pre-commit hooks)
./scripts/setup.sh
```

**Important for Development**:
- `./scripts/setup.sh` is idempotent—rerun it after pulling dependency changes
- Use `uv sync --frozen` to ensure the lockfile is respected when installing Python deps

### Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_example.py

# Run a specific test function
uv run pytest tests/test_example.py::test_version

# Run tests with coverage
uv run pytest --cov=src/pdum/hydra --cov-report=xml --cov-report=term
```

### Code Quality

```bash
# Check code with ruff
uv run ruff check .

# Format code with ruff
uv run ruff format .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Building

```bash
# Build Python 
./scripts/build.sh

# Or build just the Python distribution artifacts
uv build
```

### Publishing

```bash
# Build and publish to PyPI (requires credentials)
./scripts/publish.sh
```

### Automation scripts

- `./scripts/setup.sh` – bootstrap uv, pnpm, widget bundle, and pre-commit hooks
- `./scripts/build.sh` – reproduce the release build locally
- `./scripts/pre-release.sh` – run the full battery of quality checks
- `./scripts/release.sh` – orchestrate the release (creates tags, publishes to PyPI/GitHub)
- `./scripts/test_notebooks.sh` – execute demo notebooks (uses `./scripts/nb.sh` under the hood)

## License

MIT License - see LICENSE file for details.
