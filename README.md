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

### The mental model

A **sweep** is a grid. It has **axes** (each swept parameter and its values) and **runs** (one per point on the grid).
A **run** has three views of the same settings:

| Field | What it is | Example |
|---|---|---|
| `run.params` | Settings as a dict, typed by Hydra. Cheap: use it to filter. | `{"training.lr": 0.01, "model.layers": 50}` |
| `run.hydra_overrides` | The same settings as Hydra override strings | `["training.lr=0.01", "model.layers=50"]` |
| `run.config` | The fully composed config. Built on first access, then cached on the run. | `DictConfig` |

The sweep is lazy: `len(sweep)` and `sweep[i]` don't compose anything, and a run's config is only composed when you
read `run.config`. So you can skip combinations you don't want before paying for them.

### Basic Parameter Sweeps

```python
from pdum.hydra import generate_sweep_configs

sweep = generate_sweep_configs(
    overrides=["training.lr=0.001,0.01,0.1", "model.layers=50,101"],
    config_dir="path/to/config",
    config_name="config",
)

print(len(sweep))   # 6 (3 lr × 2 layers), nothing composed yet
print(sweep.axes)   # {'training.lr': [0.001, 0.01, 0.1], 'model.layers': [50, 101]}

for run in sweep:
    print(f"Run {run.index}: {run.params}")
    train_model(run.config)
```

### Filtering Runs

Filter in two stages: first on `run.params` (no config is composed), then, for the runs that survive, on the full
`run.config` if you need it.

```python
for run in sweep:
    # Cheap: skip combinations that don't make sense, without composing a config
    if run.params["model.layers"] == 101 and run.params["training.lr"] == 0.1:
        continue

    # Expensive: compose the config (cached on the run) and check something in it
    if run.config.training.batch_size * run.config.model.layers > 10_000:
        continue

    train_model(run.config)
```

Or as a generator expression:

```python
wanted = (run for run in sweep if run.params["training.lr"] < 0.1)
```

`sweep` is a re-iterable sequence, so you can loop over it more than once, index it (`sweep[0]`, `sweep[-1]`), and
slice it (`sweep[:3]`). Each access creates a fresh `Run`; keep a reference to a run if you'll read its config more than
once. A `Run` can be pickled (for example, to send to a worker process); its config is composed and carried along.

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
sweep = generate_sweep_configs(
    overrides=["+sweeps=experiment"],
    config_dir="path/to/config",
)

# 5 trials × 3 competitions × 3 models = 45 runs
print(len(sweep))  # 45

# Swept parameters are in sweep.axes; fixed ones are not, but every run's params include both
print(sweep.axes.keys())       # dict_keys(['trial', 'competition', 'agent.code.model'])
print(sweep[0].params["limits.steps"])  # 500

for run in sweep:
    if run.params["agent.code.model"] == "o1" and run.params["trial"] > 2:
        continue  # e.g. only 3 trials for the most expensive model
    assert run.config.limits.steps == 500
    print(f"Trial {run.params['trial']}, Model: {run.params['agent.code.model']}")
```

### Combining Sweep Files with Overrides

You can combine sweep files with additional command-line overrides:

```python
sweep = generate_sweep_configs(
    overrides=[
        "+sweeps=experiment",  # Load base sweep
        "optimizer=adam,sgd",  # Add another sweep dimension
    ],
    config_dir="path/to/config",
)

# Now: 45 runs × 2 optimizers = 90 runs
print(len(sweep))  # 90
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
