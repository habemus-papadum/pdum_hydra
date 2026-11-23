"""Unit tests for the pdum.hydra public API."""

import os
import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf

import pdum.hydra as pdum_hydra
from pdum.hydra import Run, Sweep, generate_sweep_configs


@pytest.fixture
def simple_config_dir(tmp_path):
    """Create a simple config directory with basic configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create a basic config.yaml
    config_file = config_dir / "config.yaml"
    config_file.write_text("""
model:
  name: resnet
  layers: 50

training:
  lr: 0.01
  batch_size: 32
  epochs: 100
""")
    return config_dir


@pytest.fixture
def sweep_config_dir(tmp_path):
    """Create a config directory with sweep configurations."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create base config
    config_file = config_dir / "config.yaml"
    config_file.write_text("""
model:
  name: resnet
  layers: 50

training:
  lr: 0.01
  batch_size: 32
""")

    # Create sweeps directory
    sweeps_dir = config_dir / "sweeps"
    sweeps_dir.mkdir()

    # Create a sweep config
    sweep_file = sweeps_dir / "basic.yaml"
    sweep_file.write_text("""
parameters:
  training.lr: 0.001
  training.batch_size: 64
""")

    return config_dir


@pytest.fixture
def multi_sweep_config_dir(tmp_path):
    """Create a config directory with multiple sweep parameters."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create base config
    config_file = config_dir / "config.yaml"
    config_file.write_text("""
model:
  name: resnet

training:
  lr: 0.01
  batch_size: 32
  optimizer: adam
""")

    return config_dir


@pytest.fixture
def advanced_sweep_config_dir(tmp_path):
    """Create a config directory with advanced sweep configurations using arrays and ranges."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create base config
    config_file = config_dir / "config.yaml"
    config_file.write_text("""
agent:
  code:
    model: gpt-4
    temp: 0.7
  search:
    max_debug_depth: 10
    num_drafts: 3
  k_fold_validation: 3

limits:
  steps: 100
  code_execution_time: 3600
  total_time: 7200

training:
  lr: 0.01
  batch_size: 32
  optimizer: sgd

model:
  name: default

competition: default-competition
trial: 0
""")

    # Create sweeps directory
    sweeps_dir = config_dir / "sweeps"
    sweeps_dir.mkdir()

    # Create a comprehensive sweep config with arrays and scalar values
    sweep_file = sweeps_dir / "comprehensive.yaml"
    sweep_file.write_text("""
parameters:
  trial: [0, 1, 2, 3, 4]
  competition:
    - random-acts-of-pizza
    - new-york-city-taxi-fare-prediction
    - tabular-playground-series-may-2022
  agent.code.model:
    - o1
    - o3-mini
    - o1-mini
  limits.steps: 500
  agent.search.max_debug_depth: 20
  agent.search.num_drafts: 5
  limits.code_execution_time: 32400
  agent.code.temp: 1.0
  agent.k_fold_validation: 5
  limits.total_time: 86400
""")

    # Create a sweep with mixed arrays
    sweep_file2 = sweeps_dir / "mixed.yaml"
    sweep_file2.write_text("""
parameters:
  training.lr: [0.001, 0.01, 0.1]
  training.batch_size: [16, 32, 64]
  training.optimizer: adam
  model.name: [resnet, vgg]
""")

    return config_dir


def _counting_compose():
    """Patch the config composer so tests can count how many configs get built."""
    return patch("pdum.hydra.render_sweep_config", wraps=pdum_hydra.render_sweep_config)


class TestRun:
    """Test suite for the Run class."""

    def test_run_fields(self):
        """A Run exposes index, params, hydra_overrides and a lazily composed config."""
        compose = MagicMock(return_value=OmegaConf.create({"lr": 0.01}))
        run = Run(3, {"lr": 0.01}, ["lr=0.01"], compose)

        assert run.index == 3
        assert run.params == {"lr": 0.01}
        assert run.hydra_overrides == ["lr=0.01"]
        compose.assert_not_called()

        assert run.config.lr == 0.01
        assert run.config.lr == 0.01
        compose.assert_called_once()

    def test_run_without_composer(self):
        """A Run with no composer raises when its config is accessed."""
        run = Run(0, {}, [], None)
        with pytest.raises(RuntimeError):
            run.config

    def test_run_repr_does_not_compose(self):
        """repr shows index and params without composing the config."""
        compose = MagicMock()
        run = Run(0, {"lr": 0.01}, ["lr=0.01"], compose)
        assert repr(run) == "Run(index=0, params={'lr': 0.01})"
        compose.assert_not_called()


class TestSweep:
    """Test suite for the lazy Sweep sequence."""

    def test_len_and_axes_without_composing(self, simple_config_dir):
        """len() and axes are available without composing any config."""
        with _counting_compose() as compose:
            sweep = generate_sweep_configs(
                overrides=["training.lr=0.001,0.01,0.1", "model.layers=50,101"],
                config_dir=simple_config_dir,
            )
            assert len(sweep) == 6
            assert len(sweep.runs) == 6
            assert sweep.axes == {"training.lr": [0.001, 0.01, 0.1], "model.layers": [50, 101]}
            runs = list(sweep)
            assert len(runs) == 6
            assert compose.call_count == 0

    def test_config_composed_lazily_and_cached(self, simple_config_dir):
        """Each run's config is composed on first access only."""
        with _counting_compose() as compose:
            sweep = generate_sweep_configs(overrides=["training.lr=0.001,0.01"], config_dir=simple_config_dir)
            run = sweep[1]
            assert compose.call_count == 0
            assert run.config.training.lr == 0.01
            assert run.config.training.lr == 0.01
            assert compose.call_count == 1

    def test_filtering_on_params_skips_composition(self, simple_config_dir):
        """Runs filtered out on params never have their config composed."""
        with _counting_compose() as compose:
            sweep = generate_sweep_configs(
                overrides=["training.lr=0.001,0.01,0.1", "model.layers=50,101"],
                config_dir=simple_config_dir,
            )
            kept = []
            for run in sweep:
                if run.params["model.layers"] == 101 and run.params["training.lr"] == 0.1:
                    continue
                if run.config.training.lr == 0.001:
                    continue
                kept.append(run)
            assert [(r.params["training.lr"], r.params["model.layers"]) for r in kept] == [
                (0.01, 50),
                (0.01, 101),
                (0.1, 50),
            ]
            assert compose.call_count == 5

    def test_reiterable(self, simple_config_dir):
        """A sweep can be iterated more than once."""
        sweep = generate_sweep_configs(overrides=["training.lr=0.001,0.01"], config_dir=simple_config_dir)
        assert [r.params for r in sweep] == [r.params for r in sweep]

    def test_indexing_matches_iteration(self, simple_config_dir):
        """sweep[i] gives the same run as the i-th item of iteration."""
        sweep = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01,0.1", "model.layers=50,101", "model.name=a,b"],
            config_dir=simple_config_dir,
        )
        iterated = list(sweep)
        assert [r.index for r in iterated] == list(range(12))
        for i, run in enumerate(iterated):
            assert sweep[i].params == run.params
            assert sweep[i].hydra_overrides == run.hydra_overrides
            assert sweep[i].index == i
        # last dimension varies fastest, like itertools.product
        assert iterated[0].params == {"training.lr": 0.001, "model.layers": 50, "model.name": "a"}
        assert iterated[1].params == {"training.lr": 0.001, "model.layers": 50, "model.name": "b"}

    def test_negative_index_and_slice(self, simple_config_dir):
        """Negative indices and slices work like a list."""
        sweep = generate_sweep_configs(overrides=["training.lr=0.001,0.01,0.1"], config_dir=simple_config_dir)
        assert sweep[-1].params == {"training.lr": 0.1}
        assert sweep[-1].index == 2
        assert [r.params["training.lr"] for r in sweep[::2]] == [0.001, 0.1]
        with pytest.raises(IndexError):
            sweep[3]
        with pytest.raises(IndexError):
            sweep[-4]

    def test_config_after_hydra_context_exits(self, simple_config_dir):
        """Configs compose correctly after generate_sweep_configs has returned."""
        sweep = generate_sweep_configs(overrides=["training.lr=0.001,0.01"], config_dir=simple_config_dir)
        assert not GlobalHydra.instance().is_initialized()
        assert sweep[0].config.training.lr == 0.001
        assert sweep[0].config.model.name == "resnet"
        assert "hydra" not in sweep[0].config

    def test_pickle_run(self, simple_config_dir):
        """A pickled run carries its composed config."""
        sweep = generate_sweep_configs(overrides=["training.lr=0.001,0.01"], config_dir=simple_config_dir)
        run = pickle.loads(pickle.dumps(sweep[1]))
        assert run.index == 1
        assert run.params == {"training.lr": 0.01}
        assert run.hydra_overrides == ["training.lr=0.01"]
        assert run.config.training.lr == 0.01

    def test_repr(self, simple_config_dir):
        sweep = generate_sweep_configs(overrides=["training.lr=0.001,0.01"], config_dir=simple_config_dir)
        assert repr(sweep) == "Sweep(len=2, axes={'training.lr': [0.001, 0.01]})"


class TestParams:
    """run.params values are typed by Hydra's parser."""

    def test_typed_values(self, simple_config_dir):
        """Bools, null, ints, floats and strings come through with the right types."""
        sweep = generate_sweep_configs(
            overrides=["+flag=true,false", "+nothing=null", "model.layers=50", "model.name=f", "training.lr=1e-3"],
            config_dir=simple_config_dir,
        )
        assert [r.params["flag"] for r in sweep] == [True, False]
        params = sweep[0].params
        assert params["nothing"] is None
        assert params["model.layers"] == 50
        assert params["training.lr"] == 0.001
        # Regression: strings that are substrings of "false" used to become False.
        assert params["model.name"] == "f"
        for value in ["a", "al", "fa"]:
            s = generate_sweep_configs(overrides=[f"model.name={value}"], config_dir=simple_config_dir)
            assert s[0].params["model.name"] == value

    def test_plus_prefix_stripped_from_keys(self, simple_config_dir):
        """'+key' overrides appear as 'key' in params and axes, but keep '+' in hydra_overrides."""
        sweep = generate_sweep_configs(overrides=["+extra=1,2", "++other=3"], config_dir=simple_config_dir)
        assert sweep.axes == {"extra": [1, 2]}
        assert sweep[0].params == {"extra": 1, "other": 3}
        assert sweep[0].hydra_overrides == ["+extra=1", "++other=3"]
        assert sweep[0].config.extra == 1
        assert sweep[0].config.other == 3

    def test_params_include_fixed_settings(self, simple_config_dir):
        """Fixed overrides are in params but not in axes."""
        sweep = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01", "training.epochs=5"], config_dir=simple_config_dir
        )
        assert sweep.axes == {"training.lr": [0.001, 0.01]}
        assert sweep[0].params == {"training.lr": 0.001, "training.epochs": 5}

    def test_range_sweep(self, simple_config_dir):
        """range() is a discrete sweep and is supported."""
        sweep = generate_sweep_configs(overrides=["training.epochs=range(1,4)"], config_dir=simple_config_dir)
        assert sweep.axes == {"training.epochs": [1, 2, 3]}
        assert [r.config.training.epochs for r in sweep] == [1, 2, 3]

    def test_unsupported_sweep_type(self, simple_config_dir):
        """Non-discrete sweeps such as interval() are rejected."""
        with pytest.raises(HydraException):
            generate_sweep_configs(overrides=["training.lr=interval(0.1,1.0)"], config_dir=simple_config_dir)


class TestGenerateSweepConfigs:
    """Test suite for generate_sweep_configs function."""

    def test_generate_sweep_configs_basic(self, simple_config_dir):
        """Test basic sweep config generation without any sweeps."""
        result = generate_sweep_configs(config_dir=simple_config_dir, config_name="config")

        assert isinstance(result, Sweep)
        assert isinstance(result.base_config, DictConfig)
        assert isinstance(result.runs, Sweep)
        # Without sweeps, should have exactly 1 run
        assert len(result.runs) == 1
        assert isinstance(result.runs[0], Run)

    def test_generate_sweep_configs_with_single_override(self, simple_config_dir):
        """Test sweep config generation with a single override value."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001"], config_dir=simple_config_dir, config_name="config"
        )

        assert isinstance(result, Sweep)
        assert len(result.runs) == 1
        # Check that the override was applied
        assert result.runs[0].config.training.lr == 0.001

    def test_generate_sweep_configs_with_sweep(self, simple_config_dir):
        """Test sweep config generation with sweep parameters."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01,0.1"], config_dir=simple_config_dir, config_name="config"
        )

        assert isinstance(result, Sweep)
        # Should have 3 runs for 3 learning rates
        assert len(result.runs) == 3

        # Check override map
        assert "training.lr" in result.axes
        assert result.axes["training.lr"] == [0.001, 0.01, 0.1]

        # Check that each run has the correct lr
        lrs = [run.config.training.lr for run in result.runs]
        assert set(lrs) == {0.001, 0.01, 0.1}

    def test_generate_sweep_configs_cartesian_product(self, simple_config_dir):
        """Test that sweeps create cartesian product of parameters."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01", "training.batch_size=16,32"],
            config_dir=simple_config_dir,
            config_name="config",
        )

        # 2 learning rates × 2 batch sizes = 4 runs
        assert len(result.runs) == 4

        # Check that we have all combinations
        configs = [(run.config.training.lr, run.config.training.batch_size) for run in result.runs]
        expected = [(0.001, 16), (0.001, 32), (0.01, 16), (0.01, 32)]
        assert set(configs) == set(expected)

    def test_generate_sweep_configs_params(self, simple_config_dir):
        """Test that params is correctly populated."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01"], config_dir=simple_config_dir, config_name="config"
        )

        assert len(result.runs) == 2
        for run in result.runs:
            assert "training.lr" in run.params
            assert run.params["training.lr"] in [0.001, 0.01]

    def test_generate_sweep_configs_with_string_sweep(self, simple_config_dir):
        """Test sweep with string values."""
        result = generate_sweep_configs(
            overrides=["model.name=resnet,vgg,alexnet"], config_dir=simple_config_dir, config_name="config"
        )

        assert len(result.runs) == 3
        names = [run.config.model.name for run in result.runs]
        assert set(names) == {"resnet", "vgg", "alexnet"}

    def test_generate_sweep_configs_mixed_types(self, simple_config_dir):
        """Test sweep with mixed parameter types."""
        result = generate_sweep_configs(
            overrides=["model.name=resnet,vgg", "training.lr=0.001,0.01", "training.epochs=50,100"],
            config_dir=simple_config_dir,
            config_name="config",
        )

        # 2 × 2 × 2 = 8 runs
        assert len(result.runs) == 8

        # Verify all runs have valid configs
        for run in result.runs:
            assert run.config.model.name in ["resnet", "vgg"]
            assert run.config.training.lr in [0.001, 0.01]
            assert run.config.training.epochs in [50, 100]

    def test_generate_sweep_configs_with_sweep_location(self, sweep_config_dir):
        """Test using sweep configurations from files."""
        result = generate_sweep_configs(overrides=["+sweeps=basic"], config_dir=sweep_config_dir, config_name="config")

        assert isinstance(result, Sweep)
        assert len(result.runs) >= 1
        # Verify the sweep config was loaded and applied
        # The basic.yaml sets lr=0.001 and batch_size=64
        assert result.runs[0].config.training.lr == 0.001
        assert result.runs[0].config.training.batch_size == 64

    def test_generate_sweep_configs_empty_overrides(self, simple_config_dir):
        """Test with explicitly empty overrides list."""
        result = generate_sweep_configs(overrides=[], config_dir=simple_config_dir, config_name="config")

        assert isinstance(result, Sweep)
        assert len(result.runs) == 1

    def test_generate_sweep_configs_none_overrides(self, simple_config_dir):
        """Test with None overrides (default parameter)."""
        result = generate_sweep_configs(config_dir=simple_config_dir, config_name="config")

        assert isinstance(result, Sweep)
        assert len(result.runs) == 1

    def test_sweep_file_with_arrays(self, advanced_sweep_config_dir):
        """Test sweep file with array parameters creating sweep dimensions."""
        result = generate_sweep_configs(
            overrides=["+sweeps=mixed"], config_dir=advanced_sweep_config_dir, config_name="config"
        )

        # 3 lr × 3 batch_size × 2 model.name = 18 runs
        assert len(result.runs) == 18

        # Check that sweep dimensions are in axes
        assert "training.lr" in result.axes
        assert "training.batch_size" in result.axes
        assert "model.name" in result.axes

        # Verify the sweep values
        assert set(result.axes["training.lr"]) == {0.001, 0.01, 0.1}
        assert set(result.axes["training.batch_size"]) == {16, 32, 64}
        assert set(result.axes["model.name"]) == {"resnet", "vgg"}

        # Verify all runs have the scalar value applied
        for run in result.runs:
            assert run.config.training.optimizer == "adam"

    def test_sweep_file_with_scalars_and_arrays(self, advanced_sweep_config_dir):
        """Test sweep file with both scalar values and array sweep parameters."""
        result = generate_sweep_configs(
            overrides=["+sweeps=comprehensive"], config_dir=advanced_sweep_config_dir, config_name="config"
        )

        # 5 trials × 3 competitions × 3 models = 45 runs
        assert len(result.runs) == 45

        # Check sweep dimensions
        assert "trial" in result.axes
        assert "competition" in result.axes
        assert "agent.code.model" in result.axes

        # Verify sweep values
        assert result.axes["trial"] == [0, 1, 2, 3, 4]
        assert set(result.axes["competition"]) == {
            "random-acts-of-pizza",
            "new-york-city-taxi-fare-prediction",
            "tabular-playground-series-may-2022",
        }
        assert set(result.axes["agent.code.model"]) == {"o1", "o3-mini", "o1-mini"}

        # Verify scalar values are applied to all runs
        for run in result.runs:
            assert run.config.limits.steps == 500
            assert run.config.agent.search.max_debug_depth == 20
            assert run.config.agent.search.num_drafts == 5
            assert run.config.limits.code_execution_time == 32400
            assert run.config.agent.code.temp == 1.0
            assert run.config.agent.k_fold_validation == 5
            assert run.config.limits.total_time == 86400

    def test_sweep_file_cartesian_product(self, advanced_sweep_config_dir):
        """Test that sweep file arrays create cartesian product."""
        result = generate_sweep_configs(
            overrides=["+sweeps=comprehensive"], config_dir=advanced_sweep_config_dir, config_name="config"
        )

        # Verify we have all combinations
        combinations = set()
        for run in result.runs:
            combo = (run.config.trial, run.config.competition, run.config.agent.code.model)
            combinations.add(combo)

        # Should have 5 × 3 × 3 = 45 unique combinations
        assert len(combinations) == 45

        # Verify specific combinations exist
        assert (0, "random-acts-of-pizza", "o1") in combinations
        assert (4, "tabular-playground-series-may-2022", "o1-mini") in combinations

    def test_sweep_file_params_with_arrays(self, advanced_sweep_config_dir):
        """Test that params is correctly populated for sweep files with arrays."""
        result = generate_sweep_configs(
            overrides=["+sweeps=mixed"], config_dir=advanced_sweep_config_dir, config_name="config"
        )

        # Check each run has correct params
        for run in result.runs:
            assert "training.lr" in run.params
            assert "training.batch_size" in run.params
            assert "model.name" in run.params
            assert "training.optimizer" in run.params

            # Verify values are from the sweep
            assert run.params["training.lr"] in [0.001, 0.01, 0.1]
            assert run.params["training.batch_size"] in [16, 32, 64]
            assert run.params["model.name"] in ["resnet", "vgg"]
            assert run.params["training.optimizer"] == "adam"


class TestDynamicLoadConfig:
    """Test suite for dynamic.load_config function."""

    def test_load_config_import(self):
        """Test that load_config can be imported."""
        from pdum.hydra.dynamic import load_config

        assert callable(load_config)

    def test_load_config_no_env_var(self):
        """Test load_config when environment variable is not set."""
        from pdum.hydra.dynamic import load_config

        # Make sure env var doesn't exist
        env_var_name = "TEST_CONFIG_DOES_NOT_EXIST_12345"
        if env_var_name in os.environ:
            del os.environ[env_var_name]

        # Should not raise an exception
        load_config(env_var_name, "some.module")

    def test_load_config_with_mock_module(self):
        """Test load_config with a mocked module."""
        from pdum.hydra.dynamic import load_config

        # Create a mock module with config() function
        mock_module = MagicMock()
        mock_module.config = MagicMock()

        env_var_name = "TEST_DYNAMIC_CONFIG_12345"
        os.environ[env_var_name] = "production"

        try:
            with patch("pdum.hydra.dynamic.importlib.import_module") as mock_import:
                # Make import_module update sys.modules
                def import_side_effect(name):
                    sys.modules[name] = mock_module
                    return mock_module

                mock_import.side_effect = import_side_effect

                load_config(env_var_name, "test.base")

                # Verify import was called with correct path
                mock_import.assert_called_once_with("test.base.production")
                # Verify config() was called
                mock_module.config.assert_called_once()
        finally:
            # Clean up
            if env_var_name in os.environ:
                del os.environ[env_var_name]
            if "test.base.production" in sys.modules:
                del sys.modules["test.base.production"]


class TestAPIExports:
    """Test suite for verifying the public API exports."""

    def test_version_export(self):
        """Test that __version__ is exported."""
        from pdum import hydra

        assert hasattr(hydra, "__version__")
        assert isinstance(hydra.__version__, str)

    def test_generate_sweep_configs_export(self):
        """Test that generate_sweep_configs is exported."""
        from pdum import hydra

        assert hasattr(hydra, "generate_sweep_configs")
        assert callable(hydra.generate_sweep_configs)

    def test_run_export(self):
        """Test that Run is exported."""
        from pdum import hydra

        assert hasattr(hydra, "Run")

    def test_sweep_export(self):
        """Test that Sweep is exported."""
        from pdum import hydra

        assert hasattr(hydra, "Sweep")

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from pdum import hydra

        assert hasattr(hydra, "__all__")

        expected_exports = {"__version__", "generate_sweep_configs", "Run", "Sweep"}
        actual_exports = set(hydra.__all__)

        assert expected_exports == actual_exports

    def test_dynamic_module_exists(self):
        """Test that dynamic module can be imported."""
        from pdum.hydra import dynamic

        assert dynamic is not None

    def test_dynamic_load_config_accessible(self):
        """Test that load_config is accessible from dynamic module."""
        from pdum.hydra.dynamic import load_config

        assert callable(load_config)
