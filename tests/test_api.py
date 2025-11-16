"""Unit tests for the pdum.hydra public API."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from pdum.hydra import GeneratedRuns, RunConfig, generate_sweep_configs


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


class TestRunConfig:
    """Test suite for RunConfig dataclass."""

    def test_run_config_creation(self):
        """Test that RunConfig can be instantiated with all required fields."""
        overrides = ["lr=0.01", "batch_size=32"]
        override_dict = {"lr": 0.01, "batch_size": 32}
        config = OmegaConf.create({"lr": 0.01, "batch_size": 32})

        run_config = RunConfig(
            overrides=overrides,
            override_dict=override_dict,
            config=config
        )

        assert run_config.overrides == overrides
        assert run_config.override_dict == override_dict
        assert isinstance(run_config.config, DictConfig)
        assert run_config.config.lr == 0.01

    def test_run_config_attributes(self):
        """Test that RunConfig attributes are accessible."""
        config = OmegaConf.create({"model": "resnet", "layers": 50})
        run_config = RunConfig(
            overrides=["model=resnet"],
            override_dict={"model": "resnet"},
            config=config
        )

        assert hasattr(run_config, "overrides")
        assert hasattr(run_config, "override_dict")
        assert hasattr(run_config, "config")

    def test_run_config_empty_overrides(self):
        """Test RunConfig with empty overrides."""
        run_config = RunConfig(
            overrides=[],
            override_dict={},
            config=OmegaConf.create({})
        )

        assert run_config.overrides == []
        assert run_config.override_dict == {}
        assert isinstance(run_config.config, DictConfig)


class TestGeneratedRuns:
    """Test suite for GeneratedRuns dataclass."""

    def test_generated_runs_creation(self):
        """Test that GeneratedRuns can be instantiated."""
        base_config = OmegaConf.create({"default": "value"})
        override_map = {"lr": [0.001, 0.01], "batch_size": [16, 32]}
        runs = [
            RunConfig(
                overrides=["lr=0.001", "batch_size=16"],
                override_dict={"lr": 0.001, "batch_size": 16},
                config=OmegaConf.create({"lr": 0.001, "batch_size": 16})
            )
        ]

        generated_runs = GeneratedRuns(
            base_config=base_config,
            override_map=override_map,
            runs=runs
        )

        assert isinstance(generated_runs.base_config, DictConfig)
        assert generated_runs.override_map == override_map
        assert len(generated_runs.runs) == 1

    def test_generated_runs_multiple_runs(self):
        """Test GeneratedRuns with multiple run configurations."""
        base_config = OmegaConf.create({})
        override_map = {"param": [1, 2, 3]}
        runs = [
            RunConfig(
                overrides=[f"param={i}"],
                override_dict={"param": i},
                config=OmegaConf.create({"param": i})
            )
            for i in [1, 2, 3]
        ]

        generated_runs = GeneratedRuns(
            base_config=base_config,
            override_map=override_map,
            runs=runs
        )

        assert len(generated_runs.runs) == 3
        assert all(isinstance(run, RunConfig) for run in generated_runs.runs)


class TestGenerateSweepConfigs:
    """Test suite for generate_sweep_configs function."""

    def test_generate_sweep_configs_basic(self, simple_config_dir):
        """Test basic sweep config generation without any sweeps."""
        result = generate_sweep_configs(
            config_dir=simple_config_dir,
            config_name="config"
        )

        assert isinstance(result, GeneratedRuns)
        assert isinstance(result.base_config, DictConfig)
        assert isinstance(result.runs, list)
        # Without sweeps, should have exactly 1 run
        assert len(result.runs) == 1
        assert isinstance(result.runs[0], RunConfig)

    def test_generate_sweep_configs_with_single_override(self, simple_config_dir):
        """Test sweep config generation with a single override value."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001"],
            config_dir=simple_config_dir,
            config_name="config"
        )

        assert isinstance(result, GeneratedRuns)
        assert len(result.runs) == 1
        # Check that the override was applied
        assert result.runs[0].config.training.lr == 0.001

    def test_generate_sweep_configs_with_sweep(self, simple_config_dir):
        """Test sweep config generation with sweep parameters."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01,0.1"],
            config_dir=simple_config_dir,
            config_name="config"
        )

        assert isinstance(result, GeneratedRuns)
        # Should have 3 runs for 3 learning rates
        assert len(result.runs) == 3

        # Check override map
        assert "training.lr" in result.override_map
        assert result.override_map["training.lr"] == [0.001, 0.01, 0.1]

        # Check that each run has the correct lr
        lrs = [run.config.training.lr for run in result.runs]
        assert set(lrs) == {0.001, 0.01, 0.1}

    def test_generate_sweep_configs_cartesian_product(self, simple_config_dir):
        """Test that sweeps create cartesian product of parameters."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01", "training.batch_size=16,32"],
            config_dir=simple_config_dir,
            config_name="config"
        )

        # 2 learning rates × 2 batch sizes = 4 runs
        assert len(result.runs) == 4

        # Check that we have all combinations
        configs = [(run.config.training.lr, run.config.training.batch_size)
                   for run in result.runs]
        expected = [
            (0.001, 16), (0.001, 32),
            (0.01, 16), (0.01, 32)
        ]
        assert set(configs) == set(expected)

    def test_generate_sweep_configs_override_dict(self, simple_config_dir):
        """Test that override_dict is correctly populated."""
        result = generate_sweep_configs(
            overrides=["training.lr=0.001,0.01"],
            config_dir=simple_config_dir,
            config_name="config"
        )

        assert len(result.runs) == 2
        for run in result.runs:
            assert "training.lr" in run.override_dict
            assert run.override_dict["training.lr"] in [0.001, 0.01]

    def test_generate_sweep_configs_with_string_sweep(self, simple_config_dir):
        """Test sweep with string values."""
        result = generate_sweep_configs(
            overrides=["model.name=resnet,vgg,alexnet"],
            config_dir=simple_config_dir,
            config_name="config"
        )

        assert len(result.runs) == 3
        names = [run.config.model.name for run in result.runs]
        assert set(names) == {"resnet", "vgg", "alexnet"}

    def test_generate_sweep_configs_mixed_types(self, simple_config_dir):
        """Test sweep with mixed parameter types."""
        result = generate_sweep_configs(
            overrides=[
                "model.name=resnet,vgg",
                "training.lr=0.001,0.01",
                "training.epochs=50,100"
            ],
            config_dir=simple_config_dir,
            config_name="config"
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
        result = generate_sweep_configs(
            overrides=["+sweeps=basic"],
            config_dir=sweep_config_dir,
            config_name="config"
        )

        assert isinstance(result, GeneratedRuns)
        assert len(result.runs) >= 1
        # Verify the sweep config was loaded and applied
        # The basic.yaml sets lr=0.001 and batch_size=64
        assert result.runs[0].config.training.lr == 0.001
        assert result.runs[0].config.training.batch_size == 64

    def test_generate_sweep_configs_empty_overrides(self, simple_config_dir):
        """Test with explicitly empty overrides list."""
        result = generate_sweep_configs(
            overrides=[],
            config_dir=simple_config_dir,
            config_name="config"
        )

        assert isinstance(result, GeneratedRuns)
        assert len(result.runs) == 1

    def test_generate_sweep_configs_none_overrides(self, simple_config_dir):
        """Test with None overrides (default parameter)."""
        result = generate_sweep_configs(
            config_dir=simple_config_dir,
            config_name="config"
        )

        assert isinstance(result, GeneratedRuns)
        assert len(result.runs) == 1

    def test_sweep_file_with_arrays(self, advanced_sweep_config_dir):
        """Test sweep file with array parameters creating sweep dimensions."""
        result = generate_sweep_configs(
            overrides=["+sweeps=mixed"],
            config_dir=advanced_sweep_config_dir,
            config_name="config"
        )

        # 3 lr × 3 batch_size × 2 model.name = 18 runs
        assert len(result.runs) == 18

        # Check that sweep dimensions are in override_map
        assert "training.lr" in result.override_map
        assert "training.batch_size" in result.override_map
        assert "model.name" in result.override_map

        # Verify the sweep values
        assert set(result.override_map["training.lr"]) == {0.001, 0.01, 0.1}
        assert set(result.override_map["training.batch_size"]) == {16, 32, 64}
        assert set(result.override_map["model.name"]) == {"resnet", "vgg"}

        # Verify all runs have the scalar value applied
        for run in result.runs:
            assert run.config.training.optimizer == "adam"

    def test_sweep_file_with_scalars_and_arrays(self, advanced_sweep_config_dir):
        """Test sweep file with both scalar values and array sweep parameters."""
        result = generate_sweep_configs(
            overrides=["+sweeps=comprehensive"],
            config_dir=advanced_sweep_config_dir,
            config_name="config"
        )

        # 5 trials × 3 competitions × 3 models = 45 runs
        assert len(result.runs) == 45

        # Check sweep dimensions
        assert "trial" in result.override_map
        assert "competition" in result.override_map
        assert "agent.code.model" in result.override_map

        # Verify sweep values
        assert result.override_map["trial"] == [0, 1, 2, 3, 4]
        assert set(result.override_map["competition"]) == {
            "random-acts-of-pizza",
            "new-york-city-taxi-fare-prediction",
            "tabular-playground-series-may-2022"
        }
        assert set(result.override_map["agent.code.model"]) == {"o1", "o3-mini", "o1-mini"}

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
            overrides=["+sweeps=comprehensive"],
            config_dir=advanced_sweep_config_dir,
            config_name="config"
        )

        # Verify we have all combinations
        combinations = set()
        for run in result.runs:
            combo = (
                run.config.trial,
                run.config.competition,
                run.config.agent.code.model
            )
            combinations.add(combo)

        # Should have 5 × 3 × 3 = 45 unique combinations
        assert len(combinations) == 45

        # Verify specific combinations exist
        assert (0, "random-acts-of-pizza", "o1") in combinations
        assert (4, "tabular-playground-series-may-2022", "o1-mini") in combinations

    def test_sweep_file_override_dict_with_arrays(self, advanced_sweep_config_dir):
        """Test that override_dict is correctly populated for sweep files with arrays."""
        result = generate_sweep_configs(
            overrides=["+sweeps=mixed"],
            config_dir=advanced_sweep_config_dir,
            config_name="config"
        )

        # Check each run has correct override_dict
        for run in result.runs:
            assert "training.lr" in run.override_dict
            assert "training.batch_size" in run.override_dict
            assert "model.name" in run.override_dict
            assert "training.optimizer" in run.override_dict

            # Verify values are from the sweep
            assert run.override_dict["training.lr"] in [0.001, 0.01, 0.1]
            assert run.override_dict["training.batch_size"] in [16, 32, 64]
            assert run.override_dict["model.name"] in ["resnet", "vgg"]
            assert run.override_dict["training.optimizer"] == "adam"


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
            with patch('pdum.hydra.dynamic.importlib.import_module') as mock_import:
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
            if 'test.base.production' in sys.modules:
                del sys.modules['test.base.production']


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

    def test_run_config_export(self):
        """Test that RunConfig is exported."""
        from pdum import hydra
        assert hasattr(hydra, "RunConfig")

    def test_generated_runs_export(self):
        """Test that GeneratedRuns is exported."""
        from pdum import hydra
        assert hasattr(hydra, "GeneratedRuns")

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from pdum import hydra
        assert hasattr(hydra, "__all__")

        expected_exports = {"__version__", "generate_sweep_configs", "RunConfig", "GeneratedRuns"}
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
