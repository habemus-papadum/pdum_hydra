"""
Hydra utils for structured configuration and sweep generation.

This module provides utilities for managing Hydra configurations with first-class
support for parameter sweeps, building on top of the Hydra framework while
simplifying sweep generation and configuration management.
"""
import itertools
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import OverrideType, ValueType
from hydra.errors import HydraException
from hydra.experimental.callback import Callback
from hydra.types import HydraContext, RunMode
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

DEFAULT_SWEEP_LOCATION = "sweeps"

"""
hydra provides a way to structure config convenient for experimentation with first class support for sweeps.
However, it also provides suboptimal support for cli (better to use `typer`) and job launching

This file selects / re-implements the best bits

Pre-requisites:  Read hydra manual and codebase, grok the concepts of `overrides` and `config groups`

"""


@dataclass
class RunConfig:
    """
    Configuration for a single run in a sweep.

    Attributes
    ----------
    overrides : List[str]
        List of override strings applied to this run (e.g., ["lr=0.01", "batch_size=32"]).
    override_dict : dict
        Dictionary representation of the overrides with parsed values.
    config : DictConfig
        Fully rendered OmegaConf DictConfig with all overrides applied.
    """
    overrides: List[str]
    override_dict: dict
    config: DictConfig


@dataclass
class GeneratedRuns:
    """
    Collection of all runs generated from a sweep configuration.

    Attributes
    ----------
    base_config : DictConfig
        The base Hydra configuration before sweep expansion.
    override_map : dict[str, list]
        Mapping of parameter names to their sweep values. For example,
        {"lr": [0.001, 0.01, 0.1], "batch_size": [16, 32]}.
    runs : List[RunConfig]
        List of all individual run configurations generated from the sweep,
        representing the cartesian product of all sweep parameters.
    """
    base_config: DictConfig
    override_map: dict[str, list]
    runs: List[RunConfig]


def sweep_dict_to_overrides(sweep_dict):
    assert isinstance(sweep_dict, DictConfig)
    for k, v in sweep_dict.items():
        if isinstance(v, ListConfig):
            for i in v:
                assert not isinstance(i, ListConfig) and not isinstance(i, DictConfig)
            yield f"{k}={','.join(map(str, v))}"
        else:
            assert isinstance(v, str) or isinstance(v, int) or isinstance(v, float) or isinstance(v, bool)
            if isinstance(v, bool):
                v = str(v).lower()
            yield f"{k}={v}"


def render_sweep_config(ctx: HydraContext, master_config: DictConfig, sweep_overrides: List[str]) -> DictConfig:
    sweep_config = ctx.config_loader.load_configuration(
        config_name=master_config.hydra.job.config_name,
        overrides=sweep_overrides,
        run_mode=RunMode.RUN,
    )

    # Copy old config cache to ensure we get the same resolved values (for things
    # like timestamps etc). Since `oc.env` does not cache environment variables
    # (but the deprecated `env` resolver did), the entire config should be copied
    OmegaConf.copy_cache(from_config=master_config, to_config=sweep_config)
    with open_dict(sweep_config):
        del sweep_config["hydra"]
    return sweep_config


def render_multiple_sweeps(ctx: HydraContext, config: DictConfig, sweep_location) -> dict:
    rendered = {}
    parser = OverridesParser.create(config_loader=ctx.config_loader)
    ovs = parser.parse_overrides(config.hydra.overrides.task)
    sweep_override = None
    for ov in ovs:
        if (
            ov.key_or_group == sweep_location
            and ov.type == OverrideType.ADD
            and ov.value_type == ValueType.SIMPLE_CHOICE_SWEEP
        ):
            sweep_override = ov
            break
    if sweep_override is None:
        return rendered

    for sweep in sweep_override.sweep_iterator():
        rendered = {
            **rendered,
            **OmegaConf.to_container(
                ctx.config_loader.load_configuration(f"{sweep_location}/{sweep}", [], RunMode.RUN)[
                    sweep_location
                ].parameters
            ),
        }
    return rendered


def extract_sweep_overrides(ctx: HydraContext, cfg: DictConfig, sweep_location="sweeps") -> List[str]:
    parser = OverridesParser.create(config_loader=ctx.config_loader)

    # command line overrides
    ovs = [
        ov_line for ov_line in cfg.hydra.overrides.task if parser.parse_override(ov_line).key_or_group != sweep_location
    ]

    sweeps_dict = render_multiple_sweeps(ctx, cfg, sweep_location)
    sweep_path = sweep_location.split(".")

    # check if sweep_location exists in cfg
    current = cfg
    exists = True
    for p in sweep_path:
        if p in current:
            current = current[p]
        else:
            exists = False
            break
    if exists:
        sweeps_dict = {
            **sweeps_dict,
            **OmegaConf.to_container(current.parameters),
        }
    ovs = ovs + list(sweep_dict_to_overrides(OmegaConf.create(sweeps_dict)))
    # assert that there are no duplicate keys
    keys = set()
    for ov_line in ovs:
        ov = parser.parse_override(ov_line)
        assert ov.key_or_group not in keys, f"Duplicate key in overrides: {ov.key_or_group}"
        keys.add(ov.key_or_group)
    return ovs


def generate_sweep_overrides(ctx: HydraContext, cfg: DictConfig, sweep_location="sweeps"):
    ovs = extract_sweep_overrides(ctx, cfg, sweep_location)
    parser = OverridesParser.create(config_loader=ctx.config_loader)
    choices = []
    final_overrides = OrderedDict()
    override_map = {}

    for ov_line in ovs:
        ov = parser.parse_override(ov_line)
        if ov.is_sweep_override():
            if ov.is_discrete_sweep():
                key = ov.get_key_element()
                sweep = [f"{key}={val}" for val in ov.sweep_string_iterator()]
                final_overrides[key] = sweep
                override_map[key] = list(ov.sweep_iterator())
            else:
                assert ov.value_type is not None
                raise HydraException(f"Sweep type : {ov.value_type.name} is not supported.  Line: {ov_line}")
        else:
            final_overrides[ov.key_or_group] = [ov_line]

    for _, val in final_overrides.items():
        choices.append(val)

    return [list(x) for x in itertools.product(*choices)], override_map


def simple_overrides_to_dict(ctx: HydraContext, overrides: List[str]) -> dict:
    def convert_string(s):
        if not isinstance(s, str):
            return s

        _s = s.strip().lower()

        # Try boolean
        if _s == "true":
            return True
        if _s in "false":
            return False

        # Try int
        try:
            return int(s)
        except ValueError:
            pass

        # Try float
        try:
            return float(s)
        except ValueError:
            pass

        # Return original if no conversion possible
        return s

    parser = OverridesParser.create(config_loader=ctx.config_loader)
    ovs = parser.parse_overrides(overrides)
    d = {}
    for ov in ovs:
        d[ov.get_key_element()] = convert_string(ov.get_value_element_as_str())
    return d


def _generate_sweep_configs(ctx: HydraContext, cfg: DictConfig, sweep_location="sweeps") -> GeneratedRuns:
    overrides, final_overrides = generate_sweep_overrides(ctx, cfg, sweep_location)

    runs = [
        RunConfig(
            override,
            simple_overrides_to_dict(ctx, override),
            render_sweep_config(ctx, cfg, override),
        )
        for override in overrides
    ]
    return GeneratedRuns(cfg, final_overrides, runs)


def generate_sweep_configs(*,
    overrides=None,
    config_name="config",
    config_dir="config",
    sweep_location=DEFAULT_SWEEP_LOCATION,
) -> GeneratedRuns:
    """
    Generate all configurations for a parameter sweep.

    This function loads a Hydra configuration and generates all possible
    run configurations based on sweep parameters defined either in command-line
    overrides or in sweep configuration files.

    Parameters
    ----------
    overrides : List[str], optional
        List of Hydra override strings to apply to the base configuration.
        Examples: ["lr=0.01,0.1", "model=resnet,vgg"]. Default is None.
        To use sweep files, include "+sweeps=<sweep_name>" in overrides.
    config_name : str, optional
        Name of the config file (without extension) to load. Default is "config".
    config_dir : str or Path, optional
        Directory containing Hydra configuration files. Default is "config".
    sweep_location : str, optional
        Config path where sweep parameters are defined. Default is "sweeps".

    Returns
    -------
    GeneratedRuns
        Object containing the base configuration, override mappings, and all
        generated run configurations.

    Notes
    -----
    Sweep files provide a powerful way to define parameter sweeps. Place YAML
    files in the `{config_dir}/sweeps/` directory with a `parameters` section.

    - **Array values** create sweep dimensions (cartesian product)
    - **Scalar values** are applied to all runs without sweeping

    Array syntax (creates sweep dimension)::

        parameters:
          training.lr: [0.001, 0.01, 0.1]
          model.name: [resnet, vgg]

    Alternative array syntax::

        parameters:
          competition:
            - random-acts-of-pizza
            - taxi-fare-prediction

    Scalar values (no sweep, applied to all runs)::

        parameters:
          training.epochs: 100
          agent.temperature: 1.0

    Examples
    --------
    >>> # Generate sweep from command-line overrides
    >>> runs = generate_sweep_configs(
    ...     overrides=["lr=0.001,0.01", "batch_size=16,32"],
    ...     config_dir="/path/to/config"
    ... )
    >>> print(len(runs.runs))  # Number of runs in sweep
    4
    >>> print(runs.override_map)
    {'lr': [0.001, 0.01], 'batch_size': [16, 32]}

    >>> # Use a sweep file: config/sweeps/my_sweep.yaml
    >>> # File contents:
    >>> # parameters:
    >>> #   trial: [0, 1, 2, 3, 4]
    >>> #   model: [resnet, vgg, alexnet]
    >>> #   lr: [0.001, 0.01]
    >>> #   batch_size: 32  # scalar, no sweep
    >>>
    >>> runs = generate_sweep_configs(
    ...     overrides=["+sweeps=my_sweep"],
    ...     config_dir="/path/to/config"
    ... )
    >>> print(len(runs.runs))  # 5 trials × 3 models × 2 lrs = 30
    30

    >>> # Access individual run configurations
    >>> for run in runs.runs:
    ...     print(run.override_dict)
    ...     # All runs have batch_size=32 (scalar value)
    ...     # Use run.config to access the full OmegaConf DictConfig
    ...     train_model(run.config)

    >>> # Combine sweep file with additional overrides
    >>> runs = generate_sweep_configs(
    ...     overrides=["+sweeps=my_sweep", "optimizer=adam,sgd"],
    ...     config_dir="/path/to/config"
    ... )
    >>> print(len(runs.runs))  # 30 × 2 optimizers = 60
    60
    """
    if overrides is None:
        overrides = []
    with initialize_config_dir(config_dir=str(config_dir), job_name="generate_sweep_configs", version_base=None):
        gh = GlobalHydra.instance()
        cb = Callback()
        ctx = HydraContext(gh.config_loader(), cb)
        base_cfg = gh.hydra.compose_config(config_name=config_name, overrides=overrides, run_mode=RunMode.MULTIRUN)
        return _generate_sweep_configs(ctx, base_cfg, sweep_location)


__version__ = "0.3.0"

__all__ = ["__version__", "generate_sweep_configs", "RunConfig", "GeneratedRuns"]
