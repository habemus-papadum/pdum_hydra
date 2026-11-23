"""
Hydra utils for structured configuration and sweep generation.

This module provides utilities for managing Hydra configurations with first-class
support for parameter sweeps, building on top of the Hydra framework while
simplifying sweep generation and configuration management.

A sweep is a grid: it has **axes** (each swept parameter and its values) and **runs**
(one per point on the grid). A run has **params** (its settings as a dict), **hydra_overrides**
(the same settings as Hydra override strings) and **config** (the fully composed config).

Runs are produced lazily and a run's config is composed only when first accessed, so
large sweeps can be filtered cheaply on ``run.params`` before any config is built.
"""

import itertools
import math
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property, partial
from typing import Any, List, overload

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

# One choice along one dimension of the sweep: (hydra override string, params key, typed value).
_Choice = tuple[str, str, Any]


class Run:
    """
    A single run in a sweep: one point on the sweep's grid.

    Runs are created by a :class:`Sweep`; you normally don't construct them yourself.

    Attributes
    ----------
    index : int
        Position of this run in its sweep (``sweep[run.index]`` is an equivalent run).
    params : dict[str, Any]
        This run's settings as ``{key: value}``, with values typed by Hydra's parser
        (e.g. ``{"training.lr": 0.01, "model.layers": 50}``). Includes both swept
        parameters and fixed ones applied to every run. Cheap: use it to filter runs
        before touching :attr:`config`.
    hydra_overrides : list[str]
        The same settings as Hydra override strings
        (e.g. ``["training.lr=0.01", "model.layers=50"]``), suitable for passing back to
        Hydra's command line or ``compose``.
    config : DictConfig
        The fully composed config with this run's overrides applied. Composed on first
        access and cached on this ``Run`` object.

    Notes
    -----
    Pickling a ``Run`` (e.g. to send it to a worker process) composes its config first,
    so the unpickled copy carries the config with it.
    """

    def __init__(
        self,
        index: int,
        params: dict[str, Any],
        hydra_overrides: list[str],
        compose: Callable[[], DictConfig] | None,
    ):
        self.index = index
        self.params = params
        self.hydra_overrides = hydra_overrides
        self._compose = compose

    @cached_property
    def config(self) -> DictConfig:
        if self._compose is None:
            raise RuntimeError("This Run has no way to compose its config.")
        return self._compose()

    def __getstate__(self) -> dict:
        self.config  # compose (and cache) before shipping; the composer itself isn't picklable
        state = self.__dict__.copy()
        state["_compose"] = None
        return state

    def __repr__(self) -> str:
        return f"Run(index={self.index}, params={self.params!r})"


class Sweep(Sequence[Run]):
    """
    A parameter sweep: the cartesian product of all swept parameters.

    A ``Sweep`` is a lazy, re-iterable sequence of :class:`Run` objects. Nothing is
    composed up front: ``len(sweep)`` and ``sweep[i]`` are computed directly from the
    axes, and each run's config is composed only when ``run.config`` is first read.

    Each access (iteration or indexing) creates a fresh ``Run``. A run caches its own
    config, so keep a reference to the run if you need its config more than once.

    Attributes
    ----------
    base_config : DictConfig
        The composed configuration before sweep expansion (includes the ``hydra`` node).
    axes : dict[str, list]
        The swept parameters and their values, in sweep order, e.g.
        ``{"training.lr": [0.001, 0.01, 0.1], "model.layers": [50, 101]}``. Fixed
        (non-swept) settings don't appear here, but do appear in each run's ``params``.
    runs : Sweep
        This sweep. Provided so ``for run in sweep.runs`` reads naturally;
        ``for run in sweep`` is equivalent.
    """

    def __init__(
        self,
        base_config: DictConfig,
        axes: dict[str, list],
        dimensions: list[list[_Choice]],
        compose: Callable[[list[str]], DictConfig],
    ):
        self.base_config = base_config
        self.axes = axes
        self._dimensions = dimensions
        self._compose = compose
        self._len = math.prod(len(d) for d in dimensions)

    @property
    def runs(self) -> "Sweep":
        return self

    def __len__(self) -> int:
        return self._len

    @overload
    def __getitem__(self, index: int) -> Run: ...
    @overload
    def __getitem__(self, index: slice) -> list[Run]: ...
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._len))]
        if index < 0:
            index += self._len
        if not 0 <= index < self._len:
            raise IndexError(f"Run index out of range (sweep has {self._len} runs)")
        # Mixed-radix decomposition, matching itertools.product order (last dimension varies fastest).
        choices = []
        rest = index
        for dim in reversed(self._dimensions):
            rest, i = divmod(rest, len(dim))
            choices.append(dim[i])
        choices.reverse()
        return self._make_run(index, choices)

    def __iter__(self) -> Iterator[Run]:
        for index, choices in enumerate(itertools.product(*self._dimensions)):
            yield self._make_run(index, choices)

    def _make_run(self, index: int, choices: Sequence[_Choice]) -> Run:
        hydra_overrides = [override for override, _, _ in choices]
        params = {key: value for _, key, value in choices}
        return Run(index, params, hydra_overrides, partial(self._compose, hydra_overrides))

    def __repr__(self) -> str:
        return f"Sweep(len={self._len}, axes={self.axes!r})"


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


def build_sweep_dimensions(
    ctx: HydraContext, cfg: DictConfig, sweep_location="sweeps"
) -> tuple[list[list[_Choice]], dict[str, list]]:
    """
    Turn a composed config's overrides into the dimensions of a sweep.

    Returns
    -------
    dimensions : list[list[tuple[str, str, Any]]]
        One list per override key. Each entry is ``(hydra_override, params_key, value)``.
        Swept keys have one entry per value; fixed keys have a single entry.
    axes : dict[str, list]
        ``{params_key: values}`` for the swept keys only.
    """
    ovs = extract_sweep_overrides(ctx, cfg, sweep_location)
    parser = OverridesParser.create(config_loader=ctx.config_loader)
    dimensions: OrderedDict[str, list[_Choice]] = OrderedDict()
    axes = {}

    for ov_line in ovs:
        ov = parser.parse_override(ov_line)
        hydra_key = ov.get_key_element()  # keeps Hydra's +/++/~ prefix
        key = hydra_key.lstrip("+")  # "+foo" / "++foo" just set foo
        if ov.is_sweep_override():
            if not ov.is_discrete_sweep():
                assert ov.value_type is not None
                raise HydraException(f"Sweep type : {ov.value_type.name} is not supported.  Line: {ov_line}")
            values = list(ov.sweep_iterator())
            strings = list(ov.sweep_string_iterator())
            dimensions[key] = [(f"{hydra_key}={s}", key, v) for s, v in zip(strings, values, strict=True)]
            axes[key] = values
        else:
            dimensions[key] = [(ov_line, key, ov.value())]

    return list(dimensions.values()), axes


def _build_sweep(ctx: HydraContext, cfg: DictConfig, sweep_location="sweeps") -> Sweep:
    dimensions, axes = build_sweep_dimensions(ctx, cfg, sweep_location)
    return Sweep(cfg, axes, dimensions, partial(render_sweep_config, ctx, cfg))


def generate_sweep_configs(
    *,
    overrides=None,
    config_name="config",
    config_dir="config",
    sweep_location=DEFAULT_SWEEP_LOCATION,
) -> Sweep:
    """
    Generate a lazy parameter sweep from a Hydra configuration.

    Loads a Hydra configuration and returns a :class:`Sweep` covering every
    combination of the sweep parameters, defined either in overrides or in sweep
    configuration files. Runs are produced lazily and each run's config is composed
    only when ``run.config`` is first accessed, so you can skip runs based on
    ``run.params`` without paying for config composition.

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
    Sweep
        A lazy sequence of :class:`Run` objects, with the sweep's ``axes`` and
        ``base_config``.

    Raises
    ------
    HydraException
        If an override uses a non-discrete sweep (e.g. ``range`` or ``interval``).

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
    >>> sweep = generate_sweep_configs(
    ...     overrides=["lr=0.001,0.01", "batch_size=16,32"],
    ...     config_dir="/path/to/config"
    ... )
    >>> len(sweep)  # computed without composing any config
    4
    >>> sweep.axes
    {'lr': [0.001, 0.01], 'batch_size': [16, 32]}

    >>> # Filter cheaply on params, then (if needed) on the full config
    >>> for run in sweep:
    ...     if run.params["lr"] == 0.01 and run.params["batch_size"] == 32:
    ...         continue  # skipped: no config composed
    ...     if not looks_useful(run.config):  # composed here, cached on run
    ...         continue
    ...     train_model(run.config)

    >>> # Use a sweep file: config/sweeps/my_sweep.yaml
    >>> # File contents:
    >>> # parameters:
    >>> #   trial: [0, 1, 2, 3, 4]
    >>> #   model: [resnet, vgg, alexnet]
    >>> #   lr: [0.001, 0.01]
    >>> #   batch_size: 32  # scalar, no sweep
    >>>
    >>> sweep = generate_sweep_configs(
    ...     overrides=["+sweeps=my_sweep"],
    ...     config_dir="/path/to/config"
    ... )
    >>> len(sweep)  # 5 trials × 3 models × 2 lrs = 30
    30

    >>> # Combine sweep file with additional overrides
    >>> sweep = generate_sweep_configs(
    ...     overrides=["+sweeps=my_sweep", "optimizer=adam,sgd"],
    ...     config_dir="/path/to/config"
    ... )
    >>> len(sweep)  # 30 × 2 optimizers = 60
    60
    """
    if overrides is None:
        overrides = []
    with initialize_config_dir(config_dir=str(config_dir), job_name="generate_sweep_configs", version_base=None):
        gh = GlobalHydra.instance()
        cb = Callback()
        ctx = HydraContext(gh.config_loader(), cb)
        base_cfg = gh.hydra.compose_config(config_name=config_name, overrides=overrides, run_mode=RunMode.MULTIRUN)
        # Runs compose their configs lazily, after this `with` block has exited. That works because
        # the config loader captured in `ctx` keeps its own config repository and doesn't need
        # Hydra's global state.
        return _build_sweep(ctx, base_cfg, sweep_location)


__version__ = "0.4.0-alpha"

__all__ = ["__version__", "generate_sweep_configs", "Run", "Sweep"]
