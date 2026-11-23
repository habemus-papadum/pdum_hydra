"""Tests for compose, find_config_dirs and config directories given as an ordered list."""

import threading
import warnings

import pytest
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException, InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pdum.hydra import compose, find_config_dirs, generate_sweep_configs


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def layered(tmp_path):
    """A shared config directory and a nearer one that overrides part of it.

    repo/.git/
    repo/experiments/config/          base.yaml, config.yaml, model/{small,big}.yaml
    repo/experiments/alice/trial/config/   config.yaml, model/{big,tiny}.yaml
    """
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    shared = repo / "experiments" / "config"
    near = repo / "experiments" / "alice" / "trial" / "config"
    write(shared / "base.yaml", "defaults:\n  - model: small\n  - _self_\nlookback_days: 20\ncost_bps: 1.0\n")
    write(shared / "config.yaml", "defaults:\n  - base\n  - _self_\nname: shared\n")
    write(shared / "model" / "small.yaml", "layers: 2\nwhere: shared\n")
    write(shared / "model" / "big.yaml", "layers: 8\nwhere: shared\n")
    write(near / "config.yaml", "defaults:\n  - base\n  - _self_\nname: trial\nlookback_days: 60\n")
    write(near / "model" / "big.yaml", "layers: 9\nwhere: near\n")
    write(near / "model" / "tiny.yaml", "layers: 1\nwhere: near\n")
    return {"repo": repo, "shared": shared, "near": near, "script": write(near.parent / "run.py", "")}


class TestCompose:
    def test_single_directory(self, layered):
        cfg = compose(config_dir=layered["shared"])
        assert isinstance(cfg, DictConfig)
        assert OmegaConf.to_container(cfg) == {
            "model": {"layers": 2, "where": "shared"},
            "lookback_days": 20,
            "cost_bps": 1.0,
            "name": "shared",
        }
        assert "hydra" not in cfg

    def test_overrides_and_config_name(self, layered):
        cfg = compose(overrides=["cost_bps=2.5", "model=big"], config_name="base", config_dir=layered["shared"])
        assert cfg.cost_bps == 2.5
        assert cfg.model.layers == 8
        assert "name" not in cfg

    def test_plain_yaml_without_defaults(self, tmp_path):
        write(tmp_path / "config.yaml", "data:\n  root: /data\n")
        assert compose(config_dir=tmp_path).data.root == "/data"

    def test_sweep_override_rejected(self, layered):
        with pytest.raises(HydraException):
            compose(overrides=["cost_bps=1,2"], config_dir=layered["shared"])

    def test_relative_directory_resolves_against_cwd(self, layered, monkeypatch):
        monkeypatch.chdir(layered["shared"].parent)
        assert compose(config_dir="config").name == "shared"

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope"):
            compose(config_dir=tmp_path / "nope")

    def test_empty_directory_list(self):
        with pytest.raises(ValueError):
            compose(config_dir=[])

    def test_compose_again_reads_the_files_again(self, tmp_path):
        file = write(tmp_path / "config.yaml", "value: 1\n")
        assert compose(config_dir=tmp_path).value == 1
        file.write_text("value: 2\n")
        assert compose(config_dir=tmp_path).value == 2

    def test_package_source(self, tmp_path, monkeypatch):
        write(tmp_path / "pdum_hydra_demo_conf" / "__init__.py", "")
        write(tmp_path / "pdum_hydra_demo_conf" / "base.yaml", "workers: 4\nport: 8000\n")
        deployment = tmp_path / "deployment"
        write(deployment / "config.yaml", "defaults:\n  - base\n  - _self_\nport: 9000\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        cfg = compose(config_dir=[deployment, "pkg://pdum_hydra_demo_conf"])
        assert OmegaConf.to_container(cfg) == {"workers": 4, "port": 9000}


class TestLayeredDirectories:
    def test_nearest_primary_config_wins_and_extends_base(self, layered):
        cfg = compose(config_dir=[layered["near"], layered["shared"]])
        assert cfg.name == "trial"
        assert cfg.lookback_days == 60  # set by the near config
        assert cfg.cost_bps == 1.0  # inherited from base.yaml in the shared directory
        assert cfg.model.where == "shared"  # model/small.yaml exists only in the shared directory

    def test_group_option_in_both_directories_takes_the_nearest(self, layered):
        cfg = compose(overrides=["model=big"], config_dir=[layered["near"], layered["shared"]])
        assert OmegaConf.to_container(cfg.model) == {"layers": 9, "where": "near"}

    def test_group_option_only_in_the_near_directory(self, layered):
        cfg = compose(overrides=["model=tiny"], config_dir=[layered["near"], layered["shared"]])
        assert cfg.model.layers == 1

    def test_order_decides(self, layered):
        cfg = compose(overrides=["model=big"], config_dir=[layered["shared"], layered["near"]])
        assert cfg.name == "shared"
        assert cfg.model.where == "shared"

    def test_sweep_over_layered_directories(self, layered):
        sweep = generate_sweep_configs(
            overrides=["model=tiny,big", "cost_bps=0.5,1.0,2.0"],
            config_dir=[layered["near"], layered["shared"]],
        )
        assert len(sweep) == 6
        assert sweep.axes == {"model": ["tiny", "big"], "cost_bps": [0.5, 1.0, 2.0]}
        last = sweep[-1]
        assert last.params == {"model": "big", "cost_bps": 2.0}
        assert last.config.model.where == "near"
        assert last.config.name == "trial"
        assert last.config.lookback_days == 60

    def test_sweep_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_sweep_configs(config_dir=tmp_path / "nope")


class TestFindConfigDirs:
    def test_nearest_first_up_to_the_repository_root(self, layered):
        assert find_config_dirs(layered["near"].parent) == [layered["near"], layered["shared"]]

    def test_start_may_be_a_file(self, layered):
        assert find_config_dirs(layered["script"]) == [layered["near"], layered["shared"]]

    def test_result_composes(self, layered):
        assert compose(config_dir=find_config_dirs(layered["script"])).name == "trial"

    def test_explicit_stop_is_included(self, layered):
        trial = layered["near"].parent
        assert find_config_dirs(trial, stop=trial) == [layered["near"]]
        assert find_config_dirs(trial, stop=trial.parent) == [layered["near"]]
        assert find_config_dirs(trial, stop=layered["shared"].parent) == [layered["near"], layered["shared"]]

    def test_does_not_look_above_the_repository_root(self, layered):
        write(layered["repo"].parent / "config" / "config.yaml", "name: outside\n")
        assert find_config_dirs(layered["script"]) == [layered["near"], layered["shared"]]

    def test_other_directory_name(self, layered):
        write(layered["near"].parent / "conf" / "config.yaml", "name: conf\n")
        assert find_config_dirs(layered["script"], name="conf") == [layered["near"].parent / "conf"]

    def test_none_found(self, layered):
        assert find_config_dirs(layered["script"], name="absent") == []

    def test_outside_a_repository_needs_stop(self, tmp_path):
        with pytest.raises(ValueError, match="stop"):
            find_config_dirs(tmp_path)

    def test_stop_must_be_an_ancestor(self, layered, tmp_path):
        other = tmp_path / "elsewhere"
        other.mkdir()
        with pytest.raises(ValueError, match="parents"):
            find_config_dirs(layered["script"], stop=other)

    def test_missing_start(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_config_dirs(tmp_path / "nope")


class TestNoGlobalState:
    """Hydra's own initialize_* functions install a process-wide singleton. These must not."""

    def test_global_hydra_untouched(self, layered):
        assert not GlobalHydra.instance().is_initialized()
        compose(config_dir=layered["shared"])
        sweep = generate_sweep_configs(overrides=["cost_bps=1,2"], config_dir=layered["shared"])
        assert sweep[1].config.cost_bps == 2
        assert not GlobalHydra.instance().is_initialized()

    def test_inside_an_open_hydra_context(self, layered, tmp_path):
        write(tmp_path / "outer" / "config.yaml", "name: outer\n")
        with initialize_config_dir(config_dir=str(tmp_path / "outer")):
            assert compose(config_dir=layered["shared"]).name == "shared"
            sweep = generate_sweep_configs(overrides=["cost_bps=1,2"], config_dir=layered["shared"])
            assert [run.config.cost_bps for run in sweep] == [1, 2]
            # The surrounding context still works and still sees its own config.
            assert hydra_compose("config").name == "outer"

    def test_from_several_threads(self, layered):
        results, errors = [], []

        def work(i):
            try:
                for _ in range(10):
                    cfg = compose(overrides=[f"lookback_days={i}"], config_dir=[layered["near"], layered["shared"]])
                    results.append((cfg.lookback_days, i))
            except Exception as error:  # pragma: no cover - reported by the assertion below
                errors.append(error)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert len(results) == 80
        assert all(got == wanted for got, wanted in results)


class TestInstantiate:
    """hydra.utils.instantiate on a composed config. These pin the behaviour of Hydra 1.4."""

    @pytest.fixture
    def config_dir(self, tmp_path):
        write(tmp_path / "config.yaml", "defaults:\n  - window: week\n  - _self_\n")
        write(tmp_path / "window" / "week.yaml", "_target_: datetime.timedelta\ndays: 7\n")
        write(tmp_path / "window" / "day.yaml", "_target_: datetime.timedelta\nhours: 24\n")
        return tmp_path

    def test_builds_the_object_named_in_the_config(self, config_dir):
        import datetime

        window = instantiate(compose(config_dir=config_dir).window, _execution_whitelist_="datetime.timedelta")
        assert window == datetime.timedelta(days=7)

    def test_group_choice_selects_the_class_arguments(self, config_dir):
        import datetime

        cfg = compose(overrides=["window=day"], config_dir=config_dir)
        assert instantiate(cfg.window, _execution_whitelist_="datetime.*") == datetime.timedelta(days=1)

    def test_call_site_arguments(self, config_dir):
        import datetime

        cfg = compose(config_dir=config_dir)
        window = instantiate(cfg.window, hours=12, _execution_whitelist_="datetime.timedelta")
        assert window == datetime.timedelta(days=7, hours=12)

    def test_unknown_argument_is_reported_by_the_target(self, config_dir):
        cfg = compose(overrides=["+window.dayz=3"], config_dir=config_dir)
        with pytest.raises(TypeError, match="dayz"):
            instantiate(cfg.window, _execution_whitelist_="datetime.timedelta")

    def test_target_outside_the_whitelist_is_refused(self, config_dir):
        cfg = compose(config_dir=config_dir)
        with pytest.raises(InstantiationException):
            instantiate(cfg.window, _execution_whitelist_="fractions.*")

    def test_without_a_whitelist_hydra_warns(self, config_dir):
        cfg = compose(config_dir=config_dir)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            instantiate(cfg.window)
        assert any("_execution_whitelist_" in str(w.message) for w in caught)
