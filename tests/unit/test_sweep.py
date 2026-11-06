"""Tests for sweep expansion into concrete run configs.

Covers: cartesian product count, that each expanded config carries no residual
sweep, that swept values are applied, and that a config with no axes yields one.
"""

from conftest import EXAMPLE_CONFIGS

from cadet.config import expand_sweep, load_config_data


def _braking_dict():
    return load_config_data(str(EXAMPLE_CONFIGS[0]))


def test_cartesian_product_count():
    data = _braking_dict()  # 3 vehicles x 3 objects
    configs = expand_sweep(data)
    assert len(configs) == 9


def test_expanded_configs_have_no_residual_sweep():
    configs = expand_sweep(_braking_dict())
    assert all(cfg.sweep.axes == {} for cfg in configs)


def test_swept_values_applied():
    configs = expand_sweep(_braking_dict())
    combos = {
        (cfg.source.scenario.ego_vehicle, cfg.source.scenario.target_object) for cfg in configs
    }
    assert ("audi.tt", "car") in combos
    assert ("kawasaki.ninja", "bike") in combos
    assert len(combos) == 9


def test_no_axes_yields_single_config():
    data = _braking_dict()
    data["sweep"] = {"axes": {}}
    configs = expand_sweep(data)
    assert len(configs) == 1
