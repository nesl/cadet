"""Named weather presets for CADET sources.

What this is for:
    A single ``source.weather`` field selects one of these presets by name
    (``clear`` | ``light`` | ``high``). The presets themselves live here, as
    package data, so experiment configs stay small and reproducible and so that
    tweaking a preset changes every experiment that references it.

What it contains:
    - ``WEATHER_PRESETS``: name -> CARLA ``WeatherParameters`` knobs.
    - ``WeatherName``: the ``Literal`` of valid preset names, used by the schema.
    - ``get_preset``: resolve a name to its knob dict.

How it fits into CADET:
    The config schema (``cadet.config.schema.CarlaSource``) validates the
    ``weather`` field against ``WeatherName``. A CARLA source adapter (added in a
    later phase) calls :func:`get_preset` to apply the knobs to the simulator.
    Replaces the old separate ``occlusion`` field; "visibility" is now one axis.
"""

from typing import Dict

from typing_extensions import Literal

# Preset knobs mirror CARLA's WeatherParameters fields. Values are the project's
# established visibility tiers (baseline clear, dusk+drizzle, night+storm).
WEATHER_PRESETS: Dict[str, Dict[str, float]] = {
    "clear": {
        "cloudiness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wetness": 0.0,
        "sun_altitude_angle": 50.0,
    },
    "light": {  # dusk + drizzle
        "cloudiness": 70.0,
        "precipitation": 30.0,
        "precipitation_deposits": 40.0,
        "wetness": 50.0,
        "sun_altitude_angle": 6.0,
    },
    "high": {  # night + storm
        "cloudiness": 100.0,
        "precipitation": 90.0,
        "precipitation_deposits": 90.0,
        "wetness": 100.0,
        "sun_altitude_angle": -5.0,
    },
}

WeatherName = Literal["clear", "light", "high"]


def get_preset(name: str) -> Dict[str, float]:
    """Return the weather knobs for ``name`` or raise ``KeyError`` if unknown."""
    return WEATHER_PRESETS[name]
