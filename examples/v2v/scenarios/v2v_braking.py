#!/usr/bin/env python3
"""V2V braking scenario: spawn the leader (V1) + follower (V2), then idle.

The scenario only places the two cars and holds; the CADET V2V client drives BOTH
of them (follower control + the scripted leader + the in-process intent), exactly as
the original single-process run did. Spawn points and the follower model come from
the CADET config/env (CADET_SCENARIO_VEHICLE and the CADET_V2V_* spawn vars the V2V
sim hook sets), not from pylot - so this file has no dependency on the pylot tree.
"""

import os

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    ActorTransformSetter,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.basic_scenario import BasicScenario


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# Follower (V2) model + leader (V1) model, and the spawn geometry - all from the
# environment the V2V sim hook sets, with the original v2v_braking_params defaults.
VEHICLE_NAME = os.environ.get("CADET_SCENARIO_VEHICLE", "audi.tt")      # follower (V2)
LEADER_BP = os.environ.get("CADET_V2V_LEADER", "vehicle.tesla.model3")  # leader (V1)
SPAWN_X = _envf("CADET_V2V_SPAWN_X", -122.958664)
SPAWN_Y = _envf("CADET_V2V_SPAWN_Y", 100.0)
SPAWN_Z = _envf("CADET_V2V_SPAWN_Z", 0.60)
SPAWN_YAW = _envf("CADET_V2V_SPAWN_YAW", -90.0)
V2_HEADWAY_M = _envf("CADET_V2V_HEADWAY_M", 30.0)


# =====================
# Scenario
# =====================
class V2VBrakingScenario(BasicScenario):
    """
    Minimal V2V:
      - V1 lane-snapped; autopilot OFF; single behavior: Start→Cruise(keep velocity)→Stop@Y
      - V2 spawned behind V1 and left unmanaged (external controller handles it)
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=100):
        self.world = world
        self.timeout = timeout
        super(V2VBrakingScenario, self).__init__(
            "V2VBrakingScenario",
            ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        self.v1 = None
        self.v2 = None

    def _initialize_actors(self, _config):
        # V1 spawn from config
        v1_spawn = carla.Transform(
            carla.Location(x=SPAWN_X, y=SPAWN_Y, z=SPAWN_Z),
            carla.Rotation(pitch=0.0, yaw=SPAWN_YAW, roll=0.0)
        )
        self.v1 = CarlaDataProvider.request_new_actor(LEADER_BP, v1_spawn)
        if self.v1 is None:
            # retry with a tiny nudge to avoid overlaps
            v1_spawn.location.x += 0.3
            v1_spawn.location.y += 0.3
            self.v1 = CarlaDataProvider.request_new_actor(LEADER_BP, v1_spawn)
        assert self.v1 is not None, "Failed to spawn V1"

        # Snap V1 to driving lane center
        w = self.world.get_map().get_waypoint(
            self.v1.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if w is not None:
            self.v1.set_transform(w.transform)
        v1_tf = self.v1.get_transform()

        # Make sure TM/autopilot is OFF so we fully control V1
        self.v1.set_autopilot(False)
        fwd0 = self.v1.get_transform().get_forward_vector()
        kick = 1.0  # m/s (small nudge)
        self.v1.set_target_velocity(carla.Vector3D(fwd0.x * kick, fwd0.y * kick, 0.0))
        self.v1.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
        print(f"[KICK] set_target_velocity ~{kick:.1f} m/s", flush=True)

        # V2 spawn at headway behind V1 (left unmanaged)
        fwd = v1_tf.get_forward_vector()
        back = carla.Location(x=-fwd.x * V2_HEADWAY_M, y=-fwd.y * V2_HEADWAY_M, z=0.0)
        v2_tf = carla.Transform(
            carla.Location(
                x=v1_tf.location.x + back.x,
                y=v1_tf.location.y + back.y,
                z=v1_tf.location.z
            ),
            v1_tf.rotation
        )
        self.v2 = CarlaDataProvider.request_new_actor(f"vehicle.{VEHICLE_NAME}", v2_tf)
        if self.v2 is None:
            v2_tf.location.x -= 0.5
            v2_tf.location.y -= 0.5
            self.v2 = CarlaDataProvider.request_new_actor(f"vehicle.{VEHICLE_NAME}", v2_tf)
        assert self.v2 is not None, "Failed to spawn V2"
        self.v2.set_autopilot(False)

        print(f"[Scenario] V1 snapped to lane at {v1_tf.location} (yaw={v1_tf.rotation.yaw:.1f})")
        print(f"[Scenario] V2 spawn headway={V2_HEADWAY_M:.1f} m behind V1 (no SR control)")

    def _create_behavior(self):

        class WaitUntilTimeout(py_trees.behaviour.Behaviour):
            def __init__(self, timeout_s=100):
                super().__init__(name=f"Wait({timeout_s}s)")
                self.timeout_s = timeout_s
                self.t0 = None

            def initialise(self):
                self.t0 = GameTime.get_time()

            def update(self):
                if GameTime.get_time() - self.t0 >= self.timeout_s:
                    return py_trees.common.Status.SUCCESS
                return py_trees.common.Status.RUNNING

        seq = py_trees.composites.Sequence("ScenarioSequence")

        # Lock initial poses, then just wait while the client controls the cars
        seq.add_child(ActorTransformSetter(self.v1, self.v1.get_transform()))
        seq.add_child(ActorTransformSetter(self.v2, self.v2.get_transform()))
        seq.add_child(WaitUntilTimeout(timeout_s=self.timeout))  # uses scenario timeout

        # Optional: clean up V1 when done (V2 left for client)
        seq.add_child(ActorDestroy(self.v1))
        return seq


    def _create_test_criteria(self):
        return [CollisionTest(self.v1), CollisionTest(self.v2)]

    def remove_all_actors(self):
        super().remove_all_actors()
        for a in [self.v1]:
            try:
                if a is not None:
                    CarlaDataProvider.remove_actor_by_id(a.id)
            except Exception:
                pass
        # Intentionally do NOT destroy V2
