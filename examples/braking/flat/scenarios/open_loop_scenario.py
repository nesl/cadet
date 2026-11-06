#!/usr/bin/env python
"""Open-loop straight-line driving scenario (CADET braking overlay).

What this is for:
    A ScenarioRunner scenario that spawns a single vehicle, drives it forward a
    fixed distance at a constant velocity, then destroys it. It is the simplest
    of the CADET braking-overlay scenarios — no obstacle — used to sanity-check
    the sim-hook lifecycle.

What it contains:
    - :class:`OpenLoopScenario`: a ``BasicScenario`` that spawns the vehicle in
      ``_initialize_actors`` and drives-then-destroys it in ``_create_behavior``.

How it fits into CADET:
    Loaded by ScenarioRunner via ``OpenLoop.xml`` (scenario type
    ``OpenLoopScenario``). The CADET orchestrator starts ScenarioRunner as the
    braking app's sim hook; see ``examples/braking/flat/README.md``. This file uses the
    ScenarioRunner (``srunner``) API, which is an external dependency.
"""

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    ActorTransformSetter,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

# Constant forward velocity (m/s) applied to the vehicle, and how far it drives
# before the scenario destroys it and ends.
VEHICLE_SPEED_MPS = 10.0
DRIVE_DISTANCE_M = 50


class OpenLoopScenario(BasicScenario):
    """
    Scenario class for a high-speed vehicle moving in a straight line.
    :param world: CARLA world object
    :param ego_vehicles: List of ego vehicles (not used here since there's no ego vehicle)
    :param config: Scenario configuration (ScenarioConfiguration)
    :param randomize: Select random parameters (optional, default=False)
    :param debug_mode: Enable debug mode for detailed output (optional, default=False)
    :param criteria_enable: Enable evaluation criteria (optional, default=True)
    :param timeout: Overall scenario timeout in seconds (default=1000)
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=100):
        self._map = CarlaDataProvider.get_map()
        self.vehicle_speed = VEHICLE_SPEED_MPS
        self.timeout = timeout

        for actor in world.get_actors():
            #if actor.type_id.startswith("vehicle."):
            actor.destroy()

        # Call constructor of BasicScenario
        super(OpenLoopScenario, self).__init__(
            "OpenLoopScenario",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable)

        self.vehicle = None
        #self._initialize_actors(config)

    def _initialize_actors(self, config):
        """
        Initialize the leading vehicle with a randomized or predefined spawn point.
        """
        # Get a list of spawn points and select one (e.g., random or specific index)
        spawn_points = self._map.get_spawn_points()
        for i, spawn_point in enumerate(spawn_points):
            print(f"Spawn Point {i}: Location = {spawn_point.location}, Rotation = {spawn_point.rotation}")
        #spawn_point = random.choice(spawn_points)  # or use spawn_points[5] for a specific index
        spawn_location = carla.Transform(
            carla.Location(x=9.530025, y=302.570007, z=0.500000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        # Optionally adjust the Z coordinate if necessary
        #spawn_location.location.z += 1.0

        # Try spawning the vehicle with a different model if necessary
        try:
            self.vehicle = CarlaDataProvider.request_new_actor(model = 'vehicle.audi.a2', spawn_point=spawn_location)
        except RuntimeError as e:
            print(f"Failed to spawn 'vehicle.nissan.patrol' at {spawn_point}. Error: {e}")
            print("Trying with a different vehicle model...")
            self.vehicle = CarlaDataProvider.request_new_actor('vehicle.audi.a2', spawn_location)

        self.vehicle.set_simulate_physics(enabled=True)
        #self.vehicle.set_autopilot(enabled=True)
        self.vehicle.set_target_velocity(carla.Vector3D(self.vehicle_speed, 0, 0))
        self.other_actors.append(self.vehicle)

    def _create_behavior(self):
        """
        Define the behavior of the high-speed vehicle.
        """
        # Sequence of behaviors for the vehicle
        behavior = py_trees.composites.Sequence("VehicleBehaviorSequence")

        # Set the vehicle to the initial transform and apply the speed
        start_transform = ActorTransformSetter(self.vehicle, self.vehicle.get_transform())
        drive_distance_condition = DriveDistance(self.vehicle, distance=DRIVE_DISTANCE_M, name="DriveDistance")
        stop_behavior = ActorDestroy(self.vehicle)

        # Build the behavior sequence
        behavior.add_child(start_transform)
        print("start transform")
        behavior.add_child(drive_distance_condition)
        behavior.add_child(stop_behavior)
        print("stop")

        return behavior

    def _create_test_criteria(self):
        """
        Setup evaluation criteria (optional, could be expanded).
        """
        criteria = []

        # Example criterion: Ensure the vehicle does not collide
        collision_criterion = CollisionTest(self.vehicle)
        criteria.append(collision_criterion)

        return criteria

    def remove_all_actors(self):
        """
        Remove all actors when the scenario ends.
        """
        super().remove_all_actors()
        if self.vehicle is not None:
            CarlaDataProvider.remove_actor_by_id(self.vehicle.id)
