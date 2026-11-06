#!/usr/bin/env python
"""At-speed emergency-braking scenario (CADET braking overlay).

What this is for:
    Same as the open-loop braking scenario, but the ego spawns further down the
    lane (below the ego's one-time speed-kick gate at ``y < 90``) so it is already
    at cruising speed when it approaches the obstacle — the "at-speed" variant used
    for braking-distance measurements.

What it contains:
    - :class:`AtSpeedBrakingScenario`: spawns ego + obstacle in
      ``_initialize_actors`` (obstacle class chosen by ``CADET_SCENARIO_OBJECT``)
      and drives-then-destroys the ego in ``_create_behavior``.

How it fits into CADET:
    Loaded by ScenarioRunner via ``AtSpeed.xml`` (scenario type
    ``AtSpeedBrakingScenario``). The CADET orchestrator starts ScenarioRunner as
    the braking app's sim hook; see ``examples/braking/flat/README.md``. This file uses
    the ScenarioRunner (``srunner``) API, which is an external dependency.
"""

import os

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    ActorTransformSetter,
    KeepVelocity,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

# The ego blueprint tail and the obstacle class, supplied by the CADET sim hook
# from source.scenario (see examples/braking/flat/app.py). Defaults keep the file
# runnable by hand with ScenarioRunner alone.
VEHICLE_NAME = os.environ.get("CADET_SCENARIO_VEHICLE", "audi.tt")
OBJECT_TYPE = os.environ.get("CADET_SCENARIO_OBJECT", "car")

# Constant forward velocity (m/s) applied to the ego, and how far it drives
# before the scenario destroys it and ends.
VEHICLE_SPEED_MPS = 10.0
DRIVE_DISTANCE_M = 100
# Ego spawn Y: below the braking client's one-time speed-kick gate at y < 90, so
# the ego is already at speed on approach (the "at-speed" variant).
EGO_SPAWN_Y = 85.0


class AtSpeedBrakingScenario(BasicScenario):
    """
    Scenario class for vehicle moving forward with a static pedestrian ahead.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=100):
        self._map = CarlaDataProvider.get_map()
        self.vehicle_speed = VEHICLE_SPEED_MPS
        self.timeout = timeout
        self.world = world

        for actor in world.get_actors():
            if "vehicle" in actor.type_id or "walker" in actor.type_id:
                actor.destroy()

        # blueprints = world.get_blueprint_library().filter('static.prop*')
        # for bp in blueprints:
        #     print(bp.id)
        # exit()

        super(AtSpeedBrakingScenario, self).__init__(
            "AtSpeedBrakingScenario",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable)

        self.ego_vehicle = None
        self.obstacle = None

    def _initialize_actors(self, config):
        # Get a list of spawn points and select one (e.g., random or specific index)
        # spawn_points = self._map.get_spawn_points()
        # for i, spawn_point in enumerate(spawn_points):
        #     print(f"Spawn Point {i}: Location = {spawn_point.location}, Rotation = {spawn_point.rotation}")

        # Spawn the ego vehicle
        # spawn_location = carla.Transform(
        #     carla.Location(x=9.530025, y=302.570007, z=0.5),
        #     carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))

        # Temp dummy spawn (somewhere off-road)
        temp_spawn = carla.Transform(carla.Location(x=0, y=0, z=2), carla.Rotation(yaw=0))
        dummy = CarlaDataProvider.request_new_actor(f'vehicle.{VEHICLE_NAME}', temp_spawn)
        print(f"{VEHICLE_NAME}: {dummy.bounding_box}")
        dummy.destroy()  # Clean up

        adjusted_spawn_y = EGO_SPAWN_Y
        print(f"{VEHICLE_NAME}: {adjusted_spawn_y}")
        spawn_location = carla.Transform(
            carla.Location(x=-122.958664, y=adjusted_spawn_y, z=0.60),
            carla.Rotation(pitch=0.0, yaw=-90, roll=0.0))

        self.ego_vehicle = CarlaDataProvider.request_new_actor(f'vehicle.{VEHICLE_NAME}', spawn_location)


        # spawn_location = carla.Transform(
        #     carla.Location(x=-122.958664, y=100, z=0.60),
        #     carla.Rotation(pitch=0.0, yaw=-90, roll=0.0))
        # self.ego_vehicle = CarlaDataProvider.request_new_actor(f'vehicle.{VEHICLE_NAME}', spawn_location)
        self.ego_vehicle.set_autopilot(False)
        # physics = self.ego_vehicle.get_physics_control()
        # new_wheels = []
        # for wheel in physics.wheels:
        #     wheel.tire_friction = 1.0
        #     wheel.max_brake_torque = 600
        #     new_wheels.append(wheel)
        # physics.wheels = new_wheels
        # print(physics)
        self.other_actors.append(self.ego_vehicle)

        # Spawn a static pedestrian as the obstacle
        # walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.0001')[0]
        # walker_transform = carla.Transform(
        #     carla.Location(x=-132.235184, y=90.576721, z=0.6),
        #     carla.Rotation(pitch=0.0, yaw=90, roll=0.0))
        # self.obstacle = self.world.try_spawn_actor(walker_bp, walker_transform)

        # Spawn a static car as the obstacle
        # walker_bp = self.world.get_blueprint_library().filter('vehicle.kawasaki.ninja')[0]
        # walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.0001')[0]
        walker_bp = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        if OBJECT_TYPE == "car":
            walker_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        elif OBJECT_TYPE == "person":
            walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
        elif OBJECT_TYPE == "bike":
            walker_bp = self.world.get_blueprint_library().filter("vehicle.kawasaki.ninja")[0]
        walker_transform = carla.Transform(
            carla.Location(x=-122.235184, y=-122.393723, z=0.10),
            carla.Rotation(pitch=0.0, yaw=0, roll=0.0))
        if walker_bp.has_attribute('color'):
            walker_bp.set_attribute('color', '255,0,0')  # red bike
        self.obstacle = self.world.try_spawn_actor(walker_bp, walker_transform)


        # Spawn a static bike as the obstacle
        # walker_bp = self.world.get_blueprint_library().filter('vehicle.bh.crossbike')[0]
        # walker_transform = carla.Transform(
        #     carla.Location(x=-132.235184, y=90.576721, z=1),
        #     carla.Rotation(pitch=0.0, yaw=0, roll=0.0))
        # self.obstacle = self.world.try_spawn_actor(walker_bp, walker_transform)
        # self.obstacle.set_collision_enabled(carla.CollisionType.NONE)
        if self.obstacle:
            self.obstacle.set_simulate_physics(False)
            # physics_control = self.obstacle.get_physics_control()
            # physics_control.use_sweep_wheel_collision = False  # if using vehicle
            # self.obstacle.apply_physics_control(physics_control)
            # self.obstacle.set_target_velocity(carla.Vector3D(0, 0, 0))
            # self.obstacle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            self.other_actors.append(self.obstacle)
            print("[Scenario] Spawned static obstacle")
        else:
            print("[Scenario] Failed to spawn obstacle.")

    def _create_behavior(self):
        behavior = py_trees.composites.Sequence("VehicleBehaviorSequence")

        start_transform = ActorTransformSetter(self.ego_vehicle, self.ego_vehicle.get_transform())

        parallel_drive = py_trees.composites.Parallel(
            "ParallelDrive",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        maintain_speed = KeepVelocity(self.ego_vehicle, self.vehicle_speed)
        drive_distance_condition = DriveDistance(self.ego_vehicle, distance=DRIVE_DISTANCE_M, name="DriveDistance")

        parallel_drive.add_child(maintain_speed)
        parallel_drive.add_child(drive_distance_condition)

        destroy_vehicle = ActorDestroy(self.ego_vehicle)

        behavior.add_child(start_transform)
        behavior.add_child(parallel_drive)
        behavior.add_child(destroy_vehicle)

        return behavior

    def _create_test_criteria(self):
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicle)
        criteria.append(collision_criterion)
        return criteria

    def remove_all_actors(self):
        super().remove_all_actors()
        if self.ego_vehicle is not None:
            CarlaDataProvider.remove_actor_by_id(self.ego_vehicle.id)
        if self.obstacle is not None:
            CarlaDataProvider.remove_actor_by_id(self.obstacle.id)
