#!/usr/bin/env python

"""
Custom scenario for testing purposes, based on FollowLeadingVehicle
Leading vehicle spawns 150m away
No traffic lights or intersections

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""

import random

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity,
                                                                      WaypointFollower,
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               DriveDistance,
                                                                               InTriggerDistanceToVehicle,
                                                                               StandStill)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance, generate_target_waypoint_list

class CustomScenario(BasicScenario):
    """
    Some documentation on CustomScenario
    :param world is the carla world)
    :param ego_vehicles is a list of ego vehicles for this scenario
    :param config is the scenario configuration (ScenarioConfiguration)
    :param randomize can be used to select parameters randomly (optional, default=False)
    :param debug_mode can be used to provide more comprehensive console output (optional, default=False)
    :param criteria_enable can be used to disable/enable scenario evaluation based on test criteria (optional, default=True)
    :param timeout is the overall scenario timeout (optional, default=60 seconds)
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=1000):
        
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_distance = 30
        self._first_vehicle_speed = 15
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._spawn_pts = self._map.get_spawn_points()
        self._other_actor_max_brake = 1.0
        self._other_actor_transform = None
        self._first_vehicle_drive_distance = 100
        self.timeout = timeout

        CarlaDataProvider.initialize_csv()

        # Call constructor of BasicScenario
        super(CustomScenario, self).__init__(
          "CustomScenario",
          ego_vehicles,
          config,
          world,
          debug_mode,
          criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Initialize first (leading) vehicle 
        """

        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_distance)
        self._other_actor_transform = carla.Transform(
            carla.Location(first_vehicle_waypoint.transform.location.x,
                           first_vehicle_waypoint.transform.location.y,
                           first_vehicle_waypoint.transform.location.z + 1),
            first_vehicle_waypoint.transform.rotation)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(model='vehicle.nissan.patrol', spawn_point=first_vehicle_transform, color='yellow')
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)
        
        #from colorama import Back
        #print(Back.GREEN + str(self._reference_waypoint.transform))
        #print(Back.GREEN + str(first_vehicle_waypoint.transform))
        #for i in self._spawn_pts:
            #print("Location: {} Rotation: {}", i.location, i.rotation)

    def _create_behavior(self):
        """
        Setup the behavior for CustomScenario
        """

        # to avoid the other actor blocking traffic, it was spawned elsewhere
        # reset its pose to the required one
        start_transform = ActorTransformSetter(self.other_actors[0], self._other_actor_transform)

        target_waypoint, _ = get_waypoint_in_distance(CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), self._first_vehicle_drive_distance)
        #print(target_waypoint)
        # generating waypoints until intersection (target_waypoint)
        plan, next_waypoint = generate_target_waypoint_list(target_waypoint, 0)
        #for i in plan:
            #print(i[0])
        drive_along_waypoints = WaypointFollower(self.other_actors[0], self._first_vehicle_speed, plan=plan, avoid_collision=True)
        
        # stop condition
        stop = StopVehicle(self.other_actors[0], self._other_actor_max_brake)

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], distance=20, name="FinalDistance"))
        end_condition.add_child(StandStill(self.ego_vehicles[0], name="StandStill", duration=1))
        end_condition.add_child(DriveDistance(self.other_actors[0], distance=self._first_vehicle_drive_distance, name="DriveDistance"))

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(start_transform)
        sequence.add_child(drive_along_waypoints)
        sequence.add_child(stop)
        sequence.add_child(end_condition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        Setup the evaluation criteria for CustomScenario
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria
    
    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
