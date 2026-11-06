"""World model: ego, obstacles, waypoints and prediction state.

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
from collections import deque
import copy
import enum
import itertools
import math
import os

import numpy as np

from examples.av_pipeline import params
from examples.av_pipeline.pipeline.objects.objects import Waypoints, RoadOption, Transform, Location, Rotation

# Number of predicted locations to consider when computing speed factors.
NUM_FUTURE_TRANSFORMS = 10

class World(object):
    """A representation of the world that is used by the planners."""
    def __init__(self):
        self.static_obstacles = None
        self.obstacle_predictions = []
        self._ego_obstacle_predictions = []
        self.pose = None
        self.ego_trajectory = deque(maxlen=params.tracking_num_steps)
        self.ego_transform = None
        self.ego_velocity_vector = None
        self._lanes = None
        self._map = None
        self._goal_location = Location(376, 0.0, 0.0)
        self.waypoints = Waypoints(deque([Transform(self._goal_location, Rotation())]))
        self._last_stop_ego_location = None
        self._distance_since_last_full_stop = 0
        self._num_ticks_stopped = 0

    def update(self,
               pose,
               obstacle_predictions,
               static_obstacles,
               hd_map=None,
               lanes=None):
        self.pose = pose
        self.ego_transform = pose.transform
        self.ego_velocity_vector = pose.velocity_vector
        self.ego_trajectory.append(self.ego_transform)
        self.obstacle_predictions = obstacle_predictions
        self._ego_obstacle_predictions = copy.deepcopy(obstacle_predictions)
        # Tranform predictions to world frame of reference.
        for obstacle_prediction in self.obstacle_predictions:
            obstacle_prediction.to_world_coordinates(self.ego_transform)
        # Road signs are in world coordinates. Only maintain the road signs
        # that are within the threshold.
        self.static_obstacles = []
        for obstacle in static_obstacles:
            if (obstacle.transform.location.distance(
                    self.ego_transform.location) <=
                    params.static_obstacle_distance_threshold):
                self.static_obstacles.append(obstacle)
        
        self.waypoints.remove_waypoint_if_close(self.ego_transform.location, 3)
        new_goal_location = self.get_goal_location(self.ego_transform)
        if len(self.obstacle_predictions) > 0:
            prediction = self.obstacle_predictions[0]
            waypoints_deque = deque([
                transform for transform in prediction.obstacle_trajectory.trajectory # list of transforms
            ])
            # for future_transform in prediction.predicted_trajectory:
            #    waypoints_deque.append(future_transform)
            road_options = deque([
                RoadOption.LANE_FOLLOW
                for _ in waypoints_deque
            ])
            self.waypoints = Waypoints(waypoints_deque, road_options=road_options)
        else:
            new_goal_location = self._goal_location
        
        if new_goal_location != self._goal_location:
            self._goal_location = new_goal_location
            if not self.waypoints or self.waypoints.is_empty():
                self.waypoints = Waypoints(
                    deque([self.ego_transform]),
                    road_options=deque([RoadOption.LANE_FOLLOW]))

        self._map = hd_map
        self._lanes = lanes

        # The waypoints are not received on the global trajectory stream.
        # We need to compute them using the map.
        if not self.waypoints:
            if self._map is not None and self._goal_location is not None:
                self.waypoints = Waypoints(deque(), deque())
                self.waypoints.recompute_waypoints(self._map,
                                                   self.ego_transform.location,
                                                   self._goal_location)

        if pose.forward_speed < 0.7:
            # We can't just check if forward_speed is zero because localization
            # noise can cause the forward_speed to be non zero even when the
            # ego is stopped.
            self._num_ticks_stopped += 1
            if self._num_ticks_stopped > 10:
                self._distance_since_last_full_stop = 0
                self._last_stop_ego_location = self.ego_transform.location
        else:
            self._num_ticks_stopped = 0
            if self._last_stop_ego_location is not None:
                self._distance_since_last_full_stop = \
                    self.ego_transform.location.distance(
                        self._last_stop_ego_location)
            else:
                self._distance_since_last_full_stop = 0

    def get_goal_location(self, ego_transform: Transform):
        if len(self.waypoints.waypoints) > 1:
            dist = ego_transform.location.distance(
                self.waypoints.waypoints[0].location)
            if dist < 5:
                new_goal_location = self.waypoints.waypoints[1].location
            else:
                new_goal_location = self.waypoints.waypoints[0].location
        elif len(self.waypoints.waypoints) == 1:
            new_goal_location = self.waypoints.waypoints[0].location
        else:
            new_goal_location = ego_transform.location
        return new_goal_location

    def update_waypoints(self, goal_location, waypoints):
        self._goal_location = goal_location
        self.waypoints = waypoints

    def follow_waypoints(self, target_speed: float):
        self.waypoints.remove_completed(self.ego_transform.location,
                                        self.ego_transform)
        return self.waypoints.slice_waypoints(0,
                                              params.num_waypoints_ahead,
                                              target_speed)

    def get_obstacle_list(self):
        obstacle_list = []
        for prediction in self.obstacle_predictions:
            # Use all prediction times as potential obstacles.
            previous_origin = None
            for transform in prediction.predicted_trajectory:
                # Ignore predictions that are too close.
                if (previous_origin is None
                        or previous_origin.location.l2_distance(
                            transform.location) >
                        params.obstacle_filtering_distance):
                    previous_origin = transform
                    # Ensure the prediction is nearby.
                    if (self.ego_transform.location.l2_distance(
                            transform.location) <=
                            params.dynamic_obstacle_distance_threshold):
                        obstacle = prediction.obstacle_trajectory.obstacle
                        obstacle_corners = \
                            obstacle.get_bounding_box_corners(
                                transform, params.obstacle_radius)
                        obstacle_list.append(obstacle_corners)
        if len(obstacle_list) == 0:
            return np.empty((0, 4))
        return np.array(obstacle_list)

    def draw_on_frame(self, frame):
        for obstacle_prediction in self._ego_obstacle_predictions:
            obstacle_prediction.draw_trajectory_on_frame(frame)
        for obstacle in self.static_obstacles:
            if obstacle.is_traffic_light():
                world_transform = obstacle.transform
                # Transform traffic light to ego frame of reference.
                obstacle.transform = (self.ego_transform.inverse_transform() *
                                      obstacle.transform)
                obstacle.draw_on_bird_eye_frame(frame)
                obstacle.transform = world_transform
        if self.waypoints:
            self.waypoints.draw_on_frame(
                frame, self.ego_transform.inverse_transform())
        # TODO: Draw lane markings. We do not draw them currently
        # because we need to transform them from world frame of reference
        # to ego vehicle frame of reference, which is slow to compute.
        # if self._map:
        #     lane = self._map.get_lane(self.ego_transform.location)
        #     lane.draw_on_frame(frame, self.ego_transform.inverse_transform())
        if self._lanes:
            for lane in self._lanes:
                lane.draw_on_frame(frame)

    def stop_person(self, obstacle, wp_vector) -> float:
        """Computes a stopping factor for ego vehicle given a person obstacle.

        Args:
            obstacle (:py:class:`~pylot.prediction.obstacle_prediction.ObstaclePrediction`):  # noqa: E501
                Prediction for a person.
            wp_vector (:py:class:`~pylot.utils.Vector2D`): vector from the ego
                vehicle to the target waypoint.

        Returns:
            :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
        """
        if self._map is not None:
            if not self._map.are_on_same_lane(self.ego_transform.location,
                                              obstacle.transform.location):
                # Person is not on the same lane.
                if not any(
                        map(
                            lambda transform: self._map.are_on_same_lane(
                                self.ego_transform.location, transform.location
                            ), obstacle.predicted_trajectory)):
                    return 1
        ego_location_2d = self.ego_transform.location.as_vector_2D()
        min_speed_factor_p = compute_person_speed_factor(
            ego_location_2d, obstacle.transform.location.as_vector_2D(),
            wp_vector)
        transforms = itertools.islice(
            obstacle.predicted_trajectory, 0,
            min(NUM_FUTURE_TRANSFORMS, len(obstacle.predicted_trajectory)))
        for person_transform in transforms:
            speed_factor_p = compute_person_speed_factor(
                ego_location_2d, person_transform.location.as_vector_2D(),
                wp_vector)
            min_speed_factor_p = min(min_speed_factor_p, speed_factor_p)
        return min_speed_factor_p

    def stop_vehicle(self, obstacle, wp_vector) -> float:
        """Computes a stopping factor for ego vehicle given a vehicle pos.

        Args:
            obstacle (:py:class:`~pylot.prediction.obstacle_prediction.ObstaclePrediction`):  # noqa: E501
                Prediction for a vehicle.
            wp_vector (:py:class:`~pylot.utils.Vector2D`): vector from the ego
                vehicle to the target waypoint.

        Returns:
            :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
        """
        if self.ego_transform.location.x == obstacle.transform.location.x and \
           self.ego_transform.location.y == obstacle.transform.location.y and \
           self.ego_transform.location.z == obstacle.transform.location.z:
            # Don't stop for ourselves.
            return 1

        if self._map is not None:
            if not self._map.are_on_same_lane(self.ego_transform.location,
                                              obstacle.transform.location):
                # Vehicle is not on the same lane as the ego.
                if not any(
                        map(
                            lambda transform: self._map.are_on_same_lane(
                                self.ego_transform.location, transform.location
                            ), obstacle.predicted_trajectory)):
                    return 1

        ego_location_2d = self.ego_transform.location.as_vector_2D()
        min_speed_factor_v = compute_vehicle_speed_factor(
            ego_location_2d, obstacle.transform.location.as_vector_2D(),
            wp_vector)
        v_dist = obstacle.transform.location.as_vector_2D().l2_distance(ego_location_2d)
        os.makedirs(params.OUTPUT_DIR, exist_ok=True)
        f=open(os.path.join(params.OUTPUT_DIR, "dist_sortpid_service.txt"), 'a')
        f.write(str(v_dist))
        f.write('\n')
        f.close()
        transforms = itertools.islice(
            obstacle.predicted_trajectory, 0,
            min(NUM_FUTURE_TRANSFORMS, len(obstacle.predicted_trajectory)))
        for vehicle_transform in transforms:
            speed_factor_v = compute_vehicle_speed_factor(
                ego_location_2d, vehicle_transform.location.as_vector_2D(),
                wp_vector)
            min_speed_factor_v = min(min_speed_factor_v, speed_factor_v)
        return min_speed_factor_v

    def stop_for_agents(self) -> float:
        """Calculates the speed factor in [0, 1] (0 is full stop).

        Reduces the speed factor whenever the ego vehicle's path is blocked
        by an obstacle, or the ego vehicle must stop at a traffic light.
        """
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1
        speed_factor_stop = 1

        try:
            self.waypoints.remove_completed(self.ego_transform.location)
            wp_vector = self.waypoints.get_vector(
                self.ego_transform,
                params.min_pid_steer_waypoint_distance)
            wp_angle = self.waypoints.get_angle(
                self.ego_transform,
                params.min_pid_steer_waypoint_distance)
        except (ValueError, AttributeError):
            # No more waypoints to follow.
            return (0, 0, 0, 0, 0)

        for obstacle in self.obstacle_predictions:
            if obstacle.is_person() and params.stop_for_people:
                new_speed_factor_p = self.stop_person(obstacle, wp_vector)
                if new_speed_factor_p < speed_factor_p:
                    speed_factor_p = new_speed_factor_p
            elif obstacle.is_vehicle() and params.stop_for_vehicles:
                new_speed_factor_v = self.stop_vehicle(obstacle, wp_vector)
                if new_speed_factor_v < speed_factor_v:
                    speed_factor_v = new_speed_factor_v

        semaphorized_junction = False
        for obstacle in self.static_obstacles:
            if (obstacle.is_traffic_light()
                    and params.stop_for_traffic_lights):
                valid_tl, new_speed_factor_tl = self.stop_traffic_light(
                    obstacle, wp_vector, wp_angle)
                semaphorized_junction = semaphorized_junction or valid_tl
                if new_speed_factor_tl < speed_factor_tl:
                    speed_factor_tl = new_speed_factor_tl

        if params.stop_at_uncontrolled_junctions:
            if (self._map is not None and not semaphorized_junction
                    and not self._map.is_intersection(
                        self.ego_transform.location)):
                dist_to_junction = self._map.distance_to_intersection(
                    self.ego_transform.location, max_distance_to_check=13)
                if (dist_to_junction is not None
                        and self._distance_since_last_full_stop > 13):
                    speed_factor_stop = 0

        speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v,
                           speed_factor_stop)
        return (speed_factor, speed_factor_p, speed_factor_v, speed_factor_tl,
                speed_factor_stop)

    def stop_traffic_light(self, tl, wp_vector, wp_angle) -> float:
        """Computes a stopping factor for ego vehicle given a traffic light.

        Args:
            tl (:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`):  # noqa: E501
                the traffic light.
            wp_vector (:py:class:`~pylot.utils.Vector2D`): vector from the ego
                vehicle to the target waypoint.

        Returns:
            :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
        """
        if self._map is not None:
            # The traffic light is not relevant to the ego vehicle.
            if not self._map.must_obey_traffic_light(
                    self.ego_transform.location, tl.transform.location):
                return False, 1
        
        # The ego vehicle is too close to the traffic light.
        if (self.ego_transform.location.distance(tl.transform.location) <
                params.traffic_light_min_distance):
            return True, 1
        # The ego vehicle can carry on driving.
        if (tl.state == TrafficLightColor.GREEN
                or tl.state == TrafficLightColor.OFF):
            return True, 1

        height_delta = tl.transform.location.z - self.ego_transform.location.z
        if height_delta > 4:
            # The traffic ligh is across the road. Increase the max distance.
            traffic_light_max_distance = \
                params.traffic_light_max_distance * 2.5
            traffic_light_max_angle = params.traffic_light_max_angle / 3
            american_tl = True
        else:
            traffic_light_max_distance = params.traffic_light_max_distance
            traffic_light_max_angle = params.traffic_light_max_angle
            american_tl = False
        speed_factor_tl = 1
        ego_location_2d = self.ego_transform.location.as_vector_2D()
        tl_location_2d = tl.transform.location.as_vector_2D()
        tl_vector = tl_location_2d - ego_location_2d
        tl_dist = tl_location_2d.l2_distance(ego_location_2d)
        tl_angle = tl_vector.get_angle(wp_vector)
        if (-0.2 <= tl_angle < traffic_light_max_angle
                and tl_dist < traffic_light_max_distance):
            # The traffic light is at most x radians to the right of the
            # vehicle path, and is not too far away.
            speed_factor_tl = min(
                speed_factor_tl, tl_dist /
                (params.coast_factor * traffic_light_max_distance))

        if (-0.2 <= tl_angle <
                traffic_light_max_angle / params.coast_factor and
                tl_dist < traffic_light_max_distance * params.coast_factor
                and math.fabs(wp_angle) < 0.2):
            # The ego is pretty far away, so the angle to the traffic light has
            # to be smaller, and the vehicle must be driving straight.
            speed_factor_tl = min(
                speed_factor_tl, tl_dist /
                (params.coast_factor * traffic_light_max_distance))

        if (-0.2 <= tl_angle <
                traffic_light_max_angle * params.coast_factor
                and math.fabs(wp_angle) < 0.2):
            if american_tl:
                if (-0.1 <= tl_angle < traffic_light_max_angle
                        and tl_dist < 60):
                    dist_to_intersection = self._map.distance_to_intersection(
                        self.ego_transform.location, max_distance_to_check=20)
                    if (dist_to_intersection is not None
                            and dist_to_intersection < 12):
                        if (tl.bounding_box_2D is None
                                or tl.bounding_box_2D.get_width() *
                                tl.bounding_box_2D.get_height() > 400):
                            speed_factor_tl = 0
                    if (dist_to_intersection is not None and tl_dist < 27
                            and 12 <= dist_to_intersection <= 20):
                        speed_factor_tl = 0
            else:
                if (tl_dist <
                        traffic_light_max_distance / params.coast_factor):
                    # The traffic light is nearby and the vehicle is driving
                    # straight; the angle to the traffic light can be higher.
                    speed_factor_tl = 0
        if speed_factor_tl < 1:
            dist_to_intersection = self._map.distance_to_intersection(
                self.ego_transform.location, max_distance_to_check=20)
            if dist_to_intersection is None:
                # Our lidar-based depth estimation does not work when
                # we're on a hill.
                # XXX(ionel): Hack to avoid getting stuck when we're far
                # from intersections (see scenario 28 in the challenge training
                # routes).
                return True, 1
            else:
                return True, speed_factor_tl
        else:
            # The traffic light doesn't affect the vehicle.
            return False, speed_factor_tl


class BehaviorPlannerState(enum.Enum):
    """States in which the FSM behavior planner can be in."""
    FOLLOW_WAYPOINTS = 0
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6
    OVERTAKE = 7


def compute_person_speed_factor(ego_location_2d, person_location_2d, wp_vector) -> float:
    speed_factor_p = 1
    p_vector = person_location_2d - ego_location_2d
    p_dist = person_location_2d.l2_distance(ego_location_2d)
    p_angle = p_vector.get_angle(wp_vector)
    # Maximum braking is applied if the person is in the emergency
    # hit zone. Otherwise, gradual braking is applied if the person
    # is in the hit zone.
    if (math.fabs(p_angle) < params.person_angle_hit_zone
            and p_dist < params.person_distance_hit_zone):
        # Person is in the hit zone.
        speed_factor_p = min(
            speed_factor_p,
            p_dist / (params.coast_factor * params.person_distance_hit_zone))
    if (math.fabs(p_angle) < params.person_angle_emergency_zone
            and p_dist < params.person_distance_emergency_zone):
        # Person is in emergency hit zone.
        speed_factor_p = 0
    return speed_factor_p


def compute_vehicle_speed_factor(ego_location_2d, vehicle_location_2d,
                                 wp_vector) -> float:
    speed_factor_v = 1
    v_vector = vehicle_location_2d - ego_location_2d
    v_dist = vehicle_location_2d.l2_distance(ego_location_2d)
    v_angle = v_vector.get_angle(wp_vector)
    #print('Vehicle vector {}; dist {}; angle {}'.format(v_vector, v_dist, v_angle))
    min_angle = -0.5 * params.vehicle_max_angle / params.coast_factor
    if (min_angle < v_angle < params.vehicle_max_angle
            and v_dist < params.vehicle_max_distance):
        # The vehicle is within the angle limit, and nearby.
        speed_factor_v = min(
            speed_factor_v,
            v_dist / (params.coast_factor * params.vehicle_max_distance))

    if (min_angle < v_angle < params.vehicle_max_angle / params.coast_factor
            and v_dist < params.vehicle_max_distance * params.coast_factor):
        # The vehicle is a bit far away, but it's on ego vehicle's path.
        speed_factor_v = min(
            speed_factor_v,
            v_dist / (params.coast_factor * params.vehicle_max_distance))

    min_nearby_angle = -0.5 * params.vehicle_max_angle * params.coast_factor
    if (min_nearby_angle < v_angle <
            params.vehicle_max_angle * params.coast_factor
            and v_dist < params.vehicle_max_distance / params.coast_factor):
        # The vehicle is very close; the angle can be higher.
        speed_factor_v = 0
    return speed_factor_v


class TrafficLightColor(enum.Enum):
    """Enum to represent the states of a traffic light."""
    RED = 1
    YELLOW = 2
    GREEN = 3
    OFF = 4

    def get_label(self):
        """Gets the label of a traffic light color.

        Returns:
            :obj:`str`: The label string.
        """
        if self.value == 1:
            return 'red traffic light'
        elif self.value == 2:
            return 'yellow traffic light'
        elif self.value == 3:
            return 'green traffic light'
        else:
            return 'off traffic light'

    def get_color(self):
        if self.value == 1:
            return [255, 0, 0]
        elif self.value == 2:
            return [255, 165, 0]
        elif self.value == 3:
            return [0, 255, 0]
        else:
            return [0, 0, 0]
