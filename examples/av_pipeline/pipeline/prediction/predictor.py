"""Obstacle motion prediction (linear / R2P2).

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
from examples.av_pipeline import params

import numpy as np
import time

from examples.av_pipeline.pipeline.objects.objects import Transform, Location, Rotation
from examples.av_pipeline.pipeline.objects.objects import ObstaclePrediction
from examples.av_pipeline.pipeline.objects.messages import ObstacleTrajectoriesMessage

from examples.av_pipeline.pipeline.prediction.prediction_utils import get_occupancy_grid

_r2p2 = None
# _lidar_setup = None # Likely unnecessary: create_center_lidar_setup(Transform(Location(1.3, 0.0, 1.8), Rotation(pitch=-15)))

# input and output to this will be ObstacleTrajectoriesMessage, output is ObstaclePrediction
def get_predictions(message):  
    if params.prediction_type == 'linear':
        return generate_linear_predicted_trajectories(message)
    else:
        return generate_r2p2_predicted_trajectories(message)


def generate_linear_predicted_trajectories(message: ObstacleTrajectoriesMessage):
    obstacle_predictions_list = []
    start_time = time.time()
    nearby_obstacle_trajectories, nearby_obstacles_ego_transforms = message.get_nearby_obstacles_info(params.prediction_radius)
    num_predictions = len(nearby_obstacle_trajectories)

    for idx in range(len(nearby_obstacle_trajectories)):
        obstacle_trajectory = nearby_obstacle_trajectories[idx]
        # Time step matrices used in regression.
        num_steps = min(params.prediction_num_past_steps, len(obstacle_trajectory.trajectory))
        ts = np.zeros((num_steps, 2))
        future_ts = np.zeros((params.prediction_num_future_steps, 2))
        for t in range(num_steps):
            ts[t][0] = -t
            ts[t][1] = 1
        for i in range(params.prediction_num_future_steps):
            future_ts[i][0] = i + 1
            future_ts[i][1] = 1

        xy = np.zeros((num_steps, 2))
        for t in range(num_steps):
            # t-th most recent step
            transform = obstacle_trajectory.trajectory[-(t + 1)]
            xy[t][0] = transform.location.x
            xy[t][1] = transform.location.y
        linear_model_params = np.linalg.lstsq(ts, xy, rcond=None)[0]
        # Predict future steps and convert to list of locations.
        predict_array = np.matmul(future_ts, linear_model_params)
        predictions = []
        for t in range(params.prediction_num_future_steps):
            # Linear prediction does not predict vehicle orientation, so we
            # use our estimated orientation of the vehicle at its latest
            # location.
            predictions.append(
                Transform(location=Location(x=predict_array[t][0],
                                            y=predict_array[t][1]),
                            rotation=nearby_obstacles_ego_transforms[idx].
                            rotation))

        obstacle_predictions_list.append(ObstaclePrediction(obstacle_trajectory, obstacle_trajectory.obstacle.transform, 1.0, predictions))

    return obstacle_predictions_list, (time.time() - start_time) * 1000


def generate_r2p2_predicted_trajectories(point_cloud_msg, tracking_msg):
    import torch
    from examples.av_pipeline.pipeline.prediction.r2p2 import R2P2
    if not _r2p2:
        _r2p2 = R2P2().to(params.device)
    
    start_time = time.time()
    nearby_trajectories, nearby_vehicle_ego_transforms, nearby_trajectories_tensor, binned_lidars_tensor = preprocess_input(tracking_msg, point_cloud_msg)

    num_predictions = len(nearby_trajectories)

    if num_predictions == 0:
        return

    # Run the forward pass.
    z = torch.tensor(
        np.random.normal(size=(num_predictions,
                                params.prediction_num_future_steps,
                                2))).to(torch.float32).to(params.device)
    model_start_time = time.time()
    prediction_array, _ = _r2p2.forward(
        z, nearby_trajectories_tensor, binned_lidars_tensor)
    model_runtime = (time.time() - model_start_time) * 1000

    prediction_array = prediction_array.cpu().detach().numpy()

    obstacle_predictions_list = postprocess_predictions(
        prediction_array, nearby_trajectories,
        nearby_vehicle_ego_transforms)
    runtime = (time.time() - start_time) * 1000
    return obstacle_predictions_list


def preprocess_input(tracking_msg, point_cloud_msg):

    nearby_vehicle_trajectories, nearby_vehicle_ego_transforms = tracking_msg.get_nearby_obstacles_info(
            params.prediction_radius,
            lambda t: t.obstacle.is_vehicle())
    point_cloud = point_cloud_msg.point_cloud.points
    num_nearby_vehicles = len(nearby_vehicle_trajectories)
    if num_nearby_vehicles == 0:
        return [], [], [], []

    # Pad and rotate the trajectory of each nearby vehicle to its
    # coordinate frame. Also, remove the z-coordinate of the trajectory.
    nearby_trajectories_tensor = []  # Pytorch tensor for network input.

    for i in range(num_nearby_vehicles):
        cur_trajectory = nearby_vehicle_trajectories[
            i].get_last_n_transforms(params.prediction_num_past_steps)
        cur_trajectory = np.stack(
            [[point.location.x, point.location.y, point.location.z]
                for point in cur_trajectory])

        rotated_trajectory = nearby_vehicle_ego_transforms[
            i].inverse_transform_points(cur_trajectory)[:, :2]

        nearby_trajectories_tensor.append(rotated_trajectory)

    nearby_trajectories_tensor = np.stack(nearby_trajectories_tensor)
    nearby_trajectories_tensor = torch.tensor(
        nearby_trajectories_tensor).to(torch.float32).to(params.device)

    # For each vehicle, transform the lidar point cloud to that vehicle's
    # coordinate frame for purposes of prediction.
    binned_lidars = []
    for i in range(num_nearby_vehicles):
        rotated_point_cloud = nearby_vehicle_ego_transforms[
            i].inverse_transform_points(point_cloud)
        binned_lidars.append(
            get_occupancy_grid(
                rotated_point_cloud,
                _lidar_setup.transform.location.z,
                int(_lidar_setup.get_range_in_meters())))
    binned_lidars = np.concatenate(binned_lidars)
    binned_lidars_tensor = torch.tensor(binned_lidars).to(
        torch.float32).to(params.device)

    return nearby_vehicle_trajectories, nearby_vehicle_ego_transforms, \
        nearby_trajectories_tensor, binned_lidars_tensor


def postprocess_predictions(prediction_array, vehicle_trajectories,
                                vehicle_ego_transforms):
    # The prediction_array consists of predictions with respect to each
    # vehicle. Transform each predicted trajectory to be in relation to the
    # ego-vehicle, then convert into an ObstaclePrediction.
    obstacle_predictions_list = []
    num_predictions = len(vehicle_trajectories)

    for idx in range(num_predictions):
        cur_prediction = prediction_array[idx]

        obstacle_transform = vehicle_trajectories[idx].obstacle.transform
        predictions = []
        # Because R2P2 only predicts (x,y) coordinates, we assume the
        # vehicle stays at the same height as its last location.
        for t in range(params.prediction_num_future_steps):
            cur_point = vehicle_ego_transforms[idx].transform_points(
                np.array([[
                    cur_prediction[t][0], cur_prediction[t][1],
                    vehicle_ego_transforms[idx].location.z
                ]]))[0]
            # R2P2 does not predict vehicle orientation, so we use our
            # estimated orientation of the vehicle at its latest location.
            predictions.append(
                Transform(location=Location(cur_point[0], cur_point[1],
                                            cur_point[2]),
                            rotation=vehicle_ego_transforms[idx].rotation))

        # Probability; currently a filler value because we are taking
        # just one sample from distribution
        obstacle_predictions_list.append(
            ObstaclePrediction(vehicle_trajectories[idx],
                                obstacle_transform, 1.0, predictions))
    return obstacle_predictions_list