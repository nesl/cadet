"""Detection label loading and mAP / precision-recall evaluation helpers.

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
import copy

import numpy as np

# Local copies
from examples.av_pipeline.pipeline.objects.objects import Vector2D, Transform, Rotation, BoundingBox2D, BoundingBox3D

OBSTACLE_LABELS = {
    'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle', 'person',
    'stop sign', 'parking meter', 'cat', 'dog', 'speed limit 30',
    'speed limit 60', 'speed limit 90'
}

PYLOT_BBOX_COLOR_MAP = {
    'person': [0, 128, 0],
    'vehicle': [128, 0, 0],
    'car': [128, 0, 0],
    'bicycle': [128, 0, 0],
    'motorcycle': [128, 0, 0],
    'bus': [128, 0, 0],
    'truck': [128, 0, 0],
    'stop marking': [128, 128, 0],
    'speed limit': [255, 255, 0],
    'red traffic light': [0, 0, 255],
    'yellow traffic light': [0, 255, 255],
    'green traffic light': [0, 255, 0],
    'off traffic light': [0, 0, 0],
    '': [255, 255, 255],
}

coco_bbox_color_list = np.array([
    1.000, 1.000, 1.000, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
    0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
    0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
    1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
    0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
    0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
    0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
    0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
    0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
    0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
    0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.50, 0.5, 0
]).astype(np.float32)

def load_coco_labels(labels_path):
    """Returns a map from index to label.

    Args:
        labels_path (:obj:`str`): Path to a file storing a label on each line.
    """
    labels_map = {}
    with open(labels_path) as labels_file:
        labels = labels_file.read().splitlines()
        index = 1
        for label in labels:
            labels_map[index] = label
            index += 1
    return labels_map


def load_coco_bbox_colors(coco_labels):
    """Returns a map from label to color."""
    # Transform to RGB values.
    bbox_color_list = coco_bbox_color_list.reshape((-1, 3)) * 255
    # Transform to ints
    bbox_colors = [(bbox_color_list[_]).astype(np.uint8)
                   for _ in range(len(bbox_color_list))]
    bbox_colors = np.array(bbox_colors,
                           dtype=np.uint8).reshape(len(bbox_colors), 1, 1, 3)

    colors = {}
    for category, label in coco_labels.items():
        colors[label] = bbox_colors[category - 1][0][0].tolist()
    return colors


def get_prediction_results(ground_truths, predictions, iou_threshold):
    """Calculate the number of true positives, false positives and false
    negatives from the given ground truth and predictions."""
    true_pos, false_pos, false_neg = None, None, None

    # If there are no predictions, then everything is a false negative.
    if len(predictions) == 0:
        true_pos, false_pos = 0, 0
        false_neg = len(ground_truths)
        return true_pos, false_pos, false_neg

    # If there is no ground truth, everything is a false positive.
    if len(ground_truths) == 0:
        true_pos, false_neg = 0, 0
        false_pos = len(predictions)
        return true_pos, false_pos, false_neg

    # Iterate over the predictions and calculate the IOU of each prediction
    # with each ground truth.
    ious = []
    for i, prediction in enumerate(predictions):
        for j, ground_truth in enumerate(ground_truths):
            iou = prediction.calculate_iou(ground_truth)
            if iou > iou_threshold:
                ious.append((i, j, iou))

    # If no IOUs were over the threshold, return all predictions as false
    # positives and all ground truths as false negatives.
    if len(ious) == 0:
        true_pos = 0
        false_pos, false_neg = len(predictions), len(ground_truths)
    else:
        # Sort the IOUs and match each box only once.
        ground_truths_matched, predictions_matched = set(), set()
        matched = []
        for prediction, ground_truth, iou in sorted(ious,
                                                    key=lambda x: x[-1],
                                                    reverse=True):
            if (ground_truth not in ground_truths_matched
                    and prediction not in predictions_matched):
                ground_truths_matched.add(ground_truth)
                predictions_matched.add(prediction)
                matched.append((prediction, ground_truth, iou))

        # The matches are the true positives.
        true_pos = len(matched)
        # The unmatched predictions are the false positives.
        false_pos = len(predictions) - len(predictions_matched)
        # The umatched ground truths are the false negatives.
        false_neg = len(ground_truths) - len(ground_truths_matched)

    return true_pos, false_pos, false_neg


def get_precision_recall(true_positives, false_positives, false_negatives):
    precision, recall = None, None
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = float(true_positives) / float(true_positives +
                                                  false_positives)

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = float(true_positives) / float(true_positives +
                                               false_negatives)

    return (precision, recall)


def get_precision_recall_at_iou(ground_truths, predictions, iou_threshold):
    true_pos, false_pos, false_neg = get_prediction_results(
        ground_truths, predictions, iou_threshold)
    return get_precision_recall(true_pos, false_pos, false_neg)


def get_mAP(ground_obstacles, obstacles):
    """Return mAP with IoU threshold of 0.5"""
    # Sort bboxes descending by score.
    sorted_obstacles = \
        sorted(obstacles, key=lambda o: o.confidence, reverse=True)
    detected_bboxes = [o.bounding_box_2D for o in sorted_obstacles]
    ground_bboxes = [obstacle.bounding_box_2D for obstacle in ground_obstacles]
    # Compute recall precision. The results are sorted in descending
    # order by recall.
    prec_rec = []
    while (len(detected_bboxes) > 0):
        # Get precision recall with 0.5 IoU threshold .
        precision, recall = get_precision_recall_at_iou(
            ground_bboxes, detected_bboxes, 0.5)
        prec_rec.append((precision, recall))
        detected_bboxes.pop()
    # Append (0, 0) to also cover the area from first recall point to 0 recall.
    prec_rec.append((0, 0))
    avg_precision = 0.0
    max_precision = 0.0
    max_precision = None
    last_recall = None
    for (precision, recall) in prec_rec:
        if max_precision is None:
            max_precision = precision
            last_recall = recall
        else:
            avg_precision += (last_recall - recall) * max_precision
            max_precision = max(max_precision, precision)
            last_recall = recall
    return avg_precision


def get_obstacle_locations(obstacles, depth_frame, ego_transform, camera_setup,
                           logger):
    from examples.av_pipeline.pipeline.objects.frames import PointCloud, DepthFrame
    if isinstance(depth_frame, PointCloud):
        point_cloud = depth_frame
        # Get the position of the camera in world frame of reference.
        transformed_camera_setup = copy.deepcopy(camera_setup)
        transformed_camera_setup.set_transform(
            ego_transform * transformed_camera_setup.transform)

        obstacles_with_location = []
        for obstacle in obstacles:
            location = point_cloud.get_pixel_location(
                obstacle.bounding_box_2D.get_center_point(),
                transformed_camera_setup)
            if location is not None:
                obstacle.transform = Transform(
                    location, Rotation())
                obstacles_with_location.append(obstacle)
            else:
                logger.error(
                    'Could not find world location for obstacle {}'.format(
                        obstacle))
        return obstacles_with_location
    elif isinstance(depth_frame, DepthFrame):
        depth_frame = depth_frame
        depth_frame.camera_setup.set_transform(
            ego_transform * depth_frame.camera_setup.transform)

        for obstacle in obstacles:
            center_point = obstacle.bounding_box_2D.get_center_point()
            # Sample several points around the center of the bounding box
            # in case the bounding box is not well centered on the obstacle.
            # In such situations the center point might be in between legs,
            # and thus we might overestimate the distance.
            sample_points = []
            for delta_x in range(-30, 30, 5):
                for delta_y in range(-30, 30, 5):
                    sample_point = center_point + Vector2D(
                        delta_x, delta_y)
                    if obstacle.bounding_box.is_within(sample_point):
                        sample_points.append(sample_point)
            locations = depth_frame.get_pixel_locations(sample_points)
            # Choose the closest from the locations of the sampled points.
            min_distance = np.infty
            closest_location = None
            for location in locations:
                dist = location.distance(ego_transform.location)
                if dist < min_distance:
                    min_distance = dist
                    closest_location = location
            obstacle.transform = Transform(closest_location,
                                                    Rotation())
        return obstacles
    else:
        raise ValueError('Unexpected depth message type')
