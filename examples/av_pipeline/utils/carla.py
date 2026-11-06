"""Miscellaneous CARLA/image helpers: timestamp overlay, transforms, timing.

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
import math
import time
from enum import Enum

import numpy as np

from examples.av_pipeline.pipeline.objects.objects import Transform, Location, Rotation

def add_timestamp(image_np, timestamp):
    """Adds a timestamp text to an image np array.

    Args:
        image_np: A numpy array of the image.
        timestamp (:obj:`int`): The timestamp of the image.
    """
    import cv2
    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp_txt = '{}'.format(timestamp)
    # Put timestamp text.
    cv2.putText(image_np,
                timestamp_txt, (5, 15),
                txt_font,
                0.5, (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)


def get_top_down_transform(transform, top_down_camera_altitude):
    # Calculation relies on the fact that the camera's FOV is 90.
    top_down_location = (transform.location +
                         Location(0, 0, top_down_camera_altitude))
    return Transform(top_down_location, Rotation(-90, 0, 0))


def time_epoch_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


def set_tf_loglevel(level):
    """To be used to suppress TensorFlow logging."""
    import logging
    import os
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

def verify_keys_in_dict(required_keys, arg_dict):
    assert set(required_keys).issubset(set(arg_dict.keys())), \
            "one or more of {} not found in {}".format(required_keys, arg_dict)
