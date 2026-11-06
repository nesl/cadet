"""Frame containers (camera / depth / point cloud) and conversions.

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
import copy
import os
import pickle

import numpy as np
from numpy.linalg import inv
from numpy.matlib import repmat

from typing import List, Tuple, Union

import PIL.Image as Image

import cv2

from examples.av_pipeline.pipeline.objects.objects import Location, Transform, Vector2D, Location, BoundingBox2D, CameraSetup, LidarSetup
from examples.av_pipeline.pipeline.detection.utils import PYLOT_BBOX_COLOR_MAP
from examples.av_pipeline.utils.carla import add_timestamp

class DepthFrame(object):
    """Class that stores depth frames.

    Args:
        frame: A numpy array storing the depth frame.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.DepthCameraSetup`):  # noqa: E501
            The camera setup used by the sensor that generated this frame.
        original_frame: A numpy array storing the RGB encoded depth image.

    Attributes:
        frame: A numpy array storing the depth frame.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.DepthCameraSetup`):
            The camera setup used by the sensor that generated this frame.
        original_frame: A numpy array storing the RGB encoded depth image.
    """
    def __init__(self, frame, camera_setup, original_frame=None):
        self.frame = frame
        self.camera_setup = camera_setup
        self.original_frame = None
        # Attribute used to cache the depth frame as a point cloud. We're doing
        # this because it is computationally expensive to transform a depth
        # frame to a point cloud.
        self._cached_point_cloud = None

    @classmethod
    def from_simulator_frame(cls,
                             frame,
                             camera_setup,
                             save_original_frame=False):
        """Creates a pylot depth frame from a simulator depth frame.

        Args:
            frame: An image instance containing the depth image.
            camera_setup: The setup of the depth camera.
            save_original_frame: True if the original RGB image needs to be
                saved.

        Returns:
            :py:class:`.DepthFrame`: A depth frame.
        """
        original_frame = None
        # Convert an image containing simulator encoded depth-map to a 2D
        # array containing the depth value of each pixel normalized
        # between [0.0, 1.0]
        _frame = np.frombuffer(frame.raw_data, dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame, (frame.height, frame.width, 4))
        frame = _frame.astype(np.float32)
        if save_original_frame:
            original_frame = copy.deepcopy(frame[:, :, :3])
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        frame = np.dot(frame[:, :, :3], [65536.0, 256.0, 1.0]).astype(np.float32)
        frame /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        # print("Length of depth frame: ", len(frame))
        # print(frame)
        return cls(frame, camera_setup, original_frame)

    def as_numpy_array(self):
        """Returns the depth frame as a numpy array."""
        return self.frame

    def as_point_cloud(self):
        """Converts the depth frame to a 1D array containing the 3D
        position of each pixel in world coordinates.

        See :py:class:`~pylot.drivers.sensor_setup.CameraSetup` for
        coordinate axis orientations.
        """
        far = 1000.0  # max depth in meters.
        intrinsic_mat = self.camera_setup.get_intrinsic_matrix()
        width, height = self.camera_setup.width, self.camera_setup.height
        # 2d pixel coordinates
        pixel_length = width * height
        u_coord = repmat(np.r_[0:width:1], height, 1).reshape(pixel_length)
        v_coord = repmat(np.c_[0:height:1], 1, width).reshape(pixel_length)
        normalized_depth = np.reshape(self.frame, pixel_length)

        # p2d = [u,v,1]
        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

        # P = [X,Y,Z]
        p3d = np.dot(inv(intrinsic_mat), p2d)
        p3d *= normalized_depth * far

        # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
        locations = np.asarray(np.transpose(p3d))
        # Transform the points in 3D world coordinates.
        to_world_transform = self.camera_setup.get_unreal_transform()
        point_cloud = to_world_transform.transform_points(locations)
        return point_cloud

    def get_pixel_locations(self, pixels):
        """ Gets the 3D world locations from pixel coordinates.

        Args:
            pixels: List of pylot.utils.Vector2D pixel coordinates.
        Returns:
            List of pylot.utils.Locations
        """
        if self._cached_point_cloud is None:
            self._cached_point_cloud = self.as_point_cloud()
        pixel_locations = [
            self._cached_point_cloud[pixel.y * self.camera_setup.width +
                                     pixel.x] for pixel in pixels
        ]
        return [
            Location(loc[0], loc[1], loc[2])
            for loc in pixel_locations
        ]

    def pixel_has_same_depth(self, x, y, z: float, threshold: float) -> bool:
        """Checks if the depth of pixel (y,x) is within threshold of z."""
        return abs(self.frame[int(y)][int(x)] * 1000 - z) < threshold

    def resize(self, width: int, height: int):
        """Resizes the frame."""
        import cv2
        self.camera_setup.set_resolution(width, height)
        self.frame = cv2.resize(self.frame,
                                dsize=(width, height),
                                interpolation=cv2.INTER_NEAREST)

    def visualize(self, pygame_display, timestamp=None):
        """Visualizes the frame on a pygame display."""
        if self.original_frame is not None:
            import pygame
            image_np = self.original_frame
            image_np = image_np[:, :, ::-1]
            image_np = np.transpose(image_np, (1, 0, 2))
            pygame.surfarray.blit_array(pygame_display, image_np)
            pygame.display.flip()

    def save(self, timestamp: int, data_path: str, file_base: str):
        """Saves the depth frame to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the depth frame.
            data_path (:obj:`str`): Path where to save the depth frame.
            file_base (:obj:`str`): Base name of the file.
        """
        file_name = os.path.join(data_path,
                                 '{}-{}.pkl'.format(file_base, timestamp))
        pickle.dump(self.as_numpy_array(),
                    open(file_name, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self):
        return 'DepthFrame(camera_setup: {}, frame: {})'.format(
            self.camera_setup, self.frame)

    def __str__(self):
        return 'DepthFrame(camera_setup: {})'.format(self.camera_setup)


# Semantic Labels
CITYSCAPES_LABELS = {
    0: "unlabeled",
    1: "building",
    2: "fence",
    3: "other",
    4: "person",
    5: "pole",
    6: "road_line",
    7: "road",
    8: "sidewalk",
    9: "vegetation",
    10: "car",
    11: "wall",
    12: "traffic_sign",
    13: "sky",
    14: "ground",
    15: "bridge",
    16: "rail_track",
    17: "guard_rail",
    18: "traffic_light",
    19: "static",
    20: "dynamic",
    21: "water",
    22: "terrain",
}

# Cityscapes palette.
CITYSCAPES_CLASSES = {
    0: [0, 0, 0],  # None
    1: [70, 70, 70],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [220, 20, 60],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [128, 64, 128],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0],  # TrafficSigns
    13: [70, 130, 180],  # Sky
    14: [81, 0, 81],  # Ground
    15: [150, 100, 100],  # Bridge
    16: [230, 150, 140],  # RailTrack
    17: [180, 165, 180],  # GuardRail
    18: [250, 170, 30],  # TrafficLight
    19: [110, 190, 160],  # Static
    20: [170, 120, 50],  # Dynamic
    21: [45, 60, 150],  # Water
    22: [145, 170, 100]  # Terrain
}

# XXX(ionel): Note! These classes do not cover all
# the classes from CITYSCAPES. Hence, we can't compare segmentation
# outputs to ground truth.


class SegmentedFrame(object):
    """Stores a semantically segmented frame.

    Args:
        frame: A numpy array storring the segmented frame.
        encoding (:obj:`str`): The encoding of the frame (simulator | cityscapes).  # noqa: E501
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.SegmentedCameraSetup`):  # noqa: E501
            The camera setup used by the sensor that generated this frame.

    Attributes:
        frame: A numpy array storring the segmented frame.
        encoding (:obj:`str`): The encoding of the frame (simulator | cityscapes).  # noqa: E501
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.SegmentedCameraSetup`):  # noqa: E501
            The camera setup used by the sensor that generated this frame.
    """
    def __init__(self, frame, encoding, camera_setup):
        if encoding == 'simulator' or encoding == 'cityscapes':
            self._frame = frame
            self.encoding = encoding
            self.camera_setup = camera_setup
        else:
            raise ValueError(
                'Unexpected encoding {} for segmented frame'.format(encoding))
        self._class_masks = None

    @classmethod
    def from_simulator_image(cls, simulator_image, camera_setup):
        """Creates a pylot camera frame from a simulator frame.

        Note:
            This conversion is slow.

        Returns:
            :py:class:`.SegmentedFrame`: A segmented camera frame.
        """
        # Converts the array containing simulator semantic segmentation labels
        # to a 2D array containing the label of each pixel.
        from carla import Image
        if not isinstance(simulator_image, Image):
            raise ValueError('simulator_image should be of type Image')
        __frame = np.frombuffer(simulator_image.raw_data,
                                dtype=np.dtype("uint8"))
        __frame = np.reshape(
            __frame, (simulator_image.height, simulator_image.width, 4))
        return cls(__frame[:, :, 2], 'simulator', camera_setup)

    def as_cityscapes_palette(self):
        """Returns the frame to the CARLA cityscapes pallete.

        Returns:
           A numpy array.
        """
        if self.encoding == 'cityscapes':
            return self._frame
        else:
            result = np.zeros((self._frame.shape[0], self._frame.shape[1], 3),
                              dtype=np.uint8)
            for key, value in CITYSCAPES_CLASSES.items():
                result[np.where(self._frame == key)] = value
            return result

    def as_numpy_array(self):
        """Returns the segmented frame as a numpy array."""
        return self._frame

    def transform_to_cityscapes(self):
        """Transforms the frame to a cityscapes frame."""
        self._frame = self.as_cityscapes_palette()
        self.encoding = 'cityscapes'

    def in_frame(self, point):
        """Checks if a point is within the frame."""
        return (0 <= point.x <= self.camera_setup.width
                and 0 <= point.y <= self.camera_setup.height)

    def get_traffic_sign_bounding_boxes(self, min_width=2, min_height=3):
        """Extracts traffic sign bounding boxes from the frame.

        Returns:
            list(:py:class:`~pylot.perception.detection.utils.BoundingBox2D`):
            Traffic sign bounding boxes.
        """
        assert self.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        # Set the pixels we are interested in to True.
        traffic_signs_frame = self._get_traffic_sign_pixels()
        # Extracts bounding box from frame.
        bboxes = []
        # Labels the connected segmented pixels.
        map_labeled = measure.label(traffic_signs_frame, connectivity=1)
        # Extract the regions out of the labeled frames.
        for region in measure.regionprops(map_labeled):
            x_min = region.bbox[1]
            x_max = region.bbox[3]
            y_min = region.bbox[0]
            y_max = region.bbox[2]
            # Filter the bboxes that are extremely small.
            if x_max - x_min > min_width and y_max - y_min > min_height:
                bboxes.append(BoundingBox2D(x_min, x_max, y_min, y_max))
        return bboxes

    def _get_per_class_masks(self):
        """ Build a cache of class key to frame mask."""
        assert self.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        if self._class_masks is not None:
            return self._class_masks
        else:
            self._class_masks = []
            for key, value in CITYSCAPES_CLASSES.items():
                mask = np.zeros((self._frame.shape[0], self._frame.shape[1]))
                mask[np.where(self._frame == key)] = 1
                self._class_masks.append(mask)
        return self._class_masks

    def compute_semantic_iou(self, other_frame):
        """Computes IoU for a segmented frame.

        Args:
            other_frame (:py:class:`.SegmentedFrame`): The frame for which to
            compute IoU.

        Returns:
            A tuple comprising of mIoU and a list of IoUs.
        """
        assert (self.encoding == 'cityscapes' and other_frame.encoding
                == 'cityscapes'), 'Not implemented on simulator encoding'
        iou = {}
        for key, value in CITYSCAPES_CLASSES.items():
            #  Do not include None in the mIoU
            if key == 0:
                continue
            target = np.zeros((self._frame.shape[0], self._frame.shape[1], 3))
            prediction = np.zeros(
                (self._frame.shape[0], self._frame.shape[1], 3))
            target[np.where(self._frame == value)] = 1
            prediction[np.where(other_frame._frame == value)] = 1
            intersection = np.logical_and(target, prediction)
            union = np.logical_or(target, prediction)
            sum_intersection = np.sum(intersection)
            sum_union = np.sum(union)
            # Ignore non-existing classes.
            if sum_union > 0:
                iou[key] = float(sum_intersection) / float(sum_union)
        mean_iou = np.mean(list(iou.values()))
        return (mean_iou, iou)

    def compute_semantic_iou_using_masks(self, other_frame):
        """Computes IoU for a segmented frame.

        Computes IoU from per class image masks. This method is the fastest if
        the frames already have precomputed masks.

        Args:
            other_frame (:py:class:`.SegmentedFrame`): The frame for which to
            compute IoU.

        Returns:
            A tuple comprising of mIoU and a list of IoUs.
        """
        assert self.encoding == 'simulator' and \
            other_frame.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        masks = self._get_per_class_masks()
        other_masks = other_frame._get_per_class_masks()
        iou = {}
        for i in range(1, len(CITYSCAPES_CLASSES)):
            intersection = np.logical_and(masks[i], other_masks[i])
            union = np.logical_or(masks[i], other_masks[i])
            sum_intersection = np.sum(intersection)
            sum_union = np.sum(union)

            # Ignore non-existing classes.
            if sum_union > 0:
                iou[i] = float(sum_intersection) / float(sum_union)
                mean_iou = np.mean(list(iou.values()))
        return (mean_iou, iou)

    def save_per_class_masks(self, data_path, timestamp):
        assert self.encoding == 'simulator', \
            'Not implemented on cityscapes encoding'
        masks = self._get_per_class_masks()
        assert len(timestamp.coordinates) == 1
        for k, v in CITYSCAPES_LABELS.items():
            file_name = os.path.join(
                data_path, '{}-{}.png'.format(v, timestamp.coordinates[0]))
            img = Image.fromarray(masks[k])
            img = img.convert('RGB')
            img.save(file_name)

    def save(self, timestamp, data_path, file_base):
        """Saves the segmented frame to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the segmented
                frame.
            data_path (:obj:`str`): Path where to save the segmented frame.
            file_base (:obj:`str`): Base name of the file.
        """
        file_name = os.path.join(data_path,
                                 '{}-{}.png'.format(file_base, timestamp))
        img = Image.fromarray(self.as_cityscapes_palette())
        img.save(file_name)

    def visualize(self, pygame_display, timestamp=None):
        import pygame
        cityscapes_frame = self.as_cityscapes_palette()
        if timestamp is not None:
            add_timestamp(cityscapes_frame, timestamp)
        image_np = np.transpose(cityscapes_frame, (1, 0, 2))
        pygame.surfarray.blit_array(pygame_display, image_np)
        pygame.display.flip()

    def draw_box(self, start_point, end_point, color, thickness=3):
        """Draw a colored box defined by start_point, end_point."""
        start = (int(start_point.x), int(start_point.y))
        end = (int(end_point.x), int(end_point.y))
        cv2.rectangle(self._frame, start, end, color, thickness)

    def draw_point(self, point, color, r=3):
        """Draws a colored point on the segmented frame."""
        cv2.circle(self._frame, (int(point.x), int(point.y)), r, color, -1)

    def draw_text(self, point, text, color=(255, 255, 255)):
        cv2.putText(self._frame,
                    text, (int(point.x), int(point.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def _get_traffic_sign_pixels(self):
        """Returns a frame with the traffic sign pixels set to True."""
        # Shape is height, width
        traffic_signs_frame = np.zeros(
            (self._frame.shape[0], self._frame.shape[1]), dtype=np.bool)
        # 12 is the key for TrafficSigns segmentation in CARLA.
        # Apply mask to only select traffic signs and traffic lights.
        traffic_signs_frame[np.where(
            np.logical_or(self._frame == 12, self._frame == 18))] = True
        return traffic_signs_frame

    def __repr__(self):
        return 'SegmentedFrame(encoding: {}, camera_setup: {}, frame: {})'.format(  # noqa: E501
            self.encoding, self.camera_setup, self._frame)

    def __str__(self):
        return 'SegmentedFrame(encoding: {}, camera_setup: {}, frame: {})'.format(  # noqa: E501
            self.encoding, self.camera_setup, self._frame)


class PointCloud(object):
    """Class that stores points clouds.

    Args:
        points: A (number of points) by 3 numpy array, where each row is
            the (x, y, z) coordinates of a point.
        transform (:py:class:`~Transform`): Transform of the
            point cloud, relative to the ego-vehicle.

    Attributes:
        points: A (number of points) by 3 numpy array, where each row is
            the (x, y, z) coordinates of a point.
        transform (:py:class:`~Transform`): Transform of the
            point cloud, relative to the ego-vehicle.
    """
    def __init__(self, points, lidar_setup: LidarSetup):
        # Transform point cloud from lidar to camera coordinates.
        self._lidar_setup = lidar_setup
        self.global_points = copy.deepcopy(points)
        self.points = self._to_camera_coordinates(points)
        self.transform = lidar_setup.get_transform()

    @classmethod
    def from_simulator_point_cloud(cls, simulator_pc, lidar_setup: LidarSetup):
        """Creates a pylot point cloud from a simulator point cloud.

        Returns:
          :py:class:`.PointCloud`: A point cloud.
        """
        points = np.frombuffer(simulator_pc.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        if lidar_setup.legacy:
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
        else:
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            # Remove the intensity component of the point cloud.
            points = points[:, :3]
        return cls(points, lidar_setup)

    def merge(self, point_cloud):
        """Merges this point cloud with another point cloud.

        Note:
            The method modifies the point cloud inplace.
        """
        self.global_points = np.concatenate(
            (self.global_points, point_cloud.global_points), 0)
        self.points = np.concatenate((self.points, point_cloud.points), 0)

    def _to_camera_coordinates(self, points):
        # Converts points in lidar coordinates to points in camera coordinates.
        # See CameraSetup in pylot/drivers/sensor_setup.py for coordinate
        # axis orientations.
        #
        # The Velodyne coordinate space is defined as:
        # +x into the screen, +y to the left, and +z up.
        #
        # Note: We're using the ROS velodyne driver coordinate
        # system, not the one specified in the Velodyne manual.
        # Link to the ROS coordinate system:
        # https://www.ros.org/reps/rep-0103.html#axis-orientation
        if self._lidar_setup.lidar_type == 'sensor.lidar.ray_cast':
            if self._lidar_setup.legacy:
                # The legacy CARLA Lidar coordinate space is defined as:
                # +x to right, +y out of the screen, +z down.
                to_camera_transform = Transform(matrix=np.array(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
            else:
                # The latest coordiante space is the unreal space.
                to_camera_transform = Transform(matrix=np.array(
                    [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
        elif self._lidar_setup.lidar_type == 'velodyne':
            to_camera_transform = Transform(matrix=np.array(
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
        else:
            raise ValueError('Unexpected lidar type {}'.format(
                self._lidar_setup.lidar_type))
        transformed_points = to_camera_transform.transform_points(points)
        return transformed_points

    def get_pixel_location(self, pixel, camera_setup: CameraSetup):
        """ Gets the 3D world location from pixel coordinates.

        Args:
            pixel (:py:class:`~utils.Vector2D`): Pixel coordinates.
            camera_setup (:py:class:`~sensors.CameraSetup`):
                The setup of the camera with its transform in the world frame
                of reference.

        Returns:
            :py:class:`~utils.Location`: The 3D world location, or None
            if all the point cloud points are behind.
        """
        # Select only points that are in front.
        # Setting the threshold to 0.1 because super close points cause
        # floating point errors.
        fwd_points = self.points[np.where(self.points[:, 2] > 0.1)]
        if len(fwd_points) == 0:
            return None
        intrinsic_mat = camera_setup.get_intrinsic_matrix()
        # Project our 2D pixel location into 3D space, onto the z=1 plane.
        p3d = np.dot(inv(intrinsic_mat), np.array([[pixel.x], [pixel.y],
                                                   [1.0]]))

        if self._lidar_setup.lidar_type == 'sensor.lidar.ray_cast':
            location = PointCloud.get_closest_point_in_point_cloud(
                fwd_points, Vector2D(p3d[0], p3d[1]), normalized=True)
            # Use the depth from the retrieved location.
            p3d *= np.array([location.z])
            p3d = p3d.transpose()
            # Convert from camera to unreal coordinates if the lidar type is
            # sensor.lidar.ray_cast
            to_world_transform = camera_setup.get_unreal_transform()
            camera_point_cloud = to_world_transform.transform_points(p3d)[0]
            pixel_location = Location(camera_point_cloud[0],
                                      camera_point_cloud[1],
                                      camera_point_cloud[2])
        elif self._lidar_setup.lidar_type == 'velodyne':
            location = PointCloud.get_closest_point_in_point_cloud(
                fwd_points, Vector2D(p3d[0], p3d[1]), normalized=False)
            # Use the depth from the retrieved location.
            p3d[2] = location.z
            p3d = p3d.transpose()
            pixel_location = Location(p3d[0, 0], p3d[0, 1], p3d[0, 2])
        return pixel_location

    @staticmethod
    def get_closest_point_in_point_cloud(fwd_points,
                                         pixel,
                                         normalized: bool = False):
        """Finds the closest point in the point cloud to the given point.

        Args:
            pixel (:py:class:`~utils.Vector2D`): Camera coordinates.

        Returns:
            :py:class:`~utils.Location`: Closest point cloud point.
        """
        # Select x and y.
        pc_xy = fwd_points[:, 0:2]
        # Create an array from the x, y coordinates of the point.
        xy = np.array([pixel.x, pixel.y]).transpose()

        # Compute distance
        if normalized:
            # Select z
            pc_z = fwd_points[:, 2]
            # Divize x, y by z
            normalized_pc = pc_xy / pc_z[:, None]
            dist = np.sum((normalized_pc - xy)**2, axis=1)
        else:
            dist = np.sum((pc_xy - xy)**2, axis=1)

        # Select index of the closest point.
        closest_index = np.argmin(dist)

        # Return the closest point.
        return Location(fwd_points[closest_index][0],
                        fwd_points[closest_index][1],
                        fwd_points[closest_index][2])

    def save(self, timestamp: int, data_path: str, file_base: str):
        """Saves the point cloud to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the point cloud.
            data_path (:obj:`str`): Path where to save the point cloud.
            file_base (:obj:`str`): Base name of the file.
        """
        import open3d as o3d
        file_name = os.path.join(data_path,
                                 '{}-{}.ply'.format(file_base, timestamp))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        o3d.io.write_point_cloud(file_name, pcd)

    def visualize(self, pygame_display, timestamp=None):
        """Visualizes the point cloud on a pygame display."""
        import pygame
        (width, height) = pygame_display.get_size()
        # Transform point cloud to top down view.
        lidar_data = np.array(self.global_points[:, :2])
        lidar_data *= (min(width, height) /
                       (2.0 * self._lidar_setup.get_range_in_meters()))
        lidar_data += (0.5 * width, 0.5 * height)
        lidar_data = np.fabs(lidar_data)
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (width, height, 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        pygame.surfarray.blit_array(pygame_display, lidar_img)
        pygame.display.flip()

    def __repr__(self):
        return 'PointCloud(lidar setup: {}, points: {})'.format(
            self._lidar_setup, self.points)

    def __str__(self):
        return 'PointCloud(transform: {}, number of points: {})'.format(
            self.transform, len(self.points))


class CameraFrame(object):
    """Class that stores camera frames.

    Args:
        frame: A numpy array storring the frame.
        camera_setup (:py:class:`~sensors.CameraSetup`):
            The camera setup used by the sensor that generated this frame.

    Attributes:
        frame: A numpy array storring the frame.
        encoding (:obj:`str`): The encoding of the frame (BGR | RGB).
        camera_setup (:py:class:`~sensors.CameraSetup`):
            The camera setup used by the sensor that generated this frame.
    """
    def __init__(self,
                 frame,
                 encoding: str,
                 camera_setup: Union[CameraSetup, None] = None):
        self.frame = frame
        if encoding != 'BGR' and encoding != 'RGB':
            raise ValueError('Unsupported encoding {}'.format(encoding))
        self.encoding = encoding
        self.camera_setup = camera_setup

    @classmethod
    def from_simulator_frame(cls, simulator_frame, camera_setup: CameraSetup):
        """Creates a pylot camera frame from a simulator frame.

        Returns:
            :py:class:`.CameraFrame`: A BGR camera frame.
        """
        from carla import Image
        if not isinstance(simulator_frame, Image):
            raise ValueError('simulator_frame should be of type Image')
        _frame = np.frombuffer(simulator_frame.raw_data,
                               dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame,
                            (simulator_frame.height, simulator_frame.width, 4))
        return cls(np.array(_frame[:, :, :3]), 'BGR', camera_setup)

    def as_numpy_array(self):
        """Returns the camera frame as a numpy array."""
        return self.frame.astype(np.uint8)

    def as_bgr_numpy_array(self):
        """Returns the camera frame as a BGR encoded numpy array."""
        if self.encoding == 'RGB':
            return np.array(self.frame[:, :, ::-1])
        else:
            return self.frame

    def as_rgb_numpy_array(self):
        """Returns the camera frame as a RGB encoded numpy array."""
        if self.encoding == 'BGR':
            return self.frame[:, :, ::-1]
        else:
            return self.frame

    def annotate_with_bounding_boxes(self,
                                     timestamp,
                                     detected_obstacles,
                                     transform=None,
                                     bbox_color_map=PYLOT_BBOX_COLOR_MAP):
        add_timestamp(self.frame, timestamp)
        for obstacle in detected_obstacles:
            obstacle.draw_on_frame(self,
                                   bbox_color_map,
                                   ego_transform=transform)

    def draw_box(self,
                 start_point: Vector2D,
                 end_point: Vector2D,
                 color: Tuple[int, int, int],
                 thickness: float = 2):
        """Draw a colored box defined by start_point, end_point."""
        start = (int(start_point.x), int(start_point.y))
        end = (int(end_point.x), int(end_point.y))
        cv2.rectangle(self.frame, start, end, color, thickness)

    def draw_3d_box(self, corners, color: Tuple[int, int, int]):
        for corner in corners:
            self.draw_point(corner, color)

    def draw_point(self, point: Vector2D, color, r: float = 3):
        """Draws a point on the frame.

        Args:
            point (:py:class:`~utils.Vector2D`): Where to draw the point.
            color: RGB tuple for the color of the point.
        """
        cv2.circle(self.frame, (int(point.x), int(point.y)), r, color, -1)

    def draw_text(self,
                  point: Vector2D,
                  text: str,
                  color: Tuple[int, int, int] = (255, 255, 255)):
        """Draws text on the frame.

        Args:
            point (:py:class:`~utils.Vector2D`): Where to draw the text.
            text (:obj:`str`): The text to draw.
            color: RGB tuple for the color of the text.
        """
        cv2.putText(self.frame,
                    text, (int(point.x), int(point.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def draw_line(self, points: List[Vector2D], color, thickness: float = 3):
        """Draws lines between given points on the frame.

        Args:
            points: List of points of where to draw lines between.
            color: RGB tuple for the color of he line.
        """
        draw_points = np.array([[point.x, point.y] for point in points])
        cv2.polylines(self.frame,
                      np.array([draw_points], dtype=np.int32),
                      False,
                      color,
                      thickness=thickness)

    def in_frame(self, point: Vector2D) -> bool:
        """Checks if a point is within the frame."""
        return (0 <= point.x <= self.camera_setup.width
                and 0 <= point.y <= self.camera_setup.height)

    def resize(self, width: int, height: int):
        """Resizes the frame."""
        self.camera_setup.set_resolution(width, height)
        self.frame = cv2.resize(self.frame,
                                dsize=(width, height),
                                interpolation=cv2.INTER_NEAREST)

    def visualize(self, pygame_display, timestamp=None):
        """Visualizes the frame on a pygame display."""
        import pygame
        if timestamp is not None:
            add_timestamp(self.frame, timestamp)
        if self.encoding != 'RGB':
            image_np = self.as_rgb_numpy_array()
        else:
            image_np = self.frame
        image_np = np.transpose(image_np, (1, 0, 2))
        pygame.surfarray.blit_array(pygame_display, image_np)
        pygame.display.flip()

    def save(self, timestamp: int, data_path: str, file_base: str):
        """Saves the camera frame to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the camera frame.
            data_path (:obj:`str`): Path where to save the camera frame.
            file_base (:obj:`str`): Base name of the file.
        """
        if self.encoding != 'RGB':
            image_np = self.as_rgb_numpy_array()
        else:
            image_np = self.frame
        file_name = os.path.join(data_path,
                                 '{}-{}.png'.format(file_base, timestamp))
        img = Image.fromarray(image_np)
        img.save(file_name)

    def __repr__(self):
        return 'CameraFrame(encoding: {}, camera_setup: {}, frame: {})'.format(
            self.encoding, self.camera_setup, self.frame)

    def __str__(self):
        return 'CameraFrame(encoding: {}, camera_setup: {})'.format(
            self.encoding, self.camera_setup)
