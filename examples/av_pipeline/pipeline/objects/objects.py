"""Core geometry and perception types (Transform, Location, Rotation, Obstacle, ...).

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
import numpy as np

import time
import math
import csv
import itertools
from collections import deque
from enum import Enum


from collections import deque

import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


from examples.av_pipeline.pipeline.control.mpc_utils import Vehicle

VEHICLE_LABELS = {'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle'}

class Rotation(object):
    """Used to represent the rotation of an actor or obstacle.

    Rotations are applied in the order: Roll (X), Pitch (Y), Yaw (Z).
    A 90-degree "Roll" maps the positive Z-axis to the positive Y-axis.
    A 90-degree "Pitch" maps the positive X-axis to the positive Z-axis.
    A 90-degree "Yaw" maps the positive X-axis to the positive Y-axis.

    Args:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.

    Attributes:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.
    """
    def __init__(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    @classmethod
    def from_simulator_rotation(cls, rotation):
        """Creates a pylot Rotation from a simulator rotation.

        Args:
            rotation: An instance of a simulator rotation.

        Returns:
            :py:class:`.Rotation`: A pylot rotation.
        """
        # from carla import Rotation
        # if not isinstance(rotation, Rotation):
        #     raise ValueError('rotation should be of type Rotation')
        return cls(rotation.pitch, rotation.yaw, rotation.roll)

    def as_simulator_rotation(self):
        """ Retrieves the rotation as an instance of a simulator rotation.

        Returns:
            An instance of a simulator class representing the rotation.
        """
        from carla import Rotation
        return Rotation(self.pitch, self.yaw, self.roll)

    def as_numpy_array(self):
        """Retrieves the Rotation as a numpy array."""
        return np.array([self.pitch, self.yaw, self.roll])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Rotation(pitch={}, yaw={}, roll={})'.format(
            self.pitch, self.yaw, self.roll)


class Quaternion(object):
    """ Represents the Rotation of an obstacle or vehicle in quaternion
    notation.

    Args:
        w: The real-part of the quaternion.
        x: The x-part (i) of the quaternion.
        y: The y-part (j) of the quaternion.
        z: The z-part (k) of the quaternion.

    Attributes:
        w: The real-part of the quaternion.
        x: The x-part (i) of the quaternion.
        y: The y-part (j) of the quaternion.
        z: The z-part (k) of the quaternion.
        matrix: A 3x3 numpy array that can be used to rotate 3D vectors from
            body frame to world frame.
    """
    def __init__(self, w: float, x: float, y: float, z: float):
        norm = np.linalg.norm([w, x, y, z])
        if norm < 1e-50:
            self.w, self.x, self.y, self.z = 0, 0, 0, 0
        else:
            self.w = w / norm
            self.x = x / norm
            self.y = y / norm
            self.z = z / norm
        self.matrix = Quaternion._create_matrix(self.w, self.x, self.y, self.z)

    @staticmethod
    def _create_matrix(w, x, y, z):
        """Creates a Rotation matrix that can be used to transform 3D vectors
        from body frame to world frame.

        Note that this yields the same matrix as a Transform object with the
        quaternion converted to the Euler rotation except this matrix only does
        rotation and no translation.

        Specifically, this matrix is equivalent to:
            Transform(location=Location(0, 0, 0),
                      rotation=self.as_rotation()).matrix[:3, :3]

        Returns:
            A 3x3 numpy array that can be used to rotate 3D vectors from body
            frame to world frame.
        """
        x2, y2, z2 = x * 2, y * 2, z * 2
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        m = np.array([[1.0 - (yy + zz), xy - wz, xz + wy],
                      [xy + wz, 1.0 - (xx + zz), yz - wx],
                      [xz - wy, yz + wx, 1.0 - (xx + yy)]])
        return m

    @classmethod
    def from_rotation(cls, rotation: Rotation):
        """Creates a Quaternion from a rotation including pitch, roll, yaw.

        Args:
            rotation (:py:class:`.Rotation`): A pylot rotation representing
                the rotation of the object in degrees.

        Returns:
            :py:class:`.Quaternion`: The quaternion representation of the
            rotation.
        """
        roll_by_2 = np.radians(rotation.roll) / 2.0
        pitch_by_2 = np.radians(rotation.pitch) / 2.0
        yaw_by_2 = np.radians(rotation.yaw) / 2.0

        cr, sr = np.cos(roll_by_2), np.sin(roll_by_2)
        cp, sp = np.cos(pitch_by_2), np.sin(pitch_by_2)
        cy, sy = np.cos(yaw_by_2), np.sin(yaw_by_2)

        w = cr * cp * cy + sr * sp * sy
        x = cr * sp * sy - sr * cp * cy
        y = -cr * sp * cy - sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(w, x, y, z)

    @classmethod
    def from_angular_velocity(cls, angular_velocity, dt: float):
        """Creates a Quaternion from an angular velocity vector and the time
        delta to apply it for.

        Args:
            angular_velocity (:py:class:`.Vector3D`): The vector representing
                the angular velocity of the object in the body-frame.
            dt (float): The time delta to apply the angular velocity for.

        Returns:
            :py:class:`.Quaternion`: The quaternion representing the rotation
                undergone by the object with the given angular velocity in the
                given delta time.
        """
        angular_velocity_np = angular_velocity.as_numpy_array() * dt
        magnitude = np.linalg.norm(angular_velocity_np)

        w = np.cos(magnitude / 2.0)
        if magnitude < 1e-50:
            # To avoid instabilities and nan.
            x, y, z = 0, 0, 0
        else:
            imaginary = angular_velocity_np / magnitude * np.sin(
                magnitude / 2.0)
            x, y, z = imaginary
        return cls(w, x, y, z)

    def as_rotation(self) -> Rotation:
        """Retrieve the Quaternion as a Rotation in degrees.

        Returns:
            :py:class:`.Rotation`: The euler-angle equivalent of the
                Quaternion in degrees.
        """
        SINGULARITY_THRESHOLD = 0.4999995
        RAD_TO_DEG = (180.0) / np.pi

        singularity_test = self.z * self.x - self.w * self.y
        yaw_y = 2.0 * (self.w * self.z + self.x * self.y)
        yaw_x = (1.0 - 2.0 * (self.y**2 + self.z**2))

        pitch, yaw, roll = None, None, None
        if singularity_test < -SINGULARITY_THRESHOLD:
            pitch = -90.0
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = -yaw - (2.0 * np.arctan2(self.x, self.w) * RAD_TO_DEG)
        elif singularity_test > SINGULARITY_THRESHOLD:
            pitch = 90.0
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = yaw - (2.0 * np.arctan2(self.x, self.w) * RAD_TO_DEG)
        else:
            pitch = np.arcsin(2.0 * singularity_test) * RAD_TO_DEG
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = np.arctan2(-2.0 * (self.w * self.x + self.y * self.z),
                              (1.0 - 2.0 *
                               (self.x**2 + self.y**2))) * RAD_TO_DEG
        return Rotation(pitch, yaw, roll)

    def __mul__(self, other):
        """Returns the product self * other.  The product is NOT commutative.

        The product is defined in Unreal as:
         [ (Q2.w * Q1.x) + (Q2.x * Q1.w) + (Q2.y * Q1.z) - (Q2.z * Q1.y),
           (Q2.w * Q1.y) - (Q2.x * Q1.z) + (Q2.y * Q1.w) + (Q2.z * Q1.x),
           (Q2.w * Q1.z) + (Q2.x * Q1.y) - (Q2.y * Q1.x) + (Q2.z * Q1.w),
           (Q2.w * Q1.w) - (Q2.x * Q1.x) - (Q2.y * Q1.y) - (Q2.z * Q1.z) ]
        Copied from DirectX's XMQuaternionMultiply function.
        """
        q1, q2 = other, self
        x = (q2.w * q1.x) + (q2.x * q1.w) + (q2.y * q1.z) - (q2.z * q1.y)
        y = (q2.w * q1.y) - (q2.x * q1.z) + (q2.y * q1.w) + (q2.z * q1.x)
        z = (q2.w * q1.z) + (q2.x * q1.y) - (q2.y * q1.x) + (q2.z * q1.w)
        w = (q2.w * q1.w) - (q2.x * q1.x) - (q2.y * q1.y) - (q2.z * q1.z)
        return Quaternion(w, x, y, z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Quaternion (w={}, x={}, y={}, z={})'.format(
            self.w, self.x, self.y, self.z)


class Vector3D(object):
    """Represents a 3D vector and provides useful helper functions.

    Args:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.

    Attributes:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.
    """
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    @classmethod
    def from_simulator_vector(cls, vector):
        """Creates a pylot Vector3D from a simulator 3D vector.

        Args:
            vector: An instance of a simulator 3D vector.

        Returns:
            :py:class:`.Vector3D`: A pylot 3D vector.
        """
        # from carla import Vector3D
        # if not isinstance(vector, Vector3D):
        #     raise ValueError('The vector must be a Vector3D')
        return cls(vector.x, vector.y, vector.z)

    def as_numpy_array(self):
        """Retrieves the 3D vector as a numpy array."""
        return np.array([self.x, self.y, self.z])

    def as_numpy_array_2D(self):
        """Drops the 3rd dimension."""
        return np.array([self.x, self.y])

    def as_simulator_vector(self):
        """Retrieves the 3D vector as an instance of simulator 3D vector.

        Returns:
            An instance of the simulator class representing the 3D vector.
        """
        from carla import Vector3D
        return Vector3D(self.x, self.y, self.z)

    def l1_distance(self, other):
        """Calculates the L1 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector3D`): The other vector used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z -
                                                                   other.z)

    def l2_distance(self, other) -> float:
        """Calculates the L2 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector3D`): The other vector used to
                calculate the L2 distance to.

        Returns:
            :obj:`float`: The L2 distance between the two points.
        """
        vec = np.array([self.x - other.x, self.y - other.y, self.z - other.z])
        return np.linalg.norm(vec)

    def magnitude(self):
        """Returns the magnitude of the 3D vector."""
        return np.linalg.norm(self.as_numpy_array())

    def to_camera_view(self, extrinsic_matrix, intrinsic_matrix):
        """Converts the given 3D vector to the view of the camera using
        the extrinsic and the intrinsic matrix.

        Args:
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            :py:class:`.Vector3D`: An instance with the coordinates converted
            to the camera view.
        """
        position_vector = np.array([[self.x], [self.y], [self.z], [1.0]])

        # Transform the points to the camera in 3D.
        transformed_3D_pos = np.dot(np.linalg.inv(extrinsic_matrix),
                                    position_vector)

        # Transform the points to 2D.
        position_2D = np.dot(intrinsic_matrix, transformed_3D_pos[:3])

        # Normalize the 2D points.
        location_2D = type(self)(float(position_2D[0] / position_2D[2]),
                                 float(position_2D[1] / position_2D[2]),
                                 float(position_2D[2]))
        return location_2D

    def rotate(self, angle: float):
        """Rotate the vector by a given angle.

        Args:
            angle (float): The angle to rotate the Vector by (in degrees).

        Returns:
            :py:class:`.Vector3D`: An instance with the coordinates of the
            rotated vector.
        """
        x_ = math.cos(math.radians(angle)) * self.x - math.sin(
            math.radians(angle)) * self.y
        y_ = math.sin(math.radians(angle)) * self.x - math.cos(
            math.radians(angle)) * self.y
        return type(self)(x_, y_, self.z)

    def __add__(self, other):
        """Adds the two vectors together and returns the result."""
        return type(self)(x=self.x + other.x,
                          y=self.y + other.y,
                          z=self.z + other.z)

    def __sub__(self, other):
        """Subtracts the other vector from self and returns the result."""
        return type(self)(x=self.x - other.x,
                          y=self.y - other.y,
                          z=self.z - other.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector3D(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Vector2D(object):
    """Represents a 2D vector and provides helper functions."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_numpy_array(self):
        """Retrieves the 2D vector as a numpy array."""
        return np.array([self.x, self.y])

    def get_angle(self, other) -> float:
        """Computes the angle between the vector and another vector
           in radians."""
        angle = math.atan2(self.y, self.x) - math.atan2(other.y, other.x)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def l1_distance(self, other) -> float:
        """Calculates the L1 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector2D`): The other vector used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def l2_distance(self, other) -> float:
        """Calculates the L2 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector2D`): The other vector used to
                calculate the L2 distance to.

        Returns:
            :obj:`float`: The L2 distance between the two points.
        """
        vec = np.array([self.x - other.x, self.y - other.y])
        return np.linalg.norm(vec)

    def magnitude(self):
        """Returns the magnitude of the 2D vector."""
        return np.linalg.norm(self.as_numpy_array())

    def __add__(self, other):
        """Adds the two vectors together and returns the result. """
        return type(self)(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        """Subtracts the other vector from self and returns the result. """
        return type(self)(x=self.x - other.x, y=self.y - other.y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector2D(x={}, y={})'.format(self.x, self.y)


class Location(Vector3D):
    """Stores a 3D location, and provides useful helper methods.

    Args:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.

    Attributes:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.
    """
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super(Location, self).__init__(x, y, z)

    @classmethod
    def from_simulator_location(cls, location):
        """Creates a pylot Location from a simulator location.

        Args:
            location: An instance of a simulator location.

        Returns:
            :py:class:`.Location`: A pylot location.
        """
        # from carla import Location, Vector3D
        # if not (isinstance(location, Location)
        #         or isinstance(location, Vector3D)):
        #     raise ValueError('The location must be a Location or Vector3D')
        return cls(location.x, location.y, location.z)

    @classmethod
    def from_gps(cls, latitude: float, longitude: float, altitude: float):
        """Creates Location from GPS (latitude, longitude, altitude).

        This is the inverse of the _location_to_gps method found in
        https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
        """
        EARTH_RADIUS_EQUA = 6378137.0
        # The following reference values are applicable for towns 1 through 7,
        # and are taken from the corresponding OpenDrive map files.
        # LAT_REF = 49.0
        # LON_REF = 8.0
        # TODO: Do not hardcode. Get the references from the open drive file.
        LAT_REF = 0.0
        LON_REF = 0.0

        scale = math.cos(LAT_REF * math.pi / 180.0)
        basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * LON_REF
        basey = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + LAT_REF) * math.pi / 360.0))

        x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
        y = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

        # This wasn't in the original method, but seems to be necessary.
        y *= -1

        return cls(x, y, altitude)

    def distance(self, other) -> float:
        """Calculates the Euclidean distance between the given point and the
        other point.

        Args:
            other (:py:class:`~.Location`): The other location used to
                calculate the Euclidean distance to.

        Returns:
            :obj:`float`: The Euclidean distance between the two points.
        """
        return (self - other).magnitude()

    def as_vector_2D(self) -> Vector2D:
        """Transforms the Location into a Vector2D.

        Note:
            The method just drops the z-axis.

        Returns:
            :py:class:`.Vector2D`: A 2D vector.
        """
        return Vector2D(self.x, self.y)

    def as_simulator_location(self):
        """Retrieves the location as a simulator location instance.

        Returns:
            An instance of the simulator class representing the location.
        """
        from carla import Location
        return Location(self.x, self.y, self.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Location(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Transform(object):
    """A class that stores the location and rotation of an obstacle.

    It can be created from a simulator transform, defines helper functions
    needed in Pylot, and makes the simulator transform serializable.

    A transform object is instantiated with either a location and a rotation,
    or using a matrix.

    Args:
        location (:py:class:`.Location`, optional): The location of the object
            represented by the transform.
        rotation (:py:class:`.Rotation`, optional): The rotation  (in degrees)
            of the object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.

    Attributes:
        location (:py:class:`.Location`): The location of the object
            represented by the transform.
        rotation (:py:class:`.Rotation`): The rotation (in degrees) of the
            object represented by the transform.
        forward_vector (:py:class:`.Vector3D`): The forward vector of the
            object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.
    """
    def __init__(self,
                 location: Location = None,
                 rotation: Rotation = None,
                 matrix=None):
        if matrix is not None:
            self.matrix = matrix
            self.location = Location(matrix[0, 3], matrix[1, 3], matrix[2, 3])

            # Forward vector is retrieved from the matrix.
            self.forward_vector = \
                Vector3D(self.matrix[0, 0], self.matrix[1, 0],
                         self.matrix[2, 0])
            pitch_r = math.asin(np.clip(self.forward_vector.z, -1, 1))
            yaw_r = math.acos(
                np.clip(self.forward_vector.x / math.cos(pitch_r), -1, 1))
            roll_r = math.asin(
                np.clip(matrix[2, 1] / (-1 * math.cos(pitch_r)), -1, 1))
            self.rotation = Rotation(math.degrees(pitch_r),
                                     math.degrees(yaw_r), math.degrees(roll_r))
        else:
            self.location, self.rotation = location, rotation
            self.matrix = Transform._create_matrix(self.location,
                                                   self.rotation)

            # Forward vector is retrieved from the matrix.
            self.forward_vector = \
                Vector3D(self.matrix[0, 0], self.matrix[1, 0],
                         self.matrix[2, 0])

    @classmethod
    def from_simulator_transform(cls, transform):
        """Creates a pylot transform from a simulator transform.

        Args:
            transform: A simulator transform.

        Returns:
            :py:class:`.Transform`: An instance of a pylot transform.
        """
        # from carla import Transform
        # if not isinstance(transform, Transform):
        #     raise ValueError('transform should be of type Transform')
        return cls(Location.from_simulator_location(transform.location),
                   Rotation.from_simulator_rotation(transform.rotation))

    @staticmethod
    def _create_matrix(location, rotation):
        """Creates a transformation matrix to convert points in the 3D world
        coordinate space with respect to the object.

        Use the transform_points function to transpose a given set of points
        with respect to the object.

        Args:
            location (:py:class:`.Location`): The location of the object
                represented by the transform.
            rotation (:py:class:`.Rotation`): The rotation of the object
                represented by the transform.

        Returns:
            A 4x4 numpy matrix which represents the transformation matrix.
        """
        matrix = np.identity(4)
        cy = math.cos(np.radians(rotation.yaw))
        sy = math.sin(np.radians(rotation.yaw))
        cr = math.cos(np.radians(rotation.roll))
        sr = math.sin(np.radians(rotation.roll))
        cp = math.cos(np.radians(rotation.pitch))
        sp = math.sin(np.radians(rotation.pitch))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = (cp * cy)
        matrix[0, 1] = (cy * sp * sr - sy * cr)
        matrix[0, 2] = -1 * (cy * sp * cr + sy * sr)
        matrix[1, 0] = (sy * cp)
        matrix[1, 1] = (sy * sp * sr + cy * cr)
        matrix[1, 2] = (cy * sr - sy * sp * cr)
        matrix[2, 0] = (sp)
        matrix[2, 1] = -1 * (cp * sr)
        matrix[2, 2] = (cp * cr)
        return matrix

    def __transform(self, points, matrix):
        """Internal function to transform the points according to the
        given matrix. This function either converts the points from
        coordinate space relative to the transform to the world coordinate
        space (using self.matrix), or from world coordinate space to the
        space relative to the transform (using inv(self.matrix))

        Args:
            points: An n by 3 numpy array, where each row is the
                (x, y, z) coordinates of a point.
            matrix: The matrix of the transformation to apply.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        # Needed format: [[X0,..Xn],[Y0,..Yn],[Z0,..Zn]].
        # So let's transpose the point matrix.
        points = points.transpose()

        # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)

        # Point transformation (depends on the given matrix)
        points = np.dot(matrix, points)

        # Get all but the last row in array form.
        points = np.asarray(points[0:3].transpose()).astype(np.float16)

        return points

    def transform_points(self, points):
        """Transforms the given set of points (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        For example, if the transform is at location (3, 0, 0) and the
        location passed to the argument is (10, 0, 0), this function will
        return (13, 0, 0) i.e. the location of the argument in the world
        coordinate space.

        Args:
            points: A (number of points) by 3 numpy array, where each row is
                the (x, y, z) coordinates of a point.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        return self.__transform(points, self.matrix)

    def inverse_transform_points(self, points):
        """Transforms the given set of points (specified in world coordinate
        space) to be relative to the given transform.

        For example, if the transform is at location (3, 0, 0) and the location
        passed to the argument is (10, 0, 0), this function will return
        (7, 0, 0) i.e. the location of the argument relative to the given
        transform.

        Args:
            points: A (number of points) by 3 numpy array, where each row is
                the (x, y, z) coordinates of a point.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        return self.__transform(points, np.linalg.inv(self.matrix))

    def transform_locations(self, locations):
        """Transforms the given set of locations (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        This method has the same functionality as transform_points, and
        is provided for convenience; when dealing with a large number of
        points, it is advised to use transform_points to avoid the slow
        conversion between a numpy array and list of locations.

        Args:
            points (list(:py:class:`.Location`)): List of locations.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """
        points = np.array([loc.as_numpy_array() for loc in locations])
        transformed_points = self.__transform(points, self.matrix)
        return [Location(x, y, z) for x, y, z in transformed_points]

    def inverse_transform_locations(self, locations):
        """Transforms the given set of locations (specified in world coordinate
        space) to be relative to the given transform.

        This method has the same functionality as inverse_transform_points,
        and is provided for convenience; when dealing with a large number of
        points, it is advised to use inverse_transform_points to avoid the slow
        conversion between a numpy array and list of locations.

        Args:
            points (list(:py:class:`.Location`)): List of locations.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """

        points = np.array([loc.as_numpy_array() for loc in locations])
        transformed_points = self.__transform(points,
                                              np.linalg.inv(self.matrix))
        return [Location(x, y, z) for x, y, z in transformed_points]

    def as_simulator_transform(self):
        """Converts the transform to a simulator transform.

        Returns:
            An instance of the simulator class representing the Transform.
        """
        from carla import Location, Rotation, Transform
        return Transform(
            Location(self.location.x, self.location.y, self.location.z),
            Rotation(pitch=self.rotation.pitch,
                     yaw=self.rotation.yaw,
                     roll=self.rotation.roll))

    def get_angle_and_magnitude(self, target_loc):
        """Computes relative angle between the transform and a target location.

        Args:
            target_loc (:py:class:`.Location`): Location of the target.

        Returns:
            Angle in radians and vector magnitude.
        """
        target_vec = target_loc.as_vector_2D() - self.location.as_vector_2D()
        magnitude = target_vec.magnitude()
        if magnitude > 0:
            forward_vector = Vector2D(
                math.cos(math.radians(self.rotation.yaw)),
                math.sin(math.radians(self.rotation.yaw)))
            angle = target_vec.get_angle(forward_vector)
        else:
            angle = 0
        return angle, magnitude

    def is_within_distance_ahead(self, dst_loc: Location,
                                 max_distance: float) -> bool:
        """Checks if a location is within a distance.

        Args:
            dst_loc (:py:class:`.Location`): Location to compute distance to.
            max_distance (:obj:`float`): Maximum allowed distance.

        Returns:
            bool: True if other location is within max_distance.
        """
        d_angle, norm_dst = self.get_angle_and_magnitude(dst_loc)
        # Return if the vector is too small.
        if norm_dst < 0.001:
            return True
        # Return if the vector is greater than the distance.
        if norm_dst > max_distance:
            return False
        return d_angle < 90.0

    def inverse_transform(self):
        """Returns the inverse transform of this transform."""
        new_matrix = np.linalg.inv(self.matrix)
        return Transform(matrix=new_matrix)

    def __mul__(self, other):
        new_matrix = np.dot(self.matrix, other.matrix)
        return Transform(matrix=new_matrix)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.location:
            return "Transform(location: {}, rotation: {})".format(
                self.location, self.rotation)
        else:
            return "Transform({})".format(str(self.matrix))

class BoundingBox2D(object):
    """Class that stores a 2D bounding box."""
    def __init__(self, x_min, x_max, y_min, y_max):
        assert x_min < x_max and y_min < y_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_min_point(self) -> Vector2D:
        return Vector2D(self.x_min, self.y_min)

    def get_max_point(self) -> Vector2D:
        return Vector2D(self.x_max, self.y_max)

    def get_height(self):
        return self.y_max - self.y_min

    def get_width(self):
        return self.x_max - self.x_min

    def get_center_point(self) -> Vector2D:
        return Vector2D((self.x_min + self.x_max) // 2,
                                    (self.y_min + self.y_max) // 2)

    def as_width_height_bbox(self):
        return [self.x_min, self.y_min, self.get_width(), self.get_height()]

    def is_within(self, point) -> bool:
        """Checks if a point is within the bounding box."""
        return (point.x >= self.x_min and point.x <= self.x_max
                and point.y >= self.y_min and point.y <= self.y_max)

    def calculate_iou(self, other_bbox) -> float:
        """Calculate the IoU of a single bounding box.

        Args:
            other_bbox (:py:class:`.BoundingBox2D`): The other bounding box.

        Returns:
            :obj:`float`: The IoU of the two bounding boxes.
        """
        if (other_bbox.x_min > other_bbox.x_max
                or other_bbox.y_min > other_bbox.y_max):
            raise AssertionError(
                "Other bbox is malformed {}".format(other_bbox))

        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise AssertionError("Bounding box is malformed {}".format(self))

        if (self.x_max < other_bbox.x_min or other_bbox.x_max < self.x_min
                or self.y_max < other_bbox.y_min
                or other_bbox.y_max < self.y_min):
            return 0.0

        inter_x1 = max([self.x_min, other_bbox.x_min])
        inter_x2 = min([self.x_max, other_bbox.x_max])

        inter_y1 = max([self.y_min, other_bbox.y_min])
        inter_y2 = min([self.y_max, other_bbox.y_max])

        inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        gt_area = (self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)
        pred_area = (other_bbox.x_max - other_bbox.x_min +
                     1) * (other_bbox.y_max - other_bbox.y_min + 1)
        return float(inter_area) / (gt_area + pred_area - inter_area)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'BoundingBox2D(xmin: {}, xmax: {}, ymin: {}, ymax: {})'.format(
            self.x_min, self.x_max, self.y_min, self.y_max)


class BoundingBox3D(object):
    """Class used to store a 3D bounding box.

    Args:
        transform (:py:class:`~utils.Transform`): Transform of the
            bounding box (rotation is (0, 0, 0)).
        extent (:py:class:`~utils.Vector3D`): The extent of the bounding
            box.

    Attributes:
        transform (:py:class:`~utils.Transform`): Transform of the
            bounding box (rotation is (0, 0, 0)).
        extent (:py:class:`~utils.Vector3D`): The extent of the bounding
            box.
    """
    def __init__(self,
                 transform: Transform = None,
                 extent: Vector3D = None,
                 corners=None):
        self.transform = transform
        self.extent = extent
        self.corners = corners

    @classmethod
    def from_dimensions(cls, bbox_dimensions, location, rotation_y):
        """Creates a 3D bounding box.

        Args:
            bbox_dimensions: The height, width and length of the bbox.
            location: The location of the box in the camera frame.
            rotation: The rotation of the bbox.

        Returns:
            :py:class:`.BoundingBox3D`: A bounding box instance.
        """
        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        l, w, h = bbox_dimensions[2], bbox_dimensions[1], bbox_dimensions[0]
        x_corners = [
            l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2
        ]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [
            w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2
        ]
        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners).transpose(1, 0)
        corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
            1, 3)
        return cls(corners=corners_3d)

    @classmethod
    def from_simulator_bounding_box(cls, bbox):
        """Creates a pylot bounding box from a simulator bounding box.

        Args:
            bbox: The bounding box to transform.

        Returns:
            :py:class:`.BoundingBox3D`: A bounding box instance.
        """
        transform = Transform(
            Location.from_simulator_location(bbox.location),
            Rotation())
        extent = Vector3D.from_simulator_vector(bbox.extent)
        return cls(transform, extent)

    def as_simulator_bounding_box(self):
        """Retrieves the bounding box as instance of a simulator bounding box.

        Returns:
            A instance of a simulator class that represents the bounding box.
        """
        from carla import BoundingBox
        bb_loc = self.transform.location.as_simulator_location()
        bb_extent = self.extent.as_simulator_vector()
        return BoundingBox(bb_loc, bb_extent)

    def visualize(self, world, actor_transform, time_between_frames=100):
        """Visualizes the bounding box on the world.

        Args:
            world: The simulator world instance to visualize the bounding
                box on.
            actor_transform (:py:class:`~utils.Transform`): The current
                transform of the actor that the bounding box is of.
            time_between_frames (:obj:`float`): Time in ms to show the bounding
                box for.
        """
        bb = self.as_simulator_bounding_box()
        bb.location += actor_transform.location()
        world.debug.draw_box(bb,
                             actor_transform.rotation.as_simulator_rotation(),
                             life_time=time_between_frames / 1000.0)

    def to_camera_view(self, obstacle_transform: Transform,
                       extrinsic_matrix, intrinsic_matrix):
        """Converts the coordinates of the bounding box for the given obstacle
        to the coordinates in the view of the camera.

        This method retrieves the extent of the bounding box, transforms them
        to coordinates relative to the bounding box origin, then converts those
        to coordinates relative to the obstacle.

        These coordinates are then considered to be in the world coordinate
        system, which is mapped into the camera view. A negative z-value
        signifies that the bounding box is behind the camera plane.

        Note that this function does not cap the coordinates to be within the
        size of the camera image.

        Args:
            obstacle_transform (:py:class:`~utils.Transform`): The
                transform of the obstacle that the bounding box is associated
                with.
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            A list of 8 Location instances specifying the 8 corners of the
            bounding box.
        """
        # Retrieve the eight coordinates of the bounding box with respect to
        # the origin of the bounding box.
        import numpy as np
        if self.corners is not None:
            pts_2d = np.dot(intrinsic_matrix,
                            self.corners.transpose(1, 0)).transpose(1, 0)
            pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
            camera_coordinates = [
                Vector2D(pt[0], pt[1]) for pt in pts_2d
            ]
            return camera_coordinates

        extent = self.extent
        bbox = np.array([
            Location(x=+extent.x, y=+extent.y, z=-extent.z),
            Location(x=-extent.x, y=+extent.y, z=-extent.z),
            Location(x=-extent.x, y=-extent.y, z=-extent.z),
            Location(x=+extent.x, y=-extent.y, z=-extent.z),
            Location(x=+extent.x, y=+extent.y, z=+extent.z),
            Location(x=-extent.x, y=+extent.y, z=+extent.z),
            Location(x=-extent.x, y=-extent.y, z=+extent.z),
            Location(x=+extent.x, y=-extent.y, z=+extent.z)
        ])

        # Transform the vertices with respect to the bounding box transform.
        bbox = self.transform.transform_locations(bbox)

        # Convert the bounding box relative to the world.
        bbox = obstacle_transform.transform_locations(bbox)

        # Obstacle's transform is relative to the world. Thus, the bbox
        # contains the 3D bounding box vertices relative to the world.
        camera_coordinates = []
        for vertex in bbox:
            location_2D = vertex.to_camera_view(extrinsic_matrix,
                                                intrinsic_matrix)

            # Add the points to the image.
            camera_coordinates.append(location_2D)

        return camera_coordinates

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox3D(transform: {}, extent: {})".format(
            self.transform, self.extent)


def get_bounding_box_in_camera_view(bb_coordinates, image_width, image_height):
    """Creates the bounding box in the view of the camera image using the
    coordinates expressed with respect to the camera transform.

    Args:
        bb_coordinates: 8 :py:class:`~utils.Location` coordinates of
            the bounding box relative to the camera transform.
        image_width (:obj:`int`): The width of the image being published by the
            camera.
        image_height (:obj:`int`): The height of the image being published by
            the camera.

    Returns:
        :py:class:`.BoundingBox2D`: a bounding box, or None if the bounding box
            does not fall into the view of the camera.
    """
    # Make sure that atleast 2 of the bounding box coordinates are in front.
    z_vals = [loc.z for loc in bb_coordinates if loc.z >= 0]
    if len(z_vals) < 2:
        return None

    # Create the thresholding line segments of the camera view.
    from shapely.geometry import LineString
    left = LineString(((0, 0), (0, image_height)))
    bottom = LineString(((0, image_height), (image_width, image_height)))
    right = LineString(((image_width, image_height), (image_width, 0)))
    top = LineString(((image_width, 0), (0, 0)))
    camera_thresholds = [left, bottom, right, top]

    def threshold(p1, p2):
        points = []
        # If the points are themselves within the image, add them to the
        # set of thresholded points.
        if (p1[0] >= 0 and p1[0] < image_width and p1[1] >= 0
                and p1[1] < image_height):
            points.append(p1)

        if (p2[0] >= 0 and p2[0] < image_width and p2[1] >= 0
                and p2[1] < image_height):
            points.append(p2)

        # Compute the intersection of the line segment formed by p1 -- p2
        # with all the thresholds of the camera image.
        p12 = LineString((p1, p2))
        for camera_threshold in camera_thresholds:
            p = p12.intersection(camera_threshold)
            if not p.is_empty:
                if p.geom_type == 'Point':
                    points.append((p.x, p.y))
                elif p.geom_type == 'LineString':
                    for coord in p.coords:
                        points.append((coord[0], coord[1]))
        return points

    # Go over each of the segments of the bounding box and threshold it to
    # be inside the image.
    thresholded_points = []
    points = [(int(loc.x), int(loc.y)) for loc in bb_coordinates]
    # Bottom plane thresholded.
    thresholded_points.extend(threshold(points[0], points[1]))
    thresholded_points.extend(threshold(points[1], points[2]))
    thresholded_points.extend(threshold(points[2], points[3]))
    thresholded_points.extend(threshold(points[3], points[0]))

    # Top plane thresholded.
    thresholded_points.extend(threshold(points[4], points[5]))
    thresholded_points.extend(threshold(points[5], points[6]))
    thresholded_points.extend(threshold(points[6], points[7]))
    thresholded_points.extend(threshold(points[7], points[4]))

    # Remaining segments thresholded.
    thresholded_points.extend(threshold(points[0], points[4]))
    thresholded_points.extend(threshold(points[1], points[5]))
    thresholded_points.extend(threshold(points[2], points[6]))
    thresholded_points.extend(threshold(points[3], points[7]))

    if len(thresholded_points) == 0:
        return None
    else:
        x = [int(x) for x, _ in thresholded_points]
        y = [int(y) for _, y in thresholded_points]
        if min(x) < max(x) and min(y) < max(y):
            return BoundingBox2D(min(x), max(x), min(y), max(y))
        else:
            return None


class Pose(object):
    def __init__(self,
                 transform: Transform,
                 forward_speed: float,
                 velocity_vector: Vector3D = None,
                 localization_time: float = None):
        # if not isinstance(transform, Transform):
        #     raise ValueError(
        #         'transform should be of type Transform')
        self.transform = transform
        # Forward speed in m/s.
        self.forward_speed = forward_speed
        self.velocity_vector = velocity_vector
        if localization_time is None:
            self.localization_time = time.time()
        else:
            self.localization_time = localization_time

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Pose(transform: {}, forward speed: {}, velocity vector: {})"\
            .format(self.transform, self.forward_speed, self.velocity_vector)


class LaneMarkingColor(Enum):
    """Enum that defines the lane marking colors according to OpenDrive 1.4.

    The goal of this enum is to make sure that lane colors are correctly
    propogated from the simulator to Pylot.
    """
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4
    OTHER = 5


class LaneMarkingType(Enum):
    """Enum that defines the lane marking types according to OpenDrive 1.4.

    The goal of this enum is to make sure that lane markings are correctly
    propogated from the simulator to Pylot.
    """
    OTHER = 0
    BROKEN = 1
    SOLID = 2
    SOLIDSOLID = 3
    SOLIDBROKEN = 4
    BROKENSOLID = 5
    BROKENBROKEN = 6
    BOTTSDOTS = 7
    GRASS = 8
    CURB = 9
    NONE = 10


class LaneChange(Enum):
    """ Enum that defines the permission to turn either left, right, both or
    none for a given lane.

    The goal of this enum is to make sure that the lane change types are
    correctly propogated from the simulator to Pylot.
    """
    NONE = 0
    RIGHT = 1
    LEFT = 2
    BOTH = 3


class LaneType(Enum):
    """Enum that defines the type of the lane according to OpenDrive 1.4.

    The goal of this enum is to make sure that the lane change types are
    correctly propogated from the simulator to Pylot.
    """
    NONE = 1
    DRIVING = 2
    STOP = 4
    SHOULDER = 8
    BIKING = 16
    SIDEWALK = 32
    BORDER = 64
    RESTRICTED = 128
    PARKING = 256
    BIDIRECTIONAL = 512
    MEDIAN = 1024
    SPECIAL1 = 2048
    SPECIAL2 = 4096
    SPECIAL3 = 8192
    ROADWORKS = 16384
    TRAM = 32768
    RAIL = 65536
    ENTRY = 131072
    EXIT = 262144
    OFFRAMP = 524288
    ONRAMP = 1048576
    ANY = 4294967294


class RoadOption(Enum):
    """Enum that defines the possible high-level route plans.

    RoadOptions are usually attached to waypoints we receive from
    the challenge environment.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANE_FOLLOW = 4
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name


class LaneMarking(object):
    """Used to represent a lane marking.

    Attributes:
        marking_color (:py:class:`.LaneMarkingColor`): The color of the lane
            marking
        marking_type (:py:class:`.LaneMarkingType`): The type of the lane
            marking.
        lane_change (:py:class:`.LaneChange`): The type that defines the
            permission to either turn left, right, both or none.
    """
    def __init__(self, marking_color, marking_type, lane_change):
        self.marking_color = LaneMarkingColor(marking_color)
        self.marking_type = LaneMarkingType(marking_type)
        self.lane_change = LaneChange(lane_change)

    @classmethod
    def from_simulator_lane_marking(cls, lane_marking):
        """Creates a pylot LaneMarking from a simulator lane marking.

        Args:
            lane_marking: An instance of a simulator lane marking.

        Returns:
            :py:class:`.LaneMarking`: A pylot lane-marking.
        """
        return cls(lane_marking.color, lane_marking.type,
                   lane_marking.lane_change)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "LaneMarking(color: {}, type: {}, change: {})".format(
            self.marking_color, self.marking_type, self.lane_change)


class Obstacle(object):
    """Class used to store info about obstacles.

    This class provides helper functions to detect obstacles and provide
    bounding boxes for them.

    Args:
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~Transform`, optional): Transform of
            the obstacle in the world.

    Attributes:
        bounding_box (:py:class:`~pylot.utisl.BoundingBox2D`): Bounding box of
            the obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~Transform`): Transform of the
            obstacle.
    """
    def __init__(self,
                 bounding_box,
                 confidence: float,
                 label: str,
                 id: int = -1,
                 transform: Transform = None,
                 detailed_label: str = '',
                 bounding_box_2D: BoundingBox2D = None):
        self.bounding_box = bounding_box
        if isinstance(bounding_box, BoundingBox2D):
            self.bounding_box_2D = bounding_box
        else:
            self.bounding_box_2D = bounding_box_2D
        self.confidence = confidence
        self.label = label
        self.id = id
        self.transform = transform
        self.detailed_label = detailed_label
        if label == 'vehicle':
            self.segmentation_class = 10
        elif label == 'person':
            self.segmentation_class = 4
        else:
            self.segmentation_class = None
        # Thresholds to be used for detection of the obstacle.
        self.__segmentation_threshold = 0.20
        self.__depth_threshold = 5

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates an Obstacle from a simulator actor.

        Args:
            actor: The actor to initialize the obstacle with.

        Returns:
            :py:class:`.Obstacle`: An obstacle instance.
        """
        # from carla import Vehicle, Walker
        # if not isinstance(actor, (Vehicle, Walker)):
        #     raise ValueError("The actor should be of type Vehicle or "
        #                      "Walker to initialize the Obstacle class.")
        # We do not use everywhere from_simulator* methods in order to reduce
        # runtime.
        # Convert the transform provided by the simulation to the Pylot class.
        transform = Transform.from_simulator_transform(
            actor.get_transform())
        # Convert the bounding box from the simulation to the Pylot one.
        bounding_box = BoundingBox3D.from_simulator_bounding_box(
            actor.bounding_box)
        if isinstance(actor, Vehicle):
            label = 'vehicle'
        else:
            label = 'person'
        # Get the simulator actor from type_id (e.g. vehicle.ford.mustang).
        detailed_label = actor.type_id
        # TODO (Sukrit): Move from vehicles and people to separate classes
        # for bicycles, motorcycles, cars and persons.
        return cls(bounding_box, 1.0, label, actor.id, transform,
                   detailed_label)

    def as_mot16_str(self, timestamp):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        log_line = "{},{},{},{},{},{},{},{},{},{}\n".format(
            timestamp, self.id, self.bounding_box_2D.x_min,
            self.bounding_box_2D.y_min, self.bounding_box_2D.get_width(),
            self.bounding_box_2D.get_height(), 1.0, -1, -1, -1)
        return log_line

    def _distance(self, other_transform: Transform):
        """Computes the distance from the obstacle to the other transform.

        The distance provides an estimate of the depth returned by the depth
        camera sensor in the simulator. As a result, the distance is defined
        as the displacement of the obstacle along either the X or the Y axis.

        Args:
            other_transform (:py:class:`~Transform`): The other
                transform.

        Returns:
            :obj:`float`: The distance (in metres) of the obstacle from the
            transform.
        """
        import numpy as np
        if self.transform is None:
            raise ValueError('Obstacle {} does not have a transform'.format(
                self.id))
        # Get the location of the vehicle and the obstacle as numpy arrays.
        other_location = other_transform.location.as_numpy_array()
        obstacle_location = self.transform.location.as_numpy_array()

        # Calculate the vector from the vehicle to the obstacle.
        # Scale it by the forward vector, and calculate the norm.
        relative_vector = other_location - obstacle_location
        distance = np.linalg.norm(
            relative_vector * other_transform.forward_vector.as_numpy_array())
        return distance

    def draw_on_frame(self,
                      frame,
                      bbox_color_map,
                      ego_transform: Transform = None,
                      text: str = None):
        """Annotate the image with the bounding box of the obstacle."""
        if text is None:
            text = '{}, {:.1f}'.format(self.label, self.confidence)
            if self.id != -1:
                text += ', id:{}'.format(self.id)
            if ego_transform is not None and self.transform is not None:
                text += ', {:.1f}m'.format(
                    ego_transform.location.distance(self.transform.location))
        if self.label in bbox_color_map:
            color = bbox_color_map[self.label]
        else:
            color = [255, 255, 255]
        # Show bounding box.
        if self.bounding_box_2D:
            # Draw the 2D bounding box if available.
            frame.draw_box(self.bounding_box_2D.get_min_point(),
                           self.bounding_box_2D.get_max_point(), color)
            frame.draw_text(self.bounding_box_2D.get_min_point(), text, color)
        elif isinstance(self.bounding_box, BoundingBox3D):
            if self.bounding_box.corners is None:
                raise ValueError(
                    'Obstacle {} does not have bbox corners'.format(self.id))
            corners = self.bounding_box.to_camera_view(
                None, frame.camera_setup.get_extrinsic_matrix(),
                frame.camera_setup.get_intrinsic_matrix())
            frame.draw_3d_box(corners, color)
        else:
            raise ValueError('Obstacle {} does not have bounding box'.format(
                self.id))

    def draw_trajectory_on_frame(self,
                                 trajectory,
                                 frame,
                                 point_color,
                                 draw_label: bool = False):
        # Intrinsic and extrinsic matrix of the top down camera.
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        if isinstance(self.bounding_box, BoundingBox3D):
            # Draw bounding boxes.
            start_location = self.bounding_box.transform.location - \
                self.bounding_box.extent
            end_location = self.bounding_box.transform.location + \
                self.bounding_box.extent
            for transform in trajectory:
                [start_transform,
                 end_transform] = transform.transform_locations(
                     [start_location, end_location])
                start_point = start_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                end_point = end_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(start_point) or frame.in_frame(end_point):
                    frame.draw_box(start_point, end_point, point_color)
        else:
            # Draw points.
            for transform in trajectory:
                screen_point = transform.location.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(screen_point):
                    # Draw trajectory on frame.
                    frame.draw_point(screen_point, point_color)
        if draw_label and len(trajectory) > 0:
            text = '{}, {}'.format(self.label, self.id)
            screen_point = trajectory[-1].location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_text(screen_point, text, point_color)

    def get_bounding_box_corners(self,
                                 obstacle_transform,
                                 obstacle_radius=None):
        """Gets the corners of the obstacle's bounding box.
        Note:
            The bounding box is applied on the given obstacle transfom, and not
            on the default obstacle transform.
        """
        # Use 3d bounding boxes if available, otherwise use default
        if isinstance(self.bounding_box, BoundingBox3D):
            start_location = (self.bounding_box.transform.location -
                              self.bounding_box.extent)
            end_location = (self.bounding_box.transform.location +
                            self.bounding_box.extent)
            [start_location,
             end_location] = obstacle_transform.transform_locations(
                 [start_location, end_location])
        else:
            obstacle_radius_loc = Location(obstacle_radius,
                                                       obstacle_radius)
            start_location = obstacle_transform.location - obstacle_radius_loc
            end_location = obstacle_transform.location + obstacle_radius_loc
        return [
            min(start_location.x, end_location.x),
            min(start_location.y, end_location.y),
            max(start_location.x, end_location.x),
            max(start_location.y, end_location.y)
        ]

    def get_in_log_format(self):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        min_point = self.bounding_box_2D.get_min_point()
        max_point = self.bounding_box_2D.get_max_point()
        return (self.label, self.detailed_label, self.id,
                ((min_point.x, min_point.y), (max_point.x, max_point.y)))

    def is_animal(self):
        return self.label in [
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe'
        ]

    def is_person(self):
        return self.label == 'person'

    def is_speed_limit(self):
        return self.label in [
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        ]

    def is_stop_sign(self):
        return self.label == 'stop sign' or self.label == 'stop marking'

    def is_traffic_light(self):
        return self.label in [
            'red traffic light', 'yellow traffic light', 'green traffic light',
            'off traffic light'
        ]

    def is_vehicle(self):
        # Might want to include train.
        return self.label in VEHICLE_LABELS

    def populate_bounding_box_2D(self, depth_frame, segmented_frame):
        """Populates the 2D bounding box for the obstacle.

        Heuristically uses the depth frame and segmentation frame to figure out
        if the obstacle is in view of the camera or not.

        Args:
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                Depth frame used to compare the depth to the distance of the
                obstacle from the sensor.
            segmented_frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):  # noqa: E501
                Segmented frame used to refine the conversions.

        Returns:
            :py:class:`~pylot.utisl.BoundingBox2D`: An instance representing a
            rectangle over the obstacle if the obstacle is deemed to be
            visible, None otherwise.
        """
        if self.bounding_box_2D:
            return self.bounding_box_2D
        # Convert the bounding box of the obstacle to the camera coordinates.
        bb_coordinates = self.bounding_box.to_camera_view(
            self.transform, depth_frame.camera_setup.get_extrinsic_matrix(),
            depth_frame.camera_setup.get_intrinsic_matrix())

        # Threshold the bounding box to be within the camera view.
        bbox_2d = get_bounding_box_in_camera_view(
            bb_coordinates, depth_frame.camera_setup.width,
            depth_frame.camera_setup.height)
        if not bbox_2d:
            return None
        # Crop the segmented and depth image to the given bounding box.
        cropped_image = segmented_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]
        cropped_depth = depth_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]

        # If the size of the bounding box is greater than 0, ensure that the
        # bounding box contains more than a threshold of pixels corresponding
        # to the required segmentation class.
        if cropped_image.size > 0:
            masked_image = np.zeros_like(cropped_image)
            masked_image[np.where(
                cropped_image == self.segmentation_class)] = 1
            seg_threshold = self.__segmentation_threshold * masked_image.size
            if np.sum(masked_image) >= seg_threshold:
                # The bounding box contains the required number of pixels that
                # belong to the required class. Ensure that the depth of the
                # obstacle is the depth in the image.
                masked_depth = cropped_depth[np.where(masked_image == 1)]
                mean_depth = np.mean(masked_depth) * 1000
                depth = self._distance(
                    depth_frame.camera_setup.get_transform())
                if abs(depth - mean_depth) <= self.__depth_threshold:
                    self.bounding_box_2D = bbox_2d
                    return bbox_2d
        return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        obstacle = 'Obstacle(id: {}, label: {}, confidence: {}, '\
            'bbox: {})'.format(self.id, self.label, self.confidence,
                               self.bounding_box)
        if self.transform:
            return obstacle + ' at ' + str(self.transform)
        else:
            return obstacle


class Waypoints(object):
    """Stores waypoints and provides methods to wrangle them."""
    def __init__(self, waypoints, target_speeds=None, road_options=None):
        self.waypoints = waypoints
        if target_speeds is None:
            # Set target speed to 0 if it is not specified.
            self.target_speeds = deque([0 for _ in range(len(waypoints))])
        else:
            self.target_speeds = target_speeds
        self.road_options = road_options

    @classmethod
    def read_from_csv_file(cls, csv_file_name: str, target_speed):
        """Reads waypoints from a csv file."""
        csv_file = open(csv_file_name)
        csv_reader = csv.reader(csv_file)
        waypoints = []
        for row in csv_reader:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            waypoint = Transform(Location(x, y, z),Rotation(0, 0, 0))
            waypoints.append(waypoint)
        target_speeds = deque([target_speed for _ in range(len(waypoints))])
        return cls(deque(waypoints), target_speeds)

    def apply_speed_factor(self, speed_factor: float):
        if self.target_speeds:
            self.target_speeds = [
                speed_factor * ts for ts in self.target_speeds
            ]

    def as_numpy_array_2D(self):
        """Returns the waypoints as a numpy array of lists of x and y."""
        wx = []
        wy = []
        for wp in self.waypoints:
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        return np.array([wx, wy])

    def closest_waypoint(self, location: Location):
        """Finds the closest waypoint to the location."""
        min_dist = np.infty
        min_index = 0
        for index, waypoint in enumerate(self.waypoints):
            dist = waypoint.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                min_index = index
        return min_index

    def is_empty(self) -> bool:
        return len(self.waypoints) == 0

    def remove_waypoint_if_close(self,
                                 location: Location,
                                 distance: float = 5) -> bool:
        """Removes the first waypoint if it is less than distance m away."""
        if self.waypoints is None or len(self.waypoints) == 0:
            return False
        if location.distance(self.waypoints[0].location) < distance:
            self.waypoints.popleft()
            if self.target_speeds:
                self.target_speeds.popleft()
            if self.road_options:
                self.road_options.popleft()
            return True
        return False

    def remove_completed(self,
                         location: Location,
                         ego_transform: Transform = None):
        """Removes waypoints that the ego vehicle has already completed.

        The method first finds the closest waypoint to the location,
        and then removes all waypoints that are before the closest waypoint.

        Args:
            location (:py:class:`Location`): The location
                of the ego-vehicle.
        """
        min_index = self.closest_waypoint(location)
        # Remove waypoints that are before the closest waypoint. The ego
        # vehicle already completed them.
        while min_index > 0:
            self.waypoints.popleft()
            if self.target_speeds:
                self.target_speeds.popleft()
            if self.road_options:
                self.road_options.popleft()
            min_index -= 1

        if self.waypoints is None or len(self.waypoints) == 0:
            if ego_transform is not None:
                self.waypoints = deque([ego_transform])
                self.target_speeds = deque([0])
                self.road_options = deque([RoadOption.LANE_FOLLOW])
            else:
                raise ValueError('No more waypoints to follow')

    def recompute_waypoints(self, hd_map, ego_location, goal_location):
        self.waypoints = hd_map.compute_waypoints(ego_location, goal_location)
        self.target_speeds = deque([0 for _ in range(len(self.waypoints))])

    def _get_index(self, transform: Transform,
                   min_distance: float) -> int:
        min_index = -1
        for index, wp in enumerate(self.waypoints):
            distance = wp.location.distance(transform.location)
            if distance >= min_distance:
                min_index = index
                break
        if min_index == -1:
            min_index = len(self.waypoints) - 1
        if min_index < 0:
            raise ValueError('No more waypoints')
        return min_index

    def get_angle(self, transform: Transform,
                  min_distance: float) -> float:
        """Returns the angle between the transform and the first waypoint that
        is at least min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        angle, _ = transform.get_angle_and_magnitude(
            self.waypoints[wp_index].location)
        return angle

    def get_vector(self, transform: Transform,
                   min_distance: float):
        """Returns the vector between the transform and the first waypoint that
        is at least min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        return self.waypoints[wp_index].location.as_vector_2D() - \
            transform.location.as_vector_2D()

    def get_target_speed(self, transform: Transform,
                         min_distance: float) -> float:
        """Gets the target speed at the first waypoint that is at least
        min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        return self.target_speeds[wp_index]

    def slice_waypoints(self,
                        start_index: int,
                        end_index: int,
                        target_speed: float = None):
        head_wps = deque(
            itertools.islice(self.waypoints, start_index, end_index))
        if target_speed is not None:
            # Use the specified target speed.
            head_target_speeds = deque(
                [target_speed for _ in range(len(head_wps))])
        else:
            # Otherwise use the already existing target speeds.
            head_target_speeds = deque(
                itertools.islice(self.target_speeds, start_index, end_index))
        if self.road_options is not None:
            head_road_options = deque(
                itertools.islice(self.road_options, start_index, end_index))
        else:
            head_road_options = None
        return Waypoints(head_wps, head_target_speeds, head_road_options)

    def draw_on_frame(self, bgr_frame, inverse_transform=None):
        """Draw waypoints on a frame.

        Args:
            bgr_frame: Frame on which to draw the waypoints.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = bgr_frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = bgr_frame.camera_setup.get_intrinsic_matrix()
        for index, wp in enumerate(self.waypoints):
            if inverse_transform:
                wp = inverse_transform * wp
            pixel_location = wp.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            bgr_frame.draw_point(pixel_location, [255, 255, 255])
            waypoint_txt = ''
            if self.target_speeds:
                waypoint_txt = '{:.1f}m/s'.format(self.target_speeds[index])
            # if self.road_options:
            #     waypoint_txt = '{} {}'.format(waypoint_txt,
            #                                   self.road_options[index])
            if waypoint_txt != '':
                bgr_frame.draw_text(pixel_location, waypoint_txt,
                                    [255, 255, 255])

    def draw_on_world(self, world):
        """Draw waypoints on the simulator world."""
        for index, wp in enumerate(self.waypoints):
            # Adds 0.5 to z to ensure that the point is above the road surface.
            loc = (wp.location +
                   Location(0, 0, 0.5)).as_simulator_location()
            world.debug.draw_point(loc, size=0.1, life_time=0.1)
            # if self.road_options and index < len(self.road_options):
            #     world.debug.draw_string(loc,
            #                             str(self.road_options[index]),
            #                             life_time=DEFAULT_VIS_TIME)


class ObstacleTrajectory(object):
    """Used to store the trajectory of an obstacle."""
    def __init__(self, obstacle: Obstacle, trajectory):
        self.obstacle = obstacle
        self.trajectory = trajectory

    def draw_on_frame(self,
                      frame,
                      bbox_color_map,
                      ego_transform: Transform = None):
        """Draws the tracked obstacle as a 2D bounding box."""
        self.obstacle.draw_on_frame(frame, bbox_color_map, ego_transform)

    def draw_trajectory_on_frame(self, frame, draw_label: bool = False):
        """Draws the trajectory on a bird's eye view frame."""
        if self.obstacle.is_person():
            color = [255, 0, 0]
        elif self.obstacle.is_vehicle():
            color = [128, 128, 0]
        else:
            color = [255, 255, 0]
        self.obstacle.draw_trajectory_on_frame(self.trajectory, frame, color,
                                               draw_label)

    def estimate_obstacle_orientation(self):
        """Uses the obstacle's past trajectory to estimate its angle from the
           positive x-axis (assumes trajectory points are in the ego-vehicle's
           coordinate frame)."""
        other_idx = len(self.trajectory) - 2
        # TODO: Setting a default yaw is dangerous. Find some way to estimate
        # the orientation of a stationary object (e.g. 3D object detection).
        yaw = 0.0  # Default orientation for stationary objects.
        current_loc = self.trajectory[-1].location.as_vector_2D()
        while other_idx >= 0:
            past_ref_loc = self.trajectory[other_idx].location.as_vector_2D()
            vec = current_loc - past_ref_loc
            displacement = current_loc.l2_distance(past_ref_loc)
            if displacement > 0.001:
                # Angle between displacement vector and the x-axis, i.e.
                # the (1,0) vector.
                yaw = vec.get_angle(Vector2D(1, 0))
                break
            else:
                other_idx -= 1
        return math.degrees(yaw)

    def get_last_n_transforms(self, n: int):
        """Returns the last n steps of the trajectory. If we have not seen
        enough past locations of the obstacle, pad the trajectory with the
        appropriate number of copies of the earliest location."""
        num_past_locations = len(self.trajectory)
        if num_past_locations < n:
            initial_copies = [self.trajectory[0]] * (n - num_past_locations)
            last_n_steps = initial_copies + self.trajectory
        else:
            last_n_steps = self.trajectory[-n:]
        return last_n_steps

    def to_world_coordinates(self, ego_transform: Transform):
        """Transforms the trajectory into world coordinates."""
        cur_trajectory = []
        for past_transform in self.trajectory:
            cur_trajectory.append(ego_transform * past_transform)
        self.trajectory = cur_trajectory

    @property
    def id(self):
        return self.obstacle.id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Obstacle {}, trajectory {}'.format(self.obstacle,
                                                   self.trajectory)


class ObstaclePrediction(object):
    """Class storing info about an obstacle prediction."""
    def __init__(self, obstacle_trajectory: ObstacleTrajectory,
                 transform: Transform, probability: float,
                 predicted_trajectory):
        # Trajectory in ego frame of coordinates.
        self.obstacle_trajectory = obstacle_trajectory
        # The transform is in world coordinates.
        self.transform = transform
        self.probability = probability
        # Predicted trajectory in ego frame of coordinates.
        self.predicted_trajectory = predicted_trajectory

    def draw_trajectory_on_frame(self, frame):
        """Draws the past and predicted trajectory on a bird's eye frame."""
        if self.is_person():
            color = [0, 0, 255]
        elif self.is_vehicle():
            color = [0, 255, 0]
        else:
            color = [255, 0, 0]
        self.obstacle_trajectory.obstacle.draw_trajectory_on_frame(
            self.predicted_trajectory, frame, color)
        self.obstacle_trajectory.draw_trajectory_on_frame(frame, True)

    def to_world_coordinates(self, ego_transform: Transform):
        """Transforms the trajectory and prediction into world coordinates."""
        self.obstacle_trajectory.to_world_coordinates(ego_transform)
        cur_trajectory = []
        for future_transform in self.predicted_trajectory:
            cur_trajectory.append(ego_transform * future_transform)
        self.predicted_trajectory = cur_trajectory

    @property
    def id(self):
        return self.obstacle_trajectory.obstacle.id

    @property
    def label(self):
        return self.obstacle_trajectory.obstacle.label

    def is_animal(self):
        return self.obstacle_trajectory.obstacle.is_animal()

    def is_person(self):
        return self.obstacle_trajectory.obstacle.is_person()

    def is_speed_limit(self):
        return self.obstacle_trajectory.obstacle.is_speed_limit()

    def is_stop_sign(self):
        return self.obstacle_trajectory.obstacle.is_stop_sign()

    def is_traffic_light(self):
        return self.obstacle_trajectory.obstacle.is_traffic_light()

    def is_vehicle(self):
        return self.obstacle_trajectory.obstacle.is_vehicle()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ('Prediction for obstacle {}, probability {}, '
                'predicted trajectory {}'.format(
                    self.obstacle_trajectory.obstacle, self.probability,
                    self.predicted_trajectory))


class LidarSetup(object):
    """LidarSetup stores information about an instance of LIDAR mounted on
    the vehicle."""
    def __init__(self,
                 name: str,
                 lidar_type: str,
                 transform: Transform,
                 range: float = 5000,
                 rotation_frequency: float = 20,
                 channels: int = 32,
                 upper_fov: float = 15,
                 lower_fov: float = -30,
                 points_per_second: int = 500000,
                 legacy: bool = True):
        # Ensure that the name is a string.
        assert isinstance(name, str), "The name should be of type 'str'"
        self.name = name

        self.legacy = legacy

        # Ensure that the type of LIDAR is currently supported.
        err_msg = "LIDAR must be of type 'sensor.lidar.ray_cast' or 'velodyne'"
        assert lidar_type == 'sensor.lidar.ray_cast' or \
            lidar_type == 'velodyne', err_msg
        self.lidar_type = lidar_type

        # Ensure that the transform is of the correct type.
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type Transform"
        self.transform = transform

        # Try to coerce the range to float and throw an error if not possible.
        range = float(range) if isinstance(range, int) else range
        assert isinstance(range, float), "The range should be of type 'float'"
        self.range = range

        # Try to coerce the rotation_frequency to float and throw an error,
        # if not possible.
        rotation_frequency = float(rotation_frequency) if \
            isinstance(rotation_frequency, int) else rotation_frequency
        assert isinstance(rotation_frequency, float), "The rotation_frequency"
        " should be of type 'float'"
        self.rotation_frequency = rotation_frequency

        # Ensure that the channels are of correct type.
        assert isinstance(channels, int), "The channels should be of type "
        "'int'"
        self.channels = channels

        # Try to coerce the upper_fov and lower_fov to float, and throw an
        # error if not possible.
        upper_fov = float(upper_fov) if \
            isinstance(upper_fov, int) else upper_fov
        lower_fov = float(lower_fov) if \
            isinstance(lower_fov, int) else lower_fov
        assert isinstance(upper_fov, float) and isinstance(lower_fov, float),\
            "The upper_fov and lower_fov should be of type 'float'"
        self.upper_fov, self.lower_fov = upper_fov, lower_fov

        # Ensure that the points_per_second is of type 'int'
        assert isinstance(points_per_second, int), "The points_per_second"
        " should be of type 'int'"
        self.points_per_second = points_per_second
        self._unreal_transform = LidarSetup.__create_unreal_transform(
            self.transform, self.legacy)

    @staticmethod
    def __create_unreal_transform(transform: Transform,
                                  legacy: bool) -> Transform:
        """Converts a Transform from the LIDAR coordinate space to the
        Unreal Engine coordinate space.

        The legacy LIDAR space is defined as:
            +x to the right, +y is out of the screen, +z is down.

        The Unreal Engine coordinate space is defined as:
            +x into the screen, +y to the right, +z is up."""
        if legacy:
            to_camera_transform = Transform(matrix=np.array(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
            return transform * to_camera_transform
        else:
            return transform

    def get_name(self) -> str:
        """Get the name of the LIDAR instance.

        Returns:
            str: The name of the LIDAR instance."""
        return self.name

    def get_transform(self) -> Transform:
        """Get the transform of the LIDAR with respect to the vehicle to
        which it is attached."""
        return self.transform

    def set_transform(self, transform: Transform):
        """Set the transform of the LIDAR with respect to the vehicle to which
        it is attached."""
        self.transform = transform
        self._unreal_transform = LidarSetup.__create_unreal_transform(
            self.transform, self.legacy)

    def set_legacy(self, legacy: bool):
        self.legacy = legacy
        self._unreal_transform = LidarSetup.__create_unreal_transform(
            self.transform, self.legacy)

    def get_unreal_transform(self) -> Transform:
        """Get the transform of the LIDAR with respect to the vehicle in the
        Unreal Engine coordinate space."""
        return self._unreal_transform

    def get_range_in_meters(self) -> float:
        """Get the range of the LIDAR in metres.

        Returns:
            float: The range of the LIDAR in metres.
        """
        return self.range / 100

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'LidarSetup(name: {}, type: {}, transform: {}, range: {}, '\
            'rotation freq: {}, channels: {}, upper_fov: {}, lower_fov: {}, '\
            'points_per_second: {}'.format(
                self.name, self.lidar_type, self.transform, self.range,
                self.rotation_frequency, self.channels, self.upper_fov,
                self.lower_fov, self.points_per_second)


class CameraSetup(object):
    """CameraSetup stores information about an instance of the camera
    mounted on the vehicle."""
    def __init__(self,
                 name: str,
                 camera_type: str,
                 width: int,
                 height: int,
                 transform: Transform,
                 fov: float = 90):
        assert isinstance(name, str), "The name should be of type 'str'"
        self.name = name

        # Ensure that the camera type is one of the three that we support.
        assert camera_type in (
            'sensor.camera.rgb', 'sensor.camera.depth',
            'sensor.camera.semantic_segmentation'), "The camera_type " \
            "should belong to ('sensor.camera.rgb', 'sensor.camera.depth', " \
            "'sensor.camera.semantic_segmentation')"
        self.camera_type = camera_type

        # The width of the image produced by the camera should be > 1.
        assert width > 1, "Valid camera setup should have width > 1"
        assert isinstance(width, int) and isinstance(height, int), "The width"
        " and height should be of type 'int'"
        self.width, self.height = width, height

        # Ensure that the transform is of the type pylot.Transform.
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type Transform"
        self.transform = transform

        # Ensure that the field-of-view is a float.
        fov = float(fov) if isinstance(fov, int) else fov
        assert isinstance(fov, float), "The fov should be of type 'float'"
        self.fov = fov

        # Generate the intrinsic and extrinsic matrices.
        self._intrinsic_mat = CameraSetup.__create_intrinsic_matrix(
            self.width, self.height, self.fov)
        self._unreal_transform = CameraSetup.__create_unreal_transform(
            self.transform)

    @staticmethod
    def __create_intrinsic_matrix(width: int, height: int, fov: float):
        """Creates the intrinsic matrix for a camera with the given
        parameters.

        Args:
            width (int): The width of the image returned by the camera.
            height (int): The height of the image returned by the camera.
            fov (float): The field-of-view of the camera.

        Returns:
            :py:class:`numpy.ndarray`: A 3x3 intrinsic matrix of the camera.
        """
        import numpy as np
        k = np.identity(3)
        # We use width - 1 and height - 1 to find the center column and row
        # of the image, because the images are indexed from 0.

        # Center column of the image.
        k[0, 2] = (width - 1) / 2.0
        # Center row of the image.
        k[1, 2] = (height - 1) / 2.0
        # Focal length.
        k[0, 0] = k[1, 1] = (width - 1) / (2.0 * np.tan(fov * np.pi / 360.0))
        return k

    @staticmethod
    def __create_unreal_transform(transform):
        """Converts a Transform from the camera coordinate space to the
        Unreal coordinate space.

        The camera space is defined as:
            +x to right, +y to down, +z into the screen.

        The unreal coordinate space is defined as:
            +x into the screen, +y to right, +z to up.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): The transform to
                convert to Unreal coordinate space.

        Returns:
            :py:class:`~pylot.utils.Transform`: The given transform after
                transforming to the Unreal coordinate space.
        """
        import numpy as np
        to_unreal_transform = Transform(matrix=np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        return transform * to_unreal_transform

    def get_intrinsic_matrix(self):
        """Get the intrinsic matrix of the camera denoted by the CameraSetup.

        Returns:
            :py:class:`numpy.ndarray`: The 3x3 intrinsic matrix of the camera.
        """
        return self._intrinsic_mat

    def get_focal_length(self):
        return self._intrinsic_mat[0, 0]

    def get_extrinsic_matrix(self):
        """Get the extrinsic matrix of the camera denoted by the transform
        of the camera with respect to the vehicle to which it is attached.

        Returns:
            :py:class:`numpy.ndarray`: The 4x4 extrinsic matrix of the camera.
        """
        return self._unreal_transform.matrix

    def get_name(self) -> str:
        """ Get the name of the camera instance.

        Returns:
            str: The name of the camera instance.
        """
        return self.name

    def get_unreal_transform(self) -> Transform:
        """Get the transform of the camera with respect to the vehicle in
        the Unreal Engine coordinate space.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the camera
            in the Unreal Engine coordinate space.
        """
        return self._unreal_transform

    def get_transform(self) -> Transform:
        """Get the transform of the camera with respect to the vehicle to
        which it is attached.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the camera
            with respect to the vehicle to which it is attached.
        """
        return self.transform

    def set_transform(self, transform: Transform):
        """Set the transform of the camera with respect to the vehicle to
        which it is attached.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): The new transform
                of the camera with respect to the vehicle to which it is
                attached.
        """
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type Transform"
        self.transform = transform
        self._unreal_transform = CameraSetup.__create_unreal_transform(
            self.transform)

    def set_resolution(self, width: int, height: int):
        self.width = width
        self.height = height
        self._intrinsic_mat = CameraSetup.__create_intrinsic_matrix(
            self.width, self.height, self.fov)

    def get_fov(self) -> float:
        """Get the field of view of the camera.

        Returns:
            float: The field of view of the given camera.
        """
        return self.fov

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraSetup(name: {}, type: {}, width: {}, height: {}, '\
            'transform: {}, fov: {})'.format(
                self.name, self.camera_type, self.width, self.height,
                self.transform, self.fov)


class RGBCameraSetup(CameraSetup):
    """A CameraSetup that denotes an RGB camera.

    Args:
        name (str): The name of the camera instance.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance
            with respect to the vehicle.
        fov (float): The field-of-view of the camera.

    Attributes:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.
    """
    def __init__(self,
                 name: str,
                 width: int,
                 height: int,
                 transform: Transform,
                 fov: float = 90):
        super(RGBCameraSetup, self).__init__(name, 'sensor.camera.rgb', width,
                                             height, transform, fov)


class DepthCameraSetup(CameraSetup):
    """A CameraSetup that denotes a Depth camera.

    Args:
        name (str): The name of the camera instance.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.

    Attributes:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.
    """
    def __init__(self,
                 name: str,
                 width: int,
                 height: int,
                 transform: Transform,
                 fov: float = 90):
        super(DepthCameraSetup, self).__init__(name, 'sensor.camera.depth',
                                               width, height, transform, fov)


class TrafficLightColor(Enum):
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

class TrafficLight(Obstacle):
    """Class used to store info about traffic lights.

    Args:
        confidence (:obj:`float`): The confidence of the detection.
        state (:py:class:`.TrafficLightColor`): The state of the traffic light.
        id (:obj:`int`, optional): The identifier of the traffic light.
        transform (:py:class:`~pylot.utils.Transform`, optional): Transform of
            the traffic light.
        trigger_volume_extent (:py:class:`pylot.utils.Vector3D`, optional): The
            extent of the trigger volume of the light.
        bounding_box (:py:class:`.BoundingBox2D`, optional): The bounding box
            of the traffic light in camera view.

    Attributes:
        confidence (:obj:`float`): The confidence of the detection.
        state (:py:class:`.TrafficLightColor`): The state of the traffic light.
        id (:obj:`int`): The identifier of the traffic light.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            traffic light.
        trigger_volume_extent (:py:class:`pylot.utils.Vector3D`): The extent
            of the trigger volume of the light.
        bounding_box (:py:class:`.BoundingBox2D`, optional): The bounding box
            of the traffic light in camera view.
    """
    def __init__(self,
                 confidence: float,
                 state: TrafficLightColor,
                 id: int = -1,
                 transform: Transform = None,
                 trigger_volume_extent: Vector3D = None,
                 bounding_box: BoundingBox2D = None):
        super(TrafficLight, self).__init__(bounding_box, confidence,
                                           state.get_label(), id, transform)
        self.state = state
        self.trigger_volume_extent = trigger_volume_extent

    @classmethod
    def from_simulator_actor(cls, traffic_light):
        """ Creates a TrafficLight from a simulator traffic light actor.

        Args:
            traffic_light: A simulator traffic light actor.

        Returns:
            :py:class:`.TrafficLight`: A traffic light.
        """
        from carla import TrafficLight, TrafficLightState
        if not isinstance(traffic_light, TrafficLight):
            raise ValueError('The traffic light must be a TrafficLight')
        # Retrieve the Transform of the TrafficLight.
        transform = Transform.from_simulator_transform(
            traffic_light.get_transform())
        # Retrieve the Trigger Volume of the TrafficLight.
        trigger_volume_extent = Vector3D(
            traffic_light.trigger_volume.extent.x,
            traffic_light.trigger_volume.extent.y,
            traffic_light.trigger_volume.extent.z)
        traffic_light_state = traffic_light.get_state()
        state = TrafficLightColor.OFF
        if traffic_light_state == TrafficLightState.Red:
            state = TrafficLightColor.RED
        elif traffic_light_state == TrafficLightState.Yellow:
            state = TrafficLightColor.YELLOW
        elif traffic_light_state == TrafficLightState.Green:
            state = TrafficLightColor.GREEN
        return cls(1.0, state, traffic_light.id, transform,
                   trigger_volume_extent)

    def draw_on_bird_eye_frame(self, frame):
        # Intrinsic and extrinsic matrix of the top down camera.
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        point = self.transform.location.to_camera_view(extrinsic_matrix,
                                                       intrinsic_matrix)
        frame.draw_point(point, self.state.get_color(), r=10)
        frame.draw_text(point, self.state.get_label(), self.state.get_color())

    def is_traffic_light_visible(self,
                                 camera_transform: Transform,
                                 town_name: str = None,
                                 distance_threshold: int = 70):
        """Checks if the traffic light is visible from the camera transform.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): Transform of the
                camera in the world frame of reference.
            distance_threshold (:obj:`int`): Maximum distance to the camera
                (in m).

        Returns:
            bool: True if the traffic light is visible from the camera
            transform.
        """
        # We dot product the forward vectors (i.e., orientation).
        # Note: we have to rotate the traffic light forward vector
        # so that it's pointing out from the traffic light in the
        # opposite direction in which the ligth is beamed.
        prod = np.dot([
            self.transform.forward_vector.y, -self.transform.forward_vector.x,
            self.transform.forward_vector.z
        ], [
            camera_transform.forward_vector.x,
            camera_transform.forward_vector.y,
            camera_transform.forward_vector.z
        ])
        if self.transform.location.distance(
                camera_transform.location) > distance_threshold:
            return prod > 0.4

        if town_name is None:
            return prod > -0.80
        else:
            if town_name == 'Town01' or town_name == 'Town02':
                return prod > 0.3
        return prod > -0.80

    def get_all_detected_traffic_light_boxes(self, town_name: str, depth_frame,
                                             segmented_image):
        """ Returns traffic lights for all boxes of a simulator traffic light.

        Note:
            All the traffic lights returned will have the same id and
            transform.

        Args:
            town_name (:obj:`str`): Name of the town in which the traffic light
                is.
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                 Depth frame.
            segmented_image: A segmented image np array used to refine the
                 bounding boxes.

        Returns:
            list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`):
            Detected traffic lights, one for each traffic light box.
        """
        traffic_lights = []
        bboxes = self._get_bboxes(town_name)
        # Convert the returned bounding boxes to 2D and check if the
        # light is occluded. If not, add it to the traffic lights list.
        for bbox in bboxes:
            bounding_box = [
                loc.to_camera_view(
                    depth_frame.camera_setup.get_extrinsic_matrix(),
                    depth_frame.camera_setup.get_intrinsic_matrix())
                for loc in bbox
            ]
            bbox_2d = get_bounding_box_in_camera_view(
                bounding_box, depth_frame.camera_setup.width,
                depth_frame.camera_setup.height)
            if not bbox_2d:
                continue

            # Crop the segmented and depth image to the given bounding box.
            cropped_image = segmented_image[bbox_2d.y_min:bbox_2d.y_max,
                                            bbox_2d.x_min:bbox_2d.x_max]
            cropped_depth = depth_frame.frame[bbox_2d.y_min:bbox_2d.y_max,
                                              bbox_2d.x_min:bbox_2d.x_max]

            if cropped_image.size > 0:
                masked_image = np.zeros_like(cropped_image)
                masked_image[np.where(
                    np.logical_or(cropped_image == 12,
                                  cropped_image == 18))] = 1
                if np.sum(masked_image) >= 0.20 * masked_image.size:
                    masked_depth = cropped_depth[np.where(masked_image == 1)]
                    mean_depth = np.mean(masked_depth) * 1000
                    if abs(mean_depth -
                           bounding_box[0].z) <= 2 and mean_depth < 150:
                        traffic_lights.append(
                            TrafficLight(1.0, self.state, self.id,
                                         self.transform,
                                         self.trigger_volume_extent, bbox_2d))
        return traffic_lights

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'TrafficLight(confidence: {}, state: {}, id: {}, ' \
            'transform: {}, trigger_volume_extent: {}, bbox: {})'.format(
                self.confidence, self.state, self.id, self.transform,
                self.trigger_volume_extent, self.bounding_box)

    def _relative_to_traffic_light(self, points):
        """Transforms the bounding box specified in the points relative to the
        light.

        Args:
            points: An array of length 4 representing the 4 points of the
                rectangle.
        """
        def rotate(yaw, location):
            """ Rotate a given 3D vector around the Z-axis. """
            rotation_matrix = np.identity(3)
            rotation_matrix[0, 0] = np.cos(yaw)
            rotation_matrix[0, 1] = -np.sin(yaw)
            rotation_matrix[1, 0] = np.sin(yaw)
            rotation_matrix[1, 1] = np.cos(yaw)
            location_vector = np.array([[location.x], [location.y],
                                        [location.z]])
            transformed = np.dot(rotation_matrix, location_vector)
            return Location(x=transformed[0, 0],
                                        y=transformed[1, 0],
                                        z=transformed[2, 0])

        transformed_points = [
            rotate(np.radians(self.transform.rotation.yaw), point)
            for point in points
        ]
        base_relative_points = [
            self.transform.location + point for point in transformed_points
        ]
        return base_relative_points

    def _get_bboxes(self, town_name: str):
        if town_name == 'Town01' or town_name == 'Town02':
            return self._get_bboxes_for_town1_or_2()
        elif town_name == 'Town03':
            return self._get_bboxes_for_town3()
        elif town_name == 'Town04':
            return self._get_bboxes_for_town4()
        elif town_name == 'Town05':
            return self._get_bboxes_for_town5()
        else:
            raise ValueError(
                'Could not find a town named {}'.format(town_name))

    def _get_bboxes_for_town1_or_2(self):
        points = [
            # Back Plane
            Location(x=-0.5, y=-0.1, z=2),
            Location(x=+0.1, y=-0.1, z=2),
            Location(x=+0.1, y=-0.1, z=3),
            Location(x=-0.5, y=-0.1, z=3),
            # Front Plane
            Location(x=-0.5, y=0.5, z=2),
            Location(x=+0.1, y=0.5, z=2),
            Location(x=+0.1, y=0.5, z=3),
            Location(x=-0.5, y=0.5, z=3),
        ]
        return [self._relative_to_traffic_light(points)]

    def _get_bboxes_for_town3(self):
        bboxes = []
        if (self.trigger_volume_extent.x > 2 or self.id in [
                66,
                67,
                68,
                71,
                72,
                73,
                75,
                81,
        ]):
            points = [
                # Back Plane
                Location(x=-5.2, y=-0.2, z=5.5),
                Location(x=-4.8, y=-0.2, z=5.5),
                Location(x=-4.8, y=-0.2, z=6.5),
                Location(x=-5.2, y=-0.2, z=6.5),
                # Front Plane
                Location(x=-5.2, y=0.4, z=5.5),
                Location(x=-4.8, y=0.4, z=5.5),
                Location(x=-4.8, y=0.4, z=6.5),
                Location(x=-5.2, y=0.4, z=6.5),
            ]
            bboxes.append(self._relative_to_traffic_light(points))
            right_points = [
                point + Location(x=-3.0) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(right_points))
            if self.id not in [51, 52, 53]:
                left_points = [
                    point + Location(x=-6.5) for point in points
                ]
                bboxes.append(self._relative_to_traffic_light(left_points))
        else:
            points = [
                # Back Plane
                Location(x=-0.5, y=-0.1, z=2),
                Location(x=+0.1, y=-0.1, z=2),
                Location(x=+0.1, y=-0.1, z=3),
                Location(x=-0.5, y=-0.1, z=3),
                # Front Plane
                Location(x=-0.5, y=0.5, z=2),
                Location(x=+0.1, y=0.5, z=2),
                Location(x=+0.1, y=0.5, z=3),
                Location(x=-0.5, y=0.5, z=3),
            ]
            bboxes.append(self._relative_to_traffic_light(points))

        return bboxes

    def _get_bboxes_for_town4(self):
        bboxes = []
        points = [
            # Back Plane
            Location(x=-5.2, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=6.5),
            Location(x=-5.2, y=-0.2, z=6.5),
            # Front Plane
            Location(x=-5.2, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=6.5),
            Location(x=-5.2, y=0.4, z=6.5),
        ]
        bboxes.append(self._relative_to_traffic_light(points))
        if self.trigger_volume_extent.x > 5:
            # This is a traffic light with 4 signs, we need to come up with
            # more bounding boxes.
            middle_points = [  # Light in the middle of the pole.
                # Back Plane
                Location(x=-0.5, y=-0.1, z=2.5),
                Location(x=+0.1, y=-0.1, z=2.5),
                Location(x=+0.1, y=-0.1, z=3.5),
                Location(x=-0.5, y=-0.1, z=3.5),
                # Front Plane
                Location(x=-0.5, y=0.5, z=2.5),
                Location(x=+0.1, y=0.5, z=2.5),
                Location(x=+0.1, y=0.5, z=3.5),
                Location(x=-0.5, y=0.5, z=3.5),
            ]
            right_points = [
                point + Location(x=-3.0) for point in points
            ]
            left_points = [
                point + Location(x=-5.5) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(middle_points))
            bboxes.append(self._relative_to_traffic_light(right_points))
            bboxes.append(self._relative_to_traffic_light(left_points))
        return bboxes

    def _get_bboxes_for_town5(self):
        bboxes = []
        points = [
            # Back Plane
            Location(x=-5.2, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=6.5),
            Location(x=-5.2, y=-0.2, z=6.5),
            # Front Plane
            Location(x=-5.2, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=6.5),
            Location(x=-5.2, y=0.4, z=6.5),
        ]
        # Town05 randomizes the identifiers for the traffic light at each
        # reload of the world. We cannot depend on static identifiers for
        # figuring out which lights only have a single traffic light.
        bboxes.append(self._relative_to_traffic_light(points))
        # There's a traffic light with extent.x < 2, which only has one box.
        if self.trigger_volume_extent.x >= 2:
            # This is a traffids light with 4 signs, we need to come up
            # with more bounding boxes.
            middle_points = [  # Light in the middle of the pole.
                # Back Plane
                Location(x=-0.4, y=-0.1, z=2.55),
                Location(x=+0.2, y=-0.1, z=2.55),
                Location(x=+0.2, y=-0.1, z=3.55),
                Location(x=-0.4, y=-0.1, z=3.55),
                # Front Plane
                Location(x=-0.4, y=0.5, z=2.55),
                Location(x=+0.2, y=0.5, z=2.55),
                Location(x=+0.2, y=0.5, z=3.55),
                Location(x=-0.5, y=0.5, z=3.55),
            ]
            right_points = [
                point + Location(x=-3.0) for point in points
            ]
            left_points = [
                point + Location(x=-5.5) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(middle_points))
            bboxes.append(self._relative_to_traffic_light(right_points))
            bboxes.append(self._relative_to_traffic_light(left_points))
        return bboxes


class SpeedLimitSign(Obstacle):
    def __init__(self,
                 speed_limit: float,
                 confidence: float,
                 bounding_box: BoundingBox2D = None,
                 id: int = -1,
                 transform: Transform = None):
        super(SpeedLimitSign, self).__init__(bounding_box, confidence,
                                             'speed limit', id, transform)
        self.speed_limit = speed_limit

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates a detected speed limit sign from a simulator actor.

        Args:
            actor: A simulator speed limit sign actor.

        Returns:
            :py:class:`.SpeedLimitSign`: A detected speed limit sign.
        """
        from carla import TrafficSign
        if not isinstance(actor, TrafficSign):
            raise ValueError('actor should be of type TrafficSign')
        transform = Transform.from_simulator_transform(actor.get_transform())
        speed_limit = int(actor.type_id.split('.')[-1])
        return cls(speed_limit, 1.0, id=actor.id, transform=transform)

    def get_in_log_format(self):
        min_point = self.bounding_box_2D.get_min_point()
        max_point = self.bounding_box_2D.get_max_point()
        return (self.label + ' ' + str(self.speed_limit),
                ((min_point.x, min_point.y), (max_point.x, max_point.y)))

    def draw_on_frame(self, frame, bbox_color_map, ego_transform=None):
        assert self.bounding_box_2D, \
            'Speed limit sign does not have a 2D bounding box'
        text = '{} {} {:.1f}'.format(self.speed_limit, self.label,
                                     self.confidence)
        super(SpeedLimitSign, self).draw_on_frame(frame, bbox_color_map,
                                                  ego_transform, text)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'SpeedLimitSign(label: {}, limit: {}, '\
            'confidence: {}, id: {}, transform: {}, bbox: {})'.format(
                self.label, self.speed_limit, self.confidence, self.id,
                self.transform, self.bounding_box)


class StopSign(Obstacle):
    def __init__(self,
                 confidence: float,
                 bounding_box: BoundingBox2D = None,
                 id: int = -1,
                 transform: Transform = None,
                 bounding_box_3d: BoundingBox3D = None):
        super(StopSign, self).__init__(bounding_box, confidence,
                                       'stop marking', id, transform)
        self.bounding_box_3d = bounding_box_3d

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates a stop sign from a simulator actor.

        Args:
            actor: A simulator stop sign actor.

        Returns:
            :py:class:`.StopSign`: A stop sign.
        """
        from carla import TrafficSign
        if not isinstance(actor, TrafficSign):
            raise ValueError('actor should be of type TrafficSign')
        transform = Transform.from_simulator_transform(
            actor.get_transform())
        world_trigger_volume = actor.get_transform().transform(
            actor.trigger_volume.location)
        extent = Vector3D.from_simulator_vector(
            actor.trigger_volume.extent)
        trigger_loc = Location(world_trigger_volume.x,
                                           world_trigger_volume.y,
                                           world_trigger_volume.z)
        trigger_transform = Transform(trigger_loc,
                                                  Rotation())
        bbox = BoundingBox3D(trigger_transform, extent)
        return cls(1.0, id=actor.id, transform=transform, bounding_box_3d=bbox)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'StopSign(label: {}, confidence: {}, id: {}, transform: {}, '\
            'bbox: {})'.format(self.label, self.confidence, self.id,
                               self.transform, self.bounding_box)

class Lane(object):
    """Stores information about a lane.

    Args:
        id (:obj:`int`): The id of the lane (0 for ego lane, negative for
            left lanes, and positive for right lanes).
        left_markings: List of transforms.
        right_markings: List of transforms.
    """
    def __init__(self, id: int, left_markings, right_markings):
        self.id = id
        self.left_markings = left_markings
        self.right_markings = right_markings
        self._lane_polygon = None
        self._color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (125, 125, 0), (0, 125, 125), (125, 0, 125),
                           (50, 100, 50), (100, 50, 100)]

    def draw_on_frame(self, frame, inverse_transform=None, binary_frame=None):
        """Draw lane markings on a frame.

        Args:
            bgr_frame: Frame on which to draw the waypoints.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        # change color based on lane id
        lane_color_l = self._color_map[self.id % len(self._color_map)]
        lane_color_r = self._color_map[(self.id + 2) % len(self._color_map)]

        for marking in self.left_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            else:
                marking = marking.location
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    frame.draw_point(pixel_location, lane_color_l)
                    if binary_frame:
                        binary_frame.draw_point(pixel_location,
                                                (255, 255, 255))
                except Exception:
                    continue
        for marking in self.right_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            else:
                marking = marking.location
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    frame.draw_point(pixel_location, lane_color_r)
                    if binary_frame:
                        binary_frame.draw_point(pixel_location,
                                                (255, 255, 255))
                except Exception:
                    continue

    def collect_frame_data(self,
                           frame,
                           binary_frame,
                           camera_setup,
                           inverse_transform=None):
        """Draw lane markings on input frames for lane data collection.

        Args:
            frame: Grayscale image on which to draw the waypoints.
            binary_frame: Black and white image on which to draw the waypoints.
            camera_setup: Camera setup used to generate the frame.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = camera_setup.get_intrinsic_matrix()
        gray_color_map = [(20, 20), (70, 70), (120, 120), (170, 170),
                          (220, 220), (250, 250)]
        # change color based on lane id
        lane_color_l = gray_color_map[self.id % len(gray_color_map)]
        lane_color_r = gray_color_map[(self.id + 2) % len(gray_color_map)]

        for marking in self.left_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            else:
                marking = marking.location
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    cv2.circle(frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, lane_color_l, -1)
                    cv2.circle(binary_frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, (255, 255), -1)
                except Exception:
                    continue
        for marking in self.right_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            else:
                marking = marking.location
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    cv2.circle(frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, lane_color_r, -1)
                    cv2.circle(binary_frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, (255, 255), -1)
                except Exception:
                    continue

    def draw_on_world(self, world):
        from carla import Color
        for marking in self.left_markings:
            world.debug.draw_point(marking.as_simulator_location(),
                                   size=0.1,
                                   color=Color(255, 255, 0))
        for marking in self.right_markings:
            world.debug.draw_point(marking.as_simulator_location(),
                                   size=0.1,
                                   color=Color(255, 255, 0))

    def get_closest_lane_waypoint(self, location):
        if self.is_on_lane(location):
            return Transform(location, Rotation())
        closest_transform = None
        min_dist = np.infty
        for transform in self.left_markings:
            dist = transform.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                closest_transform = transform
        for transform in self.right_markings:
            dist = transform.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                closest_transform = transform
        return closest_transform

    def get_lane_center_transforms(self):
        if len(self.left_markings) < len(self.right_markings):
            anchor_markings = self.left_markings
            other_markings = self.right_markings
        else:
            anchor_markings = self.right_markings
            other_markings = self.left_markings
        index_other = 0
        center_markings = deque([])
        for transform in anchor_markings:
            dist = transform.location.distance(
                other_markings[index_other].location)
            while (index_other + 1 < len(other_markings)
                   and dist > transform.location.distance(
                       other_markings[index_other + 1].location)):
                index_other += 1
                dist = transform.location.distance(
                    other_markings[index_other].location)
            if index_other < len(other_markings):
                other_loc = other_markings[index_other].location
                center_location = Location(
                    (transform.location.x + other_loc.x) / 2.0,
                    (transform.location.y + other_loc.y) / 2.0,
                    (transform.location.z + other_loc.z) / 2.0)
                center_markings.append(Transform(center_location, Rotation()))
        return center_markings

    def is_on_lane(self, location):
        # We only create the lane polygon if it is necessary.
        if not self._lane_polygon:
            self._create_lane_polygon()
        return self._lane_polygon.contains(Point(location.x, location.y))

    def _create_lane_polygon(self):
        points = [(0, self.left_markings[0].y)]
        for transform in self.left_markings:
            points.append((transform.location.x, transform.location.y))
        for transform in reversed(self.right_markings):
            points.append((transform.location.x, transform.location.y))
        points.append((0, self.right_markings[0].y))
        self._lane_polygon = Polygon(points)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Lane(id: {}, {})'.format(
            self.id, zip(self.left_markings, self.right_markings))