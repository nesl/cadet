"""Camera sensor setup descriptors (resolution, FOV, transform).

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
import numpy as np

from examples.av_pipeline.pipeline.objects.objects import Location, Rotation, Transform

def create_left_right_camera_setups(camera_name_prefix: str,
                                    location: Location,
                                    width: int,
                                    height: int,
                                    camera_offset,
                                    fov: float = 90):
    """Creates a dual-RGB-camera setup with the center at the given location,
    and the two cameras on either side of the center at a distance specified
    by the camera_offset.

    The Rotation is set to (pitch=0, yaw=0, roll=0).

    Args:
        camera_name_prefix (str): The name of the camera instance. A suffix
            of "_left" and "_right" is appended to the name.
        location (:py:class:`~pylot.utils.Location`): The location of the
            center of the cameras with respect to the center of the vehicle.
        width (int): The width of the image returned by the cameras.
        height (int): The height of the image returned by the cameras.
        camera_offset (float): The offset of the two cameras from the center.
        fov (float): The field of view of the image returned by the cameras.

    Returns:
        tuple: A tuple containing two instances of
        :py:class:`~pylot.drivers.sensor_setup.RGBCameraSetup` for the left
        and right camera setups with the given parameters.
    """
    rotation = Rotation()
    left_loc = location + Location(0, -camera_offset, 0)
    right_loc = location + Location(0, camera_offset, 0)
    left_transform = Transform(left_loc, rotation)
    right_transform = Transform(right_loc, rotation)
    left_camera_setup = RGBCameraSetup(camera_name_prefix + '_left',
                                       width,
                                       height,
                                       left_transform,
                                       fov=fov)
    right_camera_setup = RGBCameraSetup(camera_name_prefix + '_right',
                                        width,
                                        height,
                                        right_transform,
                                        fov=fov)
    return (left_camera_setup, right_camera_setup)


class CameraSetup(object):
    """CameraSetup stores information about an instance of the camera
    mounted on the vehicle.

    Args:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
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
        "of the type pylot.utils.Transform"
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
        "of the type pylot.utils.Transform"
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