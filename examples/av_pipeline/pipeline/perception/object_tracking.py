"""Multi-object tracker wrapper (SORT / DeepSORT).

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
import time
from collections import deque
import time

import examples.av_pipeline.utils.logging as cadet_logging
from examples.av_pipeline import params

class ObjectTracker():
    def __init__(self):

        self._config_name = "ObjectTracker"
        self._module_logger = cadet_logging.get_module_logger(self._config_name)
        self._csv_logger = cadet_logging.ModuleCompletionLogger()

        print("\nInitializing Object Tracker")
        self._module_logger.info("\nInitializing Object Tracker")

        self._tracker_type = params.tracker_type
        # Absolute time when the last tracker run completed.
        self._last_tracker_run_completion_time = 0
        try:
            if params.tracker_type == 'deep_sort':
                from examples.av_pipeline.pipeline.perception.tracker import MultiObjectDeepSORTTracker
                self._tracker = MultiObjectDeepSORTTracker()
            elif params.tracker_type == 'sort':
                from examples.av_pipeline.pipeline.perception.tracker import MultiObjectSORTTracker
                self._tracker = MultiObjectSORTTracker()
            else:
                raise ValueError(
                    'Unexpected tracker type {}'.format(self._tracker_type))
        except ImportError as error:
            self._module_logger.exception('Error importing {}'.format(self._tracker_type))
            raise error
        
        self._detection_update_count = -1
        self._server=None

    def _reinit_tracker(self, camera_frame, detected_obstacles):
        start = time.time()
        result = self._tracker.reinitialize(camera_frame, detected_obstacles)
        return (time.time() - start) * 1000, result

    def _run_tracker(self, camera_frame):
        start = time.time()
        result = self._tracker.track(camera_frame)
        return (time.time() - start) * 1000, result

    def get_tracked_obstacles(self, timestamp, camera_frame, obstacles):
        start_time = time.time()

        detected_obstacles = []
        for obstacle in obstacles:
            if obstacle.is_vehicle() or obstacle.is_person():
                detected_obstacles.append(obstacle)
        self._tracker.reinitialize(camera_frame, detected_obstacles)
        ok, tracked_obstacles = self._tracker.track(camera_frame)

        self._module_logger.info('@{}: {} obstacles: {}'.format(timestamp, self._config_name, tracked_obstacles))

        assert ok, 'Tracker failed at timestamp {}'.format(timestamp)
        tracker_runtime = 1000 * (time.time() - start_time)
        self._csv_logger.log_module_completion(self._config_name, tracker_runtime)
        return (timestamp, tracked_obstacles, tracker_runtime)