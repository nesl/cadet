"""CSV / metric logging helpers for the pipeline.

Derived from ERDOS Pylot (Apache-2.0); see the project NOTICE.
"""
import logging
import time
from datetime import datetime
import os
import numpy as np
import pandas as pd

from examples.av_pipeline import params

log_dir = os.path.join(params.OUTPUT_DIR, 'logs') + os.sep
metrics_dir = os.path.join(params.OUTPUT_DIR, 'metrics') + os.sep

def setup_module_logging(module_name):
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Define the format for the log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}-{module_name}.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='a'),
                                  logging.StreamHandler()])


def setup_pipeline_logging():
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Define the format for the pipeline log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}+pipeline.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Configure centralized logging
    logging.basicConfig(level=logging.INFO,
                        format='@%(asctime)s [%(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_filepath, mode='a'),
                                  logging.StreamHandler()])


class ModuleLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter to add the module name to log messages.
    """
    def process(self, msg, kwargs):
        return f'{self.extra["module_name"]} {msg}', kwargs


def get_module_logger(module_name):
    logger = logging.getLogger(module_name)
    adapter = ModuleLoggerAdapter(logger, {'module_name': module_name})
    return adapter


class ModuleCompletionLogger:
    _instance = None
    filename = None

    def __new__(cls):
        if cls._instance is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cls.filename = f"{timestamp}-module_completion_time.csv"
            cls._instance = super(ModuleCompletionLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'is_initialized'):
            self.log_dir = metrics_dir
            self.filepath = os.path.join(self.log_dir, self.filename)
            self.setup_central_csv_logger()
            self.is_initialized = True

    def setup_central_csv_logger(self):
        """Sets up a central CSV logger for module completion times using pandas."""
        if not os.path.isfile(self.filepath):
            df = pd.DataFrame(columns=['module_name', 'completion_time'])
            df.to_csv(self.filepath, index=False)

    def log_module_completion(self, module_name, completion_time):
        """Logs the completion time of a module to the central CSV file using pandas"""
        new_row = pd.DataFrame({
            'module_name': [module_name],
            'completion_time': [completion_time]
        })
        new_row.to_csv(self.filepath, mode='a', header=False, index=False)

def get_timestamp_logger():
    #if tl_file == None:
    tl_file = open(params.timestamp_log_file, 'a', 1)
    return tl_file

class ModuleLogParser:

    def __init__(self):
        self._timestamp_map = {}
    
    def read(self, filename):
        file = open(filename, 'r')
        lines = file.readlines()
        for line in lines:
            fields = line.split()
            if len(fields) < 3:
                print("Warning! Encountered too few fields in  - " + line)
            self._add(int(fields[0]), fields[1], float(fields[2]))
    
    def _add(self, timestamp, name, value):
        if timestamp not in self._timestamp_map:
            self._timestamp_map[timestamp] = {}
        if name in self._timestamp_map[timestamp]:
            print("Warning! Encountered duplicate value for {} at {}".format(name, timestamp))
        self._timestamp_map[timestamp][name] = float(value)
    
    def get_timestamps(self):
        return [key for key in self._timestamp_map.keys()]
    
    def get(self, timestamp):
        return self._timestamp_map[timestamp]
    
    def get_acceleration(self, timestamp):
        return self.get(timestamp)['acceleration']
    
    def get_actual_distance(self, timestamp):
        return self.get(timestamp)['actual_distance']
    
    def get_perceived_distance(self, timestamp):
        return self.get(timestamp)['perceived_distance']
    
