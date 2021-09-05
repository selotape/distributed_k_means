import logging
import subprocess
import sys

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dist_k_mean.config import M

formatter = logging.Formatter('%(asctime)s %(message)s')


def setup_logger(name, log_file, with_console=False, level=logging.INFO):
    """To setup as many loggers as you want"""

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    if with_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


LAST_RUNTIME = '_last_runtime'


def keep_time(func):
    """Instance-Method decorator that saves the methods last execution time on the instance."""

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        args[0].__dict__[func.__name__ + LAST_RUNTIME] = end - start
        return result

    return wrap


def get_kept_time(obj, func_name):
    return obj.__dict__[func_name + LAST_RUNTIME]


def log_config_file(logger):
    label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    logger.info(f"git commit: {label}")

    with open('dist_k_mean/config.py') as config_f:
        config_txt = config_f.readlines()
    [logger.info(line) for line in config_txt]


class Measurement(ABC):

    @abstractmethod
    def reducers_time(self):
        pass

    @abstractmethod
    def total_time(self):
        pass

    @abstractmethod
    def total_comps_per_machine(self):
        pass

    @abstractmethod
    def total_comps(self):
        pass

    @abstractmethod
    def iterations(self):
        pass

    @abstractmethod
    def coord_memory(self):
        pass

    @abstractmethod
    def num_centers_unfinalized(self):
        pass


@dataclass
class SimpleMeasurement(Measurement):
    reducers_time_: float = 0.0
    coordinator_time_: float = 0.0
    finalization_time_: float = 0.0
    dist_comps_: float = 0.0
    iterations_: int = 0
    num_centers_unfinalized_: int = 0
    coord_memory_: float = -1.0

    def reducers_time(self):
        return self.reducers_time_

    def coordinator_time(self):
        return self.coordinator_time_

    def total_time(self):
        return self.reducers_time_ + self.coordinator_time_ + self.finalization_time_

    def total_comps_per_machine(self):
        return self.dist_comps_

    def total_comps(self):
        return self.total_comps_per_machine() * M

    def coord_memory(self):
        return self.coord_memory_

    def num_centers_unfinalized(self):
        return self.num_centers_unfinalized_

    def iterations(self):
        return self.iterations_


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True