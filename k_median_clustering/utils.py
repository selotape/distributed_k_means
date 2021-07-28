import logging
import sys

import time

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
