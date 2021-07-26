import logging
import sys

import time

formatter = logging.Formatter('%(asctime)s %(message)s')


def setup_logger(name, log_file, another_file=None, with_console=False, level=logging.INFO):
    """To setup as many loggers as you want"""

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    if another_file:
        another_file_handler = logging.FileHandler(another_file)
        another_file_handler.setFormatter(formatter)
        logger.addHandler(another_file_handler)

    if with_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


LAST_RUNTIME = '_last_runtime'


def keep_time(func):
    '''Instance-Method decorator that saves the methods last execution time on the instance.'''

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        args[0].__dict__[func.__name__ + LAST_RUNTIME] = end - start
        return result

    return wrap


# At the beginning of every .py file in the project
def logger(fn):
    from functools import wraps
    import inspect
    @wraps(fn)
    def wrapper(*args, **kwargs):
        log = logging.getLogger(fn.__name__)
        log.info('About to run %s' % fn.__name__)

        out = fn(*args, **kwargs)

        log.info('Done running %s' % fn.__name__)
        # Return the return value
        return out

    return wrapper
