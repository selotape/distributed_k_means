import logging
import sys

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
        logger.addHandler(console)

    return logger
