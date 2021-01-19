import os
import sys
import time
import logging

from enum import Enum

class GameOutcome(Enum):
    TIE = 1
    LEFT_WIN = 2
    RIGHT_WIN = 3


def timeit(func):
    def timed_func(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        print(f'runtime for {func.__name__.upper()}: {(te - ts):.2f} s')
        return result
    return timed_func


def get_logger(logger_name = 'test'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(formatter))
    logger.addHandler(console_handler)
    return logger