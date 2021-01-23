import json

import numpy as np
import pandas as pd


def read_json(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    return data