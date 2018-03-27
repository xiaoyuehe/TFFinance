# coding=utf-8
'''
resnet implimention
'''

import math
import numpy as np
from datasource.ds1 import Ds1


def cast_float(float_value):
    if not float_value:
        return 0.0
    return float_value
