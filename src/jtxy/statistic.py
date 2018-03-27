# coding=utf-8
'''
resnet implimention
'''

import math
import numpy as np
from datasource.ds1 import Ds1

MODEL_BASE = '/Users/yaoli/StockData/11_MODEL_02/'
# MODEL_BASE = 'D:/StockData/11_MODEL_02/'

result_dict = {}
inp = Ds1(MODEL_BASE + '/train_sh.csv')
for i in range(85):
    x, y = inp.next_train_batch_origin(1000)

    for j in range(len(y)):
        key = str(math.ceil(y[j]))
        if key in result_dict:
            result_dict[key] = result_dict[key] + 1
        else:
            result_dict[key] = 1

for kk in result_dict:
    print(kk + "\t" + str(result_dict.get(kk)))
