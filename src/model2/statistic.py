# coding=utf-8
'''
resnet implimention
'''

import numpy as np
from model02_input import Model02Input

MODEL_BASE = 'D:/StockData/11_MODEL_02/'

result_dict = {}
inp = Model02Input(MODEL_BASE + '/train.csv')
for i in range(85):
    x, y = inp.next_train_batch(1000)

    for j in range(len(y)):
        key = str(np.argmax(y[j]))
        if key in result_dict:
            result_dict[key] = result_dict[key] + 1
        else:
            result_dict[key] = 1

print(result_dict)
print(85000/3/result_dict['0'])
print(85000/3/result_dict['1'])
print(85000/3/result_dict['2'])