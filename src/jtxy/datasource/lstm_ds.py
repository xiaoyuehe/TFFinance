# coding=utf-8
'''
获取原始数据
'''

import numpy as np
import pandas as pd


class Ds(object):
    def __init__(self, path=None):
        self.path = path
        self.indx = 0
        self.cache = None
        self._reset_()

    def _reset_(self):
        self.reader = pd.read_table(self.path, sep=',', header=None, iterator=True)

    def next_train_batch1(self, step_times):
        df = self.reader.get_chunk(step_times)
        actual_size = df.iloc[:, 0].size
        if actual_size < step_times:
            self._reset_()
            df = self.reader.get_chunk(step_times)

        arr = df.as_matrix()
        x = arr[:, 0:5].reshape((1, step_times, 5))
        y = arr[:, 5:].reshape((1, step_times, 1))

        return x, y

    def next_train_batch(self, step_times, batch_size=1):
        if batch_size == 1:
            return self.next_train_batch1(step_times)

        result_x, result_y = self.next_train_batch1(step_times)
        for _ in range(batch_size - 1):
            c_x, c_y = self.next_train_batch1(step_times)
            result_x = np.concatenate([result_x, c_x], axis=0)
            result_y = np.concatenate([result_y, c_y], axis=0)

        return result_x, result_y
