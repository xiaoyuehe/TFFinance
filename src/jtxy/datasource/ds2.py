# coding=utf-8
'''
获取原始数据
'''

import numpy as np
import pandas as pd


def process_y(y):
    return np.where(y >= 6, [10], np.where(y >= 3, [9], np.where(y >= 2, [8], np.where(y >= 1, [7],
                                                                                       np.where(
                                                                                           y >= 0,
                                                                                           [6],
                                                                                           np.where(
                                                                                               y >= -1,
                                                                                               [5],
                                                                                               np.where(
                                                                                                   y >= -2,
                                                                                                   [
                                                                                                       4],
                                                                                                   np.where(
                                                                                                       y >= -3,
                                                                                                       [
                                                                                                           3],
                                                                                                       np.where(
                                                                                                           y >= -6,
                                                                                                           [
                                                                                                               2],
                                                                                                           [
                                                                                                               1])))))))))


def process_y_2(y):
    return np.where(y >= 6, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    np.where(y >= 3, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             np.where(y >= 2, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                      np.where(y >= 1, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                               np.where(
                                                   y >= 0,
                                                   [0, 0, 0,
                                                    0, 1, 0,
                                                    0, 0, 0,
                                                    0],
                                                   np.where(
                                                       y >= -1,
                                                       [0,
                                                        0,
                                                        0,
                                                        0,
                                                        1,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0],
                                                       np.where(
                                                           y >= -2,
                                                           [
                                                               0,
                                                               0,
                                                               0,
                                                               1,
                                                               0,
                                                               0,
                                                               0,
                                                               0,
                                                               0,
                                                               0],
                                                           np.where(
                                                               y >= -3,
                                                               [
                                                                   0,
                                                                   0,
                                                                   1,
                                                                   0,
                                                                   0,
                                                                   0,
                                                                   0,
                                                                   0,
                                                                   0,
                                                                   0],
                                                               np.where(
                                                                   y >= -6,
                                                                   [
                                                                       0,
                                                                       1,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0],
                                                                   [
                                                                       1,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0])))))))))


class Ds2(object):
    def __init__(self, path=None):
        self.path = path
        self._reset_()

    def _reset_(self):
        self.reader = pd.read_table(self.path, sep=',', header=None, iterator=True)

    def next_train_batch(self, batch_size):
        df = self.reader.get_chunk(batch_size)
        actual_size = df.iloc[:, 0].size
        if actual_size < batch_size:
            self._reset_()

        arr = df.as_matrix()
        x = arr[:, 0:125].reshape((actual_size, 5, 5, 5))
        y = arr[:, 125:]

        # one = np.ones(y.shape)
        yy = process_y_2(y)
        return x, yy

    def next_train_batch_flat(self, batch_size):
        df = self.reader.get_chunk(batch_size)
        actual_size = df.iloc[:, 0].size
        if actual_size < batch_size:
            self._reset_()

        arr = df.as_matrix()
        x = arr[:, 0:125]
        y = arr[:, 125:]

        # one = np.ones(y.shape)
        yy = process_y(y)
        return x, yy
