# coding=utf-8
'''
获取原始数据
'''

import numpy as np
import pandas as pd

ORI_PATH = 'D:/StockData/11_MODEL_01/'


class Model01Input(object):
    def __init__(self,path=None):
        self.path = ORI_PATH + 'data3.csv'
        if path:
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
        yy = np.where(y > 6, [0, 0, 1], np.where(y > 2, [0, 1, 0], [1, 0, 0]))
        return x, yy


if __name__ == '__main__':
    inp = Model01Input()
    print(inp.next_train_batch(1))
    print('=' * 20)
    print(inp.next_train_batch(1))
    # preprocess()
    # deal_stock('000001')
    # save(BASE_PATH+'600101.csv','600101',start,end)
    # stock_list = ts.get_stock_basics()
    # for code in stock_list.index:
    #     try:
    #         file_path = BASE_PATH + str(code)+'_15_17.csv'
    #         save(file_path, code, start, end)
    #         print('saved ===> ' + str(code))
    #     except Exception:
    #         print('error!')
    pass
