# coding=utf-8
'''
获取原始数据
'''

import numpy as np
import pandas as pd
import os

ORI_PATH = 'D:/StockData/11_MODEL_01/'

def get_tran_data(batch_size):
    reader = pd.read_table(ORI_PATH+'data.csv', sep=',', chunksize=batch_size)
    for chunk in reader:
        print(chunk.as_matrix(columns=[1:125]))

get_tran_data(100)

class Model01Input(object):

    def __init__(self):
        self.train_files = os.listdir(ORI_PATH + 'train')
        self.train_files_index = 0
        self.train_x = []
        self.train_y = []
        self.test_files =os.listdir(ORI_PATH + 'test')
        self.test_files_index = 0
        self.test_x = []
        self.test_y = []

    def next_train_batch(self,batch_size):
        if len(self.train_x) >= batch_size:
            pass

        if not train_files or len(train_files) == 0:
            train_files = os.listdir(ORI_PATH + 'train')
        stock_list = pd.read_csv(ORI_PATH + 'stock.csv', dtype=object)


        for idx in stock_list.index:
            code = stock_list.loc[idx]['code']
            try:
                deal_stock(code)
            except Exception:
                print('error!')

            print('finished == > ' + code)


def deal_stock(code):
    df0 = pd.read_csv(ORI_PATH + '01_ORI/' + code + '_15_17.csv', index_col=0)
    df = df0.sort_index()

    df['lcp'] = df['close'] * 100 / (df['p_change'] + 100)
    df['r1'] = 100 * (df['open'] - df['lcp']) / df['lcp']
    df['r2'] = 100 * (df['high'] - df['lcp']) / df['lcp']
    df['r3'] = 100 * (df['low'] - df['lcp']) / df['lcp']

    df['r1_1'] = df['r1'][1:]

    # print(df.iloc[2])
    # print(df)

    lenth = df.iloc[:, 0].size - 30
    print(lenth)
    result = np.zeros((lenth, 126), dtype=np.float64)
    for i in range(lenth):
        for j in range(25):
            idx = i + j
            row = df.iloc[idx]
            np.put(result[i], [j, 25 + j, 50 + j, 75 + j, 100 + j],
                   [row['p_change'], row['r1'], row['r2'], row['r3'], row['turnover']])
        np.put(result[i], [125],
               [df.iloc[i + 25:i + 30, 6].sum()])
        # print(df.iloc[i + 25:i + 30, 6])

    pd.DataFrame(result, index=df.index[0:lenth]).to_csv(ORI_PATH + '11_MODEL_01/' + code + '.csv')
    # print(result_x[0])
    # print(result_y[0])


if __name__ == '__main__':
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
