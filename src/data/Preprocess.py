# coding=utf-8
'''
获取原始数据
'''

import numpy as np
import pandas as pd

ORI_PATH = 'D:/StockData/'
start = '2015-01-01'
end = '2017-12-31'


def preprocess():
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

    pd.DataFrame(result, index=df.index[0:lenth]).to_csv(ORI_PATH + '11_MODEL_01/' + code + '.csv', header=False,
                                                         index=False)
    # print(result_x[0])
    # print(result_y[0])


if __name__ == '__main__':
    preprocess()
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
