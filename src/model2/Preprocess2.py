# coding=utf-8
'''
获取原始数据
'''

import numpy as np
import pandas as pd

ORI_PATH = 'D:/StockData/'
start = '2015-01-01'
end = '2017-12-31'
train_path = ORI_PATH + '11_MODEL_02/train.csv'
test_path = ORI_PATH + '11_MODEL_02/test.csv'
valid_path = ORI_PATH + '11_MODEL_02/valid.csv'


def preprocess(path, start_date=None, end_date=None):
    stock_list = pd.read_csv(ORI_PATH + 'stock.csv', dtype=object)

    for idx in stock_list.index:
        code = stock_list.loc[idx]['code']
        try:
            deal_stock(code, path, start_date, end_date)
        except Exception:
            print('error!')

        print('finished == > ' + code)


def deal_stock(code, path, start_date=None, end_date=None):
    df0 = pd.read_csv(ORI_PATH + '01_ORI/' + code + '_15_17.csv', index_col=0)

    if start_date or end_date:
        df = df0[((df0.index > start_date) & (df0.index <= end_date))].sort_index()
    else:
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

    result = pd.DataFrame(result, index=df.index[0:lenth]).sample(frac=1)
    result.to_csv(path, header=False,
                  index=False, mode='a')
    # print(result_x[0])
    # print(result_y[0])


if __name__ == '__main__':
    # preprocess(train_path,'2017-04-01', '2017-07-01')
    # preprocess(test_path,'2017-07-01', '2017-08-25')
    # preprocess(valid_path,'2017-08-25', '2017-10-20')
    # deal_stock('000001', '2017-04-01', '2017-07-01')
    # save(BASE_PATH+'600101.csv','600101',start,end)
    # stock_list = ts.get_stock_basics()
    # for code in stock_list.index:
    #     try:
    #         file_path = BASE_PATH + str(code)+'_15_17.csv'
    #         save(file_path, code, start, end)
    #         print('saved ===> ' + str(code))
    #     except Exception:
    #         print('error!')
