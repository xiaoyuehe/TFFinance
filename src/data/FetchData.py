# coding=utf-8
'''
获取原始数据
'''

import tushare as ts


def save(path, code, start, end, type='D'):
    df = ts.get_hist_data(code, start, end, ktype=type)
    if df is not None:
        df.to_csv(path)


# BASE_PATH = '/Users/yaoli/StockData/01_ORI/'
BASE_PATH = 'D:/StockData/01_ORI/'
start = '2001-01-01'
end = '2018-06-31'

if __name__ == '__main__':
    save(BASE_PATH+'sh.csv','sh',start,end)
    # stock_list = ts.get_stock_basics()
    # for code in stock_list.index:
    #     try:
    #         file_path = BASE_PATH + str(code)+'_15_17.csv'
    #         save(file_path, code, start, end)
    #         print('saved ===> ' + str(code))
    #     except Exception:
    #         print('error!')
