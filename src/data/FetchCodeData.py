# coding=utf-8
'''
获取原始数据
'''

import tushare as ts


ORI_PATH = '/Users/yaoli/StockData/stock.csv'
# ORI_PATH = 'D:/StockData/stock.csv'

if __name__ == '__main__':
    stock_list = ts.get_stock_basics()
    if stock_list is not None:
        stock_list.to_csv(ORI_PATH)

