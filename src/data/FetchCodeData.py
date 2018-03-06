# coding=utf-8
'''
获取原始数据
'''

import tushare as ts

if __name__ == '__main__':
    stock_list = ts.get_stock_basics()
    if stock_list is not None:
        stock_list.to_csv('D:/StockData/stock.csv')

