# -*-coding:utf-8-*-
"""
http相关工具方法
:author xiaoyuehe
"""
import ssl
from urllib import request

ssl._create_default_https_context = ssl._create_unverified_context


def build_opener(use_proxy=False, proxy_obj=None):
    """获取url opener用于后续发送请求
    :param use_proxy: 是否需要使用代理
    :param proxy_obj: 代理对象dict
    {'http':'http://www.xxx.com:81321','https':'https://www.xxx.com:81321'}
    :return:
    """
    if use_proxy:
        proxy_handler = request.ProxyHandler(proxy_obj)
        return request.build_opener(proxy_handler)
    else:
        return request.build_opener()


def build_request(url, method='GET', headers_dict={}, data=None):
    """
    构造Request对象（包含请求头和发送的数据信息）
    :param url:
    :param method:
    :param headers_dict:
    :param data:
    :return:
    """
    return request.Request(url=url, data=data, method=method, headers=headers_dict)


def fetch_response(opener, req, timeout=None):
    """
    获取response对象
    :param opener:
    :param req:
    :param timeout:
    :return:
    """
    return opener.open(req, timeout=timeout)


def read_response_result(resp, encode='utf8'):
    """
    按照编码获取结果字符串
    :param resp:
    :param encode:
    :return:
    """
    return resp.read().decode(encode)


def response_code(resp):
    """
    按照编码获取返回码
    :param resp:
    :return:
    """
    return resp.getcode()
