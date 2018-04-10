# -*-coding:utf-8-*-
"""
http response解析相关工具方法
requere：
    pip install lxml
    pip install cssselect
如果熟悉css选择器，获取页面中的对象会非常的容易
:author xiaoyuehe
"""

from lxml import html, cssselect


def dom_array(html_str, cs_str):
    """
    获取所有符合样式选择的dom结构的列表
    :param html_str:
    :param cs_str:
    :return:
    """
    tree = html.fromstring(html_str)
    return tree.cssselect(cs_str)


def dom_xpath_array(html_str, xpath_str):
    """
    获取所有符合xpath_str的dom结构的列表
    :param html_str:
    :param xpath_str:
    :return:
    """
    tree = html.fromstring(html_str)
    return tree.xpath(xpath_str)

