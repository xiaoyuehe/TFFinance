# -*-coding:utf-8-*-
"""
自动抓取ip代理池，多线程验活，提供输出接口给具体的业务对象
:author xiaoyuehe
"""

from api/http_tool import build_opener,build_request,fetch_response,read_response_result
from api/response_parser import dom_xpath_array

class IpProxyHolder(object):
    def __init__(self):
        self.ip_list = {}
        self.valid_ip_list = {}
        self.fail_ip_list = {}

    def load(self, path):
        pass

    def restore(self, path):
        pass

    def validate(self):
        pass

    def random_proxy(self, count):
        return None


def xichi_ip_proxy():
    result = []
    url = ''
    xpath = ''
    url_opener = build_opener(use_proxy=False)
    req = build_request(url)
    result_html = read_response_result(fetch_response(url_opener,req))
    dom_array = dom_xpath_array(result_html,xpath)
    for el in dom_array:

    return result