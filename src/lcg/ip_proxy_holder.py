# -*-coding:utf-8-*-
"""
自动抓取ip代理池，多线程验活，提供输出接口给具体的业务对象
:author xiaoyuehe
"""

from api.http_tool import build_opener, build_request, read_response_result, fetch_response
from api.response_parser import dom_xpath_array
from lxml import etree


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
    # url = 'http://www.baidu.com'
    url = 'http://www.xicidaili.com/nn/2'
    xpath = '//table[@id="ip_list"]/tr'
    url_opener = build_opener(use_proxy=False)
    headers_dict = {}
    headers_dict[
        'User-Agent'] = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Mobile Safari/537.36'
    req = build_request(url, headers_dict=headers_dict)
    result_html = read_response_result(fetch_response(url_opener, req))
    print(result_html)
    dom_array = dom_xpath_array(result_html, xpath)
    i = 0
    for el in dom_array:
        if i == 0:
            i += 1
            continue
        el = etree.fromstring(etree.tostring(el))
        print(etree.tostring(el.xpath('//td[position()=2]')[0]))
        ip = el.xpath('//td[position()=2]')[0].text
        port = el.xpath('//td[position()=3]')[0].text
        ptype = el.xpath('//td[position()=6]')[0].text
        print('%s %s %s' % (ip, port, ptype))
    return result


def valid_proxy(proxy_obj):
    url = 'https://www.baidu.com'
    url_opener = build_opener(use_proxy=True, proxy_obj=proxy_obj)
    headers_dict = {}
    headers_dict[
        'User-Agent'] = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Mobile Safari/537.36'
    req = build_request(url, headers_dict=headers_dict)
    try:
        resp = fetch_response(url_opener, req)
        print(resp.read())
        if resp.getcode() == 200:
            return True
    except BaseException:
        return False


# xichi_ip_proxy()
po={'http':'http://49.64.186.185:30919'}
print(valid_proxy(po))