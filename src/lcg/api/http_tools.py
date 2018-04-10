"""
http协议相关的工具类
"""

from urllib import request


def build_proxy_url_opener(proxy_obj):
    proxy_handler = request.ProxyHandler(proxy_obj)
    return request.build_opener(proxy_handler)


def build_req_obj(url, headers, data, method):
    req = request.Request(url=url, data=data, headers=headers, method=method)
    response = self.opener.open(req, timeout=timeout)
    return response.read()
