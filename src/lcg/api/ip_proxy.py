"""
使用proxy方式进行网络数据抓取工具类
"""

import json
import ssl
from urllib import request, error


class IpProxyValidate(object):
    def __init__(self):
        pass


class IpProxyHandler(object):
    def __init__(self):
        self.opener = None
        self.last_state = 0
        self.acc_fail = 0
        self.continue_fail = 0
        ssl._create_default_https_context = ssl._create_unverified_context
        pass

    def change_proxy(self, proxy_obj):
        proxy_handler = request.ProxyHandler(proxy_obj)
        self.opener = request.build_opener(proxy_handler)

    def fetch_content(self, url, data=None, headers=None, method='POST', timeout=120):
        try:
            req = request.Request(url=url, data=data, headers=headers, method=method)
            response = self.opener.open(req, timeout=timeout)
            return response.read()
        except BaseException as  e:
            print(e)
            self.acc_fail += 1
            if self.last_state == 1:
                self.continue_fail += 1
            else:
                self.continue_fail = 1
            self.last_state = 1

        return None

    def fetch_content_retry(self, url, times, timeout, data=None, headers=None, method='POST'):
        for i in range(times):
            result = self.fetch_content(url=url, data=data, headers=headers, method=method, timeout=timeout)
            if self.last_state == 0:
                return result
        return None


url = 'https://pet-chain.baidu.com/data/pet/queryPetById'
# url = 'http://www.baidu.com'
post_data = {
    'appId': 1,
    'nounce': None,
    'petId': '1922937328933371507',
    'requestId': 1522942205543,
    'timeStamp': None,
    'token': None,
    'tpl': ""}
headers = {'Content-Type': 'application/json'}
data = bytes(json.dumps(post_data), 'utf8')
proxy = IpProxyHandler()
proxy.change_proxy({
    'http': 'http://111.155.116.217:8123',
    'https': 'https://111.155.116.217:8123'
})
print(proxy.fetch_content_retry(url, 3, 120, data=post_data, headers=headers))
