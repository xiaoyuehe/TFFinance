import ssl
from urllib import request

ssl._create_default_https_context = ssl._create_unverified_context


def url_by_proxy():
    proxy_handler = request.ProxyHandler({
        'http': 'http://111.155.116.217:8123',
        'https': 'https://111.155.116.217:8123'
    })
    opener = request.build_opener(proxy_handler)
    response = opener.open('http://www.baidu.com')
    print('Status:', response.status, response.reason)
    for k, v in response.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', response.read().decode('utf-8'))


url_by_proxy()

# proxies = {'https': 'https://123.55.176.229:25027'}
# filehandle = urllib.urlopen('http://www.baidu.com', proxies=proxies)
# print(filehandle.read().decode('utf-8'))
# Don't use any proxies
# filehandle = urllib.urlopen(some_url, proxies={})
# Use proxies from environment - both versions are equivalent
# filehandle = urllib.urlopen(some_url, proxies=None)
# filehandle = urllib.urlopen(some_url)
