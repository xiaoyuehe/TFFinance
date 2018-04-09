import urllib
import ssl
ssl._create_default_https_context=ssl._create_unverified_context

# proxy_handler = urllib2.request.ProxyHandler({'http': 'http://183.15.120.2:20379/'})
# proxy_auth_handler = request.ProxyBasicAuthHandler()
# # proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
# opener = request.build_opener(proxy_handler, proxy_auth_handler)
# with opener.open('https://www.baidu.com') as f:
#     print('Status:', f.status, f.reason)
#     for k, v in f.getheaders():
#         print('%s: %s' % (k, v))
#     print('Data:', f.read().decode('utf-8'))
#     pass

proxies = {'https': 'https://123.55.176.229:25027'}
filehandle = urllib.urlopen('http://www.baidu.com', proxies=proxies)
print(filehandle.read().decode('utf-8'))
# Don't use any proxies
# filehandle = urllib.urlopen(some_url, proxies={})
# Use proxies from environment - both versions are equivalent
# filehandle = urllib.urlopen(some_url, proxies=None)
# filehandle = urllib.urlopen(some_url)