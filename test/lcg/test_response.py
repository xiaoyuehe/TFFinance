# !/usr/bin/env python
# -*- coding:utf-8 -*-

from urllib import request
from urllib import response

URL="http://www.baidu.com/"

# 构造请求头信息
# 反反爬虫:设置User-Agent
request_headers={
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36"
}

# 构造请求对象,并添加请求头
req=request.Request(URL,headers=request_headers)
# 发起请求
resp=request.urlopen(req)


print(type(resp))   # <class 'http.client.HTTPResponse'>
# 获取HTTP协议版本号(10 for HTTP/1.0, 11 for HTTP/1.1)
print(resp.version)

# 获取响应码
print(resp.status)
print(resp.getcode())

# 获取响应描述字符串
print(resp.reason)

# 获取实际请求的页面url(防止重定向用)
print(resp.geturl())

# 获取特定响应头信息
print(resp.getheader(name="Content-Type"))
# 获取响应头信息,返回二元元组列表
print(resp.getheaders())
# 获取响应头信息,返回字符串
print(resp.info())

# 读取响应体
print(resp.readline().decode('utf-8'))
print(resp.read().decode('utf-8'))

