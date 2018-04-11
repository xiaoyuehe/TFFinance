# -*-coding:utf-8-*-
"""
自动抓取ip代理池，多线程验活，提供输出接口给具体的业务对象
:author xiaoyuehe
"""

from api.http_tool import build_opener, build_request, read_response_result, fetch_response
from api.response_parser import dom_xpath_array
from lxml import etree
import csv
import threading


class IpProxyHolder(object):
    def __init__(self):
        self.ip_list = {}
        self.valid_ip_list = {}
        self.fail_ip_list = {}

    def load(self, path):
        self.ip_list = {}
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                key_str = '_'.join(row)
                self.ip_list[key_str] = row

    def restore(self, path):
        with open(path + '.all', 'w', newline='') as f:
            writer = csv.writer(f)
            for row in self.ip_list.values():
                writer.writerow(row)
        with open(path + '.ok', 'w', newline='') as f:
            writer = csv.writer(f)
            for row in self.valid_ip_list.values():
                writer.writerow(row)

    def validate(self):
        for key, row in self.ip_list.items():
            po = row_to_proxy_obj(row)
            result = valid_proxy(po)
            print('validate url -- > ' + str(po) + (' ok ' if result else ' fail '))
            if result:
                self.valid_ip_list[key] = row
            else:
                self.fail_ip_list[key] = row

    def multi_validate(self):
        threads = []
        thread_count = 50
        batch_size = int(len(self.ip_list) / thread_count)
        i = 0
        rows = []
        for key, row in self.ip_list.items():
            rows.append(row)
            i += 1
            if i % batch_size == 0:
                t = threading.Thread(target=multi_thread_validate_proxy,
                                     args=('thread-' + str(int(i / batch_size)), self, rows,))
                t.start()
                threads.append(t)
                # thread.start_new_thread(multi_thread_validate_proxy,
                #                         ('thread-' + str(int(i / thread_count)), self, rows,))
                rows = []

        for th in threads:
            th.join()

    def get_valid_proxy(self, count, htype='http'):
        result = []
        for row in self.valid_ip_list.values():
            if len(result) == count:
                break
            if row[2] == htype:
                result.append(row_to_proxy_obj(row))
        return result

    def fetch_new(self):
        ip_proxy_list = xichi_ip_proxy()
        for row in ip_proxy_list:
            key_str = '_'.join(row)
            print(key_str)
            self.ip_list[key_str] = row

    def print(self):
        for row in self.ip_list.values():
            print(row)


def row_to_proxy_obj(row):
    po = {}
    po[row[2]] = row[2] + '://' + row[0] + ':' + row[1]
    return po


def proxy_key(row):
    return '_'.join(row)


def multi_thread_validate_proxy(name, iph, rows):
    for row in rows:
        key = proxy_key(row)
        result = valid_proxy(row_to_proxy_obj(row))
        print(name + ' -- > ' + str(row) + (' ok ' if result else ' fail '))
        if result:
            iph.valid_ip_list[key] = row
        else:
            iph.fail_ip_list[key] = row


def xichi_ip_proxy():
    result = []
    # url = 'http://www.baidu.com'
    for i in range(1, 20):
        url = 'http://www.xicidaili.com/nn/' + str(i)
        xpath = '//table[@id="ip_list"]/tr'
        url_opener = build_opener(use_proxy=False)
        headers_dict = {}
        headers_dict[
            'User-Agent'] = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Mobile Safari/537.36'
        req = build_request(url, headers_dict=headers_dict)
        result_html = read_response_result(fetch_response(url_opener, req))
        # print(result_html)
        dom_array = dom_xpath_array(result_html, xpath)
        i = 0
        for el in dom_array:
            if i == 0:
                i += 1
                continue
            el = etree.fromstring(etree.tostring(el))
            # print(etree.tostring(el.xpath('//td[position()=2]')[0]))
            ip = el.xpath('//td[position()=2]')[0].text
            port = el.xpath('//td[position()=3]')[0].text
            ptype = el.xpath('//td[position()=6]')[0].text
            result.append([ip, port, 'https' if ptype == 'HTTPS' else 'http'])
            # print('%s %s %s' % (ip, port, ptype))
    return result


def valid_proxy(proxy_obj):
    url = 'https://www.baidu.com'
    if proxy_obj.get('https') is None:
        url = 'http://www.runoob.com/'
    url_opener = build_opener(use_proxy=True, proxy_obj=proxy_obj)
    headers_dict = {}
    headers_dict[
        'User-Agent'] = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Mobile Safari/537.36'
    req = build_request(url, headers_dict=headers_dict)
    try:
        resp = fetch_response(url_opener, req, timeout=10)
        # print(resp.read())
        if resp.getcode() == 200:
            return True
    except BaseException as err:
        print(err)
        return False


# xichi_ip_proxy()
# po = {'http': 'http://49.64.186.185:30919'}
# print(valid_proxy(po))
iph = IpProxyHolder()
# iph.fetch_new()
iph.load('/Users/yaoli/lcg/ip_proxy_list.txt')
# iph.restore1('/Users/yaoli/lcg/ip_proxy_list.txt')
iph.print()
iph.multi_validate()
print(len(iph.valid_ip_list))
print(len(iph.fail_ip_list))
