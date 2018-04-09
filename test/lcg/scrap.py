import json
import re
import ssl
from urllib import request

ssl._create_default_https_context = ssl._create_unverified_context


def get_list():
    url = 'https://pet-chain.baidu.com/data/market/queryPetsOnSale'
    post_data = {"pageNo": 199, "pageSize": 5, "querySortType": "CREATETIME_ASC",
                 "petIds": ["1898052082261091213", "2015373030965946228", "2006547388568224593", "2015375951543759107",
                            "1872732597214615937", "2015385228673277532", "2000517941679259200", "2015380899346175726",
                            "2000527493686611014", "1922937328933371507"], "lastAmount": "366.00",
                 "lastRareDegree": None,
                 # "filterCondition": "{\"1\":\"1\",\"3\":\"2880 - 2881\",\"5\":\"2000 - 3000\",\"6\":\"0\"}",
                 "appId": 1, "tpl": "", "type": 1, "requestId": 1522913482922, "timeStamp": None, "nounce": None,
                 "token": None}
    headers = {'Content-Type': 'application/json'}
    fetch_post_result(url, headers, post_data)


def get_detail():
    url = 'https://pet-chain.baidu.com/data/pet/queryPetById'
    post_data = {
        'appId': 1,
        'nounce': None,
        'petId': '2015383029650000773',
        'requestId': 1522942205543,
        'timeStamp': None,
        'token': None,
        'tpl': ""}
    headers = {'Content-Type': 'application/json'}
    fetch_post_result(url, headers, post_data)


def fetch_post_result(post_url, headers_dict, data_dict):
    data = bytes(json.dumps(data_dict), 'utf8')
    req = request.Request(url=post_url, headers=headers_dict, data=data)
    with request.urlopen(req) as f:
        # return f.read().decode('utf-8')
        print('Status:', f.status, f.reason)
        for k, v in f.getheaders():
            print('%s: %s' % (k, v))

        print('Data:', f.read().decode('utf-8'))


def test_login_weibo():
    print('Login to weibo.cn...')
    email = input('Email: ')
    passwd = input('Password: ')

    post_url = 'https://passport.weibo.cn/sso/login'
    headers_dict = {}
    data_dict = {}
    data_dict['username'] = email
    data_dict['password'] = passwd
    data_dict['entry'] = 'mweibo'
    data_dict['client_id'] = ''
    data_dict['savestate'] = '1'
    data_dict['ec'] = ''
    data_dict['pagerefer'] = 'https://passport.weibo.cn/signin/welcome?entry=mweibo&r=http%3A%2F%2Fm.weibo.cn%2F'

    headers_dict['Origin'] = 'https://passport.weibo.cn'
    headers_dict[
        'User-Agent'] = 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25'
    headers_dict['Referer'] = 'https://passport.weibo.cn/signin/welcome?entry=mweibo&r=http%3A%2F%2Fm.weibo.cn%2F'

    fetch_post_result(post_url, headers_dict, data_dict)


def getHtml(url):
    page = request.urlopen(url)
    html = page.read()
    html = html.decode('utf-8')
    return html


def getImg(html):
    reg = r'<p class="img_title">(.*)</p>'
    img_title = re.compile(reg)
    imglist = re.findall(img_title, html)
    return imglist


# test_login_weibo()
get_list()
# get_detail()
# ssl._create_default_https_context = ssl._create_unverified_context
# url = "https://tieba.baidu.com"
# html = getHtml(url)
# imglist = getImg(html)
#
# print(imglist)
