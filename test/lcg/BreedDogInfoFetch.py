import json
import os
import ssl
import time
from urllib import request

ssl._create_default_https_context = ssl._create_unverified_context


class FilterCondition(object):
    def __init__(self):
        self.__gen_type__()
        pass

    def __gen_type__(self):
        self.indx = 0
        self._type = []
        dim_1 = ['0', '1', '2', '3', '4', '5']
        dim_3 = ['0-1', '1440-1441', '2880-2881', '5760-5761', '8640-8641', '11520-11521', '14400-14401', '17280-17281']
        dim_5 = ['0-1000', '1000-2000-1441', '2000-3000-2881', '3000-5000-5761', '5000-10000-8641', '10000-x']
        dim_6 = ['0', '1']

        jd = {}
        for d1 in dim_1:
            for d3 in dim_3:
                for d5 in dim_5:
                    for d6 in dim_6:
                        jd['1'] = d1
                        jd['3'] = d3
                        jd['5'] = d5
                        jd['6'] = d6
                        self._type.append(json.dumps(jd))

    def __gen_breed_type__(self):
        self.breed_indx = 0
        self._breed_type = []
        dim_1 = ['0', '1', '2', '3', '4', '5']
        dim_3 = ['0-1', '1440-1441', '2880-2881', '5760-5761', '8640-8641', '11520-11521', '14400-14401', '17280-17281']
        dim_4 = ['0-1000', '1000-2000-1441', '2000-3000-2881', '3000-5000-5761', '5000-10000-8641', '10000-x']

        jd = {}
        for d1 in dim_1:
            for d3 in dim_3:
                for d4 in dim_4:
                    jd['1'] = d1
                    jd['3'] = d3
                    jd['4'] = d4
                    self._breed_type.append(json.dumps(jd))

    def next_type(self):
        if self.indx == len(self._type):
            self.indx = 0
        else:
            self.indx += 1
        return self._type[self.indx]

    def next_breed_type(self):
        if self.breed_indx == len(self.breed_indx):
            self.breed_indx = 0
        else:
            self.breed_indx += 1
        return self._breed_type[self.breed_indx]

    def get_all(self):
        return self._type


def get_breed_list(page_no=1, page_size=10, filter_str=None):
    url = 'https://pet-chain.baidu.com/data/market/breed/pets'
    post_data = {"pageNo": page_no, "pageSize": page_size, "querySortType": "CREATETIME_DESC",
                 "petIds": ["1898052082261091213", "2015373030965946228", "2006547388568224593", "2015375951543759107",
                            "1872732597214615937", "2015385228673277532", "2000517941679259200", "2015380899346175726",
                            "2000527493686611014", "1922937328933371507"], "lastAmount": "366.00",
                 "lastRareDegree": None,
                 "filterCondition": filter_str,
                 # "filterCondition": "{\"1\":\"1\",\"3\":\"2880 - 2881\",\"4\":\"2000 - 3000\"}",
                 "appId": 1, "tpl": "", "type": 1, "requestId": 1522913482922, "timeStamp": None, "nounce": None,
                 "token": None}

    headers = {'Content-Type': 'application/json'}
    result = fetch_post_result(url, headers, post_data)
    jsr = json.loads(result)
    lr = []
    for pet in jsr['data']['pets4Breed']:
        lr.append(pet['petId'])
    return lr, jsr['data']['hasData']


def get_detail(petId):
    url = 'https://pet-chain.baidu.com/data/pet/queryPetById'
    post_data = {
        'appId': 1,
        'nounce': None,
        'petId': petId,
        'requestId': 1522942205543,
        'timeStamp': None,
        'token': None,
        'tpl': ""}
    headers = {'Content-Type': 'application/json'}
    return fetch_post_result(url, headers, post_data)


def fetch_post_result(post_url, headers_dict, data_dict):
    data = bytes(json.dumps(data_dict), 'utf8')
    req = request.Request(url=post_url, headers=headers_dict, data=data)
    with request.urlopen(req) as f:
        return f.read().decode('utf-8')


def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


def save_pet(base_path, petId, result):
    dr = base_path + '/' + petId[0:3]
    if not os.path.exists(dr):
        os.mkdir(dr)
    fp = dr + '/' + petId + '.txt'
    if not os.path.exists(fp):
        save_to_file(fp, result)


def process(base_path, pageSize=10, runtimes=-1):
    if runtimes == -1:
        runtimes = 10000000;
    fc = FilterCondition()
    filter_str = fc.next_type()
    i = 1
    c = 1
    while True:
        try:
            pet_list, has_data = get_breed_list(i, pageSize, filter_str=filter_str)
            for pet_id in pet_list:
                print('--> ' + pet_id)
                pet_info = get_detail(pet_id)
                save_pet(base_path, pet_id, pet_info)
            if not has_data:
                filter_str = fc.next_type()
                i = 1
        except BaseException as e:
            print(str(e))
            i = 1
            print('occurs error, sleep 5 s')
            time.sleep(5)

        if i % 5 == 0:
            print('>>>>>>>>> COUNT :  ' + str(count_file('D:/PetDog/SRC/')))
            time.sleep(5)
        if i % 100 == 0:
            print('step ' + str(i) + ' ,sleep 5 s')
            time.sleep(5)

        # if i == 50:
        #     filter_str = fc.next_type()
        #     i = 0

        i += 1
        if len(pet_list) < pageSize:
            i = 1
        c += 1
        if c >= runtimes:
            break
        print('===============>  ' + str(c) + ' : ' + str(i))


def count_file(bast_path):
    sub_path = os.listdir(bast_path)
    summary = 0
    for sp in sub_path:
        child = os.path.join('%s%s%s' % (bast_path, '/', sp))
        if os.path.isfile(child):
            summary += 1
        else:
            summary += len(os.listdir(child))
    return summary

    # print(filter_condition().get_all())


process('D:/PetDog/SRC/', runtimes=-1)
# print(count_file('D:/PetDog/SRC/'))
# get_list(pageNo=2, pageSize=10)
# print(get_detail('1922937328933371507'))
# ssl._create_default_https_context = ssl._create_unverified_context
# url = "https://tieba.baidu.com"
# html = getHtml(url)
# imglist = getImg(html)
#
# print(imglist)
