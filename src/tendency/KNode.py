# coding=utf-8
'''
K线数据分析
'''

import math
from enum import Enum


def parse_directory(first_val, last_val):
    return TendencyDirectory.upward if first_val > last_val else TendencyDirectory.downward


class TendencyDirectory(Enum):
    upward = 1
    downward = -1
    none = 0


def get_log(src_value, use_log, base):
    if use_log and src_value > 0:
        return math.log(src_value, base)
    return src_value


def max_val(value_a, value_b):
    return value_a if value_a > value_b else value_b


def min_val(value_a, value_b):
    return value_a if value_a < value_b else value_b


def gen_nl_summary(node_list):
    result = NLSummary()
    if node_list and len(node_list) > 0:
        result.count = len(node_list)
        result.start_dt = node_list[0].dt
        result.end_dt = node_list[result.count - 1].dt

        for node in node_list:
            result.sum_amount += node.amount
            result.sum_exchange_rate += node.exchange_rate
            result.sum_exchange_amount += node.exchange_amount

            result.hi_cp = max_val(result.hi_cp, node.cp)
            result.hi_pr = max_val(result.hi_pr, node.hp)
            result.hi_amount = max_val(result.hi_amount, node.amount)
            result.hi_exchange_amount = max_val(result.hi_exchange_amount, node.exchange_amount)
            result.hi_exchange_rate = max_val(result.hi_exchange_rate, node.exchange_rate)

            result.low_cp = min_val(result.low_cp, node.cp)
            result.low_pr = min_val(result.low_pr, node.hp)
            result.low_amount = min_val(result.low_amount, node.amount)
            result.low_exchange_amount = min_val(result.low_exchange_amount, node.exchange_amount)
            result.low_exchange_rate = min_val(result.low_exchange_rate, node.exchange_rate)

        result.avg_amount = result.sum_amount / result.count
        result.avg_exchange_amount = result.sum_exchange_amount / result.count
        result.avg_exchange_rate = result.sum_exchange_rate / result.count
        result.avg_pr = (result.sum_amount / result.sum_exchange_amount) if result.sum_exchange_amount > 0 else 0

    return result


class NLSummary(object):
    def __init__(self):
        self.count = 0
        self.start_dt = None
        self.end_dt = None

        self.hi_cp = 0.0
        self.hi_pr = 0.0
        self.hi_amount = 0.0
        self.hi_exchange_amount = 0
        self.hi_exchange_rate = 0.0

        self.low_cp = float('inf')
        self.low_pr = float('inf')
        self.low_amount = float('inf')
        self.low_exchange_amount = float('inf')
        self.low_exchange_rate = float('inf')

        self.avg_pr = 0.0
        self.avg_amount = 0.0
        self.avg_exchange_amount = 0
        self.avg_exchange_rate = 0.0

        self.sum_amount = 0.0
        self.sum_exchange_amount = 0
        self.sum_exchange_rate = 0.0

    def __str__(self):
        return "summary --> count : " + str(self.count) + " dt:[" + str(self.start_dt) + " , " + str(
            self.end_dt) + "] 均价:" + str(self.avg_pr) + " 平均成交额:" + str(self.avg_amount) + " 平均成交量:" + str(
            self.avg_exchange_amount) + "平均成交价格:" + str(self.avg_exchange_rate) + " 成交总额:" + str(
            self.sum_amount) + " 成交总量:" + str(self.sum_exchange_amount) + "\n" + "high -->   最高收盘价:" + str(
            self.hi_cp) + " 最高成交价:" + str(self.hi_pr) + " 最大成交额:" + str(self.hi_amount) + " 最高成交量:" + str(
            self.hi_exchange_amount) + " 最高成交率:" + str(self.hi_exchange_rate) + "\n" + "low --> 最低收盘价:" + str(
            self.low_cp) + " 最低成交价:" + str(self.low_pr) + " 最低成交额:" + str(self.low_amount) + " 最低成交量:" + str(
            self.low_exchange_amount) + " 最低成交率:" + str(self.low_exchange_rate) + "\n"


class KNode(object):
    def __init__(self, dt='', cp=0.0, op=0.0, hp=0.0, lp=0.0, amount=0.0, exchange_amount=0, exchange_rate=0.0,
                 use_log=True, base=2):
        self.src_op = op
        self.src_cp = cp
        self.src_hp = hp
        self.src_lp = lp
        self.amount = amount
        self.exchange_amount = exchange_amount
        self.exchange_rate = exchange_rate
        self.dt = dt
        self.use_log = use_log
        self.op = get_log(op, use_log, base)
        self.cp = get_log(cp, use_log, base)
        self.hp = get_log(hp, use_log, base)
        self.lp = get_log(lp, use_log, base)

    def __str__(self):
        return "[" + str(self.cp) + "]"


class Wave(object):
    def __init__(self, directory=TendencyDirectory.none, base=None):
        self.list = []
        self.base = base
        self.directory = directory
        pass

    def set_directory(self, directory):
        self.directory = directory

    def set_base(self, node):
        self.base = node

    def append_node(self, node):
        self.list.append(node)

    def last_node(self):
        if len(self.list) > 0:
            return self.list[len(self.list) - 1]
        return None

    def summary(self):
        return gen_nl_summary(self.list)

    def __str__(self):
        return "wave : dir " + str(self.directory) + " base:" + str(self.base.cp) + " list : " + str_list(
            self.list) + "\n" + str(self.summary())


class Tendency(object):
    def __init__(self, directory=0, base=None):
        self.list = []
        self.wave_list = []
        self.base = base
        self.directory = directory
        pass

    def append_wave(self, wave):
        if wave:
            self.wave_list.append(wave)
            for node in wave.list:
                self.list.append(node)

    def summary(self):
        return gen_nl_summary(self.list)

    def __str__(self):
        return "tendency : dir " + str(self.directory) + " base:" + str(self.base.cp) + "\n" + str(self.summary())


class KNodeList(object):
    def __init__(self, node_list):
        self.node_list = node_list
        self.wave_list = []
        self.tendency_list = []
        self.wave_index = 0

    def parse_wave(self):
        first_node = self.node_list[0]
        cur = Wave(base=first_node)
        cur.append_node(first_node)
        self.wave_list.append(cur)

        cur_value = first_node.cp
        for i in range(len(self.node_list) - 1):
            node = self.node_list[i + 1]
            new_dir = parse_directory(cur_value, node.cp)
            if new_dir is not cur.directory:
                cur = Wave(directory=new_dir, base=cur.last_node())
                self.wave_list.append(cur)
            cur.append_node(node)
            cur_value = node.cp

        for i in range(len(self.wave_list)):
            print("-" * 20)
            print(self.wave_list[i])

    def parse_tendency(self, tendency_number, delay_number):
        search_index = 1
        while search_index < len(self.wave_list):
            td = self.find_max_tendency(search_index, tendency_number, delay_number)
            if not td:
                search_index += 1
            else:
                search_index += len(td.wave_list)
                self.tendency_list.append(td)

        for i in range(len(self.tendency_list)):
            print("-" * 20)
            print(self.tendency_list[i])

    def find_max_tendency(self, wave_from_index, tendency_number, delay_number):
        first_wave = self.wave_list[wave_from_index]
        directory = first_wave.directory
        max_value = first_wave.base.cp
        tendency_count = 0
        delay_count = 0
        stopped = 0
        max_wave_index = wave_from_index

        for i in range(wave_from_index, len(self.wave_list)):
            curr_wave = self.wave_list[i]
            for j in range(len(curr_wave.list)):
                curr_node = curr_wave.list[j]
                if parse_directory(max_value, curr_node.cp) is directory:
                    delay_count = 0
                    tendency_count += 1
                    max_wave_index = i
                    max_value = curr_node.cp
                else:
                    delay_count += 1

                if delay_count >= delay_number:
                    stopped = 1
                    break
            if stopped == 1:
                break

        td = Tendency(directory=directory, base=first_wave.base)
        for i in range(wave_from_index, max_wave_index + 1):
            td.append_wave(self.wave_list[i])

        if tendency_count >= tendency_number:
            return td

        return None


def alynize(knodeList):
    mid = []
    cur = []
    cur_dir = TendencyDirectory.none
    cur_value = knodeList[0].cp
    cur.append(knodeList[0])
    mid.append(cur)

    cur = []
    mid.append(cur)
    for i in range(len(knodeList) - 1):
        new_value = knodeList[i + 1].cp
        new_dir = parse_directory(cur_value, new_value)
        cur_value = new_value
        if new_dir is not cur_dir:
            cur = []
            mid.append(cur)
        cur_dir = new_dir
        cur.append(knodeList[i + 1])

    for i in range(len(mid)):
        print("-" * 20)
        for j in range(len(mid[i])):
            print(mid[i][j].cp)

    return mid


def str_list(list_obj):
    result = ""
    for item in list_obj:
        result += str(item)
    return result


if __name__ == '__main__':
    df = []
    df.append(KNode(cp=1, dt='20180601', op=0.0, hp=0.0, lp=0.0, amount=0.0, exchange_amount=0, exchange_rate=0.0,
                    use_log=False, base=2))
    df.append(KNode(cp=3, dt='20180602', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=8, dt='20180604', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=9, dt='20180608', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=6, dt='20180610', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=4, dt='20180611', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=2, dt='20180613', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=1, dt='20180614', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=5, dt='20180616', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    df.append(KNode(cp=4, dt='20180631', op=1.0, hp=2.0, lp=3.0, amount=1.0, exchange_amount=3, exchange_rate=3.0,
                    use_log=False, base=2))
    kl = KNodeList(df)
    kl.parse_wave()
    kl.parse_tendency(3, 2)
    # for i in range(len(kl.wave_list)):
    #     print("-" * 20)
    #     for j in range(len(kl.wave_list[i].list)):
    #         print(kl.wave_list[i].list[j].cp)

    # print(str_list(df))
