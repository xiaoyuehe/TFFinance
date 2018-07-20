# coding=utf-8
'''
K线数据分析
'''


class KNode(object):
    def __init__(self, dt='', cp=0.0, op=0.0, hp=0.0, lp=0.0):
        self.op = op
        self.cp = cp
        self.hp = hp
        self.lp = lp
        self.dt = dt


class Wave(object):
    def __init__(self, directory=0, base=None):
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


class Tendency(object):
    def __init__(self, directory=0, base=None):
        self.list = []
        self.wave_list = []
        self.base = base
        self.direct = directory
        pass

    def append_wave(self, wave):
        if wave:
            self.wave_list.append(wave)
            for node in wave.list:
                self.list.append(node)


class KNodeList(object):
    def __init__(self, node_list):
        self.node_list = node_list
        self.wave_list = []
        self.tendency_list = []
        self.wave_index = 0

    def parse_wave(self):
        first_node = self.node_list[0]
        cur = Wave(directory=0, base=first_node)
        cur.append_node(first_node)
        self.wave_list.append(cur)

        cur_value = first_node.cp
        cur = Wave(directory=0, base=first_node)
        self.wave_list.append(cur)
        for i in range(len(self.node_list) - 1):
            new_value = self.node_list[i + 1].cp
            new_dir = 1 if new_value >= cur_value else -1
            cur_value = new_value
            if new_dir * cur.directory < 0:
                cur = Wave(directory=new_dir, base=cur.last_node())
                self.wave_list.append(cur)
            cur.append_node(self.node_list[i + 1])
            cur.set_directory(new_dir)

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
            for j in range(len(self.tendency_list[i].list)):
                print(self.tendency_list[i].list[j].cp)

    def find_max_tendency(self, wave_from_index, tendency_number, delay_number):
        first_wave = self.wave_list[wave_from_index]
        directory = first_wave.directory
        max_value = first_wave.base.cp
        tendency_count = 0
        delay_count = 0
        stoped = 0
        max_wave_index = wave_from_index

        for i in range(wave_from_index, len(self.wave_list)):
            curr_wave = self.wave_list[i]
            for j in range(len(curr_wave.list)):
                curr_node = curr_wave.list[j]
                if (curr_node.cp - max_value) * directory > 0:
                    delay_count = 0
                    tendency_count += 1
                    max_wave_index = i
                else:
                    delay_count += 1

                if delay_count >= delay_number:
                    stoped = 1
                    break
            if stoped == 1:
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
    cur_dir = 0
    cur_value = knodeList[0].cp
    cur.append(knodeList[0])
    mid.append(cur)

    cur = []
    mid.append(cur)
    for i in range(len(knodeList) - 1):
        new_value = knodeList[i + 1].cp
        new_dir = 1 if new_value >= cur_value else -1
        cur_value = new_value
        if new_dir * cur_dir < 0:
            cur = []
            mid.append(cur)
        cur_dir = new_dir
        cur.append(knodeList[i + 1])

    for i in range(len(mid)):
        print("-" * 20)
        for j in range(len(mid[i])):
            print(mid[i][j].cp)

    return mid


if __name__ == '__main__':
    df = []
    df.append(KNode(cp=1))
    df.append(KNode(cp=3))
    df.append(KNode(cp=8))
    df.append(KNode(cp=9))
    df.append(KNode(cp=6))
    df.append(KNode(cp=4))
    df.append(KNode(cp=2))
    df.append(KNode(cp=1))
    df.append(KNode(cp=5))
    df.append(KNode(cp=4))
    kl = KNodeList(df)
    kl.parse_wave()
    kl.parse_tendency(3, 2)
    # for i in range(len(kl.wave_list)):
    #     print("-" * 20)
    #     for j in range(len(kl.wave_list[i].list)):
    #         print(kl.wave_list[i].list[j].cp)
