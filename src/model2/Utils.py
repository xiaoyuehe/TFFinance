# coding=utf-8
"""
useful data process tools
2018年3月24日
"""

import numpy as np


def gen_noise_data(np_data, bias, count):
    result = []
    for i in range(count):
        random_bias_array = 1 - bias * np.random.random(size=np_data.size).reshape(np_data.shape)
        result.append(np_data * random_bias_array)
    return result


def big_file_shuffle(src_path, tmp_dir, tmp_count, dist_path):
    # random split file
    src = open(src_path, mode='r')
    files = []
    for i in range(tmp_count):
        files.append(open(tmp_dir + '/tmp0.' + str(i), 'w'))
    while True:
        line = src.readline()
        if not line:
            break
        partition = np.random.randint(0, tmp_count)
        files[partition].write(line)
    for f in files:
        f.close()
    src.close()
    # shuffle & merge
    for j in range(tmp_count):
        shuffle_file(tmp_dir + '/tmp0.' + str(j), dist_path, write_mode='a')


def shuffle_file(src_path, dist_path, write_mode='w'):
    src = open(src_path, mode='r')
    da = []
    while True:
        line = src.readline()
        if not line:
            break
        da.append(line)
    src.close()

    index = np.arange(len(da))
    i = 0
    while i < len(index):
        swap_index = np.random.randint(i, len(da))
        ss = index[i]
        index[i] = index[swap_index]
        index[swap_index] = ss
        i += 1
    # print(index)

    dist = open(dist_path, mode=write_mode)
    for j in index:
        dist.write(da[index[j]])
    dist.close()


# a = np.arange(15).reshape(5, 3)
# print(a)
# print('*' * 30)
# print(gen_noise_data(a, 0.1, 10))
# print(np.random.randint(3, 5))
# print(np.random.randint(3, 4))
# print(np.random.randint(3, 4))
# print(np.random.randint(3, 4))

# shuffle_file('D:/StockData/01_ORI/000001_15_17.csv', 'd:/result.csv', write_mode='a')
# big_file_shuffle('D:/StockData/11_MODEL_02/train.csv', 'D:/StockData/11_MODEL_02/tmp/', 100,
#                  'D:/StockData/11_MODEL_02/train_sh.csv')
