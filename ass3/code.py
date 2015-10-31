import random
from functools import reduce
import numpy
import heapq
import sys
import time


# 读数据
def load_data(filename):
    data_list = []
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line:
            data_list.append([float(each) for each in line.split(',')])
            line = f.readline().strip()
    return data_list


# 去掉最后一个属性数据并转换成array方便矩阵运算
def init(data_list):
    return numpy.array([each[:-1] for each in data_list])


# 两个向量的距离
def distance(v1, v2):
    # return numpy.sum(numpy.power(v1 - v2, 2))
    return numpy.linalg.norm(v1 - v2)


def kmeans(data_set, k):
    length = data_set.shape[0]
    width = data_set.shape[1]
    belong_to_set = numpy.empty([length, 1])
    quota = 0
    center_set = numpy.empty([k, width])
    for i in range(k):
        index = int(random.random() * length)
        center_set[i] = data_set[index]
    change = True
    while change:
        change = False
        quota = 0
        for i in range(length):
            min_distance = float('inf')
            for j in range(k):
                temp_distance = distance(data_set[i], center_set[j])
                quota += temp_distance
                if temp_distance < min_distance:
                    min_distance = temp_distance
                    min_index = j
            if belong_to_set[i] != min_index:
                belong_to_set[i] = min_index
                change = True
        for j in range(k):
            in_cluster = [data_set[index] for index in range(length) if belong_to_set[index] == j]
            in_cluster_number = len(in_cluster)
            new_center = [each / in_cluster_number for each in reduce(add, in_cluster)]
            center_set[j] = new_center
    return belong_to_set, quota


# reduce用到的两两相加
def add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def nmf(data_set, k):
    length = data_set.shape[0]
    width = data_set.shape[1]
    x_mat = numpy.transpose(data_set)
    u_mat = numpy.random.rand(width, k)
    v_mat = numpy.random.rand(length, k)
    count = 100
    while count:
        count -= 1
        u_mat = u_mat * (numpy.dot(x_mat, v_mat) / numpy.dot(numpy.dot(u_mat, numpy.transpose(v_mat)), v_mat))
        v_mat = v_mat * (
            numpy.dot(numpy.transpose(x_mat), u_mat) / numpy.dot(numpy.dot(v_mat, numpy.transpose(u_mat)), u_mat))
        for i in range(width):
            for j in range(k):
                if u_mat[i, j] == 0:
                    u_mat[i, j] = 1e-10
        for i in range(length):
            for j in range(k):
                if v_mat[i, j] == 0:
                    v_mat[i, j] = 1e-10
    quota = distance(x_mat, numpy.dot(u_mat, numpy.transpose(v_mat)))
    belong_to_set = numpy.empty([length, 1])
    for i in range(length):
        temp_list = list(v_mat[i])
        belong_to_set[i] = temp_list.index(max(temp_list))
    return belong_to_set, quota


def get_graph(data_set, n):
    length = len(data_set)
    distance_mat = numpy.zeros([length, length])
    graph = numpy.zeros([length, length])
    for i in range(length):
        for j in range(i, length):
            distance_mat[i, j] = distance(data_set[i], data_set[j])
            distance_mat[j, i] = distance_mat[i, j]
    for i in range(length):
        n_min_list = heapq.nsmallest(n, distance_mat[i])
        for j in range(n):
            j_index = list(distance_mat[i]).index(n_min_list[j])
            graph[i, j_index] = 1
            graph[j_index, i] = 1
    return graph


def spectral_cluster(w_mat, k):
    length = w_mat.shape[0]
    d_mat = numpy.zeros([length, length])
    for i in range(length):
        d_mat[i, i] = numpy.sum(w_mat[i])
    l_mat = d_mat - w_mat
    v, q = numpy.linalg.eig(l_mat)
    k_min_list = heapq.nsmallest(k, [each for each in v if each != 0])
    q_mat = numpy.zeros([length, k])
    for i, j in zip(range(k), k_min_list):
        q_mat[:, i] = q[:, list(v).index(j).real]
    result, quota = kmeans(q_mat, k)
    return result, quota


def print_result(quota, result, raw_data, k):
    length = len(raw_data)
    mij = dict()
    for i in range(length):
        raw_class = str(raw_data[i][-1])
        if raw_class not in mij:
            mij[raw_class] = dict()
        if len(mij) == k:
            break
    for i in range(length):
        raw_class = str(raw_data[i][-1])
        now_class = str(result[i, 0])
        if now_class not in mij[raw_class]:
            mij[raw_class][now_class] = 1
        else:
            mij[raw_class][now_class] += 1
    pj_sum = 0
    for each in mij:
        pj_sum += max(mij[each].items(), key=lambda x: x[1])[1]
    purity = pj_sum / length
    gj_sum = 0
    for i in mij:
        mj = sum(mij[i].values())
        gj = 0
        for j in mij[i]:
            gj += (mij[i][j] / mj) ** 2
        gj_sum += (1 - gj) * mj
    gini = gj_sum / length
    print("quota:", quota, "\tpurity:", purity, "\tgini index:", gini)


if __name__ == '__main__':
    _t1 = time.time()
    if len(sys.argv) < 3:
        print('wrong arguments')
        exit(1)
    _filename = sys.argv[1]
    _k = 2
    if _filename == 'german.txt':
        _k = 2
    elif _filename == 'mnist.txt':
        _k = 10
    else:
        print('wrong filename')
        exit(2)
    _raw_data = load_data(_filename)
    _data_array = init(_raw_data)
    _function = sys.argv[2]
    _t2 = time.time()
    if _function == 'kmeans':
        _result, _quota = kmeans(_data_array, _k)
        _t3 = time.time()
        print_result(_quota, _result, _raw_data, _k)
        _t4 = time.time()
        print(_t2 - _t1, _t3 - _t2, _t4 - _t3)
    elif _function == 'nmf':
        _result, _quota = nmf(_data_array, _k)
        _t3 = time.time()
        print_result(_quota, _result, _raw_data, _k)
        _t4 = time.time()
        print(_t2 - _t1, _t3 - _t2, _t4 - _t3)
    elif _function == 'spectral' and len(sys.argv) == 4:
        _n = sys.argv[3]
        if _n != '3' and _n != '6' and _n != '9':
            print('wrong spectral clustering n')
            exit(3)
        _result, _quota = spectral_cluster(get_graph(_data_array, int(_n)), _k)
        _t3 = time.time()
        print_result(_quota, _result, _raw_data, _k)
        _t4 = time.time()
        print(_t2 - _t1, _t3 - _t2, _t4 - _t3)
    else:
        print('wrong cluster function')
        exit(4)
