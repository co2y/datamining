from decisiontree import *
import numpy
import random
import sys


def get_adaboost(data_list, label_list, attribute_list, whole_feature_list):
    n = len(data_list)
    w = [1 / n] * n
    t = 0
    t_max = 100
    a_list = []
    dt_list = []
    while 1:
        sub_label_list = label_list[:]
        sub_attribute_list = attribute_list[:]
        sub_whole_feature_list = whole_feature_list[:]
        train_data_index = numpy.random.choice([i for i in range(n)], n, p=w)
        train_data_list = [data_list[i] for i in train_data_index]
        # test_data_list = [each_data for each_data in data_list if each_data not in train_data_list]
        dt = create_decision_tree(train_data_list, sub_label_list, sub_attribute_list, sub_whole_feature_list)
        correct_index = [1] * n
        c = 0
        for test_data in data_list:
            label = classify(dt, label_list, test_data, attribute_list)
            if label != test_data[-1]:
                wrong_index = data_list.index(test_data)
                correct_index[wrong_index] = 0
                c += w[wrong_index]
        if c > 0.5:
            w = [1 / n] * n
            continue
        if c == 0:
            c = 1E-20
        a = 1 / 2 * numpy.log((1 - c) / c)
        for i in range(n):
            if correct_index[i] == 0:
                w[i] *= numpy.exp(a)
            else:
                w[i] *= numpy.exp(-a)
        w_sum = numpy.sum(w)
        for i in range(n):
            w[i] /= w_sum
        a_list.append(a)
        dt_list.append(dt)
        t += 1
        if t >= t_max:
            break
    return a_list, dt_list


def adaboost_get_accuracy(a_list, dt_list, data_list, label_list, attribute_list):
    num_of_data = len(data_list)
    num_of_correct = 0
    n = len(dt_list)
    for each_data in data_list:
        vote_dict = dict()
        for i in range(n):
            label = classify(dt_list[i], label_list, each_data, attribute_list)
            if label not in vote_dict:
                vote_dict[label] = 0
            vote_dict[label] += a_list[i]
            # print(vote_dict)
        if each_data[-1] == sorted(vote_dict.items(), key=lambda xx: xx[1], reverse=True)[0][0]:
            num_of_correct += 1
    return num_of_correct / num_of_data


def cross_validation():
    attribute_list, data_list = load_data(sys.argv[1])  # breast-cancer-assignment5.txt german-assignment5.txt
    feature_list = get_all_feature(data_list)
    label_list = [str(i) for i in range(len(attribute_list))]
    n = 10
    num_of_data = len(data_list)
    ans_list = []
    for i in range(n):
        train_data_list = random.sample(data_list, int((n - 1) * num_of_data / n))
        weight_list, decision_tree_list = get_adaboost(train_data_list, label_list, attribute_list, feature_list, )
        test_data_list = [each_data for each_data in data_list if each_data not in train_data_list]
        ans = adaboost_get_accuracy(weight_list, decision_tree_list, test_data_list, label_list, attribute_list)
        ans_list.append(ans)
    return ans_list


if __name__ == '__main__':
    ansList = cross_validation()
    print(ansList)
    print(numpy.average(ansList), numpy.std(ansList))
