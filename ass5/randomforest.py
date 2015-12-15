from decisiontree import *
from math import sqrt
import random
import numpy
import sys


def get_random_forest(data_list, label_list, attribute_list, whole_feature_list):
    num_of_attribute = len(attribute_list)
    m = int(sqrt(num_of_attribute))
    rf = []
    n = len(data_list)
    # label_list_copy = label_list[:]
    # t = int(num_of_attribute / m)
    t = 10
    for i in range(t):
        sub_data_list = []
        sub_label_list = ['-1'] * num_of_attribute
        sub_attribute_list = ['-1'] * num_of_attribute
        sub_whole_feature_list = ['-1'] * num_of_attribute
        # label_list_copy = label_list[:]
        for j in range(m):
            label_list_copy = label_list[:]
            label = random.choice(label_list_copy)
            index = label_list.index(label)
            label_list_copy.remove(label)
            sub_label_list[index] = label_list[index]
            sub_attribute_list[index] = attribute_list[index]
            sub_whole_feature_list[index] = whole_feature_list[index]

        for k in range(n):
            data = random.choice(data_list)
            sub_data_list.append(data)

        dt = create_decision_tree(sub_data_list, sub_label_list, sub_attribute_list, sub_whole_feature_list)
        rf.append(dt)
    return rf


def rf_get_accuracy(rf, data_list, label_list, attribute_list):
    num_of_data = len(data_list)
    num_of_correct = 0
    for each_data in data_list:
        vote_dict = dict()
        for each_dt in rf:
            label = classify(each_dt, label_list, each_data, attribute_list)
            if label not in vote_dict:
                vote_dict[label] = 0
            vote_dict[label] += 1
            # print(vote_dict)
        if each_data[-1] == sorted(vote_dict.items(), key=lambda xx: xx[1], reverse=True)[0][0]:
            num_of_correct += 1
    return num_of_correct / num_of_data


def cross_validation():
    attribute_list, data_list = load_data(sys.argv[1])  # breast-cancer-assignment5.txt german-assignment5.txt
    whole_feature_list = get_all_feature(data_list)
    label_list = [str(i) for i in range(len(attribute_list))]
    n = 10
    num_of_data = len(data_list)
    ans_list = []
    for i in range(n):
        train_data_list = random.sample(data_list, int((n - 1) * num_of_data / n))
        random_forest = get_random_forest(train_data_list, label_list, attribute_list, whole_feature_list)
        test_data_list = [each_data for each_data in data_list if each_data not in train_data_list]
        ans = rf_get_accuracy(random_forest, test_data_list, label_list, attribute_list)
        ans_list.append(ans)
    return ans_list


if __name__ == '__main__':
    # attributeList, dataList = load_data('german-assignment5.txt')
    # wholeFeatureList = get_all_feature(dataList)
    # labelList = [str(i) for i in range(len(attributeList))]
    # randomForest = get_random_forest(dataList, labelList, attributeList, wholeFeatureList)
    # print(randomForest)
    # ans = rf_get_accuracy(randomForest, dataList, labelList, attributeList)
    # print(ans)
    ansList = cross_validation()
    print(ansList)
    print(numpy.average(ansList), numpy.std(ansList))
