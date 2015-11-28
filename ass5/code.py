from math import log


def load_data(filename):
    data_list = []
    with open(filename, 'r') as f:
        line = f.readline().strip()
        attribute_list = line.split(',')
        line = f.readline().strip()
        while line:
            data_list.append(line.split(','))
            line = f.readline().strip()
    return attribute_list, data_list


def calculate_entropy(data_list):
    num_of_data = len(data_list)
    class_dict = {}
    for each_data in data_list:
        current_class = each_data[-1]
        if current_class not in class_dict:
            class_dict[current_class] = 0
        class_dict[current_class] += 1
    entropy = 0.0
    for class_key in class_dict:
        p = float(class_dict[class_key]) / num_of_data
        entropy -= p * log(p, 2)
    return entropy


def eq_reduce_data_list(data_list, index, value):
    ret_data_list = []
    for each_data in data_list:
        if each_data[index] == value:
            reduce_data = each_data[:index]
            reduce_data.extend(each_data[index + 1:])
            ret_data_list.append(reduce_data)
    return ret_data_list


def le_reduce_data_list(data_list, index, value):
    ret_data_list = []
    for each_data in data_list:
        if float(each_data[index]) <= value:
            reduce_data = each_data[:index]
            reduce_data.extend(each_data[index + 1:])
            ret_data_list.append(reduce_data)
    return ret_data_list


def g_reduce_data_list(data_list, index, value):
    ret_data_list = []
    for each_data in data_list:
        if float(each_data[index]) > value:
            reduce_data = each_data[:index]
            reduce_data.extend(each_data[index + 1:])
            ret_data_list.append(reduce_data)
    return ret_data_list


def get_split_attribute(data_list, attribute_list):
    max_info_gain_ratio = -2
    split_index = -1
    whole_entropy = calculate_entropy(data_list)
    num_of_attribute = len(data_list[0]) - 1
    num_of_data = len(data_list)
    flag = 0
    numeric_attribute_dict = {}
    numeric_attribute = 0
    hv = 0

    for i in range(num_of_attribute):
        # 离散
        if attribute_list[i] == '1':
            feature_set = set([each_data[i] for each_data in data_list])
            split_entropy = 0
            for each_feature in feature_set:
                sub_data_list = eq_reduce_data_list(data_list, i, each_feature)
                sub_data_list_length = len(sub_data_list)
                sub_data_list_entropy = calculate_entropy(sub_data_list)
                split_entropy += sub_data_list_entropy * sub_data_list_length / num_of_data
                hv -= (sub_data_list_length / num_of_data) * log((sub_data_list_length / num_of_data), 2)
                if hv == 0:
                    hv = -1
        # 连续
        elif attribute_list[i] == '0':
            sorted_data_list = sorted(data_list, key=lambda xx: xx[i])
            feature_list = []
            for j in range(num_of_data - 1):
                if sorted_data_list[j][-1] != sorted_data_list[j + 1][-1]:
                    feature_list.append(sorted_data_list[j][i])
            feature_set = set(feature_list)
            inner_max_info_gain = -1
            split_entropy = 0
            for each_feature in feature_set:
                left_data = [data for data in sorted_data_list if float(data[i]) <= float(each_feature)]
                right_data = [data for data in sorted_data_list if float(data[i]) > float(each_feature)]
                left_entropy = calculate_entropy(left_data)
                right_entropy = calculate_entropy(right_data)
                split_entropy = (left_entropy * len(left_data) + right_entropy * len(right_data)) / num_of_data
                if whole_entropy - split_entropy > inner_max_info_gain:
                    inner_max_info_gain = whole_entropy - split_entropy
                    numeric_attribute_dict[str(i)] = each_feature
                    if len(right_data) == 0:
                        hv = -1
                    else:
                        hv = -(len(left_data) / num_of_data) * log((len(left_data) / num_of_data), 2) - \
                             (len(right_data) / num_of_data) * log((len(right_data) / num_of_data), 2)
        else:
            exit(-1)

        if (whole_entropy - split_entropy) / hv > max_info_gain_ratio:
            max_info_gain_ratio = whole_entropy - split_entropy
            split_index = i
            if attribute_list[i] == '0':
                flag = 1
                numeric_attribute = numeric_attribute_dict[str(i)]
            else:
                flag = 0
        else:
            continue
    return split_index, flag, numeric_attribute


def majority_vote(class_list):
    class_dict = {}
    for each_class in class_list:
        if each_class not in class_dict:
            class_dict[each_class] = 0
        class_dict[each_class] += 1
    sorted_class_dict = sorted(class_dict.items(), key=lambda xx: xx[1], reverse=True)
    return sorted_class_dict[0][0]


def create_decision_tree(data_list, label_list, attribute_list, whole_attribute_list):
    class_list = [each_data[-1] for each_data in data_list]

    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(data_list[0]) == 1:
        return majority_vote(class_list)

    split_index, flag, numeric_attribute = get_split_attribute(data_list, attribute_list)
    split_label = label_list[split_index]
    tree = {split_label: {}}
    del label_list[split_index]
    del attribute_list[split_index]

    if flag == 0:
        feature_list = [each_data[split_index] for each_data in data_list]
        feature_set = set(feature_list)
        for each_feature in (whole_attribute_list[split_index] - feature_set):
            tree[split_label][each_feature] = majority_vote(class_list)
        del whole_attribute_list[split_index]
        for each_feature in feature_set:
            sub_label_list = label_list[:]
            sub_attribute_list = attribute_list[:]
            sub_whole_attribute_list = whole_attribute_list[:]
            sub_data_list = eq_reduce_data_list(data_list, split_index, each_feature)
            if not sub_data_list:
                tree[split_label][each_feature] = majority_vote(class_list)
            else:
                tree[split_label][each_feature] = create_decision_tree(sub_data_list, sub_label_list,
                                                                       sub_attribute_list, sub_whole_attribute_list)
    elif flag == 1:
        del whole_attribute_list[split_index]
        feature_set = ('le' + numeric_attribute, 'g' + numeric_attribute)
        for each_feature in feature_set:
            if each_feature.startswith('le'):
                sub_label_list = label_list[:]
                sub_attribute_list = attribute_list[:]
                sub_whole_attribute_list = whole_attribute_list[:]
                value = float(each_feature.split('le')[1])
                sub_data_list = le_reduce_data_list(data_list, split_index, value)
                if not sub_data_list:
                    tree[split_label][each_feature] = majority_vote(class_list)
                else:
                    tree[split_label][each_feature] = create_decision_tree(sub_data_list, sub_label_list,
                                                                           sub_attribute_list, sub_whole_attribute_list)
            else:
                sub_label_list = label_list[:]
                sub_attribute_list = attribute_list[:]
                sub_whole_attribute_list = whole_attribute_list[:]
                value = float(each_feature.split('g')[1])
                sub_data_list = g_reduce_data_list(data_list, split_index, value)
                if not sub_data_list:
                    tree[split_label][each_feature] = majority_vote(class_list)
                else:
                    tree[split_label][each_feature] = create_decision_tree(sub_data_list, sub_label_list,
                                                                           sub_attribute_list, sub_whole_attribute_list)
    else:
        exit(-1)

    return tree


def classify(tree, label_list, test_data, attribute_list):
    label = tuple(tree.keys())[0]
    feature_dict = tree[label]
    index = label_list.index(label)
    while 1:
        if label == 'A' or label == 'B':
            return label
        else:
            if attribute_list[index] == '1':
                for each_feature in feature_dict:
                    if each_feature == test_data[index]:
                        label = feature_dict[each_feature]
                        if type(label).__name__ == 'dict':
                            label = tuple(label.keys())[0]
                            feature_dict = feature_dict[each_feature][label]
                            index = label_list.index(label)
                        break
            else:
                for each_feature in feature_dict:
                    if each_feature.startswith('le'):
                        value = float(each_feature.split('le')[1])
                        if float(test_data[index]) <= value:
                            label = feature_dict[each_feature]
                            if type(label).__name__ == 'dict':
                                label = tuple(label.keys())[0]
                                feature_dict = feature_dict[each_feature][label]
                                index = label_list.index(label)
                            break
                    else:
                        value = float(each_feature.split('g')[1])
                        if float(test_data[index]) > value:
                            label = feature_dict[each_feature]
                            if type(label).__name__ == 'dict':
                                label = tuple(label.keys())[0]
                                feature_dict = feature_dict[each_feature][label]
                                index = label_list.index(label)
                            break


def get_all_feature(data_list):
    num_of_feature = len(data_list[0]) - 1
    whole_feature_list = []
    for i in range(num_of_feature):
        whole_feature_list.append(set(each_data[i] for each_data in data_list))
    return whole_feature_list


if __name__ == '__main__':
    attributeList, dataList = load_data('german-assignment5.txt')
    attributeCopyList = attributeList[:]
    wholeFeatureList = get_all_feature(dataList)
    print('whole feature: ', wholeFeatureList)
    labelList = [str(i) for i in range(len(attributeList))]
    labelCopyList = labelList[:]
    decisionTree = create_decision_tree(dataList[:], labelList, attributeList, wholeFeatureList)
    print('decision tree: ', decisionTree)
    ansCorrect = 0
    for eachTestData in dataList[:]:
        classLabel = classify(decisionTree, labelCopyList, eachTestData[:-1], attributeCopyList)
        if classLabel == eachTestData[-1]:
            ansCorrect += 1
    print('correct number: ', ansCorrect)
