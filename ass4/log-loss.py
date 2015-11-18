import numpy
import random


def load_data(filename):
    x = list()
    y = list()
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line:
            int_list = [int(each) for each in line.split(',')]
            x.append(int_list[:-1])
            y.append(int_list[-1])
            line = f.readline().strip()
    return numpy.array(x), y


def pegasos(x, y, r, t):
    w = numpy.zeros((t + 1, x.shape[1]))
    for j in range(t):
        i = int(random.random() * x.shape[0])
        n = 1 / (r * (j + 1))
        z = numpy.inner(w[j], x[i])
        # z太大会导致exp过大使结果不准确
        if z > 500:
            w[j + 1] = (1 - n * r) * w[j]
        else:
            w[j + 1] = (1 - n * r) * w[j] + n * y[i] / (1 + numpy.exp(y[i] * z)) * x[i]
    return w[t]


def test(x, y, w):
    correct_num = 0
    for i in range(x.shape[0]):
        if numpy.inner(w, x[i]) * y[i] > 0:
            correct_num += 1
    return 1 - correct_num / x.shape[0]


if __name__ == '__main__':
    train_data_x, train_data_y = load_data('dataset1-a8a-training.txt')
    for T in range(1, 11):
        classifier = pegasos(train_data_x, train_data_y, 1E-4, round(0.5 * train_data_x.shape[0] * T))
        test_data_x, test_data_y = load_data('dataset1-a8a-testing.txt')
        result = test(test_data_x, test_data_y, classifier)
        print(result, end=' ')
    print()
    train_data_x, train_data_y = load_data('dataset1-a9a-training.txt')
    for T in range(1, 11):
        classifier = pegasos(train_data_x, train_data_y, 5E-5, round(0.5 * train_data_x.shape[0] * T))
        test_data_x, test_data_y = load_data('dataset1-a9a-testing.txt')
        result = test(test_data_x, test_data_y, classifier)
        print(result, end=' ')
