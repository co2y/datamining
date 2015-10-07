# -*- coding: utf-8 -*-
import jieba
import os
import math

SOURCE_FILE_PATH = './lily/'
STOPWORDS_FILENAME = 'Chinese-stop-words.txt'
OUTPUT_FILE_PATH = './result/'
tfMap = dict()
idfMap = dict()
each_file_number_list = list()


def get_files():
    return os.listdir(SOURCE_FILE_PATH)


def get_stopwords():
    stopwords_list = list()
    with open(STOPWORDS_FILENAME, 'r') as f:
        line = f.readline()
        while line:
            stopwords_list.append(line.strip())
            line = f.readline()
    return stopwords_list


def get_posts(files, stopwords):
    posts_list = list()
    for each_file in files:
        each_number = 0
        with open(SOURCE_FILE_PATH + each_file, 'r') as f:
            line = f.readline()
            while line:
                posts_list.append(
                    [term for term in jieba.cut(line, cut_all=False) if term.strip() not in stopwords])
                line = f.readline()
                each_number += 1
        each_file_number_list.append(each_number)
    return posts_list


def init_map(post_length):
    for i in range(post_length):
        tfMap[i] = dict()


def get_result(posts):
    post_id = 0
    for each_post in posts:
        for each in each_post:
            if each not in tfMap[post_id]:
                tfMap[post_id][each] = 1
            else:
                tfMap[post_id][each] += 1
        for key in tfMap[post_id]:
            if key not in idfMap:
                idfMap[key] = 1
            else:
                idfMap[key] += 1
        post_id += 1


def print_result():
    if not os.path.exists(OUTPUT_FILE_PATH):
        os.makedirs(OUTPUT_FILE_PATH)
    file_names = get_files()
    index_end = 0
    idf_list = list(idfMap)
    for each_file, each_post_number in zip(file_names, each_file_number_list):
        index_begin = index_end
        index_end = index_begin + each_post_number
        with open(OUTPUT_FILE_PATH + each_file, 'w') as f:
            for i in range(index_begin, index_end):
                each_line = list()
                for key in tfMap[i]:
                    each_line.append(
                        str(idf_list.index(key)) + ":" + str(
                            round(tfMap[i][key] * math.log10(post_number / idfMap[key]), 4)))
                each_line.append('\n')
                f.write(" ".join(each_line))


if __name__ == '__main__':
    all_posts = get_posts(get_files(), get_stopwords())
    post_number = all_posts.__len__()
    init_map(post_number)
    get_result(all_posts)
    print_result()