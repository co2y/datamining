import pandas as pd
import numpy as np
import csv
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from scipy.optimize import fmin


# 用于计算quadratic_weighted_kappa
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


# 用于计算quadratic_weighted_kappa
def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


# quadratic_weighted_kappa 比赛指标,用于求最优分界点
def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))
    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)
    numerator = 0.0
    denominator = 0.0
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 1.0 - numerator / denominator


# 预处理,独热编码One-Hot Encoding,将类型变量和离散变量进行0,1编码
def pre_process(data, columns, replace=False):
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in columns)
    vec_data = pd.DataFrame(vec.fit_transform(data[columns].apply(mkdict, axis=1)).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = data.index
    if replace is True:
        data = data.drop(columns, axis=1)
        data = data.join(vec_data)
    return data


# 用于求局部最优分界点的函数,根据训练集配合fmin使用,目标是使得分最高
def dist_func(x):
    length = len(y_train)
    result_list = []
    for i in range(length):
        flag = -99999
        if min_x <= yy_train_bst[i] <= x[0]:
            flag = 1
        elif x[0] <= yy_train_bst[i] < x[1]:
            flag = 2
        elif x[1] <= yy_train_bst[i] < x[2]:
            flag = 3
        elif x[2] <= yy_train_bst[i] < x[3]:
            flag = 4
        elif x[3] <= yy_train_bst[i] < x[4]:
            flag = 5
        elif x[4] <= yy_train_bst[i] < x[5]:
            flag = 6
        elif x[5] <= yy_train_bst[i] < x[6]:
            flag = 7
        elif x[6] <= yy_train_bst[i] <= max_x:
            flag = 8
        result_list.append(flag)
    return 1 - quadratic_weighted_kappa(y_train, result_list)


# 获得结果函数,根据最优分界点来对测试集的回归结果进行分类
def get_result(x, y):
    if x < y[0]:
        return 1
    elif y[0] <= x < y[1]:
        return 2
    elif y[1] <= x < y[2]:
        return 3
    elif y[2] <= x < y[3]:
        return 4
    elif y[3] <= x < y[4]:
        return 5
    elif y[4] <= x < y[5]:
        return 6
    elif y[5] <= x < y[6]:
        return 7
    else:
        return 8


if __name__ == '__main__':
    # 读数据
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # 类别变量和离散变量
    cat_dis_variables = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
                         'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
                         'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5',
                         'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2',
                         'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8',
                         'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3',
                         'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7',
                         'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11',
                         'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16',
                         'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20',
                         'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25',
                         'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29',
                         'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34',
                         'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38',
                         'Medical_History_39', 'Medical_History_40', 'Medical_History_41', 'Medical_History_1',
                         'Medical_History_15', 'Medical_History_24', 'Medical_History_32']

    # 独热编码
    train_ohd = pre_process(train, cat_dis_variables, replace=True)
    test_ohd = pre_process(test, cat_dis_variables, replace=True)

    # 去掉id和response,填充缺失数据
    features = train_ohd.columns.tolist()
    features.remove("Id")
    features.remove("Response")
    train_features = train_ohd[features]
    test_features = test_ohd[features]
    # train_features = train_features.fillna(-9999)
    # test_features = test_features.fillna(-9999)
    print(train_features)

    # 训练集
    y_train = train["Response"].values
    print(y_train)

    # 参数
    param = {'max_depth': 5, 'eta': 0.1, 'silent': 1, 'min_child_weight': 3, 'subsample': 0.7, 'seed': 15,
             "early_stopping_rounds": 10, "objective": "count:poisson", 'eval_metric': 'rmse', 'colsample_bytree': 0.65}
    num_round = 700

    # 开始训练
    dtrain = xgb.DMatrix(train_features, label=y_train, missing=np.NaN)
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    print("Training the model")

    # 根据分类器测试训练集获取最优分界点
    yy_train_bst = bst.predict(dtrain)
    print(yy_train_bst)
    min_x = min(yy_train_bst)
    max_x = max(yy_train_bst)
    x0 = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    xmin0 = fmin(dist_func, x0)
    print(xmin0)

    # 预测测试集
    dtest = xgb.DMatrix(test_features, missing=np.NaN)
    y_test_bst = bst.predict(dtest)

    # 根据最优分界点获得预测结果
    y_test_bst_result = [get_result(y, xmin0) for y in y_test_bst]

    # 输出csv文件
    ids = test.Id.values.tolist()
    n_ids = len(ids)
    prediction_file = open("result.csv", "w+")
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["Id", "Response"])
    for ii in range(0, n_ids):
        prediction_file_object.writerow([ids[ii], y_test_bst_result[ii]])
