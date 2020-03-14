import sklearn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
# import xgboost as xgb
import time

"""
24题
现有如下数据：
    [[22, 18, 4, 35, 47, 11],
    [61, 33, 20, 48, 10, 25],
    [55, 74, 29, 81, 16, 27],
    [123, 54, 73, 21, 99, 83],
    [6, 55, 79, 33, 24, 63]]
请尝试使用PCA对如上数据进行降维。
要求：分别使用小数和整数给参数n_components赋值， 并查看对比最终处理的结果。
"""


def code_24():
    print('-' * 25, '24题开始', '-' * 25)
    data = np.array([[22, 18, 4, 35, 47, 11],
                     [61, 33, 20, 48, 10, 25],
                     [55, 74, 29, 81, 16, 27],
                     [123, 54, 73, 21, 99, 83],
                     [6, 55, 79, 33, 24, 63]])
    pca1 = PCA(n_components=3)
    pca2 = PCA(n_components=0.8)
    pca1_result = pca1.fit_transform(data)
    pca2_result = pca2.fit_transform(data)
    print('源数据： \n', data)
    print('源数据： ', data.shape)
    print("*" * 30)
    print('pca整数参数结果： \n', pca1_result)
    print('pca整数参数结果： ', pca1_result.shape)
    print("*" * 30)
    print('pca小数参数结果： \n', pca2_result)
    print('pca小数参数结果： ', pca2_result.shape)
    print('-' * 25, '24题结束', '-' * 25)
    pass


"""
25题
使用鸢尾花数据集训练lightGBM分类模型 
要求：
1. 使用sklearn内置的鸢尾花数据集；
2. 对数据集进行划分，验证集比例可以自定义，保证程序每次使用的数据集都是相同的；
3. 使用合适的特征预处理方法对原始数据进行处理；
4. 使用交叉验证和网格搜索对超参数进行调优(包括但不限于K值)；
5. 评估训练好的模型；
6. 获取表现最好的模型在测试集上的准确率。
7. 获取在交叉验证中表现最好的模型及其参数。

提示:
api函数: LGBMClassifier(boosting_type = 'gbdt',objective = 'multiclass', metric="multi_logloss")
下列参数任选两个进行交叉验证和网格搜索
learning_rate                 
max_depth
sub_feature
num_leaves
colsample_bytree
n_estimators
early_stop
"""


def print_info(data):
    print(data.shape)
    print(data.info())
    print(data.columns)
    print(data.head(10))


def code_25():
    print('-' * 25, '25题开始', '-' * 25)
    start_time = time.time()
    bunch = load_iris()
    data = pd.DataFrame(data=bunch.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y_traget = pd.Series(data=bunch.target)
    x_train, x_test, y_train, y_test = train_test_split(data, y_traget, train_size=0.8, random_state=25)

    # 标准化
    transfer = StandardScaler()
    transfer.fit(x_train)
    x_train = transfer.transform(x_train)
    x_test = transfer.transform(x_test)

    param_grid = {"n_estimators": [60, 80, 100, 150, 200], "learning_rate": [0.02, 0.03, 0.04, 0.05, 0.06]}
    estimator = LGBMClassifier(boosting_type='gbdt', objective='multiclass', metric="multi_logloss", max_depth=3, )
    grids_searcher = GridSearchCV(estimator, param_grid=param_grid, cv=3)
    grids_searcher.fit(x_train, y_train)
    grids_searcher.score(x_test, y_test)
    print("准确率： ", grids_searcher.best_score_)
    print("最优参数： ", grids_searcher.best_params_)  # {'learning_rate': 0.03, 'n_estimators': 150}
    end_time = time.time()
    print(f'总共耗时： {str(round((end_time - start_time), 4))} 秒')
    print('-' * 25, '25题结束', '-' * 25)


a = [1, 3, 5, 7, 9]
b = [4, 5, 6, 7]


def list_intersection():
    intersection = list(set(a).intersection(set(b)))
    print("交集： ", intersection)
    pass


def list_union():
    union = list(set(a).union(set(b)))
    print("并集： ", union)
    pass


def list_defference():
    defference = list(set(b).difference(set(a)))
    print("差集： ", defference)
    pass


def test_list():
    print('a: ', a)
    print('b: ', b)
    list_intersection()
    list_union()
    list_defference()
    pass


def main():
    # code_24()
    # code_25()
    test_list()


if __name__ == '__main__':
    main()
