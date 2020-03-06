from tianchi.o2o.data_manager import DataManager
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from common.my_decorator import *

# 强制关闭setingwichcopywarning警告
pd.set_option('mode.chained_assignment', None)

FORMAT = '*' * 20

# @caltime_p1('pca特征降维')
def pca_dimen_reduce(train_df):
    pca = PCA(n_components=0.9)
    data_pca = pca.fit_transform(train_df)
    return data_pca


@caltime_p4('模型训练')
def train_process(x_train, y_train, x_test, y_test):
    svm = SVC()
    svm.fit(x_train, y_train)
    y_predict = svm.predict(x_test)
    accuracy = svm.score(x_test, y_test)
    print(FORMAT,'准确率 accuracy=', accuracy)
    auc = roc_auc_score(y_test, y_predict)
    print(FORMAT, 'auc=', auc)
    # print(x_train.head(50))


@caltime_p0('项目整体运行')
def o2o_train():
    data_manager = DataManager()
    data_manager.handle_data()
    print('pca 特征降维')
    data_pca = pca_dimen_reduce(data_manager.get_train_df())
    data_manager.train_test_split(data_pca)
    data_manager.print_traindf_info()
    train_process(data_manager.x_train, data_manager.y_train, data_manager.x_test, data_manager.y_test)
    data_manager.__str__()


def main():
    o2o_train()


if __name__ == '__main__':
    main()

# 部分数据归一化

# pca 特征压缩

# svm 尝试训练

# roc 获取数据（画图）

# Coupon_id 中竟然含有 fixed ，没天理
