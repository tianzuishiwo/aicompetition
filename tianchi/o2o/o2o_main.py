from hm.o2o.data_manager import DataManager
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 强制关闭setingwichcopywarning警告
pd.set_option('mode.chained_assignment', None)


# # 所有参数配置
# OFFLINE_SPLIT_DATA = 5000
# ONLINE_SPLIT_DATA = 20000
# FORMART_EMPTY = 'empty'
# FORMART_DOT = '.'
# FORMART_COLON = ':'
# FORMART_FIXED = 'fixed'
# TRAIN_COLUMNS = ['User_id', 'Merchant_id', 'Action', 'Distance', 'day', 'hour', 'weekday',
#                  'discount_fixed', 'discount_ratio', 'discount_satisfy', 'sample_type', 'user_type']
#  TRAIN_TARGET_COLUMN = 'use_coupon_15day'


# def pca_and_train(train_data, train_target_data):
#     pca = PCA(n_components=0.9)
#     data_pca = pca.fit_transform(train_data)
#     # print(data_pca)
#     x_train, x_test, y_train, y_test = train_test_split(data_pca, train_target_data)
#
#     svm = SVC()
#     svm.fit(x_train, y_train)
#     y_predict = svm.predict(x_test)
#     accuracy = svm.score(x_test, y_test)
#     print('pca n_components=', pca.n_components, ' accuracy=', accuracy)
#     auc = roc_auc_score(y_test, y_predict)
#     print('*' * 20, 'auc=', auc)
#     print('训练数据：')
#     print(train_data.head(50))


def pca_dimen_reduce(train_df):
    pca = PCA(n_components=0.9)
    data_pca = pca.fit_transform(train_df)
    return data_pca


def train_process(x_train, y_train, x_test, y_test):
    svm = SVC()
    svm.fit(x_train, y_train)
    y_predict = svm.predict(x_test)
    accuracy = svm.score(x_test, y_test)
    print('准确率 accuracy=', accuracy)
    auc = roc_auc_score(y_test, y_predict)
    print('*' * 20, 'auc=', auc)
    print('训练数据：')
    # print(x_train.head(50))


def o2o_train():
    data_manager = DataManager()
    data_manager.handle_data()
    data_pca = pca_dimen_reduce(data_manager.get_train_df())
    data_manager.train_test_split(data_pca)
    data_manager.print_traindf_info()
    train_process(data_manager.x_train, data_manager.y_train, data_manager.x_test, data_manager.y_test)


def main():
    o2o_train()


if __name__ == '__main__':
    main()

# 部分数据归一化

# pca 特征压缩

# svm 尝试训练

# roc 获取数据（画图）

# Coupon_id 中竟然含有 fixed ，没天理
