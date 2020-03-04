import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from collections import Counter
import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# 强制关闭setingwichcopywarning警告
pd.set_option('mode.chained_assignment', None)

# 所有参数配置
OFFLINE_SPLIT_DATA = 5000
ONLINE_SPLIT_DATA = 20000
FORMART_EMPTY = 'empty'
FORMART_DOT = '.'
FORMART_COLON = ':'
FORMART_FIXED = 'fixed'
TRAIN_COLUMNS = ['User_id', 'Merchant_id', 'Action', 'Distance', 'day', 'hour', 'weekday',
                 'discount_fixed', 'discount_ratio', 'discount_satisfy', 'sample_type', 'user_type']
TRAIN_TARGET_COLUMN = 'use_coupon_15day'

# 加载数据
# ccf_offline_stage1_test_revised_csv = pd.read_csv('./data/ccf_offline_stage1_test_revised.csv')
ccf_offline_stage1_train_csv = pd.read_csv('./data/small/ccf_offline_stage1_train.csv')
ccf_online_stage1_train_csv = pd.read_csv('./data/small/ccf_online_stage1_train.csv')


# sample_submission_csv = pd.read_csv('./data/sample_submission.csv')


# 工具类

# 截取部分数据训练
def split_data():
    offline_data = ccf_offline_stage1_train_csv.iloc[:OFFLINE_SPLIT_DATA, :]
    online_data = ccf_online_stage1_train_csv.iloc[:ONLINE_SPLIT_DATA, :]
    print(offline_data.shape, online_data.shape)
    return offline_data, online_data


def get_stime(input_time):
    return time.mktime(time.strptime(str(int(input_time)), '%Y%m%d'))


def is_bigger_15day(before_time, after_time):
    btime = get_stime(before_time)
    atime = get_stime(after_time)
    if atime > btime and ((atime - btime) < (15 * 24 * 60 * 60)):
        return True
    return False


def convert_value(str_value):
    if str_value is not None:
        splits = str_value.split(':')
        value = round(float(splits[0]) / float(splits[1]), 1)
        return value
    return 0


def print_counter(input_df, column_names):
    for column_name in column_names:
        counter = Counter(input_df[column_name])
        print(counter)


#     counter = Counter(offline_data[column_name])
#     print(counter)

def column_one_hot(input_df, column_name):
    one_hot_encoder = OneHotEncoder()
    onehot_column = input_df[column_name].values.reshape(-1, 1)
    input_df[column_name] = one_hot_encoder.fit_transform(onehot_column)


def columns_one_hot(input_df, column_names):
    one_hot_encoder = OneHotEncoder()
    for column_name in column_names:
        onehot_column = input_df[column_name].values.reshape(-1, 1)
        input_df[column_name] = one_hot_encoder.fit_transform(onehot_column)


# def clear():
#     print('屏幕已清空')

# 样本数据处理

# 添加一列：sample_type 负样本-1，普通消费0，正样本1
def add_sample_type(input_df):
    date_notna = input_df['Date'].notna()
    date_isna = input_df['Date'].isna()
    coupon_notna = input_df['Coupon_id'].notna()
    coupon_isna = input_df['Coupon_id'].isna()

    input_df['sample_type'] = 0
    input_df['sample_type'][date_isna & coupon_notna] = -1
    input_df['sample_type'][date_notna & coupon_isna] = 0
    input_df['sample_type'][date_notna & coupon_notna] = 1


def add_use_coupon(input_df):
    input_df['use_coupon'] = 0
    input_df['use_coupon'][input_df['sample_type'] == 1] = 1


def add_discount_type(input_dataframe):
    input_dataframe['discount_type'] = -1
    discount_rate_series = input_dataframe['Discount_rate']
    discount_rate_series = discount_rate_series.replace(np.NaN, FORMART_EMPTY)

    index_dot_list = []
    index_colon_list = []
    for index in input_dataframe.index:
        discount_des = discount_rate_series[index]
        if FORMART_DOT in discount_des:
            index_dot_list.append(index)
        elif FORMART_COLON in discount_des:
            index_colon_list.append(index)
        else:
            pass

    for i in index_dot_list:
        input_dataframe.loc[i, 'discount_type'] = 1
    for i in index_colon_list:
        input_dataframe.loc[i, 'discount_type'] = 2

    input_dataframe['discount_type'][discount_rate_series == FORMART_EMPTY] = 0
    input_dataframe['discount_type'][discount_rate_series == FORMART_FIXED] = 3

    # fixed 0.9 200:30
    # dscount_fixed,dscount_ratio,discount_satisfy


def split_discount_rate(input_df):
    input_df['discount_fixed'] = 0
    input_df['discount_ratio'] = 0
    input_df['discount_satisfy'] = 0
    discount_rate_series = input_df['Discount_rate']
    discount_rate_series = discount_rate_series.replace(np.NaN, FORMART_EMPTY)

    index_dot_list = {}
    index_colon_list = {}
    index_fixed_list = []
    #     print('input_df.index=',input_df.index)
    for index in input_df.index:
        discount_des = discount_rate_series[index]
        if FORMART_DOT in discount_des:
            index_dot_list[index] = float(discount_des)
        elif FORMART_COLON in discount_des:
            index_colon_list[index] = convert_value(discount_des)
        elif FORMART_FIXED == discount_des:
            index_fixed_list.append(index)
        else:
            pass
    for i in index_fixed_list:
        input_df.loc[i, 'discount_fixed'] = 1
    for k, v in index_dot_list.items():
        input_df.loc[k, 'discount_ratio'] = v
    for k, v in index_colon_list.items():
        input_df.loc[k, 'discount_satisfy'] = v


def add_use_coupon_15day(input_df):
    input_df['use_coupon_15day'] = 0
    # 其实这里是索引列表
    index_list = input_df[input_df['use_coupon'] == 1].index
    target_index_list = []
    for index in index_list:
        date = input_df.loc[index, 'Date']
        date_received = input_df.loc[index, 'Date_received']
        is_bigger_15d = is_bigger_15day(date_received, date)
        if is_bigger_15d:
            target_index_list.append(index)
    #     print('Date_received:',date_received,' Date:',date)
    for i in target_index_list:
        input_df.loc[i, 'use_coupon_15day'] = 1


def add_user_type(input_df, n):
    input_df['user_type'] = n


def add_receive_weekday(input_df):
    input_df['date_rcv_time'] = input_df['Date_received']
    input_df['date_rcv_time'] = input_df['date_rcv_time'].apply(lambda x: get_stime(x))
    time_ = pd.to_datetime(input_df['date_rcv_time'], unit='s')
    time_ = pd.DatetimeIndex(time_)
    input_df['day'] = time_.day
    input_df['hour'] = time_.hour
    input_df['weekday'] = time_.weekday


#  样本过采样
def handle_over_sample(input_df):
    print('过采样前：', input_df.shape)
    date_notna = input_df['Date'].notna()
    coupon_notna = input_df['Coupon_id'].notna()
    input_df['target'] = 0
    input_df['target'][date_notna & coupon_notna] = 1
    over_sample_transformer = RandomOverSampler()
    x_over_sample, y_over_sample = over_sample_transformer.fit_resample(input_df, input_df['target'])


#  样本欠采样
def handle_under_sample(input_df):
    #     print('过采样前：',input_df.shape)
    date_notna = input_df['Date'].notna()
    coupon_notna = input_df['Coupon_id'].notna()
    input_df['target'] = np.zeros_like(input_df['Coupon_id'].shape[0])
    input_df['target'][date_notna & coupon_notna] = 1
    transfer = RandomUnderSampler()
    x_under, y_under = transfer.fit_resample(input_df, input_df['target'])
    input_df = x_under.drop('target', axis=1)
    return input_df


def data_dropna(input_df):
    input_df.dropna(subset=['Date_received'], inplace=True)


def data_add_columns(input_df, user_type):
    handle_under_sample(input_df)  # 欠采样
    data_dropna(input_df)
    add_sample_type(input_df)
    add_use_coupon(input_df)
    add_use_coupon_15day(input_df)
    add_user_type(input_df, user_type)  # 0 线下 1 线上
    add_discount_type(input_df)
    split_discount_rate(input_df)
    add_receive_weekday(input_df)


# 部分数据归一化

# pca 特征压缩

# svm 尝试训练

# roc 获取数据（画图）

# Coupon_id 中竟然含有 fixed ，没天理
# 合并线上和线下
def merge_offline_online(offline_data, online_data):
    data_merge = pd.concat([offline_data, online_data],sort=True)
    data_merge['Action'].replace(np.NaN, -1, inplace=True)
    data_merge['Distance'].replace(np.NaN, -1, inplace=True)
    train_data = data_merge[TRAIN_COLUMNS]  # 'Coupon_id',
    train_target_data = data_merge[TRAIN_TARGET_COLUMN]
    return train_data, train_target_data


# def before_train():
#     train_data,train_target_data = merge_offline_online()
# #     print(train_data.columns)
#     columns_one_hot(train_data,['User_id','Merchant_id','Action','sample_type'])


def print_info(train_data):
    print('打印train_data信息')
    print(train_data.shape)
    print(train_data)
    train_data.to_excel('train_data.xls', sheet_name='sheet_name')


def pca_and_train(train_data, train_target_data):
    #     def pac_and_train(train_data,train_target_data):
    # print(train_data.columns)
    pca = PCA(n_components=0.9)
    print_info(train_data)
    data_pca = pca.fit_transform(train_data)
    # print(data_pca)
    x_train, x_test, y_train, y_test = train_test_split(data_pca, train_target_data)

    svm = SVC()
    svm.fit(x_train, y_train)
    y_predict = svm.predict(x_test)
    accuracy = svm.score(x_test, y_test)
    print('pca n_components=', pca.n_components, ' accuracy=', accuracy)
    auc = roc_auc_score(y_test, y_predict)
    print('*' * 20, 'auc=', auc)
    print('训练数据：')
    print(train_data.head(50))


#     print(train_data.head(50))


def main():
    offline_data, online_data = split_data()
    data_add_columns(offline_data, 0)
    data_add_columns(online_data, 1)
    train_data, train_target_data = merge_offline_online(offline_data, online_data)
    # columns_one_hot(train_data, ['User_id', 'Merchant_id', 'Action', 'sample_type'])
    #     before_train()
    pca_and_train(train_data, train_target_data)


if __name__ == '__main__':
    main()
