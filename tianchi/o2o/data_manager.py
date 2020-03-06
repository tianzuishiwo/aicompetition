from tianchi.o2o.user import User
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tianchi.o2o.o2o_config import *
from tianchi.o2o.o2o_tool import *


class DataManager(object):
    def __init__(self):
        self.offline_train_csv = None
        self.online_train_csv = None
        self.partial_offline_df = None
        self.partial_online_df = None
        self.train_df = None
        self.target_df = None
        self.x_train = None
        self.x_test = None
        self._load_data()
        self.__str__()
        self._print_data(COUNT)

    def _load_data(self):
        self.offline_train_csv = pd.read_csv(USE_OFFLINE_TRAIN_CSV)
        self.online_train_csv = pd.read_csv(USE_ONLINE_TRAIN_CSV)
        self.partial_offline_df = self.offline_train_csv.iloc[:OFFLINE_SPLIT_DATA, :]
        self.partial_online_df = self.online_train_csv.iloc[:ONLINE_SPLIT_DATA, :]
        # 加载数据
        # ccf_offline_stage1_test_revised_csv = pd.read_csv('./data/ccf_offline_stage1_test_revised.csv')
        # sample_submission_csv = pd.read_csv('./data/sample_submission.csv')

    def handle_data(self):
        self._print_shape('过采样前')
        self.partial_offline_df = self._handle_under_sample(self.partial_offline_df)
        self.partial_online_df = self._handle_under_sample(self.partial_online_df)
        self._print_shape('过采样后')
        self._print_shape('删除缺失数据前')
        self._drop_na(self.partial_online_df, self.partial_offline_df)
        self._print_shape('删除缺失数据后')
        self._print_shape('添加新的特征前')
        count = 2
        self._print_data(count, False)
        # offline_df, online_df = self._get_user_data(self.partial_offline_df, self.partial_online_df)
        self.partial_offline_df, self.partial_online_df = self._get_user_data(self.partial_offline_df,
                                                                              self.partial_online_df)
        self._print_shape('添加新的特征后')
        self._print_data(count, False)
        self._print_shape('线上线下合并之前')
        self.train_df, self.target_df = self._merge_offline_online(self.partial_offline_df, self.partial_online_df)
        self.print_df('线上线下合并之后', self.train_df, self.target_df)
        # self._print_shape('线上线下合并之后')
        # self._print_data(count)

    def get_train_df(self):
        return self.train_df

    def _drop_na(self, partial_online_df, partial_offline_df):
        partial_online_df.dropna(subset=[COLUMN_Date_received], inplace=True)
        partial_offline_df.dropna(subset=[COLUMN_Date_received], inplace=True)

    # def get_target_df(self):
    #     return self.target_df

    # 合并线上和线下
    def _merge_offline_online(self, offline_data, online_data):
        data_merge = pd.concat([offline_data, online_data], sort=True)
        data_merge[COLUMN_Action].replace(np.NaN, VALUE_F_1, inplace=True)
        data_merge[COLUMN_Distance].replace(np.NaN, VALUE_F_1, inplace=True)
        print(f'合并后列: {data_merge.shape}')
        print('提取待训练特征')
        print(f'特征列(len={len(TRAIN_COLUMNS)})：{TRAIN_COLUMNS}')
        print(f'标签列：{TRAIN_TARGET_COLUMN}')

        train_data = data_merge[TRAIN_COLUMNS]
        train_target_data = data_merge[TRAIN_TARGET_COLUMN]
        print('用户分类 ', Counter(train_data[COLUMN_user_type]))
        print('标签值成分 ', Counter(train_target_data))
        print('提取待训练特征完成')
        return train_data, train_target_data

    #  样本欠采样
    def _handle_under_sample(self, input_df):
        #     print('过采样前：',input_df.shape)
        date_notna = input_df[COLUMN_Date].notna()
        coupon_notna = input_df[COLUMN_Coupon_id].notna()
        input_df['target'] = np.zeros_like(input_df[COLUMN_Coupon_id].shape[0])
        input_df['target'][date_notna & coupon_notna] = 1
        transfer = RandomUnderSampler()
        x_under, y_under = transfer.fit_resample(input_df, input_df['target'])
        input_df = x_under.drop('target', axis=1)
        return input_df

    def _get_user_data(self, partial_offline_df, partial_online_df):
        # offline_user = User(self.partial_offline_df, VALUE_0)
        # online_user = User(self.partial_online_df, VALUE_1)
        offline_user = User(partial_offline_df, VALUE_0)
        online_user = User(partial_online_df, VALUE_1)
        offline_user.data_extract()
        online_user.data_extract()
        return offline_user.get_data(), online_user.get_data()

    def train_test_split(self, data_pca):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data_pca, self.target_df)

    def __str__(self):
        print(f'self.offline_train_csv = { self.offline_train_csv.shape}')
        print(f'self.online_train_csv = { self.online_train_csv.shape}')
        print(f'self.partial_offline_df = { self.partial_offline_df.shape}')
        print(f'self.partial_online_df = {self.partial_online_df.shape}')
        return ''

    def _print_data(self, count, print_detail=True):
        if print_detail:
            print(f'打印线上用户前{count}条数据')
            print(self.partial_online_df.head(count))
        print(f'线上列名(len={len(self.partial_online_df.columns)})：{self.partial_online_df.columns}')
        if print_detail:
            print(f'打印线下用户前{count}条数据')
            print(self.partial_offline_df.head(count))
        print(f'线下列名(len={len(self.partial_offline_df.columns)})：{self.partial_offline_df.columns}')

    def _print_shape(self, des=""):
        print(f'{des} self.partial_offline_df = { self.partial_offline_df.shape}')
        print(f'{des} self.partial_online_df = {self.partial_online_df.shape}')

    def print_traindf_info(self, des=""):
        # print(f'{des} x_train: {len(self.x_train)} , 列数：{len(self.x_train.columns)}')
        # print(f'{des} x_test: {len(self.x_test)} , 列数：{len(self.x_test.columns)}')
        print(f'{des} x_train: {len(self.x_train)} ')
        print(f'{des} x_test: {len(self.x_test)} ')

    def print_df(self, des, train_df, target_df):
        print(f'{des} train_df = {train_df.shape}')
        print(f'{des} target_df = {target_df.shape}')
