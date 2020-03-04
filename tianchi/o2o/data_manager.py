from hm.o2o.user import User
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from hm.o2o.o2o_config import *
from hm.o2o.o2o_tool import *


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
        self.y_train = None
        self.y_test = None
        self._load_data()
        self.__str__()

    def _load_data(self):
        self.offline_train_csv = pd.read_csv('./data/small/ccf_offline_stage1_train.csv')
        self.online_train_csv = pd.read_csv('./data/small/ccf_online_stage1_train.csv')
        self.partial_offline_df = self.offline_train_csv.iloc[:OFFLINE_SPLIT_DATA, :]
        self.partial_online_df = self.online_train_csv.iloc[:ONLINE_SPLIT_DATA, :]
        # 加载数据
        # ccf_offline_stage1_test_revised_csv = pd.read_csv('./data/ccf_offline_stage1_test_revised.csv')
        # sample_submission_csv = pd.read_csv('./data/sample_submission.csv')

    def handle_data(self):
        self.partial_offline_df = self._handle_under_sample(self.partial_offline_df)
        self.partial_online_df = self._handle_under_sample(self.partial_online_df)
        self._drop_na(self.partial_online_df, self.partial_offline_df)
        offline_df, online_df = self._get_user_data(self.partial_online_df, self.partial_offline_df)
        self.train_df, self.target_df = self._merge_offline_online(offline_df, online_df)

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
        train_data = data_merge[TRAIN_COLUMNS]
        train_target_data = data_merge[TRAIN_TARGET_COLUMN]
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

    def _get_user_data(self, partial_online_df, partial_offline_df):
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

    def _print_data(self):
        print('打印线上用户前五条数据')
        print(self.partial_online_df.head(5))
        print('打印线下用户前五条数据')
        print(self.partial_offline_df.head(5))

    def print_traindf_info(self):
        print(f'x_train: {len(self.x_train)}')
        print(f'x_test: {len(self.x_test)}')
