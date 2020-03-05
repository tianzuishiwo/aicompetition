import pandas as pd
import numpy as np
from tianchi.o2o.o2o_config import *
from tianchi.o2o.o2o_tool import *


class User(object):
    def __init__(self, data_df, user_type):
        self.data_df = data_df
        self.user_type = user_type

    # 添加一列：sample_type 负样本-1，普通消费0，正样本1
    def _add_sample_type(self):
        date_notna = self.data_df[COLUMN_Date].notna()
        date_isna = self.data_df[COLUMN_Date].isna()
        coupon_notna = self.data_df[COLUMN_Coupon_id].notna()
        coupon_isna = self.data_df[COLUMN_Coupon_id].isna()

        self.data_df[COLUMN_sample_type] = VALUE_0
        self.data_df[COLUMN_sample_type][date_isna & coupon_notna] = VALUE_F_1
        self.data_df[COLUMN_sample_type][date_notna & coupon_isna] = VALUE_0
        self.data_df[COLUMN_sample_type][date_notna & coupon_notna] = VALUE_1

    def _add_use_coupon(self):
        self.data_df[COLUMN_use_coupon] = VALUE_0
        self.data_df[COLUMN_use_coupon][self.data_df[COLUMN_sample_type] == VALUE_1] = VALUE_1

    def _add_discount_type(self):
        self.data_df[COLUMN_discount_type] = VALUE_F_1
        discount_rate_series = self.data_df[COLUMN_Discount_rate]
        discount_rate_series = discount_rate_series.replace(np.NaN, FORMART_EMPTY)

        index_dot_list = []
        index_colon_list = []
        for index in self.data_df.index:
            discount_des = discount_rate_series[index]
            if FORMART_DOT in discount_des:
                index_dot_list.append(index)
            elif FORMART_COLON in discount_des:
                index_colon_list.append(index)
            else:
                pass

        for i in index_dot_list:
            self.data_df.loc[i, COLUMN_discount_type] = VALUE_1
        for i in index_colon_list:
            self.data_df.loc[i, COLUMN_discount_type] = 2

        self.data_df[COLUMN_discount_type][discount_rate_series == FORMART_EMPTY] = VALUE_0
        self.data_df[COLUMN_discount_type][discount_rate_series == FORMART_FIXED] = 3

        # fixed 0.9 200:30
        # dscount_fixed,dscount_ratio,discount_satisfy

    def _split_discount_rate(self):
        self.data_df[COLUMN_discount_fixed] = VALUE_0
        self.data_df[COLUMN_discount_ratio] = VALUE_0
        self.data_df[COLUMN_discount_satisfy] = VALUE_0
        discount_rate_series = self.data_df[COLUMN_Discount_rate]
        discount_rate_series = discount_rate_series.replace(np.NaN, FORMART_EMPTY)

        index_dot_list = {}
        index_colon_list = {}
        index_fixed_list = []
        #     print('self.data_df.index=',self.data_df.index)
        for index in self.data_df.index:
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
            self.data_df.loc[i, COLUMN_discount_fixed] = VALUE_1
        for k, v in index_dot_list.items():
            self.data_df.loc[k, COLUMN_discount_ratio] = v
        for k, v in index_colon_list.items():
            self.data_df.loc[k, COLUMN_discount_satisfy] = v

    def _add_use_coupon_VALUE_15day(self):
        self.data_df[COLUMN_use_coupon_15day] = VALUE_0
        # 其实这里是索引列表
        index_list = self.data_df[self.data_df[COLUMN_use_coupon] == VALUE_1].index
        target_index_list = []
        for index in index_list:
            date = self.data_df.loc[index, COLUMN_Date]
            date_received = self.data_df.loc[index, COLUMN_Date_received]
            is_bigger_15d = is_bigger_15day(date_received, date)
            if is_bigger_15d:
                target_index_list.append(index)
        #     print('Date_received:',date_received,' Date:',date)
        for i in target_index_list:
            self.data_df.loc[i, COLUMN_use_coupon_15day] = VALUE_1

    def _add_user_type(self, n):
        self.data_df[COLUMN_user_type] = n

    def _add_receive_weekday(self):
        self.data_df['date_rcv_time'] = self.data_df[COLUMN_Date_received]
        self.data_df['date_rcv_time'] = self.data_df['date_rcv_time'].apply(lambda x: get_stime(x))
        time_ = pd.to_datetime(self.data_df['date_rcv_time'], unit='s')
        time_ = pd.DatetimeIndex(time_)
        self.data_df[COLUMN_day] = time_.day
        self.data_df[COLUMN_hour] = time_.hour
        self.data_df[COLUMN_weekday] = time_.weekday

    def _add_use_coupon_15day(self):
        self.data_df[COLUMN_use_coupon_15day] = VALUE_0
        # 其实这里是索引列表
        index_list = self.data_df[self.data_df[COLUMN_use_coupon] == VALUE_1].index
        target_index_list = []
        for index in index_list:
            date = self.data_df.loc[index, COLUMN_Date]
            date_received = self.data_df.loc[index, COLUMN_Date_received]
            # print(self.data_df[COLUMN_Discount_rate].head(5))
            is_bigger_15d = is_bigger_15day(date_received, date)
            if is_bigger_15d:
                target_index_list.append(index)
        #     print('Date_received:',date_received,' Date:',date)
        for i in target_index_list:
            self.data_df.loc[i, COLUMN_use_coupon_15day] = VALUE_1

    # def data_add_columns(self, user_type):
    #     # handle_under_sample(input_df)  # 欠采样
    #     # data_dropna(input_df)
    #
    #     add_sample_type(input_df)
    #     add_use_coupon(input_df)
    #     add_use_coupon_15day(input_df)
    #     add_user_type(input_df, user_type)  # 0 线下 1 线上
    #     add_discount_type(input_df)
    #     split_discount_rate(input_df)
    #     add_receive_weekday(input_df)

    def data_extract(self):
        self._add_sample_type()
        self._add_use_coupon()
        self._add_use_coupon_15day()
        self._add_user_type(self.user_type)  # 0 线下 1 线上
        self._add_discount_type()
        self._split_discount_rate()
        self._add_receive_weekday()

    def get_data(self):
        return self.data_df

    def set_data(self, df):
        self.data_df = df
