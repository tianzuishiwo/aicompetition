from common.my_decorator import *
from hm.ml.houserent.rent_config import *
import pandas as pd
import numpy as np

# 强制关闭setingwichcopywarning警告
pd.set_option('mode.chained_assignment', None)

COL_index = 'index'
COL_order_ = 'id'
COL_time = 'time'
COL_area = 'area'  # 小区名
COL_area_rent_count = 'area_rent_count'
COL_floor = 'floor'
COL_total_floor = 'total_floor'
COL_home_area = 'home_area'
COL_home_direct = 'home_direct'
COL_reside_state = 'reside_state'
COL_room_count = 'room_count'
COL_parlor_count = 'parlor_count'
COL_toilet_count = 'toilet_count'
COL_rent_type = 'rent_type'
COL_district = 'district'
COL_location = 'location'
COL_metro_num = 'metro_num'
COL_metro_station = 'metro_station'
COL_distance = 'distance'
COL_decorate_situation = 'decorate_situation'
COL_month_fee = 'month_fee'

# 以下新增
COL_room_area = 'room_area'
COL_total_rtp = 'total_room_toilet_parlor'  # 房+卫+厅
COL_room_rate = 'room_rate'  # 房/总
COL_toilet_rate = 'toilet_rate'  # 卫/总
COL_parlor_rate = 'parlor_rate'  # 厅/总
COL_floor_rate = 'floor_rate'  # 楼层比
COL_home_type = 'home_type'  # 户型
COL_metro_exist = 'metro_exist'  # 有地铁
COL_location_path_count = 'location_path_count'  # 位置线路数
COL_area_path_count = 'area_path_count'  # 小区线路数

# COL_ = ''
# 训练集列名
COLUMNS_TRAIN = [
    # COL_order_,
    # COL_index,
    COL_time, COL_area, COL_area_rent_count, COL_floor,
    COL_total_floor, COL_home_area, COL_home_direct, COL_reside_state, COL_room_count,
    COL_parlor_count, COL_toilet_count, COL_rent_type, COL_district, COL_location,
    COL_metro_num, COL_metro_station, COL_distance, COL_decorate_situation, COL_month_fee,
]

# 测试集列名
COLUMNS_TEST = [
    # COL_index,
    COL_order_, COL_time, COL_area, COL_area_rent_count, COL_floor,
    COL_total_floor, COL_home_area, COL_home_direct, COL_reside_state, COL_room_count,
    COL_parlor_count, COL_toilet_count, COL_rent_type, COL_district, COL_location,
    COL_metro_num, COL_metro_station, COL_distance, COL_decorate_situation,
    # COL_month_fee,
]

"""
# ['Unnamed: 0', '时间', '小区名', '小区房屋出租数量', '楼层',
#  '总楼层', '房屋面积', '房屋朝向','居住状态', '卧室数量',
#  '厅的数量', '卫的数量', '出租方式', '区', '位置',
#  '地铁线路', '地铁站点', '距离','装修情况', '月租金']           
"""


class BaseDataHandle(object):
    def __init__(self, df,is_train):
        self.data = df
        self.is_train = is_train
        self.handle()

    def handle(self):
        pass

    def fill_mean(self, col_name):
        self.data[col_name][self.data[col_name].isna()] = self.data[col_name].mean()

    def fill_value(self, col_name, value):
        self.data[col_name][self.data[col_name].isna()] = value

    def feature_str2int(self, col_name):
        self.data[col_name] = self.data[col_name].astype('category').cat.codes

    def del_col(self, col_name):
        self.data.drop(col_name, axis=1, inplace=True)   # 删除列

    def del_row(self, col_name):  # 删除行，根据空数据
        self.data.drop(self.data[col_name][self.data[col_name].isna()].index, axis=0, inplace=True)

    def col_one_hot(self, col_name):
        self.data = pd.get_dummies(self.data, columns=[col_name])

    def get_data(self):
        return self.data

    def pshape(self):
        print(self.data.shape)


class MissValuer(BaseDataHandle):
    """缺失值处理"""

    def handle(self):
        self.fill_mean(COL_area_rent_count)
        self.fill_mean(COL_metro_station)
        self.fill_mean(COL_distance)
        self.fill_value(COL_rent_type, 3)
        self.fill_value(COL_reside_state, 0)
        self.fill_value(COL_metro_num, 0)
        self.fill_value(COL_decorate_situation, 0)


class AbnormalValuer(BaseDataHandle):
    """剔除异常值"""

    @caltime_p1('剔除异常值')
    def handle(self):
        if self.is_train:
            self.del_row(COL_district)  # 极少数有空值
            self.del_row(COL_location)  # 极少数有空值
        else:
            print('测试集不能删除数据')


class FeatureExtractor(BaseDataHandle):
    """特征提取"""

    @caltime_p1('特征提取')
    def handle(self):
        self.feature_str2int(COL_home_direct)

        # time home_direct 不能onehot，因为测试集可分类型与训练集不一致
        self.col_one_hot(COL_floor)
        self.col_one_hot(COL_reside_state)
        self.col_one_hot(COL_rent_type)
        self.col_one_hot(COL_metro_num)
        self.col_one_hot(COL_decorate_situation)

        self.add_room_area()
        self.add_total_rtp()
        self.add_room_rate()
        self.add_toilet_rate()
        self.add_parlor_rate()
        # self.add_floor_rate() # 楼层比先 ,有问题，先放
        self.add_home_type()
        self.add_metro_exist()
        # self.add_x_path_count(COL_area, COL_area_path_count)  # 有问题，先放
        # self.add_x_path_count(COL_location, COL_location_path_count)  # 有问题，先放
        pass

    def add_home_type(self):
        self.data[COL_home_type] = self.data[[COL_room_count, COL_parlor_count, COL_toilet_count]] \
            .apply(lambda x: str(x[COL_room_count]) + str(x[COL_parlor_count]) + str(x[COL_toilet_count]), axis=1)

    def add_room_area(self):
        self.data[COL_room_area] = self.data[COL_home_area] / (self.data[COL_room_count] + 1)

    def add_total_rtp(self):
        self.data[COL_total_rtp] = self.data[COL_room_count] + self.data[COL_toilet_count] + self.data[COL_parlor_count]

    def add_room_rate(self):
        self.data[COL_room_rate] = self.data[COL_room_count] / (self.data[COL_total_rtp] + 1)

    def add_toilet_rate(self):
        self.data[COL_toilet_rate] = self.data[COL_toilet_count] / (self.data[COL_total_rtp] + 1)

    def add_parlor_rate(self):
        self.data[COL_parlor_rate] = self.data[COL_parlor_count] / (self.data[COL_total_rtp] + 1)

    def add_floor_rate(self):
        self.data[COL_floor_rate] = self.data[COL_floor] / (self.data[COL_total_floor] + 1)

    def add_metro_exist(self):
        self.data[COL_metro_exist] = (self.data[COL_metro_station] > -1).map(int)

    def add_x_path_count(self, col_name, new_col_name):
        # COL_location_path_count = 'location_path_count'  # 位置线路数
        # COL_area_path_count = 'area_path_count'  # 小区线路数
        lines_count = self.data[[col_name, COL_metro_num]].drop_duplicates().groupby(col_name).count()
        lines_count.columns = [new_col_name]
        self.data = pd.merge(self.data, lines_count, how='left', on=[col_name])
        pass


class ElseHander(BaseDataHandle):
    """数据前置处理"""

    @caltime_p1('数据前置处理')
    def handle(self):
        self.del_col(COL_area)
        pass


class DataManager(object):
    """数据管理类"""

    def __init__(self, is_train=True):
        self.data = None
        self.is_train = is_train  # True 训练集 False 测试集
        self.load_data()
        self.raw_shape = self.data.shape

    @caltime_p1('加载数据集')
    def load_data(self):
        if self.is_train:
            print('@' * 30, '加载训练集数据')
            self.data = pd.read_csv(USE_TRAIN_PATH)
            self.data.columns = COLUMNS_TRAIN
        else:
            print('@' * 30, '加载测试集数据')
            self.data = pd.read_csv(USE_TEST_PATH)
            self.data.columns = COLUMNS_TEST

    @caltime_p1('生成特征数据')
    def generate_feature_df(self):
        self.print_info(self.data)
        self.data = ElseHander(self.data,self.is_train).get_data()
        self.data = MissValuer(self.data,self.is_train).get_data()
        self.data = AbnormalValuer(self.data,self.is_train).get_data()
        self.data = FeatureExtractor(self.data,self.is_train).get_data()

    def auto(self):
        self.generate_feature_df()
        # self.data = self.sample() # 数据截取
        self.print_info(self.data)
        self.save_feature()

    @caltime_p1('保存特征数据')
    def save_feature(self):
        self.data.to_csv(self.get_new_csv_path(), index=None)
        # self.data.to_csv(self.get_new_csv_path(self.raw_shape, self.data.shape), index=None)

    def print_info(self, df):
        print(df.shape)
        print(df.columns)

    def pshape(self):
        print(self.data.shape)

    def get_new_csv_path(self):
        # 生成文件带有时间戳
        head = 'train_small.csv'
        if not self.is_train:
            head = 'test_small.csv'
        csv_new = head
        print(f'生成特征数据文件(名称写死)： {csv_new}')
        path = RENT_TRAIN_PATH + csv_new
        return path

    # def get_new_csv_path(self, raw_shape, new_shape):
    #     # 生成文件带有时间戳
    #     # time-1000_23_to_333_45.csv
    #     head = 'train'
    #     if not self.is_train:
    #         head = 'test'
    #     csv_new = f'{head}_{str(int(time.time()))}-{raw_shape[0]}_{raw_shape[1]}_to_{new_shape[0]}_{new_shape[1]}.csv'
    #     print(f'生成特征数据文件： {csv_new}')
    #     path = RENT_TRAIN_PATH + csv_new
    #     return path

    @caltime_p1('数据截取')
    def sample(self):
        count = 10000
        return self.data.sample(count)


@caltime_p0('数据处理全部流程')
def generate_feature_csv():
    train = DataManager(True)
    train.auto()
    del train
    test = DataManager(False)
    test.auto()


def main():
    generate_feature_csv()


if __name__ == '__main__':
    main()
