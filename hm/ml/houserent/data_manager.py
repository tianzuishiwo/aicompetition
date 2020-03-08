from common.my_decorator import *
from hm.ml.houserent.rent_config import *
import pandas as pd
import numpy as np

# 强制关闭setingwichcopywarning警告
pd.set_option('mode.chained_assignment', None)

COL_index = 'index'
COL_order_ = 'id'
COL_time = 'time'
COL_area = 'area'
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


# COL_ = ''

COLUMNS_TRAIN = [
    # COL_order_,
    # COL_index,
    COL_time, COL_area, COL_area_rent_count, COL_floor,
    COL_total_floor, COL_home_area, COL_home_direct, COL_reside_state, COL_room_count,
    COL_parlor_count, COL_toilet_count, COL_rent_type, COL_district, COL_location,
    COL_metro_num, COL_metro_station, COL_distance, COL_decorate_situation, COL_month_fee,
]

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

x_columns=['时间', '新小区名', '小区房屋出租数量', '楼层', '总楼层', '房屋面积','居住状态', '卧室数量',
       '厅的数量', '卫的数量', '出租方式', '区', '位置', '地铁线路', '地铁站点', '距离', '装修情况', 
       '新朝向', '房+卫+厅', '房/总', '卫/总', '厅/总', '卧室面积', '楼层比', '户型','平均值特征1',
       '平均值特征2','有地铁','小区线路数','位置线路数','小区条数大于100','小区平均值特征','朝向平均值特征',
           '站点平均值特征','位置平均值特征']
           
           '新小区名', '小区房屋出租数量','新朝向','房+卫+厅', '房/总', '卫/总', '厅/总','卧室面积','户型','平均值特征1',
       '平均值特征2','有地铁','小区线路数','位置线路数','小区条数大于100','小区平均值特征','朝向平均值特征',
           '站点平均值特征','位置平均值特征'
           
           '小区线路数'(4),'位置线路数'(4),
           
           '有地铁'(2,3,4),
           '房+卫+厅'(4),
           '卧室面积'(4),
           
"""


class BaseDataHandle(object):
    def __init__(self, df):
        self.data = df
        self.handle()

    def handle(self):
        pass

    def fill_mean(self, col_name):
        self.data[col_name][self.data[col_name].isna()] = self.data[col_name].mean()

    def fill_value(self, col_name, value):
        # data[data['metro_num'].isna()]=0
        self.data[col_name][self.data[col_name].isna()] = value

    def feature_str2int(self, col_name):
        self.data[col_name] = self.data[col_name].astype('category').cat.codes

    def del_col(self, col_name):
        # 删除列
        self.data.drop(col_name, axis=1, inplace=True)

    def del_row(self, col_name):
        # 删除行，根据空数据
        self.data.drop(self.data[col_name][self.data[col_name].isna()].index, axis=0, inplace=True)
        # newdf = traindf.drop(traindf['区'][traindf['区'].isna()].index,axis=0)

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
        # data[data['metro_station'].isna()]=data['metro_station'].mean()


class AbnormalValuer(BaseDataHandle):
    """剔除异常值"""

    @caltime_p1('剔除异常值')
    def handle(self):
        self.del_row(COL_district)  # 极少数有空值
        self.del_row(COL_location)  # 极少数有空值


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
        # self.add_floor_rate() # 楼层比先屏蔽
        self.add_home_type()
        self.add_metro_exist()
        pass

    def add_home_type(self):
        # train['户型'] = train[['卧室数量', '厅的数量', '卫的数量']].
        # apply(lambda x: str(x['卧室数量']) + str(x['厅的数量']) + str(x['卫的数量']),axis=1)
        self.data[COL_home_type] = self.data[[COL_room_count, COL_parlor_count, COL_toilet_count]] \
            .apply(lambda x: str(x[COL_room_count]) + str(x[COL_parlor_count]) + str(x[COL_toilet_count]), axis=1)

    def add_room_area(self):
        #  data['room_area'] = data['home_area']/(data['room_count']+1)
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


class ElseHander(BaseDataHandle):
    """
    关于暂未处理
    1.以下列暂未one-hot编码
    onehot 处理：“时间”，“楼层”，“房屋朝向”（dummy），“居住状态”，“出租方式”，‘地铁线路’，“装修情况”
    """

    @caltime_p1('数据前置处理')
    def handle(self):
        # self.data.columns = COLUMNS
        self.del_col(COL_area)
        pass


class DataManager(object):

    def __init__(self, is_train=True):
        self.data = None
        # self.is_train = True
        self.is_train = is_train
        self.load_data()
        self.raw_shape = self.data.shape

    @caltime_p1('加载数据集')
    def load_data(self):
        if self.is_train:
            print('@' * 30, '加载训练集数据')
            self.data = pd.read_csv(USE_TRAIN_PATH)
            print('')
            self.data.columns = COLUMNS_TRAIN
        else:
            print('@' * 30, '加载测试集数据')
            self.data = pd.read_csv(USE_TEST_PATH)
            self.data.columns = COLUMNS_TEST

    @caltime_p1('生成特征数据')
    def generate_feature_df(self):
        self.print_info(self.data)
        self.data = ElseHander(self.data).get_data()
        self.data = MissValuer(self.data).get_data()
        self.data = AbnormalValuer(self.data).get_data()
        self.data = FeatureExtractor(self.data).get_data()

    def auto(self):
        self.generate_feature_df()
        # self.data = self.sample() # 数据截取
        self.print_info(self.data)
        self.save_feature()

    @caltime_p1('保存特征数据')
    def save_feature(self):
        self.data.to_csv(self.get_new_csv_path(self.raw_shape, self.data.shape), index=None)

    def print_info(self, df):
        print(df.shape)
        print(df.columns)
        # print(df.info())

    def pshape(self):
        print(self.data.shape)

    def get_new_csv_path(self, raw_shape, new_shape):
        # 生成文件带有时间戳
        # time-1000_23_to_333_45.csv
        head = 'train_small.csv'
        if not self.is_train:
            head = 'test_small.csv'
        # csv_new = f'{head}_{str(int(time.time()))}-{raw_shape[0]}_{raw_shape[1]}_to_{new_shape[0]}_{new_shape[1]}.csv'
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
def test_data():
    train = DataManager(True)
    train.auto()
    del train
    test = DataManager(False)
    test.auto()


def main():
    test_data()


if __name__ == '__main__':
    main()
