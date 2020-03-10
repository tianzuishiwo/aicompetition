import os
from ml.main.config import *
from common.my_decorator import *

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# from pylab import mpl
# mpl.rcParams['font.ttf'] = ['fangsong_GB2312'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# 支持中文
plt.rcParams['font.sans-serif'] = ['fangsong_GB2312']  # 用来正常显示中文标签

import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告

IMAGE_PATH = 'image/'
ROOT_PATH = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/ml/main/data/'

TRAIN_PATH = ROOT_PATH + 'source/' + TRAIN_NAME
TEST_PATH = ROOT_PATH + 'source/' + TEST_NAME


class BaseAnalysis(object):
    TARGET_NAME = COL_month_fee
    FONT_SIZE = 18

    def __init__(self, data):
        self.data = data
        self.analisis()
        pass

    def analisis(self):
        pass

    def info(self):
        print('训练数据.info()')
        print(self.data.info())

    def describe(self):
        print('训练数据.describe()')
        print(self.data.describe())

    def divider(self):
        print("-" * 30, ' 分割线 ', '-' * 30)

    def columns(self):
        print(self.data.columns)

    def shape(self):
        print('训练数据.shape ', self.data.shape)

    def head(self, n=5, pd=None):
        print(f'训练数据.head({n})')
        if pd is None:
            print(self.data.head(n))
        else:
            print(pd.head(n))

    def col_head(self, col_name, n=5):
        print(f'{col_name}列的前{5}个数据')
        print(self.data[col_name].head(n))

    def miss_rate(self):
        print('构造缺失比例统计表：')
        # 每列的缺失值个数/总行数
        train_missing = (self.data.isnull().sum() / len(self.data)) * 100
        # 去掉缺失比例为0的列
        train_missing = \
            train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending=False)
        # .sort_values(ascending=False)
        miss_data = pd.DataFrame({'缺失百分比': train_missing})
        print(miss_data)
        pass

    def get_data(self):
        return self.data


class RelationshipAnalysis(BaseAnalysis):
    # 连续性特征
    CONTINUOUS_COLS = [COL_time, COL_area_rent_count, COL_total_floor, COL_home_area, COL_room_count, COL_parlor_count,
                       COL_toilet_count, COL_distance]

    @caltime_p1('相关性分析')
    def analisis(self):
        self.columns()
        self.continuous_values()

        pass

    @caltime_p1('通过散点图观察特征和目标值之间的关系-生成多个图例')
    def continuous_values(self):
        continuous_cols = self.CONTINUOUS_COLS
        # continuous_cols = ['时间', '小区房屋出租数量', '总楼层', '房屋面积', '卧室数量','厅的数量', '卫的数量', '距离']
        for col in continuous_cols:
            sns.jointplot(x=col, y=self.TARGET_NAME, data=self.data, alpha=0.3, size=4)
        plt.show()


class DataDistribute(BaseAnalysis):

    @caltime_p1('数据分布')
    def analisis(self):
        self.col_head(self.TARGET_NAME)
        self.target_value_distribute()
        self.head()
        self.all_col_distribute()

        pass

    @caltime_p1('生成目标值分布图例')
    def target_value_distribute(self):
        # print('目标值分布：')
        plt.figure(figsize=(20, 6))

        plt.subplot(221)
        plt.title(f'{self.TARGET_NAME}占比分布', fontsize=self.FONT_SIZE)
        sns.distplot(self.data[self.TARGET_NAME])

        plt.subplot(222)
        plt.title(f"{self.TARGET_NAME}排序图", fontsize=self.FONT_SIZE)
        plt.scatter(range(self.data.shape[0]), np.sort(self.data[self.TARGET_NAME].values))

        plt.show()
        pass

    @caltime_p1('生成所有特征分布图例')
    def all_col_distribute(self):
        # print('所有特征分布：')
        self.data.hist(figsize=(20, 20), bins=50, grid=False)
        plt.show()
        pass


class DataDescribe(BaseAnalysis):

    @caltime_p1('打印基本信息')
    def analisis(self):
        self.divider()
        self.shape()
        self.divider()
        self.info()
        self.divider()
        self.describe()
        # testdf.shape
        self.divider()
        self.head(n=10)
        self.divider()
        self.miss_rate()
        pass


class AnalysisManager(object):
    def __init__(self):
        self.data = None
        self.load()
        pass

    def start_analysis(self):
        DataDescribe(self.data)
        DataDistribute(self.data)
        RelationshipAnalysis(self.data)
        pass

    @caltime_p1('加载训练数据')
    def load(self):
        self.data = pd.read_csv(TRAIN_PATH, encoding='utf-8')


def test_something():
    # data = pd.read_csv(TRAIN_PATH)
    # print(data.columns)
    pass


@caltime_p0('数据分析所有步骤')
def analysis_run():
    da = AnalysisManager()
    da.start_analysis()


def main():
    # test_something()
    analysis_run()
    pass


if __name__ == '__main__':
    main()
