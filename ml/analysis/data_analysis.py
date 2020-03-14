import os
from ml.main.config import *
from common.my_decorator import *
# from ml.analysis.data_analysis_rough import ResearchQuestionDataRough


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['fangsong_GB2312']  # 用来正常显示中文标签

import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告

"""
待优化点：

"""

SAVE_IMG = True  # 图例保存到本地
IMG_SHOW = not SAVE_IMG
IMG_SKIP = False  # 跳过所有画图

IMAGE_PATH = 'image/'
ROOT_PATH = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/ml/main/data/'

TRAIN_PATH = ROOT_PATH + 'source/' + TRAIN_NAME
TEST_PATH = ROOT_PATH + 'source/' + TEST_NAME

img_order = 1


class BaseObject(object):
    FIG_SIZE = (20, 12)
    FONT_SIZE = 18
    TARGET_NAME = None
    # 连续性特征
    CONTINUOUS_COLS = None
    # 与目标值相关性最大列
    COL_MAX_RELATIONSHIP_WITH_TARGET = None

    pass


class BaseAnalysis(BaseObject):
    TARGET_NAME = COL_month_fee
    # 连续性特征
    CONTINUOUS_COLS = [COL_time, COL_area_rent_count, COL_total_floor,
                       COL_home_area, COL_room_count, COL_parlor_count,
                       COL_toilet_count, COL_distance]

    def __init__(self, data):
        self.data = data
        self.analisis()
        pass

    # 子类必实现接口
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

    def col_value_counts(self, col_name):
        # 查看某列，关于列内属性的统计，例如：'东' 24749 '南' 41769 '西' 7559 ...
        self.data[col_name].value_counts()

    def barh(self, y_values, x_values, x_label, title=''):
        """水平直方图"""
        plt.figure(figsize=self.FIG_SIZE)
        plt.barh(y_values, x_values)
        plt.xlabel(x_label)
        plt.title(title)
        self.show_or_save('水平直方图-'+title)
        # self.show_or_save_img(title, True, True)

    def boxline(self, x_label, y_label, data):
        """绘制箱线图"""
        plt.figure(figsize=(12, 10))
        sns.boxplot(y=y_label, x=x_label, data=data, orient='h')
        self.show_or_save('箱线图-' + x_label + 'vs' +y_label )
        pass

    def show_or_save_img(self, img_name, save_img, show_img, img_skip=False):
        """是否保存图片，是否显示图片，是否直接跳过"""
        global img_order
        if img_skip:
            self._plt_clear()
            return None
        if save_img:
            plt.savefig(IMAGE_PATH + str(img_order) + "_" + img_name + '.png')
        if show_img:
            plt.show()
        else:
            self._plt_clear()
        img_order = img_order + 1

    def _plt_clear(self):
        '''plt重置（绘图过程不调用plt.show，下一次绘图会叠加当前数据显示）'''
        plt.cla()
        plt.clf()
        plt.close()

    def show_or_save(self, img_name):
        # 所有图片默认配置
        self.show_or_save_img(img_name, SAVE_IMG, IMG_SHOW, img_skip=IMG_SKIP)

    def get_data(self):
        return self.data


class ResearchQuestionData(BaseAnalysis):

    @caltime_p1(LEVEL_1 + '问题数据分析')
    def analisis(self):
        pass


class RelationshipAnalysis(BaseAnalysis):

    @caltime_p1(LEVEL_1 + '相关性分析')
    def analisis(self):
        self.columns()
        self.continuous_values()
        self.heatmap()
        self.pearson_relationship()
        self.box_line_map()
        self.outlier_analysis()
        pass

    @caltime_p1(LEVEL_2 + '通过散点图观察特征和目标值之间的关系-生成多个图例')
    def continuous_values(self):
        continuous_cols = self.CONTINUOUS_COLS
        # continuous_cols = ['时间', '小区房屋出租数量', '总楼层', '房屋面积', '卧室数量','厅的数量', '卫的数量', '距离']
        for col_name in continuous_cols:
            sns.jointplot(x=col_name, y=self.TARGET_NAME, data=self.data, alpha=0.3, size=8)
            self.show_or_save(f'二维散点图-{self.TARGET_NAME}vs{col_name}之间的关系')

    @caltime_p1(LEVEL_2 + '皮尔森相关性热力图')
    def heatmap(self):
        # 计算皮尔森相关性
        corrmat = self.data.corr()
        plt.figure(figsize=(20, 20))
        sns.heatmap(corrmat, square=True, linewidths=0.1, annot=True)
        self.show_or_save('皮尔森相关性热力图')
        pass

    @caltime_p1(LEVEL_2 + '特征与目标值皮尔森相关性分析图')
    def pearson_relationship(self):
        corr = self.data.corr()[self.TARGET_NAME][self.CONTINUOUS_COLS].sort_values(ascending=False)
        self.barh(corr.index.values, corr.values, self.TARGET_NAME, title='全部特征与目标值皮尔森相关性分析')

        self.COL_MAX_RELATIONSHIP_WITH_TARGET = corr.index[0]
        print('皮尔森相关性(特征与目标值):', type(corr))  # corr 类型为Series
        print(corr)
        print('目标值相关性最强特征：  ', "<" * 5, self.COL_MAX_RELATIONSHIP_WITH_TARGET, ">" * 5)
        pass

    @caltime_p1(LEVEL_2 + '绘制箱线图')
    def box_line_map(self):
        """
         所有列特征：   ['时间', '小区名', '小区房屋出租数量', '楼层', '总楼层', '房屋面积', '房屋朝向', '居住状态', '卧室数量',
               '厅的数量', '卫的数量', '出租方式', '区', '位置', '地铁线路', '地铁站点', '距离', '装修情况', '月租金']

         连续列特征：   continuous_cols = ['时间', '小区房屋出租数量', '总楼层', '房屋面积',  '卧室数量','厅的数量', '卫的数量', '距离']

         非连续性特征：  ['区', '装修情况', '地铁站点', '房屋朝向', '小区名', '位置', '楼层', '地铁线路', '出租方式', '居住状态', '月租金']

         老师使用列：['时间', '楼层', '居住状态', '出租方式', '区', '地铁线路', '装修情况']
        """
        col_list = ['时间', '楼层', '居住状态', '出租方式', '区', '地铁线路', '装修情况']  # 老师使用列
        print('箱线图待绘制特征列表(仅选择子类别个数不多的)： ', col_list)
        for i in range(len(col_list)):
            self.boxline(self.TARGET_NAME, col_list[i], self.data)
        print(f'箱线图绘制完成,共{len(col_list)}张！')
        pass

    @caltime_p1(LEVEL_2 + '异常值分析图')
    def outlier_analysis(self):
        corr = self.data.corr()[self.TARGET_NAME][self.CONTINUOUS_COLS].sort_values(ascending=False)
        col_name = corr.index[0]
        self.COL_MAX_RELATIONSHIP_WITH_TARGET = col_name

        plt.figure(figsize=(10, 10))
        # plt.title = f'异常值分析图: {col_name}---{self.TARGET_NAME}'  # 不起作用
        sns.regplot(x=self.data[col_name], y=self.data[self.TARGET_NAME])
        self.show_or_save('线性回归图-异常值分析图')
        pass


class DataDistribute(BaseAnalysis):

    @caltime_p1(LEVEL_1 + '数据分布')
    def analisis(self):
        # self.col_head(self.TARGET_NAME)
        self.target_value_distribute()
        self.head()
        self.all_col_distribute()
        pass

    @caltime_p1(LEVEL_2 + '生成目标值分布图例')
    def target_value_distribute(self):
        # print('目标值分布：')
        plt.figure(figsize=self.FIG_SIZE)

        plt.subplot(221)
        plt.title(f'{self.TARGET_NAME}占比分布', fontsize=self.FONT_SIZE)
        sns.distplot(self.data[self.TARGET_NAME])

        plt.subplot(222)
        plt.title(f"{self.TARGET_NAME}排序图", fontsize=self.FONT_SIZE)
        plt.scatter(range(self.data.shape[0]), np.sort(self.data[self.TARGET_NAME].values))

        self.show_or_save('二维图-目标值分布图例')
        # plt.show()
        pass

    @caltime_p1(LEVEL_2 + '生成所有特征分布图例')
    def all_col_distribute(self):
        # print('所有特征分布：')  # '生成所有特征分布图例',
        self.data.hist(figsize=(20, 20), bins=50, grid=False)
        self.show_or_save('直方图-所有特征分布')
        pass


class DataDescribe(BaseAnalysis):

    @caltime_p1(LEVEL_1 + '打印基本信息')
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
        self.miss_rate()  # 打印缺失率： 可构造成条形图或bar图
        pass

    def miss_rate(self):
        """缺失比例统计水平直方图"""
        map_name = '缺失比例统计'
        col_name = '缺失百分比'
        # 每列的缺失值个数/总行数
        train_missing = (self.data.isnull().sum() / len(self.data)) * 100
        # 去掉缺失比例为0的列
        train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending=False)
        miss_data = pd.DataFrame({col_name: train_missing})
        self.barh(miss_data.index.values, miss_data[col_name].values, col_name, map_name)
        print('构造', map_name,'水平直方图')
        print(miss_data)
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
        ResearchQuestionData(self.data)
        pass

    @caltime_p1(LEVEL_0 + '加载训练数据')
    def load(self):
        self.data = pd.read_csv(TRAIN_PATH, encoding='utf-8')


def test_something():
    # data = pd.read_csv(TRAIN_PATH)
    # print(data.columns)
    pass


@caltime_p0(LEVEL_0 + '数据分析所有步骤')
def analysis_run():
    da = AnalysisManager()
    da.start_analysis()


def main():
    # test_something()
    analysis_run()
    pass


if __name__ == '__main__':
    main()
