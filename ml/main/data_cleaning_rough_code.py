import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # 忽略一些警告

TRAIN_PATH = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/ml/main/data/source/train.csv'
TEST_PATH = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/ml/main/data/source/test.csv'

space_threshold = 0.3

dist_value_for_fill = 2  # 为什么是2,因为距离的最大值是1,没有地铁 意味着很远

line_value_for_fill = 0

station_value_for_fill = 0

state_value_for_fill = 0  # train["居住状态"].mode().values[0]

decration_value_for_fill = -1  # train["装修情况"].mode().values[0]

rent_value_for_fill = -1  # train["出租方式"].mode().values[0]

class FuckCode(object):
    def __init__(self):
        self.train = None

    def fuck(self):
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        self.step6()
        pass

    def step1(self):
        self.step11()
        self.step12()
        pass

    def step2(self):
        self.step21()
        self.step22()
        self.step23()
        self.step24()
        self.step25()
        pass

    def step3(self):
        pass

    def step4(self):
        pass

    def step5(self):
        pass

    def step6(self):
        pass

    def step11(self):
        train = pd.read_csv(TRAIN_PATH)
        train.head()
        train.info()
        print(train.shape)
        # 出租方式中有很多缺失值

        train["出租方式"].value_counts()
        train["装修情况"].value_counts()
        train["居住状态"].value_counts()

        pass

    def step12(self):
        # 1.2  设置后面要用的填充量 (已经放到最上面了)
        # 拿到每个区的位置众数

        area_value_for_fill = train["区"].mode().values[0]
        position_by_area = train.groupby('区').apply(lambda x: x["位置"].mode())
        # print(position_by_area)

        position_value_for_fill = position_by_area[position_by_area.index ==
                                                   area_value_for_fill].values[0][0]
        # print(position_value_for_fill)

        # 拿到每个小区房屋出租数量的众数

        ratio_by_neighbor = train.groupby('小区名').apply(lambda x: x["小区房屋出租数量"].mode())
        index = [x[0] for x in ratio_by_neighbor.index]
        ratio_by_neighbor.index = index
        ratio_by_neighbor = ratio_by_neighbor.to_dict()
        ratio_mode = train["小区房屋出租数量"].mode().values[0]

        pass

    def step21(self):
        train_missing = (train.isnull().sum() / len(train)) * 100
        train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending=False)
        da= pd.DataFrame({'缺失百分比': train_missing})
        print(da)
        pass

    def step22(self):
        pass

    def step23(self):
        pass

    def step24(self):
        pass

    def step25(self):
        pass


def main():
    FuckCode().fuck()


if __name__ == '__main__':
    main()