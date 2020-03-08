import pandas as pd
import numpy as np
from hm.ml.pubg.model_train import MyModel
from common.my_decorator import *
from hm.ml.pubg.pubg_config import *
from hm.ml.pubg.data_manager import COL_winPlacePerc
from hm.ml.pubg.data_manager import COL_ID
from sklearn.model_selection import train_test_split

# 迷你测试文件
USE_TRAIN_CSV = PUBG_TRAIN_PATH + '1583625497-10000_30_to_9998_47.csv'
USE_TEST_CSV = PUBG_SMALL_PATH + 'small_test_V2.csv'


# 源文件
# USE_TRAIN_CSV = PUBG_TRAIN_PATH + 'train_V2.csv'
# USE_TEST_CSV = PUBG_SOURCE_PATH + 'test_V2.csv'


class DataSet(object):
    def __init__(self):
        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None
        self.testdf = None
        self.is_load = False  # 模型是否加载

    def __str__(self):
        print(f'x_train.shape: {self.x_train.shape}')
        print(f'x_valid.shape: {self.x_valid.shape}')
        print(f'y_train.shape: {self.y_train.shape}')
        print(f'y_valid.shape: {self.y_valid.shape}')
        print(f'testdf.shape: {self.testdf.shape}')
        return ''


class PubgController(object):
    def __init__(self):
        self.data = None
        self.dataset = DataSet()
        # self.x_train = None
        # self.x_valid = None
        # self.y_train = None
        # self.y_valid = None
        self.load_data()

    @caltime_p1('删除目标列与切分数据')
    def pre_process(self):
        y_target = self.data[COL_winPlacePerc]
        train = self.data.drop([COL_winPlacePerc, COL_ID], axis=1)
        print(train.head(3))
        print(y_target.head(3))
        self.split(train, y_target)

    def split(self, train, target):
        self.dataset.x_train, self.dataset.x_valid, self.dataset.y_train, self.dataset.y_valid = train_test_split(train,
                                                                                                                  target,
                                                                                                                  test_size=0.2)

    @caltime_p1('加载特征数据')
    def load_data(self):
        self.data = pd.read_csv(USE_TRAIN_CSV)
        self.dataset.testdf = pd.read_csv(USE_TEST_CSV)
        # 待处理测试集

    def train(self):
        self.pre_process()
        self.dataset.__str__()
        model = MyModel(self.dataset)
        model.train()


@caltime_p0('模型整体训练')
def test_train():
    controller = PubgController()
    controller.train()


def test_train_load_testdf():
    model = MyModel()
    model.load_model_trian()


def main():
    test_train()
    # test_train_load_testdf()


if __name__ == '__main__':
    main()
