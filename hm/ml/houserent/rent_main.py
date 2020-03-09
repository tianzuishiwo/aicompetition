import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from common.my_decorator import *
from hm.ml.houserent.data_manager import *
from hm.ml.houserent.rent_config import *
from hm.ml.houserent.model_train import MyModel

# 迷你测试文件
# USE_TRAIN_CSV = RENT_TRAIN_PATH + 'train_1583685793-10000_20_to_10000_19.csv'
# USE_TEST_CSV = RENT_TRAIN_PATH + 'test_1583685793-10000_20_to_10000_19.csv'


USE_TRAIN_CSV = RENT_TRAIN_PATH + 'train_small.csv'  # 训练集特征数据路径
USE_TEST_CSV = RENT_TRAIN_PATH + 'test_small.csv'  # 测试集集特征数据路径

# 源文件
# USE_TRAIN_CSV = RENT_TRAIN_PATH + 'train.csv'
# USE_TEST_CSV = RENT_SOURCE_PATH + 'test.csv'

pd.set_option('mode.chained_assignment', None)


class DataSet(object):
    """数据类（用于传递数据）"""

    def __init__(self):
        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None
        self.X_test = None
        self.X_test_id = None  # 测试集id列
        self.is_load = False  # 模型是否加载（没使用）

    def __str__(self):
        print(f'x_train.shape: {self.x_train.shape}')
        print(f'x_valid.shape: {self.x_valid.shape}')
        print(f'y_train.shape: {self.y_train.shape}')
        print(f'y_valid.shape: {self.y_valid.shape}')
        print(f'x_train.columns: {self.x_train.columns}')
        return ''


class RentController(object):
    """模型控制器"""

    def __init__(self):
        self.data = None  # 训练集
        self.testdf = None  # 测试集
        self.dataset = DataSet()
        self.load_data()

    @caltime_p1('训练集特征-预处理-删除目标列与切分数据')
    def train_pre_process(self):
        y_target = self.data[COL_month_fee]
        train = self.data.drop([COL_month_fee], axis=1)
        print(train.head(3))
        print(y_target.head(3))
        self.split(train, y_target)

    def split(self, train, target):
        """切分训练集和验证集"""
        self.dataset.x_train, self.dataset.x_valid, self.dataset.y_train, self.dataset.y_valid = train_test_split(train,
                                                                                                                  target,
                                                                                                                  test_size=0.2)

    @caltime_p1('加载特征数据')
    def load_data(self):
        self.data = pd.read_csv(USE_TRAIN_CSV)
        self.testdf = pd.read_csv(USE_TEST_CSV)

    def run(self):
        """模型启动入口方法"""
        self.train_pre_process()
        self.dataset.__str__()
        self.test_pre_process()
        model = MyModel(self.dataset)
        model.train()

    @caltime_p1('测试集特征-预处理')
    def test_pre_process(self):
        self.dataset.X_test_id = self.testdf[COL_order_]
        self.dataset.X_test = self.testdf.drop([COL_order_], axis=1)
        print(f'测试集 shape: {self.dataset.X_test.shape}')
        print(f'测试集 列名: {self.dataset.X_test.columns}')
        pass


@caltime_p0('模型整体训练')
def model_train():
    controller = RentController()
    controller.run()


MODE_NAME = 'Xgboost_1583697336.pkl'


@caltime_p0('本地加载模型，获取测试集预测数据')
def load_pkl_and_predict():
    # 加载模型
    estimator = joblib.load(RENT_PKL_PATH + MODE_NAME)
    # 加载特征数据（测试集）
    testpd = pd.read_csv(USE_TEST_CSV, encoding='utf-8')
    X_test_id = testpd[COL_order_]
    X_test = testpd.drop([COL_order_], axis=1)
    # 预测
    y_predict = estimator.predict(X_test)
    result_df = pd.DataFrame(X_test_id)
    result_df['月租金'] = y_predict
    print(f'测试结果文件shape： {result_df.shape}')
    print(f'测试结果文件columns： {result_df.columns}')
    # 保存测试集预测数据
    path = RENT_RESULT_PATH + 'local_pkl_' + str(int(time.time())) + '_' + RENT_RESULT_NAME
    result_df.to_csv(path, encoding='utf-8', index=None)
    print('本地加载模型，预测测试集数据完成！')


def main():
    model_train()
    # load_pkl_and_predict()


if __name__ == '__main__':
    main()
