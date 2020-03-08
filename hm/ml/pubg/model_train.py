from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common.my_decorator import *


class BaseModel(object):
    def __init__(self, data):
        self.dataset = data

    def print_result(self, model_des, accuracy, result):
        print(FORMAT_ARROW, '使用模型：', model_des)
        # print(FORMAT_ARROW, f'训练数据：{len(self.x_train)}', f' 验证数据:{len(self.x_test)}')
        print(FORMAT_ARROW, '准确率：', accuracy)
        print(FORMAT_ARROW, '平均绝对误差：', result)
        # self.record_dict[model_des] = auc

    def save_model(self):

        pass


class MySvm(BaseModel):
    pass
    # def __init__(self):
    #     pass


class MyRandomForestRegressor(BaseModel):

    @caltime_p1('随机森林回归训练')
    def train(self):
        estimator = RandomForestRegressor(n_estimators=40, min_samples_leaf=30, max_features='sqrt', n_jobs=1)
        estimator.fit(self.dataset.x_train, self.dataset.y_train)
        y_pre = estimator.predict(self.dataset.x_valid)
        accuray = estimator.score(self.dataset.x_valid, self.dataset.y_valid)
        result = mean_absolute_error(self.dataset.y_valid, y_pre)
        self.print_result('MyRandomForestRegressor', accuray, result)


class MyModel(object):

    def __init__(self, data=None,is_load = False):
        self.dataset = data
        self.is_load = is_load

    def train(self):
        MyRandomForestRegressor(self.dataset).train()

    def load_model_trian(self):
        pass
