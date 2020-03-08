from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
import time
from sklearn.externals import joblib

from common.my_decorator import *
from hm.ml.houserent.rent_config import *

FORMAT_ARROW = '-' * 20 + '> '

pd.set_option('mode.chained_assignment', None)


# RENT_RESULT_NAME_PATH

class BaseModel(object):
    def __init__(self, data, model_name):
        self.dataset = data
        self.model_name = model_name

    def print_result(self, model_des, accuracy, result):
        print(FORMAT_ARROW, '使用模型：', model_des)
        # print(FORMAT_ARROW, f'训练数据：{len(self.x_train)}', f' 验证数据:{len(self.x_test)}')
        print(FORMAT_ARROW, '准确率：', accuracy)
        print(FORMAT_ARROW, '均平方根误差：', result)
        # self.record_dict[model_des] = auc

    def is_all_traindata(self):
        # self.dataset.x_train
        if self.dataset.x_train.shape[0] > 10 * 10000:
            return True
        return False

    def get_result_csv_path(self):
        path = RENT_RESULT_PATH + str(int(time.time())) + '_' + RENT_RESULT_NAME
        print(f'输出结果文件：{path}')
        return path
        # path = f'{RENT_RESULT_PATH} {str(time.time())}_'

    def _to_result_csv(self, estimator):
        if self.dataset.X_test is not None:
            y_predict = estimator.predict(self.dataset.X_test)
            result_df = pd.DataFrame(self.dataset.X_test_id)
            result_df['月租金'] = y_predict
            print(f'测试结果文件shape： {result_df.shape}')
            print(f'测试结果文件columns： {result_df.columns}')
            if OUTPUT_RESULT_ENABLE:
                result_df.to_csv(self.get_result_csv_path(), index=None)
            else:
                print('暂不打印结果文件！')

    def _save_model(self, estimator):
        all_data_des = ''
        if self.is_all_traindata():
            all_data_des = '_all_data'
        model_name = f'{self.model_name + all_data_des}_{str(int(time.time()))}.pkl'
        path = RENT_PKL_PATH + model_name
        joblib.dump(estimator, path)
        print(f'{self.model_name} 训练模型保存完成！')

    def save_and_result(self, estimator):
        self._to_result_csv(estimator)
        self._save_model(estimator)

    def rmse(self, y, y_pre):
        return np.sqrt(mean_absolute_error(y, y_pre))


class MyRandomForestRegressor(BaseModel):

    @caltime_p1('随机森林回归训练')
    def train(self):
        estimator = RandomForestRegressor(n_estimators=40, min_samples_leaf=30, max_features='sqrt', n_jobs=1)
        estimator.fit(self.dataset.x_train, self.dataset.y_train)
        y_pre = estimator.predict(self.dataset.x_valid)
        accuray = estimator.score(self.dataset.x_valid, self.dataset.y_valid)
        # result = np.sqrt(mean_absolute_error(self.dataset.y_valid, y_pre))
        result = self.rmse(self.dataset.y_valid, y_pre)
        # 测试集 预测标签值
        # if self.dataset.
        # self.to_result_csv()
        self.print_result(self.model_name, accuray, result)
        self.save_and_result(estimator)


class MyLGBMRegressor(BaseModel):

    @caltime_p1('LGBMRegressor训练')
    def train(self):
        estimator = lgb.LGBMRegressor(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=20)
        estimator.fit(self.dataset.x_train, self.dataset.y_train,
                      eval_set=[(self.dataset.x_valid, self.dataset.y_valid)], eval_metric="l1",
                      early_stopping_rounds=5)
        # estimator = lgb.LGBMRegressor(boosting_type='gbdt',
        #                               num_leaves=31,
        #                               max_depth=5,
        #                               learning_rate=0.1,
        #                               n_estimators=100,
        #                               min_child_samples=20,
        #                               n_jobs=-1
        #                               )
        # estimator.fit(self.dataset.x_train, self.dataset.y_train,
        #               eval_set=[(self.dataset.x_valid, self.dataset.y_valid)],
        #               eval_metric='l1',
        #               early_stopping_rounds=5)
        accuracy = estimator.score(self.dataset.x_valid, self.dataset.y_valid)
        y_valid_pre = estimator.predict(self.dataset.x_valid)
        result = self.rmse(self.dataset.y_valid, y_valid_pre)
        self.print_result(self.model_name, accuracy, result)
        self.save_and_result(estimator)


class MyXgboost(BaseModel):

    @caltime_p1('Xgboost训练')
    def train(self):
        estimator = XGBRegressor(learning_rate=0.1, n_estimators=550, max_depth=4, min_child_weight=5, seed=0,
                                 subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)
        estimator.fit(self.dataset.x_train, self.dataset.y_train)
        accuracy = estimator.score(self.dataset.x_valid, self.dataset.y_valid)
        y_valid_pre = estimator.predict(self.dataset.x_valid)
        result = self.rmse(self.dataset.y_valid, y_valid_pre)
        self.print_result(self.model_name, accuracy, result)
        self.save_and_result(estimator)
        pass


class MyModel(object):

    def __init__(self, data=None, is_load=False):
        self.dataset = data
        self.is_load = is_load

    def train(self):
        # MyRandomForestRegressor(self.dataset, 'randomForestRegressor').train()
        # MyLGBMRegressor(self.dataset, 'LGBMRegressor').train()
        MyXgboost(self.dataset, 'Xgboost').train()
