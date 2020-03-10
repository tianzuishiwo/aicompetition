from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import numpy as np
import pandas as pd
import time
from sklearn.externals import joblib

from common.my_decorator import *
from hm.ml.houserent.rent_config import *

FORMAT_ARROW = '-' * 20 + '> '

pd.set_option('mode.chained_assignment', None)


class BaseModel(object):
    def __init__(self, data, model_name):
        self.dataset = data
        self.model_name = model_name

    def print_result(self, model_des, accuracy, result):
        print(FORMAT_ARROW, '使用模型：', model_des)
        print(FORMAT_ARROW, '准确率：', accuracy)
        print(FORMAT_ARROW, '均平方根误差：', result)

    def is_all_traindata(self):
        """是否使用全部数据训练"""
        if self.dataset.x_train.shape[0] > 10 * 10000:
            return True
        return False

    def get_result_csv_path(self):
        path = RENT_RESULT_PATH + str(int(time.time())) + '_' + RENT_RESULT_NAME
        print(f'输出结果文件：{path}')
        return path

    def _to_result_csv(self, estimator):
        """生成结果文件"""
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
        """保存模型"""
        all_data_des = ''
        if self.is_all_traindata():
            all_data_des = '_all_data'
        model_name = f'{self.model_name + all_data_des}_{str(int(time.time()))}.pkl'
        path = RENT_PKL_PATH + model_name
        joblib.dump(estimator, path)
        print(f'{self.model_name} 训练模型保存完成！')

    def save_and_result(self, estimator):
        """保存模型和生成结果文件"""
        self._to_result_csv(estimator)
        self._save_model(estimator)

    def rmse(self, y, y_pre):
        """均方根误差"""
        return np.sqrt(mean_absolute_error(y, y_pre))


class MyLightGBM(BaseModel):

    @caltime_p1('LightGBM优化')
    def train(self):
        model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=900,
                                      learning_rate=0.1, n_estimators=3141, bagging_fraction=0.7,
                                      feature_fraction=0.6, reg_alpha=0.3, reg_lambda=0.3,
                                      min_data_in_leaf=18, min_sum_hessian_in_leaf=0.001)

        model_lgb.fit(self.dataset.x_train, self.dataset.y_train)
        accuracy = model_lgb.score(self.dataset.x_valid, self.dataset.y_valid)
        y_valid_pre = model_lgb.predict(self.dataset.x_valid)
        result = self.rmse(self.dataset.y_valid, y_valid_pre)
        # result = self.rmse(self.dataset.y_valid, self.dataset.x_valid)
        self.print_result(self.model_name, accuracy, result)
        # # model_lgb.fit(train, train_label)
        print(f'accuracy: {accuracy}')
        pass


class MyRandomForestRegressor(BaseModel):
    """回归森林类"""

    @caltime_p1('随机森林回归训练')
    def train(self):
        estimator = RandomForestRegressor(n_estimators=40, min_samples_leaf=30, max_features='sqrt', n_jobs=1)
        estimator.fit(self.dataset.x_train, self.dataset.y_train)
        y_pre = estimator.predict(self.dataset.x_valid)
        accuray = estimator.score(self.dataset.x_valid, self.dataset.y_valid)
        result = self.rmse(self.dataset.y_valid, y_pre)
        self.print_result(self.model_name, accuray, result)
        self.save_and_result(estimator)


class MyLGBMRegressor(BaseModel):
    """LGBMRegressor"""

    @caltime_p1('LGBMRegressor训练')
    def train(self):
        estimator = lgb.LGBMRegressor(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=20)
        estimator.fit(self.dataset.x_train, self.dataset.y_train,
                      eval_set=[(self.dataset.x_valid, self.dataset.y_valid)], eval_metric="l1",
                      early_stopping_rounds=5)
        accuracy = estimator.score(self.dataset.x_valid, self.dataset.y_valid)
        y_valid_pre = estimator.predict(self.dataset.x_valid)
        result = self.rmse(self.dataset.y_valid, y_valid_pre)
        self.print_result(self.model_name, accuracy, result)
        self.save_and_result(estimator)


class MyXgboostOptimize(BaseModel):
    """xgboost模型优化类"""

    @caltime_p1('MyXgboostOptimize训练')
    def train(self):
        """
        调试最佳参数：
        n_estimators=750,
        max_depth=7，
        learning_rate=0.15，
        min_child_weight= 4，
        gamma= 0.13 第二次训练0.17怀疑过拟合
        """
        param_grid = {
            # 'n_estimators': [550, 650, 720, 800, 880]
            # 'max_depth': [3, 4, 5, 7]
            # 'learning_rate': [0.05, 0.1, 0.15]
            # 'min_child_weight': [4, 5, 6]
            # 'gamma':[0.07,0.1,0.13]
            # 'gamma':[0.13,0.15,0.17]
        }

        estimator = XGBRegressor(
            n_estimators=750,
            learning_rate=0.15,
            max_depth=7,
            min_child_weight=4,
            gamma=0.13,
            seed=0, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=1)

        grid_search_cv = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search_cv.fit(self.dataset.x_train, self.dataset.y_train)

        accuracy = grid_search_cv.score(self.dataset.x_valid, self.dataset.y_valid)
        y_valid_pre = grid_search_cv.predict(self.dataset.x_valid)
        result = self.rmse(self.dataset.y_valid, y_valid_pre)
        self.print_result(self.model_name, accuracy, result)
        print(f'参数的最佳取值: {grid_search_cv.best_params_}')
        print(f'最佳模型得分: {grid_search_cv.best_score_}')
        pass


class MyXgboostBest(BaseModel):
    """xgboost最优参数模型类"""

    @caltime_p1('MyXgboostBest训练')
    def train(self):
        estimator = XGBRegressor(
            n_estimators=750,
            learning_rate=0.15,
            max_depth=7,
            min_child_weight=4,
            gamma=0.13,
            reg_alpha=1, seed=0, subsample=0.7, colsample_bytree=0.7, reg_lambda=1)
        estimator.fit(self.dataset.x_train, self.dataset.y_train)
        accuracy = estimator.score(self.dataset.x_valid, self.dataset.y_valid)
        y_valid_pre = estimator.predict(self.dataset.x_valid)
        result = self.rmse(self.dataset.y_valid, y_valid_pre)
        self.print_result(self.model_name, accuracy, result)
        self.save_and_result(estimator)


class MyModel(object):
    """模型训练类"""

    def __init__(self, data=None, is_load=False):
        self.dataset = data
        self.is_load = is_load

    def train(self):
        """模型训练启动方法"""
        # MyXgboostOptimize(self.dataset, 'Xgboost-optimize').train()  # xgboost调优类
        # MyXgboostBest(self.dataset, 'Xgboost-best').train()  # xgboost最优类（直接跑最优结果）
        MyLightGBM(self.dataset, 'LightGBM-best').train()

        # MyRandomForestRegressor(self.dataset, 'randomForestRegressor').train() # 回归森林测试
        # MyLGBMRegressor(self.dataset, 'LGBMRegressor').train()  # LGBMRegressor测试
