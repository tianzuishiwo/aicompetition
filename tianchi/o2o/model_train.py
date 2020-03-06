from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# from sklearn.linear_model import  LinearRegression

from sklearn.metrics import roc_auc_score
from common.my_decorator import *
from tianchi.o2o.o2o_config import *


class ModelController(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.record_dict = {}

    @caltime_p1('全部模型训练')
    def train(self):
        # self.svm()
        # self.logistic_regression()
        # self.adaboost_classifier()
        # self.xgb_classifier()
        self.decision_tree_classifier()
        self.random_forest_classifier()
        print('模型训练结果：', self.record_dict)

    @caltime_p1('svm训练')
    def svm(self):
        svm = SVC()
        svm.fit(self.x_train, self.y_train)
        y_predict = svm.predict(self.x_test)
        accuracy = svm.score(self.x_test, self.y_test)
        auc = roc_auc_score(self.y_test, y_predict)
        self.print_auc('svm', accuracy, auc)

    @caltime_p1('DecisionTreeClassifier训练')
    def decision_tree_classifier(self):
        classifier = DecisionTreeClassifier()
        classifier.fit(self.x_train, self.y_train)
        accuracy = classifier.score(self.x_test, self.y_test)
        y_predict = classifier.predict(self.x_test)
        auc = roc_auc_score(self.y_test, y_predict)
        self.print_auc('DecisionTreeClassifier', accuracy, auc)

    # LogisticRegression
    @caltime_p1('LogisticRegression训练')
    def logistic_regression(self):
        estimator = LogisticRegression()
        estimator.fit(self.x_train, self.y_train)
        accuracy = estimator.score(self.x_test, self.y_test)
        y_predict = estimator.predict(self.x_test)
        auc = roc_auc_score(self.y_test, y_predict)
        self.print_auc('LogisticRegression', accuracy, auc)

    # RandomForestClassifier
    @caltime_p1('RandomForestClassifier训练')
    def random_forest_classifier(self):
        estimator = RandomForestClassifier()
        estimator.fit(self.x_train, self.y_train)
        accuracy = estimator.score(self.x_test, self.y_test)
        y_predict = estimator.predict(self.x_test)
        auc = roc_auc_score(self.y_test, y_predict)
        self.print_auc('RandomForestClassifier', accuracy, auc)

    # AdaBoostClassifier
    @caltime_p1('AdaBoostClassifier')
    def adaboost_classifier(self):
        estimator = AdaBoostClassifier()
        estimator.fit(self.x_train, self.y_train)
        accuracy = estimator.score(self.x_test, self.y_test)
        y_predict = estimator.predict(self.x_test)
        auc = roc_auc_score(self.y_test, y_predict)
        self.print_auc('AdaBoostClassifier', accuracy, auc)


    # XGBClassifier
    @caltime_p1('XGBClassifier')
    def xgb_classifier(self):
        estimator = XGBClassifier()
        estimator.fit(self.x_train,self.y_train)
        accuracy = estimator.score(self.x_test,self.y_test)
        y_predict = estimator.predict(self.x_test)
        auc = roc_auc_score(self.y_test,y_predict)
        self.print_auc('XGBClassifier',accuracy,auc)



    def print_auc(self, model_des, accuracy, auc):
        print(FORMAT_ARROW, '使用模型：', model_des)
        print(FORMAT_ARROW, f'训练数据：{len(self.x_train)}', f' 测试数据={len(self.x_test)}')
        print(FORMAT_ARROW, '准确率：', accuracy)
        print(FORMAT_ARROW, 'auc：', auc)
        self.record_dict[model_des] = auc
