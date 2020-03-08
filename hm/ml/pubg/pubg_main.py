import pandas as pd
import numpy as np
from hm.ml.pubg.model_train import MyModel
from common.my_decorator import *


class PubgController(object):
    def __init__(self):
        self.data = None
        self.load_data()
        pass

    def pre_process(self):
        pass

    @caltime_p1('加载特征数据')
    def load_data(self):
        pass

    @caltime_p1('模型整体训练')
    def train(self):
        self.pre_process()
        model = MyModel(self.data)
        model.train()


def main():
    controller = PubgController()
    controller.train()


if __name__ == '__main__':
    main()
