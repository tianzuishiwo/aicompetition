import pandas as pd
from sklearn.decomposition import PCA
from data.test import *
import numpy as np
import os
import sys

PATH_ROOT = './data/'
PATH_TARGET = './data/small/'
FILE_1 = 'ccf_offline_stage1_train.csv'
FILE_2 = 'ccf_online_stage1_train.csv'

DATA_ROOT_PAHT = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/data/test'


def get_source_path(file_path):
    return PATH_ROOT + file_path


def get_target_path(file_path):
    return PATH_TARGET + file_path


def test_split_csv():
    data = pd.read_csv('train_data.csv')
    # data=pd.read_excel('train_data.xls')
    print(f'data.shape={data.shape}')
    pca = PCA(n_components=0.9)
    data_new = pca.fit_transform(data)
    print(f'data_new.shape={data_new.shape}')


DATA_FILE_ROOT = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/data/test/'


def test_outer_data():
    # print(sys.path)
    print('test_outer_data')
    # data = pd.DataFrame(np.random.normal(0, 1, [3, 3]))
    path = os.path.join(DATA_FILE_ROOT, 'test_data.csv')
    data = pd.read_csv(path)
    print('路径： ', path)
    print(data)
    # data.to_csv(path)
    print('测试 ok')


def main():
    # test_split_csv()
    test_outer_data()
    pass


if __name__ == '__main__':
    main()
