import pandas as pd
import numpy as np
import os
import datetime
from collections import Counter
from tianchi.o2o.o2o_config import *
from common.my_decorator import *
from hm.ml.pubg.pubg_config import *


# SOURCE_PATH = O2O_PATH
# TARGET_PATH = O2O_SMALL_PATH

def get_time_now():
    return datetime.datetime.now()


def print_counter(series, des="单列成分分析："):
    series = series.replace(np.NaN, 0)
    counter = Counter(series)
    print(des, counter)


def print_counter_allcols(input_df, column_names):
    for column_name in column_names:
        counter = Counter(input_df[column_name])
        print(counter)


class CsvSpliter(object):
    def __init__(self, file_name, small, median=None, big=None, load_data=True, source_path=None, small_path=None):
        self.file_name = file_name
        self.small = small
        self.median = median
        self.big = big
        self.source_csv = None
        self.small_csv = None
        self.median_csv = None
        self.big_csv = None
        self.source_path = source_path
        self.small_path = small_path
        self.median_path = None
        self.big_path = None
        # self.set_path(SOURCE_PATH, SMALL_PATH, MEDIAN_PATH, BIG_PATH)
        if load_data:
            self.read_csv()

    def print_sourcecsv_info(self, print_head=True):
        if self.source_csv is not None:
            print(f'源样本名称：{self.file_name} shape={self.source_csv.shape} size={self.source_csv.size}')
            print(f'源样本列名({len(self.source_csv.columns)})：{self.source_csv.columns}')
            if print_head:
                print(self.source_csv.head(5))

    @caltime_p1('读取源文件')
    def read_csv(self):
        self.source_csv = pd.read_csv(self.get_source_path())
        self.small_csv = self._csv_split(self.source_csv, self.small)
        if self.median_path is not None:
            self.median_csv = self._csv_split(self.source_csv, self.median)
        if self.big_path is not None:
            self.big_csv = self._csv_split(self.source_csv, self.big)

    def _csv_split(self, source_csv, pos):
        if pos is not None:
            return source_csv.iloc[:pos, :]
        return None

    def print_all_info(self):
        self.print_sourcecsv_info(False)
        self.print_csv('source', self.source_csv)
        self.print_csv('samll', self.small_csv)
        if self.median_path_not_empty():
            self.print_csv('median', self.median_csv)
        if self.big_path_not_empty():
            self.print_csv('big', self.big_csv)

    def print_csv(self, des, csv):
        if csv is not None:
            print(f' {des}: shape={csv.shape} size={csv.size}')

    def get_source_path(self):
        return self._get_join_path(self.source_path, self.file_name)

    def get_small_path(self):
        return self._get_join_path(self.small_path, self.file_name)

    def get_median_path(self):
        return self._get_join_path(self.median_path, self.file_name)

    def get_big_path(self):
        return self._get_join_path(self.big_path, self.file_name)

    def generate_all_csv(self):
        self.generate_new_csv(self.source_csv, self.small, self.get_small_path())
        if self.median_path_not_empty():
            self.generate_new_csv(self.source_csv, self.median, self.get_median_path())
        if self.big_path_not_empty():
            self.generate_new_csv(self.source_csv, self.big, self.get_big_path())

    def median_path_not_empty(self):
        return self.median_path is not None

    def big_path_not_empty(self):
        return self.big_path is not None

    @caltime_p4('单份样本切割')
    def generate_new_csv(self, csv, count, target_file):
        if csv is not None and (count is not None):
            new_csv = csv.iloc[:count, :]
            new_csv.to_csv(target_file)
            line_count = 30
            print('-' * line_count, '开始', '-' * line_count)
            print(f'保留样本：{target_file}')
            print(f'切割： shape={csv.shape} -----> shape={new_csv.shape}')
            print(f'切割： size={csv.size} -----> size={new_csv.size}')
            print(f'保留样本完成!')
            print('-' * line_count, '结束', '-' * line_count)
        else:
            if csv is None:
                print('csv is null! generate new csv fail!!!')

    def _get_join_path(self, dir_path, file_name):
        return os.path.join(dir_path, file_name)

    def auto(self):
        # self.read_csv()
        self.generate_all_csv()
        self.print_all_info()

    def set_path(self, source_path, small_path, median_path=None, big_path=None):
        self.source_path = source_path
        self.small_path = small_path
        self.median_path = median_path
        self.big_path = big_path
        # self.source_path = SOURCE_PATH
        # self.small_path = SMALL_PATH
        # self.median_path = MEDIAN_PATH
        # self.big_path = BIG_PATH


# def split_csv(file_name, copy_count):
#     print('*' * 25, ' start ', '*' * 25)
#     print(f'源样本文件: {get_source_path(file_name)} , 保留样本数量：{copy_count}')
#     data_csv = pd.read_csv(get_source_path(file_name))
#     print(f'源样本文件shape={data_csv.shape}')
#     print(f'源样本文件columns={data_csv.columns}')
#     small_data = data_csv.iloc[:copy_count, :]
#     print(f'保留样本shape={small_data.shape}')
#     small_data.to_csv(get_target_path(file_name))
#     print(f'保留样本完成！ {get_target_path(file_name)}')
#     print('*' * 25, ' end ', '*' * 25)


# def test_split_csv():
#     split_csv(OFFLINE_TRAIN_NAME, 5000)
#     split_csv(ONLINE_TRAIN_NAME, 5000)


SMALL_COUNT = 1 * 10000
MEDIAN_COUNT = 50 * 10000
BIG_COUNT = 100 * 10000


@caltime_p0('切割所有样本')
def test_Csv():
    # spliter1 = CsvSpliter(OFFLINE_TRAIN_NAME, SMALL_COUNT, MEDIAN_COUNT, BIG_COUNT)
    # spliter2 = CsvSpliter(ONLINE_TRAIN_NAME, SMALL_COUNT, MEDIAN_COUNT, BIG_COUNT)
    # spliter1 = CsvSpliter(OFFLINE_TRAIN_NAME, SMALL_COUNT, MEDIAN_COUNT)
    # spliter2 = CsvSpliter(ONLINE_TRAIN_NAME, SMALL_COUNT, MEDIAN_COUNT)
    spliter1 = CsvSpliter(PUBG_TRAIN_NAME, SMALL_COUNT, source_path=PUBG_SOURCE_PATH, small_path=PUBG_SMALL_PATH)
    spliter2 = CsvSpliter(PUBG_TEST_NAME, SMALL_COUNT, source_path=PUBG_SOURCE_PATH, small_path=PUBG_SMALL_PATH)
    spliter1.auto()
    spliter2.auto()


def test_counter():
    spliter1 = CsvSpliter(OFFLINE_TRAIN_NAME, SMALL_COUNT, load_data=False)
    data = pd.read_csv(spliter1.get_small_path())
    print(data.shape)
    print_counter(data[COLUMN_Date])
    # series=data[COLUMN_Date].replace(np.NaN,0)
    # counter = Counter(series)
    # print(counter)
    # print_counter(data[COLUMN_Date])


def test_time():
    print(get_time_now())


def main():
    test_Csv()
    # test_counter()
    # test_time()


if __name__ == '__main__':
    main()
