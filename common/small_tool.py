import pandas as pd
import os
from tianchi.o2o.o2o_config import *
from common.my_decorator import *

# SOURCE_PATH = O2O_PATH
# TARGET_PATH = O2O_SMALL_PATH


class CsvSpliter(object):
    def __init__(self, file_name, small, median=None, big=None):
        self.file_name = file_name
        self.small = small
        self.median = median
        self.big = big
        self.source_csv = None
        self.small_csv = None
        self.median_csv = None
        self.big_csv = None
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
        self.median_csv = self._csv_split(self.source_csv, self.median)
        self.big_csv = self._csv_split(self.source_csv, self.big)

    def _csv_split(self, source_csv, pos):
        if pos is not None:
            return source_csv.iloc[:pos, :]
        return None

    def print_all_info(self):
        self.print_sourcecsv_info(False)
        self.print_csv('source', self.source_csv)
        self.print_csv('samll', self.small_csv)
        self.print_csv('median', self.median_csv)
        self.print_csv('big', self.big_csv)

    def print_csv(self, des, csv):
        if csv is not None:
            print(f' {des}: shape={csv.shape} size={csv.size}')

    def get_source_path(self):
        return self._get_join_path(SOURCE_PATH, self.file_name)

    def get_small_path(self):
        return self._get_join_path(SMALL_PATH, self.file_name)

    def get_median_path(self):
        return self._get_join_path(MEDIAN_PATH, self.file_name)

    def get_big_path(self):
        return self._get_join_path(BIG_PATH, self.file_name)

    def generate_all_csv(self):
        self.generate_new_csv(self.source_csv, self.small, self.get_small_path())
        self.generate_new_csv(self.source_csv, self.median, self.get_median_path())
        self.generate_new_csv(self.source_csv, self.big, self.get_big_path())

    def generate_new_csv(self, csv, count, target_file):
        if csv is not None:
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
            print('csv is null! generate new csv fail!!!')

    def _get_join_path(self, dir_path, file_name):
        return os.path.join(dir_path, file_name)

    def auto(self):
        # self.read_csv()
        self.generate_all_csv()
        self.print_all_info()


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


def test_Csv():
    spliter1 = CsvSpliter(OFFLINE_TRAIN_NAME, 30, 300, 3000)
    spliter2 = CsvSpliter(ONLINE_TRAIN_NAME, 50, 500, 5000)
    # spliter1.print_sourcecsv_info()
    # spliter2.print_sourcecsv_info()
    spliter1.auto()
    spliter2.auto()


def main():
    test_Csv()


if __name__ == '__main__':
    main()
