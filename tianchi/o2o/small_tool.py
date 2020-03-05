import pandas as pd
import os
from tianchi.o2o.o2o_config import *

SOURCE_PATH = O2O_PATH
TARGET_PATH = O2O_SMALL_PATH


def get_source_path(file_name):
    return get_join_path(SOURCE_PATH, file_name)


def get_target_path(file_name):
    return get_join_path(TARGET_PATH, file_name)


def get_join_path(dir_path, file_name):
    return os.path.join(dir_path, file_name)


def split_csv(file_name, copy_count):
    print('*' * 25, ' start ', '*' * 25)
    print(f'源样本文件: {get_source_path(file_name)} , 保留样本数量：{copy_count}')
    data_csv = pd.read_csv(get_source_path(file_name))
    print(f'源样本文件shape={data_csv.shape}')
    print(f'源样本文件columns={data_csv.columns}')
    small_data = data_csv.iloc[:copy_count, :]
    print(f'保留样本shape={small_data.shape}')
    small_data.to_csv(get_target_path(file_name))
    print(f'保留样本完成！ {get_target_path(file_name)}')
    print('*' * 25, ' end ', '*' * 25)


def test_split_csv():
    split_csv(OFFLINE_TRAIN_NAME, 5000)
    split_csv(ONLINE_TRAIN_NAME, 5000)


def main():
    test_split_csv()


if __name__ == '__main__':
    main()
