import pandas as pd

PATH_ROOT = './data/'
PATH_TARGET = './data/small/'
FILE_1 = 'ccf_offline_stage1_train.csv'
FILE_2 = 'ccf_online_stage1_train.csv'


def get_source_path(file_path):
    return PATH_ROOT + file_path


def get_target_path(file_path):
    return PATH_TARGET + file_path


def split_csv(file_path, copy_count):
    print('*' * 25, ' start ', '*' * 25)
    print(f'源样本文件: {get_source_path(file_path)} , 保留样本数量：{copy_count}')
    data_csv = pd.read_csv(get_source_path(file_path))
    print(f'源样本文件shape={data_csv.shape}')
    print(f'源样本文件columns={data_csv.columns}')
    small_data = data_csv.iloc[:copy_count, :]
    print(f'保留样本shape={small_data.shape}')
    small_data.to_csv(get_target_path(file_path))
    print(f'保留样本完成！ {get_target_path(file_path)}')
    print('*' * 25, ' end ', '*' * 25)


def test_split_csv():
    split_csv(FILE_1, 5000)
    split_csv(FILE_2, 5000)


def main():
    test_split_csv()


if __name__ == '__main__':
    main()
