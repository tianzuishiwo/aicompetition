
import time
from collections import Counter
from sklearn.preprocessing import OneHotEncoder


def get_stime(input_time):
    # print(f'input_time={input_time}')
    return time.mktime(time.strptime(str(int(input_time)), '%Y%m%d'))


def is_bigger_15day(before_time, after_time):
    btime = get_stime(before_time)
    atime = get_stime(after_time)
    if atime > btime and ((atime - btime) < (15 * 24 * 60 * 60)):
        return True
    return False


def convert_value(str_value):
    if str_value is not None:
        splits = str_value.split(':')
        value = round(float(splits[0]) / float(splits[1]), 1)
        return value
    return 0



def column_one_hot(input_df, column_name):
    one_hot_encoder = OneHotEncoder()
    onehot_column = input_df[column_name].values.reshape(-1, 1)
    input_df[column_name] = one_hot_encoder.fit_transform(onehot_column)


def columns_one_hot(input_df, column_names):
    one_hot_encoder = OneHotEncoder()
    for column_name in column_names:
        onehot_column = input_df[column_name].values.reshape(-1, 1)
        input_df[column_name] = one_hot_encoder.fit_transform(onehot_column)

