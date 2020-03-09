import time, datetime
# from common.small_tool import get_time_now
# from tianchi.o2o.o2o_config import *

CALCULATE_START = 'recorde_start'
CALCULATE_END = 'recorde_end'
caltime_record = []


# 被装饰方法：有参数 decorator Y func N
def caltime_p0(*decorator_param):
    def median(func):
        def inner():
            start = time.time()
            func()
            print_caltime(start, decorator_param[0])

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime_p1(*decorator_param):
    def median(func):
        def inner(p):
            start = time.time()
            func(p)
            print_caltime(start, decorator_param[0])

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime_p2(*decorator_param):
    def median(func):
        def inner(p1, p2):
            start = time.time()
            func(p1, p2)
            print_caltime(start, decorator_param[0])

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime_p3(*decorator_param):
    def median(func):
        def inner(p1, p2, p3):
            start = time.time()
            func(p1, p2, p3)
            print_caltime(start, decorator_param[0])

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime_p4(*decorator_param):
    def median(func):
        def inner(p1, p2, p3, p4):
            start = time.time()
            func(p1, p2, p3, p4)
            print_caltime(start, decorator_param[0])

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime2(*decorator_param):
    def median(func):
        def inner(*args):
            start = time.time()
            func(args)
            print_caltime(start, decorator_param[0])

        return inner

    return median


# 被装饰方法：有参数（含键值对参数）
def caltime3(*decorator_param):
    def median(func):
        def inner(a, *args, **kwargs):
            start = time.time()
            func(a, args, kwargs)
            print_caltime(start, decorator_param[0])

        return inner

    return median


MIN = 60
HOUR = 60 * 60


def get_delta_des(delta):
    des = ''
    if delta < 0.01:
        des = f'{str(round(delta, 4))}秒'
        # des = '0.0秒'
    elif delta < MIN:
        des = f'{str(round(delta, 2))}秒'
    else:
        delta = int(delta)
        if delta < HOUR:
            des = f' {str(int(delta / MIN))}分{str(delta % MIN)}秒'
        else:
            des = f' {str(int(delta / HOUR))}小时{str(int(delta % HOUR/ MIN))}分{str(delta % MIN)}秒'
    # print(delta, ' ', des)
    return des

# get_delta_des(0.00123346)
# get_delta_des(23.123346)
# get_delta_des(123.123346)
# get_delta_des(789.00123346)
# get_delta_des(2555.00123346)
# get_delta_des(4590.00123346)
# get_delta_des(18234.00123346)


def print_caltime(start, des):
    end = time.time()
    delta = end - start
    # if delta < MIN_TIME:
    #     delta = 0.0
    # delta = round(delta, 2)
    delta_des = get_delta_des(delta)
    func_time_des = f'>>>>>>>>>> {des} 耗时：{delta_des} '
    print(func_time_des)
    # record(func_time_des)


# def record(des):
#     if CALCULATE_END in des:
#         # print('+'*30,' ',CALCULATE_END)
#         file_path = LOG_PATH + str(time.time()) + '.txt'
#         # print(file_path)
#         # print(caltime_record)
#         with open(file_path, 'w') as f:
#             for line in caltime_record:
#                 f.write(line)
#                 f.write('\r\n')
#         caltime_record.clear()
#     caltime_record.append(des)
    # print('-'*25,'添加日志信息')

# decorator N func N
# def caltime(func):
#     def inner():
#         # des = 'des'
#         start = time.time()
#         func()
#         end = time.time()
#         delta = end - start
#         if delta < 0.5:
#             delta = 0.0
#         print(f' 耗时：{delta} 秒')
#
#     return inner
#
#
# # decorator N func Y
# def caltime2(func):
#     def inner(a, *args, **kwargs):
#         # des = 'des'
#         start = time.time()
#         func(a, args, kwargs)
#         end = time.time()
#         delta = end - start
#         if delta < 0.5:
#             delta = 0.0
#         print(f' 耗时：{delta} 秒')
#
#     return inner
