import time

MIN_TIME = 0.01

# 被装饰方法：有参数 decorator Y func N
def caltime_p0(*decorator_param):
    def median(func):
        def inner():
            des = decorator_param[0]
            start = time.time()
            func()
            end = time.time()
            delta = end - start
            if delta < MIN_TIME:
                delta = 0.0
            print(f'>>>>> {des} 耗时：{delta} 秒')

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime_p1(*decorator_param):
    def median(func):
        def inner(p):
            des = decorator_param[0]
            start = time.time()
            func(p)
            end = time.time()
            delta = end - start
            if delta < MIN_TIME:
                delta = 0.0
            print(f'>>>>> {des} 耗时：{delta} 秒')

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime_p2(*decorator_param):
    def median(func):
        def inner(p1, p2):
            des = decorator_param[0]
            start = time.time()
            func(p1, p2)
            end = time.time()
            delta = end - start
            if delta < MIN_TIME:
                delta = 0.0
            print(f'{des} 耗时：{delta} 秒')

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime_p3(*decorator_param):
    def median(func):
        def inner(p1, p2, p3):
            des = decorator_param[0]
            start = time.time()
            func(p1, p2, p3)
            end = time.time()
            delta = end - start
            if delta < MIN_TIME:
                delta = 0.0
            print(f'{des} 耗时：{delta} 秒')

        return inner

    return median


# 被装饰方法：有参数（无键值对参数）
def caltime2(*decorator_param):
    def median(func):
        def inner(*args):
            des = decorator_param[0]
            start = time.time()
            func(args)
            end = time.time()
            delta = end - start
            if delta < MIN_TIME:
                delta = 0.0
            print(f'{des} 耗时：{delta} 秒')

        return inner

    return median


# 被装饰方法：有参数（含键值对参数）
def caltime3(*decorator_param):
    def median(func):
        def inner(a, *args, **kwargs):
            des = decorator_param[0]
            start = time.time()
            func(a, args, kwargs)
            end = time.time()
            delta = end - start
            if delta < MIN_TIME:
                delta = 0.0
            print(f'{des} 耗时：{delta} 秒')

        return inner

    return median

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
