import time


def get_cur_timedes():
    ft1 = '%Y%m%d-%H_%M_%S'
    return time.strftime(ft1)


def test_print_time():
    # ft1 = '%Y-%m-%d_%H%M%S'
    # ft1 = '%Y%m%d-%H_%M_%S'
    # print("1: ", time.strftime(ft1))
    # print("2: ", time.strftime(ft1), time.localtime(time.time()))
    print(get_cur_timedes())
    pass


def main():
    test_print_time()
    pass


if __name__ == '__main__':
    main()
