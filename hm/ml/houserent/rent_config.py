DATA_ROOT = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/data/'
RENT_PATH = DATA_ROOT + 'hm/ml/houserent/'

RENT_SOURCE_PATH = RENT_PATH + 'source/'
RENT_SMALL_PATH = RENT_PATH + 'small/'
RENT_TRAIN_PATH = RENT_PATH + 'train/'
RENT_RESULT_PATH = RENT_PATH + 'result/'
RENT_PKL_PATH = RENT_PATH + 'pkl/'

RENT_TRAIN_NAME = 'train.csv'
RENT_TEST_NAME = 'test.csv'
RENT_RESULT_NAME = 'result.csv'  # 结果文件

OUTPUT_RESULT_ENABLE = True
# OUTPUT_RESULT_ENABLE = False

# 源文件
# USE_TRAIN_PATH = RENT_SOURCE_PATH + RENT_TRAIN_NAME
# USE_TEST_PATH = RENT_SOURCE_PATH + RENT_TEST_NAME
# small文件
USE_TRAIN_PATH = RENT_SMALL_PATH + RENT_TRAIN_NAME
USE_TEST_PATH = RENT_SMALL_PATH + RENT_TEST_NAME

# train文件
FEATURE_TRAIN_PATH = RENT_TRAIN_PATH + RENT_TRAIN_NAME
# USE_TEST_PATH = RENT_TRAIN_PATH + RENT_TEST_NAME

# ['Unnamed: 0', '时间', '小区名', '小区房屋出租数量', '楼层',
#  '总楼层', '房屋面积', '房屋朝向','居住状态', '卧室数量',
#  '厅的数量', '卫的数量', '出租方式', '区', '位置',
#  '地铁线路', '地铁站点', '距离','装修情况', '月租金']
#
# [
#     'order_','time','area','area_rent_count','floor',
#     'total_floor','home_area','home_direct','reside_state','room_count',
#     'parlor_count','toilet_count','rent_type','district','location',
#     'metro_num','metro_station','distance','decorate_situation','month_fee',
#  ]