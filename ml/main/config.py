# ROOT_PATH = './data/'
ROOT_PATH = 'data/'

SOURCE_PATH = ROOT_PATH + 'source/'
CLEAN_PATH = ROOT_PATH + 'clean/'
FEATURE_PATH = ROOT_PATH + 'feature/'
PKL_PATH = ROOT_PATH + 'pkl/'
RESULT_PATH = ROOT_PATH + 'result/'
SMALL_PATH = ROOT_PATH + 'small/'

TRAIN_NAME = 'train.csv'
TEST_NAME = 'test.csv'

"""
['时间', '小区名', '小区房屋出租数量', '楼层', '总楼层', '房屋面积', '房屋朝向', '居住状态', '卧室数量',
       '厅的数量', '卫的数量', '出租方式', '区', '位置', '地铁线路', '地铁站点', '距离', '装修情况', '月租金']
"""

COL_index = 'index'
COL_order_ = 'id'
COL_time = '时间'
COL_area = '小区名'  # 小区名
COL_area_rent_count = '小区房屋出租数量'
COL_floor = '楼层'
COL_total_floor = '总楼层'
COL_home_area = '房屋面积'
COL_home_direct = '房屋朝向'
COL_reside_state = '居住状态'
COL_room_count = '卧室数量'
COL_parlor_count = '厅的数量'
COL_toilet_count = '卫的数量'
COL_rent_type = '出租方式'
COL_district = '区'
COL_location = '位置'
COL_metro_num = '地铁线路'
COL_metro_station = '地铁站点'
COL_distance = '距离'
COL_decorate_situation = '装修情况'
COL_month_fee = '月租金'

# 以下新增
COL_room_area = 'room_area'
COL_total_rtp = 'total_room_toilet_parlor'  # 房+卫+厅
COL_room_rate = 'room_rate'  # 房/总
COL_toilet_rate = 'toilet_rate'  # 卫/总
COL_parlor_rate = 'parlor_rate'  # 厅/总
COL_floor_rate = 'floor_rate'  # 楼层比
COL_home_type = 'home_type'  # 户型
COL_metro_exist = 'metro_exist'  # 有地铁
COL_location_path_count = 'location_path_count'  # 位置线路数
COL_area_path_count = 'area_path_count'  # 小区线路数











