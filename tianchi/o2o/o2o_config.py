
DATA_ROOT = '/Users/wushaohua/my/workplace/ailearn/hm/aicompetition/data/'
O2O_PATH = DATA_ROOT + 'tianchi/o2o/'
SOURCE_PATH = O2O_PATH + 'source/'
SMALL_PATH = O2O_PATH + 'small/'
MEDIAN_PATH = O2O_PATH + 'median/'
BIG_PATH = O2O_PATH + 'big/'

OFFLINE_TRAIN_NAME = 'ccf_offline_stage1_train.csv'
ONLINE_TRAIN_NAME = 'ccf_online_stage1_train.csv'


# 所有参数配置
OFFLINE_SPLIT_DATA = 2000000
ONLINE_SPLIT_DATA = 2000000

# USE_WHICH_DATA = O2O_PATH  # data 下的源数据
USE_WHICH_DATA = SMALL_PATH  # small 文件下的小数据
# USE_WHICH_DATA = MEDIAN_PATH  # median 文件下的小数据
# USE_WHICH_DATA = O2O_PATH


# USE_OFFLINE_TRAIN_CSV = O2O_SMALL_PATH + OFFLINE_TRAIN_NAME
# USE_ONLINE_TRAIN_CSV = O2O_SMALL_PATH + ONLINE_TRAIN_NAME

USE_OFFLINE_TRAIN_CSV = USE_WHICH_DATA + OFFLINE_TRAIN_NAME
USE_ONLINE_TRAIN_CSV = USE_WHICH_DATA + ONLINE_TRAIN_NAME
COUNT = 5  # log中打印dataframe数量

VALUE_F_1 = -1
VALUE_0 = 0
VALUE_1 = 1
VALUE_2 = 2
VALUE_3 = 3
FORMART_EMPTY = 'empty'
FORMART_DOT = '.'
FORMART_COLON = ':'
FORMART_FIXED = 'fixed'

COLUMN_Coupon_id = 'Coupon_id'
COLUMN_User_id = 'User_id'
COLUMN_Merchant_id = 'Merchant_id'
COLUMN_Discount_rate = 'Discount_rate'
COLUMN_Action = 'Action'
COLUMN_Distance = 'Distance'
COLUMN_Date = 'Date'
COLUMN_Date_received = 'Date_received'
COLUMN_day = 'day'
COLUMN_hour = 'hour'
COLUMN_weekday = 'weekday'
COLUMN_discount_fixed = 'discount_fixed'
COLUMN_discount_ratio = 'discount_ratio'
COLUMN_discount_satisfy = 'discount_satisfy'
COLUMN_sample_type = 'sample_type'
COLUMN_discount_type = 'discount_type'
COLUMN_user_type = 'user_type'
COLUMN_use_coupon = 'use_coupon'
COLUMN_use_coupon_15day = 'use_coupon_15day'
TRAIN_TARGET_COLUMN = 'use_coupon_15day'


TRAIN_COLUMNS = [
                 # COLUMN_User_id,
                 # COLUMN_Merchant_id,
                 COLUMN_Action,
                 COLUMN_Distance,
                 COLUMN_day,
                 COLUMN_hour,
                 COLUMN_weekday,
                 COLUMN_discount_fixed,
                 COLUMN_discount_ratio,
                 COLUMN_discount_satisfy,
                 COLUMN_sample_type,
                 COLUMN_user_type
                ]



# TRAIN_COLUMNS = ['User_id', 'Merchant_id', 'Action', 'Distance', 'day', 'hour', 'weekday',
#                  'discount_fixed', 'discount_ratio', 'discount_satisfy', 'sample_type', 'user_type']
# TRAIN_TARGET_COLUMN = 'use_coupon_15day'