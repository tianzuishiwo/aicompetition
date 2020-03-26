ROOT_DATA_PATH = '/Users/wushaohua/my/hm/数据集/深度学习/Sign-Language-Digits-Dataset-master/Dataset/'

ROOT_MODEL_FILE_PATH = './hdf5/'
MODEL_WEIGHT_NAME = '_weights.{epoch:02d}-acc_{val_acc:.2f}.hdf5'
# MODEL_FILE_PATH = ROOT_MODEL_FILE_PATH + '_weights.{epoch:02d}-acc_{val_acc:.2f}.hdf5'
MONITOR_VALUE = 'val_acc'
LOG_DIR = 'logs'
ROOT_SMODEL_PAHT = './saved_model/'

BATCH_SIZE = 16 * 1
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224, 3)
DATA_RATE = 1
WARMUP_EPOCH = 5

"""测试时使用"""
WARMUP_EPOCH = 2
EPOCHS = 3
BATCH_SIZE = 16 * 1
DATA_RATE = 0.1
