BATCH_SIZE = 16*1
# batch_size = 16 * 1
# epochs = 6
# EPOCHS = 10
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224, 3)
DATA_RATE = 1
# DATA_RATE = 0.5

WARMUP_EPOCH = 5

ROOT_MODEL_FILE_PATH = './hdf5/'
MODEL_FILE_PATH = ROOT_MODEL_FILE_PATH+'weights.{epoch:02d}-acc_{val_acc:.2f}.hdf5'
MONITOR_VALUE = 'val_acc'
LOG_DIR = 'logs'

ROOT_SMODEL_PAHT = './saved_model/'