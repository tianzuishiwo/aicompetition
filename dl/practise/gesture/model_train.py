import tensorflow as tf
from dl.practise.gesture.my_model import MyModel
from dl.practise.gesture.data_loader import DataLoader
from dl.practise.gesture.config import *
from dl.practise.gesture.my_optimizer import get_warm_lr
import os
from common.my_decorator import *


class ModelManager(object):
    def __init__(self):
        self.mode = MyModel().get_mode()
        self.data_loader = DataLoader(BATCH_SIZE, IMAGE_SIZE)
        pass

    def compile(self):
        self.mode.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            # loss=tf.keras.losses.CategoricalCrossentropy(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['acc']
        )

    def fit(self):
        # self.clear_logs()
        model_checkpoit = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_FILE_PATH,
            monitor=MONITOR_VALUE,
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
        )
        tensor_borad = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        )

        self.mode.fit_generator(
            self.data_loader.get_train_sequence(),
            validation_data=self.data_loader.get_test_sequence(),
            max_queue_size=10,
            epochs=EPOCHS,
            steps_per_epoch=int(len(self.data_loader.get_train_sequence())),
            verbose=1,
            # workers=int(multiprocessing.cpu_count() * 0.7),
            # use_multiprocessing=True,
            shuffle=True,
            callbacks=[model_checkpoit, tensor_borad,
                       get_warm_lr(len(self.data_loader.get_train_sequence()))]
        )

    def train(self):
        self.compile()
        print(self.mode.summary())
        self.fit()
        self.saved_model()
        pass

    def saved_model(self):
        timedes = int(time.time())
        path = ROOT_SMODEL_PAHT + timedes
        os.mkdir(path)
        tf.saved_model.save(self.mode, path)
        print('保存模型成功！')
        pass

    @caltime_p1('加载本地权重进行预测')
    def predict_by_load_weights(self):
        mode_path = ROOT_MODEL_FILE_PATH + 'weights.40-acc_1.00.hdf5'
        self.mode.load_weights(mode_path)
        self.predict(self.mode, self.data_loader.get_test_sequence())

    @caltime_p1('加载本地模型进行预测')
    def predict_by_load_saved_mode(self):
        # mode_path = './saved_model/'
        # mode = tf.saved_model.load(mode_path)
        # self.predict(mode, self.data_loader.get_test_sequence())
        pass

    @caltime_p3('预测数据耗时：')
    def predict(self, mode, test_sequence):
        print('batch_size: ', test_sequence.batch_size)
        for i in range(test_sequence.__len__()):
            test_x, test_y = test_sequence.__getitem__(i)
            y_pre = mode.predict(test_x)
            sca = tf.keras.metrics.SparseCategoricalAccuracy()
            sca.update_state(y_true=test_y, y_pred=y_pre)
            print(f"预测结果({i})： ", sca.result().numpy())


def test_predict1():
    manager = ModelManager()
    manager.predict_by_load_weights()
    pass


def test_mkdir():
    os.mkdir('./saved_model/testdir2')
    pass


def main():
    # test_predict1()
    test_mkdir()
    pass


"""
参数抽取到config
Tensorboard 调试
HParams超参数调优
某些方法抽取到工具类 sequence，余弦退火

mixup（）调试

"""

if __name__ == '__main__':
    main()
