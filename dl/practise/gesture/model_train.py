import tensorflow as tf
from dl.practise.gesture.my_model import MyModel
from dl.practise.gesture.data_loader import DataLoader
from dl.practise.gesture.config import *
from dl.practise.gesture.my_optimizer import WarmUpCosineDecayScheduler


class ModelManager(object):
    def __init__(self):
        self.mode = MyModel().get_mode()
        self.data_loader = DataLoader(BATCH_SIZE,IMAGE_SIZE)
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
            filepath='./hdf5/weights.{epoch:02d}-acc6_{val_acc:.2f}.hdf5',
            monitor='val_acc',
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
        )
        tensor_borad = tf.keras.callbacks.TensorBoard(
            log_dir='logs',
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
            callbacks=[model_checkpoit, tensor_borad, self.get_cos_lr()]
        )

    def get_cos_lr(self):
        # 余弦退回warmup
        # 得到总样本数
        batch_size = BATCH_SIZE
        sample_count = len(self.data_loader.get_train_sequence()) * batch_size

        # 第二阶段学习率以及总步数
        learning_rate_base = LEARNING_RATE
        total_steps = int(EPOCHS * sample_count) / batch_size
        # 计算第一阶段的步数需要多少 warmup_steps
        warmup_epoch = 5
        warmup_steps = int(warmup_epoch * sample_count) / batch_size

        warm_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                             total_steps=total_steps,
                                             warmup_learning_rate=0,
                                             warmup_steps=int(warmup_steps),
                                             hold_base_rate_steps=0, )
        return warm_lr

    def train(self):
        self.compile()
        print(self.mode.summary())
        self.fit()
        pass


def main():
    pass


if __name__ == '__main__':
    main()
