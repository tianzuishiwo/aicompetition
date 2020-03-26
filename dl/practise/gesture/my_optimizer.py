import numpy as np
import tensorflow as tf
from dl.practise.gesture.config import *


def get_warm_lr(sequence_len):
    # 余弦退回warmup
    # 得到总样本数
    batch_size = BATCH_SIZE
    # sample_count = len(self.data_loader.get_train_sequence()) * batch_size
    sample_count = sequence_len * batch_size

    # 第二阶段学习率以及总步数
    learning_rate_base = LEARNING_RATE
    total_steps = int(EPOCHS * sample_count) / batch_size
    # 计算第一阶段的步数需要多少 warmup_steps
    # warmup_epoch = 5
    warmup_steps = int(WARMUP_EPOCH * sample_count) / batch_size

    warm_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                         total_steps=total_steps,
                                         warmup_learning_rate=0,
                                         warmup_steps=int(warmup_steps),
                                         hold_base_rate_steps=0, )
    return warm_lr

class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """带有warnup的余弦退火学习率实现
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        # 是否在每次训练结束打印学习率
        self.verbose = verbose
        # 记录所有批次下来的每次准确的学习率，可以用于打印显示
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        # 记录当前训练到走到第几步数
        self.global_step = self.global_step + 1
        # 记录下所有每次的学习到列表，要统计画图可以使用

        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        # 计算这批次开始的学习率 lr
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)

        # 设置模型的学习率为lr
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\n批次数 %05d: 设置学习率为'
                  ' %s.' % (self.global_step + 1, lr))


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
    每批次带有warmup余弦退火学习率计算
    :param global_step: 当前到达的步数
    :param learning_rate_base: warmup之后的基础学习率
    :param total_steps: 总需要批次数
    :param warmup_learning_rate: warmup开始的学习率
    :param warmup_steps:warmup学习率 步数
    :param hold_base_rate_steps: 预留总步数和warmup步数间隔
    :return:
    """
    if total_steps < warmup_steps:
        raise ValueError("总步数要大于wamup步数")
    # 1、余弦退火学习率计算
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi * (global_step - warmup_steps - hold_base_rate_steps) / float(
            total_steps - warmup_steps - hold_base_rate_steps)
    ))

    # 2、warmup之后的学习率计算
    # 预留步数阶段
    # 如果预留大于0，判断目前步数是否 > warmup步数+预留步数，是的话返回刚才上面计算的学习率，不是的话使用warmup之后的基础学习率
    learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps, learning_rate, learning_rate_base)
    # 3、warmup学习率计算，并判断大小
    # 第一个阶段的学习率计算
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError("第二阶段学习率要大于第一阶段学习率")

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)
    # 4、如果最后当前到达的步数大于总步数，则归0，否则返回当前的计算出来的学习率（可能是warmup学习率也可能是余弦衰减结果）

    return np.where(global_step > total_steps, 0.0, learning_rate)

   # def get_warm_lr(self):
    #     # 余弦退回warmup
    #     # 得到总样本数
    #     batch_size = BATCH_SIZE
    #     sample_count = len(self.data_loader.get_train_sequence()) * batch_size
    #
    #     # 第二阶段学习率以及总步数
    #     learning_rate_base = LEARNING_RATE
    #     total_steps = int(EPOCHS * sample_count) / batch_size
    #     # 计算第一阶段的步数需要多少 warmup_steps
    #     # warmup_epoch = 5
    #     warmup_steps = int(WARMUP_EPOCH * sample_count) / batch_size
    #
    #     warm_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
    #                                          total_steps=total_steps,
    #                                          warmup_learning_rate=0,
    #                                          warmup_steps=int(warmup_steps),
    #                                          hold_base_rate_steps=0, )
    #     return warm_lr
