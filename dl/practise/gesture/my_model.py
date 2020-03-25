# import efficientnet
from dl.efficientnet import EfficientNetB0
import tensorflow as tf


class MyModel(object):
    def __init__(self):
        self.base_mode, self.mode = self._create_mode_efficientnet()
        pass

    def get_mode(self):
        return self.mode

    def _create_mode_efficientnet(self):
        base_mode = EfficientNetB0(include_top=False)
        output = base_mode.output
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(output)
        y_pre = tf.keras.layers.Dense(units=10, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_mode.input, outputs=y_pre)
        return base_mode, model
