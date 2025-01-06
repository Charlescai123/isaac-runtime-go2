import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Layer


class MLPModel(Model):
    def __init__(self, shape_input, shape_output, name='', output_activation=None):
        super(MLPModel, self).__init__()

        self.model = self.build_mlp_model(shape_input, shape_output, name, output_activation)

    # def __new__(cls, shape_input, shape_output, name='', output_activation=None):
    #     return cls.build_mlp_model(shape_input, shape_output, name, output_activation)

    @staticmethod
    def build_mlp_model(shape_input, shape_output, name='', output_activation=None):
        input = Input(shape=(shape_input,), name=name + '_input', dtype=tf.float16)
        dense1 = Dense(512, activation='elu', name=name + '_dense1')(input)
        dense2 = Dense(256, activation='elu', name=name + '_dense2')(dense1)
        dense3 = Dense(128, activation='elu', name=name + '_dense3')(dense2)
        output = Dense(shape_output, activation=output_activation, name=name + '_output')(dense3)
        model = Model(inputs=input, outputs=output, name=name)
        return model

    def call(self, inputs, **kwargs):
        return self.model(inputs)


if __name__ == '__main__':
    actor = MLPModel(shape_input=48, shape_output=12, name="actor", output_activation=None).model
    actor.summary()

