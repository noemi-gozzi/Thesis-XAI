
from keras.layers import Conv2D, LocallyConnected2D, Conv2DTranspose, Flatten, Dense, LeakyReLU, PReLU, Input, add, Layer
import keras.backend as K
import numpy as np

class RReLU(Layer):
    '''Randomized Leaky Rectified Linear Unit
    that uses a random alpha in training while using the average of alphas
    in testing phase:
    During training
    `f(x) = alpha * x for x < 0, where alpha ~ U(l, u), l < u`,
    `f(x) = x for x >= 0`.
    During testing:
    `f(x) = (l + u) / 2 * x for x < 0`,
    `f(x) = x for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        l: lower bound of the uniform distribution, default is 1/8
        u: upper bound of the uniform distribution, default is 1/3
    # References
        - [Empirical Evaluation of Rectified Activations in Convolution Network](https://arxiv.org/pdf/1505.00853v2.pdf)
    '''

    def __init__(self, l=1 / 10., u=1 / 9., **kwargs):
        self.supports_masking = True
        self.l = l
        self.u = u
        self.average = (l + u) / 2
        self.uses_learning_phase = True
        super(RReLU, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.in_train_phase(K.relu(x, np.random.uniform(self.l, self.u)),
                                K.relu(x, self.average))

    def build(self, input_shape):
        super(RReLU, self).build(input_shape)

    def get_config(self):
        config = {'l': self.l, 'u': self.u}
        base_config = super(RReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def compute_output_shape(self, input_shape):
    #     return input_shape
