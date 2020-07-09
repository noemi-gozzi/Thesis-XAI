import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
#from keras.applications import resnet50
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#model=ResNet50((weights='imagenet'))
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import ops
from keras.layers import Layer
#import kapre
import keras
import keras.utils as ku

from keras.layers import Conv2D, LocallyConnected2D, Conv2DTranspose, Flatten, Dense, LeakyReLU, PReLU, Input, add, Layer
from keras.layers import BatchNormalization, UpSampling2D, Activation, Dropout, MaxPooling2D, AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras.models import Sequential, load_model, Model




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

saved_model = load_model('./best_modelinc0.h5',custom_objects={'RReLU': RReLU})


# Define model here ---------------------------------------------------
def build_model():
    """Function returning keras model instance.
    
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    return saved_model


#H, W = 299, 299 # Input shape, defined by the model (model.input_shape)
#H, W = 224, 224
H,W=10, 512

# ---------------------------------------------------------------------

def load_image(path, H, W, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model():
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)
    # tf.compat.v1.get_default_graph()
    #g = tf.compat.v1.get_default_graph()
    g=tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max
    return cam


def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


def compute_saliency(model, guided_model, img_path, preprocessed_input, layer_name='block5_conv3', cls=-1, visualize=True, save=True):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    #preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)
    # top_n = 5
    # top = decode_predictions(predictions, top=top_n)[0]
    # classes = np.argsort(predictions[0])[-top_n:][::-1]
    # print('Model prediction:')
    # for c, p in zip(classes, top):
    #     print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    if cls == -1:
        cls = np.argmax(predictions)
        print(cls)
    # class_name = decode_predictions(np.eye(1, 1000, cls))[0][0][1]
    # print("Explanation for '{}'".format(class_name))

    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path, H, W, preprocess=False)) / 2
        cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
        cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
        cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))

    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, H, W, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))

        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()

    return gradcam, gb, guided_gradcam


if __name__ == '__main__':
    model = build_model()
    guided_model = build_guided_model()
    print(guided_model.summary())

    #plt.imshow(load_image(image_path, H, W)[0])
    # pre_processed_input = load_image(image_path, H=224, W=224)
    # gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, image_path, pre_processed_input, layer_name='activation_49',
    #                                                 cls=-1, visualize=False, save=False)
# # img_path=sys.argv[1]
#     print(gradcam.shape)
