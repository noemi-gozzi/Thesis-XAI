from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    H, W = 10, 512
    img_path = path
    img = image.load_img(img_path, target_size=(H,W))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def grad_cam2(input_model, image, category_index, layer_name):
    nb_classes = 8
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    #model.summary()
    loss = K.sum(model.output)
    #conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    conv_output=input_model.get_layer(layer_name).output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    # output, grads_val = gradient_function([image])
    # output, grads_val = output[0, :], grads_val[0, :, :, :]
    #
    # weights = np.mean(grads_val, axis = (0, 1))
    # cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    # print(cam.shape)
    #
    # for i, w in enumerate(weights):
    #     cam += w * output[:, :, i]
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (512, 6), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0) #RELU
    #heatmap = cam / np.max(cam)


    # #Return to BGR [0..255] from the preprocessed image
    # image = image[0, :]
    # image -= np.min(image)
    # image = np.minimum(image, 255)

    # cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # cam = np.float32(cam) + np.float32(image)
    # cam = 255 * cam / np.max(cam)
    return cam

def compute_gradcam_2(model, preprocessed_input, layer_name='block5_conv3', cls=-1):
    """Compute saliency using all three approaches.
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
        #print(cls)
    # class_name = decode_predictions(np.eye(1, 1000, cls))[0][0][1]
    # print("Explanation for '{}'".format(class_name))

    cam= grad_cam2(model, preprocessed_input, cls, layer_name)
    return cam

# preprocessed_input = load_image("../resources/images/cat_dog.png")
#
# model = VGG16(weights='imagenet')
#
# predictions = model.predict(preprocessed_input)
# top_1 = decode_predictions(predictions)[0][0]
# print('Predicted class:')
# print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
#
# predicted_class = np.argmax(predictions)
# cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
# cv2.imwrite("gradcam.jpg", cam)
#
# register_gradient()
# guided_model = modify_backprop(model, 'GuidedBackProp')
# saliency_fn = compile_saliency_function(guided_model)
# saliency = saliency_fn([preprocessed_input, 0])
# gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))