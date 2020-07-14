import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
from IPython.display import display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import kapre
import keras
import keras.utils as ku

from keras.layers import Conv2D, LocallyConnected2D, Conv2DTranspose, Flatten, Dense, LeakyReLU, PReLU, Input, add, Layer
from keras.layers import BatchNormalization, UpSampling2D, Activation, Dropout, MaxPooling2D, AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical, plot_model
from keras.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop, SGD
from keras.initializers import RandomNormal
import keras.backend as K

from utils.utils_classes import RReLU
from utils.load_data_CNN import load_data_CNN

Xtrainset, Ytrainset, Xtestset, Ytestset=load_data_CNN()
num_patient=len(Xtrainset)
for patient in range(1):
    #patient 0
    saved_model = load_model(('C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\best_models\\best_modelinc{}.h5'.format(patient)), custom_objects={'RReLU': RReLU})
    #saved_model.summary()
    Xtest=Xtestset[patient]
    ytest=Ytestset[patient]
    ytest=to_categorical(ytest-1,8)
    # evaluate the model
    scores = saved_model.evaluate(Xtest, ytest, verbose=0)
    print("Result on test set for taregt domain: %s: %.2f%%" % (saved_model.metrics_names[1], scores[1] * 100))
    preds=saved_model.predict(Xtest)
    #print(to_categorical(preds-1,8))
    #print(y_test[:10])

#import guided_grad_cam_our_model
Xtest_expand=np.expand_dims(Xtest[0], axis=0)
from guided_grad_cam_our_model import build_model,load_image, deprocess_image, normalize,build_guided_model, guided_backprop, grad_cam, grad_cam_batch, compute_saliency
model_tmp=build_model(saved_model)
guided_model = build_guided_model(saved_model)
guided_model.summary()
#plt.imshow(load_image(image_path, H, W)[0])
gradcam, gb, guided_gradcam = compute_saliency(saved_model, guided_model, './ciao.png', Xtest_expand, layer_name='conv2d_140',
                                                cls=-1, visualize=False, save=False)
print(to_categorical((np.argmax((preds), axis=1)),8)[0])
plt.figure(figsize=(30,10))
plt.imshow(Xtest_expand[0,:,:,0], interpolation='nearest', cmap='Greys')
plt.imshow(gradcam, cmap='jet', aspect='auto', alpha=0.5)
#plt.imshow(x[0,:,:,0], cmap='Greys')
plt.savefig('C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\images\\ex.jpg')
