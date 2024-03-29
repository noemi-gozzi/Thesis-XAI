"""
Temporary file to test different gradcam function. it works on one image
the model is uploaded and so Xtest

"""
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from utils.utils_classes import RReLU
from guided_grad_cam_our_model import build_model,load_image, deprocess_image, normalize,build_guided_model, guided_backprop, grad_cam, grad_cam_batch, compute_saliency
from utils.load_data_CNN import load_data_CNN
from original_grad_cam import grad_cam2

# Xtrainset, Ytrainset, Xtestset, Ytestset=load_data_CNN()
# num_patient=len(Xtrainset)
with open('../resources/data_deep/data_not_ordered/Xtest_correct.pkl', 'rb') as f:
    Xtestset = pickle.load(f)
##model


for patient in range(1):
    #patient 0
    saved_model = load_model(('../resources/Conv1D/Conv1D_pat_{}.h5'.format(patient)), custom_objects={'RReLU': RReLU})
    #saved_model.summary()
    Xtest=Xtestset[patient]
    # ytest=Ytestset[patient]
    # ytest=to_categorical(ytest-1,8)
    # # evaluate the model
    # scores = saved_model.evaluate(Xtest, ytest, verbose=0)
    # print("Result on test set for taregt domain: %s: %.2f%%" % (saved_model.metrics_names[1], scores[1] * 100))
    # preds=saved_model.predict(Xtest)
    #print(to_categorical(preds-1,8))
    #print(y_test[:10])

#import guided_grad_cam_our_model
preds=saved_model.predict(Xtest)
Xtest_expand=np.expand_dims(Xtest[0], axis=0)
print(np.argmax(preds[0]))

predicted_class=np.argmax(saved_model.predict(np.expand_dims(Xtest[0], axis=0)))
model_tmp=build_model(saved_model)
guided_model = build_guided_model(saved_model)
guided_model.summary()
#plt.imshow(load_image(image_path, H, W)[0])
gradcam, gb, guided_gradcam = compute_saliency(saved_model, guided_model,
                                                           Xtest_expand, H=10, W=512,
                                                           layer_name='conv5',
                                                           cls=-1, visualize=False, save=False)
#predictions = saved_model.predict(preprocessed_input)
# top_1 = decode_predictions(predictions)[0][0]
# print('Predicted class:')
# print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
#
# predicted_class = np.argmax(predictions)
# print(to_categorical((np.argmax((preds), axis=1)),8)[0])
heatmap = grad_cam2(saved_model, Xtest_expand, predicted_class, "conv5")
cv2.imwrite("gradcam3.jpg", cam)
plt.figure(figsize=(30,10))
plt.imshow(Xtest_expand[0,:,:,0], interpolation='nearest', cmap='Greys')
plt.imshow(cam, cmap='jet', aspect='auto', alpha=0.5)
plt.show()

plt.figure(figsize=(30,10))
plt.imshow(Xtest_expand[0,:,:,0], interpolation='nearest', cmap='Greys')
plt.imshow(heatmap, cmap='jet', aspect='auto', alpha=0.5)
plt.title("heatmap")
plt.show()

plt.figure(figsize=(30,10))
plt.imshow(Xtest_expand[0,:,:,0], interpolation='nearest', cmap='Greys')
plt.imshow(gradcam, cmap='jet', aspect='auto', alpha=0.5)
plt.title("gradcam")
plt.show()
#plt.imshow(x[0,:,:,0], cmap='Greys')
#plt.savefig('C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\images\\ex.jpg')
