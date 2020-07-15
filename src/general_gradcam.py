import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical, plot_model
from utils.utils_classes import RReLU
from utils.load_data_CNN import load_data_CNN
from guided_grad_cam_our_model import build_model,load_image, deprocess_image, normalize,build_guided_model, guided_backprop, grad_cam, grad_cam_batch, compute_saliency
import pickle
import pandas as pd


# Xtrainset, Ytrainset, Xtestset, Ytestset=load_data_CNN()
# num_patient=len(Xtrainset)
# #for patient in range(1):

# #patient 0
#
# #saved_model.summary()
# Xtest=Xtestset[patient]
# ytest=Ytestset[patient]
# ytest=to_categorical(ytest-1,8)
#
# ##temp xtest for some tests.
# with open(
#         '/tmp/Xtest_tmp.pkl',
#         'wb') as f:
#     pickle.dump(Xtest, f)
# #


def compute_gradcam(patient, Xtest):

    saved_model = load_model(('../resources/best_models/best_modelinc{}.h5'.format(patient)), custom_objects={'RReLU': RReLU})
    ####preds array
    preds=saved_model.predict(Xtest)
    labels_preds=np.argmax((preds), axis=1)
    print(labels_preds)
    print((labels_preds.shape))

    ####backprop model
    model_tmp=build_model(saved_model)
    guided_model = build_guided_model(saved_model)
    guided_model.summary()

    ###divide dataset per label
    Xtest_list=[]
    for label in range(len(np.unique(labels_preds))):
        Xtest_list.append(Xtest[labels_preds==label])

    # for label in range (np.unique(labels_preds)):
    #     gradcam, gb, guided_gradcam = compute_saliency(saved_model, guided_model, './ciao.png', np.expand_dims(test_1[0,:,:,:],
    #                                                    layer_name='conv2d_140',
    #                                                    cls=-1, visualize=False, save=False)

    label=0

    gradcam_result={}
    for label in range(7,8):
    # for label in range(len(np.unique(labels_preds))):
        gradcam_value = np.zeros((50, 10, 512))
        X_tmp = Xtest_list[label]
        number_instances=X_tmp.shape[0]
        for index in range(50):
            gradcam, gb, guided_gradcam = compute_saliency(saved_model, guided_model, './ciao.png',
                                                       np.expand_dims(X_tmp[index, :, :, :], axis=0),
                                                            layer_name='conv2d_140',
                                                            cls=-1, visualize=False, save=False)
            gradcam_value[index, :, :]=gradcam
        mean_val = np.mean((gradcam_value), axis=0)
        plt.figure(figsize=(30, 10))
        plt.imshow(mean_val, cmap='jet', aspect='auto', alpha=0.5)
        # plt.imshow(x[0,:,:,0], cmap='Greys')
        plt.savefig(
            '../resources/images/gradcam_label{}_meanvalue_50.jpg'.format(
                label))

        plt.show()
        gradcam_result[label]={'gradcam_values': gradcam_value, 'mean': mean_val}
    return gradcam_result

if __name__ == "__main__":
    # tempraory dataset
    with open('/tmp/Xtest_tmp.pkl', 'rb') as f:
        Xtest = pickle.load(f)
    ##model

    compute_gradcam(patient=0, Xtest=Xtest)
