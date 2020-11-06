import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical, plot_model
from utils.utils_classes import RReLU
from utils.load_data_CNN import load_data_CNN
from guided_grad_cam_our_model import build_model, load_image, deprocess_image, normalize, build_guided_model, guided_backprop, grad_cam, grad_cam_batch, compute_saliency
import pickle
import pandas as pd
import cv2
from utils.customMultiprocessing import customMultiprocessing
from original_grad_cam import compute_gradcam_2

# Xtrainset, Ytrainset, Xtestset, Ytestset=load_data_CNN()
# num_patient=len(Xtrainset)

def compute_gradcam(patient, Xtest):
    """

    :param patient: patient for which gradcam is computed (different model and test data)
    :param Xtest: Xtest for patient (dimension Nx10x5120x1)
    :return: gradcam result (list of 8 dict with gradcam values for each instance)
    """
    saved_model = load_model('../resources/Conv1D/Conv1D_pat_{}.h5'.format(patient), custom_objects={'RReLU': RReLU})

    #### Divide the Xtest dataset depending on the predicted label. list of 8 datasets
    ####backprop model
    model_tmp=build_model(saved_model)
    guided_model = build_guided_model(saved_model)
    guided_model.summary()

    ##preds array
    preds = saved_model.predict(Xtest)
    labels_preds = np.argmax((preds), axis=1)
    print(labels_preds)
    print((labels_preds.shape))
    ##divide dataset per label
    Xtest_list = []
    for label in np.unique(labels_preds):
        Xtest_list.append(Xtest[labels_preds == label])
    ####

    label = 0
    try:
        with open('../resources/gradcam_results_patient{}_conv_1D_bp.pkl'.format(patient), 'rb') as f:
            gradcam_result = pickle.load(f)
    except FileNotFoundError:
        print("create file")
        gradcam_result = {}

    for label in np.unique(labels_preds):
        #if label already computed go to the next label
        if label in gradcam_result:
            print(label)
            continue
        X_tmp = Xtest_list[label]
        gradcam_value = np.zeros((X_tmp.shape[0], X_tmp.shape[1], 512))
        gb_value = np.zeros((X_tmp.shape[0], X_tmp.shape[1], 512))
        guided_gradcam_value = np.zeros((X_tmp.shape[0], X_tmp.shape[1], 512))
        print(X_tmp.shape[0])
        for index in range(X_tmp.shape[0]):
            #print(index)
            if (index==X_tmp.shape[0]-1): print("Patient{} class {} ended".format(patient, label))
            gradcam, gb, guided_gradcam = compute_saliency(saved_model, guided_model,
                                                           np.expand_dims(X_tmp[index, :, :, :], axis=0), H=X_tmp.shape[1], W=512,
                                                           layer_name='conv5',
                                                           cls=-1, visualize=False, save=False)

            # gradcam = compute_gradcam_2(saved_model,
            #                            np.expand_dims(X_tmp[index, :, :, :], axis=0),
            #                            layer_name='conv3',
            #                            cls=-1)

            gradcam_value[index, :, :] = gradcam
            gb_value[index, :, :] = gb[0,:,:,0]
            guided_gradcam_value[index, :, :] = guided_gradcam[0,:,:,0]


        gradcam_result[label] = {'gradcam_values': gradcam_value, 'gb_values': gb_value, 'guided_gradcam_values': guided_gradcam_value}

        with open(
                '../resources/gradcam_results_patient{}_conv_1D_bp.pkl'.format(patient),
                'wb') as f:
            pickle.dump(gradcam_result, f)
    return 1

if __name__ == "__main__":
    # tempraory datasetdd
    with open('../resources/data_deep/Xtest_correct_ordered.pkl', 'rb') as f:
        Xtest = pickle.load(f)
    Xtestset_drop=[]
#    for patient in range(11):
#        X_tmp=np.zeros((len(Xtest[patient]), 6, 512,1))
#        for index in range(len(Xtest[patient])):
#            X_tmp[index]=np.delete(Xtest[patient][index], [6,7,8,9], axis=0)
#        Xtestset_drop.append(X_tmp)
    ##model
    patient=list(range(11))
    params_list = [list(range(0,11)), Xtest]
    out=customMultiprocessing(compute_gradcam, params_list, pool_size=11)
    #compute_gradcam(patient=patient, Xtest=Xtest[patient])
