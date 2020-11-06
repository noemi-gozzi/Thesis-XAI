import pickle
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from utils.utils_classes import RReLU

root_path = "/home/tesi/Thesis-XAI"


def visualize_gradcam(Xtest, patient):
    """
    Load gradcam values for *patient*, for each label and for each instance;
    weights each instance depending on the prediction value (pred*gradcam),
    average all instances grouped by label

    :param Xtest: dataset of correct instances
    :param patient: patient
    :return:
    """
    path = root_path + "/resources/gradcam_results_patient{}_conv_1D_method2.pkl".format(patient)
    with open(path, 'rb') as f:
        gradcam = pickle.load(f)

    saved_model = load_model(root_path + '/resources/Conv1D/Conv1D_pat_{}.h5'.format(patient),
                             custom_objects={'RReLU': RReLU})
    classes = 8
    preds = saved_model.predict(Xtest[patient])
    labels_preds = np.argmax((preds), axis=1)
    print(labels_preds)
    print((labels_preds.shape))
    Xtest_list = []
    for label in np.unique(labels_preds):
        Xtest_list.append(Xtest[patient][labels_preds == label])
    #for cls in range(classes):
    for cls in range(8):
        gradcam_value = gradcam[cls]["gradcam_values"]
        ####preds array
        preds_tmp = saved_model.predict(Xtest_list[cls])[:, cls]
        # weighting each result depending on how much sure was the prediction

        # for index in range(len(preds_tmp)):
        #for index in range(15,16):
        index=30
        gradcam_value[index, :, :] = (gradcam_value[index, :, :] * preds_tmp[index])/np.max(gradcam_value[index, :, :])
        mean_val = np.average(gradcam_value, axis=0)
        # plt.figure(figsize=(30, 10))
        # plt.imshow(mean_val, cmap='jet', aspect='auto', alpha=0.5)
        # #plt.savefig(root_path + "/resources/GRAD-CAM method 2/6 channels 3conv/gradcam_patient{}_label{}_conv5.jpg".format(patient, cls))
        # plt.title("Class{}".format(cls), fontsize=20)
        # plt.show()
        plt.figure(figsize=(30, 10))
        plt.imshow(gradcam_value[index, :, :], cmap='jet', aspect='auto', alpha=0.5)
        plt.savefig(root_path + "/resources/images/gradcam_conv1D_patient{}_label{}_ex_{}.jpg".format(patient, cls,index))
        plt.title("Class{}".format(cls), fontsize=20)
        plt.show()
    return

if __name__ == "__main__":
    with open(root_path + '/resources/data_deep/Xtest_correct_ordered.pkl', 'rb') as f:
        Xtest = pickle.load(f)
    #Xtestset_drop=[]
    # for patient in range(11):
    #     X_tmp=np.zeros((len(Xtest[patient]), 6, 512,1))
    #     for index in range(len(Xtest[patient])):
    #         X_tmp[index]=np.delete(Xtest[patient][index], [6,7,8,9], axis=0)
    #     Xtestset_drop.append(X_tmp)

    visualize_gradcam(Xtest, patient=0)
