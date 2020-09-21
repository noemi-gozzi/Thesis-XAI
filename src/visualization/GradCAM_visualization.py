import pickle
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from utils.utils_classes import RReLU

root_path = "C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI"


def visualize_gradcam(Xtest, patient):
    """
    Load gradcam values for *patient*, for each label and for each instance;
    weights each instance depending on the prediction value (pred*gradcam),
    average all instances grouped by label

    :param Xtest: dataset of correct instances
    :param patient: patient
    :return:
    """
    path = root_path + "/resources/Grad-CAM/gradcam_results_patient{}_conv_1D.pkl".format(patient)
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
    for cls in range(classes):
        gradcam_value = gradcam[cls]["gradcam_values"]
        ####preds array
        preds_tmp = saved_model.predict(Xtest_list[cls])[:, cls]
        # weighting each result depending on how much sure was the prediction

        for index in range(len(preds_tmp)):
            gradcam_value[index, :, :] = gradcam_value[index, :, :] * preds_tmp[index]
        mean_val = np.average(gradcam_value, axis=0)
        plt.figure(figsize=(30, 10))
        plt.imshow(mean_val, cmap='jet', aspect='auto', alpha=0.5)
        plt.savefig(root_path + "/resources/Grad-CAM/gradcam_patient{}_label{}.jpg".format(patient, cls))
        plt.show()
    return


if __name__ == "__main__":
    with open(root_path + '/resources/data/Xtest_correct.pkl', 'rb') as f:
        Xtest = pickle.load(f)
    visualize_gradcam(Xtest, patient=9)
