import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_featureset_ALL.pkl'
with open(file_path, 'rb') as f:
    scores = pickle.load(f)

file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_featureset_wo_ssc_hpc.pkl '
with open(file_path, 'rb') as f:
    scores_wo = pickle.load(f)


file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_SSC_ZC.pkl'
with open(file_path, 'rb') as f:
    scores_bad = pickle.load(f)

file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_featureset_no_corr.pkl'
with open(file_path, 'rb') as f:
    scores_50 = pickle.load(f)

file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_RMS_HP_C_M.pkl'
with open(file_path, 'rb') as f:
    scores_30 = pickle.load(f)

file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_RMS.pkl'
with open(file_path, 'rb') as f:
    scores_RMS= pickle.load(f)
file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_MAV.pkl'
with open(file_path, 'rb') as f:
    scores_MAV= pickle.load(f)
#
# plt.figure()
# for patient in scores:
#     print("\t\t\t\t Patient {}".format(patient))
#     accuracy=np.zeros((5,))
#     for index,model in enumerate(scores[patient].keys()):
#         accuracy[index]=scores[patient][model]['acc']
#         # plt.plot(patient_cv, scores[patient][]["test_accuracy"])
#         print ("model: ", model)
#         print("\t\t\t\t accuracy: ", scores[patient][model]['acc'])
#         print("\t\t\t\t f1: ", scores[patient][model]['f1'])
#     plt.figure()
#     plt.bar(scores[patient].keys(), accuracy)
#     plt.show()
#plt.legend(patient, fontsize=5)


models = ['LDA',
#'Random Forest',
'Extremely Randomized Trees',
'SVM_tuned']
#'KNN']
accuracy = np.zeros((3,11))
accuracy_wo = np.zeros((3,11))
accuracy_50 = np.zeros((3,11))
accuracy_30 = np.zeros((3,11))
accuracy_bad=np.zeros((3,11))
accuracy_RMS=np.zeros((3,11))
accuracy_MAV=np.zeros((3,11))
p_val={}
for index,model in enumerate(models):
    for patient in scores:
        accuracy[index,patient]=scores[patient][model]['acc']
        #accuracy_wo[index, patient] = scores_ordered[patient][model]['acc']
        accuracy_wo[index,patient]=scores_wo[patient][model]['acc']
        accuracy_50[index, patient] = scores_50[patient][model]['acc']
        accuracy_30[index, patient] = scores_30[patient][model]['acc']
        accuracy_bad[index, patient] = scores_bad[patient][model]['acc']
        accuracy_RMS[index, patient] = scores_RMS[patient][model]['acc']
        accuracy_MAV[index, patient] = scores_MAV[patient][model]['acc']

        # p_val[model]={"wo_ssc_hpc":stats.ttest_rel(accuracy[index], accuracy_wo[index]),
        #               "w_rms_wl_iemg":stats.ttest_rel(accuracy[index], accuracy_30[index]),
        #               "no corr": stats.ttest_rel(accuracy_30[index], accuracy_50[index]),
        #               "RMS": stats.ttest_rel(accuracy[index], accuracy_RMS[index])
        #
        #               }
        p_val[model]={"RMS_HP_C_HP_M":stats.ttest_rel(accuracy[index], accuracy_30[index]),
                      "no corr": stats.ttest_rel(accuracy[index], accuracy_50[index]),
                      #"RMS": stats.ttest_rel(accuracy[index], accuracy_RMS[index]),
                      #"MAV-RMS": stats.ttest_rel(accuracy_RMS[index], accuracy_MAV[index]),
                      "SHAP useless vs SHAP 50":stats.ttest_rel(accuracy_50[index], accuracy_bad[index]),
                      "50 no corr vs 30 no corr": stats.ttest_rel(accuracy_50[index], accuracy_30[index])

                      }


    plt.figure()
    plt.plot(accuracy[index])

    #plt.plot(accuracy_RMS[index])
    #plt.plot(accuracy_MAV[index])
    plt.plot(accuracy_50[index])
    plt.plot(accuracy_bad[index])
    plt.xlabel('Patient', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0.7,1])
    plt.legend(["Complete set of features", "50 not corr", "SHAP not important (ZC SSC)"], fontsize=10, loc="lower right")
    plt.title(model)
    plt.savefig("C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\results_classification\\ZC_SSC{}.jpg".format(model))
    #
    plt.show()
print(p_val)
