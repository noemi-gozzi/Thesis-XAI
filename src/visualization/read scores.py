import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_featureset_ALL.pkl'
with open(file_path, 'rb') as f:
    scores = pickle.load(f)
file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_150.pkl'
with open(file_path, 'rb') as f:
    scores_full = pickle.load(f)

# file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
#             '\\results_classification\\no_mav_scc_zc.pkl'
# with open(file_path, 'rb') as f:
#     scores_wo = pickle.load(f)
#
#
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
#
# file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
#             '\\results_classification\\classification_RMS.pkl'
# with open(file_path, 'rb') as f:
#     scores_RMS= pickle.load(f)
# file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
#             '\\results_classification\\classification_MAV.pkl'
# with open(file_path, 'rb') as f:
#     scores_MAV= pickle.load(f)
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
accuracy_full=np.zeros((3,11))
accuracy = np.zeros((3,11))
f1=np.zeros((3,11))
accuracy_wo = np.zeros((3,11))
f1_wo=np.zeros((3,11))

accuracy_bad=np.zeros((3,11))
f1_bad=np.zeros((3,11))

accuracy_50 = np.zeros((3,11))
f1_50=np.zeros((3,11))
accuracy_30 = np.zeros((3,11))
f1_30=np.zeros((3,11))
accuracy_RMS=np.zeros((3,11))
accuracy_MAV=np.zeros((3,11))
p_val={}
metric1="acc"
metric2="f1"

heigth=3;
width=2;
f, axes = plt.subplots(heigth, width,figsize=(30, 40))
f.subplots_adjust(hspace=0.4, wspace=0.3)


for index,model in enumerate(models):
    for patient in scores:
        accuracy_full[index,patient]=scores_full[patient][model][metric1]
        accuracy[index,patient]=scores[patient][model][metric1]
        #accuracy_wo[index, patient] = scores_ordered[patient][model]['acc']
        accuracy_50[index,patient]=scores_50[patient][model][metric1]
        f1[index,patient]=scores[patient][model][metric2]
        #accuracy_wo[index, patient] = scores_ordered[patient][model]['acc']
        f1_50[index,patient]=scores_50[patient][model][metric2]
        # accuracy_50[index, patient] = scores_50[patient][model][metric]
        accuracy_30[index, patient] = scores_30[patient][model][metric1]
        f1_30[index,patient]=scores_30[patient][model][metric2]
        accuracy_bad[index, patient] = scores_bad[patient][model][metric1]
        f1_bad[index, patient] = scores_bad[patient][model][metric2]
        # accuracy_RMS[index, patient] = scores_RMS[patient][model][metric]
        # accuracy_MAV[index, patient] = scores_MAV[patient][model][metric]

        # p_val[model]={"wo_ssc_hpc":stats.ttest_rel(accuracy[index], accuracy_wo[index]),
        #               "w_rms_wl_iemg":stats.ttest_rel(accuracy[index], accuracy_30[index]),
        #               "no corr": stats.ttest_rel(accuracy_30[index], accuracy_50[index]),
        #               "RMS": stats.ttest_rel(accuracy[index], accuracy_RMS[index])
        #
        #               }
        # p_val[model]={"RMS_HP_C_HP_M":stats.ttest_rel(accuracy[index], accuracy_30[index]),
        #               "no corr": stats.ttest_rel(accuracy[index], accuracy_50[index]),
        #               #"RMS": stats.ttest_rel(accuracy[index], accuracy_RMS[index]),
        #               #"MAV-RMS": stats.ttest_rel(accuracy_RMS[index], accuracy_MAV[index]),
        #               "SHAP 50 vs SHAP useless":stats.ttest_rel(accuracy_50[index], accuracy_bad[index]),
        #               "50 no corr vs 30 no corr": stats.ttest_rel(accuracy_50[index], accuracy_30[index])
        #
        #               }
    print("original vs not corr model{}, pval:{}".format(model, stats.ttest_rel(accuracy_full[index], accuracy_50[index])))
    print("original vs best model{}, pval:{} ".format(model, stats.ttest_rel(accuracy_full[index], accuracy_30[index])))
    print("original vs worst model{}, pval:{} ".format(model, stats.ttest_rel(accuracy_full[index], accuracy_bad[index])))
    print("original vs ITD model{}, pval:{} ".format(model, stats.ttest_rel(accuracy_full[index], accuracy[index])))
    # print("not corr vs best model{}, pval:{}".format(model, stats.ttest_rel(accuracy_50[index], accuracy_30[index])))
    # print("not corr vs worst model{}, pval:{}".format(model, stats.ttest_rel(accuracy_50[index], accuracy_bad[index])))
    # print("best vs worst model{}, pval:{}".format(model, stats.ttest_rel(accuracy_30[index], accuracy_bad[index])))


    axes[index,0].plot(accuracy[index], linewidth=3)
    axes[index,0].plot(accuracy_50[index], linewidth=3)
    axes[index, 0].plot(accuracy_30[index], linewidth=3)
    axes[index, 0].plot(accuracy_bad[index], linewidth=3)
    #axes[index,0].plot(accuracy_bad[index], linewidth=3)
    #plt.plot(accuracy_RMS[index])
    #plt.plot(accuracy_MAV[index])
    # plt.plot(accuracy_50[index])
    # plt.plot(accuracy_bad[index])
    # plt.plot(accuracy_30[index])
    axes[index,0].set_xlabel('Patient', fontsize=30)
    axes[index,0].set_ylabel('Accuracy score', fontsize=30)
    axes[index,0].set_ylim([0.6,1])
    axes[index,0].legend(["ITD features", "ITD without correlated features", "RMS, HPM, HPC", "ZC, SSC"],  fontsize=30, loc="lower right")
    axes[index,0].set_title(model, fontsize=30)

    axes[index,1].plot(f1[index], linewidth=3)
    axes[index,1].plot(f1_50[index], linewidth=3)
    axes[index, 1].plot(f1_30[index], linewidth=3)
    axes[index,1].plot(f1_bad[index], linewidth=3)
    #plt.plot(accuracy_RMS[index])
    #plt.plot(accuracy_MAV[index])
    # plt.plot(accuracy_50[index])
    # plt.plot(accuracy_bad[index])
    # plt.plot(accuracy_30[index])
    axes[index,1].set_xlabel('Patient', fontsize=30)
    axes[index,1].set_ylabel('F1-score', fontsize=30)
    axes[index,1].set_ylim([0.6,1])
    axes[index,1].legend(["ITD features", "ITD without correlated features", "RMS, HPM, HPC", "ZC, SSC"],  fontsize=30, loc="lower right")
    axes[index,1].set_title(model, fontsize=30)

#plt.savefig("C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\images_tesi\\no_corr.png")
    #
plt.show()
print(p_val)
