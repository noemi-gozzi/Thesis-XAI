import pickle
import numpy as np
import matplotlib.pyplot as plt
file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\classification_featureset_wo_ssc_hpc.pkl '
with open(file_path, 'rb') as f:
    scores_wo = pickle.load(f)

file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\scores_personalized_models.pkl'
with open(file_path, 'rb') as f:
    scores = pickle.load(f)
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
'Random Forest',
'Extremely Randomized Trees',
'SVM_tuned',
'KNN']
accuracy = np.zeros((5,11))
accuracy_wo = np.zeros((5,11))
for index,model in enumerate(models):
    for patient in scores:
        accuracy[index,patient]=scores[patient][model]['acc']
        accuracy_wo[index,patient]=scores_wo[patient][model]['acc']
    plt.figure()
    plt.plot(accuracy[index])
    plt.plot(accuracy_wo[index])
    plt.xlabel('Patient', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0.75,1])
    plt.legend(["Complete set of features", "Feature set without SSC and HP_C"], fontsize=10, loc="lower right")
    plt.title(model)
    plt.show()
    plt.savefig("C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\results_classification\\{}.jpg".format(model))

