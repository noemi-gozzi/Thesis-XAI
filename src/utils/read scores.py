import pickle
import numpy as np
import matplotlib.pyplot as plt
file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\scores_personalized_models.pkl '
with open(file_path, 'rb') as f:
    scores = pickle.load(f)
plt.figure()
for patient in scores:
    print("\t\t\t\t Patient {}".format(patient))
    accuracy=np.zeros((7,))
    for index,model in enumerate(scores[patient].keys()):
        accuracy[index]=scores[patient][model]['acc']
        # plt.plot(patient_cv, scores[patient][]["test_accuracy"])
        # print ("model: ", model)
        # print("\t\t\t\t accuracy: ", scores[patient][model]['acc'])
        # print("\t\t\t\t f1: ", scores[patient][model]['f1'])
    plt.figure()
    plt.bar(scores[patient].keys(), accuracy)
    plt.show()
#plt.legend(patient, fontsize=5)

