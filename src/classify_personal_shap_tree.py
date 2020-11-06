import pandas as pd  # version 0.24.2
import numpy as np  # version 1.16.4
import random
import matplotlib.pyplot as plt  # matplotlib version 3.1.0
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from utils.customMultiprocessing import customMultiprocessing
from utils.load_data import load_dataset
import pickle
import copy
#import shap.explainers._sampling as shap if this, use Sampling
import shap #if this use samplingexplainer
from sklearn.model_selection import GridSearchCV

def classify_personalized(file_path):
    X_train, y_train, X_test, y_test = load_dataset()
    num_patient = len(X_train)
    random_seed = 0;
    n_estimators = 50;
    models = {'LDA': LinearDiscriminantAnalysis(solver='svd'),
              # 'Decision Tree': DecisionTreeClassifier(max_depth=None,
              #                                        random_state=random_seed),
              'Bagging': BaggingClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=n_estimators,
                                           random_state=random_seed),
              'Random Forest': RandomForestClassifier(n_estimators=n_estimators,
                                                      random_state=random_seed),
              'Extremely Randomized Trees': ExtraTreesClassifier(n_estimators=n_estimators,
                                                                 random_state=random_seed),
              'Ada Boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=n_estimators,
                                              random_state=random_seed),
              'SVM_tuned': SVC(kernel='linear', C=1, class_weight="balanced", gamma='auto', probability=True),
              'KNN': KNeighborsClassifier(n_neighbors=40)
              # 'Logistic Regression': LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0,
              #                                          random_state=random_seed),
              }
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']
    try:
        with open(file_path, 'rb') as f:
            scores = pickle.load(f)
    except FileNotFoundError:
        scores = {}

    for patient in range(num_patient):
        if patient in scores:
            print(patient)
            continue
        print("patient:", patient)
        # x_train = X_train[patient].copy()
        # y_train = y_train[patient].copy()
        # x_test = X_test[patient].copy()
        # y_test = y_test[patient].copy()

        score = {}
        # for model_name in models:
        #params_list = [list(models.values()), x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), list(models.keys())]
        params_list = [list(models.values()), X_train[patient], y_train[patient], X_test[patient], y_test[patient],
                       list(models.keys())]
        results = customMultiprocessing(model_fit, params_list, pool_size=8)
        score = {}
        for result in results:
            score.update(result)
        scores[patient] = score

        print("Save patient {}".format(patient))
        with open(file_path, 'wb') as f:
            pickle.dump(scores, f)
    return


def model_fit(model, x_train, y_train, x_test, y_test, model_name):
    print(model_name)
    score = {}
    model.fit(x_train, y_train)
    print(("fit"))
    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    score[model_name] = {'acc': acc, 'f1': f1, 'model': copy.copy(model),
                         'confusion matrix': classification_report(y_test, y_pred)}

    print("model\t", model_name, "\n accuracy:", acc, "f1:", f1)
    print(classification_report(y_test, y_pred))

    return score

def shap_values (path_shap):
    with open("../X_train_sess_2.pkl", 'rb') as f:
        X_train = pickle.load(f)
    with open("../X_test_sess_2.pkl", 'rb') as f:
        X_test = pickle.load(f)
    with open("../y_train_sess_2.pkl", 'rb') as f:
        y_train = pickle.load(f)
    with open("../y_test_sess_2.pkl", 'rb') as f:
        y_test = pickle.load(f)

    num_patient=len(X_train)
    num_channels=10
    
    for patient in range(num_patient):
        for i in range (1,num_channels+1):
            feature_set_drop=["SE{}".format(i), "CCI_{}".format(i), "CCII_{}".format(i),"CCIII_{}".format(i), "CCIV_{}".format(i), "SKEW{}".format(i)]
            X_test[patient] = X_test[patient].drop(feature_set_drop, axis=1)
            X_train[patient] = X_train[patient].drop(feature_set_drop, axis=1)
            
    random_seed = 0;
    n_estimators = 50;

    models = {#'LDA': LinearDiscriminantAnalysis(solver='svd')
              #'SVM_tuned': SVC(kernel='linear', C=1, class_weight="balanced", gamma='auto', probability=True)
        #'KNN': KNeighborsClassifier(n_neighbors=40)
        'ExtremelyRandomizedTrees': ExtraTreesClassifier(n_estimators=n_estimators,
                                                                 random_state=random_seed)
    }
    for model_name in models.keys():
        print(model_name)
        #for each model train the model for each patient and compute shap values
        model = models[model_name]
        params_list = [list(range(0,num_patient)), model, X_train, y_train, X_test, y_test]
        #SHAP=customMultiprocessing(shap_multiprocessing, params_list, pool_size=11)
        SHAP=customMultiprocessing(shap_multiprocessing_tree, params_list, pool_size=11)
        # with open(
        #         '../resources/shap_SVM_all_patients.pkl',
        #         'wb') as f:
        #     pickle.dump(SHAP, f)
        # SHAP_TOT = {}
        # for shap in SHAP:
        #     SHAP_TOT.update(shap)
            # for value in range(len(shap_values)):
            #     path_img_dot = path_shap + "dot_patient{}_class{}.png".format(patient, value)
            #     plt.figure()
            #     shap.summary_plot(shap_values[value], X_test[patient], plot_type="dot", show=False)
            #     plt.savefig(path_img_dot)
            #     plt.close()
            #     plt.figure()
            #     path_img_bar = path_shap + "bar_patient{}_class{}.png".format(patient, value)
            #     shap.summary_plot(shap_values[value], X_test[patient], plot_type="bar", show=False)
            #     plt.savefig(path_img_bar)
            #     plt.close()

    return


def shap_multiprocessing_tree(patient, model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)
    # create explainer
    #shap_explainer = shap.KernelExplainer(model_list[patient].predict_proba, X_train.iloc[0:500, :])
    shap_explainer = shap.TreeExplainer(model, X_train)
    shap_values = shap_explainer.shap_values(X_test)


    with open(
            '../resources/results_ordered/shap_XRT_ordered_patient_{}_sess_2.pkl'.format(patient+6),
            'wb') as f:
        pickle.dump(shap_values, f)
    return 1


if __name__ == "__main__":

    #####CLASSIFY each patient separately
    #file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
    #            '\\results_classification\\scores_personalized_models.pkl '
    #classify_personalized(file_path)

    #####calculate shap values for each patient for some type of trees.

    path_shap = "C:\\Users\\"
    shap_values(path_shap)

    # X_train, y_train, X_test, y_test = load_dataset()
    # num_patient = len(X_train)
    # random_seed = 0;
    # n_estimators = 100;
    # model = ExtraTreesClassifier()
    # parameters = {'criterion': [ 'gini'],
    #               'n_estimators':[100],
    #               #'max_depth': [15, 30, 50, 100],
    #               #'min_samples_split': [0.003, 0.004, 0.005, 0.007],
    #               #'min_samples_leaf': [0.001, 0.002],
    #               'random_state': [0]}
    # scores={}
    # for patient in range(10,num_patient):
    #     scores[patient]=grid_search_algorithm(model, parameters, X_train[patient], y_train[patient], X_test[patient], y_test[patient])
    #     print(scores)
