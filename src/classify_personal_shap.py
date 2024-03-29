"""
File for classification with classical machine learning and SHAP values extraction
for svm and tree ensemble
"""

import pandas as pd  # version 0.24.2
import numpy as np  # version 1.16.4
import random
import matplotlib.pyplot as plt  # matplotlib version 3.1.0
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from utils.customMultiprocessing import customMultiprocessing
from utils.load_data import load_dataset
import pickle
import copy
import time
# import shap.explainers._sampling as shap if this, use Sampling
import shap  # if this use samplingexplainer
from sklearn.model_selection import GridSearchCV


def classify_personalized(file_path, X_train, y_train, X_test, y_test):

    #X_train, y_train, X_test, y_test = load_dataset()
    num_patient = len(X_train)
    random_seed = 0;
    n_estimators = 50;
    models = {'LDA': LinearDiscriminantAnalysis(solver='svd'),
              # 'Decision Tree': DecisionTreeClassifier(max_depth=None,
              #                                        random_state=random_seed),
              #'Bagging': BaggingClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=n_estimators,
              #                             random_state=random_seed),
              'Random Forest': RandomForestClassifier(n_estimators=n_estimators,
                                                      random_state=random_seed),
              'Extremely Randomized Trees': ExtraTreesClassifier(n_estimators=n_estimators,
                                                                 random_state=random_seed),
              #'Ada Boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=n_estimators,
              #                                random_state=random_seed),
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

        score = {}
        # for model_name in models:
        # params_list = [list(models.values()), x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), list(models.keys())]
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
    t_1=time.clock()
    model.fit(x_train, y_train)
    t_2=time.clock()
    y_pred = model.predict(x_test)
    t_3=time.clock()
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    score[model_name] = {'acc': acc, 'f1': f1, 'model': copy.copy(model),
                         'confusion matrix': classification_report(y_test, y_pred),
                         'training time': t_2-t_1,
                         'test time': t_3-t_2}

    print("model\t", model_name, "\n accuracy:", acc, "f1:", f1)
    print(classification_report(y_test, y_pred))

    return score


def shap_values(path_shap):
    X_train, y_train, X_test, y_test = load_dataset()
    num_patient = len(X_train)
    random_seed = 0;
    n_estimators = 100;
    model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_seed)
    shap_list = []
    for patient in range(num_patient):
        model.fit(X_train[patient], y_train[patient])
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(X_test[patient])
        # path_img_bar = path_shap + "patient{}_allclasses.png".format(patient)
        # plt.figure()
        # shap.summary_plot(shap_values, X_test[patient], plot_type="bar", show=False)
        # plt.savefig(path_img_bar)
        for i in range(len(shap_values)):
            shap_df = pd.DataFrame(data=shap_values[i], columns=X_test[i].columns.values)
            shap_list.append(shap_df)
        with open(
                '../resources/shap_extratrees_patient_{}_tmp.pkl'.format(patient),
                'wb') as f:
            pickle.dump(shap_list, f)

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


def shap_values_svm(path_shap):
    X_train, y_train, X_test, y_test = load_dataset()
    num_patient = len(X_train)
    model = SVC(kernel='linear', C=1, class_weight="balanced", gamma='auto', probability=True)
    for patient in range(num_patient):
        model.fit(X_train[patient], y_train[patient])
        explainer = shap.SamplingExplainer(model.predict_proba, X_train[patient].iloc[0:100, :])
        shap_values = explainer.shap_values(X_test[patient])

        with open(
                '../resources/shap_SVM/shap_svm_patient_{}_SVM.pkl'.format(patient),
                'wb') as f:
            pickle.dump(shap_values, f)
        path_img_bar = path_shap + "patient{}_svm.png".format(patient)
        plt.figure()
        shap.summary_plot(shap_values, X_test[patient], plot_type="bar", show=False)
        plt.savefig(path_img_bar)

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


def grid_search_algorithm(classifier, parameters, X_train, y_train, X_test, y_test):
    gs = GridSearchCV(classifier, parameters, cv=3, scoring='accuracy', n_jobs=-1, refit=True)
    gs = gs.fit(X_train, y_train)
    # summarize the results of your GRIDSEARCH
    print('***GRIDSEARCH RESULTS***')

    print("Best score: %f using %s" % (gs.best_score_, gs.best_params_))
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    params = gs.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # TEST ON YOUR TEST SET
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    return {'acc': acc, 'f1': f1, 'best model': best_model}


if __name__ == "__main__":
    #####CLASSIFY each patient separately
    file_path = '../resources/results_classification/RMS_HPC_HPM_ch6.pkl'
    with open("../data_shap/X_train_ordered_150.pkl", 'rb') as f:
        X_train = pickle.load(f)
    with open("../data_shap/X_test_ordered_150.pkl", 'rb') as f:
        X_test = pickle.load(f)
    with open("../data_shap/y_train_ordered_150.pkl", 'rb') as f:
        y_train = pickle.load(f)
    with open("../data_shap/y_test_ordered_150.pkl", 'rb') as f:
        y_test = pickle.load(f)
    # with open("../X_train_sess_2.pkl", 'rb') as f:
    #     X_train = pickle.load(f)
    # with open("../X_test_sess_2.pkl", 'rb') as f:
    #     X_test = pickle.load(f)
    # with open("../y_train_sess_2.pkl", 'rb') as f:
    #     y_train = pickle.load(f)
    # with open("../y_test_sess_2.pkl", 'rb') as f:
    #     y_test = pickle.load(f)

    num_patient=len(X_train)
    #drop_set=["HP_C{}".format(i), "HP_M{}".format(i),"WL{}".format(i), "IEMG{}".format(i), "HP_A{}".format(i), "MAV{}".format(i), "RMS{}".format(i)]

    for patient in range(num_patient):
        for i in range(1,11):
            drop_set = ["SSC{}".format(i), "ZC{}".format(i), "MAV{}".format(i), "HP_A{}".format(i), "IEMG{}".format(i), "WL{}".format(i)]
            X_test[patient] = X_test[patient].drop(drop_set, axis=1)
            X_train[patient] = X_train[patient].drop(drop_set, axis=1)


        for i in range (1,11):
            X_test[patient] = X_test[patient].drop(["SE{}".format(i), "CCI_{}".format(i), "CCII_{}".format(i),"CCIII_{}".format(i), "CCIV_{}".format(i), "SKEW{}".format(i)], axis=1)
            X_train[patient] = X_train[patient].drop(["SE{}".format(i), "CCI_{}".format(i), "CCII_{}".format(i),"CCIII_{}".format(i), "CCIV_{}".format(i), "SKEW{}".format(i)], axis=1)
        for i in range(6,11):
            drop_set = ["RMS{}".format(i), "HP_C{}".format(i), "HP_M{}".format(i)]
            X_test [patient]= X_test[patient].drop(drop_set, axis=1)
            X_train [patient]=X_train[patient].drop(drop_set, axis=1)
    classify_personalized(file_path,  X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    #####calculate shap values for each patient for some type of trees.

    # path_shap = "../resources/shap_SVM/"
    # shap_values(path_shap)

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

# 7.925x10-2 8.927 x10-2 6.776 x10-2 9.979 x10-2 8.788 x10-2 1.833 x10-1
# 2.558 x10-1 147.9 125.3 9.5456e-02 3.1459e-02 1.3