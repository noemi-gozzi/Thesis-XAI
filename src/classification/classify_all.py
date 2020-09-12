import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn import metrics
from utils.load_data import load_dataset
import pickle
import copy
import shap
import matplotlib.pyplot as plt




## combine datasets
from utils.customMultiprocessing import customMultiprocessing


def classify(file_path):
    '''

    :param file_path: path where the scores (accuracy, f1 and model) are saved

    :return:
    '''
    #load data and concatenate them to form one single dataset

    X_train, y_train, X_test, y_test = load_dataset()
    num_patient = len(X_train)
    print("number of patient: ", num_patient)
    X = []
    y = []
    for patient in range(num_patient):
        X_patient = pd.concat([X_train[patient], X_test[patient]], axis=0)
        X.append(X_patient)
        y_patient = pd.concat([y_train[patient], y_test[patient]], axis=0)
        y.append(y_patient)

    print(len(X), len(y))
    random_seed = 0;
    n_estimators = 50;
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']

    #models used to classify data
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
              'KNN': KNeighborsClassifier(n_neighbors=40),
              # 'Logistic Regression': LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0,
              #                                          random_state=random_seed),
              'Naive Bayes': GaussianNB(var_smoothing=0.1)

              }
    #open score file. if not present create the dictionary
    try:
        with open(file_path, 'rb') as f:
            scores = pickle.load(f)
    except FileNotFoundError:
        scores = {}

    #for each patient, create the dataset [10 patient in training and 1 in test] LOSO approach
    for patient in range(num_patient):
        if patient in scores:
            print (patient)
            continue
        print("patient:", patient)
        #LOSO
        x_train = X.copy()
        y_train = y.copy()
        x_test = x_train.pop(patient)
        y_test = y_train.pop(patient)
        x_train = pd.concat(x_train)
        y_train = pd.concat((y_train))


        #create parameter list for multiprocessing. data must be lists of the same length or single instances
        params_list=[list(models.values()), x_train, y_train, x_test, y_test, list(models.keys())]
        #apply multiprocessing to function model fit. results is a list
        results = customMultiprocessing(model_fit,params_list, pool_size=8)
        #convert result list in a dict
        score = {}
        for result in results:
            score.update(result)
        #dict "patient":, dict "model":, dict "score": acc, f1, model
        scores[patient] = score

        print("Save patient {}".format(patient))
        with open(file_path, 'wb') as f:
            pickle.dump(scores, f)
    return

def model_fit(model, x_train, y_train, x_test, y_test, model_name):
    '''

    :param model: classifier
    :param x_train: data used to train the classifier [num_instances*num_features]
    :param y_train: labels of x_train data [num_instances*1]
    :param x_test: data used to test the classifier
    :param y_test: labels of x_test data
    :param model_name: model name used as dict key when saving results
    :return:
    score: dict {'model_name': score}, where score is again a dict
    '''

    print(model_name)
    score={}
    model.fit(x_train, y_train)
    print(("fit"))
    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    score[model_name] = {'acc': acc, 'f1': f1, 'model': copy.copy(model)}

    print("model\t", model_name, "\n accuracy:", acc, "f1:", f1)
    return score
def shap_values (path_shap):
    X_train, y_train, X_test, y_test = load_dataset()
    num_patient = len(X_train)
    X = []
    y = []
    for patient in range(num_patient):
        X_patient = pd.concat([X_train[patient], X_test[patient]], axis=0)
        X.append(X_patient)
        y_patient = pd.concat([y_train[patient], y_test[patient]], axis=0)
        y.append(y_patient)

    random_seed = 0;
    n_estimators = 100;
    model=ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_seed)
    for patient in range(10,num_patient):
        x_train = X.copy()
        y_train = y.copy()
        x_test = x_train.pop(patient)
        y_test = y_train.pop(patient)
        x_train = pd.concat(x_train)
        y_train = pd.concat((y_train))

        model.fit(x_train, y_train)
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(x_test[:3000])

        with open(
                'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\shap_patient11_3000instances_extratrees.pkl',
                'wb') as f:
                pickle.dump(shap_values, f)


        for value in range(len(shap_values)):
            path_img_dot = path_shap + "patient{}_class{}_dot_general.png".format(patient, value)
            plt.figure()
            shap.summary_plot(shap_values[value], x_test[:3000], plot_type="dot", show=False)
            plt.savefig(path_img_dot)
            plt.close()
            plt.figure()
            path_img_bar = path_shap + "patient{}_class{}_bar_general.png".format(patient, value)
            shap.summary_plot(shap_values[value], x_test[:3000], plot_type="bar", show=False)
            plt.savefig(path_img_bar)
            plt.close()
    return

if __name__ == "__main__":
    # file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
    #             '\\results_classification\\scores.pkl '
    # classify(file_path)

    path_shap = "C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\images\\result_shap\\"
    shap_values(path_shap)
