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


## combine datasets
from utils.customMultiprocessing import customMultiprocessing


def classify(file_path):
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

    try:
        with open(file_path, 'rb') as f:
            scores = pickle.load(f)
    except FileNotFoundError:
        scores = {}

    for patient in range(num_patient):
        if patient in scores:
            print (patient)
            continue
        print("patient:", patient)
        x_train = X.copy()
        y_train = y.copy()
        x_test = x_train.pop(patient)
        y_test = y_train.pop(patient)
        x_train = pd.concat(x_train)
        y_train = pd.concat((y_train))

        score = {}
        #for model_name in models:
        params_list=[list(models.values()), x_train, y_train, x_test, y_test, list(models.keys())]
        results = customMultiprocessing(model_fit,params_list, pool_size=8)
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
    score={}
    model.fit(x_train, y_train)
    print(("fit"))
    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    score[model_name] = {'acc': acc, 'f1': f1, 'model': copy.copy(model)}

    print("model\t", model_name, "\n accuracy:", acc, "f1:", f1)
    return score

if __name__ == "__main__":
    file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
                '\\results_classification\\scores.pkl '
    classify(file_path)
