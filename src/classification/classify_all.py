import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn import metrics
import sys
sys.path.append('..')
#from utils.load_data import load_dataset
import pickle
import copy
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler




## combine datasets
from utils.customMultiprocessing import customMultiprocessing

def load_dataset_all():
    # this version of database had 15 features for each channel (in total 10 channel) making 150 features
    # for each subject we have 3 seperate databases for 3 rounds
    print("LOAD DATASETS\n\n:")
    N_Subj = 11  # 11 subjects
    df_r1 = [0] * N_Subj  # dataframe for round1 for all subjects
    df_r2 = [0] * N_Subj  # dataframe for round2 for all subjects
    df_r3 = [0] * N_Subj  # dataframe for round3 for all subjects

    # making a column name for dataframe
    # the database is organized in the way that first feature of all 10 channels are at the beginning, then the second feature for all channels and so on
    cl_names = []
    for i in range(1, 11):
        cl_names.append('MAV' + str(i))
    for i in range(1, 11):
        cl_names.append('ZC' + str(i))
    for i in range(1, 11):
        cl_names.append('SSC' + str(i))
    for i in range(1, 11):
        cl_names.append('WL' + str(i))
    for i in range(1, 11):
        cl_names.append('HP_A' + str(i))
    for i in range(1, 11):
        cl_names.append('HP_M' + str(i))
    for i in range(1, 11):
        cl_names.append('HP_C' + str(i))
    for i in range(1, 11):
        cl_names.append('SE' + str(i))
    for i in range(1, 11):
        cl_names.append('CCI_' + str(i))
    for i in range(1, 11):
        cl_names.append('CCII_' + str(i))
    for i in range(1, 11):
        cl_names.append('CCIII_' + str(i))
    for i in range(1, 11):
        cl_names.append('CCIV_' + str(i))
    for i in range(1, 11):
        cl_names.append('RMS' + str(i))
    for i in range(1, 11):
        cl_names.append('IEMG' + str(i))
    for i in range(1, 11):
        cl_names.append('SKEW' + str(i))

    cl_names.append('class')  # column dedicated to show which class of movement this sample belongs to
    cl_names.append(
        'move')  # column dedicated to show which number of movement (out of 40 in each round) this sample belongs to

    # loading sepearte databases for each round and subject
    for i in range(0, N_Subj):
        df_r1[i] = pd.read_csv(
            "/home/tesi/Thesis-XAI/resources/data_features/Version 3(V2 without shuffle)/database_v3_sbj{}_s1_WL512_S128_r1.csv".format(
                i + 1))
        df_r1[i].columns = cl_names

        df_r2[i] = pd.read_csv(
            "/home/tesi/Thesis-XAI/resources/data_features/Version 3(V2 without shuffle)/database_v3_sbj{}_s1_WL512_S128_r2.csv".format(
                i + 1))
        df_r2[i].columns = cl_names

        df_r3[i] = pd.read_csv(
            "/home/tesi/Thesis-XAI/resources/data_features/Version 3(V2 without shuffle)/database_v3_sbj{}_s1_WL512_S128_r3.csv".format(
                i + 1))
        df_r3[i].columns = cl_names

    print("Subject1")
    print("Round1", np.shape(df_r1[0]))
    print("Round2", np.shape(df_r2[0]))
    print("round3", np.shape(df_r3[0]))
    print(df_r1[0].head())
    # in each round for each movement we have almost 5 repetitions,
    # we keep 2/3 of repetitions for train and rest for test
    print("TRAIN AND TEST\n\n:")
    train_df_r1 = [pd.DataFrame(columns=cl_names)] * N_Subj
    test_df_r1 = [pd.DataFrame(columns=cl_names)] * N_Subj
    train_df_r2 = [pd.DataFrame(columns=cl_names)] * N_Subj
    test_df_r2 = [pd.DataFrame(columns=cl_names)] * N_Subj
    train_df_r3 = [pd.DataFrame(columns=cl_names)] * N_Subj
    test_df_r3 = [pd.DataFrame(columns=cl_names)] * N_Subj

    # 3 seperate for for the rounds:
    # round1:
    for s in range(0, N_Subj):  # for each subject we do:
        df = df_r1[s]
        for i in range(1, 9):
            class_group = df.loc[df['class'] == i]  # first we keep all the samples belonging to movement class number i
            class_group.index = range(class_group.shape[0])  # how many samples are in class i (range)
            target_dist = class_group.groupby(
                ['move']).size()  # how many times the movement i was repeated (it can be 5,4,3)
            # this depends on if the subject performed a movemnt we eliminated it from database so some of movements instead of having 5 repetitions have less!

            mv_indx = target_dist.index
            train_size = int((2 / 3) * len(
                mv_indx))  # 2/3 of number of repetitions is for training. e.g. if 5 times was repeated we keep 3 for training rest for test
            train_mv = mv_indx[0:train_size]
            for j in train_mv:
                train_df_r1[s] = train_df_r1[s].append(class_group.loc[class_group['move'] == j],
                                                       ignore_index=True)  # adding to trainset 2/3 of repetitions of one movement
            test_mv = mv_indx[train_size:]
            for k in test_mv:
                test_df_r1[s] = test_df_r1[s].append(class_group.loc[class_group['move'] == k],
                                                     ignore_index=True)  # adding to test set rest of the movement repetitions

        train_df_r1[s] = train_df_r1[s].astype(float)
        train_df_r1[s]['class'] = train_df_r1[s]['class'].astype(int)
        test_df_r1[s] = test_df_r1[s].astype(float)
        test_df_r1[s]['class'] = test_df_r1[s]['class'].astype(int)

        # train and test set no longer need the 'move' column
        train_df_r1[s].drop('move', inplace=True, axis=1)
        test_df_r1[s].drop('move', inplace=True, axis=1)

    print("Subject1:")
    print("round 1 whole dataset:", df_r1[0].shape)
    print("round 1 test dataset:", test_df_r1[0].shape)
    print("round 1 train dataset:", train_df_r1[0].shape)

    # round2: TRAIN AND TEST
    for s in range(0, N_Subj):
        df = df_r2[s]
        for i in range(1, 9):
            class_group = df.loc[df['class'] == i]
            class_group.index = range(class_group.shape[0])
            target_dist = class_group.groupby(['move']).size()
            mv_indx = target_dist.index
            train_size = int((2 / 3) * len(mv_indx))
            train_mv = mv_indx[0:train_size]
            for j in train_mv:
                train_df_r2[s] = train_df_r2[s].append(class_group.loc[class_group['move'] == j], ignore_index=True)
            test_mv = mv_indx[train_size:]
            for k in test_mv:
                test_df_r2[s] = test_df_r2[s].append(class_group.loc[class_group['move'] == k], ignore_index=True)

        train_df_r2[s] = train_df_r2[s].astype(float)
        train_df_r2[s]['class'] = train_df_r2[s]['class'].astype(int)
        test_df_r2[s] = test_df_r2[s].astype(float)
        test_df_r2[s]['class'] = test_df_r2[s]['class'].astype(int)
        train_df_r2[s].drop('move', inplace=True, axis=1)
        test_df_r2[s].drop('move', inplace=True, axis=1)

    print("round 2 whole dataset:", df_r2[0].shape)
    print("round 2 test dataset:", test_df_r2[0].shape)
    print("round 2 train dataset:", train_df_r2[0].shape)

    # round3:
    for s in range(0, N_Subj):
        df = df_r3[s]
        for i in range(1, 9):
            class_group = df.loc[df['class'] == i]
            class_group.index = range(class_group.shape[0])
            target_dist = class_group.groupby(['move']).size()
            mv_indx = target_dist.index
            train_size = int((2 / 3) * len(mv_indx))
            train_mv = mv_indx[0:train_size]
            for j in train_mv:
                train_df_r3[s] = train_df_r3[s].append(class_group.loc[class_group['move'] == j], ignore_index=True)
            test_mv = mv_indx[train_size:]
            for k in test_mv:
                test_df_r3[s] = test_df_r3[s].append(class_group.loc[class_group['move'] == k], ignore_index=True)

        train_df_r3[s] = train_df_r3[s].astype(float)
        train_df_r3[s]['class'] = train_df_r3[s]['class'].astype(int)
        test_df_r3[s] = test_df_r3[s].astype(float)
        test_df_r3[s]['class'] = test_df_r3[s]['class'].astype(int)
        train_df_r3[s].drop('move', inplace=True, axis=1)
        test_df_r3[s].drop('move', inplace=True, axis=1)

    print("round 3 whole dataset:", df_r3[0].shape)
    print("round 3 test dataset:", test_df_r3[0].shape)
    print("round 3 train dataset:", train_df_r3[0].shape)
    print("OUTLIERS\n\n")
    #train_df_r1[2].iloc[:, :10].boxplot()
    # outlier detection based on only two features: WL and MAV
    # cleaning train data
    train_df_r1_clean = [pd.DataFrame(columns=train_df_r1[0].columns)] * N_Subj
    train_df_r2_clean = [pd.DataFrame(columns=train_df_r1[0].columns)] * N_Subj
    train_df_r3_clean = [pd.DataFrame(columns=train_df_r1[0].columns)] * N_Subj

    # round 1:
    for s in range(0, N_Subj):
        df = train_df_r1[s]
        for i in range(1,
                       9):  # cleaning for each class of movement must be based only on the samples belonging to that class
            class_group = df.loc[df['class'] == i]
            mean = class_group.mean()
            std = class_group.std()
            x = 2.5  # threshold is considered 2.5 times of std
            for j in range(1, 11):
                class_group = class_group[
                    np.abs(class_group['MAV' + str(j)] - mean['MAV' + str(j)]) <= x * std['MAV' + str(j)]]
                class_group = class_group[
                    np.abs(class_group['WL' + str(j)] - mean['WL' + str(j)]) <= x * std['WL' + str(j)]]
            train_df_r1_clean[s] = train_df_r1_clean[s].append(class_group, ignore_index=True)
        train_df_r1_clean[s]['class'] = train_df_r1_clean[s]['class'].astype(int)

    print("Subject1:")
    print("round1 train data:", train_df_r1[0].shape)
    print("round1 cleaned train data", train_df_r1_clean[0].shape)

    # round 2:
    for s in range(0, N_Subj):
        df = train_df_r2[s]
        for i in range(1, 9):
            class_group = df.loc[df['class'] == i]
            mean = class_group.mean()
            std = class_group.std()
            x = 2.5
            for j in range(1, 11):
                class_group = class_group[
                    np.abs(class_group['MAV' + str(j)] - mean['MAV' + str(j)]) <= x * std['MAV' + str(j)]]
                class_group = class_group[
                    np.abs(class_group['WL' + str(j)] - mean['WL' + str(j)]) <= x * std['WL' + str(j)]]
            train_df_r2_clean[s] = train_df_r2_clean[s].append(class_group, ignore_index=True)
        train_df_r2_clean[s]['class'] = train_df_r2_clean[s]['class'].astype(int)

    print("round2 train data:", train_df_r2[0].shape)
    print("round2 cleaned train data", train_df_r2_clean[0].shape)

    # round 3:
    for s in range(0, N_Subj):
        df = train_df_r3[s]
        for i in range(1, 9):
            class_group = df.loc[df['class'] == i]
            mean = class_group.mean()
            std = class_group.std()
            x = 2.5
            for j in range(1, 11):
                class_group = class_group[
                    np.abs(class_group['MAV' + str(j)] - mean['MAV' + str(j)]) <= x * std['MAV' + str(j)]]
                class_group = class_group[
                    np.abs(class_group['WL' + str(j)] - mean['WL' + str(j)]) <= x * std['WL' + str(j)]]
            train_df_r3_clean[s] = train_df_r3_clean[s].append(class_group, ignore_index=True)
        train_df_r3_clean[s]['class'] = train_df_r3_clean[s]['class'].astype(int)

    print("round3 train data:", train_df_r3[0].shape)
    print("round3 cleaned train data", train_df_r3_clean[0].shape)
    #train_df_r1_clean[2].iloc[:, :10].boxplot()
    print("DEFINE FEATURES SET\n\n")
    # features: MAV, ZC, SSC, WL, HP-A, HP-M, HP-C, SE, CCI, CCII, CCIII, CCIV, RMS, IEMG, SKEW
    TD = list(range(0, 40))  # MAV,ZC,SSC,WL
    ITD = list(range(0, 70)) + list(range(120, 140))  # MAV, ZC, SSC, WL, HP-A, HP-M, HP-C,RMS,IEMG
    CB = list(range(10, 40)) + list(range(50, 90))  # ZC, SSC, WL, HP_M. HP_C, ,SE, CC1
    full = list(range(0, 70)) + list(range(120, 140)) +list(range(70, 120)) + list(range(140, 150))
    full_move=list(range(0, 70)) + list(range(120, 140)) +list(range(70, 120)) + list(range(140, 152))

    Samp_pipline = list(range(30, 40)) + list(range(70, 90)) + list(range(120, 130))  # WL,SE,CCI,IEMG
    HP = list(range(40, 70))
    good = list(range(30, 40)) + list(range(50, 70))
    opt = list(range(30, 40)) + list(range(50, 70)) + list(range(80, 90))  # WL, HP-M, HP-C, CCI
    full_skew = list(range(0, 140))

    f_set = full  # this should be changed when you want to use different feature set (and following cells must be executed again)
    f_name = "full"  # this will be used if you want to save the scaler model and trained model for future used
    # after changing feature set this cell and following ones must be executed again
    # extract x and y for training and testing sets
    X_train_r1 = [0] * N_Subj
    X_train_r2 = [0] * N_Subj
    X_train_r3 = [0] * N_Subj

    y_train_r1 = [0] * N_Subj
    y_train_r2 = [0] * N_Subj
    y_train_r3 = [0] * N_Subj

    X_test_r1 = [0] * N_Subj
    X_test_r2 = [0] * N_Subj
    X_test_r3 = [0] * N_Subj

    y_test_r1 = [0] * N_Subj
    y_test_r2 = [0] * N_Subj
    y_test_r3 = [0] * N_Subj

    for s in range(0, N_Subj):
        X_train_r1[s] = train_df_r1_clean[s].iloc[:, f_set]
        X_train_r2[s] = train_df_r2_clean[s].iloc[:, f_set]
        X_train_r3[s] = train_df_r3_clean[s].iloc[:, f_set]
        y_train_r1[s] = train_df_r1_clean[s].iloc[:, -1]
        y_train_r2[s] = train_df_r2_clean[s].iloc[:, -1]
        y_train_r3[s] = train_df_r3_clean[s].iloc[:, -1]

        X_test_r1[s] = test_df_r1[s].iloc[:, f_set]
        y_test_r1[s] = test_df_r1[s].iloc[:, -1]
        y_test_r1[s] = y_test_r1[s].astype(int)
        X_test_r2[s] = test_df_r2[s].iloc[:, f_set]
        y_test_r2[s] = test_df_r2[s].iloc[:, -1]
        y_test_r2[s] = y_test_r2[s].astype(int)
        X_test_r3[s] = test_df_r3[s].iloc[:, f_set]
        y_test_r3[s] = test_df_r3[s].iloc[:, -1]
        y_test_r3[s] = y_test_r3[s].astype(int)

    print("Subject1:")
    print("train cleaned:", X_train_r1[0].shape, "test", X_test_r1[0].shape)
    print("CONCATENATION ROUNDS\n\n")
    # concatinating the rounds for train and test
    X_train = [0] * N_Subj
    y_train = [0] * N_Subj

    X_test = [0] * N_Subj
    y_test = [0] * N_Subj

    for s in range(0, N_Subj):
        X_train[s] = pd.concat([X_train_r1[s], X_train_r2[s], X_train_r3[s]], ignore_index=True)
        y_train[s] = pd.concat([y_train_r1[s], y_train_r2[s], y_train_r3[s]], ignore_index=True)

        X_test[s] = pd.concat([X_test_r1[s], X_test_r2[s], X_test_r3[s]], ignore_index=True)
        y_test[s] = pd.concat([y_test_r1[s], y_test_r2[s], y_test_r3[s]], ignore_index=True)
    print("X_test list of:", len(X_train))
    print("patient 1 instances:", X_train[0].shape)

    num_patient = len(X_train)
    print("number of patient: ", num_patient)

    X = []
    y = []
    for patient in range(num_patient):
        X_patient = pd.concat([X_train[patient], X_test[patient]], axis=0)
        X.append(X_patient)
        y_patient = pd.concat([y_train[patient], y_test[patient]], axis=0)
        y.append(y_patient)

    # print("STANDARD SCALER FOR ALL PATIENT\n\n")
    # scaler = StandardScaler(copy=False).fit(X)
    # scaler.transform(X)
        # for some algorithms (e.g. perceptron) shuffling the samples is very important

    #from sklearn.utils import shuffle
    # for i in range(N_Subj):
    #     X_train[i], y_train[i] = shuffle(X_train[i], y_train[i])
    #     X_test[i], y_test[i] = shuffle(X_test[i], y_test[i])
    #     print("full test label shape : " + str(y_test[i].shape))
    return X,y

def classify(file_path):
    '''

    :param file_path: path where the scores (accuracy, f1 and model) are saved

    :return:
    '''
    #load data and concatenate them to form one single dataset

    X,y= load_dataset_all()
    num_patient = len(X)
    for patient in range(num_patient):
        for i in range(1,11):
            drop_set = ["SSC{}".format(i), "ZC{}".format(i), "MAV{}".format(i), "HP_A{}".format(i), "IEMG{}".format(i), "WL{}".format(i)]
            X[patient] = X[patient].drop(drop_set, axis=1)
            X[patient] = X[patient].drop(["SE{}".format(i), "CCI_{}".format(i), "CCII_{}".format(i),"CCIII_{}".format(i), "CCIV_{}".format(i), "SKEW{}".format(i)], axis=1)

    print(len(X), len(y))
    random_seed = 0;
    n_estimators = 50;
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']

    #models used to classify data
    models = {'LDA': LinearDiscriminantAnalysis(solver='svd'),
              # 'Decision Tree': DecisionTreeClassifier(max_depth=None,
              #                                        random_state=random_seed),
              # 'Bagging': BaggingClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=n_estimators,
              #                              random_state=random_seed),
              # 'Random Forest': RandomForestClassifier(n_estimators=n_estimators,
              #                                         random_state=random_seed),
              'Extremely Randomized Trees': ExtraTreesClassifier(n_estimators=n_estimators,
                                                                 random_state=random_seed),
              #'Ada Boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=n_estimators,
                                              #random_state=random_seed),
              'SVM_tuned': SVC(kernel='linear', C=1, class_weight="balanced", gamma='auto', probability=True),
              'KNN': KNeighborsClassifier(n_neighbors=40)
              # 'Logistic Regression': LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0,
              #                                          random_state=random_seed),
              #'Naive Bayes': GaussianNB(var_smoothing=0.1)

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

        scaler = StandardScaler(copy=False).fit(x_train)
        scaler.transform(x_train)
        scaler = StandardScaler(copy=False).fit(x_test)
        scaler.transform(x_test)

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
    score[model_name] = {'acc': acc, 'f1': f1}

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

    path = "/home/tesi/Thesis-XAI/resources/classify_all_LOSO_2.pkl"
    classify(path)

