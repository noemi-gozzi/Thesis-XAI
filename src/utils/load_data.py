
import pandas as pd # version 0.24.2
import numpy as np #version 1.16.4
import random
from sklearn.preprocessing import StandardScaler


def load_dataset():
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
            "C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\data_features\\Version 3(V2 without shuffle)\\database_v3_sbj{}_s1_WL512_S128_r1.csv".format(
                i + 1))
        df_r1[i].columns = cl_names

        df_r2[i] = pd.read_csv(
            "C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\data_features\\Version 3(V2 without shuffle)\\database_v3_sbj{}_s1_WL512_S128_r2.csv".format(
                i + 1))
        df_r2[i].columns = cl_names

        df_r3[i] = pd.read_csv(
            "C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\data_features\\Version 3(V2 without shuffle)\\database_v3_sbj{}_s1_WL512_S128_r3.csv".format(
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
    print("STANDARD SCALER FOR EACH PATIENT\n\n")
    for s in range(0, N_Subj):
        scaler = StandardScaler(copy=False).fit(X_train[s])
        scaler.transform(X_train[s])
        scaler.transform(X_test[s])
        # for some algorithms (e.g. perceptron) shuffling the samples is very important

    from sklearn.utils import shuffle
    # for i in range(N_Subj):
    #     X_train[i], y_train[i] = shuffle(X_train[i], y_train[i])
    #     X_test[i], y_test[i] = shuffle(X_test[i], y_test[i])
    #     print("full test label shape : " + str(y_test[i].shape))
    return X_train, y_train, X_test, y_test