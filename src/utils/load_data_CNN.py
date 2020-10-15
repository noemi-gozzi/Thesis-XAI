import numpy as np


def load_data_CNN():

    dataset = np.load('../resources/data/final_dataset-004.npy', allow_pickle=True)
    label = np.load('../resources/data/final_label.npy', allow_pickle=True)
    move = np.load('../resources/data/final_move.npy', allow_pickle=True)
    xtrain_r = []
    xtest_r = []
    ytrain_r = []
    ytest_r = []

    Xtrainset = []
    Xtestset = []
    Ytrainset = []
    Ytestset = []

    for subject in range(11):
        xtrain_rr = []
        xtest_rr = []
        ytrain_rr = []
        ytest_rr = []
        for r in range(3):
            xtrain_rrr = np.array([[[0] * 512] * 10])
            xtest_rrr = np.array([[[0] * 512] * 10])
            ytrain_rrr = np.array([0])
            ytest_rrr = np.array([0])

            tempset = dataset[subject][r]
            tempmove = move[subject][r]
            templabel = label[subject][r]
            for clas in range(1, 9):
                movenum = np.unique(tempmove[templabel == clas])
                trainsize = int((2 / 3) * len(movenum))
                for i in range(trainsize):
                    xtrain_rrr = np.append(xtrain_rrr, tempset[tempmove == movenum[i]], axis=0)
                    ytrain_rrr = np.append(ytrain_rrr, templabel[tempmove == movenum[i]], axis=0)
                for i in range(trainsize, len(movenum)):
                    xtest_rrr = np.append(xtest_rrr, tempset[tempmove == movenum[i]], axis=0)
                    ytest_rrr = np.append(ytest_rrr, templabel[tempmove == movenum[i]], axis=0)

            xtrain_rr.append(xtrain_rrr[1:])
            xtest_rr.append(xtest_rrr[1:])
            ytrain_rr.append(ytrain_rrr[1:])
            ytest_rr.append(ytest_rrr[1:])
        xtrain_r.append(xtrain_rr)
        xtest_r.append(xtest_rr)
        ytrain_r.append(ytrain_rr)
        ytest_r.append(ytest_rr)

        Xtrainset.append(np.append(np.append(xtrain_rr[0], xtrain_rr[1], axis=0), xtrain_rr[2], axis=0))
        Xtestset.append(np.append(np.append(xtest_rr[0], xtest_rr[1], axis=0), xtest_rr[2], axis=0))
        Ytrainset.append(np.append(np.append(ytrain_rr[0], ytrain_rr[1], axis=0), ytrain_rr[2], axis=0))
        Ytestset.append(np.append(np.append(ytest_rr[0], ytest_rr[1], axis=0), ytest_rr[2], axis=0))
    # from sklearn.utils import shuffle
    # for i in range(11):
    #     Xtrainset[i] , Ytrainset[i] = shuffle(Xtrainset[i],Ytrainset[i])
    #     Xtestset[i] , Ytestset[i] = shuffle(Xtestset[i],Ytestset[i])
      # Xtrainset[i] = np.swapaxes(Xtrainset[i],1,2)
      # Xtestset[i] = np.swapaxes(Xtestset[i],1,2)
        #print("full test label shape : " + str(Xtestset[i].shape))
    for i in range (11):
        Xtrainset[i] = Xtrainset[i].reshape(Xtrainset[i].shape[0],Xtrainset[i].shape[1],Xtrainset[i].shape[2],1)
        Xtestset[i] = Xtestset[i].reshape(Xtestset[i].shape[0],Xtestset[i].shape[1],Xtestset[i].shape[2],1)
    print("full test label shape : " + str(Xtestset[0].shape))
    return Xtrainset, Ytrainset, Xtestset, Ytestset

