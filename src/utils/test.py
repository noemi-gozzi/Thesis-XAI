# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:46:52 2020

@author: noemi
"""


import sys
import os, os.path
import numpy as np
import matplotlib.pyplot as plt

time=np.linspace(0, 250, 512)
root_path="C:/Users/noemi/Desktop/university/university/tesi/Thesis-XAI"
dataset = np.load(root_path+'/resources/data_deep/data_ordered/final_dataset_V3.npy', allow_pickle=True)
label = np.load(root_path+'/resources/data_deep/data_ordered/final_label_V3.npy', allow_pickle=True)
move = np.load(root_path+'/resources/data_deep/data_ordered/final_move_V3.npy', allow_pickle=True)

print('Data base Shape', np.shape(dataset))
print('subject 1 database shape', dataset[0][0].shape)

print('lable Shape', np.shape(label))
print('subject 1 lable shape', label[0][0].shape)

print('move Shape', np.shape(move))
print('subject 1 move shape', move[0][0].shape)

# restore np.load for future normal usage
# np.load = np_load_old
with open(root_path + '/resources/data_deep/data_ordered/Xtest_big.pkl', 'rb') as f:
    Xtest = pickle.load(f)
Xtestset_drop = []
for patient in range(11):
    X_tmp = np.zeros((len(Xtest[patient]), 6, 512, 1))
    for index in range(len(Xtest[patient])):
        X_tmp[index] = np.delete(Xtest[patient][index], [6, 7, 8, 9], axis=0)
    Xtestset_drop.append(X_tmp)
with open(root_path + '/resources/data_deep/data_ordered/ytest_big.pkl', 'rb') as f:
    ytest = pickle.load(f)

patient=8
ex=9
single_instance=Xtestset_drop[patient]
single_instance=single_instance[ex, :, :]
cls=ytest[patient][ex]
print(cls)
for channel in range(6):
    plt.plot(time, single_instance[channel,:])


plt.legend(("1", "2", "3", "4", "5", "6"))
plt.title('patient {}, label {}'.format(patient, cls))
plt.xlabel('TIME ms')
#plt.savefig(root_path+"/EMG_patient{}_ex{}_label{}.png".format(patient, ex, cls))

plt.show()

