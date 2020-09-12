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

dataset = np.load('../resources/data/final_dataset-004.npy', allow_pickle=True)
label = np.load('../resources/data/final_label.npy', allow_pickle=True)
move = np.load('../resources/data/final_move.npy', allow_pickle=True)

print('Data base Shape', np.shape(dataset))
print('subject 1 database shape', dataset[0][0].shape)

print('lable Shape', np.shape(label))
print('subject 1 lable shape', label[0][0].shape)

print('move Shape', np.shape(move))
print('subject 1 move shape', move[0][0].shape)

# restore np.load for future normal usage
# np.load = np_load_old

single_instance=dataset[0][0]
single_instance=single_instance[0, :, :]

plt.plot(time, single_instance[0,:])
plt.title('patient 1, channel 1 (0)')
plt.xlabel('TIME ms')
plt.show()

plt.imshow(single_instance, interpolation='nearest', cmap='Greys')
plt.imshow(single_instance, cmap='Greys')