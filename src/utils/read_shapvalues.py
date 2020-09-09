import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
num_pat=11
shap_LDA=[]

shap_means_patients=[]
for pat in range(num_pat):
    file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources\\shap_LDA_patient_{}_tmp.pkl'.format(pat)
    with open(file_path, 'rb') as f:
        shap_values = pickle.load(f)
    #print(len(shap_values))
    #print(shap_values[0])
    #plt.figure()
    #plt.imshow(shap_values[0], cmap='jet', aspect='auto', alpha=0.5)
    #plt.show()
    shap_mean=pd.DataFrame(columns=shap_values[0].columns.values)
    for i in range(8):
        shap_mean.loc[i]=shap_values[i].mean(axis=0)
        print(shap_mean.shape)
        #print(np.max(shap_mean))
        #print(np.argmax(shap_mean))
    shap_means_patients.append((shap_mean))
    print(len(shap_means_patients))
    print(len(shap_means_patients[0]))
    shap_LDA.append(shap_values)
np.argmax

