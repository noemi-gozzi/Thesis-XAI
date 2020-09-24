import pandas as pd
import pickle

with open("../X_train.pkl", 'rb') as f:
    X_train = pickle.load(f)
with open("../X_test.pkl", 'rb') as f:
    X_test = pickle.load(f)
with open("../y_train.pkl", 'rb') as f:
    y_train = pickle.load(f)
with open("../y_test.pkl", 'rb') as f:
    y_test = pickle.load(f)

num_pat=len(X_train)
shap_values_indexed=[]
for patient in range(num_pat):
    file_path = '../resources/SHAP/SHAP_EXTRA/shap_ExtremelyRandomizedTrees_patient_{}_tmp.pkl'.format(patient)
    with open(file_path, 'rb') as f:
        shap_values = pickle.load(f)

    for cls in range(len(shap_values)):
        shap_values[cls]=shap_values[cls].reindex(X_test[patient].index)

    shap_values_indexed.append(shap_values)

for patient in range(num_pat):
    for cls in range(8):
        shap_values_indexed[patient][cls] = shap_values_indexed[patient][cls].sort_index(ascending=True)



with open('../resources/SHAP/SHAP_EXTRA/shap_XTR_Treepath_ascending_indexed.pkl','wb') as f:
    pickle.dump(shap_values_indexed, f)