import pickle
file_path = 'C:\\Users\\noemi\\Desktop\\university\\university\\tesi\\Thesis-XAI\\resources' \
            '\\results_classification\\scores_personalized_models.pkl '
with open(file_path, 'rb') as f:
    scores = pickle.load(f)

for patient in scores:
    print("\t\t\t\t Patient {}".format(patient))
    for model in scores[patient].keys():
        print ("model: ", model)
        print("\t\t\t\t accuracy: ", scores[patient][model]['acc'])
        print("\t\t\t\t f1: ", scores[patient][model]['f1'])
