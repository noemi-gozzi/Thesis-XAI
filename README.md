# Thesis
AI has recently found a fertile ground in EMG signal decoding for prosthesis control: AI-powered myo-controlled robotic limbs are showing promising performances in restoring missing functions providing conscious control of the movement. Nevertheless, their acceptance is strongly limited by the notion of AI models as black-boxes. In critical fields, such as medicine and neuroscience, understanding the neurophysiological phenomena underlying models outcomes is as relevant as the classification performance. We propose to put XAI into the EMG hand movement classification task to understand the outcome of machine learning models (LDA, SVM and XRT) and deep learning models such as CNNs with respect to physiological processes. We adapt and generalise SHAP and Grad-CAM to our specific type of data and problem, evaluating the contribution of each input feature or input pixel to the prediction. 

## Virtualenv

- pip3 install virtualenv
- source venv/bin/activate
- pip install -r requirements.txt
- pip3 freeze > requirements.txt
