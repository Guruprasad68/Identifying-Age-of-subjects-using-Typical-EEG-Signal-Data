# Identifying-Age-of-subjects-using-Typical-EEG-Signal-Data





### This project is a part of the course ECE 539: Introduction to Artificial Neural Neural Networks and Fuzzy Systems offered by Prof Yu Hen Hu, UW Madison in Fall 2021. Other collaborators were Siddharth Subramani and Sowmya Vemparala.



#### ALL THE CODES BELOW WERE DEVELOPED SOLELY BY THE TEAM USING PYTHON LANGAUGE AND COMMON ML PACKAGES LIKE KERAS, SKLEARN ETC. NO CODE WAS TAKEN FROM ANOTHER REPOSITORY.

This project aims to classify EEG recordings of healthy subjects into the following age groups 10-30, 31-40, 41-50, 51-60, 61-90.

The dataset used is a part of the Kaggle Dataset "EEG for Age Prediction"

This repository contains python codes developed for the project. Includes:

beta_waves_extraction.py, blstm_lstm_beta.py : A Bi-LSTM model based on P. Kaushik, A. Gupta, P. P. Roy and D. P. Dogra (2019): "EEG-Based Age and Gender Prediction Using Deep BLSTM-LSTM Network Model," in IEEE Sensors Journal, vol. 19, no. 7, pp. 2634-2641, doi: 10.1109/JSEN.2018.2885582.

create_m70_32rmse_feat.pkl.py : Preprocessing code, to generate the RMSE features on a local scale and split the data into train/test/validation.

tradML.py : Explores various ML apporaches like KNN, SVM, Logistic Regression, XGBoost, Random Forest using Scikit Learn

DNN.py: Code for the fullyconnected Network

cnn.py: Code for the convolutional network approach

lstm.py: Code for Bi-LSTM approach

mtl_model.py: Code for a Multi Task Learning approach, using a Convolutional Autoencoder to reconsstruct the input and predict the age groups jointly.

feature_importance_SHAP.py: Code for analyzing feature importance.




