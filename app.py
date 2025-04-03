import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open("iris_svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("iris_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# APP UI
st.title("iris Flower Classifier Using Support Vector Machines")
st.write("Input sepal and petal measurements below")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 4.0)
sepal_width = st.slider("Sepal width (cm)", 2.0, 5.0, 2.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.0)
petal_width = st.slider("Petal width (cm)", 0.1, 3.0, 0.1)

#Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

labels = ["iris-setosa", "Iris-versicolor", "Iris-virginica"]
st.subheader("Prediction")
st.success(f"This flower is likely **{labels[prediction]}**")
