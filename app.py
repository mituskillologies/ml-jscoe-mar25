# pip install streamlit
import streamlit as st 
import pandas as pd
import joblib
import numpy as np

st.title("Iris Flower Prediction")

sepal_length = st.number_input("Sepal Length")

sepal_width = st.number_input("Sepal Width")

petal_length = st.number_input("Petal Length")

petal_width = st.number_input("Petal Width")

df = pd.DataFrame({
    "sepal_length":[sepal_length],
    "sepal_width":[sepal_width],
    "petal_length":[petal_length],
    "petal_width":[petal_width]
})



model = joblib.load("model.joblib")

pred = model.predict(df)

st.write(pred[0])