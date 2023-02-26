import streamlit as st
import pandas as pd
import pickle
import numpy as np
st.write('''
# Penguin Species Classifier application
This app is used to classify if a particular species based on the body features.
''')
st.sidebar.header("Input Features")
st.sidebar.markdown('''
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
''')
uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def get_input_features():
        island = st.sidebar.selectbox("Island",("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex",("male", "female"))
        bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4207.0)
        data = {"island":island,
                "sex":sex,
                "bill_length_mm":bill_length_mm,
                "bill_depth_mm":bill_depth_mm,
                "flipper_length_mm":flipper_length_mm,
                "body_mass_g":body_mass_g}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = get_input_features()

penguins_raw = pd.read_csv("penguins_cleaned.csv")
penguins_raw = penguins_raw.drop("species", axis=1)
df = pd.concat([input_df, penguins_raw], axis=0)
encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]
st.subheader("User Input features")
if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)")
    st.write(df)
load_clf = pickle.load(open("penguins_classifier.pkl", "rb"))
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
st.subheader("Prediction")
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])
st.subheader("Prediction Probability")
st.write(prediction_proba)