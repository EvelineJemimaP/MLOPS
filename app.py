import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

st.title('Web Deployment of Medical Diagnostic App ⚕️')
st.subheader('Is the person diabetic?')
df=pd.read_csv('diabetes.csv')
if st.sidebar.checkbox('View Data',False):
    st.write(df)
if st.sidebar.checkbox('Vew Distributions',False):
    df.hist()
    plt.tight_layout()
    st.pyplot()
# Step 1: Load the pickled model
model=open('rfc.pickle','rb')
clf=pickle.load(model)
model.close()

# Step 2: Get the front-end user input
pregs=st.number_input('Pregnancies',0,17,0)
glucose=st.slider('Glucose',44,199,44)
bp=st.slider('BloodPressure',20,140,20)
skin=st.slider('SkinThickness',7,99,7)
insulin=st.slider('Insulin',14,850,14)
bmi=st.slider('BMI',18,67,18)
dpf=st.slider('DiabetesPedigreeFunction',0.05,2.50,0.05)
age=st.slider('Age',21,85,21)

# Step 3: Convert user-input to model input
data={
    'Pregnancies':pregs, 'Glucose':glucose, 'BloodPressure':bp, 'SkinThickness':skin, 'Insulin':insulin,
       'BMI':bmi, 'DiabetesPedigreeFunction':dpf, 'Age':age
}
input_data=pd.DataFrame([data])

# Step 4: Get the predictions and print the result
predictions=clf.predict(input_data)[0]
if st.button('Predict'):
    if predictions==1:
        st.subheader('Diabetic')
    else:
        st.subheader('Healthy')
