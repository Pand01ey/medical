import numpy as np
import joblib
import streamlit as st
import sklearn
import pickle

# loading the saved model
loaded_model = pickle.load(open("Medical1.pkl","rb"))
print(loaded_model)


#creating a function for Prediction
def medical_insurance_cost_prediction(input_data,loaded_model):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.array(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data)

    return prediction

#giving a title
st.title('Medical Insurance Cost Prediction')

#getting input from the user

age = st.text_input('Age')
sex = st.text_input('Sex: 0 -> Female, 1 -> Male')
bmi = st.text_input('Body Mass Index')
children = st.text_input('Number of Children')
smoker = st.text_input('Smoker: 0 -> No, 1 -> Yes')
region = st.text_input('Region of Living: 0 -> NorthEast, 1-> NorthWest, 2-> SouthEast, 3-> SouthWest')

#code for prediction
diagnosis = ''

# getting the input data from the user
if st.button('Predicted Medical Insurance Cost: '):
    diagnosis = medical_insurance_cost_prediction([int(age),int(sex),float(bmi),int(children),int(smoker),int(region)],loaded_model=loaded_model)
    
st.success(diagnosis)
