
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import streamlit as st
st.title('Student Exam Performance Indicator')
st.subheader('Student Exam Performance Prediction')
# Define the function to make predictions
def predict_datapoint(gender, ethnicity, parental_education, lunch, test_preparation, writing_score, reading_score):
   
        data=CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_preparation,
            reading_score=writing_score,    
            writing_score=reading_score

        )
        pred_df=data.get_data_as_data_frame()
        st.write(pred_df)
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        result=predict_datapoint()
        return result
   

# Streamlit UI
def main():
    st.title('Student Exam Performance Indicator')
    st.subheader('Student Exam Performance Prediction')

    # Form inputs
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ethnicity = st.selectbox('Race or Ethnicity', ['Group A', 'Group B', 'Group C', 'Group D', 'Group E'])
    parental_education = st.selectbox('Parental Level of Education', ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"])
    lunch = st.selectbox('Lunch Type', ['Free/Reduced', 'Standard'])
    test_preparation = st.selectbox('Test Preparation Course', ['None', 'Completed'])
    writing_score = st.number_input('Writing Score out of 100', min_value=0, max_value=100)
    reading_score = st.number_input('Reading Score out of 100', min_value=0, max_value=100)

    # Predict button
    if st.button('Predict your Maths Score'):
        # Call the prediction function
        math_score_prediction = predict_datapoint(gender, ethnicity, parental_education, lunch, test_preparation, writing_score, reading_score)
        
        # Display the prediction result
        st.success(f'The predicted math score is {math_score_prediction}')

if __name__ == '__main__':
    main()
