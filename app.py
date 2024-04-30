import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import streamlit as st

# Streamlit UI
def main():
    st.title('Student Exam Performance Indicator')
    st.subheader('Student Exam Performance Prediction')

    # Form inputs
    gender = st.selectbox('gender', ['male', 'female'])
    ethnicity = st.selectbox('ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_level_of_education= st.selectbox('Parental LeveEducation', ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"])
    lunch = st.selectbox('Lunch Type', ['free/reduced', 'standard'])
    test_preparation = st.selectbox('Test Preparation Course', ['none', 'completed'])
    writing_score = st.slider("Enter Writing Marks", min_value=0, max_value=100, value=50)
    reading_score = st.slider("Enter Reading Marks", min_value=0, max_value=100, value=50)
    # Predict button
    if st.button('Predict your Maths Score'):
        try:
            # Create CustomData object
            data = CustomData(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation,
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Get data as DataFrame
            pred_df = data.get_data_as_data_frame()
            st.write(pred_df)

            # Perform prediction
            predict_pipeline = PredictPipeline()
            math_score_prediction = predict_pipeline.predict(pred_df)

            # Display the prediction result
            st.success(f'The predicted math score is {math_score_prediction[0]}')

        except Exception as e:
            st.error(f'Error occurred: {str(e)}')

if __name__ == '__main__':
    main()

