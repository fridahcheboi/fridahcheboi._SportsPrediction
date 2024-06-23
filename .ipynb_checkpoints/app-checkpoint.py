import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained models
def load_model():
    rf_model = joblib.load('regression_model.pkl')
    return rf_model

# Define a function to predict rating
def predict_rating(model, input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

def main():
    # Load models
    rf_model= load_model()

    # Title and description
    st.title('Football Player Rating Prediction')
    st.write('Enter the features of a football player to predict their rating.')

    # Input features for prediction
    movement_reactions = st.number_input('Movement Reactions', min_value=0, max_value=100, value=50)
    potential = st.number_input('Potential', min_value=0, max_value=100, value=50)
    mentality_composure = st.number_input('Mentality Composure', min_value=0, max_value=100, value=50)
    passing = st.number_input('Passing', min_value=0, max_value=100, value=50)
    wage_eur = st.number_input('Wage (EUR)', min_value=0, value=0)
    dribbling = st.number_input('Dribbling', min_value=0, max_value=100, value=50)
    value_eur = st.number_input('Value (EUR)', min_value=0, value=0)
    physic = st.number_input('Physic', min_value=0, max_value=100, value=50)

    # Add more features as needed

    # Predict
    if st.button('Predict Rating'):
        input_features = [movement_reactions, potential, mentality_composure, passing, wage_eur, dribbling, value_eur, physic]
        # Add more features to input_features as needed
        rf_prediction = predict_rating(rf_model, input_features)
        gb_prediction = predict_rating(gb_model, input_features)

        st.success(f'Random Forest Predicted Rating: {rf_prediction:.2f}')
        st.success(f'Gradient Boosting Predicted Rating: {gb_prediction:.2f}')

if __name__ == '__main__':
    main()
