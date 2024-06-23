'''
Code of ethics affirmation:

Before writing this code, I consulted Chat GPT and Bing – Chat for how to code for an app on streamlit
'''

import streamlit as st # scikit-learn version used is 1.5.0
import pandas as pd
import numpy as np
import joblib
import os
import sklearn
import sklearn.metrics as metrics 

def load_model(model_path='my_model.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# This function is used to make a prediction
def predict_rating(model, input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

def main():
    try:
        rf_model = load_model('my_model.pkl')  # Call the loading model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return

    st.title("Predict a football player's rating here:")
    st.write('Adjust the values of the 15 features below as necessary to find the overall rating')
    
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffd1df;
        color: #333333;
    }
    .stButton > button {
        background-color: #B24BB2;
        color: white;
    }
    
    """,
    unsafe_allow_html=True
    )

    # Input features
    movement_reactions = st.number_input('Movement Reactions', min_value=0, max_value=100, value=50, key='movement_reactions')
    mentality_composure = st.number_input('Mentality Composure', min_value=0, max_value=100, value=50, key='mentality_composure')
    potential = st.number_input('Potential', min_value=0, max_value=100, value=50, key='potential')
    lf = st.number_input('LF', min_value=0, max_value=100, value=50, key='lf')
    cf = st.number_input('CF', min_value=0, max_value=100, value=50, key='cf')
    rf = st.number_input('RF', min_value=0, max_value=100, value=50, key='rf')
    wage_eur = st.number_input('Wage (EUR)', min_value=0, value=0, key='wage_eur')
    lw = st.number_input('LW', min_value=0, max_value=100, value=50, key='lw')
    rw = st.number_input('RW', min_value=0, max_value=100, value=50, key='rw')
    power_shot_power = st.number_input('Power Shot Power', min_value=0, max_value=100, value=50, key='power_shot_power')
    lwb = st.number_input('LWB', min_value=0, max_value=100, value=50, key='lwb')
    rwb = st.number_input('RWB', min_value=0, max_value=100, value=50, key='rwb')
    value_eur = st.number_input('Value (EUR)', min_value=0, value=0, key='value_eur')
    release_clause_eur = st.number_input('Release Clause (EUR)', min_value=0, value=0, key='release_clause_eur')
    passing = st.number_input('Passing', min_value=0, max_value=100, value=50, key='passing')

    if st.button('Overall Rating'):
        input_features = [
            movement_reactions, mentality_composure, potential, lf, cf, rf,
            wage_eur, lw, rw, power_shot_power, lwb, rwb, value_eur, release_clause_eur, passing
        ]
        try:
            rf_prediction = predict_rating(rf_model, input_features)
            st.success(f'Random Forest predicted overall rating: {rf_prediction:.2f}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
