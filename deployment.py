import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    df = pd.read_csv('data/predictive_maintenance.csv')
    return df

def clean_columns(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('[\[\]<]', '', regex=True).str.replace(' ', '_')
    return df

def prepare_data(df):
    df = clean_columns(df)
    
    A = df[['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']]
    b = df['Target']
    
    return A, b

def train_model(A_train, b_train, A_test, b_test):
    model = XGBClassifier(random_state=42)
    model.fit(A_train, b_train)
    
    b_pred = model.predict(A_test)
    
    accuracy = accuracy_score(b_test, b_pred)
    precision = precision_score(b_test, b_pred)
    recall = recall_score(b_test, b_pred)
    f1 = f1_score(b_test, b_pred)
    
    return model, accuracy, precision, recall, f1

def make_prediction(model, user_input):
    prediction = model.predict(np.array([user_input]))
    return prediction[0]

def main():
    st.title('Predictive Maintenance: XGBoost Model Evaluation')

    df = load_data()

    st.subheader('Dataset Preview:')
    st.write(df.head())

    A, b = prepare_data(df)

    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42, stratify=b)
    model, accuracy, precision, recall, f1 = train_model(A_train, b_train, A_test, b_test)

    if st.checkbox('Show Model Evaluation Metrics'):
        
        st.subheader('Model Evaluation Metrics:')
        st.write(f'Accuracy: {accuracy:.4f}')
        st.write(f'Precision: {precision:.4f}')
        st.write(f'Recall: {recall:.4f}')
        st.write(f'F1 Score: {f1:.4f}')

    
    st.subheader('Make a Prediction')

    air_temp = st.slider('Air Temperature (K)', min_value=0.0, max_value=400.0, value=300.0, step=0.1)
    proc_temp = st.slider('Process Temperature (K)', min_value=0.0, max_value=400.0, value=300.0, step=0.1)
    rot_speed = st.slider('Rotational Speed (rpm)', min_value=0, max_value=5000, value=2500, step=1)
    torque = st.slider('Torque (Nm)', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    tool_wear = st.slider('Tool Wear (min)', min_value=0.0, max_value=500.0, value=250.0, step=0.1)

    user_input = [air_temp, proc_temp, rot_speed, torque, tool_wear]

    if st.button('Predict'):

        model = XGBClassifier(random_state=42)
        model.fit(A_train, b_train)

        prediction = make_prediction(model, user_input)

        if prediction == 1:
            st.write("Prediction: The machine is predicted to fail!")
        else:
            st.write("Prediction: The machine is predicted to not fail!")

if __name__ == '__main__':
    main()
