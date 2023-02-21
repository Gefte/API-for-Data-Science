import streamlit as st
import requests
import json

def make_prediction(json_data):
    
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    }
    
    url = 'http://127.0.0.1:8000/predict'
    
    response = requests.post(url, headers=headers, json=json_data)
    if response.status_code == 200:
        data = response.json()
        prediction = data['predictions'][0]
        probability = data['probabilities'][0]
        return prediction, probability
    else:
        return response.status_code, None

st.title('Predição de Doenças Cardíacas')
st.write('Insira os dados do paciente abaixo:')

age = st.slider('Age', min_value=0, max_value=100, step=1, value=40)
resting_bp = st.slider('Resting Blood Pressure [RestingBP]', min_value=0, max_value=200, step=1, value=140)
cholesterol = st.slider('Cholesterol', min_value=0, max_value=600, step=1, value=289)
fasting_bs = st.radio('Fasting Blood Sugar [FastingBS]', options=['Yes', 'No'], index=1)
max_hr = st.slider('Maximum Heart Rate Achieved [MaxHR]', min_value=0, max_value=250, step=1, value=178)

sex = st.selectbox('Sex', options=['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type [ChestPainType]', options=['ATA', 'NAP', 'ASY', 'TA'])
resting_ecg = st.selectbox('Resting Electrocardiographic Results [RestingECG]', options=['Normal', 'ST', 'LVH'],)
exercise_angina = st.selectbox('Exercise Induced Angina [ExerciseAngina]', options=['Y', 'N'])
st_slope = st.selectbox('ST Slope [ST_Slope]', options=['Up', 'Flat', 'Down'])


if fasting_bs == 'Yes':
    fasting_bs = 1
else:
    fasting_bs = 0

categorical_data = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    
    
}

numerical_data = {
    'Sex': sex[0],
    'ChestPainType': chest_pain_type,
    'RestingECG': resting_ecg,
    'ExerciseAngina': exercise_angina[0],
    'ST_Slope': st_slope
}

json_data = {
    'categorical_features': categorical_data,
    'numerical_features': numerical_data,
}


if st.button('Submit'):
    # enviar a solicitação de previsão
    prediction, probability = make_prediction(json_data)
    
    if prediction == 0:
        color = 'green'
        message = 'You are healthy!'
    else:
        color = 'red'
        message = 'You are sick!' 
    st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px;">'
                f'<h2 style="color: white;">{message}</h2>'
                f'<h3 style="color: white;">Probability: {probability:.2f}</h3>'
                '</div>', unsafe_allow_html=True)