# API-for-Data-Science

Este projeto é um estudo de caso de machine learning com uma API construída com FastAPI e orientada a microserviço. A API permite aos usuários fazer previsões baseadas em dados de treinamento e fornecer resultados em tempo real. O objetivo é fornecer uma plataforma de fácil uso para aplicativos e sistemas que precisam de análise preditiva.


# Dados de previsão
    "ge": Número de anos de idade
    "sex": Gênero (0 = feminino, 1 = masculino)
    "chest_pain_type": Tipo de dor no peito (0-3)
    "resting_bp": Pressão arterial em repouso
    "cholesterol": Nível de colesterol
    "fasting_bs": Açúcar no sangue em jejum
    "resting_ecg": Resultado do ECG em repouso
    "max_hr": Batimentos cardíacos máximos durante o exercício
    "exercise_angina": Presença de angina durante o exercício (0 = não, 1 = sim)
    "oldpeak": Depressão do segmento ST
    "st_slope": Inclinação do segmento ST

# Testar a api 

``` python

import requests

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

json_data = {
    'categorical_features': {
        'Age': 40,
        'RestingBP': 140,
        'Cholesterol': 289,
        'FastingBS': 0,
        'MaxHR': 178,
    },
    'numerical_features': {
        'Sex': 'M',
        'ChestPainType': 'ATA',
        'RestingECG': 'Normal',
        'ExerciseAngina': 'N',
        'ST_Slope': 'Up',
    },
}

response = requests.post('http://127.0.0.1:8000/predict', headers=headers, json=json_data)
print(response.status_code)
print(response.json())
```