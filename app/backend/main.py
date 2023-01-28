from fastapi import FastAPI, HTTPException
from keras.models import load_model
import numpy as np

app = FastAPI()

model = load_model("backend/model.h5")


@app.get("/predict")
async def predict(age: float, 
                  sex: int, 
                  chest_pain_type: int, 
                  resting_bp: float, 
                  cholesterol: float, 
                  fasting_bs: float, 
                  resting_ecg: int, 
                  max_hr: float, 
                  exercise_angina: int, 
                  oldpeak: float, 
                  st_slope: int):
    input_vector = np.array([age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]).reshape(1, 11)

    prediction = model.predict(input_vector)
    class_idx = np.argmax(prediction[0])
    class_probability = prediction[0][class_idx]

    if class_probability:
        return {"prediction": class_idx, "probability": class_probability, "data":input_vector}
    else:
        return {"prediction": "No prediction"}

