from fastapi import FastAPI, Body
from typing import List, Tuple, Dict
import pickle
import numpy as np
from pydantic import BaseModel
from xgboost import XGBClassifier

app = FastAPI()


class NumericalFeatures(BaseModel):
    Sex: str
    ChestPainType: str
    RestingECG: str
    ExerciseAngina: str
    ST_Slope: str


class CategoricalFeatures(BaseModel):
    Age:int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int


class Features(BaseModel):
    categorical_features: CategoricalFeatures
    numerical_features: NumericalFeatures


class Model:
    
    scaler_file = 'models/scaler.pkl'
    encoder_file = 'models/encoder.pkl'
    model_file = 'models/model.json'

    @classmethod
    def load_models(cls) -> Tuple:
        """
        Load the scaler, encoder, and XGBoost model from disk.

        Returns:
        - A tuple containing the scaler, encoder, and XGBoost model objects.
        """
        with open(cls.scaler_file, 'rb') as f:
            scaler = pickle.load(f)
       
        with open(cls.encoder_file, 'rb') as f:
            encoder = pickle.load(f)
                
        model = XGBClassifier()
        model.load_model(cls.model_file)
        
        return scaler, encoder, model
    

@app.post("/predict")
async def predict(features: Features = Body(...)):
    """
    Makes predictions using a saved machine learning model.

    Parameters:
    - features (Features): The data to make predictions on.
    Returns:
    - A dictionary containing the predicted class and probability for each sample in the input data.
    """
    models_predict = Model()
    scaler, encoder, model =  models_predict.load_models()
    
    scaled_data = scaler.transform(np.array([list(features.categorical_features.dict().values())]))
    encoded_data = encoder.transform(np.array([list(features.numerical_features.dict().values())])).toarray()
    preprocessed_data = np.concatenate([scaled_data, encoded_data], axis=1)
    probabilities = model.predict_proba(preprocessed_data)
    predictions = model.predict(preprocessed_data)

    return {"predictions": predictions.tolist(), "probabilities": probabilities[:, 1].tolist()}
