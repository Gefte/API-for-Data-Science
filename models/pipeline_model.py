import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import logging
from typing import List, Tuple, Dict
import pickle


logger = logging.getLogger(__name__)


def load_data(data_file: str) -> pd.DataFrame:
    """
    Loads the data from a CSV file.

    Parameters:
    - data_file (str): The path to the data file.

    Returns:
    - A pandas DataFrame with the data.
    """
    logger.info(f"Loading data from file {data_file}")
    return pd.read_csv(data_file)


def preprocess_data(X: pd.DataFrame) -> (pd.DataFrame, ColumnTransformer):
    """
    Preprocesses the data by scaling numeric features and one-hot-encoding categorical features.

    Parameters:
    - X (pd.DataFrame): The input data.

    Returns:
    - A tuple containing the preprocessed data and the ColumnTransformer used to preprocess it.
    """
    logger.info("Preprocessing data")
    df_index = pd.DataFrame(X.dtypes).reset_index()
    
    categorical_cols = df_index.loc[df_index[0]=='object']['index'].to_list()
    numeric_cols = df_index.loc[df_index[0]=='int64']['index'].to_list()
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)])
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, preprocessor


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """
    Trains a model to predict heart disease.

    Parameters:
    - X (pd.DataFrame): The preprocessed input data.
    - y (pd.Series): The target variable.

    Returns:
    - The trained XGBClassifier model.
    """
    logger.info("Training model")
    model = XGBClassifier(learning_rate=0.01, n_estimators=100, max_depth=4)
    model.fit(X, y)
    return model


def save_models(model: XGBClassifier, preprocessor: ColumnTransformer, model_file: str,
                scaler_file: str, encoder_file: str) -> None:
    """
    Saves the trained model and its preprocessing transformers to disk.

    Parameters:
    - model (XGBClassifier): The trained model.
    - preprocessor (ColumnTransformer): The preprocessing ColumnTransformer.
    - model_file (str): The filename to use for the model file.
    - scaler_file (str): The filename to use for the scaler file.
    - encoder_file (str): The filename to use for the encoder file.
    """
    logger.info(f"Saving model to file {model_file}")
    model.save_model(model_file)
    
    logger.info(f"Saving scaler to file {scaler_file}")
    with open(scaler_file, 'wb') as f:
        pickle.dump(preprocessor.named_transformers_['num'], f)
        
    logger.info(f"Saving encoder to file {encoder_file}")
    with open(encoder_file, 'wb') as f:
        pickle.dump(preprocessor.named_transformers_['cat'], f)

def load_models(scaler_file: str, encoder_file: str, model_file: str) -> Tuple:
    """
    Load the scaler, encoder, and XGBoost model from disk.

    Parameters:
    - scaler_file (str): The path to the file containing the scaler object.
    - encoder_file (str): The path to the file containing the encoder object.
    - model_file (str): The path to the file containing the XGBoost model.

    Returns:
    - A tuple containing the scaler, encoder, and XGBoost model objects.
    """
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(encoder_file, 'rb') as f:
        encoder = pickle.load(f)
        
    model = XGBClassifier()
    model.load_model(model_file)
    
    return scaler, encoder, model

def predict(data: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder, model: XGBClassifier, numerical_features: List[str], categorical_features: List[str]) -> Dict[str, List[float]]:
    """
    Makes predictions using a saved machine learning model.

    Parameters:
    - data (pd.DataFrame): The data to make predictions on.
    - scaler (StandardScaler): The trained scaler used to scale the data.
    - encoder (OneHotEncoder): The trained encoder used to encode categorical features.
    - model (XGBClassifier): The trained machine learning model.
    - numerical_features (List[str]): The names of the numerical features in the input data.
    - categorical_features (List[str]): The names of the categorical features in the input data.

    Returns:
    - A dictionary containing the predicted class and probability for each sample in the input data.
    """

    scaled_data = scaler.transform(data[numerical_features])
    encoded_data = encoder.transform(data[categorical_features]).toarray()
    preprocessed_data = np.concatenate([scaled_data, encoded_data], axis=1)
    probabilities = model.predict_proba(preprocessed_data)
    predictions = model.predict(preprocessed_data)

    return {"predictions": predictions, "probabilities": probabilities[:, 1].tolist()}

def main():
    # Load data
    data = load_data("models/heart.csv")

    # Preprocess data
    X, preprocessor = preprocess_data(data.drop("HeartDisease", axis=1))
    y = data["HeartDisease"]

    # Train model
    model = train_model(X, y)

    # Save models
    save_models(model, preprocessor, "models/model.json", "models/scaler.pkl", "models/encoder.pkl")

    # Load models
    scaler, encoder, model = load_models("models/scaler.pkl", "models/encoder.pkl", "models/model.json")

    # Make predictions
    df_index = pd.DataFrame(data.drop("HeartDisease", axis=1).dtypes).reset_index()
    numerical_features = df_index.loc[df_index[0]=='int64']['index'].to_list()
    categorical_features = df_index.loc[df_index[0]=='object']['index'].to_list()
    new_data = pd.DataFrame(data.drop("HeartDisease", axis=1).loc[0]).T
    predictions = predict(new_data, scaler, encoder, model, numerical_features, categorical_features)
    print(predictions)

if __name__ == "__main__":
    main()