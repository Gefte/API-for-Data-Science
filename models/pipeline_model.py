import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import joblib
import logging

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
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
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
    joblib.dump(model, model_file)
    logger.info(f"Saving scaler to file {scaler_file}")
    joblib.dump(preprocessor.named_transformers_['num'], scaler_file)
    logger.info(f"Saving encoder to file {encoder_file}")
    joblib.dump(preprocessor.named_transformers_['cat'], encoder_file)


def load_models(scaler_file: str, encoder_file: str, model_file: str):
    """
    Load the scaler, encoder, and XGBoost model from disk.

    Parameters:
    - scaler_file (str): The path to the file containing the scaler object.
    - encoder_file (str): The path to the file containing the encoder object.
    - model_file (str): The path to the file containing the XGBoost model.

    Returns:
    - A tuple containing the scaler, encoder, and XGBoost model objects.
    """
    scaler = joblib.load(scaler_file)
    encoder = joblib.load(encoder_file)
    model = XGBClassifier()
    model.load_model(model_file)

    return scaler, encoder, model

def predict(data: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder, model: XGBClassifier) -> List[float]:
    """
    Makes predictions using a saved machine learning model.

    Parameters:
    - data (pd.DataFrame): The data to make predictions on.
    - scaler (StandardScaler): The trained scaler used to scale the data.
    - encoder (OneHotEncoder): The trained encoder used to encode categorical features.
    - model (XGBClassifier): The trained machine learning model.

    Returns:
    - A list with the predicted probabilities for each sample in the input data.
    """
    
    scaled_data = scaler.transform(data[numerical_features])

 
    encoded_data = encoder.transform(data[categorical_features]).toarray()

    
    preprocessed_data = np.concatenate([scaled_data, encoded_data], axis=1)

    predictions = model.predict_proba(preprocessed_data)[:, 1]

    return predictions


def main(train_data_path: str, saved_model_path: str, input_data_path: str = None, output_data_path: str = None):
    """
    Train, save, load and use a machine learning model to make predictions.

    Parameters:
    - train_data_path (str): The path to the CSV file containing the training data.
    - saved_model_path (str): The path to save or load the trained model.
    - input_data_path (str): The path to the CSV file containing the input data to make predictions on.
    - output_data_path (str): The path to save the predicted data.

    Returns:
    - None.
    """

    train_data = pd.read_csv(train_data_path)

    
    X = train_data.drop(columns=['target'])
    y = train_data['target']

 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    encoder = OneHotEncoder()

   
    scaled_train_data = scaler.fit_transform(X_train[numerical_features])
    encoded_train_data = encoder.fit_transform(X_train[categorical_features]).toarray()

   
    preprocessed_train_data = np.concatenate([scaled_train_data, encoded_train_data], axis=1)

 
    model = XGBClassifier()
    model.fit(preprocessed_train_data, y_train)


    scaled_val_data = scaler.transform(X_val[numerical_features])
    encoded_val_data = encoder.transform(X_val[categorical_features]).toarray()
    preprocessed_val_data = np.concatenate([scaled_val_data, encoded_val_data], axis=1)
    val_preds = model.predict(preprocessed_val_data)
    print(f"Validation accuracy: {accuracy_score(y_val, val_preds)}")


    with open(saved_model_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'encoder': encoder, 'model': model}, f)


    if input_data_path:
        input_data



if __name__ == "__main__":
    main()