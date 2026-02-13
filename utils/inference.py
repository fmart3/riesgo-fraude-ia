import joblib
import pandas as pd
import logging
import os
from enum import Enum

# Imports necesarios para deserializar el pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

_MODEL_PIPELINE = None
MODEL_PATH = "model_fraude.pkl"

def load_model_assets():
    global _MODEL_PIPELINE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No existe el modelo en {MODEL_PATH}")

    _MODEL_PIPELINE = joblib.load(MODEL_PATH)
    logger.info("✅ Pipeline de fraude cargado correctamente.")

def predict(input_data: dict):
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is None:
        load_model_assets()

    try:
        # Limpieza segura de Enums
        t_type = input_data["transaction_type"].value if isinstance(input_data["transaction_type"], Enum) else str(input_data["transaction_type"])
        c_segment = input_data["customer_segment"].value if isinstance(input_data["customer_segment"], Enum) else str(input_data["customer_segment"])

        # DataFrame EXACTO al entrenamiento
        df_input = pd.DataFrame([{
            "amount": float(input_data["amount"]),
            "hour": int(input_data["hour"]),
            "account_age": float(input_data["account_age"]),
            "transaction_type": t_type,
            "customer_segment": c_segment
        }])

        # DEBUG
        logger.info(f"🔍 Inferencia | {df_input.to_dict(orient='records')[0]}")

        # Predicción (el pipeline hace TODO)
        prob_fraude = _MODEL_PIPELINE.predict_proba(df_input)[0, 1]

        # Threshold final entrenado
        THRESHOLD = 0.488
        is_fraud = prob_fraude >= THRESHOLD

        return prob_fraude, bool(is_fraud)

    except Exception as e:
        logger.error(f"❌ Error en inferencia: {e}")
        return 0.0, False
