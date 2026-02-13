import os
import joblib
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "model_fraude.pkl"
_MODEL_PIPELINE = None
THRESHOLD = 0.329  # ← usa el umbral óptimo que encontraste

# =========================================================
# CARGA DEL MODELO
# =========================================================
def load_model_assets():
    global _MODEL_PIPELINE

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No existe el modelo en {MODEL_PATH}")

    _MODEL_PIPELINE = joblib.load(MODEL_PATH)
    logger.info("✅ Modelo cargado correctamente.")

# =========================================================
# FEATURE ENGINEERING (idéntico al entrenamiento)
# =========================================================
def aplicar_feature_engineering_api(df):
    df = df.copy()

    df["amount_log"] = np.log1p(df["amount"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_night"] = df["hour"].apply(lambda x: 1 if (x >= 23 or x <= 5) else 0)

    bins = [-1, 2, 10, 100]
    labels = ["New", "Established", "Veteran"]
    df["tenure_group"] = pd.cut(df["account_age"], bins=bins, labels=labels)
    df["segment_tenure_profile"] = (
        df["customer_segment"] + "_" + df["tenure_group"].astype(str)
    )

    return df

# =========================================================
# FUNCIÓN COMPATIBLE CON app.py
# =========================================================
def predict(input_data: dict):
    global _MODEL_PIPELINE

    if _MODEL_PIPELINE is None:
        load_model_assets()

    try:
        df_raw = pd.DataFrame([input_data])

        df_processed = aplicar_feature_engineering_api(df_raw)

        prob = _MODEL_PIPELINE.predict_proba(df_processed)[0, 1]
        is_fraud = bool(prob >= THRESHOLD)

        logger.info(f"Probabilidad: {prob:.4f} | Fraude: {is_fraud}")

        return prob, is_fraud

    except Exception as e:
        logger.error(f"Error en inferencia: {e}")
        return 0.0, False
