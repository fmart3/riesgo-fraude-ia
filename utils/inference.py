import os
import joblib
import pandas as pd
import logging

# ======================================================================
# LOGGING
# ======================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================
# CONFIGURACIÓN
# ======================================================================
MODEL_PATH = "model_fraude.pkl"
THRESHOLD = 0.46
_MODEL_PIPELINE = None

# ======================================================================
# CARGA DEL MODELO
# ======================================================================
def load_model_assets():
    global _MODEL_PIPELINE

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ No existe el modelo en {MODEL_PATH}")

    _MODEL_PIPELINE = joblib.load(MODEL_PATH)
    logger.info("✅ Modelo cargado correctamente (Pipeline completo).")

# ======================================================================
# INFERENCIA
# ======================================================================
def predict_fraud(input_data: dict):
    """
    input_data:
    {
        "amount": 950000,
        "hour": 3,
        "account_age": 2.0,
        "transaction_type": "Bank Transfer",
        "customer_segment": "Business"
    }
    """

    if _MODEL_PIPELINE is None:
        raise RuntimeError("Modelo no cargado. Llama a load_model_assets()")

    try:
        # 1️⃣ DataFrame crudo (SIN feature engineering manual)
        df_input = pd.DataFrame([input_data])

        # 2️⃣ Predicción directa
        prob_fraude = _MODEL_PIPELINE.predict_proba(df_input)[0, 1]

        # 3️⃣ Decisión de negocio
        is_fraud = prob_fraude >= THRESHOLD

        result = {
            "fraud_probability": round(prob_fraude * 100, 2),
            "is_fraud": bool(is_fraud),
            "threshold_used": THRESHOLD,
            "action": "BLOCK" if is_fraud else "APPROVE"
        }

        logger.info(
            f"📊 Prob={result['fraud_probability']}% | Acción={result['action']}"
        )

        return result

    except Exception as e:
        logger.error(f"❌ Error en inferencia: {e}")
        raise e
