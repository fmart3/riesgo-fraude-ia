import joblib
import pandas as pd
import logging
import os
import sys
from enum import Enum

# Imports necesarios para deserializar el pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# BLOQUE DE AJUSTE DE RUTAS (AGREGAR ESTO AL INICIO)
# ==============================================================================
# 1. Obtener la ruta absoluta de la carpeta donde está este script (misc)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Obtener la ruta raíz del proyecto (un nivel arriba de misc)
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

# 3. Agregar la raíz al 'sys.path' para poder importar 'utils'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. CAMBIAR EL DIRECTORIO DE TRABAJO A LA RAÍZ
# Esto es vital: hace que cuando los otros scripts busquen "questions.json" 
# o ".env", los encuentren en la raíz y no busquen en 'misc'.
os.chdir(project_root)
# ==============================================================================

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
