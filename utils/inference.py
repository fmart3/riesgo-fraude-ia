import joblib
import pandas as pd
import numpy as np # <--- Necesario para operaciones numéricas
import logging
import os

# --- IMPORTACIONES VITALES PARA DESERIALIZAR EL PIPELINE ---
# Aunque no las uses explícitamente, joblib las necesita para reconstruir el modelo .pkl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

# --- CARGA DEL MODELO ---
_MODELO_PIPELINE = None
MODEL_PATH = 'fraude.pkl'

def load_model_assets():
    global _MODELO_PIPELINE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No existe: {MODEL_PATH}. Asegúrate de subirlo a Render.")
    
    # Al cargar, joblib necesita ver las librerías de sklearn importadas arriba
    try:
        _MODELO_PIPELINE = joblib.load(MODEL_PATH)
        logger.info("✅ Pipeline Híbrido cargado exitosamente.")
    except Exception as e:
        logger.error(f"❌ Error cargando el pickle: {e}")
        raise e

def predict(input_data: dict):
    """
    Recibe un diccionario con los datos, lo convierte al formato exacto
    que espera el modelo y devuelve (probabilidad, es_fraude).
    """
    global _MODELO_PIPELINE
    if _MODELO_PIPELINE is None:
        load_model_assets()

    try:
        # 1. Definir el orden EXACTO de columnas que usaste en el Notebook
        # Notebook: cols_usuario = ['amount', 'hour', 'account_age', 'transaction_type', 'customer_segment']
        column_order = ['amount', 'hour', 'account_age', 'transaction_type', 'customer_segment']

        # 2. Construir el DataFrame
        # Aseguramos que los tipos de datos sean correctos
        df_input = pd.DataFrame([{
            'amount': float(input_data['amount']),
            'hour': int(input_data['hour']),
            'account_age': float(input_data['account_age']),
            'transaction_type': str(input_data['transaction_type']), 
            'customer_segment': str(input_data['customer_segment'])
        }])

        # 3. Reordenar columnas para evitar errores silenciosos
        df_for_model = df_input[column_order]

        # 4. PREDICCIÓN
        # El pipeline hace todo: IsolationForest -> Nuevas Columnas -> Scaling -> Predicción
        # predict_proba devuelve [[prob_0, prob_1]]
        prob_ia = _MODELO_PIPELINE.predict_proba(df_for_model)[0, 1]
        
        # 5. Decisión
        # Usamos 0.50 como corte estándar
        is_fraud = prob_ia > 0.50
        
        return prob_ia, bool(is_fraud)

    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        # En caso de pánico, devolvemos valores seguros
        return 0.0, False