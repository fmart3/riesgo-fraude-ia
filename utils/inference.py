import joblib
import pandas as pd
import numpy as np # <--- Necesario para operaciones numéricas
import logging
import os
from enum import Enum

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

        # 2. EXTRACCIÓN SEGURA DE STRINGS (Fix del Enum)
        # Si viene de Pydantic, es un Enum. Usamos .value para sacar el texto real.
        t_type = input_data['transaction_type']
        if isinstance(t_type, Enum):
            t_type = t_type.value
        else:
            t_type = str(t_type)

        c_segment = input_data['customer_segment']
        if isinstance(c_segment, Enum):
            c_segment = c_segment.value
        else:
            c_segment = str(c_segment)

        # 3. Construir DataFrame
        df_input = pd.DataFrame([{
            'amount': float(input_data['amount']),
            'hour': int(input_data['hour']),
            'account_age': float(input_data['account_age']),
            'transaction_type': t_type,      # <--- Texto limpio ("ATM Withdrawal")
            'customer_segment': c_segment    # <--- Texto limpio ("Retail")
        }])

        # 4. DEBUG: Ver qué está entrando realmente (Mira esto en los logs de Render)
        print(f"🔍 DEBUG INFERENCIA: Type='{t_type}', Segment='{c_segment}', Amount={df_input['amount'].iloc[0]}")

        # 5. APLICAR LOGARITMO (Vital)
        # El modelo fue entrenado con log, si no lo pones, el monto domina todo.
        df_input['amount'] = np.log1p(df_input['amount'])

        # 6. Reordenar y Predecir
        df_for_model = df_input[column_order]
        prob_ia = _MODELO_PIPELINE.predict_proba(df_for_model)[0, 1]
        
        # Umbral optimizado
        is_fraud = prob_ia >= 0.5071 

        return prob_ia, bool(is_fraud)
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        # En caso de pánico, devolvemos valores seguros
        return 0.0, False