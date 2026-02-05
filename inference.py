import joblib
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

# --- CARGA DEL MODELO ---
_MODELO_PIPELINE = None
MODEL_PATH = 'fraude.pkl'

def load_model_assets():
    global _MODELO_PIPELINE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No existe: {MODEL_PATH}")
    
    # Cargar el pipeline completo
    _MODELO_PIPELINE = joblib.load(MODEL_PATH)
    logger.info("✅ Pipeline cargado exitosamente.")

def predict(input_data: dict):
    """
    Recibe un diccionario con los datos de la transacción
    y devuelve (probabilidad, es_fraude).
    """
    global _MODELO_PIPELINE
    if _MODELO_PIPELINE is None:
        load_model_assets()

    try:
        # 1. Construir el DataFrame EXACTAMENTE como lo pide el modelo
        # El modelo espera: ['amount', 'transaction_type', 'account_age', 'customer_segment', 'hour']
        
        # Aseguramos que los Enums se conviertan a string (p.ej. TransactionType.ATM -> "ATM Withdrawal")
        df_for_model = pd.DataFrame([{
            'amount': float(input_data['amount']),
            'transaction_type': str(input_data['transaction_type']), # Importante: string
            'account_age': float(input_data['account_age']),
            'customer_segment': str(input_data['customer_segment']), # Importante: string
            'hour': int(input_data['hour'])
        }])

        # 2. PREDICCIÓN
        # El pipeline se encarga de todo (OneHot, Escalar, etc.)
        # predict_proba devuelve [[prob_no_fraude, prob_fraude]] -> tomamos el índice 1
        prob_ia = _MODELO_PIPELINE.predict_proba(df_for_model)[0, 1]
        
        # 3. Decisión (Umbral del 50% o el que definiste en entrenamiento)
        is_fraud = prob_ia >= 0.5 
        
        return prob_ia, is_fraud

    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        # En caso de error, fallamos seguro (fail-open o fail-closed según política)
        return 0.0, False