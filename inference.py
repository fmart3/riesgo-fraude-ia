import joblib
import pandas as pd
import logging
import os
import certifi
from pymongo import MongoClient
from datetime import datetime

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN MONGODB ---
_MONGO_COLLECTION = None

try:
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME", "FraudGuard_DB")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "predicciones")

    if MONGO_URI:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        _MONGO_COLLECTION = db[COLLECTION_NAME]
        logger.info("✅ Conexión a MongoDB exitosa.")
    else:
        logger.warning("⚠️ MONGO_URI no definido. Sin persistencia.")
except Exception as e:
    logger.error(f"⚠️ Error MongoDB (No fatal): {e}")
    _MONGO_COLLECTION = None

# --- CARGA DEL MODELO ---
_MODELO_PIPELINE = None
MODEL_PATH = 'modelo_fraude.pkl'

def load_model_assets():
    global _MODELO_PIPELINE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No existe: {MODEL_PATH}")
    try:
        loaded = joblib.load(MODEL_PATH)
        # Soporte para formatos dict o pipeline directo
        _MODELO_PIPELINE = loaded['modelo'] if (isinstance(loaded, dict) and 'modelo' in loaded) else loaded
        logger.info("✅ Pipeline cargado.")
    except Exception as e:
        logger.error(f"❌ Error carga modelo: {e}")
        raise e

def predict(input_df: pd.DataFrame):
    """
    Recibe datos con formato REAL (Online Purchase, location, etc.)
    Los traduce al formato del MODELO (type numérico, balances, etc.)
    """
    global _MODELO_PIPELINE
    if _MODELO_PIPELINE is None: load_model_assets()
    
    try:
        # 1. PREPARACIÓN DEL DATAFRAME PARA EL MODELO
        # Creamos un DF nuevo solo para la predicción
        df_for_model = pd.DataFrame()

        # --- A. TRADUCCIÓN DE TUS TIPOS REALES AL IDIOMA DEL MODELO ---
        # El modelo .pkl actual solo entiende números (1,2,3...). 
        # Aquí conectamos tus nombres reales con lo que el modelo espera.
        
        mapping_tipos = {
            'Online Purchase': 1,    # Tratado como PAYMENT
            'POS Purchase': 1,       # Tratado como PAYMENT
            'Bank Transfer': 2,      # Tratado como TRANSFER (Riesgo Medio)
            'ATM Withdrawal': 3,     # Tratado como CASH_OUT (Riesgo Alto)
            'Cash Advance': 5        # Tratado como CASH_IN
        }

        # Buscamos la columna, sea 'transaction_type' o 'type'
        col_tipo = 'transaction_type' if 'transaction_type' in input_df.columns else 'type'
        
        if col_tipo in input_df.columns:
            # Mapeamos. Si llega algo nuevo (ej: "Crypto"), le ponemos 1 por defecto.
            df_for_model['type'] = input_df[col_tipo].map(mapping_tipos).fillna(1).astype(int)
        else:
            df_for_model['type'] = 1 # Valor por defecto

        # --- B. LLENADO DE LAS OTRAS 7 COLUMNAS OBLIGATORIAS ---
        # El modelo exige 8 columnas exactas.
        
        # 'amount': Si no viene, es 0
        df_for_model['amount'] = input_df['amount'] if 'amount' in input_df.columns else 0.0
        
        # 'step': Simulamos hora 1
        df_for_model['step'] = 1 
        
        # Balances y Nombres: Ponemos 0 para que el modelo numérico no explote
        cols_relleno = ['nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest']
        for col in cols_relleno:
            if col in input_df.columns:
                # Si tienes los datos numéricos, úsalos. Si no, 0.
                df_for_model[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            else:
                df_for_model[col] = 0 

        # --- C. ORDENAR COLUMNAS (CRÍTICO) ---
        # El orden EXACTO de las 8 columnas
        expected_cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest"]
        df_for_model = df_for_model[expected_cols]

        # 2. INFERENCIA
        # Ahora df_for_model tiene solo números. No fallará.
        prob_ia = _MODELO_PIPELINE.predict_proba(df_for_model)[0, 1] 
        final_probability = prob_ia
        is_fraud = final_probability >= 0.5
        
        # Guardamos el tipo original para el log
        tipo_original = input_df.iloc[0].get(col_tipo, 'Unknown')
        logger.info(f"🧠 Predicción: {final_probability:.4f} | Tipo: {tipo_original} -> {df_for_model.iloc[0]['type']}")

        # 3. GUARDADO EN DB (Guardamos TODO el input rico original)
        if _MONGO_COLLECTION is not None:
            try:
                record = input_df.astype(object).to_dict(orient='records')[0]
                record["prediction_prob"] = float(final_probability)
                record["prediction_is_fraud"] = bool(is_fraud)
                record["timestamp"] = datetime.utcnow()
                
                _MONGO_COLLECTION.insert_one(record)
                logger.info("💾 Guardado en MongoDB.")
            except Exception as e:
                logger.error(f"Error DB: {e}")

        return final_probability, is_fraud

    except Exception as e:
        logger.error(f"Error CRÍTICO: {e}")
        if 'df_for_model' in locals():
            logger.error(f"Datos transformados: \n{df_for_model.head()}")
        raise ValueError(f"Error procesando datos: {e}")