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
        _MODELO_PIPELINE = loaded['modelo'] if (isinstance(loaded, dict) and 'modelo' in loaded) else loaded
        logger.info("✅ Pipeline cargado.")
    except Exception as e:
        logger.error(f"❌ Error carga modelo: {e}")
        raise e

def predict(input_df: pd.DataFrame):
    """
    Adapta tus datos reales (hour, account_age, tipos string) 
    al formato rígido del modelo (step, type int, sin account_age).
    """
    global _MODELO_PIPELINE
    if _MODELO_PIPELINE is None: load_model_assets()
    
    try:
        # 1. PREPARAR DATAFRAME PARA EL MODELO
        df_for_model = pd.DataFrame()

        # --- A. MAPEO DE HORA -> STEP (INT) ---
        # Tu dato 'hour' se convierte en el 'step' del modelo
        if 'hour' in input_df.columns:
            df_for_model['step'] = input_df['hour'].astype(int)
        else:
            df_for_model['step'] = 1 # Default si falta

        # --- B. MAPEO DE AMOUNT (FLOAT) ---
        if 'amount' in input_df.columns:
            df_for_model['amount'] = input_df['amount'].astype(float)
        else:
            df_for_model['amount'] = 0.0

        # --- C. TRADUCCIÓN DE TRANSACTION_TYPE -> TYPE (INT) ---
        mapping_tipos = {
            'Online Purchase': 1,    # PAYMENT
            'POS Purchase': 1,       # PAYMENT
            'Bank Transfer': 2,      # TRANSFER
            'ATM Withdrawal': 3,     # CASH_OUT
            'Cash Advance': 5        # CASH_IN
        }

        col_tipo = 'transaction_type' if 'transaction_type' in input_df.columns else 'type'
        
        if col_tipo in input_df.columns:
            # Convertimos string a int según el mapa
            df_for_model['type'] = input_df[col_tipo].map(mapping_tipos).fillna(1).astype(int)
        else:
            df_for_model['type'] = 1

        # --- D. RELLENO DE COLUMNAS FALTANTES (Balancess y Nombres) ---
        # account_age NO se incluye aquí porque el modelo .pkl no fue entrenado con ella.
        cols_relleno = ['nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest']
        for col in cols_relleno:
            if col in input_df.columns:
                df_for_model[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            else:
                df_for_model[col] = 0 

        # --- E. ORDEN FINAL OBLIGATORIO (8 Columnas) ---
        expected_cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest"]
        df_for_model = df_for_model[expected_cols]

        # 2. INFERENCIA
        prob_ia = _MODELO_PIPELINE.predict_proba(df_for_model)[0, 1] 
        final_probability = prob_ia
        is_fraud = final_probability >= 0.5
        
        logger.info(f"🧠 Predicción: {final_probability:.4f} | Hour(Step): {df_for_model.iloc[0]['step']}")

        # 3. GUARDADO EN MONGODB (Aquí SÍ guardamos account_age y todo lo demás)
        if _MONGO_COLLECTION is not None:
            try:
                # Convertimos el DF original (que tiene account_age) a dict
                record = input_df.astype(object).to_dict(orient='records')[0]
                
                # Aseguramos tipos correctos para Mongo
                if 'account_age' in record:
                    record['account_age'] = float(record['account_age'])
                if 'hour' in record:
                    record['hour'] = int(record['hour'])
                
                # Agregamos resultados
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
        raise ValueError(f"Error procesando datos: {e}")