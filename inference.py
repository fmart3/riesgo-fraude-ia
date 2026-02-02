import joblib
import pandas as pd
import logging
import os
import certifi
from pymongo import MongoClient
from datetime import datetime

# Configuración de logs
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
        logger.warning("⚠️ Variable MONGO_URI no encontrada. No se guardarán datos.")
except Exception as e:
    logger.error(f"⚠️ Error al conectar con MongoDB: {e}")
    _MONGO_COLLECTION = None


# VARIABLES GLOBALES DEL MODELO
_MODELO_PIPELINE = None
MODEL_PATH = 'modelo_fraude.pkl'

def load_model_assets():
    global _MODELO_PIPELINE
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No se encontró el archivo del modelo en: {MODEL_PATH}")

    try:
        logger.info(f"⏳ Cargando Pipeline de IA desde {MODEL_PATH}...")
        loaded_object = joblib.load(MODEL_PATH)
        
        # COMPATIBILIDAD
        if isinstance(loaded_object, dict) and 'modelo' in loaded_object:
            logger.warning("⚠️ Detectado formato antiguo.")
            _MODELO_PIPELINE = loaded_object['modelo']
        else:
            _MODELO_PIPELINE = loaded_object
        
        logger.info("✅ Pipeline de IA cargado exitosamente.")
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar el pickle: {e}")
        raise e

def predict(input_df: pd.DataFrame):
    global _MODELO_PIPELINE
    
    if _MODELO_PIPELINE is None:
        load_model_assets()
    
    try:
        # --- 1. DEFINIR LAS 8 COLUMNAS EXACTAS ---
        expected_cols = [
            "step", "type", "amount", 
            "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
            "nameDest", "oldbalanceDest"
        ]
        
        # --- 2. PREPARAR DATOS (REPARACIÓN DE COLUMNAS) ---
        df_for_model = input_df.copy()

        # Rellenamos faltantes
        for col in expected_cols:
            if col not in df_for_model.columns:
                if col == "step": 
                    df_for_model[col] = 1 # Step siempre 1 (inicio)
                elif col == "type":
                    # Si falta el tipo, asumimos TRANSFER por seguridad (o el más común)
                    df_for_model[col] = "TRANSFER" 
                else:
                    # ¡EL CAMBIO CLAVE!: 
                    # Usamos 0 para nameOrig/nameDest también. 
                    # Esto evita el error "could not convert string to float".
                    df_for_model[col] = 0 

        # Filtramos para tener SOLO las 8 columnas ordenadas
        df_for_model = df_for_model[expected_cols]

        # --- 3. SANITIZACIÓN DE TIPOS ---
        # Solo convertimos a category la columna 'type'. 
        # Los 'names' ahora son 0 (números), así que no los tocamos para evitar conflictos.
        if 'type' in df_for_model.columns:
            df_for_model['type'] = df_for_model['type'].astype('category')

        # --- 4. INFERENCIA ---
        prob_ia = _MODELO_PIPELINE.predict_proba(df_for_model)[0, 1] 
        final_probability = prob_ia
        is_fraud = final_probability >= 0.5
        
        logger.info(f"🧠 Predicción: {final_probability:.4f} | Es Fraude: {is_fraud}")

        # --- 5. GUARDADO EN MONGODB (Usamos el DataFrame original rico en datos) ---
        if _MONGO_COLLECTION is not None:
            try:
                record = input_df.astype(object).to_dict(orient='records')[0]
                record["prediction_is_fraud"] = bool(is_fraud)
                record["prediction_prob"] = float(final_probability)
                record["timestamp"] = datetime.utcnow()
                _MONGO_COLLECTION.insert_one(record)
                logger.info("💾 Guardado en DB.")
            except Exception as db_err:
                logger.error(f"❌ Error DB: {db_err}")

        return final_probability, is_fraud

    except Exception as e:
        logger.error(f"Error CRÍTICO: {e}")
        # Logueamos los tipos de datos para debuggear si falla de nuevo
        if 'df_for_model' in locals():
            logger.error(f"Tipos de datos enviados: \n{df_for_model.dtypes}")
        raise ValueError(f"Error modelo: {e}")