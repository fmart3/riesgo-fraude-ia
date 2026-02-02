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
        # --- 0. CORRECCIÓN DE TIPOS (EL FIX PARA TU ERROR) ---
        # XGBoost no acepta 'object' (strings). Convertimos explícitamente a 'category'.
        # Esto soluciona el error: "DataFrame.dtypes for data must be int, float, bool or category"
        cols_problematicas = ['transaction_type', 'customer_segment']
        
        for col in cols_problematicas:
            if col in input_df.columns and input_df[col].dtype == 'object':
                input_df[col] = input_df[col].astype('category')
        
        # También convertimos cualquier otra columna 'object' por seguridad
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = input_df[col].astype('category')

        # --- 1. INFERENCIA ---
        prob_ia = _MODELO_PIPELINE.predict_proba(input_df)[0, 1] 

        # --- 2. DECISIÓN ---
        final_probability = prob_ia
        is_fraud = final_probability >= 0.5
        
        logger.info(f"🧠 Predicción: {final_probability:.4f} | Es Fraude: {is_fraud}")

        # --- 3. GUARDADO EN BASE DE DATOS ---
        if _MONGO_COLLECTION is not None:
            try:
                # OJO: Para guardar en MongoDB, necesitamos los datos originales (Strings),
                # no los tipos 'category' de Pandas, porque Mongo no entiende 'category'.
                # Convertimos de nuevo a diccionario estándar de Python.
                record = input_df.astype(object).to_dict(orient='records')[0]
                
                record["prediction_is_fraud"] = bool(is_fraud)
                record["prediction_prob"] = float(final_probability)
                record["timestamp"] = datetime.utcnow()
                
                _MONGO_COLLECTION.insert_one(record)
                logger.info("💾 Predicción guardada en MongoDB.")
                
            except Exception as db_err:
                logger.error(f"❌ Error guardando en DB: {db_err}")

        return final_probability, is_fraud

    except Exception as e:
        logger.error(f"Error CRÍTICO en predicción: {e}")
        # Agregamos el mensaje técnico para que sepas qué pasó
        raise ValueError(f"El modelo no pudo procesar los datos: {e}")