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
# Intentamos conectar una sola vez al inicio.
# Si falla (por internet o credenciales), la variable queda en None y la app sigue funcionando.
_MONGO_COLLECTION = None

try:
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME", "FraudGuard_DB")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "predicciones")

    if MONGO_URI:
        # 'tlsCAFile=certifi.where()' es vital para que no falle en Docker/Nube
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
    """
    Carga el Pipeline completo (Preprocesador + Modelo) desde el archivo .pkl.
    Se llama una sola vez al iniciar App.py.
    """
    global _MODELO_PIPELINE
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No se encontró el archivo del modelo en: {MODEL_PATH}")

    try:
        logger.info(f"⏳ Cargando Pipeline de IA desde {MODEL_PATH}...")
        
        # Cargamos el objeto completo.
        loaded_object = joblib.load(MODEL_PATH)
        
        # COMPATIBILIDAD:
        if isinstance(loaded_object, dict) and 'modelo' in loaded_object:
            logger.warning("⚠️ Detectado formato antiguo (diccionario). Se recomienda usar Pipeline completo.")
            _MODELO_PIPELINE = loaded_object['modelo']
        else:
            _MODELO_PIPELINE = loaded_object
        
        logger.info("✅ Pipeline de IA cargado exitosamente en memoria.")
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar el pickle: {e}")
        raise e

def predict(input_df: pd.DataFrame):
    """
    Recibe un DataFrame con los datos CRUDOS.
    Guarda en DB y devuelve la probabilidad.
    """
    global _MODELO_PIPELINE
    
    if _MODELO_PIPELINE is None:
        load_model_assets()
    
    try:
        # --- 1. INFERENCIA ---
        # El Pipeline transforma y predice
        prob_ia = _MODELO_PIPELINE.predict_proba(input_df)[0, 1] 

        # --- 2. DECISIÓN ---
        final_probability = prob_ia
        is_fraud = final_probability >= 0.5
        
        logger.info(f"🧠 Predicción: {final_probability:.4f} | Es Fraude: {is_fraud}")

        # --- 3. GUARDADO EN BASE DE DATOS (NUEVO) ---
        if _MONGO_COLLECTION is not None:
            try:
                # Convertimos el DataFrame (una fila) a diccionario simple
                # Esto guarda los datos de entrada TAL CUAL llegaron (amount, type, etc.)
                record = input_df.to_dict(orient='records')[0]
                
                # Agregamos los resultados de la IA
                record["prediction_is_fraud"] = bool(is_fraud) # Convertir a bool de Python (True/False)
                record["prediction_prob"] = float(final_probability)
                record["timestamp"] = datetime.utcnow()
                
                # Insertar
                _MONGO_COLLECTION.insert_one(record)
                logger.info("💾 Predicción guardada en MongoDB.")
                
            except Exception as db_err:
                # Si falla guardar, SOLO logueamos el error. NO detenemos la app.
                logger.error(f"❌ Error guardando en DB, pero se devuelve predicción: {db_err}")

        return final_probability, is_fraud

    except Exception as e:
        logger.error(f"Error CRÍTICO en predicción: {e}")
        raise ValueError(f"El modelo no pudo procesar los datos: {e}")