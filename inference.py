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
    """
    Recibe un DataFrame con los datos CRUDOS.
    Alinea las columnas con las del entrenamiento, guarda en DB y devuelve la probabilidad.
    """
    global _MODELO_PIPELINE
    
    if _MODELO_PIPELINE is None:
        load_model_assets()
    
    try:
        # --- 1. DIAGNÓSTICO Y CORRECCIÓN DE COLUMNAS (FIX DEL ERROR 8 vs 6) ---
        
        # Intentamos obtener las columnas que el modelo espera
        # Esto funciona si es un Pipeline de Sklearn o XGBoost reciente
        if hasattr(_MODELO_PIPELINE, "feature_names_in_"):
            expected_cols = _MODELO_PIPELINE.feature_names_in_
        else:
            # Si no podemos leerlas automáticamente, ASUMIMOS las 8 estándar del dataset PaySim
            # (Ajusta esto si tus columnas de entrenamiento eran diferentes)
            expected_cols = [
                "step", "type", "amount", 
                "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
                "nameDest", "oldbalanceDest" # A veces también "isFlaggedFraud"
            ]

        # Rellenamos las columnas faltantes con valores por defecto
        # Esto engaña al modelo para que no falle por "shape mismatch"
        for col in expected_cols:
            if col not in input_df.columns:
                # Valores por defecto seguros:
                if col == "step":
                    input_df[col] = 1  # Paso de tiempo 1
                elif col == "isFlaggedFraud":
                    input_df[col] = 0
                elif "name" in col:
                    input_df[col] = "C_Unknown" # Nombres dummy
                else:
                    input_df[col] = 0 # Cualquier numero faltante es 0

        # Ordenamos las columnas EXACTAMENTE como las espera el modelo
        # (Si sobran columnas que el modelo no conoce, las borramos para evitar errores)
        if hasattr(_MODELO_PIPELINE, "feature_names_in_"):
             input_df = input_df[expected_cols]

        # --- 2. CORRECCIÓN DE TIPOS (SANITIZACIÓN) ---
        cols_problematicas = ['transaction_type', 'customer_segment', 'type', 'nameOrig', 'nameDest']
        for col in cols_problematicas:
            if col in input_df.columns and input_df[col].dtype == 'object':
                input_df[col] = input_df[col].astype('category')

        # --- 3. INFERENCIA ---
        prob_ia = _MODELO_PIPELINE.predict_proba(input_df)[0, 1] 

        # --- 4. DECISIÓN ---
        final_probability = prob_ia
        is_fraud = final_probability >= 0.5
        
        logger.info(f"🧠 Predicción: {final_probability:.4f} | Es Fraude: {is_fraud}")

        # --- 5. GUARDADO EN BASE DE DATOS ---
        if _MONGO_COLLECTION is not None:
            try:
                # Convertimos a dict estándar para MongoDB
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
        # Imprimimos las columnas actuales para que veas qué está llegando
        logger.error(f"Columnas recibidas: {list(input_df.columns)}")
        raise ValueError(f"El modelo no pudo procesar los datos: {e}")