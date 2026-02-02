import joblib
import pandas as pd
import logging
import os

# Configuración de logs
logger = logging.getLogger(__name__)

# VARIABLES GLOBALES
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
        # Asumimos que es un Pipeline de Sklearn o un objeto que tiene el método .predict()
        loaded_object = joblib.load(MODEL_PATH)
        
        # COMPATIBILIDAD:
        # Si por alguna razón es el formato antiguo (diccionario), extraemos el modelo,
        # pero idealmente debería ser un Pipeline único.
        if isinstance(loaded_object, dict) and 'modelo' in loaded_object:
            logger.warning("⚠️ Detectado formato antiguo (diccionario). Se recomienda usar Pipeline completo.")
            _MODELO_PIPELINE = loaded_object['modelo']
            # Nota: Si es el formato antiguo, podría fallar si falta el preprocesador.
        else:
            _MODELO_PIPELINE = loaded_object
        
        logger.info("✅ Pipeline de IA cargado exitosamente en memoria.")
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar el pickle: {e}")
        raise e

def predict(input_df: pd.DataFrame):
    """
    Recibe un DataFrame con los datos CRUDOS (tal cual vienen del formulario).
    Devuelve la probabilidad y la clasificación.
    """
    global _MODELO_PIPELINE
    
    if _MODELO_PIPELINE is None:
        load_model_assets()
    
    try:
        # --- INFERENCIA ---
        # El Pipeline se encarga de todo:
        # 1. Recibe 'Transfer', 'Retail' (Texto)
        # 2. Transforma internamente (OneHotEncoding, Scaling)
        # 3. Predice
        
        # Obtenemos probabilidad de la clase 1 (Fraude)
        prob_ia = _MODELO_PIPELINE.predict_proba(input_df)[0, 1] 

        # --- DECISIÓN ---
        # Probabilidad directa del modelo
        final_probability = prob_ia

        # Umbral (0.5 es el estándar, puedes ajustarlo si quieres ser más estricto)
        is_fraud = final_probability >= 0.5
        
        logger.info(f"🧠 Predicción: {final_probability:.4f} | Es Fraude: {is_fraud}")

        return final_probability, is_fraud

    except Exception as e:
        logger.error(f"Error CRÍTICO en predicción: {e}")
        raise ValueError(f"El modelo no pudo procesar los datos: {e}")