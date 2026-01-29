import joblib
import pandas as pd
import logging
import os

# Configuración de logs
logger = logging.getLogger(__name__)

# VARIABLES GLOBALES
# Guardamos el modelo aquí para que persista en memoria RAM
_MODELO = None
_PREPROCESADOR = None
MODEL_PATH = 'modelo_fraude_final.pkl'

def load_model_assets():
    """
    Carga el modelo y el preprocesador desde el archivo .pkl.
    Se llama una sola vez al iniciar App.py.
    """
    global _MODELO, _PREPROCESADOR
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No se encontró el archivo del modelo en: {MODEL_PATH}")

    try:
        logger.info("⏳ Cargando modelo de inteligencia artificial...")
        data = joblib.load(MODEL_PATH)
        
        # Extraemos las partes del diccionario guardado
        _MODELO = data['modelo']
        _PREPROCESADOR = data['preprocesador']
        
        logger.info("✅ Modelo cargado exitosamente en memoria.")
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar el pickle: {e}")
        raise e

def get_model_assets():
    """Devuelve el modelo y preprocesador para que otros módulos (como SHAP) los usen."""
    if _MODELO is None or _PREPROCESADOR is None:
        load_model_assets()
    return _MODELO, _PREPROCESADOR

def predict(input_df: pd.DataFrame):
    """
    Recibe un DataFrame pre-procesado (con risk_score calculado)
    y devuelve la probabilidad (float) y la clasificación (bool).
    """
    modelo, preprocesador = get_model_assets()
    
    try:
        # 1. Transformar datos (OneHotEncoding, Scaling, etc.)
        # El preprocesador espera un DataFrame y devuelve un array numpy
        input_processed = preprocesador.transform(input_df)
        
        # 2. Predecir Probabilidad (Clase 1 = Fraude)
        # predict_proba devuelve [[prob_no_fraude, prob_fraude]]
        probability = modelo.predict_proba(input_processed)[0, 1]
        
        # 3. Definir umbral (Podríamos hacerlo configurable, por ahora 0.5)
        threshold = 0.5
        is_fraud = probability >= threshold
        
        return probability, is_fraud

    except Exception as e:
        logger.error(f"Error durante la inferencia: {e}")
        raise ValueError("Error al ejecutar el modelo matemático.")