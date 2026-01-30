import joblib
import pandas as pd
import logging
import os

# Configuración de logs
logger = logging.getLogger(__name__)

# VARIABLES GLOBALES
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
    """Devuelve el modelo y preprocesador para que otros módulos los usen."""
    if _MODELO is None or _PREPROCESADOR is None:
        load_model_assets()
    return _MODELO, _PREPROCESADOR

def predict(input_df: pd.DataFrame):
    """
    Recibe un DataFrame pre-procesado (con risk_score calculado)
    y devuelve la probabilidad FINAL ajustada y la clasificación.
    """
    modelo, preprocesador = get_model_assets()
    
    try:
        # --- PASO 1: Obtener la "Opinión" de la Inteligencia Artificial ---
        # Transformamos datos y predecimos
        input_processed = preprocesador.transform(input_df)
        prob_ia = modelo.predict_proba(input_processed)[0, 1] # Valor entre 0.0 y 1.0

        # --- PASO 2: Obtener la "Opinión" de la Lógica de Negocio (Logic.py) ---
        # input_df ya trae el 'risk_score' calculado en logic.py (viene de 0 a 100)
        risk_score_reglas = input_df['risk_score'].iloc[0]
        prob_reglas = risk_score_reglas / 100.0  # Normalizamos a 0.0 - 1.0

        # --- PASO 3: FUSIÓN HÍBRIDA (La Magia) 🪄 ---
        
        final_probability = 0.0

        # CASO A: Override de Seguridad (Muerte Súbita)
        # Si las reglas dicen que es MUY peligroso (> 85), ignoramos a la IA y hacemos caso a la regla.
        if risk_score_reglas >= 85:
            logger.warning("🚨 ALERTA: Regla de alto riesgo detectada. Ignorando modelo ML.")
            final_probability = prob_reglas
        
        # CASO B: Consenso Ponderado
        # En casos normales, mezclamos ambas opiniones.
        # Le damos más peso a tus reglas (60%) que al modelo (40%) para que el demo se sienta controlado.
        else:
            peso_reglas = 0.6
            peso_ia = 0.4
            final_probability = (prob_reglas * peso_reglas) + (prob_ia * peso_ia)

        # --- PASO 4: Resultado Final ---
        # Aseguramos que no se salga de 0-1
        final_probability = min(max(final_probability, 0.0), 1.0)
        
        # Umbral de decisión
        threshold = 0.5
        is_fraud = final_probability >= threshold
        
        logger.info(f"📊 IA: {prob_ia:.2f} | Reglas: {prob_reglas:.2f} -> Final: {final_probability:.2f}")

        return final_probability, is_fraud

    except Exception as e:
        logger.error(f"Error durante la inferencia: {e}")
        # En caso de error fatal del modelo, fallback a la regla pura
        try:
            fallback_score = input_df['risk_score'].iloc[0] / 100.0
            return fallback_score, fallback_score > 0.5
        except:
            raise ValueError("Error al ejecutar el modelo matemático y el fallback.")