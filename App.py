# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# --- IMPORTAMOS NUESTROS MÓDULOS (Los crearemos en los siguientes pasos) ---
import schemas          # Formulario y validación de datos
import logic            # Lógica de negocio y reglas duras
import inference        # Carga del modelo y predicción
import explainability   # SHAP y explicaciones

# 1. Configuración Inicial
app = FastAPI(title="FraudGuard AI API", version="2.0")

# Configurar Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Configuración de CORS
# Esto permite que tu HTML (Frontend) hable con este Python (Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia esto por la URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Evento de Inicio: Cargar el modelo una sola vez
@app.on_event("startup")
def startup_event():
    """Carga el modelo y el preprocesador en memoria al iniciar la app."""
    try:
        inference.load_model_assets()
        logger.info("✅ Modelo y Preprocesador cargados correctamente.")
    except Exception as e:
        logger.error(f"❌ Error fatal al cargar el modelo: {e}")
        raise e

# 4. Endpoint Principal: Análisis de Transacción
@app.post("/analyze", response_model=schemas.PredictionResponse)
def analyze_transaction(form_data: schemas.TransactionRequest):
    """
    Flujo principal:
    1. Recibe datos del formulario (validados por schemas).
    2. Aplica lógica de negocio y Risk Score (logic).
    3. Predice con el modelo IA (inference).
    4. Genera explicación SHAP (explainability).
    """
    try:
        # A. LÓGICA DE NEGOCIO (Manejo)
        # Calcula el risk_score interno y valida reglas duras (ej: ATM < 20)
        processed_data, business_warnings = logic.process_business_rules(form_data)
        
        # B. INFERENCIA (Resultado Modelo)
        # Usamos los datos procesados (que ya incluyen el risk_score calculado)
        probability, is_fraud = inference.predict(processed_data)
        
        # C. EXPLICABILIDAD (SHAP)
        # Generamos la imagen en base64 y el texto explicativo
        shap_image, shap_text = explainability.generate_explanation(processed_data)
        
        # D. CONSTRUIR RESPUESTA
        # Combinamos las alertas de negocio con las de la IA
        final_messages = business_warnings + [shap_text]
        
        return {
            "probability_percent": probability * 100,
            "is_fraud": is_fraud,
            "risk_score_input": processed_data['risk_score'].iloc[0], # Devolvemos el score calculado para mostrarlo
            "alert_messages": final_messages,
            "shap_image_base64": shap_image
        }

    except ValueError as ve:
        # Errores de validación de negocio (ej: ATM monto invalido)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error en el análisis: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

# 5. Ejecución local (opcional, para probar sin Docker)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)