from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging

# --- IMPORTAMOS NUESTROS MÓDULOS ---
import schemas          # Formulario y validación de datos
import logic            # Lógica de negocio y reglas duras
import inference        # Carga del modelo y predicción
import explainability   # SHAP y explicaciones (incluye la función del LLM)

# 1. Configuración Inicial
app = FastAPI(title="FraudGuard AI API", version="2.0")

# Configurar Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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
    4. Genera explicación SHAP y Texto LLM (explainability).
    """
    try:
        # A. LÓGICA DE NEGOCIO (Manejo)
        # Calcula el risk_score interno y valida reglas duras
        processed_data, business_warnings = logic.process_business_rules(form_data)
        
        # B. INFERENCIA (Resultado Modelo)
        probability, is_fraud = inference.predict(processed_data)
        
        # C. EXPLICABILIDAD (SHAP Visual)
        # Obtenemos la imagen y un resumen de texto básico de SHAP
        shap_image, shap_text = explainability.generate_explanation(processed_data)
        
        # D. EXPLICABILIDAD GENERATIVA (LLM - Gemini)
        # Preparamos un texto resumen para que la IA entienda el contexto
        # CORRECCIÓN: Usamos form_data (no input_data)
        top_factors_text = f"Factores técnicos detectados: {shap_text}. Monto: {form_data.amount}, Hora: {form_data.hour}."

        # CORRECCIÓN: Usamos form_data.dict() y pasamos None en shap_values si no los tenemos crudos, 
        # o usamos el shap_text como sustituto.
        try:
            explicacion_texto = explainability.generar_explicacion_llm(
                form_data.dict(), 
                [], # Pasamos lista vacía si no tenemos los valores crudos a mano, el prompt usará el texto
                top_factors_text
            )
        except Exception as e:
            explicacion_texto = "No se pudo generar la explicación por IA."
            logger.error(f"Error LLM: {e}")

        # E. CONSTRUIR RESPUESTA
        # Combinamos las alertas de negocio
        final_messages = business_warnings
        
        return {
            "probability_percent": probability * 100,
            "is_fraud": is_fraud,
            "risk_score_input": int(processed_data['risk_score'].iloc[0]), # Convertimos a int
            "alert_messages": final_messages,
            "shap_image_base64": shap_image,  # <--- AQUÍ FALTABA LA COMA
            "ai_explanation": explicacion_texto
        }

    except ValueError as ve:
        # Errores de validación de negocio
        logger.warning(f"Validación de negocio falló: {ve}") 
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

# 5. Endpoint para mostrar el Frontend (HTML)
@app.get("/")
def read_root():
    return FileResponse('index.html')

# Ejecución local
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)