from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging

# --- IMPORTAMOS NUESTROS MÓDULOS ---
import schemas          # Formulario y validación de datos
import logic            # Feature Engineering (Risk Score)
import inference        # Carga del modelo y predicción
import explainability   # SHAP y explicaciones LLM

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
    Flujo PURO de IA:
    1. Recibe datos.
    2. Calcula variables derivadas (Risk Score) sin aplicar vetos.
    3. Predice usando SOLO el modelo.
    4. Genera explicación con IA Generativa.
    """
    try:
        # A. PROCESAMIENTO DE DATOS (Feature Engineering)
        # Llamamos a logic solo para preparar los datos (Risk Score), 
        # ignoramos las advertencias (_) para no imponer reglas duras.
        processed_data, _ = logic.process_business_rules(form_data)
        
        # B. INFERENCIA (Resultado Puro del Modelo)
        # La probabilidad viene 100% del cerebro matemático del modelo
        probability, is_fraud = inference.predict(processed_data)
        
        # C. EXPLICABILIDAD VISUAL (SHAP)
        shap_image, shap_text = explainability.generate_explanation(processed_data)
        
        # D. EXPLICABILIDAD TEXTUAL (LLM - Gemini)
        # Generamos el análisis narrativo para el usuario
        top_factors_text = f"Factores técnicos detectados: {shap_text}. Monto: {form_data.amount}, Hora: {form_data.hour}."

        try:
            explicacion_texto = explainability.generar_explicacion_llm(
                form_data.dict(), 
                [], 
                top_factors_text
            )
        except Exception as e:
            explicacion_texto = "Análisis IA no disponible temporalmente."
            logger.error(f"Error LLM: {e}")

        # E. CONSTRUIR RESPUESTA
        # alert_messages se deja vacío para no mostrar reglas rígidas ("Regla X violada")
        # Solo devolvemos la probabilidad pura y la explicación de la IA.
        return {
            "probability_percent": probability * 100,
            "is_fraud": is_fraud,
            "risk_score_input": int(processed_data['risk_score'].iloc[0]),
            "alert_messages": [], # Sin reglas estrictas
            "shap_image_base64": shap_image,
            "ai_explanation": explicacion_texto # Este es el mensaje que se mantiene debajo
        }

    except ValueError as ve:
        logger.warning(f"Error de validación de datos: {ve}") 
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error crítico en análisis: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

# 5. Endpoint para mostrar el Frontend (HTML)
@app.get("/")
def read_root():
    return FileResponse('index.html')

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)