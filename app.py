from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse # <--- IMPORTAR ESTO
import uvicorn
import logging

# IMPORTACIONES
import schemas
import inference
# import logic  <-- ELIMINADO PARA QUE NO INTERFIERA

# Configuración
app = FastAPI(title="FraudGuard AI - Pure Model", version="3.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        inference.load_model_assets()
        logger.info("✅ Modelo cargado.")
    except Exception as e:
        logger.error(f"❌ Error carga modelo: {e}")
        
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "FraudGuard AI está activo.",
        "usage": "Envía un POST a /analyze con los datos de la transacción."
    }

@app.post("/analyze", response_model=schemas.PredictionResponse)
def analyze_transaction(form_data: schemas.TransactionRequest):
    """
    Endpoint directo: Request -> Modelo -> Response.
    Sin reglas de negocio intermedias.
    """
    try:
        # 1. Convertir Pydantic a Dict puro
        data_dict = form_data.dict()
        
        # 2. INFERENCIA PURA (El modelo decide 100%)
        probability, is_fraud = inference.predict(data_dict)
        
        # 3. Construir respuesta
        return {
            "probability_percent": probability * 100,
            "is_fraud": is_fraud,
            # Ya no hay risk_score calculado manualmente, enviamos 0 o null
            "risk_score_input": 0, 
            "alert_messages": [],
            "shap_image_base64": None,
            "ai_explanation": "Predicción basada 100% en el modelo matemático."
        }

    except Exception as e:
        logger.error(f"Error endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)