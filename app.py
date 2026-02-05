import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # <--- IMPORTANTE
from fastapi.staticfiles import StaticFiles # <--- IMPORTANTE

# Importaciones locales
import schemas
import inference

# Configuración de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la App
app = FastAPI(title="FraudGuard AI Dashboard", version="3.0")

# Configuración CORS (Permite que funcione desde cualquier origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CARGA DEL MODELO AL INICIO ---
@app.on_event("startup")
def startup_event():
    try:
        inference.load_model_assets()
        logger.info("✅ Modelo cargado correctamente en memoria.")
    except Exception as e:
        logger.error(f"❌ Error crítico cargando modelo: {e}")

# --- RUTA PRINCIPAL (FRONTEND) ---
# Antes devolvía JSON, ahora devuelve tu HTML
@app.get("/")
def read_root():
    # Asegúrate de que index.html esté en la misma carpeta que app.py
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "Archivo index.html no encontrado. Súbelo al servidor."}

# --- ENDPOINT DE ANÁLISIS (BACKEND) ---
@app.post("/analyze", response_model=schemas.PredictionResponse)
def analyze_transaction(form_data: schemas.TransactionRequest):
    """
    Recibe los datos del formulario web, consulta al modelo y devuelve resultado.
    """
    try:
        # 1. Convertir Pydantic a Diccionario
        data_dict = form_data.dict()
        
        # 2. Obtener predicción del modelo (inference.py ya corregido)
        probability, is_fraud = inference.predict(data_dict)
        
        # 3. Construir respuesta para el Frontend
        # Nota: Puedes agregar lógica extra aquí si quieres mensajes personalizados
        return {
            "probability_percent": round(probability * 100, 2),
            "is_fraud": is_fraud,
            "risk_score_input": int(probability * 100), # Usamos la prob como score
            "alert_messages": ["Transacción de Alto Riesgo detectada"] if is_fraud else [],
            "shap_image_base64": None, # Si reactivas la explicabilidad, va aquí
            "ai_explanation": f"El modelo calculó una probabilidad de fraude del {probability*100:.1f}% basado en patrones históricos."
        }

    except Exception as e:
        logger.error(f"Error en endpoint /analyze: {e}")
        # En caso de error, devolvemos una respuesta segura
        return {
            "probability_percent": 0.0,
            "is_fraud": False,
            "risk_score_input": 0,
            "alert_messages": ["Error interno al procesar"],
            "ai_explanation": "Error en el servidor."
        }

if __name__ == "__main__":
    # Configuración para ejecución local
    uvicorn.run(app, host="0.0.0.0", port=5000)