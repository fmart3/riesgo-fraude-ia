import os
import logging
import uvicorn
from datetime import datetime # <--- IMPORTANTE: Para guardar fecha y hora
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# --- NUEVO: Importar MongoDB ---
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient

# Importaciones locales
import utils.schemas as schemas
import utils.inference as inference
import utils.explainability as explainability

# Configuración de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la App
app = FastAPI(title="FraudGuard AI Dashboard", version="3.1")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CONEXIÓN A MONGODB ---
# Buscamos la URL en las variables de entorno (En Render debes configurar esta variable)
MONGO_URI = os.getenv("MONGO_URI")
db_collection = None

@app.on_event("startup")
def startup_event():
    global db_collection
    
    # A) Cargar Modelo
    try:
        inference.load_model_assets()
        logger.info("✅ Modelo cargado correctamente.")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")

    # B) Conectar a Base de Datos
    if MONGO_URI:
        try:
            client = MongoClient(MONGO_URI)
            db = client.get_database("FraudGuardDB") # Nombre de tu Base de Datos
            db_collection = db.get_collection("transacciones") # Nombre de tu Colección
            logger.info("✅ Conexión a MongoDB exitosa.")
        except Exception as e:
            logger.error(f"⚠️ Error conectando a MongoDB: {e}")
    else:
        logger.warning("⚠️ No se encontró MONGO_URI. Los datos NO se guardarán.")

# --- RUTA PRINCIPAL ---
@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Frontend no encontrado"}

# ================================
# ENDPOINT PRINCIPAL
# ================================
@app.post("/analyze", response_model=schemas.PredictionResponse)
async def analyze(data: schemas.TransactionRequest):
    try:
        # Extraer .value de los Enums para obtener strings planos
        input_dict = {
            "amount": data.amount,
            "hour": data.hour,
            "account_age": data.account_age,
            "transaction_type": data.transaction_type.value,
            "customer_segment": data.customer_segment.value
        }

        # 🔥 1. Predicción
        prediction = inference.predict(input_dict)

        # 🔥 2. SHAP
        shap_img, shap_text = explainability.generate_explanation(input_dict)

        # 🔥 3. Construir respuesta alineada al schema
        response = {
            "probability_percent": prediction["probability_percent"],
            "is_fraud": prediction["is_fraud"],
            "risk_score_input": prediction["risk_score_input"],
            "alert_messages": prediction["alert_messages"],
            "shap_image_base64": shap_img,
            "ai_explanation": shap_text,
            "risk_level": prediction.get("risk_level", "LOW"),
            "threshold_used": prediction.get("threshold_used", 0.329)
        }

        # 🔥 4. Guardar en Mongo si existe
        if db_collection is not None:
            try:
                db_collection.insert_one({
                    **input_dict,
                    **response,
                    "timestamp": datetime.utcnow()
                })
            except Exception as mongo_error:
                logger.error(f"Error guardando en MongoDB: {mongo_error}")

        return response

    except Exception as e:
        logger.error(f"Error en endpoint /analyze: {e}")

        return {
            "probability_percent": 0.0,
            "is_fraud": False,
            "risk_score_input": 0,
            "alert_messages": ["Error generando análisis."],
            "shap_image_base64": None,
            "ai_explanation": "Error interno del sistema."
        }