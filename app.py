import os
import logging
import uvicorn
from datetime import datetime # <--- IMPORTANTE: Para guardar fecha y hora
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# --- NUEVO: Importar MongoDB ---
from pymongo import MongoClient

# Importaciones locales
import schemas
import inference

# Configuración de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la App
app = FastAPI(title="FraudGuard AI Dashboard", version="3.0")

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
db_collection = "predicciones"

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
            db_collection = db.get_collection("transactions") # Nombre de tu Colección
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

# --- RUTA DE ANÁLISIS ---
@app.post("/analyze", response_model=schemas.PredictionResponse)
def analyze_transaction(form_data: schemas.TransactionRequest):
    try:
        # 1. Convertir datos
        data_dict = form_data.dict()
        
        # 2. Predecir
        probability, is_fraud = inference.predict(data_dict)
        
        # 3. Preparar Respuesta
        response_payload = {
            "probability_percent": round(probability * 100, 2),
            "is_fraud": is_fraud,
            "risk_score_input": int(probability * 100),
            "alert_messages": ["Transacción de Alto Riesgo detectada"] if is_fraud else [],
            "shap_image_base64": None,
            "ai_explanation": f"Probabilidad calculada: {probability*100:.1f}%"
        }

        # --- 4. GUARDAR EN MONGODB (NUEVO) ---
        if db_collection is not None:
            try:
                # Creamos el documento a guardar
                record = {
                    "timestamp": datetime.utcnow(),     # Cuándo ocurrió
                    "input_data": data_dict,            # Qué datos envió el usuario
                    "prediction": {                     # Qué dijo la IA
                        "is_fraud": is_fraud,
                        "probability": probability
                    },
                    "source": "web_app"
                }
                # Insertamos
                db_collection.insert_one(record)
                logger.info("💾 Transacción guardada en MongoDB.")
            except Exception as e:
                logger.error(f"❌ Error guardando en DB: {e}")
        
        return response_payload

    except Exception as e:
        logger.error(f"Error en endpoint /analyze: {e}")
        return {
            "probability_percent": 0.0,
            "is_fraud": False,
            "risk_score_input": 0,
            "alert_messages": ["Error interno"],
            "ai_explanation": "Error"
        }