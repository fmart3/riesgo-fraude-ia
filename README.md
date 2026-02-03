https://fraudgaurd-ai.onrender.com/


Actualizar image para Render

docker build --no-cache -t fmart3/fraud-gaurd:latest .

docker push fmart3/fraud-gaurd:latest



---------------------------------------------------------------------

Actualizar modelo: python update_model.py



Markdown
# 🛡️ FraudGuard AI: Real-Time Financial Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![Render](https://img.shields.io/badge/Deploy-Render-white)

Sistema de detección de fraude financiero en tiempo real. Utiliza un modelo de **Machine Learning (XGBoost/Sklearn)** servido a través de una API RESTful con **FastAPI**, containerizado en **Docker** y con persistencia de datos en **MongoDB Atlas**.

El sistema cuenta con una **capa lógica de traducción** que permite ingerir datos de negocio crudos (ej: "Online Purchase", "Retail") y adaptarlos dinámicamente a los tensores numéricos que requiere el modelo, resolviendo problemas de *Feature Mismatch* en producción.

---

## 📐 Arquitectura del Sistema

El flujo de datos conecta al cliente (Postman/Web) con el modelo de IA, pasando por una capa de saneamiento y traducción, asegurando que el modelo numérico (`.pkl`) pueda procesar datos semánticos del mundo real.

```mermaid
graph TD
    User((Cliente / Postman)) -->|POST JSON Payload| API[FastAPI Endpoint]
    
    subgraph "FraudGuard Service (Docker)"
        API -->|Datos Crudos| Translator{Capa de Traducción}
        Translator -->|Mapeo: Texto -> Int| Preproc[Sanitización]
        Preproc -->|Features Numéricas (8 cols)| Model[Modelo ML (.pkl)]
        Model -->|Probabilidad de Fraude| API
    end
    
    API -->|Guarda Predicción + Datos Ricos| DB[(MongoDB Atlas)]
    API -->|Respuesta JSON| User
```

🚀 Características Clave
Traducción Inteligente de Features: Convierte automáticamente términos de negocio (e.g., "ATM Withdrawal") a los códigos numéricos que el modelo aprendió durante el entrenamiento.

Manejo de Shape Mismatch: Rellena y alinea dinámicamente las columnas faltantes para evitar errores de dimensión en el modelo.

Persistencia Híbrida: Guarda en MongoDB tanto la predicción del modelo como los datos originales del usuario (que el modelo ignoró), permitiendo re-entrenamientos futuros más ricos.

API Rápida y Asíncrona: Construida sobre FastAPI para alta performance.

🛠️ Tech Stack
Lenguaje: Python 3.9

Framework Web: FastAPI + Uvicorn

ML Core: Scikit-Learn / Joblib

Base de Datos: MongoDB Atlas (Nube)

Infraestructura: Docker & Render

⚡ Instalación y Uso Local
1. Clonar el repositorio
Bash
git clone [https://github.com/tu-usuario/fraudguard-ai.git](https://github.com/tu-usuario/fraudguard-ai.git)
cd fraudguard-ai
2. Configurar Variables de Entorno
Crea un archivo .env en la raíz:

Fragmento de código
MONGO_URI=mongodb+srv://usuario:pass@cluster.mongodb.net/?retryWrites=true&w=majority
DB_NAME=FraudGuard_DB
COLLECTION_NAME=predicciones
3. Ejecutar con Docker (Recomendado)
Bash
# Construir la imagen
docker build -t fraudguard-ai .

# Correr el contenedor
docker run -p 8000:8000 --env-file .env fraudguard-ai
🔌 Consumo de la API
Una vez desplegado (en Render o Local), puedes probar el endpoint principal.

Endpoint: POST /analyze

Ejemplo de Request (JSON)
El sistema acepta datos de negocio reales:

JSON
{
  "amount": 150.0,
  "transaction_type": "Online Purchase",
  "account_age": 2.5,
  "risk_score": 60,
  "hour": 14,
  "customer_segment": "Retail",
  "oldbalanceOrg": 500.00
}
Ejemplo de Response
JSON
{
    "prediction_prob": 0.966,
    "is_fraud": true,
    "message": "Transacción analizada correctamente"
}
📂 Estructura del Proyecto
Bash
fraudguard-ai/
├── App.py              # Punto de entrada FastAPI
├── inference.py        # Lógica de traducción e inferencia (Cerebro)
├── modelo_fraude.pkl   # Artefacto del modelo entrenado
├── Dockerfile          # Configuración de la imagen
├── requirements.txt    # Dependencias
└── README.md           # Documentación
🔄 Flujo de Mantenimiento
Nuevos Datos: Los datos reales enviados a la API se guardan en MongoDB.

Re-entrenamiento: Periódicamente, se descargan los datos de Mongo para re-entrenar el modelo con nuevas tipologías de fraude.

Despliegue: Se actualiza el archivo .pkl y se hace push a Docker/Render.
