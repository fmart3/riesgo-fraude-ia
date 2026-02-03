https://fraudgaurd-ai.onrender.com/


Actualizar image para Render

docker build --no-cache -t fmart3/fraud-gaurd:latest .

docker push fmart3/fraud-gaurd:latest



---------------------------------------------------------------------

Actualizar modelo: python update_model.py



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
