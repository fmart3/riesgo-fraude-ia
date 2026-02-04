import joblib
import pandas as pd
import logging
import os
import certifi
from pymongo import MongoClient
from datetime import datetime

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN MONGODB ---
_MONGO_COLLECTION = None
try:
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME", "FraudGuard_DB")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "predicciones")
    if MONGO_URI:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        _MONGO_COLLECTION = client[DB_NAME][COLLECTION_NAME]
        logger.info("✅ Conexión a MongoDB exitosa.")
except Exception as e:
    logger.error(f"⚠️ Error MongoDB: {e}")

# --- CARGA DEL MODELO ---
_MODELO_PIPELINE = None
MODEL_PATH = 'modelo_fraude.pkl'

def load_model_assets():
    global _MODELO_PIPELINE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No existe: {MODEL_PATH}")
    loaded = joblib.load(MODEL_PATH)
    _MODELO_PIPELINE = loaded['modelo'] if (isinstance(loaded, dict) and 'modelo' in loaded) else loaded

def predict(input_data: dict):
    """
    Recibe un diccionario directo del usuario.
    Prepara los datos para que el modelo sea NEUTRO con los saldos.
    """
    global _MODELO_PIPELINE
    if _MODELO_PIPELINE is None: load_model_assets()
    
    try:
        # Convertir dict a DataFrame
        df = pd.DataFrame([input_data])
        
        # 1. TRADUCCIÓN DE TIPOS (Texto -> Número)
        raw_type = df.iloc[0].get('transaction_type', 'Online Purchase')
        tipo_str = str(raw_type)  # <--- ESTO ASEGURA QUE SEA TEXTO "Online Purchase"
        
        mapping_tipos = {
            'Online Purchase': 3,                    # PAYMENT                              
            'POS Purchase': 2,                       # DEBIT
            'Bank Transfer': 4,                      # TRANSFER
            'ATM Withdrawal': 1,                     # CASH_OUT
            'Cash Advance': 0                        # CASH_IN
        }
        
        # Si viene 'transaction_type', lo usamos. Si no, default a 1.
        tipo_str = df.iloc[0].get('transaction_type', 'Online Purchase')
        tipo_num = mapping_tipos.get(tipo_str, 1)

        # 2. CONSTRUIR DATAFRAME EXACTO PARA EL MODELO (8 Columnas)
        df_model = pd.DataFrame()
        
        # -- Step (Hora) --
        df_model['step'] = df['hour'].astype(int) if 'hour' in df.columns else 1
        
        # -- Type --
        df_model['type'] = int(tipo_num)
        
        # -- Amount --
        amount = float(df.iloc[0].get('amount', 0))
        df_model['amount'] = amount
        
        # -- NameOrig / NameDest (Siempre 0) --
        df_model['nameOrig'] = 0
        df_model['nameDest'] = 0

        # -- BALANCES (EL FIX CRÍTICO) --
        # Si no traen saldo, asumimos que tiene fondos suficientes.
        # oldbalanceOrg = amount -> Significa que tenía justo lo que gastó.
        # newbalanceOrig = 0 -> Significa que se quedó en 0.
        # Esto hace que el modelo NO sospeche por falta de fondos.
        
        if 'oldbalanceOrg' in df.columns:
            df_model['oldbalanceOrg'] = float(df.iloc[0]['oldbalanceOrg'])
            val_old = df_model['oldbalanceOrg']
        else:
            # SIMULACIÓN: El cliente tiene el monto + $5,000 extra en la cuenta
            df_model['oldbalanceOrg'] = amount + 5000.0
            val_old = df_model['oldbalanceOrg']

        if 'newbalanceOrig' in df.columns:
            df_model['newbalanceOrig'] = float(df.iloc[0]['newbalanceOrig'])
        else:
            # El saldo final es lo que tenía menos lo que gastó (Quedan $5,000)
            df_model['newbalanceOrig'] = float(val_old - amount)
            
        df_model['oldbalanceDest'] = 0 # Destino irrelevante para fraude origen
        df_model['oldbalanceDest'] = 0 # (Repetido por seguridad del shape)

        # Asegurar orden exacto de columnas
        expected_cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest"]
        df_model = df_model[expected_cols]

        # 3. PREDICCIÓN PURA
        prob = _MODELO_PIPELINE.predict_proba(df_model)[0, 1]
        is_fraud = prob >= 0.5
        
        logger.info(f"🧠 Predicción Pura: {prob:.4f} (Sin reglas externas)")

        # 4. GUARDAR EN MONGO (Datos originales + Predicción)
        if _MONGO_COLLECTION is not None:
            record = input_data.copy()
            record["prediction_prob"] = float(prob)
            record["prediction_is_fraud"] = bool(is_fraud)
            record["timestamp"] = datetime.utcnow()
            _MONGO_COLLECTION.insert_one(record)

        return prob, is_fraud

    except Exception as e:
        logger.error(f"Error Infer: {e}")
        raise e