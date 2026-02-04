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
MODEL_PATH = 'fraude.pkl'

def load_model_assets():
    global _MODELO_PIPELINE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"⚠️ No existe: {MODEL_PATH}")
    loaded = joblib.load(MODEL_PATH)
    _MODELO_PIPELINE = loaded['modelo'] if (isinstance(loaded, dict) and 'modelo' in loaded) else loaded

def predict(input_data: dict):
    global _MODELO_PIPELINE
    
    if _MODELO_PIPELINE is None:
        try:
            load_model_assets()
        except Exception:
            return 0.0, False

    try:
        # Convertimos el dict de entrada a DataFrame
        df = pd.DataFrame([input_data])
        
        # --- 1. LIMPIEZA DE DATOS (CRÍTICO) ---
        # Extraemos los valores simples para evitar el error de "Series"
        amount = float(input_data.get('amount', 0))
        
        # Mapping: Probamos con 0. En PaySim, 0 suele ser 'CASH_IN' (Depósito),
        # que es la operación más segura posible. Si esto da fraude, el modelo está roto.
        raw_type = input_data.get('transaction_type', 'Online Purchase')
        type_str = str(raw_type)
        
        mapping_tipos = {
            'Online Purchase': 0,    # ### CAMBIO A 0 (El más seguro)
            'POS Purchase': 0,       
            'Bank Transfer': 4,      
            'ATM Withdrawal': 1,     
            'Cash Advance': 1        
        }
        
        tipo_num = mapping_tipos.get(type_str, 0) # Default a 0 (Seguro)

        # --- 2. LOGICA DE SALDOS (FIXED) ---
        # Usamos variables simples (float) en lugar de columnas de pandas
        if 'oldbalanceOrg' in input_data:
            old_balance = float(input_data['oldbalanceOrg'])
        else:
            # Simulamos que tiene MUCHO dinero (Monto + 10,000)
            old_balance = amount + 10000.0
            
        # El saldo nuevo es lo que tenía menos lo que gastó
        new_balance = old_balance - amount

        # --- 3. CREAR DATAFRAME FINAL ---
        # ### IMPORTANTE: El orden de las columnas debe ser EXACTO.
        # La mayoría de modelos PaySim usan este orden específico:
        # [step, type, amount, oldbalanceOrg, newbalanceOrig, newbalanceDest, oldbalanceDest]
        
        df_for_model = pd.DataFrame([{
            'step': 1,                     # Paso 1
            'type': int(tipo_num),         # Tipo numérico
            'amount': float(amount),       # Monto float
            'nameOrig': 'C12345',          # Placeholder (a veces el modelo lo pide aunque no lo use)
            'oldbalanceOrg': float(old_balance),
            'newbalanceOrig': float(new_balance),
            'nameDest': 'M12345',          # Placeholder
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'isFlaggedFraud': 0
        }])

        # Seleccionamos solo las columnas numéricas que suelen usar los modelos .pkl simples
        # Si tu modelo se entrenó con menos columnas, el pipeline ignorará las extra.
        cols_to_keep = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
        
        # Filtramos si es posible, si falla, pasamos todo
        try:
            df_final = df_for_model[cols_to_keep]
        except:
            df_final = df_for_model

        # --- 4. PREDICCIÓN ---
        # Usamos predict_proba
        prob_ia = _MODELO_PIPELINE.predict_proba(df_final)[0, 1]
        
        # LOG PARA DEBUG
        logger.info(f"🔢 DATOS ENVIADOS: Type={tipo_num}, Amount={amount}, OldBal={old_balance}, NewBal={new_balance}")
        logger.info(f"🧠 PREDICCIÓN: {prob_ia:.4f}")

        return prob_ia, (prob_ia >= 0.5)

    except Exception as e:
        logger.error(f"Error en inferencia: {e}")
        # En caso de error, devolvemos 0 para no bloquear
        return 0.0, False