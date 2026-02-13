from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional
import os
import sys

# ==============================================================================
# BLOQUE DE AJUSTE DE RUTAS (AGREGAR ESTO AL INICIO)
# ==============================================================================
# 1. Obtener la ruta absoluta de la carpeta donde está este script (misc)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Obtener la ruta raíz del proyecto (un nivel arriba de misc)
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

# 3. Agregar la raíz al 'sys.path' para poder importar 'utils'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. CAMBIAR EL DIRECTORIO DE TRABAJO A LA RAÍZ
# Esto es vital: hace que cuando los otros scripts busquen "questions.json" 
# o ".env", los encuentren en la raíz y no busquen en 'misc'.
os.chdir(project_root)
# ==============================================================================

# --- 1. ENUMS (Listas cerradas de opciones) ---
# Usamos Enum para obligar a que el usuario solo pueda enviar
# estos valores exactos. Si envía "Cajero Automatico" (con espacio o tilde diferente), fallará.

class TransactionType(str, Enum):
    ONLINE = 'Online Purchase'
    ATM = 'ATM Withdrawal'
    POS = 'POS Purchase'
    TRANSFER = 'Bank Transfer'

class CustomerSegment(str, Enum):
    RETAIL = 'Retail'
    BUSINESS = 'Business'
    CORPORATE = 'Corporate'

# --- 2. INPUT (Datos que recibimos del formulario) ---
class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="El monto de la transacción. Debe ser mayor a 0.")
    
    account_age: float = Field(..., ge=0, description="Antigüedad de la cuenta en años.")
    
    hour: int = Field(..., ge=0, le=23, description="Hora del día en formato 24h (0 a 23).")
    
    transaction_type: TransactionType = Field(..., description="Tipo de movimiento (Select).")
    
    customer_segment: CustomerSegment = Field(..., description="Segmento del cliente.")

    # Nota: No pedimos 'risk_score' aquí porque eso lo calculamos nosotros internamente.
    # Tampoco pedimos 'gender' porque lo eliminamos del modelo.

    class Config:
        # Esto sirve para mostrar un ejemplo en la documentación automática (Swagger)
        json_schema_extra = {
            "example": {
                "amount": 150.50,
                "account_age": 2.5,
                "hour": 22,
                "transaction_type": "ATM Withdrawal",
                "customer_segment": "Retail"
            }
        }

# --- 3. OUTPUT (Datos que devolvemos al frontend) ---
class PredictionResponse(BaseModel):
    probability_percent: float = Field(..., description="Probabilidad de fraude en porcentaje (0-100).")
    is_fraud: bool = Field(..., description="Booleano final: True si se debe bloquear, False si se aprueba.")
    risk_score_input: int = Field(..., description="El puntaje de riesgo calculado por nuestras reglas de negocio.")
    alert_messages: List[str] = Field(default=[], description="Lista de mensajes explicativos o advertencias.")
    shap_image_base64: Optional[str] = Field(None, description="Imagen del gráfico SHAP codificada en Base64 para mostrar en HTML.")
    ai_explanation: Optional[str] = Field(None, description="Explicación generada por IA sobre la decisión tomada.")