import pandas as pd
from schemas import TransactionRequest, TransactionType

# --- 1. SE ELIMINÓ "validate_hard_constraints" ---
# Antes había aquí una función que bloqueaba la app si el monto era < 20.
# La hemos borrado para que el modelo juzgue TODO.

def calculate_risk_score(data: TransactionRequest) -> int:
    """
    Calcula un puntaje (0-100) que sirve como INPUT para el modelo de IA.
    NOTA: Esto no decide si es fraude, solo genera una variable numérica
    que ayuda al modelo a entender el nivel de riesgo heurístico.
    """
    score = 50  # Puntaje base neutral
    
    # Variables auxiliares
    hour = data.hour
    es_noche = (hour >= 21) or (hour <= 6)
    es_madrugada = (0 <= hour <= 5)

    # --- Lógica de Puntuación (Feature Engineering) ---
    
    # Casos de alta sospecha (Suben el score, pero no deciden el resultado final)
    if data.transaction_type == TransactionType.ATM and es_noche:
        return 95 
    
    if data.transaction_type == TransactionType.ONLINE and es_madrugada:
        return 80

    # Ajustes estándar
    if data.transaction_type == TransactionType.TRANSFER:
        score -= 5
        if es_madrugada:
            score += 20
            
    elif data.transaction_type == TransactionType.POS:
        if es_madrugada:
            score += 15
            
    elif data.transaction_type == TransactionType.ONLINE:
        score += 10

    # Ajuste por Antigüedad
    if data.account_age < 0.5: score += 25
    elif data.account_age < 1.0: score += 15
    elif data.account_age > 5.0: score -= 15

    # Ajuste por Monto
    if data.transaction_type == TransactionType.TRANSFER:
        if data.amount > 5000: score += 10
        if data.amount > 20000: score += 20
    else:
        if data.amount > 1000: score += 15
        if data.amount > 5000: score += 20

    # Retornamos el entero limitado entre 0 y 100
    return int(min(max(score, 0), 100))

def get_business_warnings(data: TransactionRequest, risk_score: int) -> list[str]:
    """
    Genera mensajes de contexto para mostrar en el Dashboard.
    Estos son informativos y no afectan la decisión del modelo.
    """
    warnings = []
    hour = data.hour
    es_noche = (hour >= 21) or (hour <= 6)
    
    # Mensajes explicativos sobre por qué el Score puede ser alto
    if risk_score >= 90:
        warnings.append("⚠️ Contexto: Patrón inusual detectado en reglas básicas.")
    
    if data.transaction_type == TransactionType.ATM and es_noche:
        warnings.append("🏧 Contexto: Retiro nocturno suele elevar el riesgo.")
        
    if data.transaction_type == TransactionType.TRANSFER and (0 <= hour <= 5):
        warnings.append("🕒 Contexto: Transferencia de madrugada.")
        
    if data.account_age < 0.5:
        warnings.append("👶 Contexto: Cuenta reciente (< 6 meses).")

    return warnings

def process_business_rules(data: TransactionRequest):
    """
    Función Maestra.
    Prepara los datos para que el modelo pueda ingerirlos.
    """
    # 1. ELIMINADO: Ya no llamamos a validate_hard_constraints(data)
    # El flujo nunca se bloquea aquí.

    # 2. Calcular Variable Auxiliar (Score)
    # Necesario porque el modelo se entrenó usando esta columna.
    risk_score = calculate_risk_score(data)
    
    # 3. Generar advertencias informativas
    warnings = get_business_warnings(data, risk_score)

    # 4. Crear DataFrame para el modelo
    processed_df = pd.DataFrame([{
        'amount': data.amount,
        'account_age': data.account_age,
        'risk_score': risk_score,  # Se pasa como dato, no como veredicto
        'hour': data.hour,
        'transaction_type': data.transaction_type.value, 
        'customer_segment': data.customer_segment.value
    }])
    
    return processed_df, warnings