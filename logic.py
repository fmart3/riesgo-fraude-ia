import pandas as pd
from schemas import TransactionRequest, TransactionType

def validate_hard_constraints(data: TransactionRequest):
    """
    Validaciones bloqueantes antes de calcular nada.
    Si esto falla, se lanza un error y la app se detiene (Error 400).
    """
    # REGLA: Cajeros Automáticos (ATM)
    if data.transaction_type == TransactionType.ATM:
        if data.amount < 20:
            raise ValueError("⛔ Error ATM: El monto mínimo de retiro es $20.")
        
        # Verificamos si tiene decimales (ej: 20.50)
        if data.amount % 1 != 0:
            raise ValueError("⛔ Error ATM: El cajero solo entrega billetes enteros (sin centavos).")

def calculate_risk_score(data: TransactionRequest) -> int:
    """
    Calcula el puntaje de riesgo base (0-100) usando reglas heurísticas.
    Devuelve un entero.
    """
    score = 50  # Puntaje base neutral
    
    # Variables auxiliares de tiempo
    hour = data.hour
    es_noche = (hour >= 21) or (hour <= 6)      # 9 PM - 6 AM
    es_madrugada = (0 <= hour <= 5)             # 12 AM - 5 AM

    # --- 1. REGLAS DE MUERTE SÚBITA (KILL SWITCH) ---
    # Si ocurren, ignoramos el resto y devolvemos riesgo alto inmediato.

    # ATM de Noche -> Muy peligroso
    if data.transaction_type == TransactionType.ATM and es_noche:
        return 95 
    
    # Compra Online de Madrugada -> Sospechoso
    if data.transaction_type == TransactionType.ONLINE and es_madrugada:
        return 80

    # --- 2. CÁLCULO ESTÁNDAR (Si no hay kill switch) ---
    
    # A. Ajuste por Tipo
    if data.transaction_type == TransactionType.TRANSFER:
        score -= 5  # Transferencias suelen ser más seguras
        if es_madrugada:
            score += 20 # ...a menos que sea a las 3 AM (posible vaciado)
            
    elif data.transaction_type == TransactionType.POS:
        if es_madrugada:
            score += 15 # Tarjeta física usada de madrugada
            
    elif data.transaction_type == TransactionType.ONLINE:
        score += 10 # Online siempre tiene un riesgo base mayor

    # B. Ajuste por Antigüedad (Mitigante)
    if data.account_age < 0.5: score += 25   # Cuenta nueva = alto riesgo
    elif data.account_age < 1.0: score += 15
    elif data.account_age > 5.0: score -= 15 # Cliente fiel = bajo riesgo

    # C. Ajuste por Monto
    if data.transaction_type == TransactionType.TRANSFER:
        # Transferencias toleran montos altos
        if data.amount > 5000: score += 10
        if data.amount > 20000: score += 20
    else:
        # ATM, POS, Online son sensibles a montos medios
        if data.amount > 1000: score += 15
        if data.amount > 5000: score += 20

    # D. Limitar rango 0-100
    return int(min(max(score, 0), 100))

def get_business_warnings(data: TransactionRequest, risk_score: int) -> list[str]:
    """
    Genera mensajes explicativos para el humano basados en las reglas activadas.
    """
    warnings = []
    hour = data.hour
    es_noche = (hour >= 21) or (hour <= 6)
    
    # Alertas específicas
    if risk_score >= 90:
        warnings.append("⚠️ CRÍTICO: Patrón de alto riesgo detectado (Regla de Muerte Súbita).")
    
    if data.transaction_type == TransactionType.ATM and es_noche:
        warnings.append("🏧 Alerta ATM: Retiro en horario nocturno incrementa drásticamente el riesgo.")
        
    if data.transaction_type == TransactionType.TRANSFER and (0 <= hour <= 5):
        warnings.append("🕒 Alerta Hora: Transferencia realizada en horario inusual (madrugada).")
        
    if data.account_age < 0.5:
        warnings.append("👶 Cuenta Nueva: La baja antigüedad (< 6 meses) penaliza el puntaje.")

    return warnings

def process_business_rules(data: TransactionRequest):
    """
    Función Maestra llamada por App.py.
    Orquesta validaciones, cálculos y preparación de datos.
    """
    # 1. Validar restricciones duras (Lanza error si falla)
    validate_hard_constraints(data)

    # 2. Calcular Score
    risk_score = calculate_risk_score(data)
    
    # 3. Generar advertencias de negocio
    warnings = get_business_warnings(data, risk_score)

    # 4. Crear DataFrame para el modelo (Enriquecimiento)
    # Convertimos el objeto Pydantic a un DataFrame de 1 fila
    # Nota: Usamos data.transaction_type.value para obtener el string "ATM Withdrawal" en vez del Enum
    processed_df = pd.DataFrame([{
        'amount': data.amount,
        'account_age': data.account_age,
        'risk_score': risk_score,  # <--- Aquí inyectamos nuestra lógica experta
        'hour': data.hour,
        'transaction_type': data.transaction_type.value, 
        'customer_segment': data.customer_segment.value
    }])
    
    return processed_df, warnings