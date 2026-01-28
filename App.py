import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. ESTILOS CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
    .metric-box { padding: 10px; background-color: white; border-radius: 5px; border-left: 5px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# --- 3. CARGAR MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        artefactos = joblib.load('modelo_fraude_final.pkl')
        return artefactos
    except FileNotFoundError:
        st.error("⚠️ No se encuentra el archivo 'modelo_fraude_final.pkl'.")
        return None

data = cargar_modelo()

# --- 4. FUNCIONES AUXILIARES (LÓGICA PURA) ---

def analizar_contexto_hora(hour, transaction_type):
    """
    Función pura: Recibe hora y tipo, devuelve penalización de riesgo.
    Separa la lógica compleja del cálculo general.
    """
    penalizacion = 0
    
    # Definimos qué es "Horario Nocturno" (21:00 a 06:00)
    es_noche = (hour >= 21) or (hour <= 6)
    es_madrugada = (0 <= hour <= 5)

    # REGLA A: Cajeros Automáticos (ATM)
    if transaction_type == 'ATM Withdrawal':
        if es_noche:
            # 🚨 ALERTA ROJA: Sacar plata de noche es muy sospechoso
            penalizacion += 30  
        else:
            # De día es comportamiento normal
            penalizacion += 0

    # REGLA B: Compras Online
    elif transaction_type == 'Online Purchase':
        if es_madrugada:
            # ⚠️ ALERTA AMARILLA: Comprar a las 3 AM es raro, pero no ilegal
            penalizacion += 15
            
    # REGLA C: Puntos de Venta (POS)
    elif transaction_type == 'POS Purchase':
        if es_madrugada:
            # Puede ser una fiesta/bar, riesgo medio
            penalizacion += 10

    return penalizacion

def simular_risk_score(amount, account_age_years, hour, transaction_type):
    """
    Calculadora principal que orquesta las reglas.
    """
    base_score = 50
    
    # 1. Llamamos a la función modularizada (Hora + Tipo)
    base_score += analizar_contexto_hora(hour, transaction_type)
    
    # 2. Regla de Antigüedad (Float)
    if account_age_years < 0.5: base_score += 25  # Cuenta muy nueva (< 6 meses)
    elif account_age_years < 1.0: base_score += 15
    elif account_age_years > 5.0: base_score -= 10 # Fidelidad
        
    # 3. Regla de Montos
    if amount > 1000: base_score += 15
    if amount > 5000: base_score += 20
        
    # Limitar el score entre 0 y 100 para que sea realista
    return min(max(base_score, 0), 100)


# --- 5. INTERFAZ DE USUARIO (FRONTEND) ---
st.title("🛡️ FraudGuard AI: Monitor de Transacciones")

if data:
    modelo = data['modelo']
    preprocessor = data['preprocesador']
    umbral_sugerido = data.get('umbral_optimo', 0.5)

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Configuración")
    umbral_usuario = st.sidebar.slider(
        "Sensibilidad (Umbral)", 
        0.0, 1.0, float(umbral_sugerido), 0.01,
        help="Ajusta el nivel de tolerancia del modelo."
    )
    st.sidebar.markdown(f"**Umbral Sugerido:** `{umbral_sugerido:.2f}`")

    # --- INPUTS ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Datos de Operación")
        
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0, step=10.0)
        
        # Input de Float para años
        account_age = st.number_input(
            "Antigüedad Cuenta (Años)", 
            min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.1f"
        )
        
        hour = st.slider("Hora (0-23h)", 0, 23, 22) # Por defecto a las 22:00 para probar tu caso
        
        # IMPORTANTE: Estos strings deben coincidir con tu entrenamiento
        transaction_type = st.selectbox("Tipo de Movimiento", 
                                        ['Online Purchase', 'Bank Transfer', 'ATM Withdrawal', 'POS Purchase'])
        
        customer_segment = st.selectbox("Segmento Cliente", 
                                        ['Retail', 'Business', 'Corporate'])
        
        gender = st.selectbox("Género", ['M', 'F'])

    with col2:
        st.subheader("🔍 Análisis en Tiempo Real")
        
        if st.button("ANALIZAR AHORA"):
            # 1. Calculamos el Risk Score usando la nueva estructura
            risk_score_calculado = simular_risk_score(amount, account_age, hour, transaction_type)
            
            # Mostramos el detalle del cálculo para que se entienda la lógica
            st.markdown(f"""
                <div class="metric-box">
                    ℹ️ <b>Backend Risk Score:</b> {risk_score_calculado}/100<br>
                    <small>Se detectó: {transaction_type} a las {hour}:00 h.</small>
                </div>
                <br>
            """, unsafe_allow_html=True)

            # 2. Preparamos el DataFrame (Inputs + Score Calculado)
            input_data = pd.DataFrame({
                'amount': [amount],
                'account_age': [account_age],
                'risk_score': [risk_score_calculado],
                'hour': [hour],
                'transaction_type': [transaction_type],
                'customer_segment': [customer_segment],
                'gender': [gender]
            })

            # 3. Predicción IA
            try:
                input_processed = preprocessor.transform(input_data)
                probabilidad = modelo.predict_proba(input_processed)[0, 1]
                es_fraude = probabilidad >= umbral_usuario

                # 4. Visualización
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probabilidad * 100,
                    title = {'text': "Probabilidad de Fraude (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if es_fraude else "green"},
                        'steps': [
                            {'range': [0, umbral_usuario*100], 'color': "lightgreen"},
                            {'range': [umbral_usuario*100, 100], 'color': "salmon"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': umbral_usuario * 100}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                if es_fraude:
                    st.error("🚨 TRANSACCIÓN BLOQUEADA")
                else:
                    st.success("✅ TRANSACCIÓN APROBADA")

            except Exception as e:
                st.error(f"Error técnico: {e}")