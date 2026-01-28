import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- CARGAR MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        artefactos = joblib.load('modelo_fraude_final.pkl')
        return artefactos
    except FileNotFoundError:
        st.error("⚠️ Falta el archivo 'modelo_fraude_final.pkl'.")
        return None

data = cargar_modelo()

# --- FUNCIÓN DE SIMULACIÓN DE BACKEND ---
def simular_risk_score(amount, account_age, hour):
    """
    Simula el cálculo interno del banco. 
    El usuario NO ingresa esto; el sistema lo deduce.
    """
    base_score = 50  # Riesgo base
    
    # Regla 1: Cuentas nuevas son más riesgosas
    if account_age < 30:
        base_score += 30
    elif account_age < 90:
        base_score += 10
        
    # Regla 2: Montos altos suben el riesgo
    if amount > 1000:
        base_score += 20
    if amount > 5000:
        base_score += 20
        
    # Regla 3: Horas de madrugada (00:00 a 05:00) son sospechosas
    if 0 <= hour <= 5:
        base_score += 15
        
    # Limitar entre 0 y 100
    return min(max(base_score, 0), 100)

# --- INTERFAZ ---
st.title("🛡️ FraudGuard AI: Monitor de Transacciones")
st.markdown("---")

if data:
    modelo = data['modelo']
    preprocessor = data['preprocesador']
    umbral_sugerido = data.get('umbral_optimo', 0.5)

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Configuración del Sistema")
    umbral_usuario = st.sidebar.slider(
        "Sensibilidad del Modelo", 
        0.0, 1.0, float(umbral_sugerido), 0.01,
        help="Ajusta qué tan estricto es el modelo."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Modelo: XGBoost v1.0")

    # --- PANTALLA PRINCIPAL ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Datos de la Operación")
        
        # Inputs que SÍ ingresa el usuario/cajero
        amount = st.number_input("Monto de Transacción ($)", min_value=0.0, value=150.0)
        account_age = st.number_input("Antigüedad de la Cuenta (días)", min_value=0, value=365)
        hour = st.slider("Hora de la Transacción (0-23h)", 0, 23, 14)
        
        transaction_type = st.selectbox("Tipo de Movimiento", 
                                        ['purchase', 'transfer', 'withdrawal', 'payment'])
        customer_segment = st.selectbox("Segmento del Cliente", 
                                        ['low', 'medium', 'high'])
        gender = st.selectbox("Género Titular", ['M', 'F'])

    with col2:
        st.subheader("🔍 Análisis en Tiempo Real")
        
        if st.button("ANALIZAR AHORA"):
            # 1. SIMULACIÓN DE BACKEND (Cálculo automático del Risk Score)
            risk_score_calculado = simular_risk_score(amount, account_age, hour)
            
            # Mostramos el score calculado para transparencia
            st.info(f"ℹ️ **Score de Riesgo Interno (Backend):** {risk_score_calculado}/100")

            # 2. Preparar datos para el modelo
            input_data = pd.DataFrame({
                'amount': [amount],
                'account_age': [account_age],
                'risk_score': [risk_score_calculado], # Usamos el calculado
                'hour': [hour],
                'transaction_type': [transaction_type],
                'customer_segment': [customer_segment],
                'gender': [gender]
            })

            # 3. Predicción
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
                    st.error("🚨 TRANSACCIÓN BLOQUEADA POR RIESGO DE FRAUDE")
                else:
                    st.success("✅ TRANSACCIÓN APROBADA")

            except Exception as e:
                st.error(f"Error en predicción: {e}")