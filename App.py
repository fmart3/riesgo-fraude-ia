import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")

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

# --- LÓGICA DE NEGOCIO (CORREGIDA A AÑOS) ---
def simular_risk_score(amount, account_age_years, hour):
    base_score = 50
    
    # Regla 1: Antigüedad (Ajustada a AÑOS)
    # Si tiene menos de 1 año, es más riesgoso
    if account_age_years < 1: 
        base_score += 30
    elif account_age_years < 2:
        base_score += 10
    else:
        # Cuentas viejas (>5 años) bajan el riesgo
        if account_age_years > 5:
            base_score -= 10
        
    # Regla 2: Montos
    if amount > 1000: base_score += 20
    if amount > 5000: base_score += 20
        
    # Regla 3: Horario
    if 0 <= hour <= 5: base_score += 15
        
    return min(max(base_score, 0), 100)

# --- INTERFAZ ---
st.title("🛡️ FraudGuard AI: Monitor de Transacciones")

if data:
    modelo = data['modelo']
    preprocessor = data['preprocesador']
    umbral_sugerido = data.get('umbral_optimo', 0.5)

    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    umbral_usuario = st.sidebar.slider("Sensibilidad", 0.0, 1.0, float(umbral_sugerido), 0.01)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Datos de la Operación")
        
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0)
        
        # --- CORRECCIÓN AQUÍ: INPUT EN AÑOS ---
        account_age = st.number_input("Antigüedad Cuenta (Años)", min_value=0, max_value=100, value=2)
        
        hour = st.slider("Hora (0-23h)", 0, 23, 14)
        transaction_type = st.selectbox("Tipo", ['purchase', 'transfer', 'withdrawal', 'payment'])
        customer_segment = st.selectbox("Segmento", ['low', 'medium', 'high'])
        gender = st.selectbox("Género", ['M', 'F'])

    with col2:
        st.subheader("🔍 Análisis")
        
        if st.button("ANALIZAR AHORA"):
            # Calculamos riesgo usando la lógica de AÑOS
            risk_score_calculado = simular_risk_score(amount, account_age, hour)
            st.info(f"ℹ️ **Risk Score Calculado:** {risk_score_calculado}/100")

            # Crear DF
            input_data = pd.DataFrame({
                'amount': [amount],
                'account_age': [account_age], # Enviamos AÑOS al modelo
                'risk_score': [risk_score_calculado],
                'hour': [hour],
                'transaction_type': [transaction_type],
                'customer_segment': [customer_segment],
                'gender': [gender]
            })

            try:
                input_processed = preprocessor.transform(input_data)
                probabilidad = modelo.predict_proba(input_processed)[0, 1]
                es_fraude = probabilidad >= umbral_usuario

                # Gauge Chart
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
                    st.error("🚨 BLOQUEADO POR RIESGO")
                else:
                    st.success("✅ APROBADO")

            except Exception as e:
                st.error(f"Error: {e}")