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

# --- 2. ESTILOS CSS (CORREGIDO) ---
# Se fuerza el color de texto a negro en los botones para contraste
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        width: 100%;
        height: 60px;
        font-size: 20px;
    }
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

# --- 4. FUNCIONES DE LÓGICA (MODULARIZADAS) ---

def analizar_contexto_hora(hour, transaction_type):
    """Calcular penalización por hora y tipo."""
    penalizacion = 0
    es_noche = (hour >= 21) or (hour <= 6)
    es_madrugada = (0 <= hour <= 5)

    if transaction_type == 'ATM Withdrawal':
        if es_noche:
            penalizacion += 30  # 🚨 ATM de noche
    elif transaction_type == 'Online Purchase':
        if es_madrugada:
            penalizacion += 15  # ⚠️ Compra madrugada
    elif transaction_type == 'POS Purchase':
        if es_madrugada:
            penalizacion += 10
            
    return penalizacion

def simular_risk_score(amount, account_age_years, hour, transaction_type):
    """Calculadora del Risk Score Simulado (Backend)."""
    base_score = 50
    
    # 1. Contexto Hora/Tipo
    base_score += analizar_contexto_hora(hour, transaction_type)
    
    # 2. Antigüedad
    if account_age_years < 0.5: base_score += 25
    elif account_age_years < 1.0: base_score += 15
    elif account_age_years > 5.0: base_score -= 10
        
    # 3. Montos
    if amount > 1000: base_score += 15
    if amount > 5000: base_score += 20
        
    # Limitar estrictamente entre 0 y 100
    return int(min(max(base_score, 0), 100))

# --- 5. INTERFAZ GRÁFICA ---
st.title("🛡️ FraudGuard AI: Monitor de Transacciones")

if data:
    modelo = data['modelo']
    preprocessor = data['preprocesador']
    umbral_sugerido = data.get('umbral_optimo', 0.5)

    # BARRA LATERAL
    st.sidebar.header("⚙️ Configuración")
    umbral_usuario = st.sidebar.slider(
        "Sensibilidad (Umbral)", 
        0.0, 1.0, float(umbral_sugerido), 0.01
    )
    st.sidebar.info(f"Umbral IA Sugerido: **{umbral_sugerido:.2f}**")

    # COLUMNAS PRINCIPALES
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Datos de Operación")
        
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0, step=10.0)
        
        account_age = st.number_input(
            "Antigüedad Cuenta (Años)", 
            min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.1f"
        )
        
        hour = st.slider("Hora (0-23h)", 0, 23, 22) # Default 22 para probar ATM nocturno
        
        # Strings exactos del entrenamiento
        transaction_type = st.selectbox("Tipo de Movimiento", 
                                        ['Online Purchase', 'Bank Transfer', 'ATM Withdrawal', 'POS Purchase'])
        
        customer_segment = st.selectbox("Segmento Cliente", 
                                        ['Retail', 'Business', 'Corporate'])
        
        gender = st.selectbox("Género", ['M', 'F'])

    with col2:
        st.subheader("🔍 Análisis en Tiempo Real")
        
        # Botón de acción
        if st.button("ANALIZAR AHORA"):
            
            # 1. Calcular Risk Score
            risk_score_calculado = simular_risk_score(amount, account_age, hour, transaction_type)
            
            # 2. MOSTRAR RESULTADO DEL BACKEND (Usando st.metric nativo para evitar error visual)
            # Creamos 3 columnas pequeñas para mostrar los datos calculados limpiamente
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Score (Backend)", f"{risk_score_calculado}/100", delta_color="inverse")
            m2.metric("Hora Detectada", f"{hour}:00 hrs")
            m3.metric("Tipo", transaction_type)

            # Mensaje explicativo si el riesgo es alto por lógica interna
            if risk_score_calculado > 80:
                st.warning("⚠️ **Nota:** El sistema interno ha elevado el riesgo debido a la hora y tipo de transacción.")

            st.divider() # Línea divisoria visual

            # 3. Prepara DataFrame para la IA
            input_data = pd.DataFrame({
                'amount': [amount],
                'account_age': [account_age],
                'risk_score': [risk_score_calculado],
                'hour': [hour],
                'transaction_type': [transaction_type],
                'customer_segment': [customer_segment],
                'gender': [gender]
            })

            # 4. Predicción IA
            try:
                input_processed = preprocessor.transform(input_data)
                probabilidad = modelo.predict_proba(input_processed)[0, 1]
                es_fraude = probabilidad >= umbral_usuario

                # 5. Gráfico Gauge
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

                # Veredicto Final
                if es_fraude:
                    st.error("🚨 TRANSACCIÓN BLOQUEADA")
                else:
                    st.success("✅ TRANSACCIÓN APROBADA")

            except Exception as e:
                st.error(f"Error técnico: {e}")