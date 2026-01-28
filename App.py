import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")

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
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. CARGAR MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        # Asegúrate de que este archivo sea el NUEVO que re-entrenaste
        return joblib.load('modelo_fraude_final.pkl')
    except FileNotFoundError:
        st.error("⚠️ Falta el archivo 'modelo_fraude_final.pkl'.")
        return None

data = cargar_modelo()

# --- 3. CÁLCULOS INTERNOS (BACKEND SIMULADO) ---
def analizar_contexto_hora(hour, transaction_type):
    penalizacion = 0
    es_noche = (hour >= 21) or (hour <= 6)
    es_madrugada = (0 <= hour <= 5)

    if transaction_type == 'ATM Withdrawal' and es_noche:
        penalizacion += 30
    elif transaction_type == 'Online Purchase' and es_madrugada:
        penalizacion += 15
    elif transaction_type == 'POS Purchase' and es_madrugada:
        penalizacion += 10
    return penalizacion

def simular_risk_score(amount, account_age_years, hour, transaction_type):
    """Calcula el score internamente sin mostrarlo."""
    base_score = 50
    base_score += analizar_contexto_hora(hour, transaction_type)
    
    if account_age_years < 0.5: base_score += 25
    elif account_age_years < 1.0: base_score += 15
    elif account_age_years > 5.0: base_score -= 10
        
    if amount > 1000: base_score += 15
    if amount > 5000: base_score += 20
        
    return int(min(max(base_score, 0), 100))

# --- 4. INTERFAZ ---
st.title("🛡️ FraudGuard AI")
st.markdown("### Monitor de Seguridad Transaccional")

if data:
    modelo = data['modelo']
    preprocessor = data['preprocesador']
    umbral_sugerido = data.get('umbral_optimo', 0.5)

    st.sidebar.header("⚙️ Ajuste de Sensibilidad")
    umbral_usuario = st.sidebar.slider("Nivel de Estrictez", 0.0, 1.0, float(umbral_sugerido), 0.01)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("#### 📝 Detalles de la Operación")
        
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0, step=10.0)
        
        account_age = st.number_input(
            "Antigüedad Cuenta (Años)", 
            min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.1f"
        )
        
        hour = st.slider("Hora (0-23h)", 0, 23, 22)
        
        transaction_type = st.selectbox("Tipo de Movimiento", 
                                        ['Online Purchase', 'Bank Transfer', 'ATM Withdrawal', 'POS Purchase'])
        
        customer_segment = st.selectbox("Segmento Cliente", 
                                        ['Retail', 'Business', 'Corporate'])
        
        # ELIMINADO: Ya no pedimos ni simulamos el Género.

    with col2:
        st.write("#### 🔍 Resultado del Análisis")
        st.write("") 
        
        if st.button("ANALIZAR RIESGO"):
            
            # 1. Cálculo silencioso del Risk Score
            risk_score_interno = simular_risk_score(amount, account_age, hour, transaction_type)
            
            # 2. Armar datos (SIN GÉNERO)
            # Verifica que estas columnas coincidan 100% con tu X_train del Colab
            input_data = pd.DataFrame({
                'amount': [amount],
                'account_age': [account_age],
                'risk_score': [risk_score_interno],
                'hour': [hour],
                'transaction_type': [transaction_type],
                'customer_segment': [customer_segment]
                # 'gender': ELIMINADO TOTALMENTE
            })

            try:
                # 3. Predicción
                input_processed = preprocessor.transform(input_data)
                probabilidad = modelo.predict_proba(input_processed)[0, 1]
                es_fraude = probabilidad >= umbral_usuario

                # 4. Visualización
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probabilidad * 100,
                    title = {'text': "Probabilidad de Fraude Detectada"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if es_fraude else "#00CC96"},
                        'steps': [
                            {'range': [0, umbral_usuario*100], 'color': "#E5F5F9"}, 
                            {'range': [umbral_usuario*100, 100], 'color': "#FFE4E1"} 
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': umbral_usuario * 100}
                    }
                ))
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                if es_fraude:
                    st.error(f"🚨 **TRANSACCIÓN BLOQUEADA**")
                    st.markdown(f"Se recomienda verificación de identidad.")
                else:
                    st.success(f"✅ **APROBADA**")
                    st.markdown(f"Operación dentro de los parámetros normales.")

            except Exception as e:
                st.error("Error en el procesamiento.")
                st.info("Posible causa: ¿Subiste el archivo .pkl nuevo a la carpeta?")
                st.caption(f"Error técnico: {e}")