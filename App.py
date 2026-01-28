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

# --- CARGAR EL CEREBRO (MODELO) ---
@st.cache_resource
def cargar_modelo():
    try:
        artefactos = joblib.load('modelo_fraude_final.pkl')
        return artefactos
    except FileNotFoundError:
        st.error("⚠️ No se encuentra el archivo 'modelo_fraude_final.pkl'.")
        return None

data = cargar_modelo()

# --- LÓGICA DE NEGOCIO (BACKEND SIMULADO) ---
def simular_risk_score(amount, account_age_years, hour):
    """
    Simula el cálculo interno del banco.
    Recibe float en account_age_years (ej: 1.5 años).
    """
    base_score = 50
    
    # Regla 1: Antigüedad (Ahora soporta decimales)
    # Cuentas con menos de 6 meses (0.5 años) son muy riesgosas
    if account_age_years < 0.5:
        base_score += 40
    elif account_age_years < 1.0:
        base_score += 20
    elif account_age_years < 2.0:
        base_score += 10
    else:
        # Premiar fidelidad (>5 años)
        if account_age_years > 5.0:
            base_score -= 10
        
    # Regla 2: Montos altos
    if amount > 1000: base_score += 20
    if amount > 5000: base_score += 20
        
    # Regla 3: Horario de madrugada
    if 0 <= hour <= 5: base_score += 15
        
    # Limitar entre 0 y 100
    return min(max(base_score, 0), 100)

# --- INTERFAZ DE USUARIO ---
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
        help="Menor valor = Más estricto (detecta más, pero más falsas alarmas)."
    )
    st.sidebar.caption(f"Umbral Sugerido por IA: {umbral_sugerido:.2f}")

    # --- PANTALLA PRINCIPAL ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Datos de la Operación")
        
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0, step=10.0)
        
        # --- CORRECCIÓN FLOAT APLICADA ---
        # value=2.0 fuerza a que sea float. step=0.1 permite decimales.
        account_age = st.number_input(
            "Antigüedad Cuenta (Años)", 
            min_value=0.0, 
            max_value=100.0, 
            value=2.0, 
            step=0.1,
            format="%.1f" # Muestra 1 decimal visualmente
        )
        
        hour = st.slider("Hora (0-23h)", 0, 23, 14)
        
        transaction_type = st.selectbox("Tipo de Movimiento", 
                                        ['purchase', 'transfer', 'withdrawal', 'payment'])
        customer_segment = st.selectbox("Segmento Cliente", 
                                        ['low', 'medium', 'high'])
        gender = st.selectbox("Género", ['M', 'F'])

    with col2:
        st.subheader("🔍 Análisis en Tiempo Real")
        
        if st.button("ANALIZAR AHORA"):
            # 1. Simulación Backend
            risk_score_calculado = simular_risk_score(amount, account_age, hour)
            
            st.info(f"ℹ️ **Risk Score Calculado:** {risk_score_calculado}/100")

            # 2. DataFrame para el modelo
            input_data = pd.DataFrame({
                'amount': [amount],
                'account_age': [account_age], # Ahora va como float (ej: 1.5)
                'risk_score': [risk_score_calculado],
                'hour': [hour],
                'transaction_type': [transaction_type],
                'customer_segment': [customer_segment],
                'gender': [gender]
            })

            # 3. Predicción
            try:
                # Transformar datos
                input_processed = preprocessor.transform(input_data)
                
                # Obtener probabilidad
                probabilidad = modelo.predict_proba(input_processed)[0, 1]
                
                # Decisión
                es_fraude = probabilidad >= umbral_usuario

                # 4. Gráfico Gauge
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

                # Mensaje Final
                if es_fraude:
                    st.error("🚨 ALERTA: TRANSACCIÓN BLOQUEADA")
                    st.write("El modelo ha detectado patrones de alto riesgo.")
                else:
                    st.success("✅ TRANSACCIÓN APROBADA")

            except Exception as e:
                st.error(f"Error técnico: {e}")
                st.write("Verifica que las columnas coincidan con el entrenamiento.")