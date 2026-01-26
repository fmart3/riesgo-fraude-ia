import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Detector de Fraude IA", page_icon="🕵️‍♂️", layout="centered")

# --- CARGAR EL CEREBRO DE LA IA ---
@st.cache_resource
def cargar_modelo():
    # Asegúrate de que el nombre coincida con el que descargaste
    return joblib.load('modelo_fraude_v1.pkl')

try:
    data = cargar_modelo()
    model = data["modelo"]
    preprocessor = data["preprocesador"]
    threshold = 0.35
    st.success("✅ Sistema de IA cargado correctamente.")
except FileNotFoundError:
    st.error("❌ No se encuentra 'modelo_fraude_v1.pkl'. Asegúrate de que esté en la misma carpeta.")
    st.stop()

# --- INTERFAZ GRÁFICA ---
st.title("🕵️‍♂️ Sistema de Detección de Fraude")
st.markdown("Ingrese los detalles de la transacción para evaluar el riesgo en tiempo real.")

with st.form("fraude_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Monto de Transacción ($)", min_value=0.0, value=100.0)
        hour = st.slider("Hora del día (0-23)", 0, 23, 14)
        risk_score = st.slider("Risk Score (Interno)", 0, 100, 50)
    
    with col2:
        trans_type = st.selectbox("Tipo de Transacción", ['Online Purchase', 'ATM Withdrawal', 'POS Purchase', 'Bank Transfer'])
        account_age = st.number_input("Antigüedad de Cuenta (años)", min_value=0.0, value=5.0)
        # Location y Segment son menos relevantes según SHAP, ponemos valores por defecto o inputs simples
        customer_segment = st.selectbox("Segmento Cliente", ['Retail', 'Business', 'Corporate'])

    # Botón de predicción
    submitted = st.form_submit_button("🔍 Analizar Transacción")

if submitted:
    # 1. Crear DataFrame con los datos (¡Nombres de columnas IGUALES al entrenamiento!)
    # Nota: Location la eliminamos en el entrenamiento, así que no la incluimos aquí
    input_data = pd.DataFrame({
        'amount': [amount],
        'transaction_type': [trans_type],
        'account_age': [account_age],
        'customer_segment': [customer_segment],
        'risk_score': [risk_score],
        'hour': [hour]
    })

    # 2. Preprocesamiento (Usamos el mismo scaler/encoder que aprendió el modelo)
    try:
        X_processed = preprocessor.transform(input_data)
        
        # 3. Predicción (Probabilidad)
        probabilidad = model.predict_proba(X_processed)[:, 1][0]
        
        # 4. Decisión basada en el Umbral Optimizado
        es_fraude = probabilidad >= threshold
        
        st.divider()
        st.subheader("Resultado del Análisis")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="Probabilidad de Fraude", value=f"{probabilidad:.2%}")
            st.caption(f"Umbral de alerta: {threshold:.2%}")
            
        with col_res2:
            if es_fraude:
                st.error("🚨 ALERTA: TRANSACCIÓN FRAUDULENTA")
                st.write("Se recomienda bloquear la tarjeta inmediatamente.")
            else:
                st.success("✅ TRANSACCIÓN SEGURA")
                st.write("El riesgo es bajo, se puede proceder.")
                
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")