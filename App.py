import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

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
        return joblib.load('modelo_fraude_final.pkl')
    except FileNotFoundError:
        st.error("⚠️ Falta el archivo 'modelo_fraude_final.pkl'.")
        return None

data = cargar_modelo()

# --- 3. LÓGICA DE NEGOCIO (KILL SWITCH / HARD RULES) ---
def simular_risk_score(amount, account_age_years, hour, transaction_type):
    """
    Define el Risk Score. Si hay alertas rojas, impone un puntaje alto directo.
    Si no, calcula por sumas y restas.
    """
    
    # Definiciones de tiempo
    es_noche = (hour >= 21) or (hour <= 6)      # 9 PM - 6 AM
    es_madrugada = (0 <= hour <= 5)             # 12 AM - 5 AM
    
    # --- REGLAS DE MUERTE SÚBITA (Hard Rules) ---
    # Estas reglas ignoran la antigüedad o el monto, van directo al riesgo alto.
    
    # CASO 1: ATM DE NOCHE -> Riesgo Crítico Inmediato
    if transaction_type == 'ATM Withdrawal' and es_noche:
        return 95  # Casi el máximo posible (Riesgo Crítico)

    # CASO 2: COMPRA ONLINE DE MADRUGADA -> Riesgo Alto
    if transaction_type == 'Online Purchase' and es_madrugada:
        return 80  # Entra en zona de riesgo alto directamente
    
    # --- CÁLCULO ESTÁNDAR (Si no es una alerta roja) ---
    base_score = 50
    
    # Penalización menor por POS de madrugada (fiesta/bar)
    if transaction_type == 'POS Purchase' and es_madrugada:
        base_score += 15
    
    # Antigüedad (Factor mitigante)
    if account_age_years < 0.5: base_score += 25  # Cuenta nueva = riesgo
    elif account_age_years < 1.0: base_score += 15
    elif account_age_years > 5.0: base_score -= 15 # Cliente fiel = menos riesgo
    
    # Montos altos
    if amount > 1000: base_score += 15
    if amount > 5000: base_score += 20
        
    # Limitar entre 0 y 100
    return int(min(max(base_score, 0), 100))

# --- 4. FUNCIÓN SHAP ---
def mostrar_explicacion_shap(modelo, preprocessor, input_df):
    try:
        X_processed = preprocessor.transform(input_df)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"Feature {i}" for i in range(X_processed.shape[1])]

        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer(X_processed)
        shap_values.feature_names = feature_names
        
        single_shap = shap_values[0]
        
        indices_a_mostrar = []
        nombres_numericos = ['amount', 'account_age', 'risk_score', 'hour']
        
        for i, nombre_columna in enumerate(feature_names):
            valor_input = single_shap.data[i]
            es_numerica = any(num in nombre_columna for num in nombres_numericos)
            es_categoria_activa = abs(valor_input) > 0.01 
            
            if es_numerica or es_categoria_activa:
                indices_a_mostrar.append(i)
        
        shap_filtrado = single_shap[indices_a_mostrar]
        
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_filtrado, max_display=10, show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"No se pudo generar SHAP: {e}")

# --- 5. INTERFAZ ---
st.title("🛡️ FraudGuard AI")
st.markdown("### Monitor de Seguridad Transaccional")

if data:
    modelo = data['modelo']
    preprocessor = data['preprocesador']
    umbral_sugerido = data.get('umbral_optimo', 0.5)

    st.sidebar.header("⚙️ Configuración")
    umbral_usuario = st.sidebar.slider("Nivel de Estrictez", 0.0, 1.0, float(umbral_sugerido), 0.01)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("#### 📝 Detalles de Operación")
        transaction_type = st.selectbox("Tipo de Movimiento", 
                                        ['Online Purchase', 'Bank Transfer', 'ATM Withdrawal', 'POS Purchase'])
        
        # Validaciones visuales
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0, step=1.0, format="%.0f")
        account_age = st.number_input("Antigüedad (Años)", 0.0, 100.0, 2.0, 0.1, "%.1f")
        hour = st.slider("Hora", 0, 23, 22)
        customer_segment = st.selectbox("Segmento", ['Retail', 'Business', 'Corporate'])

    with col2:
        st.write("#### 🔍 Resultado del Análisis")
        
        if st.button("ANALIZAR RIESGO"):
            
            # --- VALIDACIÓN ATM ---
            if transaction_type == 'ATM Withdrawal':
                if amount < 20:
                    st.error("⛔ **Error:** Monto mínimo en cajero es $20.")
                    st.stop()
                elif amount % 1 != 0: 
                    st.error("⛔ **Error:** Cajero solo entrega billetes enteros.")
                    st.stop()