import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Detector de Fraude IA", page_icon="🛡️", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .risk-low { color: #2ecc71; font-weight: bold; }
    .risk-med { color: #f1c40f; font-weight: bold; }
    .risk-high { color: #e74c3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- CARGAR EL CEREBRO DE LA IA ---
@st.cache_resource
def cargar_modelo():
    try:
        data = joblib.load('modelo_fraude_final.pkl')
        return data
    except FileNotFoundError:
        st.error("❌ No se encuentra 'modelo_fraude_final.pkl'.")
        return None

data = cargar_modelo()

if data:
    model = data["modelo"]
    preprocessor = data["preprocesador"]
    # Si calculaste un umbral nuevo, úsalo aquí. Si no, usa el del pkl
    threshold = data.get("umbral_optimo", 0.35) 

    # --- BARRA LATERAL (INPUTS) ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)
    st.sidebar.title("Panel de Control")
    st.sidebar.markdown("Ingrese los datos de la transacción:")

    with st.sidebar.form("fraude_form"):
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0)
        hour = st.slider("Hora (0-23)", 0, 23, 14)
        #risk_score = st.slider("Risk Score Interno", 0, 100, 20)
        
        st.markdown("---")
        trans_type = st.selectbox("Tipo Transacción", ['Online Purchase', 'ATM Withdrawal', 'POS Purchase', 'Bank Transfer'])
        account_age = st.number_input("Antigüedad Cta (años)", min_value=0.0, value=3.0)
        customer_segment = st.selectbox("Segmento", ['Retail', 'Business', 'Corporate'])
        
        submitted = st.form_submit_button("🛡️ ANALIZAR RIESGO")

    # --- PANTALLA PRINCIPAL ---
    st.title("Sistema de Detección de Fraude")
    st.markdown("Dashboard de análisis en tiempo real basado en **XGBoost**.")

    if submitted:
        # 1. Preparar datos
        input_data = pd.DataFrame({
            'amount': [amount],
            'transaction_type': [trans_type],
            'account_age': [account_age],
            'customer_segment': [customer_segment],
            #'risk_score': [risk_score],
            'hour': [hour]
        })

        # 2. Preprocesar y Predecir
        try:
            X_processed = preprocessor.transform(input_data)
            probabilidad = model.predict_proba(X_processed)[:, 1][0]
            
            # --- LÓGICA DE NIVELES DE RIESGO ---
            st.divider()
            c1, c2, c3 = st.columns([1, 2, 2])
            
            with c1:
                st.metric("Probabilidad de Fraude", f"{probabilidad:.1%}")
            
            with c2:
                if probabilidad < threshold:
                    st.success("✅ RIESGO BAJO")
                    st.markdown("Transacción **Aprobada** automáticamente.")
                elif probabilidad < 0.70: # Entre umbral y 70%
                    st.warning("⚠️ RIESGO MEDIO")
                    st.markdown("Se recomienda **Revisión Manual**.")
                else:
                    st.error("🚨 RIESGO ALTO")
                    st.markdown("BLOQUEO INMEDIATO sugerido.")

            # --- EXPLICABILIDAD (SHAP) ---
            with c3:
                st.info("💡 ¿Por qué este resultado?")
                st.caption("Variables que más influyeron:")
            
            # Generar gráfico SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)
            
            # Recuperar nombres de columnas para que se vea bonito
            try:
                feature_names = ['amount', 'account_age', 'risk_score', 'hour'] + \
                                list(preprocessor.named_transformers_['cat'].get_feature_names_out())
            except:
                feature_names = [f"Feature {i}" for i in range(X_processed.shape[1])]

            # Graficar
            fig, ax = plt.subplots(figsize=(5, 3))
            shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar", show=False, max_display=5)
            st.pyplot(fig)
            
            # Explicación textual dinámica
            top_index = np.abs(shap_values[0]).argmax()
            top_feature = feature_names[top_index]
            impacto = "aumentó" if shap_values[0][top_index] > 0 else "redujo"
            
            st.markdown(f"> La variable **{top_feature}** fue la que más influyó y **{impacto}** el riesgo.")

        except Exception as e:
            st.error(f"Error técnico: {e}")
else:
    st.info("Esperando carga del modelo...")