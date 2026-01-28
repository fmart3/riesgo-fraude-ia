# /App.py

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
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- CARGAR EL CEREBRO DE LA IA ---
@st.cache_resource
def cargar_modelo():
    try:
        data = joblib.load('modelo_fraude_v1.pkl')
        return data
    except FileNotFoundError:
        st.error("❌ No se encuentra 'modelo_fraude_final.pkl'. Asegúrate de que esté en la misma carpeta.")
        return None

data = cargar_modelo()

if data:
    model = data["modelo"]
    preprocessor = data["preprocesador"]
    
    # Usamos el umbral calibrado (0.080) que guardamos en el pkl
    threshold = data.get("umbral_optimo", 0.08) 

    # --- BARRA LATERAL (INPUTS) ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)
    st.sidebar.title("Panel de Control")
    st.sidebar.markdown("Ingrese los datos de la transacción:")

    with st.sidebar.form("fraude_form"):
        # 1. ELIMINADO EL INPUT DE RISK SCORE
        # El usuario solo ingresa lo que ve en pantalla
        
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0)
        hour = st.slider("Hora (0-23)", 0, 23, 14)
        
        st.markdown("---")
        trans_type = st.selectbox("Tipo Transacción", ['Online Purchase', 'ATM Withdrawal', 'POS Purchase', 'Bank Transfer'])
        account_age = st.number_input("Antigüedad Cta (años)", min_value=0.0, value=3.0)
        customer_segment = st.selectbox("Segmento", ['Retail', 'Business', 'Corporate'])
        
        submitted = st.form_submit_button("🛡️ ANALIZAR RIESGO")

    # --- PANTALLA PRINCIPAL ---
    st.title("Sistema de Detección de Fraude")
    st.markdown(f"Modelo activo: **SVM Calibrado** (Umbral de corte: {threshold})")

    if submitted:
        # 2. INYECTAR RISK SCORE OCULTO
        # El modelo necesita esta columna obligatoriamente.
        # Le damos un valor neutro (ej: 50) o el promedio de tus datos.
        risk_score_default = 50 
        
        input_data = pd.DataFrame({
            'amount': [amount],
            'transaction_type': [trans_type],
            'account_age': [account_age],
            'customer_segment': [customer_segment],
            'risk_score': [risk_score_default], # <--- Aquí está el truco
            'hour': [hour]
        })

        # 3. Preprocesar y Predecir
        try:
            X_processed = preprocessor.transform(input_data)
            
            # Obtener probabilidad (Clase 1 = Fraude)
            probabilidad = model.predict_proba(X_processed)[:, 1][0]
            
            # --- LÓGICA DE NIVELES DE RIESGO ---
            st.divider()
            c1, c2, c3 = st.columns([1, 2, 2])
            
            with c1:
                st.metric("Score de Riesgo", f"{probabilidad:.1%}")
            
            with c2:
                # Usamos el threshold calibrado manualmente
                if probabilidad < threshold:
                    st.success("✅ RIESGO BAJO (Aprobado)")
                    st.markdown("Transacción segura dentro de los parámetros.")
                elif probabilidad < (threshold + 0.15): # Zona gris pequeña
                    st.warning("⚠️ RIESGO MEDIO (Verificar)")
                    st.markdown("Se recomienda enviar SMS de confirmación.")
                else:
                    st.error("🚨 RIESGO ALTO (Bloquear)")
                    st.markdown("**ACCIÓN REQUERIDA:** Bloqueo preventivo.")

            # --- EXPLICABILIDAD (Adaptada para SVM) ---
            with c3:
                st.info("💡 Análisis de Factores")
                
                try:
                    # CORRECCIÓN DE SHAP:
                    # SVM no usa TreeExplainer. Usamos un método genérico o coeficientes.
                    # Para producción rápida, si es lineal, mostramos coeficientes directos.
                    
                    if hasattr(model, 'coef_'):
                        # Si es SVM Lineal (el ganador), esto es más rápido que SHAP
                        coefs = model.coef_[0]
                        # Recuperar nombres
                        try:
                            cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
                            feature_names = ['amount', 'account_age', 'risk_score', 'hour'] + list(cat_names)
                        except:
                            feature_names = [f"F{i}" for i in range(len(coefs))]
                            
                        # Crear dataframe de importancia local (Valor * Coeficiente)
                        # Nota: X_processed es sparse o denso, aseguramos array
                        if hasattr(X_processed, "toarray"):
                            vals = X_processed.toarray()[0]
                        else:
                            vals = X_processed[0]
                            
                        impacto = vals * coefs
                        
                        # Graficar top 5
                        top_indices = np.argsort(np.abs(impacto))[-5:]
                        top_names = [feature_names[i] for i in top_indices]
                        top_values = [impacto[i] for i in top_indices]
                        
                        fig, ax = plt.subplots(figsize=(5, 3))
                        colors = ['red' if x > 0 else 'green' for x in top_values]
                        ax.barh(top_names, top_values, color=colors)
                        ax.set_title("Contribución al Riesgo (Rojo=Peligro)")
                        st.pyplot(fig)
                        
                    else:
                        # Fallback genérico si no es lineal
                        st.text("El modelo es no-lineal, gráfico simplificado no disponible.")
                        
                except Exception as e:
                    st.caption(f"Detalles gráficos no disponibles: {e}")

        except Exception as e:
            st.error(f"Error técnico en el procesamiento: {e}")
            st.write("Verifica que las columnas del Excel coincidan con las del entrenamiento.")

else:
    st.info("⚠️ Esperando archivo 'modelo_fraude_final.pkl'...")