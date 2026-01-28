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

# --- 3. LÓGICA DE NEGOCIO (BACKEND) ---
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
    base_score = 50
    base_score += analizar_contexto_hora(hour, transaction_type)
    
    if account_age_years < 0.5: base_score += 25
    elif account_age_years < 1.0: base_score += 15
    elif account_age_years > 5.0: base_score -= 10
        
    if amount > 1000: base_score += 15
    if amount > 5000: base_score += 20
        
    return int(min(max(base_score, 0), 100))

# --- 4. FUNCIÓN SHAP (FILTRADA Y LIMPIA) ---
def mostrar_explicacion_shap(modelo, preprocessor, input_df):
    """
    Genera un gráfico Waterfall filtrando las categorías que valen 0.
    """
    try:
        # A. Transformar datos
        X_processed = preprocessor.transform(input_df)
        
        # B. Obtener nombres de columnas
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"Feature {i}" for i in range(X_processed.shape[1])]

        # C. Calcular valores SHAP
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer(X_processed)
        
        # Asignar nombres al objeto SHAP
        shap_values.feature_names = feature_names
        
        # --- LÓGICA DE FILTRADO ---
        # Obtenemos la explicación para la única fila que tenemos (índice 0)
        single_shap = shap_values[0]
        
        indices_a_mostrar = []
        nombres_numericos = ['amount', 'account_age', 'risk_score', 'hour']
        
        # Recorremos cada variable para decidir si la mostramos
        for i, nombre_columna in enumerate(feature_names):
            valor_input = single_shap.data[i] # El valor real (0 o 1 en categorías)
            
            # Condición 1: Es numérica? (Siempre mostramos numéricas)
            # Buscamos si el nombre limpio está dentro del nombre de la columna (ej: "amount" en "scaler__amount")
            es_numerica = any(num in nombre_columna for num in nombres_numericos)
            
            # Condición 2: Es una categoría activa? (Valor cercano a 1)
            # En OneHotEncoding, si vale > 0.5 es que está presente.
            es_categoria_activa = abs(valor_input) > 0.01 
            
            if es_numerica or es_categoria_activa:
                indices_a_mostrar.append(i)
        
        # D. Crear un nuevo objeto SHAP solo con las columnas filtradas
        shap_filtrado = single_shap[indices_a_mostrar]
        
        # E. Graficar
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_filtrado, max_display=10, show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico SHAP: {e}")

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
        amount = st.number_input("Monto ($)", min_value=0.0, value=150.0, step=10.0)
        account_age = st.number_input("Antigüedad (Años)", 0.0, 100.0, 2.0, 0.1, "%.1f")
        hour = st.slider("Hora", 0, 23, 22)
        transaction_type = st.selectbox("Tipo", ['Online Purchase', 'Bank Transfer', 'ATM Withdrawal', 'POS Purchase'])
        customer_segment = st.selectbox("Segmento", ['Retail', 'Business', 'Corporate'])

    with col2:
        st.write("#### 🔍 Resultado del Análisis")
        
        if st.button("ANALIZAR RIESGO"):
            
            # Back-end simulation
            risk_score_interno = simular_risk_score(amount, account_age, hour, transaction_type)
            
            # Datos para el modelo
            input_data = pd.DataFrame({
                'amount': [amount],
                'account_age': [account_age],
                'risk_score': [risk_score_interno],
                'hour': [hour],
                'transaction_type': [transaction_type],
                'customer_segment': [customer_segment]
            })

            try:
                # Predicción
                input_processed = preprocessor.transform(input_data)
                probabilidad = modelo.predict_proba(input_processed)[0, 1]
                prob_pct = probabilidad * 100
                es_fraude = probabilidad >= umbral_usuario

                # --- RESULTADO VISUAL ---
                st.write("")
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob_pct,
                        number = {'suffix': "%"},
                        title = {'text': "Probabilidad"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if es_fraude else "#00CC96"},
                            'steps': [
                                {'range': [0, 30], 'color': "#E5F5F9"}, 
                                {'range': [30, 70], 'color': "#FFF8DD"},
                                {'range': [70, 100], 'color': "#FFE4E1"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': umbral_usuario * 100}
                        }
                    ))
                    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=200)
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    if prob_pct < 30:
                        nivel = "BAJO"; color = "#d4edda"; txt = "#155724"
                    elif prob_pct < 70:
                        nivel = "MEDIO"; color = "#fff3cd"; txt = "#856404"
                    else:
                        nivel = "ALTO"; color = "#f8d7da"; txt = "#721c24"

                    st.markdown(f"""
                        <div style="background-color: {color}; color: {txt}; padding: 15px; border-radius: 10px; text-align: center; margin-top: 40px;">
                            <h3 style="margin:0;">Riesgo: {nivel}</h3>
                            <p style="margin:0;">Probabilidad: {prob_pct:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if es_fraude:
                        st.error("🚨 **BLOQUEAR TRANSACCIÓN**")
                    else:
                        st.success("✅ **APROBAR TRANSACCIÓN**")

                # --- SHAP WATERFALL ---
                st.divider()
                st.subheader("🤖 Análisis de Factores Clave")
                st.markdown("**¿Qué variables empujaron esta decisión?**")
                
                with st.spinner("Calculando impacto de variables..."):
                    mostrar_explicacion_shap(modelo, preprocessor, input_data)

            except Exception as e:
                st.error("Error técnico al procesar.")
                st.caption(e)