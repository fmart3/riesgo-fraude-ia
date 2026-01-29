import shap
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import inference  # Importamos para acceder al modelo cargado

# Configuración para servidores sin pantalla (Headless)
# Esto evita el error "TclError" en Docker o servidores Linux
matplotlib.use('Agg') 

def generate_explanation(input_df):
    """
    Genera el gráfico SHAP Waterfall y una explicación textual.
    Devuelve: (imagen_base64, texto_explicativo)
    """
    try:
        # 1. Obtener modelo y preprocesador
        modelo, preprocesador = inference.get_model_assets()
        
        # 2. Transformar los datos para que SHAP los entienda
        X_processed = preprocesador.transform(input_df)
        
        # Intentar recuperar nombres de columnas (útil si usamos OneHotEncoder)
        try:
            feature_names = preprocesador.get_feature_names_out()
        except:
            # Fallback si no tiene nombres (usamos índices genéricos)
            feature_names = [f"Feature {i}" for i in range(X_processed.shape[1])]

        # 3. Calcular valores SHAP
        # Usamos TreeExplainer porque XGBoost/RandomForest son modelos de árboles
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer(X_processed)
        
        # Asignar nombres a los valores para que el gráfico tenga etiquetas
        shap_values.feature_names = feature_names

        # --- A. GENERAR IMAGEN ---
        # Creamos una figura nueva (importante limpiar para no sobreponer gráficos)
        plt.figure(figsize=(8, 5))
        
        # Dibujamos el Waterfall (max_display=7 para que no sea gigante)
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        
        # Guardar en un buffer de memoria (RAM) en lugar de disco
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        plt.close() # Cerramos para liberar memoria
        buf.seek(0)
        
        # Convertir a Base64 (String)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        # --- B. GENERAR TEXTO ---
        # Encontramos la característica que más empujó hacia el fraude
        # shap_values.values[0] es el array de impactos numéricos
        vals = shap_values.values[0]
        max_impact_idx = np.argmax(vals) # Índice del mayor impacto positivo (hacia fraude)
        
        if vals[max_impact_idx] > 0:
            top_feature = feature_names[max_impact_idx]
            explanation_text = f"🤖 Factor IA: El elemento '{top_feature}' fue el principal detonante del riesgo."
        else:
            explanation_text = "🤖 Factor IA: No se detectaron factores fuertes que empujen hacia el fraude."

        return image_base64, explanation_text

    except Exception as e:
        print(f"Error generando SHAP: {e}")
        # En caso de error, devolvemos nulo para no romper toda la app
        return None, "No se pudo generar la explicación gráfica."