import matplotlib
# Configuración para servidores sin pantalla (Headless) - VITAL para Render
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import os
import sys

# ==============================================================================
# BLOQUE DE AJUSTE DE RUTAS (AGREGAR ESTO AL INICIO)
# ==============================================================================
# 1. Obtener la ruta absoluta de la carpeta donde está este script (misc)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Obtener la ruta raíz del proyecto (un nivel arriba de misc)
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

# 3. Agregar la raíz al 'sys.path' para poder importar 'utils'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. CAMBIAR EL DIRECTORIO DE TRABAJO A LA RAÍZ
# Esto es vital: hace que cuando los otros scripts busquen "questions.json" 
# o ".env", los encuentren en la raíz y no busquen en 'misc'.
os.chdir(project_root)
# ==============================================================================


def generate_explanation(data_dict):
    """
    Genera un gráfico de interpretabilidad tipo SHAP-proxy y un texto explicativo
    basado en reglas alineadas con el feature engineering del modelo.

    ⚠️ IMPORTANTE:
    Esta explicación NO representa los pesos internos reales del modelo.
    Es una capa interpretativa diseñada para humanos, basada en patrones
    comúnmente asociados al fraude según el entrenamiento del sistema.
    
    Retorna: (imagen_base64, texto_explicativo)
    """
    try:
        # ------------------------------------------------------------------
        # 1. Extracción de variables crudas (las únicas disponibles en la app)
        # ------------------------------------------------------------------
        monto = float(data_dict.get('amount', 0))
        hora = int(data_dict.get('hour', 0))
        antiguedad = float(data_dict.get('account_age', 0))
        tipo = str(data_dict.get('transaction_type', ''))

        # ------------------------------------------------------------------
        # 2. Construcción de contribuciones (proxy del razonamiento del modelo)
        #    Positivo  -> empuja a FRAUDE
        #    Negativo  -> empuja a SEGURO
        # ------------------------------------------------------------------
        contributions = {}

        # --- A. MONTO (el modelo usa log1p, aquí explicamos en términos relativos)
        if monto > 10000:
            contributions['Monto Relativamente Alto'] = 0.6
        elif monto > 1000:
            contributions['Monto Moderado'] = 0.25
        else:
            contributions['Monto Bajo'] = -0.2

        # --- B. HORA (patrón cíclico: madrugada vs horario típico)
        # Representa indirectamente hour_sin / hour_cos
        if hora in [23, 0, 1, 2, 3, 4, 5]:
            contributions['Patrón Horario Atípico (Madrugada)'] = 0.4
        elif 9 <= hora <= 18:
            contributions['Patrón Horario Habitual'] = -0.3
        else:
            contributions['Patrón Horario Intermedio'] = 0.05

        # --- C. ANTIGÜEDAD (buckets reales del modelo)
        # tenure_group = New / Established / Veteran
        if antiguedad <= 2:
            contributions['Cliente Nuevo (Tenure Bajo)'] = 0.45
        elif antiguedad > 10:
            contributions['Cliente Veterano'] = -0.4
        else:
            contributions['Cliente Establecido'] = 0.1

        # --- D. TIPO DE TRANSACCIÓN
        if tipo in ['ATM Withdrawal', 'Online Purchase']:
            contributions['Canal Transaccional de Mayor Riesgo'] = 0.15
        else:
            contributions['Canal Transaccional Habitual'] = -0.1

        # ------------------------------------------------------------------
        # 3. Gráfico Horizontal (interpretabilidad visual)
        # ------------------------------------------------------------------
        features = list(contributions.keys())
        values = list(contributions.values())
        colors = ['#ff4b4b' if v > 0 else '#1e88e5' for v in values]

        plt.figure(figsize=(8, 6))
        plt.barh(features, values, color=colors, height=0.6)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')

        plt.title('Factores de Influencia en la Decisión', fontsize=12)
        plt.xlabel('Impacto en el Riesgo (← Seguro | Fraude →)', fontsize=10)
        plt.yticks(fontsize=9)

        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.subplots_adjust(left=0.30, right=0.95, top=0.9, bottom=0.15)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=100)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # ------------------------------------------------------------------
        # 4. Generación de texto explicativo (alineado al FE real)
        # ------------------------------------------------------------------
        text_parts = []

        if monto > 1000:
            text_parts.append("el monto es elevado en relación al comportamiento típico del sistema")

        if hora in [23, 0, 1, 2, 3, 4, 5]:
            text_parts.append(f"la operación ocurre en un patrón horario atípico ({hora}:00 hrs)")

        if antiguedad <= 2:
            text_parts.append("la cuenta pertenece a un grupo de clientes nuevos")

        if text_parts:
            explanation_text = (
                "⚠️ Factores de Riesgo Detectados: "
                "La alerta se genera principalmente porque "
                + " y ".join(text_parts)
                + ".\n\n"
                "ℹ️ Nota: Esta explicación es una aproximación interpretativa "
                "basada en patrones generales aprendidos por el modelo, "
                "y no corresponde a los pesos matemáticos internos exactos."
            )
        else:
            explanation_text = (
                "✅ Factores de Seguridad: "
                "La transacción presenta un patrón consistente con el comportamiento esperado "
                "para clientes similares, sin señales claras de riesgo inmediato.\n\n"
                "ℹ️ Nota: Esta explicación resume factores generales y no representa "
                "directamente los cálculos internos del modelo."
            )

        return image_base64, explanation_text

    except Exception as e:
        print(f"Error generando explicación: {e}")
        return None, "No se pudo generar la explicación."
