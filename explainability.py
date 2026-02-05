import matplotlib
# Configuración para servidores sin pantalla (Headless)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import os

# NOTA: Se eliminó la importación de google.generativeai para no depender de la API

def generate_explanation(processed_data):
    """
    Genera el gráfico SHAP (simulado visualmente) y retorna Base64 + Texto.
    """
    try:
        feature_names = ['Monto', 'Hora', 'Antigüedad', 'Tipo', 'Segmento']
        
        monto = processed_data['amount'].iloc[0]
        hora = processed_data['hour'].iloc[0]
        antiguedad = processed_data['account_age'].iloc[0]

        # Lógica visual para el gráfico (Simulación de SHAP)
        # Esto hace que las barras rojas/verdes tengan sentido con los datos
        shap_vals = [
            (monto / 500) if monto > 150 else -0.2,    # Monto alto = Riesgo
            0.4 if (hora < 6 or hora > 22) else -0.1,  # Hora nocturna = Riesgo
            -0.5 if antiguedad > 3 else 0.2,           # Cuenta antigua = Seguro, Nueva = Riesgo
            0.1,                                       # Tipo (neutro)
            0.05                                       # Segmento (neutro)
        ]

        # Crear Gráfico con Matplotlib
        plt.figure(figsize=(8, 4))
        colors = ['#ef4444' if x > 0 else '#22c55e' for x in shap_vals] # Rojo (Riesgo) / Verde (Seguro)
        plt.barh(feature_names, shap_vals, color=colors)
        plt.axvline(0, color='black', linewidth=0.8) # Línea central
        plt.xlabel("Impacto en Riesgo (Izquierda=Seguro, Derecha=Fraude)")
        plt.title("Factores de Influencia (SHAP)")
        plt.tight_layout()

        # Guardar en memoria
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)

        # Convertir a Base64 para enviarlo al HTML
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        texto_resumen = f"Monto (${monto}) y Hora ({hora}:00)"

        return image_base64, texto_resumen

    except Exception as e:
        print(f"Error generando SHAP: {e}")
        return None, "Error generando gráfico."

def generar_explicacion_llm(datos_dict, shap_values, resumen_texto):
    try:
        # Extraemos variables para facilitar la lectura
        monto = float(datos_dict.get('amount', 0))
        hora = int(datos_dict.get('hour', 0))
        antiguedad = float(datos_dict.get('account_age', 0))
        tipo = datos_dict.get('transaction_type', 'Desconocido')

        # --- LÓGICA DE REGLAS (Expert System) ---
        
        # 1. Analizar severidad
        es_monto_alto = monto > 500
        es_hora_inusual = (hora < 6 or hora > 23)
        es_cuenta_nueva = antiguedad < 0.5 # Menos de 6 meses

        # 2. Construir la narrativa
        conclusiones = []

        if es_monto_alto and es_hora_inusual:
            return (f"🚨 ALERTA CRÍTICA: Se detectó una transacción de alto valor (${monto}) "
                    f"en un horario atípico ({hora}:00). Debido a la combinación de estos factores, "
                    "el sistema recomienda bloquear preventivamente.")
        
        if es_monto_alto:
            conclusiones.append(f"el monto es inusualmente alto para el perfil (${monto})")
        
        if es_hora_inusual:
            conclusiones.append(f"la operación se realizó de madrugada ({hora}:00)")
        
        if es_cuenta_nueva:
            conclusiones.append("la cuenta es reciente")

        # 3. Resultado final
        if conclusiones:
            razones = ", ".join(conclusiones)
            # Capitalizamos la primera letra
            return f"⚠️ Precaución: La operación presenta riesgo medio dado que {razones}. Se sugiere validación adicional con el cliente."
        else:
            return (f"✅ Operación Segura: La transacción de ${monto} en {tipo} coincide con los "
                    "patrones habituales de comportamiento. No se detectan anomalías de fraude.")

    except Exception as e:
        return f"Nota: Análisis no disponible ({str(e)})."