import matplotlib
# Configuración para servidores sin pantalla (Headless) - VITAL para Render
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

def generate_explanation(data_dict):
    """
    Genera un gráfico estilo SHAP y un texto explicativo basado en la lógica del modelo.
    Retorna: (imagen_base64, texto_explicativo)
    """
    try:
        # 1. Extraer datos
        monto = float(data_dict.get('amount', 0))
        hora = int(data_dict.get('hour', 0))
        antiguedad = float(data_dict.get('account_age', 0))
        tipo = str(data_dict.get('transaction_type', ''))
        # segmento = str(data_dict.get('customer_segment', '')) # No lo usamos en la lógica visual actual

        # 2. Calcular "Importancia" (Simulación de pesos del modelo)
        # Valores positivos (Rojo) = Empujan hacia FRAUDE
        # Valores negativos (Azul) = Empujan hacia SEGURO
        
        contributions = {}
        
        # A. Lógica del Monto
        if monto > 10000:
            contributions['Monto Alto'] = 0.6  # Muy riesgoso
        elif monto > 1000:
            contributions['Monto Medio'] = 0.3
        else:
            contributions['Monto Bajo'] = -0.2 # Seguro

        # B. Lógica de la Hora
        if hora < 6 or hora > 22:
            contributions['Horario Nocturno'] = 0.4 # Sospechoso
        elif 9 <= hora <= 18:
            contributions['Horario Oficina'] = -0.3 # Seguro
        else:
            contributions['Horario Neutro'] = 0.05

        # C. Lógica de Cuenta
        if antiguedad == 0:
            contributions['Cuenta Nueva'] = 0.5 # Muy riesgoso
        elif antiguedad > 3:
            contributions['Cuenta Antigua'] = -0.4 # Cliente fiel
        else:
            contributions['Antigüedad Normal'] = 0.1

        # D. Lógica de Tipo
        if tipo in ['ATM Withdrawal', 'Online Purchase']:
            contributions['Canal Riesgoso'] = 0.15
        else:
            contributions['Canal Seguro'] = -0.1

        # 3. Crear Gráfico (Horizontal Bar Chart)
        features = list(contributions.keys())
        values = list(contributions.values())
        colors = ['#ff4b4b' if x > 0 else '#1e88e5' for x in values] # Rojo vs Azul

        # --- CAMBIO AQUÍ: AUMENTAMOS EL TAMAÑO DE LA FIGURA ---
        # Antes era (6, 4), ahora es (10, 6) para que sea más grande y legible.
        plt.figure(figsize=(8, 6)) 
        
        bars = plt.barh(features, values, color=colors, height=0.6)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        # Aumentamos un poco el tamaño de las fuentes también
        plt.title('Factores de Influencia (Interpretabilidad)', fontsize=12)
        plt.xlabel('Impacto en el Riesgo (Izquierda=Seguro, Derecha=Fraude)', fontsize=10)
        plt.yticks(fontsize=9)
        
        # Ajustes estéticos
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        # Ajustamos los márgenes para que las etiquetas largas se lean bien
        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)
        # plt.tight_layout() # A veces tight_layout corta etiquetas en este tamaño, subplots_adjust es mejor aquí

        # 4. Guardar en Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=100) # DPI 100 para buena resolución web
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # 5. Generar Texto Explicativo (Natural Language Generation simple)
        text_parts = []
        riesgo_detectado = False
        if monto > 1000:
            text_parts.append(f"el monto es inusualmente alto (${monto})")
            riesgo_detectado = True
        if hora < 6 or hora > 22:
            text_parts.append(f"la operación ocurre en horario nocturno ({hora}:00)")
            riesgo_detectado = True
        if antiguedad == 0:
            text_parts.append("la cuenta es nueva (0 años de antigüedad)")
            riesgo_detectado = True
        
        if riesgo_detectado:
            explanation_text = "⚠️ Factores de Riesgo: El modelo ha elevado la alerta principalmente porque " + " y ".join(text_parts) + "."
        else:
            explanation_text = "✅ Factores de Seguridad: El comportamiento parece normal. El monto es moderado y los patrones de tiempo/antigüedad no indican riesgo inmediato."

        return image_base64, explanation_text

    except Exception as e:
        print(f"Error generando SHAP: {e}")
        return None, "No se pudo generar la explicación."