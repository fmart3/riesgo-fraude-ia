import requests
import json
import itertools
import time

# URL de tu API en producción
# NOTA: Si usas el plan gratuito de Render, la primera petición puede tardar 1 minuto en despertar.
#URL = "https://fraudgaurd-ai.onrender.com/analyze"
URL = "http://127.0.0.1:8000/analyze"

# 1. Definimos las variables que el modelo conoce
tipos = ['ATM Withdrawal', 'Bank Transfer', 'Online Purchase', 'POS Purchase']
segmentos = ['Business', 'Corporate', 'Retail']
horas = [14, 3]       # 2 PM (Día) vs 3 AM (Noche)
montos = [50, 950000] # Monto bajo vs Robo millonario

# Generamos todas las combinaciones posibles
combinaciones = list(itertools.product(tipos, segmentos, horas, montos))

print(f"🚀 Iniciando sondeo masivo a: {URL}")
print(f"📊 Total de combinaciones a probar: {len(combinaciones)}")
print("-" * 85)
print(f"{'TYPE':<20} {'SEGMENT':<12} {'HOUR':<5} {'AMOUNT':<10} | {'PROB':<8} {'RESULT'}")
print("-" * 85)

fraud_found = False

for t, s, h, a in combinaciones:
    payload = {
        "amount": a,
        "transaction_type": t,
        "account_age": 0.0, # Asumimos cuenta nueva (riesgo alto)
        "hour": h,
        "customer_segment": s
    }
    
    try:
        response = requests.post(URL, json=payload, timeout=10) # Timeout de 10s
        
        if response.status_code == 200:
            data = response.json()
            prob = data.get("probability_percent", 0.0)
            
            # --- AQUÍ ESTÁ EL FILTRO QUE PEDISTE ---
            if prob > 50.0:
                print(f"🚨 {t:<20} {s:<12} {h:<5} ${a:<9} | {prob:.2f}%   ¡FRAUDE DETECTADO!")
                fraud_found = True
            else:
                # Opcional: Imprimir los seguros en color neutro para ver que el script avanza
                print(f"✅ {t:<20} {s:<12} {h:<5} ${a:<9} | {prob:.2f}%   Seguro")
        else:
            print(f"❌ Error del servidor: {response.status_code}")

    except Exception as e:
        print(f"⚠️ Error de conexión (¿La API está despierta?): {e}")
        break

print("-" * 85)
if not fraud_found:
    print("😱 El modelo no detectó nada sobre el 50%.")
else:
    print("🏁 Diagnóstico completado.")