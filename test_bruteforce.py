import requests
import json
import itertools

# URL de tu API local
#URL = "https://fraudgaurd-ai.onrender.com/analyze"
URL = "http://127.0.0.1:8000/analyze"

# 1. Definimos las variables que el modelo conoce (según tu output anterior)
tipos = ['ATM Withdrawal', 'Bank Transfer', 'Online Purchase', 'POS Purchase']
segmentos = ['Business', 'Corporate', 'Retail']
horas = [14, 3] # Una hora normal (2pm) y una de madrugada (3am)
montos = [50, 950000] # Un monto hormiga y un robo millonario

# Generamos todas las combinaciones posibles
combinaciones = list(itertools.product(tipos, segmentos, horas, montos))

print(f"🚀 Iniciando sondeo con {len(combinaciones)} combinaciones...")
print("-" * 60)
print(f"{'TYPE':<20} {'SEGMENT':<12} {'HOUR':<5} {'AMOUNT':<10} | {'PROB':<8} {'RESULT'}")
print("-" * 60)

fraud_found = False

for t, s, h, a in combinaciones:
    payload = {
        "amount": a,
        "transaction_type": t,
        "account_age": 0.0, # Asumimos cuenta nueva siempre (más riesgoso)
        "hour": h,
        "customer_segment": s
    }
    
    try:
        response = requests.post(URL, json=payload)
        data = response.json()
        prob = data.get("probability_percent", 0.0)
        
        # Filtro visual: Solo mostramos si hay ALGÚN riesgo (> 1%)
        if prob > 1.0:
           print(f"🚨 {t:<20} {s:<12} {h:<5} ${a:<9} | {prob:.2f}%   ¡FRAUDE DETECTADO!")
           fraud_found = True
        else:
           print(f"   {t:<20} {s:<12} {h:<5} ${a:<9} | {prob:.1f}%")
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        break

print("-" * 60)
if not fraud_found:
    print("😱 El modelo devolvió 0.0% para TODO. Algo está mal en el entrenamiento.")
else:
    print("✅ Diagnóstico completado. Usa las combinaciones marcadas con 🚨.")