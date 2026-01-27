import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

# 1. Cargar lo que ya entrenaste
try:
    data = joblib.load('modelo_fraude_final.pkl')
    model = data["modelo"]
    preprocessor = data["preprocesador"]
    
    # Cargar datos originales para probar
    df = pd.read_excel("Fraud_Risk_Dataset.xlsx")
    # Limpieza rápida (igual que antes)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    if 'time_of_transaction' in df.columns:
        df['time_of_transaction'] = pd.to_datetime(df['time_of_transaction'])
        df['hour'] = df['time_of_transaction'].dt.hour
        df = df.drop(columns=['time_of_transaction'])
    
    X = df.drop(columns=['fraudulent', 'transaction_id', 'location'], errors='ignore')
    y = df['fraudulent']
    
    # Preprocesar
    X_processed = preprocessor.transform(X)
    
    # 2. Obtener probabilidades reales
    y_prob = model.predict_proba(X_processed)[:, 1]
    
    # 3. Calcular métricas para distintos umbrales
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    
    # Calcular F1 para cada punto
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Encontrar el mejor F1
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    
    print("\n--- ⚖️ TABLA DE DECISIÓN ---")
    print(f"{'Umbral':<10} | {'Recall (Atrapados)':<20} | {'Precision (Exactitud)':<20}")
    print("-" * 60)
    
    # Mostramos algunos puntos clave
    for i in range(0, len(thresholds), len(thresholds)//10):
        print(f"{thresholds[i]:.4f}     | {recall[i]:.2%}               | {precision[i]:.2%}")
        
    print("-" * 60)
    print(f"💎 MEJOR EQUILIBRIO (Max F1): {best_threshold:.4f}")
    print(f"   -> Atrapas al: {recall[best_idx]:.2%} de los fraudes")
    print(f"   -> Tu precisión es: {precision[best_idx]:.2%}")

except FileNotFoundError:
    print("❌ Falta el archivo .pkl o el Excel en la carpeta.")