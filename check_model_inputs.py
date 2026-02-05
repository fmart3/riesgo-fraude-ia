import joblib
import pandas as pd

# Cargar el modelo
try:
    pipeline = joblib.load("fraude.pkl")
    print("✅ Modelo cargado correctamente.\n")

    # Intentar obtener los nombres de características de entrada
    if hasattr(pipeline, "feature_names_in_"):
        print("🚨 EL MODELO ESPERA ESTAS COLUMNAS EXACTAS (Copia esto):")
        print("-" * 50)
        print(pipeline.feature_names_in_)
        print("-" * 50)
    else:
        # Si es una versión vieja de sklearn, intentamos mirar el primer paso
        print("⚠️ No se encontró metadata directa. Buscando en el preprocesador...")
        try:
            print(pipeline.named_steps['preprocessor'].transformers_)
        except:
            print("No se pudo inspeccionar. Necesitamos ver tu código de entrenamiento.")

except Exception as e:
    print(f"❌ Error: {e}")