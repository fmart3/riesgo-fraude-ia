import os
import sys
import shutil

# --- CARGA DE SECRETOS ---
try:
    from dotenv import load_dotenv
    load_dotenv() 
    print("🔐 Secretos cargados desde .env.")
except ImportError:
    print("⚠️ 'python-dotenv' no instalado. Usando variables de entorno del sistema.")

# --- CONFIGURACIÓN ---
# 🚨 IMPORTANTE: Estos deben coincidir con lo que usaste en Databricks
CATALOGO = "phishing"      # <--- Tu catálogo
ESQUEMA = "default"        # <--- Tu esquema
NOMBRE_MODELO = "Fraud_Detector_Production"
FULL_MODEL_NAME = f"{CATALOGO}.{ESQUEMA}.{NOMBRE_MODELO}"

ALIAS = "Champion"         # La etiqueta que le pusimos al ganador
OUTPUT_FILE = "fraude.pkl" # Nombre del archivo local

print("--- ACTUALIZADOR DE MODELO (MODO UNITY CATALOG) ---")

try:
    import mlflow
    import mlflow.sklearn
    import joblib
    print("✅ Librerías cargadas.")
except ImportError:
    print("❌ Faltan librerías. Ejecuta: pip install mlflow pandas python-dotenv joblib")
    sys.exit(1)

def download_champion_model():
    # 1. Validar Credenciales
    token = os.environ.get("DATABRICKS_TOKEN")
    host = os.environ.get("DATABRICKS_HOST")
    
    if not token or not host:
        print("❌ ERROR: Faltan credenciales DATABRICKS_HOST o DATABRICKS_TOKEN.")
        return

    print(f"🔄 Conectando a Databricks ({host})...")
    
    # 2. Configurar MLflow para Unity Catalog
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc") # <--- CLAVE: Activar modo UC
    
    try:
        # 3. Construir la URI del Modelo Champion
        # Formato: models:/<catalogo>.<esquema>.<modelo>@<alias>
        model_uri = f"models:/{FULL_MODEL_NAME}@{ALIAS}"
        
        print(f"🔍 Buscando modelo certificado: {model_uri}")
        print(f"📥 Descargando Pipeline completo... (esto incluye el preprocesador)")
        
        # 4. Cargar el Pipeline directamente desde Databricks
        loaded_pipeline = mlflow.sklearn.load_model(model_uri)
        
        # 5. Guardar en disco local
        joblib.dump(loaded_pipeline, OUTPUT_FILE)
        
        print("-" * 50)
        print(f"🎉 ¡ÉXITO! Se ha descargado la versión '{ALIAS}' de Unity Catalog.")
        print(f"📂 Archivo guardado: {OUTPUT_FILE}")
        print("   (Ahora tu app puede recibir datos crudos, el pipeline los transformará)")
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ ERROR DE DESCARGA:\n{e}")
        print("\nPosibles causas:")
        print("1. ¿Pusiste el nombre correcto del catálogo ('phishing')?")
        print("2. ¿Tu token tiene permisos de lectura sobre ese modelo?")

if __name__ == "__main__":
    download_champion_model()