import os
import sys

# --- NUEVO: Carga de secretos desde .env ---
try:
    from dotenv import load_dotenv
    load_dotenv() # <--- Esto lee el archivo .env y carga las variables
    print("🔐 Secretos cargados desde .env localmente.")
except ImportError:
    print("⚠️ No se tiene 'python-dotenv'. Si estás en Docker o Producción, asegúrate de tener las variables de entorno configuradas en el sistema.")

# --- CONFIGURACIÓN ---
USER_EMAIL = "felipe.martinez@cybertrust.one" 
EXPERIMENT_NAME = "FraudGuard_Project_Final"
OUTPUT_FILE = "fraude.pkl"

print("--- ACTUALIZADOR DE MODELO (Automático con .env) ---")

try:
    import mlflow
    import mlflow.sklearn
    import joblib
    print("✅ Librerías cargadas.")
except ImportError:
    print("❌ Faltan librerías. Ejecuta: pip install mlflow pandas python-dotenv")
    sys.exit(1)

def download_latest_model():
    # 1. Validar Credenciales (Ahora vienen del .env)
    token = os.environ.get("DATABRICKS_TOKEN")
    host = os.environ.get("DATABRICKS_HOST")
    
    if not token or not host:
        print("❌ ERROR: Faltan credenciales.")
        print("   Asegúrate de haber creado el archivo .env con DATABRICKS_HOST y DATABRICKS_TOKEN.")
        return

    print(f"🔄 Conectando a Databricks ({host})...")
    mlflow.set_tracking_uri("databricks")
    
    try:
        # 2. Encontrar el Experimento
        experiment_path = f"/Users/{USER_EMAIL}/{EXPERIMENT_NAME}"
        print(f"🔍 Buscando experimento: {experiment_path}")
        
        experiment = mlflow.get_experiment_by_name(experiment_path)
        if experiment is None:
            print("❌ No se encontró el experimento. Verifica el email y el nombre.")
            return

        # 3. Buscar la ÚLTIMA corrida exitosa
        print("🔍 Buscando el último entrenamiento exitoso...")
        df_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if df_runs.empty:
            print("❌ No se encontraron corridas exitosas.")
            return
            
        latest_run_id = df_runs.iloc[0].run_id
        print(f"   ✅ Última corrida encontrada. ID: {latest_run_id}")

        # 4. Descargar
        model_uri = f"runs:/{latest_run_id}/model"
        print(f"📥 Descargando modelo... (puede tardar un poco)")
        
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # 5. Guardar
        joblib.dump(loaded_model, OUTPUT_FILE)
        
        print("-" * 50)
        print(f"🎉 ¡LISTO! {OUTPUT_FILE} actualizado correctamente.")
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO:\n{e}")

if __name__ == "__main__":
    download_latest_model()