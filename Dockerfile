# 1. Imagen base ligera de Python
FROM python:3.11-slim

# 2. Definir directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar herramientas de compilación del sistema
# (Necesarias para compilar SHAP y XGBoost en Linux)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar y procesar dependencias primero (para aprovechar caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar los archivos del código fuente
COPY . .

# 6. Exponer el puerto estándar de FastAPI
EXPOSE 8000

# 7. Comando de arranque: Uvicorn servidor de producción
# host 0.0.0.0 permite conexiones externas (desde tu PC al contenedor)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]