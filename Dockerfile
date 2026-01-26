# 1. Usamos una imagen base de Python ligera
FROM python:3.9-slim

# 2. Directorio de trabajo en el contenedor
WORKDIR /app

# 3. Copiamos los archivos necesarios
COPY requirements.txt .
COPY app.py .
COPY modelo_fraude_v1.pkl .

# 4. Instalamos dependencias
# (Añadimos build-essential para compilar algunas librerías si hace falta)
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exponemos el puerto de Streamlit
EXPOSE 8501

# 6. Comando para iniciar la app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]