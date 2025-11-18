#  Configuración e instalación MLflow
MLflow es una herramienta muy útil para gestionar experimentos de machine learning: registro de métricas, parámetros, modelos y hasta despliegues.
## 1) Preparar el entorno(macOS)
Usa python3 (no python), también se crea un entorno virtual

```bash
#Creo el entorno virtual - mlflow
python3 --version
which python3
brew install python
python3 -m venv .venv-mlflow
source .venv-mlflow/bin/activate # Activo el aentorno virtual
deactivate #Desactivo el entorno virtual

```
### 2) Instalar MLflow y dependencias mínimas
```bash
pip install mlflow # Hacer la inslación normal
pip install mlflow scikit-learn pandas joblib

```
### 3) Configuración del puerto y el servidor local
```bash
#Definir el servidor y puerto según la configuración de la máquina.
mlflow server --host 127.0.0.1 --port 8080
mlflow server --host 127.0.0.1 --port 5000
# Para terminar el proceso de servicio en terminal, se escribe con el comando:
Control + c
```
### 4)Configuración de un gestor de sesiones de terminal (screen)
Es un programa que permite, crear múltiples terminales virtuales dentro de una sola ventana de terminal.
```bash
#DInstalación  y configuración screen
screen -S <Nombre- proceso> #Crear una nueva sesión, para realizar cualquier proceso o tarea, que necesitamos que siga ejecutandose en el servidor.
screen -ls #Listar los procesos actuales
screen -r <Nombre- proceso> #Acceder un proceso
Ctrl + A luego D #Salir de sesión, sin terminal el proceso.
exit # Termino el proceso
screen -r python-server -X quit #Cerrar una sesión particular, desde la terminal origen.

```
### 5) Ejecutar el script

```bash
#Ejecuto el script, para entrenar los datos del modelo y definir las métricas

python3 train_mlflow.py
```


# 6) En resumen

 1. Carga un dataset limpio.
 2. Separar variables.
 3. Entrena un modelo de regresión logística.
 4. Calcula precisión.
 5. Guarda parámetros, métricas y modelo en MLFlow.
 6. Registra y versiona el modelo en el Model Registry.
 7. Carga el modelo desde Mlflow. 
 8. Genera predicciones y tabla comparativa.  



