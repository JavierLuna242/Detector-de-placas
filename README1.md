# Detección y Reconocimiento de Placas Vehiculares con YOLOv8 + FastAPI
## Objetivo

Este proyecto implementa un sistema de detección automática de placas de vehículos y reconocimiento de caracteres (OCR) utilizando un modelo YOLOv8 entrenado mediante transfer learning y un servicio FastAPI para exponer un endpoint de inferencia.

## Tecnologías usadas

**Python 3.13**

**FastAPI** (framework backend)

**Ultralytics** YOLOv8 (detección de objetos)

**EasyOCR** (reconocimiento de texto)

**OpenCV (cv2)** (procesamiento de imágenes)

**Torch** / torchvision

**vUvicorn** (servidor ASGI)

**SQLite** (opcional) para guardar resultados con timestamp

---
## 1. **Backend - FastAPI en AWS EC2**

Si tienes acceso a Learner LAb, incia el Learner Lab
![alt text](https://raw.githubusercontent.com/adiacla/Deployment-Mobile-Yolo/refs/heads/main/imagenes/learnerlab.JPG))

### 1.1 **Configurar la Instancia EC2 en AWS**

1. En la consola de administración de AWS seleccione el servicio de EC2 (servidor virtual) o escriba en buscar.
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraEC2.JPG?raw=true)

2. Ve a la opción para lanzar la instancia

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irainstancias.JPG?raw=true)

3. Lanza una istancia nueva

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iralanzarinstancia.JPG?raw=true)

4. Inicia una nueva **instancia EC2** en AWS (elige Ubuntu como sistema operativo), puede dejar la imagen por defecto. 

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Instancia%20Ubuntu.PNG?raw=true)

5. Para este proyecto dado que el tamaño del modelo a descargar es grande necesitamos una maquina con más memoria y disco.
   con nuesra licencia tenemos permiso desde un micro lanzar hasta un T2.Large. 


![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iratipodeinstancia.JPG?raw=true)


6. seleccione el par de claves ya creado, o cree uno nuevo (Uno de los dos, pero recuerde guardar esa llave que la puede necesitar, no la pierda)

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraparclaves.JPG?raw=true)

7. Habilite los puertos de shh, web y https, para este proyecto no lo vamos a usar no es necesario, pero si vas a publicar una web es requerido.
   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irfirewall.JPG?raw=true)

8. Configure el almacenamiento. Este proyecto como se dijo requere capacidad en disco. Aumente el disco minimo a **32** GiB.

   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraconfiguraralmacenamiento.JPG?raw=true)

9. Finalmente lance la instancia (no debe presentar error, si tiene error debe iniciar de nuevo). Si todo sale bien, por favor haga click en instancias en la parte superior.

   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/lanzarinstanciafinal.PNG?raw=true)


10. Dado que normalmente en la lista de instancias NO VE la nueva instancia lanzada por favor actualice la pagina Web o en ir a instancias
    
 ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iracutualizarweb.JPG?raw=true)
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irainstancias.JPG?raw=true)

11. Vamos a seleccionar el servidor ec2 lanzado.
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irseleccionarinstancia.JPG?raw=true)

12. Verificar la dirección IP pública y el DNS en el resumen de la instancia
    
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irresumeninstancia.JPG?raw=true)

13. Debido a que vamos a lanzar un API rest debemos habilitar el puerto. Vamos al seguridad

    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraseguirdad.JPG?raw=true)

14. Vamos al grupo de seguridad

   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iragruposeguridad.JPG?raw=true)

   15. Vamos a ir a Editar la regla de entrada

       ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraregladeentrada.JPG?raw=true)

16. Ahora vamos a agregar un regla de entrada para habilitar el puerto, recuerden poner IPV 4

    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iragregarregla.JPG?raw=true)

     


17. Abre un puerto en el grupo de seguridad (por ejemplo, puerto **8080** o si requiere el **8720** así está en alguos ejemplos) para permitir acceso a la API.

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Puerto.PNG?raw=true)

18. Guardemos la regla de entrada.
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irguardarreglas.JPG?raw=true)

19. Ve nuevamente a instancias
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iralanzarinstanciaB.JPG?raw=true)

20. Vamos a conectar con la consola del servidor
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irconectar.JPG?raw=true)

    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irconsola.JPG?raw=true)
    
3. Si no puedes conectarse directamente a la instancia EC2, conectate  con SSH, es decir en la consola de administración de instancia creada hay una opcion de "Conectar", has clic y luego conectar otra vez. Si no puede conectarse puede hacerlo con el SSH:
   

   ```bash
   ssh -i "tu_clave.pem" ubuntu@<tu_ip_ec2>

---

### 1.2 Instalar Dependencias en el Servidor EC2
Una vez dentro de tu instancia EC2, instalar las librerias y complementos como FastAPI y las dependencias necesarias para ello debes crear una carpeta en donde realizaras las instalaciones:

**Ver las carpetas**
 ```bash
ls -la
 ```
**Ver la version de python**
 ```bash
python3 -V
 ```

**Si se requiere, puede actualizar los paquetes**
 ```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0

 ```
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/aptUpdate.PNG?raw=true)


**Si se requiere: Instalar pip y virtualenv**
 ```bash
sudo apt install python3-pip python3-venv
 ```

**Crear la carpeta del proyecto**
 ```bash
mkdir proyecto
 ```

**Accede a tu carpeta**
 ```bash
cd proyecto
 ```

**Crear y activar un entorno virtual**
 ```bash

python3 -m venv venv
source venv/bin/activate
 ```
Recuerda que en el prompt debe obersar que el env debe quedar activo

**Instalar FastAPI, Uvicorn, Joblib, TensorFlow, Python-Multipart, Pillow**
 ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install ultralytics fastapi uvicorn easyocr opencv-python-headless pillow numpy python-multipart

 ```
```bash
yolo check
 ```

### Subir el archivo del modelo
**Subir archivos con scp**
El formato general del comando es:
```bash
scp -i "llavewebici.pem" <archivo_local> ubuntu@<DNS_PUBLICO>:/home/ubuntu/<carpeta_destino>
```
Por ejemplo, si quieres subir:

best.pt

app.py

ejecuta en tu sesion cmd de tu pc:
```
scp -i "llavewebici.pem" best.pt app.py ubuntu@ec2-98-81-166-76.compute-1.amazonaws.com:/home/ubuntu/proyecto/
```
Esto copia ambos archivos al directorio /home/ubuntu/proyecto/ dentro de tu instancia EC2.

### 1.3 Crear la API FastAPI

Crea un archivo app.py en tu instancia EC2 para definir la API que servirá las predicciones.

 ```bash
nano app.py
 ```

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/nanoApp.PNG?raw=true)


## Desarrollo del Backend API
Usaremos FastAPI por su rendimiento y facilidad de uso. El backend aceptará una imagen, la procesará con el modelo Yolo8n con el modelo best.pt y devolverá la predicción.
Puede copiar este codigo en tu editor de nano.


## API: Detector de Placas Vehiculares con YOLOv8 y OCR (FastAPI)

Este servicio expone un API REST basado en FastAPI que combina la detección de objetos con 
el reconocimiento óptico de caracteres (OCR) para identificar placas vehiculares en imágenes.

**Flujo general:**
1. El usuario envía una imagen (JPG o PNG) mediante un `POST /predict/`.
2. El modelo YOLOv8 detecta los objetos en la imagen (por ejemplo, vehículos y placas).
3. Si se identifica una placa, se extrae el recorte y se procesa con EasyOCR.
4. El servicio devuelve:
   - El texto leído de la placa (`placa`),
   - Una lista con las detecciones (etiqueta, confianza, coordenadas, texto detectado),
   - La imagen procesada codificada en Base64 (opcional).

**Endpoints principales:**
- `GET /` → Verifica que el servidor esté activo.
- `POST /predict/` → Realiza la detección y OCR sobre una imagen enviada.


```python
#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + EasyOCR para detección de placas
# Requiere: fastapi uvicorn ultralytics easyocr opencv-python-headless pillow numpy python-multipart

import os
import logging
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")  # Ruta al modelo YOLO
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
RETURN_IMAGE = True  # Devolver imagen con detecciones

# -------------------------
# App init
# -------------------------
app = FastAPI(title="YOLOv8 - Detector de Placas (OCR)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ En producción cambia esto por tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelo y OCR
# -------------------------
logger.info("🔹 Cargando modelo YOLOv8 desde %s ...", MODEL_PATH)
model = YOLO(MODEL_PATH)
logger.info("✅ Modelo YOLOv8 cargado correctamente.")

logger.info("🔹 Inicializando EasyOCR con idiomas: %s", OCR_LANGS)
reader = easyocr.Reader(OCR_LANGS, gpu=False)
logger.info("✅ EasyOCR listo.")

# -------------------------
# Helpers
# -------------------------
def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """Ejecuta EasyOCR sobre un ROI y devuelve texto limpio (mayúsculas y sin espacios)."""
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        result = reader.readtext(roi_rgb)
        if not result:
            return None
        best = max(result, key=lambda x: x[2])
        text = best[1]
        text = "".join(ch for ch in text if ch.isalnum())
        return text.upper() if text else None
    except Exception as e:
        logger.exception("OCR error: %s", e)
        return None


def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    """Convierte imagen BGR a base64 (JPG)."""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')


# -------------------------
# Rutas
# -------------------------
@app.get("/")
def home():
    return {"message": "YOLOv8 + OCR server running"}


@app.post("/predict/")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Recibe una imagen (multipart o base64) y devuelve:
    {
        "success": True,
        "placas": ["ABC123", "XYZ987"],
        "num_placas": 2,
        "image": "...",  # base64 de la imagen procesada
        "message": "OK"
    }
    """
    try:
        logger.info("📩 Petición recibida en /predict/")

        # Leer imagen desde form-data o base64
        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
        elif image_base64:
            # Limpiar base64 (por si tiene prefijo tipo "data:image/jpeg;base64,")
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            image_base64 = image_base64.strip()
            try:
                img_data = base64.b64decode(image_base64 + "===")
            except Exception as e:
                logger.error("❌ Base64 inválido: %s", e)
                return {"error": "Base64 inválido o corrupto."}
            nparr = np.frombuffer(img_data, np.uint8)
        else:
            return {"error": "No se recibió ninguna imagen"}

        # Decodificar imagen
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        logger.info("🧠 Procesando imagen con YOLOv8...")
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        if not results:
            return {"placas": [], "image": None, "success": True, "message": "Sin detecciones"}

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])

        placas_detectadas: List[str] = []

        # Dibujar cajas sobre la imagen
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else None
            label = model.names[cls_id] if cls_id is not None and cls_id < len(model.names) else "objeto"
            conf = confs[i] if len(confs) > i else 0

            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            roi = frame[y1c:y2c, x1c:x2c].copy()

            # OCR solo si el label coincide con "placa"/"plate"/"license"
            if any(k in label.lower() for k in ["placa", "plate", "license"]):
                text_detected = ocr_read_text_from_roi(roi)
                if text_detected:
                    placas_detectadas.append(text_detected)
                    cv2.putText(frame, text_detected, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None

        logger.info("✅ Placas detectadas: %s", placas_detectadas)

        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "image": img_b64,
            "message": "OK" if placas_detectadas else "No se detectaron placas"
        }

    except Exception as e:
        logger.exception("Error en /predict/: %s", e)
        return {"error": str(e)}


# -------------------------
# Ruta alternativa JSON pura
# -------------------------
@app.post("/predict_json/")
async def predict_json(request: Request):
    """Permite enviar imagen como JSON con campo 'image_base64'."""
    try:
        body = await request.json()
        image_base64 = body.get("image_base64")
        if not image_base64:
            return {"error": "No se recibió ninguna imagen"}

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        image_base64 = image_base64.strip()
        img_data = base64.b64decode(image_base64 + "===")
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        if not results:
            return {"placas": [], "image": None, "success": True, "message": "Sin detecciones"}

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])

        placas_detectadas = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else None
            label = model.names[cls_id] if cls_id is not None and cls_id < len(model.names) else "objeto"
            conf = confs[i] if len(confs) > i else 0
            h, w = frame.shape[:2]
            roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

            if any(k in label.lower() for k in ["placa", "plate", "license"]):
                text_detected = ocr_read_text_from_roi(roi)
                if text_detected:
                    placas_detectadas.append(text_detected)
                    cv2.putText(frame, text_detected, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None

        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "image": img_b64,
            "message": "OK" if placas_detectadas else "No se detectaron placas"
        }

    except Exception as e:
        logger.exception("Error en /predict_json/: %s", e)
        return {"error": str(e)}


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("🚀 Iniciando servidor en 0.0.0.0:%s", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
```
Con nano haga
nano app.py
luego copia y pega el codigo
lo guarda con CTRL-X y luego y
Finalmente inicia el servicio del API usando python3 app.py
---
### docs#
La URL http://3.80.229.31:8080/docs (reemplazando la IP por la tuya) es una interfaz automática de documentación interactiva que FastAPI genera por defecto.

Qué puedes hacer en /docs:

Explorar todos los endpoints disponibles

Por ejemplo:

GET / → Prueba que el servidor está corriendo.

POST /predict/ → Permite subir una imagen y ver la respuesta.

**Ver los parámetros esperados y sus tipos**
FastAPI usa type hints de Python para documentar los parámetros (por ejemplo file: UploadFile = File(...)).

**Subir archivos directamente desde el navegador**
En POST /predict/, verás un campo para seleccionar una imagen y probar el modelo sin usar Postman.

**Observar la respuesta estructurada**
Swagger muestra automáticamente la respuesta JSON del servidor (por ejemplo, la placa detectada y la imagen codificada).

**Generar pruebas rápidas o debugging**
Si algo no funciona, /docs te ayuda a verificar si el backend está recibiendo los archivos correctamente.

### 1.5 Ejecutar el Servidor FastAPI

Para ejecutar el servidor de FastAPI, usa Uvicorn:

 ```bash

Recuerda simepre tener el enviroment activo:
source venv/bin/activate
python3 app.py

 ```

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/ServidorAws.PNG?raw=true)

### 1.6 Error en el Servidor

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Error.PNG?raw=true)

Si al momento de ejecutar el servidor te da un error como en el de la anterior imagen en el cual se excede la memoria del sistema utiliza el siguiente comando y vuelve a intentarlo

```bash
sudo sync; sudo sysctl -w vm.drop_caches=3
 ```

## Pueba del Backend
Puedes usar la prueba manual
Descargue esta imagen de prueba a su pc
![](https://raw.githubusercontent.com/adiacla/Deployment-Mobile-Yolo/refs/heads/main/imagenes/carroprueba.JPG)

**Prueba manual:**

Usa herramientas como Postman o cURL para probar la API antes de integrarla con el frontend. Ejemplo de prueba con cURL:

curl -X POST -F "file=@image.jpg" http://ec2-54-164-41-174.compute-1.amazonaws.com:8080/predict/
Espera un JSON como respuesta con las predicciones.

Si vas a utilizar postman entra en el siguiente enlance https://www.postman.com , crea o ingresa a tu cuenta y sigue los siguientes pasos:
1. Dale click en new request

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/NewRequest.PNG?raw=true)
   
2. Poner las siguientes opciones en la request

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/PostRequest.PNG?raw=true)
   
Recuerda que debes poner la URL de tu EC2 acompañado con el :8080 que es el puerto y con el /predict que es el endpoint que queremos probar.

![alt text](https://raw.githubusercontent.com/adiacla/Deployment-Mobile-Yolo/refs/heads/main/imagenes/postmanprueba.JPG)

La API estará disponible en http://<tu_ip_ec2>:8080.

---

# Guía rápida de instalación — React Native en Windows 11

## Requisitos previos

1. **Node.js y npm**  
   Verifica versiones (deben ser ≥ Node 18, npm ≥ 9):
   ```bash
   node -v
   npm -v
   ```
   > Si no los tienes, instala la versión **LTS** desde [https://nodejs.org](https://nodejs.org)

2. **Java Development Kit (JDK)**  
   Instala **OpenJDK 17** o superior:  
   [https://adoptium.net](https://adoptium.net)

3. **Android Studio**  
   Descarga desde  [https://developer.android.com/studio](https://developer.android.com/studio)

   Luego, abre **SDK Manager** y verifica que estén instaladas:
   - ✅ Android SDK Platform **35**
   - ✅ Android SDK Build-Tools **35.0.0**
   - ✅ Android Emulator
   - ✅ Android SDK Command-line Tools (latest)
   - ✅ NDK (Side by side)
   - ✅ CMake
---

## Configurar variables de entorno

Abre “Editar variables de entorno del sistema” → “Variables de usuario”.

Agrega o verifica las siguientes rutas:

| Variable | Valor sugerido |
|-----------|----------------|
| **ANDROID_HOME** | `%LOCALAPPDATA%\Android\Sdk` |
| **Path** | `%ANDROID_HOME%\platform-tools`|
|**Path** |  `%ANDROID_HOME%\emulator`|
|**Path**  | `%ANDROID_HOME%\cmdline-tools\latest\bin` |

---

## Crear un emulador (AVD)

1. Abre **Android Studio → More Actions → Virtual Device Manager**
2. Crea un dispositivo tipo **Pixel 6a / API 33 o superior**
3. Inicia el emulador **antes** de ejecutar la app.

> También puedes conectar tu teléfono Android con la depuración USB activada.

---

# Activar modo desarrollador a tu telefono
Recomendaciones específicas para tu caso

## Activa el modo desarrollador

Sigue los pasos:

Ajustes → Acerca del teléfono → Información de software → Toca 7 veces Número de compilación.

Verás el mensaje “Ahora eres desarrollador”.

## Activa la depuración USB

Ajustes → Opciones de desarrollador → activa Depuración USB.

Conecta el teléfono por cable USB

Usa un cable de datos (no solo de carga).

Cuando aparezca el mensaje “¿Permitir depuración USB?” → pulsa Permitir siempre y Aceptar.

## Verifica la conexión
En la consola (CMD o terminal):
```bash
adb devices
```

Si ves algo como:
```bash
List of devices attached
R58N123ABC	device
````
Todo está correcto.

Si dice “unauthorized”, toca Permitir depuración USB en tu celular.
---

##  Limpieza de instalaciones previas (solo si tuviste errores antes)

```bash
#Nota: Si estas en un proxy, de lo contrario continua verificando la versión del expo Cli

npm config set proxy http://proxyaulas.unab.edu.com:8080
npm config set https-proxy http://proxyaulas.unab.edu.com:8080
npm config list

npx expo --version
npm cache clean --force
```
# Lector de Placas - Instalación y Ejecución

## Requisitos

No necesitas crear nada manualmente en Android Studio. Solo asegúrate de tener:

- **Android Studio** (para los SDKs y emuladores).  
- **Dispositivo físico** con Depuración USB activada.
---

## Crear el proyecto en React Native / Expo

Si vas a crear una app móvil con React Native / Expo (como el siguiente `App.tsx`), **NO necesitas crear un proyecto nuevo en Android Studio desde cero**.  
React Native se encarga de generar todo lo necesario (Gradle, manifest, APK, etc.).

**Crear proyecto con Expo (recomendado):**

```bash
npx create-expo-app lector-placas --template expo-template-blank-typescript
cd lector-placas
```

Reemplaza App.tsx con el código del lector de placas.

**Estructura del proyecto:**
```lua
lector-placas/
├── app
     ├── {tabs}
     ├── index.tsx
├── ...              
├── app.json
├── babel.config.js
├── package.json
├── tsconfig.json
├── node_modules/
└── assets/
    ├── icon.png
    ├── splash.png
    └── ...
```


**1. Crear proyecto**

Cree una carpeta de proyecto
luego


```bash
npx create-expo-app@latest DetectorPlacas
cd DetectorPlacas
```
**Edite el archivo app.json**

```json
{
  "expo": {
    "name": "detector-placas",
    "slug": "detector-placas",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/images/icon.png",
    "scheme": "detectorplacas",
    "userInterfaceStyle": "automatic",
    "newArchEnabled": true,
    "ios": {
      "supportsTablet": true,
      "infoPlist": {
        "NSCameraUsageDescription": "Permite a la app acceder a la cámara para detectar placas.",
        "NSMicrophoneUsageDescription": "Permite a la app usar el micrófono para reproducir voz."
      }
    },
    "android": {
      "adaptiveIcon": {
        "foregroundImage": "./assets/images/android-icon-foreground.png",
        "backgroundColor": "#E6F4FE"
      },
      "permissions": [
        "CAMERA",
        "RECORD_AUDIO",
        "INTERNET",
        "android.permission.CAMERA",
        "android.permission.RECORD_AUDIO"
      ],
      "package": "com.unab.detectorplacas",
      "edgeToEdgeEnabled": true,
      "predictiveBackGestureEnabled": false
    },
    "web": {
      "output": "static",
      "favicon": "./assets/images/favicon.png"
    },
    "plugins": [
      [
        "expo-camera",
        {
          "cameraPermission": "Permite a $(PRODUCT_NAME) acceder a tu cámara.",
          "microphonePermission": "Permite a $(PRODUCT_NAME) acceder a tu micrófono.",
          "recordAudioAndroid": true
        }
      ],
      "expo-router",
      [
        "expo-splash-screen",
        {
          "image": "./assets/images/splash-icon.png",
          "imageWidth": 200,
          "resizeMode": "contain",
          "backgroundColor": "#ffffff",
          "dark": {
            "backgroundColor": "#000000"
          }
        }
      ],
      [
        "expo-build-properties",
        {
          "android": {
            "usesCleartextTraffic": true
          }
        }
      ]
    ],
    "experiments": {
      "typedRoutes": true,
      "reactCompiler": true
    },
    "extra": {
      "router": {},
      "eas": {
        "projectId": "faeae102-4caa-45a3-b563-581b49aea017"
      }
    }
  }
}

```

Instale los paquetes requeridos
``bash
npx expo install expo-camera expo-image-manipulator expo-speech
npm install axios
```

Copia tu código en index.tsx:

```tsx
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';
import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Button,
  Image,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

export default function CameraScreen() {
  const cameraRef = useRef<any>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [ip, setIp] = useState('');
  const [port, setPort] = useState('8080');
  const [image, setImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [plates, setPlates] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const apiUrl = ip && port ? `http://${ip}:${port}` : '';

  useEffect(() => {
    if (!permission) {
      requestPermission();
    }
  }, [permission]);

  const handleCapture = async () => {
    if (!cameraRef.current) return;
    if (!ip) {
      Alert.alert('Error', 'Por favor ingresa la dirección IP del servidor.');
      return;
    }

    try {
      setLoading(true);
      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      setImage(photo.uri);
      setPlates([]);
      setProcessedImage(null);

      const fullUrl = `${apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl}/predict/`;
      console.log('📤 Enviando imagen base64 a:', fullUrl);

      const formBody = new URLSearchParams();
      formBody.append('image_base64', photo.base64);

      const response = await fetch(fullUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Accept: 'application/json',
        },
        body: formBody.toString(),
      });

      if (!response.ok) {
        const text = await response.text();
        console.error('❌ Error HTTP:', response.status, text);
        Alert.alert('Error HTTP', `Código: ${response.status}`);
        Speech.speak('Ocurrió un error al contactar el servidor.');
        return;
      }

      const data = await response.json();
      console.log('📥 Respuesta del servidor:', data);

      if (data?.placas && data.placas.length > 0) {
        const detected = data.placas;
        setPlates(detected);

        if (data.image) {
          setProcessedImage(`data:image/jpeg;base64,${data.image}`);
        }

        const textToSpeak =
          detected.length === 1
            ? `La placa detectada es ${detected[0].split('').join(' ')}`
            : `Se detectaron ${detected.length} placas: ${detected.join(', ')}`;

        if (Platform.OS !== 'web') {
          Speech.speak(textToSpeak, { language: 'es-ES' });
        }
      } else if (data?.placas?.length === 0) {
        if (Platform.OS !== 'web') Speech.speak('No se detectaron placas.');
        Alert.alert('Resultado', 'No se detectaron placas.');
        setPlates([]);
        setProcessedImage(null);
      } else if (data?.error) {
        Alert.alert('Error del servidor', data.error);
        if (Platform.OS !== 'web') Speech.speak('Ocurrió un error en el servidor.');
      } else {
        console.warn('⚠️ Respuesta inesperada:', data);
        Alert.alert('Respuesta inesperada', JSON.stringify(data));
      }
    } catch (error) {
      console.error('❌ Error enviando imagen:', error);
      Alert.alert('Error', 'No se pudo conectar al servidor.');
      if (Platform.OS !== 'web') Speech.speak('No se pudo conectar al servidor.');
    } finally {
      setLoading(false);
    }
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text>Solicitando permisos...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text>Se necesita permiso para usar la cámara.</Text>
        <Button title="Conceder permiso" onPress={requestPermission} />
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.label}>Dirección IP del servidor:</Text>
      <TextInput
        style={styles.input}
        placeholder="Ej: 192.168.1.45"
        value={ip}
        onChangeText={setIp}
      />

      <Text style={styles.label}>Puerto:</Text>
      <TextInput
        style={styles.input}
        placeholder="8080"
        value={port}
        onChangeText={setPort}
        keyboardType="numeric"
      />

      <CameraView ref={cameraRef} style={styles.camera} facing="back" />
      <Button title="Tomar foto" onPress={handleCapture} color="#007AFF" />

      {loading && <ActivityIndicator size="large" color="#007AFF" style={{ marginTop: 20 }} />}

      {image && (
        <View style={styles.imageContainer}>
          <Text style={styles.label}>📷 Imagen capturada:</Text>
          <Image source={{ uri: image }} style={styles.image} />
        </View>
      )}

      {processedImage && (
        <View style={styles.imageContainer}>
          <Text style={styles.label}>🖼️ Imagen procesada por el servidor:</Text>
          <Image source={{ uri: processedImage }} style={styles.image} resizeMode="contain" />
        </View>
      )}

      {plates.length > 0 && (
        <View style={styles.resultContainer}>
          <Text style={styles.label}>🚘 Placas detectadas:</Text>
          {plates.map((p, i) => (
            <Text key={i} style={styles.plateText}>{p}</Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'flex-start',
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  label: {
    fontWeight: 'bold',
    marginBottom: 6,
    color: '#333',
  },
  input: {
    width: '90%',
    height: 40,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 10,
    marginBottom: 10,
    backgroundColor: '#fff',
  },
  camera: {
    width: '100%',
    height: 400,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 16,
  },
  imageContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 200,
    borderRadius: 10,
  },
  resultContainer: {
    marginTop: 20,
    backgroundColor: '#007AFF20',
    padding: 12,
    borderRadius: 8,
    width: '90%',
  },
  plateText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#007AFF',
    textAlign: 'center',
  },
});

```

**ejecutar con npx expo start -c**

Asegúrate de haber guardado todos los cambios en package.json y index.tsx.
Abre tu terminal en el directorio raíz de tu proyecto placas.
Ejecuta el comando:

npx expo start -c

Verás un mensaje que indica que se está limpiando el caché antes de iniciar el packager.
Luego, como de costumbre, aparecerá el código QR.
Escanea el código QR con la aplicación Expo Go en tu teléfono.


# 5. Ejecutar en Android
### Inicia Metro bundler

npx react-native start -c

La app se instalará en tu emulador o dispositivo físico.

Solicitará permisos automáticamente en Android.

6. Notas iOS (opcional)

Agrega en ios/DetectorPlacas/Info.plist:

<key>NSCameraUsageDescription</key>
<string>Necesitamos acceder a la cámara para capturar placas</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Necesitamos acceder a la galería para guardar fotos</string>
<key>NSMicrophoneUsageDescription</key>
<string>Necesitamos usar el micrófono para TTS</string>
---

## Solución de errores comunes

| Error | Solución |
|-------|-----------|
| `SDK location not found` | Revisa la variable `ANDROID_HOME`. |
| `JAVA_HOME not set` | Configura la ruta del JDK (`C:\Program Files\Eclipse Adoptium\jdk-17\`). |
| `Emulator not found` | Abre Android Studio y corre el AVD manualmente. |
| `Build failed` | Ejecuta `cd android && gradlew clean` y vuelve a intentar. |

---

## Recomendaciones

- Usa **VS Code** como editor principal.  
- No instales `react-native-cli` globalmente.  
- Usa siempre `npx react-native ...` para evitar conflictos.  
- Mantén Android Studio y las SDK Tools actualizadas.  


# Prepara tu proyecto Expo para compilación nativa

Tu proyecto actualmente usa Expo Managed Workflow (funciona con expo start), pero para compilar un APK real necesitas usar EAS Build o convertirlo a React Native prebuild (bare).

Vamos con la forma recomendada 

### Opción A – (RECOMENDADA) usar EAS Build

EAS (Expo Application Services) genera el APK o AAB directamente en la nube sin configurar gradle manualmente.

Instálalo:

npm install -g eas-cli


Inicia sesión con tu cuenta Expo:

eas login


Inicializa EAS en tu proyecto:

eas build:configure

### Conecta tu teléfono Android

Conecta el teléfono por USB.

Acepta el aviso de “Permitir depuración USB”.

Verifica que Android Studio detecta el dispositivo:

Abre Android Studio → Device Manager → debe aparecer tu celular.

También puedes verificar con:

adb devices


Si ves tu dispositivo listado, todo está correcto.

### Genera el APK con EAS

Ejecuta:

eas build -p android --profile preview


Esto:

Creará una compilación en la nube de Expo

Y al final te mostrará un enlace para descargar el .apk o .aab.

Si quieres el APK directamente instalable:

eas build -p android --profile preview --local


Este último requiere tener el SDK de Android local y configurado en el PATH.


### Instala el APK en tu celular

Una vez tengas el archivo .apk, puedes:

Instalarlo desde Android Studio → Device File Explorer → Install APK

O más fácil, desde terminal:

adb install path/a/tu/app.apk


El dispositivo mostrará la app instalada con el icono y nombre que definiste:

## Detector de Placas

**(Opcional)** Ejecutar directamente en tu teléfono desde Expo CLI**

Mientras desarrollas, puedes ejecutar:

npm run android


Esto:

Construirá la app nativa temporalmente

La instalará automáticamente en tu teléfono conectado por USB

Abrirá el modo debug (sin necesidad de EAS)

Este comando requiere haber hecho una prebuild del proyecto:

npx expo prebuild

Esto genera las carpetas /android y /ios dentro de tu proyecto.

Luego:

npm run android

La app se compilará localmente usando Gradle y se ejecutará en el dispositivo.

Paso	      Acción	         Comando
1	         Instalar EAS	   npm install -g eas-cli
2	         Configurar	      eas build:configure
3	         Conectar teléfono	adb devices
4	         Construir app	   eas build -p android --profile preview
5	         Instalar en el celular	adb install app.apk

Cuando ya funcione todo, puedes:

Firmar tu app con tu propia key (eas credentials)
Generar un AAB para subirlo a Google Play:
eas build -p android --profile production







