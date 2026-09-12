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


En la carpeta del repositorio abre el archivo app.py y copia el snippert y con nano haga:
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

# Guía Rápida — Instalación y Ejecución del Proyecto con Expo (Windows 11)

> **Proyecto compatible con Expo Go (Android / iOS / Web)**  
> **Sin Android Studio — Sin modificar `app.json` ni `index.tsx`**  
> **Compilación final con EAS Build (.apk / .aab)**

---

## Requisitos Previos

### 1. Node.js y npm
Verifica tus versiones (mínimo **Node 18**, **npm 9**):

```bash
node -v
npm -v
```

Si no los tienes, instala la versión **LTS** desde:  
 [https://nodejs.org](https://nodejs.org)

---

### 2 Instalar Expo CLI y EAS CLI

```bash
npm install -g expo-cli eas-cli
```

Verifica versiones:

```bash
npx expo --version
eas --version
```

---

### 3 (Opcional) Android Studio
Solo es necesario si deseas compilar **localmente** el APK.  
Para desarrollo con **Expo Go** o **web**, puedes **omitirse** completamente.  

---

##  Configuración del Entorno

> Este proyecto usa **Expo Managed Workflow**  
> No requiere carpetas `/android` ni `/ios`.

1. **Crear o clonar el proyecto:**
   ```bash
   npx create-expo-app DetectorPlacas --template expo-template-blank-typescript
   cd DetectorPlacas
   ```

2. **Instalar dependencias necesarias:**
Tu proyecto usa cámara, manipulación de imágenes, voz y plugins nativos de Expo. Instálalos todos:

```bash
npx expo install expo-camera expo-speech expo-image-manipulator
npx expo install expo-router expo-splash-screen expo-build-properties
npm install axios
```

3. **Copiar el código fuente** en `index.tsx` y el `app.json`).

4. **Estructura esperada del proyecto:**
   ```
   DetectorPlacas/
   ├── app/
   │   ├── index.tsx
   ├── assets/
   │   ├── icon.png
   │   ├── splash.png
   ├── app.json
   ├── package.json
   ├── tsconfig.json
   └── ...
   ```

---

## Ejecución Durante el Desarrollo

###  1. Ejecutar en navegador (modo web)
```bash
npx expo start --web
```

### 2. Ejecutar con Expo Go (Android / iOS)
```bash
npx expo start -c
```

Esto abrirá el **Metro Bundler** y mostrará un **código QR**.  
Escanéalo con la app **Expo Go**:

 [Expo Go — Android (Google Play)](https://play.google.com/store/apps/details?id=host.exp.exponent)  
 [Expo Go — iOS (App Store)](https://apps.apple.com/app/expo-go/id982107779)

>  Asegúrate de que tu **celular y tu PC estén conectados a la misma red Wi-Fi**.

---

##  Solución de Problemas Comunes

| Error | Solución |
|-------|-----------|
| `Device unauthorized` | Revisa la conexión USB o red Wi-Fi. |
| La cámara no funciona | Acepta los permisos solicitados por Expo Go. |
| Error HTTP / conexión | Verifica la IP y el puerto del servidor backend. |
| Caché dañado | Ejecuta `npx expo start -c` para limpiar. |
| `SDK location not found` | No aplica en modo Expo Go. Solo relevante para build local. |

---

##  Compilación Final con EAS Build

Cuando la app funcione correctamente en Expo Go, puedes generar tu **APK o AAB** oficial usando **EAS Build**.

---

###  1. Inicia sesión en Expo
```bash
eas login
```

###  2. Configura EAS en tu proyecto
```bash
eas build:configure
```

Esto crea el archivo `eas.json` con perfiles de compilación.

---

###  3. Generar una build en la nube (recomendada)
Compila directamente en los servidores de Expo:
```bash
eas build -p android --profile preview
```

Al finalizar, obtendrás un enlace para descargar el `.apk` o `.aab`.

---

###  4. (Opcional) Build local
Si tienes el SDK de Android instalado localmente:
```bash
eas build -p android --profile preview --local
```

---

###  5. Instalar la app en tu dispositivo
```bash
adb install path/a/tu/app.apk
```

Tu app aparecerá con el ícono y nombre definidos en `app.json`.

---

##  Publicación (opcional)

Para firmar y subir tu app a **Google Play**:
```bash
eas credentials
eas build -p android --profile production
```

---

##  Consejos Finales

✅ No necesitas Android Studio para ejecutar ni probar la app.  
✅ Usa siempre `npx expo start` o `npx expo start --web` en desarrollo.  
✅ No modifiques `app.json` ni `index.tsx` si ya están configurados.  
✅ Usa `expo install` en lugar de `npm install` para dependencias de Expo.  
✅ EAS requiere iniciar sesión con tu cuenta de Expo para generar builds.  

---

##  Resumen Rápido de Comandos

| Acción | Comando |
|--------|----------|
| Crear proyecto | `npx create-expo-app DetectorPlacas --template expo-template-blank-typescript` |
| Instalar dependencias | `npx expo install expo-camera expo-image-manipulator expo-speech && npm install axios` |
| Ejecutar en Expo Go | `npx expo start -c` |
| Ejecutar en navegador | `npx expo start --web` |
| Instalar EAS | `npm install -g eas-cli` |
| Configurar EAS | `eas build:configure` |
| Generar build nube | `eas build -p android --profile preview` |
| Generar build local | `eas build -p android --profile preview --local` |

---
