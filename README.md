# 🚘 Sistema de Detección y Reconocimiento de Placas (ALPR / ANPR)

Sistema de Inteligencia Artificial para la detección automática y lectura óptica de caracteres (OCR) en placas vehiculares mediante **YOLOv8** y **EasyOCR / PyTesseract**. El proyecto incluye una API REST robusta construida con **FastAPI** en el Backend y una interfaz de usuario interactiva en el Frontend web con **React + Vite**.

---

## 🛠️ Tecnologías Utilizadas

- **Backend**: Python 3.10+, FastAPI, YOLOv8 (Ultralytics), EasyOCR, PyTesseract, OpenCV, PyTorch.
- **Frontend**: React 18, Vite, TailwindCSS, Lucide Icons.
- **Despliegue & Orquestación**: Docker, Docker Compose, AWS EC2.
- **Entrenamiento**: Google Colab (`reentrenamiento_yolo_colab.ipynb`).

---

## 📁 Estructura del Proyecto

```text
Deployment-Mobile-Yolo/
├── 🔵 backend/                        # API de FastAPI y Modelo de Inteligencia Artificial
│   ├── models/
│   │   └── best.pt                    # Pesos entrenados del modelo YOLOv8
│   ├── src/api/
│   │   └── main.py                    # Lógica de endpoints, preprocesamiento ROI y OCR
│   ├── Dockerfile                     # Dockerfile para la API Python
│   └── requirements.txt               # Dependencias de Python
│
├── 🟢 frontend/                       # Aplicación Web (React + Vite)
│   ├── src/
│   │   ├── App.jsx                    # Componente principal con visor de cámara, video e imágenes
│   │   └── index.css                  # Estilos visuales
│   ├── Dockerfile                     # Dockerfile del cliente Web
│   ├── package.json                   # Dependencias de Node.js
│   └── vite.config.js                 # Configuración del servidor de desarrollo Vite
│
├── 🐳 docker-compose.yml              # Configuración para ejecutar Backend + Frontend en Docker
├── 📓 reentrenamiento_yolo_colab.ipynb# Notebook de entrenamiento en Google Colab
└── 📄 README.md                       # Documentación general del proyecto
```

---

## 🌟 Características Clave

1. **Detección de Objetos en Tiempo Real (YOLOv8)**:
   - Identifica y ubica la región de interés (ROI) correspondiente a la placa del vehículo.
   - Filtro por relación de aspecto (ancho/alto) para descartar falsos positivos.

2. **Pipeline de Preprocesamiento de Imágenes y OCR**:
   - Generación de múltiples variantes de imagen (RGB, Grayscale CLAHE, Binarización Otsu y Filtro Bilateral).
   - Inyección de márgenes/bordes sintéticos para mejorar la tasa de lectura de caracteres por Tesseract / EasyOCR.
   - Corrección y validación de formatos de placas vehiculares (e.g. estándar colombiano `ABC123` / `ABC12D`).

3. **Cálculo de Porcentaje de Acierto**:
   - Integración de confianza del detector YOLOv8 con la coincidencia estricta de patrones OCR para estimar la precisión final.

4. **Interfaz Web Amigable**:
   - Subida de archivos de imagen y video.
   - Streaming en tiempo real desde la cámara web.
   - Renderizado dinámico de Bounding Boxes y métricas sobre la imagen.

---

## 🚀 Instalación y Ejecución

### Opción 1: Ejecución con Docker Compose (Recomendado)

Asegúrate de tener instalados [Docker](https://www.docker.com/) y [Docker Compose](https://docs.docker.com/compose/).

1. Clona este repositorio:
   ```bash
   git clone https://github.com/JavierLuna242/Detector-de-placas.git
   cd Detector-de-placas
   ```

2. Inicia los servicios con Docker Compose:
   ```bash
   docker compose up --build
   ```

3. Accede a las aplicaciones desde tu navegador:
   - **Frontend Web**: [http://localhost:3000](http://localhost:3000)
   - **Backend API**: [http://localhost:8080](http://localhost:8080)
   - **Documentación Interactiva (Swagger)**: [http://localhost:8080/docs](http://localhost:8080/docs)

---

### Opción 2: Ejecución Manual en Entorno Local

#### 1. Backend (FastAPI)
```bash
cd backend

# (Opcional) Crear y activar entorno virtual
python -m venv venv
# En Windows: venv\Scripts\activate | En Linux/Mac: source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el servidor
python src/api/main.py
```
El servidor backend escuchará por defecto en `http://localhost:8080`.

#### 2. Frontend (React + Vite)
```bash
cd frontend

# Instalar dependencias de Node
npm install

# Iniciar servidor de desarrollo
npm run dev
```
La aplicación web se abrirá en `http://localhost:3000` (o el puerto configurado por Vite).

---

## 📡 Endpoints de la API Backend

| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Estado del servidor y modelo cargado. |
| `POST` | `/predict/` | Recibe imagen via `multipart/form-data` o `image_base64` y retorna detecciones, texto OCR e imagen procesada en Base64. |
| `POST` | `/predict_json/` | Recibe un JSON `{ "image_base64": "..." }` y retorna la lectura de placas. |

---

## 🔧 Variables de Entorno

#### Backend (`/backend/.env`)
- `PORT`: Puerto en el que corre la API (Por defecto: `8080`).
- `MODEL_PATH`: Ruta al archivo de pesos `.pt` del modelo YOLOv8.
- `CONF_THRESH`: Umbral mínimo de confianza para detecciones YOLO (Por defecto: `0.45`).
- `IOU_THRESH`: Umbral de NMS IOU (Por defecto: `0.45`).

#### Frontend (`/frontend/.env`)
- `VITE_API_URL`: URL base de la API backend (Por defecto: `http://localhost:8080`).

---

## 🎓 Créditos y Entrenamiento
El modelo YOLOv8 fue entrenado usando el notebook [reentrenamiento_yolo_colab.ipynb](reentrenamiento_yolo_colab.ipynb) disponible en este repositorio.
