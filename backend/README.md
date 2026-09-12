# Deployment Mobile YOLO

Este proyecto tiene una API FastAPI con detección de placas usando YOLOv8 + EasyOCR y un frontend React/Vite.

## Requisitos

- Docker
- Docker Compose

## Ejecutar con contenedores

1. Copia el archivo `.env.example` a `.env` si quieres personalizar la configuración.
2. Ejecuta:

```bash
docker compose up --build
```

3. Abre:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8080

## Servicios

- Backend: Python + FastAPI + YOLOv8 + EasyOCR
- Frontend: Vite + React

## Variables de entorno

- `PORT`: puerto del backend
- `VITE_API_URL`: URL del backend visto desde el navegador
- `MODEL_PATH`: ruta del modelo YOLO

## Detección

La API expone estos endpoints:

- `GET /`
- `POST /predict/`
- `POST /predict_json/`
