#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + EasyOCR para detección de placas
# Requiere: fastapi uvicorn ultralytics easyocr opencv-python-headless pillow numpy python-multipart

import os
import re
import logging
import base64
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract

BASE_DIR = Path(__file__).resolve().parents[2]

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

_default_model = BASE_DIR / "models" / "best.pt"
if not _default_model.exists() and (BASE_DIR / "best.pt").exists():
    _default_model = BASE_DIR / "best.pt"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_default_model)))
OCR_LANGS = os.getenv("OCR_LANGS", "eng").split(",")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
RETURN_IMAGE = True  # Devolver imagen con detecciones

# Tesseract suele estar instalado en el sistema, no como dependencia de Python
# Ruta por defecto en Ubuntu/Debian:
# /usr/bin/tesseract
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

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
model = YOLO(str(MODEL_PATH))
logger.info("✅ Modelo YOLOv8 cargado correctamente.")

logger.info("🔹 OCR configurado con Tesseract y idiomas: %s", OCR_LANGS)

# -------------------------
# Helpers de Preprocesamiento y OCR Mejorado
# -------------------------
def preprocess_roi_variants(roi_bgr: np.ndarray) -> List[np.ndarray]:
    """Genera variaciones de la imagen ROI para maximizar la precisión de EasyOCR."""
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    variants = []
    h, w = roi_bgr.shape[:2]

    # Reescalar si el recorte es pequeño (menos de 160px de ancho)
    scale = 1.0
    if w < 160:
        scale = 160.0 / w
    elif w > 800:
        scale = 800.0 / w

    if abs(scale - 1.0) > 0.05:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized_bgr = cv2.resize(roi_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        resized_bgr = roi_bgr

    # Variante 1: RGB original
    roi_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    variants.append(roi_rgb)

    # Variante 2: Escala de grises + CLAHE (mejora de contraste)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    variants.append(gray_clahe)

    # Variante 3: Binarización Adaptativa
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(adaptive_thresh)

    return variants


# Regex para detectar formato estándar de placas vehiculares (3 letras + 3 números, 3 letras + 2 números + 1 letra, etc.)
PLATE_REGEX = re.compile(r'([A-Z]{3}\d{3}|[A-Z]{3}\d{2}[A-Z]|[A-Z]{2}\d{4}|[A-Z]{3}\d{3}[A-Z]?)')

def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """Lee texto de una ROI usando Tesseract con varias variantes de imagen."""
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        variants = preprocess_roi_variants(roi_bgr)
        best_text = None
        best_char_count = 0

        for img in variants:
            if isinstance(img, np.ndarray) and len(img.shape) == 2:
                gray_img = img
            else:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # Tesseract suele funcionar mejor con texto en blanco sobre fondo negro.
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            texts = []

            for config in [
                "--psm 7 --oem 3",
                "--psm 6 --oem 3",
                "--psm 11 --oem 3",
            ]:
                try:
                    text = pytesseract.image_to_string(binary, config=config, lang=OCR_LANGS[0])
                except Exception:
                    text = ""
                cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
                if cleaned:
                    texts.append(cleaned)

            if not texts:
                continue

            combined = "".join(texts)
            match = PLATE_REGEX.search(combined)
            if match:
                return match.group(1)

            if len(combined) >= 3:
                if len(combined) > best_char_count:
                    best_char_count = len(combined)
                    best_text = combined

        if best_text:
            match = PLATE_REGEX.search(best_text)
            return match.group(1) if match else (best_text if len(best_text) >= 2 else None)

        return None

    except Exception as e:
        logger.exception("OCR error: %s", e)
        return None


def crop_roi_with_padding(frame: np.ndarray, box: np.ndarray, pad_pct: float = 0.10) -> np.ndarray:
    """Extrae el ROI agregando un margen (padding) porcentual para evitar recortar bordes de letras."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    box_w = x2 - x1
    box_h = y2 - y1

    pad_w = int(box_w * pad_pct)
    pad_h = int(box_h * pad_pct)

    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(w, x2 + pad_w)
    y2_pad = min(h, y2 + pad_h)

    return frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()


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
        if file is not None and hasattr(file, "read"):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
        elif image_base64:
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

            # Extraer ROI con padding para no cortar bordes
            roi = crop_roi_with_padding(frame, box, pad_pct=0.10)

            # Verificar si la etiqueta corresponde a una placa (o si el modelo solo tiene 1 clase)
            is_plate = (len(model.names) == 1) or any(k in label.lower() for k in ["placa", "plate", "license", "0", "vehic"])
            if is_plate:
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

            # Extraer ROI con padding
            roi = crop_roi_with_padding(frame, box, pad_pct=0.10)

            is_plate = (len(model.names) == 1) or any(k in label.lower() for k in ["placa", "plate", "license", "0", "vehic"])
            if is_plate:
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
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
  
