#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + OCR Optimizado para detección de placas
# Requiere: fastapi uvicorn ultralytics easyocr opencv-python-headless pillow numpy python-multipart pytesseract

import os
import re
import logging
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
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
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.45))
IOU_THRESH = float(os.getenv("IOU_THRESH", 0.45))
RETURN_IMAGE = True

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

# -------------------------
# App init
# -------------------------
app = FastAPI(title="YOLOv8 + OCR Optimizado - Detector de Placas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelo
# -------------------------
logger.info("🔹 Cargando modelo YOLOv8 desde %s ...", MODEL_PATH)
model = YOLO(str(MODEL_PATH))
logger.info("✅ Modelo YOLOv8 cargado correctamente.")

# Intentar cargar EasyOCR si está instalado
easyocr_reader = None
try:
    import easyocr
    logger.info("🔹 Inicializando EasyOCR (detector DL)...")
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    logger.info("✅ EasyOCR inicializado correctamente.")
except Exception as e:
    logger.info("ℹ️ EasyOCR no disponible. Se usará exclusivamente PyTesseract.")

# -------------------------
# Helpers de Filtrado y Preprocesamiento
# -------------------------
PLATE_REGEX = re.compile(r'([A-Z]{2,4}\d{2,4}[A-Z]?)')

def clean_and_correct_plate(text: str) -> Optional[str]:
    """Limpia el texto y busca patrones de placas válidas dentro del texto extraído."""
    if not text:
        return None

    text_upper = text.upper()

    # 1. Buscar coincidencia de placa colombiana estándar (3 letras + 3 números o 3 letras + 2 números + 1 letra)
    # Ej: "CEK705", "CEK 705", "COLOMBIA CEK-705", "GARZON 7259"
    match = re.search(r'[A-Z]{3}\s*[-–]?\s*\d{2,3}[A-Z0-9]?', text_upper)
    if match:
        cleaned_match = re.sub(r"[^A-Za-z0-9]", "", match.group(0))
        if 5 <= len(cleaned_match) <= 7:
            return cleaned_match

    # 2. Si no hay match directo, limpiar la cadena entera sin símbolos
    cleaned_all = re.sub(r"[^A-Za-z0-9]", "", text_upper)

    # Buscar una subcadena de 3 letras y 3 números dentro de toda la cadena pegada
    sub_match = re.search(r'[A-Z]{3}\d{2,3}[A-Z0-9]?', cleaned_all)
    if sub_match:
        return sub_match.group(0)

    # 3. Si la palabra limpia mide entre 5 y 7 caracteres y tiene al menos 2 letras y 2 números
    if 5 <= len(cleaned_all) <= 7:
        letras = sum(c.isalpha() for c in cleaned_all)
        numeros = sum(c.isdigit() for c in cleaned_all)
        if letras >= 2 and numeros >= 2:
            return cleaned_all

    # 4. Probar palabra por palabra por si venía separada por espacios
    words = re.split(r'[\s\n\r\t]+', text_upper)
    for word in words:
        w_clean = re.sub(r"[^A-Za-z0-9]", "", word)
        if 5 <= len(w_clean) <= 7:
            letras = sum(c.isalpha() for c in w_clean)
            numeros = sum(c.isdigit() for c in w_clean)
            if letras >= 2 and numeros >= 2:
                return w_clean

    return None


def is_valid_plate_box(box: np.ndarray, frame_shape: tuple) -> bool:
    """Filtra detecciones falsas según relación de aspecto (ancho/alto) y tamaño mínimo."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w < 15 or h < 8:
        return False
    
    aspect_ratio = w / float(h)
    if aspect_ratio < 1.1 or aspect_ratio > 6.0:
        return False

    return True


def preprocess_roi_variants(roi_bgr: np.ndarray) -> List[np.ndarray]:
    """Genera variaciones de la imagen ROI para maximizar precisión del OCR."""
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    h, w = roi_bgr.shape[:2]
    # Escalar a un ancho de 300px, ideal para que Tesseract lea letras (aprox 30-40px de alto)
    scale = 300.0 / float(w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(roi_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 🚨 TRUCO CLAVE: Tesseract ignora las letras que tocan los bordes.
    # Añadimos un margen/borde gris claro alrededor de la placa de 20 píxeles.
    border = 20
    padded = cv2.copyMakeBorder(resized, border, border, border, border, cv2.BORDER_CONSTANT, value=[200, 200, 200])

    variants = []
    
    # 1. RGB (A veces lee mejor a color)
    variants.append(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    
    # Aplicar ecualización de histograma (mejora contraste entre letras negras y fondo amarillo/blanco)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    # 2. Escala de grises con contraste mejorado
    variants.append(gray_clahe)

    # 3. Binarización Otsu directa
    _, binary_otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary_otsu)

    # 4. Filtro Bilateral + Otsu (limpia el ruido conservando bordes afilados de las letras)
    blur = cv2.bilateralFilter(gray_clahe, 11, 17, 17)
    _, binary_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary_blur)

    # 🚨 GUARDAR IMÁGENES PARA DEPURACIÓN EN EL SERVIDOR
    cv2.imwrite("debug_roi_0_rgb.jpg", variants[0])
    cv2.imwrite("debug_roi_1_gray.jpg", variants[1])
    cv2.imwrite("debug_roi_2_otsu.jpg", variants[2])
    cv2.imwrite("debug_roi_3_blur.jpg", variants[3])

    return variants


def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """Ejecuta OCR sobre variantes de la ROI y devuelve la lectura real exacta sin adivinar."""
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        variants = preprocess_roi_variants(roi_bgr)
        candidates = []

        # 1. Probar EasyOCR si está instalado en el servidor
        if easyocr_reader is not None:
            try:
                for img in variants[:2]: # Probar en variante RGB y Grayscale
                    results = easyocr_reader.readtext(img, detail=0)
                    raw_easy = " ".join(results)
                    if raw_easy:
                        logger.info(f"🔎 EasyOCR Raw: '{raw_easy}'")
                        cleaned_easy = clean_and_correct_plate(raw_easy)
                        if cleaned_easy:
                            candidates.append(cleaned_easy)
            except Exception as e:
                logger.warning("Error en EasyOCR: %s", e)

        # 2. Probar PyTesseract
        whitelist = " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        for i, img in enumerate(variants):
            gray_img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for psm in ["--psm 6 --oem 3", "--psm 11 --oem 3", "--psm 7 --oem 3", "--psm 8 --oem 3"]:
                config_str = psm + whitelist
                try:
                    raw_text = pytesseract.image_to_string(gray_img, config=config_str, lang=OCR_LANGS[0])
                    texto_limpio_raw = raw_text.strip().replace('\n', ' ')
                    if texto_limpio_raw:
                        logger.info(f"🔎 OCR Raw [Var:{i} {psm}]: '{texto_limpio_raw}'")
                except Exception as e:
                    logger.warning("Error en pytesseract: %s", e)
                    raw_text = ""

                cleaned = clean_and_correct_plate(raw_text)
                if cleaned:
                    candidates.append(cleaned)

        if candidates:
            # Ordenar para preferir coincidencia estricta de placa y mayor longitud
            candidates.sort(key=lambda c: (re.search(r'[A-Z]{3}[0-9]{2}[A-Z0-9]', c) is not None, len(c)), reverse=True)
            logger.info("🏆 Mejor candidato OCR: %s", candidates[0])
            return candidates[0]

        return None

    except Exception as e:
        logger.exception("Error en OCR: %s", e)
        return None


def crop_roi_with_padding(frame: np.ndarray, box: np.ndarray, pad_pct: float = 0.08) -> np.ndarray:
    """Extrae la sub-imagen agregando margen porcentual."""
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
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return base64.b64encode(buffer).decode('utf-8')


# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def home():
    return {"message": "YOLOv8 + OCR Optimizado server running", "model": str(MODEL_PATH)}


@app.post("/predict/")
@app.post("/predict")
@app.post("/api/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    try:
        logger.info("📩 Petición recibida en /predict/")

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

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        logger.info("🧠 Procesando imagen con YOLOv8 (conf=%.2f, iou=%.2f)...", CONF_THRESH, IOU_THRESH)
        results = model.predict(source=frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
        if not results:
            return {"placas": [], "num_placas": 0, "details": [], "image": None, "success": True, "message": "Sin detecciones"}

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])

        placas_detectadas: List[str] = []
        detalles: List[Dict[str, Any]] = []

        for i, box in enumerate(boxes):
            if not is_valid_plate_box(box, frame.shape):
                continue

            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else 0
            label = model.names[cls_id] if cls_id < len(model.names) else "placa"
            conf = float(confs[i]) if len(confs) > i else 0.0

            roi = crop_roi_with_padding(frame, box, pad_pct=0.08)
            text_detected = ocr_read_text_from_roi(roi)

            display_text = text_detected if text_detected else ("PLACA NO DETECTADA" if len(boxes) == 1 else f"PLACA NO DETECTADA #{i+1}")
            if display_text not in placas_detectadas:
                placas_detectadas.append(display_text)

            # Calcular el porcentaje de acierto basado en la confianza de YOLO y la validez del OCR
            es_placa_valida = bool(text_detected and re.search(r'[A-Z]{3}[0-9]{2}[A-Z0-9]', text_detected))
            # Si el OCR es válido y concuerda con la placa, la precisión ronda 90-99%
            porcentaje_acierto = round((conf * 0.5 + (0.45 if es_placa_valida else 0.1)) * 100, 1)

            detalles.append({
                "caja": [x1, y1, x2, y2],
                "confianza_yolo": round(conf, 4),
                "confianza_yolo_pct": f"{round(conf * 100, 1)}%",
                "texto_ocr": text_detected,
                "es_placa_valida": es_placa_valida,
                "porcentaje_acierto": porcentaje_acierto,
                "porcentaje_acierto_str": f"{porcentaje_acierto}%",
                "clase": label
            })

            color_bgr = (0, 255, 0) if es_placa_valida else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{display_text} ({porcentaje_acierto}%)", (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None

        logger.info("✅ Placas detectadas: %s", placas_detectadas)

        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "details": detalles,
            "image": img_b64,
            "message": "OK" if placas_detectadas else "No se detectaron placas válidas"
        }

    except Exception as e:
        logger.exception("Error en /predict/: %s", e)
        return {"error": str(e)}


@app.post("/predict_json/")
@app.post("/predict_json")
@app.post("/api/predict_json")
async def predict_json(request: Request):
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

        results = model.predict(source=frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
        if not results:
            return {"placas": [], "num_placas": 0, "details": [], "image": None, "success": True, "message": "Sin detecciones"}

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])

        placas_detectadas = []
        detalles = []

        for i, box in enumerate(boxes):
            if not is_valid_plate_box(box, frame.shape):
                continue

            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else 0
            label = model.names[cls_id] if cls_id < len(model.names) else "placa"
            conf = float(confs[i]) if len(confs) > i else 0.0

            roi = crop_roi_with_padding(frame, box, pad_pct=0.08)
            text_detected = ocr_read_text_from_roi(roi)

            display_text = text_detected if text_detected else ("PLACA NO DETECTADA" if len(boxes) == 1 else f"PLACA NO DETECTADA #{i+1}")
            if display_text not in placas_detectadas:
                placas_detectadas.append(display_text)

            # Calcular el porcentaje de acierto basado en la confianza de YOLO y la validez del OCR
            es_placa_valida = bool(text_detected and re.search(r'[A-Z]{3}[0-9]{2}[A-Z0-9]', text_detected))
            porcentaje_acierto = round((conf * 0.5 + (0.45 if es_placa_valida else 0.1)) * 100, 1)

            detalles.append({
                "caja": [x1, y1, x2, y2],
                "confianza_yolo": round(conf, 4),
                "confianza_yolo_pct": f"{round(conf * 100, 1)}%",
                "texto_ocr": text_detected,
                "es_placa_valida": es_placa_valida,
                "porcentaje_acierto": porcentaje_acierto,
                "porcentaje_acierto_str": f"{porcentaje_acierto}%",
                "clase": label
            })

            color_bgr = (0, 255, 0) if es_placa_valida else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{display_text} ({porcentaje_acierto}%)", (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None

        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "details": detalles,
            "image": img_b64,
            "message": "OK" if placas_detectadas else "No se detectaron placas válidas"
        }

    except Exception as e:
        logger.exception("Error en /predict_json/: %s", e)
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("🚀 Iniciando servidor en 0.0.0.0:%s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
  
