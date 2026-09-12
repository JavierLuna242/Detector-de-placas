#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import uvicorn

# Incluir la raíz del proyecto en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 Iniciando servidor FastAPI en 0.0.0.0:{port}...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
