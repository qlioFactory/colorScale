import os
import base64
from typing import Optional, Dict, Any

import certifi

import numpy as np
import cv2
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

API_KEY = os.getenv("API_KEY", "")  # ponlo como env var en Cloud Run

app = FastAPI(title="ColorScale API", version="0.1.0")

# CORS: para que el frontend (Base44) pueda llamar con fetch()
# Para empezar lo dejamos abierto; luego puedes restringir a tu dominio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)

class AnalyzeReq(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    debug: bool = False
    client_id: Optional[str] = None
    scan_id: Optional[str] = None

def load_image_from_request(req: AnalyzeReq) -> np.ndarray:
    if req.image_base64:
        try:
            data = base64.b64decode(req.image_base64)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64: {e}")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Invalid image_base64 (cannot decode)")
        return img

    if req.image_url:
        try:
            r = requests.get(req.image_url, timeout=25, verify=certifi.where())
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(400, f"Cannot download image_url: {e}")
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Invalid image_url content (cannot decode)")
        return img

    raise HTTPException(400, "Provide image_url or image_base64")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.post("/analyze-strip")
def analyze_strip(req: AnalyzeReq, x_api_key: str = Header(default="")) -> Dict[str, Any]:
    # Seguridad mínima
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Unauthorized")

    img_bgr = load_image_from_request(req)

    # TODO: aquí irá el pipeline real (franjas, calibración, pads, DeltaE)
    # Por ahora devolvemos una respuesta “stub” para que Base44 pueda integrar ya.

    h, w = img_bgr.shape[:2]
    return {
        "ok": False,
        "orientation": "",
        "results": [],
        "diagnostics": {
            "imageSize": [w, h],
            "foundBars": False,
            "calibrationError": None,
            "foundPads": 0,
            "blurScore": None,
            "warnings": ["Pipeline not implemented yet"]
        },
        "retake_reason": "Backend pipeline pendiente de implementar",
        "retake_tips": [
            "Integrad el endpoint en Base44 con fetch() y luego activamos el análisis real",
            "Aseguraos de que image_url sea pública o signed; si no, usad image_base64"
        ]
    }
