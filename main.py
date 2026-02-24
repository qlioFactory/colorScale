import os
import base64
import json
from typing import Optional, Dict, Any, List, Tuple

import certifi
import numpy as np
import cv2
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

API_KEY = os.getenv("API_KEY", "")
SWATCHES_PATH = os.getenv("SWATCHES_PATH", "swatches.json")

PARAM_ORDER = [
    "pH", "chloride", "gh", "alkalinity", "iron",
    "copper", "aluminium", "sulfate", "nitrate", "free_chlorine"
]

TARGET_BARS_RGB = {
    "ROJO": [255, 0, 0],
    "VERDE": [0, 128, 0],
    "AZUL": [0, 0, 255],
    "GRIS": [128, 128, 128],
}

app = FastAPI(title="ColorScale API", version="0.2.0")

# CORS abierto para integración con Base44/frontend
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


_SWATCHES_CACHE = None


def load_swatches() -> Dict[str, Any]:
    global _SWATCHES_CACHE
    if _SWATCHES_CACHE is not None:
        return _SWATCHES_CACHE

    if not os.path.exists(SWATCHES_PATH):
        raise FileNotFoundError(f"swatches file not found: {SWATCHES_PATH}")

    with open(SWATCHES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for p in PARAM_ORDER:
        if p not in data:
            raise ValueError(f"Missing swatches for parameter: {p}")

    _SWATCHES_CACHE = data
    return data


def smooth1d(x: np.ndarray, k: int = 31) -> np.ndarray:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode="same")


def local_max_peaks(
    signal: np.ndarray,
    y_min: int,
    y_max: int,
    min_height: float,
    min_dist: int = 15
) -> List[int]:
    cand = []
    y_min = max(1, y_min)
    y_max = min(len(signal) - 2, y_max)
    for y in range(y_min, y_max):
        if signal[y] >= signal[y - 1] and signal[y] > signal[y + 1] and signal[y] >= min_height:
            cand.append(y)

    picked: List[int] = []
    for y in sorted(cand, key=lambda yy: float(signal[yy]), reverse=True):
        if all(abs(y - p) >= min_dist for p in picked):
            picked.append(y)

    return sorted(picked)


def find_peak_span(score: np.ndarray, peak_idx: int, frac: float = 0.55) -> Tuple[int, int]:
    peak = float(score[peak_idx])
    if peak <= 0:
        return max(0, peak_idx - 5), min(len(score) - 1, peak_idx + 5)

    th = peak * frac
    l = int(peak_idx)
    while l > 0 and score[l] >= th:
        l -= 1
    r = int(peak_idx)
    while r < len(score) - 1 and score[r] >= th:
        r += 1
    return l, r


def sample_patch_median(img_rgb: np.ndarray, cx: int, cy: int, rx: int = 4, ry: int = 6) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x0 = max(0, int(cx - rx))
    x1 = min(w, int(cx + rx + 1))
    y0 = max(0, int(cy - ry))
    y1 = min(h, int(cy + ry + 1))
    patch = img_rgb[y0:y1, x0:x1]

    if patch.size == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)

    med = np.median(patch.reshape(-1, 3), axis=0)
    return med.astype(np.float32)


def sample_bar_color(img_rgb: np.ndarray, x_center: int, y_top: int, y_bottom: int, rx: int = 8) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x0 = max(0, int(x_center - rx))
    x1 = min(w, int(x_center + rx + 1))

    margin = max(8, int((y_bottom - y_top) * 0.12))
    yy0 = max(0, int(y_top + margin))
    yy1 = min(h, int(y_bottom - margin))
    if yy1 <= yy0:
        yy0 = max(0, int(y_top))
        yy1 = min(h, int(y_bottom))

    patch = img_rgb[yy0:yy1, x0:x1]
    if patch.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    med = np.median(patch.reshape(-1, 3), axis=0)
    return med.astype(np.float32)


def fit_channel_linear_correction(obs_bars: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Ajuste lineal por canal: out_c = a_c * in_c + b_c
    (más estable que una matriz 3x3 completa en esta fase)
    """
    names = ["ROJO", "VERDE", "AZUL", "GRIS"]
    obs_mat = np.array([obs_bars[n] for n in names], dtype=np.float32)
    tgt_mat = np.array([TARGET_BARS_RGB[n] for n in names], dtype=np.float32)

    params = np.zeros((3, 2), dtype=np.float32)
    for c in range(3):
        x = obs_mat[:, c]
        y = tgt_mat[:, c]
        A = np.vstack([x, np.ones_like(x)]).T
        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        params[c, 0] = float(sol[0])  # a
        params[c, 1] = float(sol[1])  # b

    return params


def apply_channel_linear(rgb: np.ndarray, params: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    out = np.array([
        params[0, 0] * rgb[0] + params[0, 1],
        params[1, 0] * rgb[1] + params[1, 1],
        params[2, 0] * rgb[2] + params[2, 1],
    ], dtype=np.float32)
    return np.clip(out, 0, 255)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    RGB (0..255) -> CIELab aproximado usando OpenCV.
    """
    rgb_u8 = np.uint8([[np.clip(np.round(rgb), 0, 255).astype(np.uint8)]])
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0], dtype=np.float32)


def deltae76(rgb1: np.ndarray, rgb2: List[int]) -> float:
    l1 = rgb_to_lab(np.asarray(rgb1, dtype=np.float32))
    l2 = rgb_to_lab(np.asarray(rgb2, dtype=np.float32))
    return float(np.linalg.norm(l1 - l2))


def confidence_from_deltae(d: float) -> str:
    if d < 10:
        return "high"
    if d < 22:
        return "medium"
    return "low"


def match_swatch(param: str, rgb: np.ndarray, swatches: Dict[str, Any]) -> Dict[str, Any]:
    best = None
    for sw in swatches.get(param, []):
        d = deltae76(rgb, sw["rgb"])
        if best is None or d < best["deltaE"]:
            best = {
                "value": sw["value"],
                "reference_rgb": sw["rgb"],
                "deltaE": d
            }
    return best or {"value": None, "reference_rgb": None, "deltaE": 999.0}


def detect_geometry(img_rgb: np.ndarray) -> Dict[str, Any]:
    h, w = img_rgb.shape[:2]

    # Perfil por columnas en zona central para detectar franjas
    ys0, ys1 = int(h * 0.15), int(h * 0.85)
    col = img_rgb[ys0:ys1].astype(np.float32).mean(axis=0)
    R, G, B = col[:, 0], col[:, 1], col[:, 2]

    red_score = smooth1d(R - (G + B) / 2.0, 41)
    green_score = smooth1d(G - (R + B) / 2.0, 41)
    blue_score = smooth1d(B - (R + G) / 2.0, 41)

    red_x = int(np.argmax(red_score))
    green_x = int(np.argmax(green_score))
    blue_x = int(np.argmax(blue_score))

    red_span = find_peak_span(red_score, red_x, 0.55)
    green_span = find_peak_span(green_score, green_x, 0.55)
    blue_span = find_peak_span(blue_score, blue_x, 0.55)

    if red_score[red_x] < 25 or green_score[green_x] < 20 or blue_score[blue_x] < 20:
        raise ValueError("No se detectan correctamente las franjas de referencia RGB")

    # La tira SIEMPRE está entre azul y verde, aunque el usuario invierta izquierda/derecha
    if blue_x < green_x:
        orientation = "gray-blue-strip-green-red"
        strip_left = blue_span[1] + 8
        strip_right = green_span[0] - 8
        gray_dir = -1
    else:
        orientation = "red-green-strip-blue-gray"
        strip_left = green_span[1] + 8
        strip_right = blue_span[0] - 8
        gray_dir = +1

    strip_left = max(0, int(strip_left))
    strip_right = min(w - 1, int(strip_right))

    if strip_right - strip_left < 20:
        raise ValueError("No se pudo aislar la zona de la tira entre azul y verde")

    # Altura útil de las franjas de color (para saber el rango vertical de análisis)
    def color_dom_profile(x: int, mode: str) -> np.ndarray:
        c = img_rgb[:, max(0, x - 2):min(w, x + 3)].astype(np.float32).mean(axis=1)
        rr, gg, bb = c[:, 0], c[:, 1], c[:, 2]
        if mode == "red":
            s = rr - (gg + bb) / 2.0
        elif mode == "green":
            s = gg - (rr + bb) / 2.0
        elif mode == "blue":
            s = bb - (rr + gg) / 2.0
        else:
            m = c.mean(axis=1)
            sat = c.max(axis=1) - c.min(axis=1)
            s = -sat - 0.25 * np.abs(m - 128.0)
        return smooth1d(s, 21)

    tops, bottoms = [], []
    for mode, x in [("blue", blue_x), ("green", green_x), ("red", red_x)]:
        prof = color_dom_profile(x, mode)
        py = int(np.argmax(prof))
        peak = float(prof[py])
        th = peak * 0.35

        l = py
        while l > 0 and prof[l] >= th:
            l -= 1
        r = py
        while r < len(prof) - 1 and prof[r] >= th:
            r += 1

        tops.append(l)
        bottoms.append(r)

    bars_y_top = int(np.median(np.array(tops)))
    bars_y_bottom = int(np.median(np.array(bottoms)))

    # Detectar top/bottom de los pads a partir de bordes horizontales en el centro de la tira
    strip_center_x = int((strip_left + strip_right) / 2)

    gray_img = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY).astype(np.float32)
    center_band = gray_img[:, max(0, strip_center_x - 12):min(w, strip_center_x + 13)]
    dy = np.abs(np.diff(center_band, axis=0)).mean(axis=1)
    edge_profile = smooth1d(dy, 5)

    min_h = max(4.5, float(edge_profile.max()) * 0.28)
    strong_peaks = local_max_peaks(
        edge_profile,
        y_min=max(0, bars_y_top - 40),
        y_max=min(h - 2, bars_y_bottom + 40),
        min_height=min_h,
        min_dist=18,
    )

    if len(strong_peaks) >= 2:
        pads_y_top = int(strong_peaks[0])
        pads_y_bottom = int(strong_peaks[-1])
    else:
        pads_y_top = int(bars_y_top)
        pads_y_bottom = int(bars_y_bottom)

    # Centro aproximado de la franja gris (para corrección de color)
    bg_gap = abs(green_x - blue_x)
    gray_x = int(round(blue_x + gray_dir * 0.45 * bg_gap))
    gray_x = max(0, min(w - 1, gray_x))

    return {
        "orientation": orientation,
        "red_x": red_x,
        "green_x": green_x,
        "blue_x": blue_x,
        "gray_x": gray_x,
        "red_span": red_span,
        "green_span": green_span,
        "blue_span": blue_span,
        "strip_left": strip_left,
        "strip_right": strip_right,
        "strip_center_x": strip_center_x,
        "bars_y_top": bars_y_top,
        "bars_y_bottom": bars_y_bottom,
        "pads_y_top": pads_y_top,
        "pads_y_bottom": pads_y_bottom,
        "edge_peaks_count": len(strong_peaks),
    }


def analyze_image(img_bgr: np.ndarray, debug: bool = False) -> Dict[str, Any]:
    swatches = load_swatches()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    geom = detect_geometry(img_rgb)

    # Muestreo de franjas y corrección de color por canal (suave)
    obs_bars = {
        "ROJO": sample_bar_color(img_rgb, geom["red_x"], geom["bars_y_top"], geom["bars_y_bottom"], rx=8),
        "VERDE": sample_bar_color(img_rgb, geom["green_x"], geom["bars_y_top"], geom["bars_y_bottom"], rx=8),
        "AZUL": sample_bar_color(img_rgb, geom["blue_x"], geom["bars_y_top"], geom["bars_y_bottom"], rx=8),
        "GRIS": sample_bar_color(img_rgb, geom["gray_x"], geom["bars_y_top"], geom["bars_y_bottom"], rx=8),
    }
    color_params = fit_channel_linear_correction(obs_bars)

    pads_top = int(geom["pads_y_top"])
    pads_bottom = int(geom["pads_y_bottom"])
    pads_h = max(10, pads_bottom - pads_top)
    pitch = pads_h / 10.0

    strip_center_x = int(geom["strip_center_x"])
    strip_width = max(10, int(geom["strip_right"] - geom["strip_left"]))
    rx = max(3, min(6, strip_width // 14))
    ry = max(4, min(7, int(pitch * 0.10)))

    results = []
    deltaes = []

    for i, param in enumerate(PARAM_ORDER):
        cy = int(round(pads_top + (i + 0.5) * pitch))

        rgb_raw = sample_patch_median(img_rgb, strip_center_x, cy, rx=rx, ry=ry)
        rgb_corr = apply_channel_linear(rgb_raw, color_params)

        m_raw = match_swatch(param, rgb_raw, swatches)
        m_corr = match_swatch(param, rgb_corr, swatches)

        # Usar corregido solo si mejora claramente y no "revienta" la saturación
        corr_clip_count = int(np.sum((rgb_corr <= 2) | (rgb_corr >= 253)))
        use_corrected = (
            (m_corr["deltaE"] + 2.0 < m_raw["deltaE"])
            and (m_corr["deltaE"] < 30.0)
            and (corr_clip_count <= 1)
        )

        if use_corrected:
            chosen_mode = "corrected"
            chosen_rgb = rgb_corr
            chosen = m_corr
        else:
            chosen_mode = "raw"
            chosen_rgb = rgb_raw
            chosen = m_raw

        d = float(chosen["deltaE"])
        deltaes.append(d)

        results.append({
            "index": i + 1,
            "parameter": param,
            "value": str(chosen["value"]),
            "confidence": confidence_from_deltae(d),
            "deltaE": round(d, 2),
            "mode": chosen_mode,
            "sample_rgb_raw": [int(round(x)) for x in rgb_raw.tolist()],
            "sample_rgb_used": [int(round(x)) for x in chosen_rgb.tolist()],
            "reference_rgb": chosen["reference_rgb"],
            "sample_point": {"x": int(strip_center_x), "y": int(cy)},
        })

    mean_delta = float(np.mean(deltaes)) if deltaes else None
    low_count = sum(1 for d in deltaes if d > 22)

    warnings = []
    if low_count >= 4:
        warnings.append("Varias coincidencias con baja confianza: revisa iluminación/posición")
    if geom["edge_peaks_count"] < 6:
        warnings.append("Detección de bordes débil en los pads; posible desenfoque")
    if geom["strip_right"] - geom["strip_left"] < 35:
        warnings.append("Zona de tira estrecha; la foto puede estar muy lejos")

    return {
        "ok": True,
        "orientation": geom["orientation"],
        "results": results,
        "diagnostics": {
            "imageSize": [int(w), int(h)],
            "foundBars": True,
            "calibrationError": None,
            "foundPads": 10,
            "blurScore": None,
            "matchMeanDeltaE": round(mean_delta, 2) if mean_delta is not None else None,
            "warnings": warnings,
            "geometry": {
                "stripX": [int(geom["strip_left"]), int(geom["strip_right"])],
                "barsY": [int(geom["bars_y_top"]), int(geom["bars_y_bottom"])],
                "padsY": [int(geom["pads_y_top"]), int(geom["pads_y_bottom"])],
                "barCentersX": {
                    "red": int(geom["red_x"]),
                    "green": int(geom["green_x"]),
                    "blue": int(geom["blue_x"]),
                    "grayApprox": int(geom["gray_x"]),
                }
            },
            "referenceBarsObserved": {
                k: [int(round(v)) for v in arr.tolist()]
                for k, arr in obs_bars.items()
            },
        },
        "retake_reason": None,
        "retake_tips": [] if not warnings else [
            "Usa luz uniforme (sin sombras ni reflejos)",
            "Alinea la tira centrada entre las franjas azul y verde",
            "Mantén la cámara perpendicular y con enfoque nítido",
        ],
    }


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
            headers = {
                "User-Agent": "ColorScale/1.0 (+https://github.com/qlioFactory/colorScale)",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            }
            r = requests.get(
                req.image_url,
                timeout=25,
                verify=certifi.where(),
                headers=headers,
                allow_redirects=True,
            )
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
    return {
        "ok": True,
        "swatchesLoaded": os.path.exists(SWATCHES_PATH)
    }


@app.post("/analyze-strip")
def analyze_strip(req: AnalyzeReq, x_api_key: str = Header(default="")) -> Dict[str, Any]:
    # Seguridad mínima
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Unauthorized")

    img_bgr = load_image_from_request(req)

    try:
        return analyze_image(img_bgr, debug=bool(req.debug))

    except FileNotFoundError as e:
        return {
            "ok": False,
            "orientation": "",
            "results": [],
            "diagnostics": {
                "imageSize": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
                "foundBars": False,
                "calibrationError": "swatches_missing",
                "foundPads": 0,
                "blurScore": None,
                "warnings": [str(e)],
            },
            "retake_reason": "Falta swatches.json en el contenedor",
            "retake_tips": [
                "Sube swatches.json al repositorio junto a main.py"
            ],
        }

    except Exception as e:
        return {
            "ok": False,
            "orientation": "",
            "results": [],
            "diagnostics": {
                "imageSize": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
                "foundBars": False,
                "calibrationError": "analysis_exception",
                "foundPads": 0,
                "blurScore": None,
                "warnings": [str(e)],
            },
            "retake_reason": "No se pudo analizar la tira en esta imagen",
            "retake_tips": [
                "Alinea la tira entre las franjas azul y verde",
                "Evita reflejos y sombras",
                "Asegura buena nitidez",
            ],
        }
