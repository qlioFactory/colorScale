"""
ColorScale backend - overwrite v2-calibration

Compatible with the existing Base44 frontend endpoint:
  POST /analyze-strip

Main goals:
- No generative AI for color matching.
- Detect the fixed template bars (gray, blue, green, red).
- Calibrate the photographed sample color using the fixed reference bars.
- Compare in CIELAB using Delta E 2000.
- Return the nearest value, status, confidence and diagnostics.
- Store historical results in a lightweight SQLite DB by default.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import certifi
import cv2
import numpy as np
import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

APP_VERSION = "2.1.0-overwrite-calibration"
BASE_DIR = Path(__file__).resolve().parent

API_KEY = os.getenv("API_KEY", "")
SWATCHES_PATH = Path(os.getenv("SWATCHES_PATH", str(BASE_DIR / "swatches.json")))
HISTORY_DB_PATH = Path(os.getenv("HISTORY_DB_PATH", str(BASE_DIR / "colorscale_history.sqlite3")))
SAVE_HISTORY_DEFAULT = os.getenv("SAVE_HISTORY_DEFAULT", "true").lower() not in {"0", "false", "no"}
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(12 * 1024 * 1024)))

PARAM_ORDER = [
    "alkalinity",
    "pH",
    "gh",
    "free_chlorine",
    "nitrate",
    "copper",
    "iron",
    "aluminium",
    "sulfate",
    "chloride",
]

# Digital RGB targets of the printed template bars, as confirmed by the user.
TARGET_BARS_RGB: Dict[str, List[int]] = {
    "gray": [128, 128, 128],
    "blue": [0, 0, 255],
    "green": [0, 128, 0],
    "red": [255, 0, 0],
}

BAR_ORDER_CANONICAL = ["gray", "blue", "strip", "green", "red"]
WHITE_LIKE_PARAMS = {"free_chlorine", "nitrate", "copper", "iron", "aluminium"}

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------

app = FastAPI(title="ColorScale API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class AnalyzeReq(BaseModel):
    # Existing frontend-compatible inputs
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    debug: bool = False
    client_id: Optional[str] = None
    scan_id: Optional[str] = None

    # New optional inputs. Defaults preserve the old usage.
    save_history: Optional[bool] = None
    operator_id: Optional[str] = None
    location: Optional[str] = None


class AnalyzeResponse(BaseModel):
    ok: bool
    status: str
    quality_score: float = 0.0
    analysis_id: Optional[str] = None
    orientation: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    retake_reason: Optional[str] = None
    retake_tips: List[str] = Field(default_factory=list)


_SWATCHES_CACHE: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def check_api_key(x_api_key: str = "") -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def load_swatches() -> Dict[str, Any]:
    global _SWATCHES_CACHE
    if _SWATCHES_CACHE is not None:
        return _SWATCHES_CACHE
    if not SWATCHES_PATH.exists():
        raise FileNotFoundError(f"swatches file not found: {SWATCHES_PATH}")
    with SWATCHES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in PARAM_ORDER:
        if key not in data:
            raise ValueError(f"Missing parameter in swatches.json: {key}")
        meta = data[key]
        if "values" not in meta or "rgb" not in meta:
            raise ValueError(f"Invalid swatches for {key}: expected values and rgb")
        if len(meta["values"]) != len(meta["rgb"]):
            raise ValueError(f"Invalid swatches for {key}: values and rgb length differ")
    _SWATCHES_CACHE = data
    return data


def b64_to_bytes(value: str) -> bytes:
    # Accept both raw base64 and data URL format.
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc


def download_image_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "ColorScale/2.1 (+https://github.com/qlioFactory/colorScale)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        with requests.get(url, timeout=25, headers=headers, verify=certifi.where(), stream=True) as resp:
            resp.raise_for_status()
            chunks: List[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Image is too large")
                chunks.append(chunk)
            return b"".join(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot download image_url: {exc}") from exc


def load_image_from_request(req: AnalyzeReq) -> np.ndarray:
    if req.image_base64:
        raw = b64_to_bytes(req.image_base64)
    elif req.image_url:
        raw = download_image_bytes(req.image_url)
    else:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image content; OpenCV cannot decode it")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_int_list(rgb: Iterable[float]) -> List[int]:
    return [int(round(float(v))) for v in rgb]


def to_float_list(rgb: Iterable[float], digits: int = 1) -> List[float]:
    return [round(float(v), digits) for v in rgb]


def numeric_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def smooth_1d(values: np.ndarray, k: int = 31) -> np.ndarray:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def robust_patch_stats_rgb(
    img_rgb: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    reject_bright_pct: float = 90.0,
    reject_dark_pct: Optional[float] = 2.0,
) -> Tuple[np.ndarray, float, float]:
    """Return median RGB, bright rejected pct and dark rejected pct.

    The median is intentionally used instead of mean to reduce the effect of
    shadows, printed edges, glare and dust specks.
    """

    h, w = img_rgb.shape[:2]
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32), 0.0, 0.0

    patch = img_rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(patch) < 10:
        return np.median(patch, axis=0).astype(np.float32), 0.0, 0.0

    luminance = 0.2126 * patch[:, 0] + 0.7152 * patch[:, 1] + 0.0722 * patch[:, 2]
    mask = np.ones(len(patch), dtype=bool)

    bright_rej = 0.0
    dark_rej = 0.0
    if reject_bright_pct is not None:
        hi = np.percentile(luminance, reject_bright_pct)
        bright_mask = luminance <= hi
        bright_rej = 100.0 * float(np.sum(~bright_mask)) / float(len(mask))
        mask &= bright_mask
    if reject_dark_pct is not None:
        lo = np.percentile(luminance, reject_dark_pct)
        dark_mask = luminance >= lo
        dark_rej = 100.0 * float(np.sum(~dark_mask)) / float(len(mask))
        mask &= dark_mask

    good = patch[mask]
    if len(good) < 10:
        good = patch
        bright_rej = 0.0
        dark_rej = 0.0
    return np.median(good, axis=0).astype(np.float32), float(bright_rej), float(dark_rej)


# -----------------------------------------------------------------------------
# Color math
# -----------------------------------------------------------------------------


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin.astype(np.float32), 0.0, 1.0)
    x = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.float32)


def fit_per_channel_calibration(measured_refs: Dict[str, np.ndarray]) -> np.ndarray:
    """Fit per-channel linear calibration in linearized sRGB.

    The result has shape (3, 2): output_linear = slope * input_linear + intercept.
    This is less aggressive than a full 3x4 RGB affine matrix, so it tends to
    be more stable for phone photos and printed references.
    """

    names = ["red", "green", "blue", "gray"]
    obs = np.array([measured_refs[n] for n in names], dtype=np.float32)
    tgt = np.array([TARGET_BARS_RGB[n] for n in names], dtype=np.float32)
    obs_lin = srgb_to_linear(obs)
    tgt_lin = srgb_to_linear(tgt)

    params = np.zeros((3, 2), dtype=np.float32)
    for channel in range(3):
        x = obs_lin[:, channel]
        y = tgt_lin[:, channel]
        A = np.vstack([x, np.ones_like(x)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        # Limit extreme gains. Bad photos should be flagged, not over-corrected.
        params[channel, 0] = float(np.clip(sol[0], -3.0, 5.0))
        params[channel, 1] = float(np.clip(sol[1], -0.6, 0.6))
    return params


def apply_per_channel_calibration(rgb: np.ndarray, params: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float32))
    out_lin = params[:, 0] * lin + params[:, 1]
    return linear_to_srgb(out_lin)


def rgb_to_lab_cv(rgb: np.ndarray) -> np.ndarray:
    # OpenCV LAB for uint8: L 0..255, a/b offset by 128. Convert to standard-ish ranges.
    arr = np.uint8([[np.clip(np.round(rgb), 0, 255)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0], dtype=np.float32)


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.linalg.norm(lab1.astype(np.float32) - lab2.astype(np.float32)))


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIEDE2000 implementation for one LAB pair."""

    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]

    kL = kC = kH = 1.0
    C1 = (a1 * a1 + b1 * b1) ** 0.5
    C2 = (a2 * a2 + b2 * b2) ** 0.5
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - (C_bar ** 7 / (C_bar ** 7 + 25.0 ** 7)) ** 0.5) if C_bar > 0 else 0.0

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = (a1p * a1p + b1 * b1) ** 0.5
    C2p = (a2p * a2p + b2 * b2) ** 0.5

    def hp(ap: float, b: float) -> float:
        if ap == 0 and b == 0:
            return 0.0
        h = np.degrees(np.arctan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180.0:
            dh -= 360.0
        elif dh < -180.0:
            dh += 360.0
        dhp = dh
    dHp = 2.0 * (C1p * C2p) ** 0.5 * np.sin(np.radians(dhp / 2.0))

    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        hbp = h1p + h2p
    else:
        dh_abs = abs(h1p - h2p)
        if dh_abs <= 180.0:
            hbp = (h1p + h2p) / 2.0
        elif h1p + h2p < 360.0:
            hbp = (h1p + h2p + 360.0) / 2.0
        else:
            hbp = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbp))
        + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0))
    )
    delta_theta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    R_C = 2.0 * (Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7)) ** 0.5 if Cbp > 0 else 0.0
    S_L = 1.0 + (0.015 * ((Lbp - 50.0) ** 2)) / (20.0 + ((Lbp - 50.0) ** 2)) ** 0.5
    S_C = 1.0 + 0.045 * Cbp
    S_H = 1.0 + 0.015 * Cbp * T
    R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

    return float(
        (
            (dLp / (kL * S_L)) ** 2
            + (dCp / (kC * S_C)) ** 2
            + (dHp / (kH * S_H)) ** 2
            + R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
        )
        ** 0.5
    )


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    # CIEDE2000 is better perceptually. If any numerical issue happens, fall back to DeltaE76.
    try:
        return delta_e_2000(lab1, lab2)
    except Exception:
        return delta_e_76(lab1, lab2)


def rgb_luminance_chroma(rgb: np.ndarray) -> Tuple[float, float, float]:
    lab = rgb_to_lab_cv(rgb)
    L = float(lab[0])
    chroma = float((lab[1] ** 2 + lab[2] ** 2) ** 0.5)
    lum_rgb = float(np.mean(rgb))
    return L, chroma, lum_rgb


# -----------------------------------------------------------------------------
# Geometry detection
# -----------------------------------------------------------------------------


def find_colored_bar_by_mask(img_rgb: np.ndarray, color: str) -> Tuple[int, int, int, int, float]:
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    if color == "red":
        mask = cv2.inRange(hsv, (0, 45, 35), (14, 255, 255)) | cv2.inRange(hsv, (168, 45, 35), (180, 255, 255))
    elif color == "green":
        mask = cv2.inRange(hsv, (35, 35, 25), (92, 255, 245))
    elif color == "blue":
        mask = cv2.inRange(hsv, (92, 35, 25), (145, 255, 255))
    elif color == "gray":
        # Saturation low, medium brightness. The large gray bar is expected near the blue bar.
        mask = cv2.inRange(hsv, (0, 0, 40), (180, 65, 205))
    else:
        raise ValueError(f"Unsupported bar color: {color}")

    # Remove browser UI and small colored details by requiring a tall vertical rectangle.
    kernel = np.ones((13, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, int, int, int, int]] = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(bw * bh)
        if area < max(350.0, 0.0015 * h * w):
            continue
        if bh < 0.25 * h:
            continue
        if bw <= 0 or bh / max(1, bw) < 3.0:
            continue
        # Favor large vertical regions near the center of the photo, not phone/browser UI.
        center_penalty = 1.0 - min(abs((x + bw / 2.0) - (w / 2.0)) / max(w / 2.0, 1.0), 0.8) * 0.18
        candidates.append((area * center_penalty, x, y, bw, bh))

    if not candidates:
        raise ValueError(f"No se detecta la barra {color}")

    score, x, y, bw, bh = max(candidates, key=lambda item: item[0])
    return int(x), int(y), int(bw), int(bh), float(score)


def refine_bar_span_from_column_score(img_rgb: np.ndarray, x_center: int, y0: int, y1: int, mode: str) -> Tuple[int, int]:
    h, w = img_rgb.shape[:2]
    x0 = max(0, x_center - 3)
    x1 = min(w, x_center + 4)
    col = img_rgb[:, x0:x1].astype(np.float32).mean(axis=1)
    r, g, b = col[:, 0], col[:, 1], col[:, 2]
    if mode == "red":
        score = r - (g + b) / 2.0
    elif mode == "green":
        score = g - (r + b) / 2.0
    elif mode == "blue":
        score = b - (r + g) / 2.0
    else:
        saturation = col.max(axis=1) - col.min(axis=1)
        luminance = col.mean(axis=1)
        score = -saturation - 0.15 * np.abs(luminance - 128.0)
    score = smooth_1d(score, 23)

    # Restrict to the rough contour region first.
    lo = max(0, int(y0 - 0.08 * h))
    hi = min(h - 1, int(y1 + 0.08 * h))
    roi = score[lo:hi]
    if len(roi) < 5:
        return int(y0), int(y1)
    peak_y = int(lo + np.argmax(roi))
    peak = float(score[peak_y])
    if mode == "gray":
        threshold = peak - abs(peak) * 0.30
        is_inside = lambda v: v >= threshold
    else:
        threshold = max(8.0, peak * 0.35)
        is_inside = lambda v: v >= threshold

    top = peak_y
    while top > 0 and is_inside(float(score[top])):
        top -= 1
    bottom = peak_y
    while bottom < h - 1 and is_inside(float(score[bottom])):
        bottom += 1
    return int(top), int(bottom)


def local_edge_peaks(edge_profile: np.ndarray, y_min: int, y_max: int, min_height: float, min_dist: int) -> List[int]:
    y_min = max(1, y_min)
    y_max = min(len(edge_profile) - 2, y_max)
    candidates: List[int] = []
    for y in range(y_min, y_max):
        if edge_profile[y] >= edge_profile[y - 1] and edge_profile[y] > edge_profile[y + 1] and edge_profile[y] >= min_height:
            candidates.append(y)
    picked: List[int] = []
    for y in sorted(candidates, key=lambda yy: float(edge_profile[yy]), reverse=True):
        if all(abs(y - existing) >= min_dist for existing in picked):
            picked.append(y)
    return sorted(picked)


def _find_peak_span(score: np.ndarray, peak_idx: int, frac: float = 0.55, min_half_width: int = 6) -> Tuple[int, int]:
    peak = float(score[peak_idx])
    if peak <= 0:
        return max(0, peak_idx - min_half_width), min(len(score) - 1, peak_idx + min_half_width)
    threshold = peak * frac
    left = int(peak_idx)
    while left > 0 and score[left] >= threshold:
        left -= 1
    right = int(peak_idx)
    while right < len(score) - 1 and score[right] >= threshold:
        right += 1
    if right - left < 2 * min_half_width:
        left = max(0, peak_idx - min_half_width)
        right = min(len(score) - 1, peak_idx + min_half_width)
    return int(left), int(right)


def detect_geometry(img_rgb: np.ndarray) -> Dict[str, Any]:
    h, w = img_rgb.shape[:2]

    # Column-score detection is more robust than contour detection for this template,
    # because some sample pads can be blue/green/red-ish and may touch the bars after
    # morphology. We look for the dominant vertical color bands across the usable
    # central image area.
    ys0, ys1 = int(h * 0.15), int(h * 0.85)
    col = img_rgb[ys0:ys1].astype(np.float32).mean(axis=0)
    r, g, b = col[:, 0], col[:, 1], col[:, 2]

    red_score = smooth_1d(r - (g + b) / 2.0, 41)
    green_score = smooth_1d(g - (r + b) / 2.0, 41)
    blue_score = smooth_1d(b - (r + g) / 2.0, 41)

    red_x = int(np.argmax(red_score))
    green_x = int(np.argmax(green_score))
    blue_x = int(np.argmax(blue_score))

    if red_score[red_x] < 25 or green_score[green_x] < 15 or blue_score[blue_x] < 20:
        raise ValueError("No se detectan correctamente las franjas de referencia RGB")

    red_span = _find_peak_span(red_score, red_x, 0.55, 8)
    green_span = _find_peak_span(green_score, green_x, 0.55, 8)
    blue_span = _find_peak_span(blue_score, blue_x, 0.55, 8)

    if blue_x < green_x:
        orientation = "gray-blue-strip-green-red"
        strip_left = blue_span[1] + 8
        strip_right = green_span[0] - 8
        # Empirically the gray bar center is about 0.60-0.70 of the blue-green
        # center distance to the left of the blue bar in the fixed template.
        gray_expected_x = int(round(blue_x - 0.63 * abs(green_x - blue_x)))
    else:
        orientation = "red-green-strip-blue-gray"
        strip_left = green_span[1] + 8
        strip_right = blue_span[0] - 8
        gray_expected_x = int(round(blue_x + 0.63 * abs(green_x - blue_x)))

    strip_left = max(0, int(strip_left))
    strip_right = min(w - 1, int(strip_right))
    if strip_right <= strip_left or strip_right - strip_left < max(16, int(w * 0.015)):
        raise ValueError("No se pudo aislar la zona de la tira entre las franjas azul y verde")

    # Estimate bar widths from the colored spans. Gray is not saturated, so direct
    # detection against a white background is less reliable than template geometry.
    color_bar_w = int(np.median([red_span[1] - red_span[0], green_span[1] - green_span[0], blue_span[1] - blue_span[0]]))
    color_bar_w = max(12, color_bar_w)
    gray_span = (max(0, gray_expected_x - color_bar_w // 2), min(w - 1, gray_expected_x + color_bar_w // 2))

    # Refine shared top/bottom of the colored reference bars from vertical color profiles.
    rough_y0, rough_y1 = ys0, ys1
    refined_spans = []
    for name, x in [("blue", blue_x), ("green", green_x), ("red", red_x)]:
        refined_spans.append(refine_bar_span_from_column_score(img_rgb, x, rough_y0, rough_y1, name))
    bars_y_top = int(np.median([s[0] for s in refined_spans]))
    bars_y_bottom = int(np.median([s[1] for s in refined_spans]))
    if bars_y_bottom - bars_y_top < 0.25 * h:
        bars_y_top = ys0
        bars_y_bottom = ys1

    # Rectangles for reference-bar sampling.
    rects: Dict[str, Tuple[int, int, int, int]] = {
        "red": (int(red_span[0]), bars_y_top, int(red_span[1] - red_span[0]), int(bars_y_bottom - bars_y_top)),
        "green": (int(green_span[0]), bars_y_top, int(green_span[1] - green_span[0]), int(bars_y_bottom - bars_y_top)),
        "blue": (int(blue_span[0]), bars_y_top, int(blue_span[1] - blue_span[0]), int(bars_y_bottom - bars_y_top)),
        "gray": (int(gray_span[0]), bars_y_top, int(gray_span[1] - gray_span[0]), int(bars_y_bottom - bars_y_top)),
    }
    centers_x = {
        "red": red_x,
        "green": green_x,
        "blue": blue_x,
        "gray": int((gray_span[0] + gray_span[1]) / 2),
    }

    strip_center_x = int((strip_left + strip_right) / 2)
    strip_width = int(strip_right - strip_left)

    # Detect top/bottom of the 10 pads using horizontal edge energy around the strip.
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    band_half = max(6, int(strip_width * 0.30))
    band = gray_img[:, max(0, strip_center_x - band_half):min(w, strip_center_x + band_half + 1)]
    dy = np.abs(np.diff(band, axis=0)).mean(axis=1)
    edge_profile = smooth_1d(dy, 7)
    y_min = max(0, bars_y_top - int(0.08 * h))
    y_max = min(h - 2, bars_y_bottom + int(0.08 * h))
    min_height = max(3.5, float(np.percentile(edge_profile[y_min:y_max], 90)) * 0.60)
    peaks = local_edge_peaks(edge_profile, y_min, y_max, min_height=min_height, min_dist=max(10, int((bars_y_bottom - bars_y_top) / 16)))

    # The physical template fixes the strip stack height to the reference bars.
    # Edge peaks are useful diagnostics, but in real phone photos they can also
    # pick up browser UI, sticker shadows or the white handle. For production
    # sampling we therefore use the calibrated bar vertical span as the pad span.
    pads_y_top = bars_y_top
    pads_y_bottom = bars_y_bottom

    return {
        "orientation": orientation,
        "rects": {k: tuple(int(vv) for vv in v) for k, v in rects.items()},
        "centers_x": {k: int(v) for k, v in centers_x.items()},
        "strip_left": int(strip_left),
        "strip_right": int(strip_right),
        "strip_center_x": int(strip_center_x),
        "strip_width": int(strip_width),
        "bars_y_top": int(bars_y_top),
        "bars_y_bottom": int(bars_y_bottom),
        "pads_y_top": int(pads_y_top),
        "pads_y_bottom": int(pads_y_bottom),
        "edge_peaks_count": len(peaks),
        "edge_peaks": [int(v) for v in peaks[:20]],
    }

def sample_bar_color(img_rgb: np.ndarray, rect: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, float]:
    x, y, w, h = rect
    # Use the central portion only, avoiding borders and shadows.
    mx = max(2, int(w * 0.22))
    my = max(8, int(h * 0.18))
    return robust_patch_stats_rgb(img_rgb, x + mx, y + my, x + w - mx, y + h - my, reject_bright_pct=92.0, reject_dark_pct=2.0)


def sample_pad_rects(geom: Dict[str, Any]) -> List[Tuple[int, int, int, int, int, int]]:
    strip_left = int(geom["strip_left"])
    strip_right = int(geom["strip_right"])
    strip_center_x = int(geom["strip_center_x"])
    strip_width = max(10, int(strip_right - strip_left))
    pads_top = int(geom["pads_y_top"])
    pads_bottom = int(geom["pads_y_bottom"])
    span = max(20, pads_bottom - pads_top)
    pitch = span / 10.0

    # Samples should be small and central to avoid pad borders, glue, shadows and the plastic separator.
    sample_w = max(7, min(24, int(strip_width * 0.42)))
    sample_h = max(7, min(20, int(pitch * 0.32)))

    rects: List[Tuple[int, int, int, int, int, int]] = []
    for i, param in enumerate(PARAM_ORDER):
        cy = int(round(pads_top + (i + 0.5) * pitch))
        cx = strip_center_x

        # The last pad is usually close to the white handle; lift slightly to avoid the handle boundary.
        if param == "chloride":
            cy = int(round(cy - 0.10 * pitch))

        # The white/very pale tests are particularly sensitive to border shadows.
        w = sample_w
        h = sample_h
        if param in WHITE_LIKE_PARAMS:
            w = max(6, int(w * 0.85))
            h = max(6, int(h * 0.85))
        rects.append((int(cx - w / 2), int(cy - h / 2), int(w), int(h), int(cx), int(cy)))
    return rects


# -----------------------------------------------------------------------------
# Matching and status
# -----------------------------------------------------------------------------


def classify_result(param: str, numeric: float, meta: Dict[str, Any]) -> str:
    danger = meta.get("danger") or []
    recommended = meta.get("recommended") or []

    # Values printed in red in the PDF are maximum limits. When a danger limit is
    # given, values at or above that limit are classified as peligro.
    if danger:
        threshold = min(float(v) for v in danger)
        if numeric >= threshold:
            return "peligro"

    if len(recommended) == 2:
        lo, hi = float(recommended[0]), float(recommended[1])
        if lo <= numeric <= hi:
            return "ok"
        return "fuera_de_rango"

    return "ok"


def confidence_from_distances(best: float, second: float, quality_score: float) -> str:
    gap = max(0.0, second - best)
    if quality_score < 0.38:
        return "low"
    if best <= 6.0 and gap >= 2.0:
        return "high"
    if best <= 12.0 and gap >= 1.5:
        return "medium"
    if best <= 18.0 and gap >= 3.0:
        return "medium"
    return "low"


def build_swatch_distances(rgb: np.ndarray, meta: Dict[str, Any]) -> List[Tuple[float, int]]:
    lab = rgb_to_lab_cv(rgb)
    distances: List[Tuple[float, int]] = []
    for idx, ref_rgb in enumerate(meta["rgb"]):
        ref_lab = rgb_to_lab_cv(np.array(ref_rgb, dtype=np.float32))
        distances.append((delta_e(lab, ref_lab), idx))
    return sorted(distances, key=lambda x: x[0])


def match_swatch(param: str, raw_rgb: np.ndarray, cal_rgb: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    raw_distances = build_swatch_distances(raw_rgb, meta)
    cal_distances = build_swatch_distances(cal_rgb, meta)

    # Default to calibrated color. Use raw only when calibration clearly makes a
    # pale patch worse or clips too aggressively.
    cal_clip_count = int(np.sum((cal_rgb <= 2) | (cal_rgb >= 253)))
    raw_best, raw_idx = raw_distances[0]
    cal_best, cal_idx = cal_distances[0]
    mode = "calibrated"
    distances = cal_distances
    used_rgb = cal_rgb

    L_cal, chroma_cal, lum_cal = rgb_luminance_chroma(cal_rgb)
    L_raw, chroma_raw, lum_raw = rgb_luminance_chroma(raw_rgb)

    if param in WHITE_LIKE_PARAMS:
        # If the photographed pad is near-white, favor the lowest values. This
        # prevents small shadows/phone white balance from being interpreted as a
        # higher pink/yellow concentration.
        numeric_values = meta.get("numeric_values") or [numeric_value(v) for v in meta["values"]]
        lowest_idx = int(np.argmin(np.array(numeric_values, dtype=np.float32)))
        if (L_cal >= 91.0 and chroma_cal <= 7.0) or (L_raw >= 90.0 and chroma_raw <= 7.5):
            forced_distances = [(delta_e(rgb_to_lab_cv(cal_rgb), rgb_to_lab_cv(np.array(meta["rgb"][lowest_idx], dtype=np.float32))), lowest_idx)]
            for d, idx in cal_distances:
                if idx != lowest_idx:
                    forced_distances.append((d, idx))
            distances = forced_distances
            mode = "calibrated_white_bias"
            used_rgb = cal_rgb
        elif cal_clip_count >= 2 and raw_best < cal_best + 4.0:
            mode = "raw_white_fallback"
            distances = raw_distances
            used_rgb = raw_rgb
    else:
        if cal_clip_count >= 2 and raw_best + 3.0 < cal_best:
            mode = "raw_clip_fallback"
            distances = raw_distances
            used_rgb = raw_rgb

    distances = sorted(distances, key=lambda x: x[0])
    best, idx = distances[0]
    second = distances[1][0] if len(distances) > 1 else 99.0
    value = meta["values"][idx]
    numeric = (meta.get("numeric_values") or [numeric_value(v) for v in meta["values"]])[idx]

    return {
        "index": int(idx),
        "value": value,
        "numeric_value": float(numeric),
        "reference_rgb": meta["rgb"][idx],
        "deltaE": float(best),
        "deltaE2": float(second),
        "mode": mode,
        "used_rgb": used_rgb,
    }


# -----------------------------------------------------------------------------
# History storage
# -----------------------------------------------------------------------------


def init_history_db() -> None:
    try:
        HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(HISTORY_DB_PATH) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    client_id TEXT,
                    scan_id TEXT,
                    operator_id TEXT,
                    location TEXT,
                    status TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_analyses_client_id ON analyses(client_id)")
            con.commit()
    except Exception:
        # Do not fail app startup if DB is read-only. The endpoint will return analysis without history.
        pass


def save_analysis(req: AnalyzeReq, result: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    analysis_id = str(uuid.uuid4())
    try:
        HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(HISTORY_DB_PATH) as con:
            con.execute(
                """
                INSERT INTO analyses
                (id, created_at, client_id, scan_id, operator_id, location, status, quality_score, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    utc_now_iso(),
                    req.client_id,
                    req.scan_id,
                    req.operator_id,
                    req.location,
                    result.get("status", "unknown"),
                    float(result.get("quality_score", 0.0)),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            con.commit()
        return analysis_id, None
    except Exception as exc:
        return None, str(exc)


@app.on_event("startup")
def startup() -> None:
    init_history_db()


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------


def analyze_image(img_rgb: np.ndarray, debug: bool = False) -> Dict[str, Any]:
    swatches = load_swatches()
    h, w = img_rgb.shape[:2]
    geom = detect_geometry(img_rgb)

    measured_refs: Dict[str, np.ndarray] = {}
    bar_rejections: Dict[str, Dict[str, float]] = {}
    for name in ["gray", "blue", "green", "red"]:
        rgb, bright_rej, dark_rej = sample_bar_color(img_rgb, geom["rects"][name])
        measured_refs[name] = rgb
        bar_rejections[name] = {
            "brightRejectedPct": round(float(bright_rej), 2),
            "darkRejectedPct": round(float(dark_rej), 2),
        }

    cal_params = fit_per_channel_calibration(measured_refs)

    def calibrate(rgb: np.ndarray) -> np.ndarray:
        return apply_per_channel_calibration(rgb, cal_params)

    # Reference calibration residual. This cannot be zero with per-channel model,
    # which is good: it still detects unstable photos.
    ref_errors_de: Dict[str, float] = {}
    ref_corrected: Dict[str, List[int]] = {}
    for name, observed in measured_refs.items():
        corrected = calibrate(observed)
        ref_corrected[name] = to_int_list(corrected)
        ref_errors_de[name] = round(delta_e(rgb_to_lab_cv(corrected), rgb_to_lab_cv(np.array(TARGET_BARS_RGB[name], dtype=np.float32))), 2)

    ref_error_mean = float(np.mean(list(ref_errors_de.values())))
    ref_error_max = float(np.max(list(ref_errors_de.values())))

    sample_rects = sample_pad_rects(geom)
    results: List[Dict[str, Any]] = []
    match_errors: List[float] = []
    warnings: List[str] = []
    low_conf_count = 0

    for i, (param, sample) in enumerate(zip(PARAM_ORDER, sample_rects)):
        x, y, sw, sh, cx, cy = sample
        raw_rgb, bright_rej, dark_rej = robust_patch_stats_rgb(
            img_rgb,
            x,
            y,
            x + sw,
            y + sh,
            reject_bright_pct=88.0,
            reject_dark_pct=2.0,
        )
        cal_rgb = calibrate(raw_rgb)
        meta = swatches[param]
        match = match_swatch(param, raw_rgb, cal_rgb, meta)
        match_errors.append(float(match["deltaE"]))

        # Temporary confidence; refined after global quality is known.
        confidence_tmp = "low"
        if match["deltaE"] < 8 and match["deltaE2"] - match["deltaE"] > 2:
            confidence_tmp = "high"
        elif match["deltaE"] < 15 and match["deltaE2"] - match["deltaE"] > 1:
            confidence_tmp = "medium"
        if confidence_tmp == "low":
            low_conf_count += 1

        numeric = float(match["numeric_value"])
        status = classify_result(param, numeric, meta)
        used_rgb = match["used_rgb"]

        results.append(
            {
                "index": i + 1,
                "parameter": param,
                "label": meta.get("label", param),
                "short_label": meta.get("short_label", meta.get("label", param)),
                "value": str(match["value"]),
                "numeric_value": numeric,
                "unit": meta.get("unit", ""),
                "status": status,
                "confidence": confidence_tmp,
                "deltaE": round(float(match["deltaE"]), 2),
                "deltaE2": round(float(match["deltaE2"]), 2),
                "mode": match["mode"],
                "reference_rgb": match["reference_rgb"],
                "sample_point": {"x": int(cx), "y": int(cy)},
                "sample_rect": {"x": int(x), "y": int(y), "w": int(sw), "h": int(sh)},
                "sample_rgb_raw": to_int_list(raw_rgb),
                "sample_rgb_calibrated": to_int_list(cal_rgb),
                "sample_rgb_used": to_int_list(used_rgb),
                "glareRejectedPct": round(float(bright_rej), 2),
                "darkRejectedPct": round(float(dark_rej), 2),
            }
        )

    # Quality score combines reference calibration stability, match distance and geometry.
    match_mean = float(np.mean(match_errors)) if match_errors else 99.0
    match_penalty = min(match_mean / 20.0, 1.0)
    ref_penalty = min(ref_error_mean / 18.0, 1.0)
    edge_penalty = 0.0 if geom["edge_peaks_count"] >= 8 else min((8 - geom["edge_peaks_count"]) / 8.0, 1.0)
    strip_penalty = 0.0 if geom["strip_width"] >= 24 else 0.25

    quality_score = max(0.0, 1.0 - (0.48 * match_penalty + 0.36 * ref_penalty + 0.11 * edge_penalty + 0.05 * strip_penalty))
    quality_score = round(float(quality_score), 3)

    # Update confidence labels using final quality.
    low_conf_count = 0
    for r in results:
        r["confidence"] = confidence_from_distances(float(r["deltaE"]), float(r["deltaE2"]), quality_score)
        if r["confidence"] == "low":
            low_conf_count += 1

    if ref_error_mean > 13.0:
        warnings.append("Calibración de color sensible: la luz no parece uniforme sobre la plantilla")
    if ref_error_max > 25.0:
        warnings.append("Una o más barras de referencia no calibran bien; posible sombra, brillo o desenfoque")
    if geom["edge_peaks_count"] < 8:
        warnings.append("Detección débil de separaciones entre pads; posible foto desenfocada o tira mal colocada")
    if low_conf_count >= 4:
        warnings.append("Varias coincidencias tienen baja confianza; conviene repetir la foto con luz más uniforme")
    if geom["strip_width"] < 24:
        warnings.append("La tira aparece estrecha; acerca la cámara o usa mayor resolución")

    photo_status = "ok" if quality_score >= 0.42 else "foto_no_fiable"
    ok = photo_status == "ok"

    diagnostics: Dict[str, Any] = {
        "version": APP_VERSION,
        "imageSize": [int(w), int(h)],
        "foundBars": True,
        "foundPads": 10,
        "transformType": "per_channel_linear_srgb",
        "deltaEFormula": "CIEDE2000",
        "referenceErrorMean": round(ref_error_mean, 2),
        "referenceErrorMax": round(ref_error_max, 2),
        "referenceErrorsDeltaE": ref_errors_de,
        "barsObservedRGB": {k: to_int_list(v) for k, v in measured_refs.items()},
        "barsCorrectedRGB": ref_corrected,
        "barsTargetRGB": TARGET_BARS_RGB,
        "barsRejectedPct": bar_rejections,
        "warnings": warnings,
        "geometry": {
            "orientation": geom["orientation"],
            "stripX": [int(geom["strip_left"]), int(geom["strip_right"])],
            "stripWidth": int(geom["strip_width"]),
            "barsY": [int(geom["bars_y_top"]), int(geom["bars_y_bottom"])],
            "padsY": [int(geom["pads_y_top"]), int(geom["pads_y_bottom"])],
            "barRects": {k: [int(x) for x in v] for k, v in geom["rects"].items()},
            "barCentersX": {k: int(v) for k, v in geom["centers_x"].items()},
            "edgePeaksCount": int(geom["edge_peaks_count"]),
            "edgePeaks": geom["edge_peaks"] if debug else [],
        },
    }
    if debug:
        diagnostics["calibrationParams"] = cal_params.round(6).tolist()
        diagnostics["sampleRects"] = [
            {"parameter": p, "x": int(x), "y": int(y), "w": int(sw), "h": int(sh), "cx": int(cx), "cy": int(cy)}
            for p, (x, y, sw, sh, cx, cy) in zip(PARAM_ORDER, sample_rects)
        ]

    return {
        "ok": ok,
        "status": photo_status,
        "quality_score": quality_score,
        "analysis_id": None,
        "orientation": geom["orientation"],
        "results": results,
        "diagnostics": diagnostics,
        "retake_reason": None if ok else "La foto no es suficientemente fiable para una medición automática",
        "retake_tips": []
        if ok and not warnings
        else [
            "Usa luz uniforme y evita reflejos directos sobre la tira",
            "Mantén la cámara perpendicular a la plantilla",
            "Centra la tira entre las barras azul y verde",
            "Asegura enfoque nítido y que se vean completas las barras de referencia",
        ],
    }


def failure_response(img_rgb: Optional[np.ndarray], message: str) -> Dict[str, Any]:
    size = [0, 0]
    if img_rgb is not None:
        h, w = img_rgb.shape[:2]
        size = [int(w), int(h)]
    return {
        "ok": False,
        "status": "foto_no_fiable",
        "quality_score": 0.0,
        "analysis_id": None,
        "orientation": None,
        "results": [],
        "diagnostics": {
            "version": APP_VERSION,
            "imageSize": size,
            "foundBars": False,
            "foundPads": 0,
            "warnings": [message],
        },
        "retake_reason": "No se pudo analizar la tira en esta imagen",
        "retake_tips": [
            "Asegura que la plantilla completa aparece en la foto",
            "Alinea la tira entre las franjas azul y verde",
            "Evita sombras/reflejos fuertes",
            "Haz la foto perpendicular y con buen enfoque",
        ],
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "ColorScale API", "version": APP_VERSION}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": APP_VERSION,
        "swatchesLoaded": SWATCHES_PATH.exists(),
        "swatchesPath": str(SWATCHES_PATH),
        "historyDbPath": str(HISTORY_DB_PATH),
    }


@app.post("/analyze-strip", response_model=AnalyzeResponse)
def analyze_strip(req: AnalyzeReq, x_api_key: str = Header(default="")) -> Dict[str, Any]:
    check_api_key(x_api_key)
    img_rgb: Optional[np.ndarray] = None
    start = time.time()
    try:
        img_rgb = load_image_from_request(req)
        result = analyze_image(img_rgb, debug=bool(req.debug))
    except HTTPException:
        raise
    except Exception as exc:
        result = failure_response(img_rgb, str(exc))

    result["diagnostics"]["processingMs"] = int(round((time.time() - start) * 1000))

    save_history = SAVE_HISTORY_DEFAULT if req.save_history is None else bool(req.save_history)
    if save_history:
        analysis_id, err = save_analysis(req, result)
        result["analysis_id"] = analysis_id
        if err:
            result.setdefault("diagnostics", {}).setdefault("warnings", []).append(f"No se pudo guardar histórico: {err}")
    return result


@app.get("/history")
def history(
    client_id: Optional[str] = Query(default=None),
    scan_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    x_api_key: str = Header(default=""),
) -> Dict[str, Any]:
    check_api_key(x_api_key)
    init_history_db()
    where: List[str] = []
    params: List[Any] = []
    if client_id:
        where.append("client_id = ?")
        params.append(client_id)
    if scan_id:
        where.append("scan_id = ?")
        params.append(scan_id)
    sql = "SELECT id, created_at, client_id, scan_id, operator_id, location, status, quality_score, result_json FROM analyses"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    try:
        with sqlite3.connect(HISTORY_DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(sql, params).fetchall()
        items = []
        for row in rows:
            payload = json.loads(row["result_json"])
            items.append(
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "client_id": row["client_id"],
                    "scan_id": row["scan_id"],
                    "operator_id": row["operator_id"],
                    "location": row["location"],
                    "status": row["status"],
                    "quality_score": row["quality_score"],
                    "summary": [
                        {
                            "parameter": r.get("parameter"),
                            "label": r.get("short_label") or r.get("label"),
                            "value": r.get("value"),
                            "unit": r.get("unit"),
                            "status": r.get("status"),
                            "confidence": r.get("confidence"),
                        }
                        for r in payload.get("results", [])
                    ],
                }
            )
        return {"ok": True, "items": items}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read history: {exc}") from exc


@app.get("/history/{analysis_id}")
def history_detail(analysis_id: str, x_api_key: str = Header(default="")) -> Dict[str, Any]:
    check_api_key(x_api_key)
    init_history_db()
    try:
        with sqlite3.connect(HISTORY_DB_PATH) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT id, created_at, client_id, scan_id, operator_id, location, status, quality_score, result_json FROM analyses WHERE id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return {
            "ok": True,
            "id": row["id"],
            "created_at": row["created_at"],
            "client_id": row["client_id"],
            "scan_id": row["scan_id"],
            "operator_id": row["operator_id"],
            "location": row["location"],
            "status": row["status"],
            "quality_score": row["quality_score"],
            "result": json.loads(row["result_json"]),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read analysis: {exc}") from exc
