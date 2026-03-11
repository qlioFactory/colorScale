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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SWATCHES_PATH = os.getenv("SWATCHES_PATH", os.path.join(BASE_DIR, "swatches.json"))

# Orden actual (pendiente de confirmar con el cliente antes de cambiarlo)
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

TARGET_BARS_RGB = {
    "ROJO": [255, 0, 0],
    "VERDE": [0, 128, 0],
    "AZUL": [0, 0, 255],
    "GRIS": [128, 128, 128],
}
TARGET_WHITE_RGB = [255, 255, 255]

app = FastAPI(title="ColorScale API", version="0.3.0")

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


def local_max_peaks(signal: np.ndarray, y_min: int, y_max: int, min_height: float, min_dist: int = 15) -> List[int]:
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


# ---------- Robust sampling (glare reject) ----------
def robust_patch_stats_rgb(img_rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, float]:
    """
    Devuelve (median_rgb, glare_rejected_pct).
    Rechaza el 10% más brillante (reflejos).
    """
    h, w = img_rgb.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h, int(y1)))

    if x1 <= x0 or y1 <= y0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32), 0.0

    patch = img_rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if patch.shape[0] < 10:
        return np.median(patch, axis=0).astype(np.float32), 0.0

    lum = patch.mean(axis=1)
    thr = np.percentile(lum, 90)  # rechazar top 10% brillo
    mask = lum <= thr
    good = patch[mask]
    rejected_pct = 100.0 * float(np.sum(~mask)) / float(mask.shape[0])

    if good.shape[0] < 10:
        good = patch
        rejected_pct = 0.0

    return np.median(good, axis=0).astype(np.float32), float(rejected_pct)


def sample_patch_median(img_rgb: np.ndarray, cx: int, cy: int, rx: int = 6, ry: int = 8) -> Tuple[np.ndarray, float]:
    x0, x1 = cx - rx, cx + rx + 1
    y0, y1 = cy - ry, cy + ry + 1
    return robust_patch_stats_rgb(img_rgb, x0, y0, x1, y1)


def sample_bar_color(img_rgb: np.ndarray, x_center: int, y_top: int, y_bottom: int, rx: int = 10) -> Tuple[np.ndarray, float]:
    margin = max(10, int((y_bottom - y_top) * 0.12))
    y0 = y_top + margin
    y1 = y_bottom - margin
    x0 = x_center - rx
    x1 = x_center + rx + 1
    return robust_patch_stats_rgb(img_rgb, x0, y0, x1, y1)


def rgb_lum_sat(rgb: np.ndarray) -> Tuple[float, float]:
    rgb = np.asarray(rgb, dtype=np.float32)
    lum = float(rgb.mean())
    sat = float(rgb.max() - rgb.min())
    return lum, sat


def whiteness_score(rgb: np.ndarray) -> float:
    lum, sat = rgb_lum_sat(rgb)
    return lum - 1.2 * sat


# ---------- Color space helpers (linear RGB) ----------
def srgb_to_linear_u8(rgb_u8: np.ndarray) -> np.ndarray:
    x = np.clip(rgb_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    a = 0.055
    lin = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
    return lin


def linear_to_srgb_u8(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin.astype(np.float32), 0.0, 1.0)
    a = 0.055
    x = np.where(lin <= 0.0031308, 12.92 * lin, (1 + a) * (lin ** (1 / 2.4)) - a)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


# ---------- Matching (DeltaE76) ----------
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb_u8 = np.uint8([[np.clip(np.round(rgb), 0, 255).astype(np.uint8)]])
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0], dtype=np.float32)


def deltae76(rgb1: np.ndarray, rgb2: List[int]) -> float:
    l1 = rgb_to_lab(np.asarray(rgb1, dtype=np.float32))
    l2 = rgb_to_lab(np.asarray(rgb2, dtype=np.float32))
    return float(np.linalg.norm(l1 - l2))


def confidence_label(best: float, second: float) -> str:
    gap = float(second - best)
    if best < 10 and gap >= 4:
        return "high"
    if best < 22 and gap >= 2:
        return "medium"
    return "low"


def match_swatch(param: str, rgb: np.ndarray, swatches: Dict[str, Any]) -> Dict[str, Any]:
    ds = []
    for sw in swatches.get(param, []):
        d = deltae76(rgb, sw["rgb"])
        ds.append((d, sw))
    if not ds:
        return {"value": None, "reference_rgb": None, "deltaE": 999.0, "deltaE2": 999.0}

    ds.sort(key=lambda t: t[0])
    best_d, best_sw = ds[0]
    second_d = ds[1][0] if len(ds) > 1 else 999.0

    return {
        "value": best_sw["value"],
        "reference_rgb": best_sw["rgb"],
        "deltaE": float(best_d),
        "deltaE2": float(second_d),
    }


# ---------- Geometry detection (same as before) ----------
def detect_geometry(img_rgb: np.ndarray) -> Dict[str, Any]:
    h, w = img_rgb.shape[:2]

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


# ---------- Calibration models ----------
def fit_per_channel_linear(obs_bars_rgb: Dict[str, np.ndarray], obs_white_rgb: np.ndarray) -> np.ndarray:
    """
    Ajuste lineal por canal en linear RGB: out_lin = a*in_lin + b
    Usamos 5 puntos: ROJO/VERDE/AZUL/GRIS/BLANCO.
    """
    names = ["ROJO", "VERDE", "AZUL", "GRIS"]
    obs = np.array([obs_bars_rgb[n] for n in names] + [obs_white_rgb], dtype=np.float32)
    tgt = np.array([TARGET_BARS_RGB[n] for n in names] + [TARGET_WHITE_RGB], dtype=np.float32)

    obs_lin = srgb_to_linear_u8(obs)
    tgt_lin = srgb_to_linear_u8(tgt)

    params = np.zeros((3, 2), dtype=np.float32)
    for c in range(3):
        x = obs_lin[:, c]
        y = tgt_lin[:, c]
        A = np.vstack([x, np.ones_like(x)]).T
        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        params[c, 0] = float(sol[0])  # a
        params[c, 1] = float(sol[1])  # b
    return params


def apply_per_channel(rgb: np.ndarray, params: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    lin = srgb_to_linear_u8(rgb)
    out_lin = np.array([
        params[0, 0] * lin[0] + params[0, 1],
        params[1, 0] * lin[1] + params[1, 1],
        params[2, 0] * lin[2] + params[2, 1],
    ], dtype=np.float32)
    out_u8 = linear_to_srgb_u8(out_lin)
    return out_u8.astype(np.float32)


def fit_affine_3x3(obs_bars_rgb: Dict[str, np.ndarray], obs_white_rgb: np.ndarray) -> np.ndarray:
    """
    Ajuste afín en linear RGB:
    [r g b 1] @ M(4x3) = [R' G' B']  (todo en 0..1 linear)
    """
    names = ["ROJO", "VERDE", "AZUL", "GRIS"]
    obs = np.array([obs_bars_rgb[n] for n in names] + [obs_white_rgb], dtype=np.float32)
    tgt = np.array([TARGET_BARS_RGB[n] for n in names] + [TARGET_WHITE_RGB], dtype=np.float32)

    obs_lin = srgb_to_linear_u8(obs)
    tgt_lin = srgb_to_linear_u8(tgt)

    X = np.concatenate([obs_lin, np.ones((obs_lin.shape[0], 1), dtype=np.float32)], axis=1)  # Nx4
    Y = tgt_lin  # Nx3
    M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # 4x3
    return M.astype(np.float32)


def apply_affine(rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    lin = srgb_to_linear_u8(rgb)
    x = np.array([lin[0], lin[1], lin[2], 1.0], dtype=np.float32)  # 4
    out_lin = x @ M  # 3
    out_u8 = linear_to_srgb_u8(out_lin)
    return out_u8.astype(np.float32)


def calibration_error(obs_points: List[Tuple[str, np.ndarray]], apply_fn) -> Tuple[float, float, Dict[str, List[int]]]:
    """
    Calcula error medio y max (L2 en sRGB) contra los targets para puntos de calibración.
    Retorna también puntos corregidos (para diagnóstico).
    """
    errs = []
    corr_map = {}
    for name, rgb_obs in obs_points:
        rgb_corr = apply_fn(rgb_obs)
        tgt = np.array(TARGET_BARS_RGB.get(name, TARGET_WHITE_RGB), dtype=np.float32)
        e = float(np.linalg.norm(rgb_corr - tgt))
        errs.append(e)
        corr_map[name] = [int(round(x)) for x in rgb_corr.tolist()]
    return float(np.mean(errs)), float(np.max(errs)), corr_map


def estimate_white_from_background(img_rgb: np.ndarray, geom: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Busca un blanco de referencia en el fondo (cartulina) en varios puntos.
    Devuelve (rgb, score). Si score es bajo, es blanco poco fiable.
    """
    h, w = img_rgb.shape[:2]
    y = int((geom["bars_y_top"] + geom["bars_y_bottom"]) / 2)
    candidates = [
        (int(0.08 * w), y),
        (int(0.92 * w), y),
        (int(0.50 * w), int(0.10 * h)),
        (int(0.50 * w), int(0.90 * h)),
    ]

    best_rgb = None
    best_score = -1e9

    for (cx, cy) in candidates:
        rgb, rej = sample_patch_median(img_rgb, cx, cy, rx=16, ry=16)
        score = whiteness_score(rgb)
        if score > best_score:
            best_score = score
            best_rgb = rgb

    if best_rgb is None:
        best_rgb = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        best_score = -999.0

    return best_rgb.astype(np.float32), float(best_score)


def local_white_near_pad(img_rgb: np.ndarray, x: int, cy: int, pitch: float) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Estima blanco local por fila (para corregir gradientes): muestrea arriba y abajo del pad
    (en el espacio blanco entre pads). Devuelve (rgb or None, whiteness_score, glareRejectedPct).
    """
    off = int(round(0.35 * pitch))
    candidates = [(x, cy - off), (x, cy + off)]
    best = None
    best_score = -1e9
    best_rej = 0.0

    for (cx, yy) in candidates:
        rgb, rej = sample_patch_median(img_rgb, cx, yy, rx=10, ry=8)
        score = whiteness_score(rgb)
        # aceptar solo si parece "bastante blanco"
        lum, sat = rgb_lum_sat(rgb)
        if lum < 140 or sat > 70:
            continue
        if score > best_score:
            best_score = score
            best = rgb
            best_rej = rej

    return (best.astype(np.float32) if best is not None else None), float(best_score), float(best_rej)


def apply_local_white_correction(pad_rgb_corr: np.ndarray, white_local_corr: np.ndarray) -> np.ndarray:
    """
    Corrige por fila: ajusta el pad corregido globalmente usando el blanco local corregido.
    Operación en linear: pad *= (white_target / white_local)
    Limita el factor para no amplificar ruido.
    """
    eps = 1e-4
    pad_lin = srgb_to_linear_u8(pad_rgb_corr)
    wl_lin = srgb_to_linear_u8(white_local_corr)
    wt_lin = srgb_to_linear_u8(np.array(TARGET_WHITE_RGB, dtype=np.float32))

    scale = wt_lin / np.maximum(wl_lin, eps)
    scale = np.clip(scale, 0.70, 1.40)  # limitar

    out_lin = np.clip(pad_lin * scale, 0.0, 1.0)
    out_u8 = linear_to_srgb_u8(out_lin)
    return out_u8.astype(np.float32)


# ---------- Main analysis ----------
def analyze_image(img_bgr: np.ndarray, debug: bool = False) -> Dict[str, Any]:
    swatches = load_swatches()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    geom = detect_geometry(img_rgb)

    # Barras observadas (robust)
    obs_bars = {}
    obs_bars_rej = {}
    for k, x in [("ROJO", geom["red_x"]), ("VERDE", geom["green_x"]), ("AZUL", geom["blue_x"]), ("GRIS", geom["gray_x"])]:
        rgb, rej = sample_bar_color(img_rgb, x, geom["bars_y_top"], geom["bars_y_bottom"], rx=12)
        obs_bars[k] = rgb
        obs_bars_rej[k] = rej

    # Blanco global observado
    obs_white, white_score = estimate_white_from_background(img_rgb, geom)

    # Modelo por canal (siempre disponible)
    ch_params = fit_per_channel_linear(obs_bars, obs_white)
    apply_ch = lambda rgb: apply_per_channel(rgb, ch_params)

    # Modelo 3x3+offset (si blanco es fiable)
    lumW, satW = rgb_lum_sat(obs_white)
    white_reliable = (white_score > 120.0 and lumW > 160.0 and satW < 60.0)

    model_type = "per_channel"
    affine_M = None
    apply_cal = apply_ch

    # Error del modelo por canal (en puntos de calibración)
    cal_points = [(k, obs_bars[k]) for k in ["ROJO", "VERDE", "AZUL", "GRIS"]] + [("BLANCO", obs_white)]
    err_ch_mean, err_ch_max, corr_ch_map = calibration_error(cal_points, apply_ch)

    err_aff_mean = None
    err_aff_max = None
    corr_aff_map = None

    if white_reliable:
        affine_M = fit_affine_3x3(obs_bars, obs_white)
        apply_aff = lambda rgb: apply_affine(rgb, affine_M)
        err_aff_mean, err_aff_max, corr_aff_map = calibration_error(cal_points, apply_aff)

        # Selección: usar affine solo si mejora o no empeora
        if err_aff_mean is not None and err_aff_mean <= (err_ch_mean * 0.95):
            model_type = "affine_3x3"
            apply_cal = apply_aff
        else:
            model_type = "per_channel"
            apply_cal = apply_ch

    # Retake si calibración global mala (sombras/reflejos severos)
    calib_err_mean = float(err_aff_mean if model_type == "affine_3x3" else err_ch_mean)
    calib_err_max = float(err_aff_max if model_type == "affine_3x3" else err_ch_max)

    if calib_err_mean > 35.0 or calib_err_max > 55.0:
        return {
            "ok": False,
            "orientation": geom["orientation"],
            "results": [],
            "diagnostics": {
                "imageSize": [int(w), int(h)],
                "foundBars": True,
                "transformType": model_type,
                "calibrationErrorMean": round(calib_err_mean, 2),
                "calibrationErrorMax": round(calib_err_max, 2),
                "barsObservedRGB": {k: [int(round(x)) for x in v.tolist()] for k, v in obs_bars.items()},
                "barsCorrectedRGB": (corr_aff_map if model_type == "affine_3x3" else corr_ch_map),
                "whiteObservedRGB": [int(round(x)) for x in obs_white.tolist()],
                "whiteReliable": bool(white_reliable),
                "whiteScore": round(float(white_score), 2),
                "warnings": ["Calibración global inestable (posibles reflejos/sombras)."],
            },
            "retake_reason": "Iluminación no estable (sombras/reflejos). Repite la foto.",
            "retake_tips": [
                "Evita reflejos directos sobre la tira",
                "Usa luz uniforme",
                "Alinea la tira centrada entre las franjas azul y verde",
                "Asegura buen enfoque",
            ],
        }

    pads_top = int(geom["pads_y_top"])
    pads_bottom = int(geom["pads_y_bottom"])
    pads_h = max(10, pads_bottom - pads_top)
    pitch = pads_h / 10.0

    strip_center_x = int(geom["strip_center_x"])
    strip_width = max(10, int(geom["strip_right"] - geom["strip_left"]))
    rx = max(5, min(10, strip_width // 10))
    ry = max(6, min(12, int(pitch * 0.14)))

    results = []
    warnings = []
    low_conf_count = 0

    for i, param in enumerate(PARAM_ORDER):
        cy = int(round(pads_top + (i + 0.5) * pitch))

        rgb_raw, rej_pad = sample_patch_median(img_rgb, strip_center_x, cy, rx=rx, ry=ry)

        # Corrección global
        rgb_glob = apply_cal(rgb_raw)

        # Corrección por blanco local (por fila) para sombras/gradiente
        wl_raw, wl_score, wl_rej = local_white_near_pad(img_rgb, strip_center_x, cy, pitch)
        rgb_local = None
        if wl_raw is not None:
            wl_glob = apply_cal(wl_raw)
            rgb_local = apply_local_white_correction(rgb_glob, wl_glob)

        # Comparación: probamos raw / global / local y elegimos mejor
        m_raw = match_swatch(param, rgb_raw, swatches)
        m_glob = match_swatch(param, rgb_glob, swatches)
        m_loc = match_swatch(param, rgb_local, swatches) if rgb_local is not None else None

        # Elegir la variante con menor deltaE (con pequeñas reglas para evitar sobre-correcciones)
        choices = [("raw", rgb_raw, m_raw)]
        choices.append(("global", rgb_glob, m_glob))
        if m_loc is not None:
            choices.append(("local_white", rgb_local, m_loc))

        choices.sort(key=lambda t: t[2]["deltaE"])
        chosen_mode, chosen_rgb, chosen_m = choices[0]

        best = float(chosen_m["deltaE"])
        second = float(chosen_m["deltaE2"])
        conf = confidence_label(best, second)
        if conf == "low":
            low_conf_count += 1

        result_item = {
            "index": i + 1,
            "parameter": param,
            "value": str(chosen_m["value"]),
            "confidence": conf,
            "deltaE": round(best, 2),
            "deltaE2": round(second, 2),
            "mode": chosen_mode,
            "reference_rgb": chosen_m["reference_rgb"],
            "sample_point": {"x": int(strip_center_x), "y": int(cy)},
        }

        # Diagnóstico ampliado (siempre lo devolvemos; Base44 puede ocultarlo si no lo necesita)
        result_item["sample_rgb_raw"] = [int(round(x)) for x in rgb_raw.tolist()]
        result_item["sample_rgb_global"] = [int(round(x)) for x in rgb_glob.tolist()]
        if rgb_local is not None:
            result_item["sample_rgb_local"] = [int(round(x)) for x in rgb_local.tolist()]
        result_item["glareRejectedPct"] = round(float(rej_pad), 2)

        if wl_raw is not None:
            result_item["localWhite_raw"] = [int(round(x)) for x in wl_raw.tolist()]
            result_item["localWhite_score"] = round(float(wl_score), 2)
            result_item["localWhite_glareRejectedPct"] = round(float(wl_rej), 2)

        results.append(result_item)

    if low_conf_count >= 4:
        warnings.append("Varias coincidencias con baja confianza: revisa iluminación/posición")
    if geom["edge_peaks_count"] < 6:
        warnings.append("Detección de bordes débil en los pads; posible desenfoque")
    if strip_width < 35:
        warnings.append("Zona de tira estrecha; la foto puede estar muy lejos")

    # Diagnóstico global completo
    diagnostics = {
        "imageSize": [int(w), int(h)],
        "foundBars": True,
        "foundPads": 10,
        "transformType": model_type,
        "calibrationErrorMean": round(calib_err_mean, 2),
        "calibrationErrorMax": round(calib_err_max, 2),
        "barsObservedRGB": {k: [int(round(x)) for x in v.tolist()] for k, v in obs_bars.items()},
        "barsObserved_glareRejectedPct": {k: round(float(v), 2) for k, v in obs_bars_rej.items()},
        "barsCorrectedRGB": (corr_aff_map if model_type == "affine_3x3" else corr_ch_map),
        "whiteObservedRGB": [int(round(x)) for x in obs_white.tolist()],
        "whiteScore": round(float(white_score), 2),
        "whiteReliable": bool(white_reliable),
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
    }

    return {
        "ok": True,
        "orientation": geom["orientation"],
        "results": results,
        "diagnostics": diagnostics,
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
    return {"ok": True, "swatchesLoaded": os.path.exists(SWATCHES_PATH)}


@app.post("/analyze-strip")
def analyze_strip(req: AnalyzeReq, x_api_key: str = Header(default="")) -> Dict[str, Any]:
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
                "warnings": [str(e)],
            },
            "retake_reason": "Falta swatches.json en el contenedor",
            "retake_tips": ["Sube swatches.json al repositorio junto a main.py"],
        }

    except Exception as e:
        return {
            "ok": False,
            "orientation": "",
            "results": [],
            "diagnostics": {
                "imageSize": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
                "foundBars": False,
                "warnings": [str(e)],
            },
            "retake_reason": "No se pudo analizar la tira en esta imagen",
            "retake_tips": [
                "Alinea la tira entre las franjas azul y verde",
                "Evita reflejos y sombras",
                "Asegura buena nitidez",
            ],
        }
