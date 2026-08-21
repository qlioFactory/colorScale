import base64
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import certifi
import cv2
import numpy as np
import requests
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

VERSION = "0.4.1"

app = FastAPI(title="ColorScale API", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

WHITE_LIKE_PARAMS = {"iron", "free_chlorine", "aluminium", "copper"}

# v0.4.0: espacio cromático canónico aprendido a partir de 38 fotografías
# válidas sobre la tarjeta de referencia (lotes 2026-06-11 y 2026-07-17).
# No son los colores de las tiras: son los anclajes reales de la tarjeta impresa.
ANCHOR_NAMES = ["red", "green", "blue", "gray", "white"]
NOMINAL_ANCHORS_RGB = np.array(
    [
        [255.0, 0.0, 0.0],
        [0.0, 128.0, 0.0],
        [0.0, 0.0, 255.0],
        [128.0, 128.0, 128.0],
        [255.0, 255.0, 255.0],
    ],
    dtype=np.float64,
)
CANONICAL_ANCHORS_RGB = np.array(
    [
        [208.0, 40.0, 28.0],
        [50.0, 86.0, 20.0],
        [63.0, 58.0, 128.0],
        [129.0, 102.0, 91.0],
        [188.0, 184.0, 173.0],
    ],
    dtype=np.float64,
)

SWATCHES_PATH = os.environ.get("SWATCHES_PATH", "swatches.json")
MAX_ANALYSIS_DIM = int(os.environ.get("MAX_ANALYSIS_DIM", "1400"))


def clamp_rgb(rgb: Any) -> np.ndarray:
    return np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 255.0)


def srgb_to_linear(rgb: Any) -> np.ndarray:
    x = clamp_rgb(rgb) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: Any) -> np.ndarray:
    x = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(s * 255.0, 0.0, 255.0)


def fit_diag_affine(obs_rgb: np.ndarray, tgt_rgb: np.ndarray) -> np.ndarray:
    """Ajuste por canal y = a*x+b en RGB lineal. Devuelve shape (3,2)."""
    x = srgb_to_linear(obs_rgb)
    y = srgb_to_linear(tgt_rgb)
    coeffs = []
    for c in range(3):
        a_mat = np.column_stack([x[:, c], np.ones(len(x), dtype=np.float64)])
        coef, *_ = np.linalg.lstsq(a_mat, y[:, c], rcond=None)
        coeffs.append(coef)
    return np.asarray(coeffs, dtype=np.float64)


def apply_diag_affine(rgb: Any, coeffs: np.ndarray) -> np.ndarray:
    x = srgb_to_linear(rgb)
    y = x * coeffs[:, 0] + coeffs[:, 1]
    return linear_to_srgb(y)


NOMINAL_TO_CANONICAL = fit_diag_affine(NOMINAL_ANCHORS_RGB, CANONICAL_ANCHORS_RGB)


def rgb_to_lab(rgb: Any) -> np.ndarray:
    arr = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1] -= 128.0
    lab[:, 2] -= 128.0
    return lab


def delta_e76(rgb1: Any, rgb2: Any) -> float:
    a = rgb_to_lab(np.asarray(rgb1).reshape(1, 3))[0]
    b = rgb_to_lab(np.asarray(rgb2).reshape(1, 3))[0]
    return float(np.linalg.norm(a - b))


def parse_rgb(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) >= 3:
            return clamp_rgb([float(nums[0]), float(nums[1]), float(nums[2])])
        return None
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        return clamp_rgb(value[:3])
    if isinstance(value, dict):
        if all(k in value for k in ("r", "g", "b")):
            return clamp_rgb([value["r"], value["g"], value["b"]])
        if "rgb" in value:
            return parse_rgb(value["rgb"])
    return None


def parse_numeric_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:[\.,]\d+)?", str(value))
    if not m:
        return float("inf")
    try:
        return float(m.group(0).replace(",", "."))
    except Exception:
        return float("inf")


def normalize_swatches(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Normaliza el swatches.json real de ColorScale.

    Formato principal:
      {
        "alkalinity": {
          "values": [...],
          "rgb": [[r,g,b], ...],
          "numeric_values": [...]  # opcional; usado por GH
        },
        ...
      }

    Mantiene además compatibilidad con formatos antiguos basados en objetos
    individuales {value, rgb}.
    """
    out: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PARAM_ORDER}

    aliases = {
        "ph": "pH",
        "p_h": "pH",
        "hardness": "gh",
        "total_hardness": "gh",
        "freechlorine": "free_chlorine",
        "free chlorine": "free_chlorine",
        "nitrate+nitrite": "nitrate",
        "nitrate_nitrite": "nitrate",
        "aluminum": "aluminium",
    }

    def canonical_param(param_raw: Any) -> str:
        text = str(param_raw).strip()
        return aliases.get(text.lower(), text)

    def add_item(param: str, value: Any, rgb_value: Any, numeric_value: Any = None) -> None:
        if param not in out or value is None:
            return
        rgb = parse_rgb(rgb_value)
        if rgb is None:
            return
        can_rgb = apply_diag_affine(rgb.reshape(1, 3), NOMINAL_TO_CANONICAL).reshape(3)
        numeric = parse_numeric_value(numeric_value if numeric_value is not None else value)
        out[param].append(
            {
                "value": value,
                "rgb": rgb.astype(np.float64),
                "canonical_rgb": can_rgb.astype(np.float64),
                "numeric": numeric,
            }
        )

    if isinstance(raw, dict):
        for param_raw, block in raw.items():
            param = canonical_param(param_raw)
            if param not in out or not isinstance(block, dict):
                continue

            # Formato real del fichero suministrado por el cliente:
            # arrays paralelos values + rgb (+ numeric_values opcional).
            values = block.get("values")
            rgbs = block.get("rgb")
            numeric_values = block.get("numeric_values")

            if isinstance(values, list) and isinstance(rgbs, list):
                n = min(len(values), len(rgbs))
                for i in range(n):
                    numeric = None
                    if isinstance(numeric_values, list) and i < len(numeric_values):
                        numeric = numeric_values[i]
                    add_item(param, values[i], rgbs[i], numeric)
                continue

            # Compatibilidad con formatos anteriores tipo:
            # {"0": {"rgb": [...]}, "40": {"rgb": [...]}}
            for value_key, item in block.items():
                if isinstance(item, dict):
                    value = item.get("value", value_key)
                    rgb_value = item.get("rgb") or item.get("color_rgb") or item.get("color")
                    add_item(param, value, rgb_value, item.get("numeric_value"))

    elif isinstance(raw, list):
        # Compatibilidad con lista plana: {parameter, value, rgb}
        for item in raw:
            if not isinstance(item, dict):
                continue
            param_raw = item.get("parameter") or item.get("param") or item.get("name")
            if not param_raw:
                continue
            param = canonical_param(param_raw)
            value = item.get("value")
            if value is None:
                value = item.get("label")
            rgb_value = item.get("rgb") or item.get("color_rgb") or item.get("color")
            add_item(param, value, rgb_value, item.get("numeric_value"))

    for p in out:
        out[p].sort(key=lambda x: x["numeric"])
    return out

def load_swatches() -> Dict[str, List[Dict[str, Any]]]:
    try:
        with open(SWATCHES_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return normalize_swatches(raw)
    except Exception:
        return {p: [] for p in PARAM_ORDER}


SWATCHES = load_swatches()


def swatches_loaded() -> bool:
    return all(len(SWATCHES.get(p, [])) > 0 for p in PARAM_ORDER)


def resize_for_analysis(img: np.ndarray, max_dim: int = MAX_ANALYSIS_DIM) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img, scale


def contour_geometry(contour: np.ndarray) -> Dict[str, Any]:
    pts = contour.reshape(-1, 2).astype(np.float64)
    center = pts.mean(axis=0)
    x = pts - center
    cov = np.cov(x.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    lp = x @ axis
    sp = x @ perp
    return {
        "area": float(cv2.contourArea(contour)),
        "center": center,
        "axis": axis,
        "long": float(lp.max() - lp.min()),
        "short": float(sp.max() - sp.min()),
        "contour": contour,
    }


def elongated_components(mask: np.ndarray) -> List[Dict[str, Any]]:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        geom = contour_geometry(contour)
        if geom["area"] < 120:
            continue
        if geom["long"] < 40 or geom["short"] < 2:
            continue
        if geom["long"] / max(geom["short"], 1e-6) < 4.0:
            continue
        result.append(geom)
    result.sort(key=lambda x: x["area"], reverse=True)
    return result


def sample_binary_line(mask: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray,
                       center_u: float, half_u: float, vv: np.ndarray) -> np.ndarray:
    us = np.linspace(center_u - half_u, center_u + half_u, 9)
    uu, vgrid = np.meshgrid(us, vv)
    map_x = origin[0] + uu * u[0] + vgrid * v[0]
    map_y = origin[1] + uu * u[1] + vgrid * v[1]
    sampled = cv2.remap(
        mask,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.mean(sampled > 0, axis=1)


def longest_relevant_interval(binary: np.ndarray, center_index: int) -> Optional[Tuple[int, int]]:
    x = binary.astype(np.uint8).reshape(-1)
    intervals = []
    start = None
    for i, value in enumerate(x):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(x) - 1):
            end = i if value and i == len(x) - 1 else i - 1
            length = end - start + 1
            mid = (start + end) / 2.0
            intervals.append((length, start, end, abs(mid - center_index)))
            start = None
    if not intervals:
        return None
    intervals.sort(key=lambda z: (z[3], -z[0]))
    _, start, end, _ = intervals[0]
    return start, end


def detect_geometry(img_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    img, scale = resize_for_analysis(img_bgr)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masks = {
        "red": cv2.inRange(hsv, np.array([0, 65, 50]), np.array([20, 255, 255]))
               | cv2.inRange(hsv, np.array([165, 65, 50]), np.array([179, 255, 255])),
        "green": cv2.inRange(hsv, np.array([25, 35, 25]), np.array([105, 255, 255])),
        "blue": cv2.inRange(hsv, np.array([88, 30, 20]), np.array([170, 255, 255])),
    }
    comps = {name: elongated_components(mask) for name, mask in masks.items()}

    best = None
    for red in comps["red"][:12]:
        for green in comps["green"][:12]:
            for blue in comps["blue"][:12]:
                axes = [red["axis"].copy(), green["axis"].copy(), blue["axis"].copy()]
                for i in (1, 2):
                    if np.dot(axes[i], axes[0]) < 0:
                        axes[i] *= -1
                parallel = min(abs(float(np.dot(axes[i], axes[j]))) for i in range(3) for j in range(i + 1, 3))
                if parallel < 0.92:
                    continue

                v = np.mean(axes, axis=0)
                v = v / (np.linalg.norm(v) + 1e-12)
                u = np.array([-v[1], v[0]], dtype=np.float64)

                cr, cg, cb = red["center"], green["center"], blue["center"]
                if np.dot(cb - cr, u) < 0:
                    u *= -1

                sr, sg, sb = [float(np.dot(c, u)) for c in (cr, cg, cb)]
                lr, lg, lb = [float(np.dot(c, v)) for c in (cr, cg, cb)]
                if not (sr < sg < sb):
                    continue

                d_rg = sg - sr
                d_gb = sb - sg
                if d_rg < 5 or d_gb < 10:
                    continue
                ratio = d_gb / d_rg
                if not (1.3 < ratio < 6.5):
                    continue

                lengths = np.array([red["long"], green["long"], blue["long"]], dtype=np.float64)
                align = float(np.std([lr, lg, lb]) / (np.mean(lengths) + 1e-6))
                len_var = float(np.std(lengths) / (np.mean(lengths) + 1e-6))
                if align > 0.30 or len_var > 0.45:
                    continue

                candidate_score = align * 6.0 + len_var * 2.0 + abs(ratio - 2.7) * 0.05
                if best is None or candidate_score < best[0]:
                    best = (candidate_score, red, green, blue, u, v)

    if best is None:
        return None

    score, red, green, blue, u, v = best
    origin = np.mean([red["center"], green["center"], blue["center"]], axis=0)
    coords = {}
    for name, geom in (("red", red), ("green", green), ("blue", blue)):
        delta = geom["center"] - origin
        coords[name] = (float(np.dot(delta, u)), float(np.dot(delta, v)))

    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
    projected = (corners - origin) @ v
    vv = np.linspace(projected.min(), projected.max(), int(projected.max() - projected.min()) + 1)

    line_scores = []
    for name, geom in (("red", red), ("green", green), ("blue", blue)):
        half_u = max(2.5, 0.22 * geom["short"])
        line_scores.append(sample_binary_line(masks[name], origin, u, v, coords[name][0], half_u, vv))

    combined = np.mean(np.stack(line_scores), axis=0)
    smooth = np.convolve(combined, np.ones(21) / 21.0, mode="same")
    good = (smooth > 0.18).astype(np.uint8)
    close_len = max(15, int(0.03 * len(vv)))
    good = cv2.morphologyEx(good.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, close_len), np.uint8))[0]

    center_v = float(np.mean([coords["red"][1], coords["green"][1], coords["blue"][1]]))
    center_idx = int(np.argmin(np.abs(vv - center_v)))
    interval = longest_relevant_interval(good, center_idx)

    if interval is not None:
        a, b = interval
        vmin, vmax = float(vv[a]), float(vv[b])
    else:
        extents = []
        for geom in (red, green, blue):
            vals = (geom["contour"].reshape(-1, 2).astype(np.float64) - origin) @ v
            extents.append((float(vals.min()), float(vals.max())))
        vmin = min(x[0] for x in extents)
        vmax = max(x[1] for x in extents)

    if vmax - vmin < 80:
        return None

    return {
        "img": img,
        "scale": scale,
        "score": float(score),
        "red": red,
        "green": green,
        "blue": blue,
        "u": u,
        "v": v,
        "origin": origin,
        "coords": coords,
        "vmin": vmin,
        "vmax": vmax,
    }


def sample_oriented_rect(img_bgr: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray,
                         center_u: float, center_v: float, half_u: float, half_v: float,
                         nu: int = 19, nv: int = 19) -> np.ndarray:
    us = np.linspace(center_u - half_u, center_u + half_u, nu)
    vs = np.linspace(center_v - half_v, center_v + half_v, nv)
    uu, vv = np.meshgrid(us, vs)
    map_x = origin[0] + uu * u[0] + vv * v[0]
    map_y = origin[1] + uu * u[1] + vv * v[1]
    bgr = cv2.remap(
        img_bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return bgr[:, :, ::-1]


def robust_patch_stats_rgb(patch_rgb: np.ndarray, reject_top: float = 0.12,
                           reject_bottom: float = 0.02) -> Tuple[np.ndarray, float]:
    pixels = patch_rgb.reshape(-1, 3).astype(np.float64)
    lum = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    lo = np.quantile(lum, reject_bottom) if reject_bottom > 0 else -1.0
    hi = np.quantile(lum, 1.0 - reject_top) if reject_top > 0 else 256.0
    keep = (lum >= lo) & (lum <= hi)
    if np.sum(keep) < max(10, int(0.30 * len(pixels))):
        keep[:] = True
    rgb = np.median(pixels[keep], axis=0)
    rejected = 100.0 * (1.0 - float(np.sum(keep)) / float(len(pixels)))
    return clamp_rgb(rgb), rejected


def strip_profile(geometry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = geometry["img"]
    u, v, origin = geometry["u"], geometry["v"], geometry["origin"]
    ug = geometry["coords"]["green"][0]
    ub = geometry["coords"]["blue"][0]
    center_u = (ug + ub) / 2.0
    gap = ub - ug
    length = geometry["vmax"] - geometry["vmin"]

    vv = np.linspace(
        geometry["vmin"] - 0.08 * length,
        geometry["vmax"] + 0.08 * length,
        int(length * 1.16) + 1,
    )
    us = np.linspace(center_u - 0.12 * gap, center_u + 0.12 * gap, 21)
    uu, vgrid = np.meshgrid(us, vv)
    map_x = origin[0] + uu * u[0] + vgrid * v[0]
    map_y = origin[1] + uu * u[1] + vgrid * v[1]
    bgr = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    rgb = bgr[:, :, ::-1]
    median_rgb = np.median(rgb, axis=1)
    lab = rgb_to_lab(median_rgb)
    chroma = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
    chroma = np.convolve(chroma, np.ones(9) / 9.0, mode="same")
    dif = np.linalg.norm(np.diff(lab, axis=0), axis=1)
    edge = np.r_[dif[0] if len(dif) else 0.0, dif]
    edge = np.convolve(edge, np.ones(7) / 7.0, mode="same")
    return vv, chroma, edge


def detect_pad_centers(geometry: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
    """Busca una rejilla regular de 10 pads; no depende del borde exacto del primer pad."""
    vv, chroma, edge = strip_profile(geometry)
    length = geometry["vmax"] - geometry["vmin"]

    best_score = -1e18
    best_centers = None
    best_pitch = None

    # 61 x 81 ~= 5.000 combinaciones: rápido y estable en Cloud Run.
    pitches = np.linspace(0.085 * length, 0.105 * length, 61)
    first_centers = np.linspace(geometry["vmin"], geometry["vmin"] + 0.14 * length, 81)

    for pitch in pitches:
        boundary_offset = 0.31 * pitch
        for first in first_centers:
            centers = first + np.arange(10, dtype=np.float64) * pitch
            if centers[-1] > geometry["vmax"] + 0.08 * length:
                continue

            idx = np.clip(np.searchsorted(vv, centers), 0, len(vv) - 1)
            gaps = (centers[:-1] + centers[1:]) / 2.0
            gap_idx = np.clip(np.searchsorted(vv, gaps), 0, len(vv) - 1)
            b1 = np.clip(np.searchsorted(vv, centers - boundary_offset), 0, len(vv) - 1)
            b2 = np.clip(np.searchsorted(vv, centers + boundary_offset), 0, len(vv) - 1)

            score = float(np.sum(chroma[idx]) - 0.65 * np.sum(chroma[gap_idx]))
            score += 0.45 * float(np.sum(edge[b1]) + np.sum(edge[b2]))
            score -= 0.20 * float(np.sum(edge[idx]))

            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_pitch = float(pitch)

    if best_centers is None:
        raise ValueError("No se ha podido localizar la rejilla de 10 pads")

    return best_centers, float(best_pitch), float(best_score)


def sample_reference_anchors(geometry: Dict[str, Any]) -> Dict[str, np.ndarray]:
    img = geometry["img"]
    u, v, origin = geometry["u"], geometry["v"], geometry["origin"]
    coords = geometry["coords"]
    length = geometry["vmax"] - geometry["vmin"]
    center_v = (geometry["vmin"] + geometry["vmax"]) / 2.0

    anchors = {}
    for name in ("red", "green", "blue"):
        geom = geometry[name]
        patch = sample_oriented_rect(
            img, origin, u, v,
            coords[name][0], center_v,
            max(2.0, 0.18 * geom["short"]), 0.30 * length,
            15, 81,
        )
        anchors[name], _ = robust_patch_stats_rgb(patch, 0.10, 0.02)

    d_rg = coords["green"][0] - coords["red"][0]
    median_short = float(np.median([geometry["red"]["short"], geometry["green"]["short"], geometry["blue"]["short"]]))

    gray_u = coords["blue"][0] + 0.95 * d_rg
    patch = sample_oriented_rect(
        img, origin, u, v,
        gray_u, center_v,
        max(2.0, 0.18 * median_short), 0.30 * length,
        15, 81,
    )
    anchors["gray"], _ = robust_patch_stats_rgb(patch, 0.10, 0.02)

    white_u = coords["red"][0] - 0.95 * d_rg
    patch = sample_oriented_rect(
        img, origin, u, v,
        white_u, center_v,
        max(3.0, 0.22 * d_rg), 0.20 * length,
        15, 61,
    )
    anchors["white"], _ = robust_patch_stats_rgb(patch, 0.12, 0.02)
    return anchors


def sample_pads(geometry: Dict[str, Any], centers_v: np.ndarray, pitch: float) -> Tuple[np.ndarray, List[float], List[List[int]]]:
    img = geometry["img"]
    u, v, origin = geometry["u"], geometry["v"], geometry["origin"]
    ug = geometry["coords"]["green"][0]
    ub = geometry["coords"]["blue"][0]
    center_u = (ug + ub) / 2.0
    gap = ub - ug

    rgbs = []
    glare = []
    points = []
    for center_v in centers_v:
        patch = sample_oriented_rect(
            img, origin, u, v,
            center_u, float(center_v),
            0.13 * gap, 0.20 * pitch,
            19, 19,
        )
        rgb, rejected = robust_patch_stats_rgb(patch, 0.12, 0.02)
        rgbs.append(rgb)
        glare.append(float(rejected))
        p = origin + center_u * u + float(center_v) * v
        points.append([int(round(float(p[0] / geometry["scale"]))), int(round(float(p[1] / geometry["scale"])))])

    return np.asarray(rgbs, dtype=np.float64), glare, points


def nearest_match(param: str, rgb: np.ndarray, space: str) -> Dict[str, Any]:
    items = SWATCHES.get(param, [])
    if not items:
        return {"item": None, "delta": 999.0, "delta2": 999.0, "ratio": 1.0}

    key = "canonical_rgb" if space == "canonical" else "rgb"
    distances = [(delta_e76(rgb, item[key]), item) for item in items]
    distances.sort(key=lambda x: x[0])
    d1, best = distances[0]
    d2 = distances[1][0] if len(distances) > 1 else d1 + 20.0
    ratio = float(d1 / max(d2, 1e-6))
    return {"item": best, "delta": float(d1), "delta2": float(d2), "ratio": ratio}


def orientation_score(raw_seq: np.ndarray, canonical_seq: np.ndarray) -> float:
    score = 0.0
    weights = [1.25, 1.25, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.15, 1.15]
    for i, param in enumerate(PARAM_ORDER):
        mr = nearest_match(param, raw_seq[i], "raw")
        mc = nearest_match(param, canonical_seq[i], "canonical")
        # Usamos el espacio que mejor explica el color para decidir solo la orientación.
        local = min(mr["delta"], mc["delta"])
        local += 4.0 * min(mr["ratio"], mc["ratio"])
        score += weights[i] * min(local, 45.0)
    return float(score)


def choose_orientation(raw_seq: np.ndarray, canonical_seq: np.ndarray,
                       points: List[List[int]], glare: List[float]) -> Tuple[np.ndarray, np.ndarray, List[List[int]], List[float], str, float, float]:
    forward = orientation_score(raw_seq, canonical_seq)
    reverse = orientation_score(raw_seq[::-1], canonical_seq[::-1])
    if reverse + 1.0 < forward:
        return raw_seq[::-1], canonical_seq[::-1], points[::-1], glare[::-1], "reversed", reverse, forward
    return raw_seq, canonical_seq, points, glare, "forward", forward, reverse


def relative_neutrality(rgb: np.ndarray, white_rgb: np.ndarray) -> Tuple[float, float]:
    rgb = clamp_rgb(rgb)
    white_rgb = clamp_rgb(white_rgb)
    lum = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    wlum = float(0.2126 * white_rgb[0] + 0.7152 * white_rgb[1] + 0.0722 * white_rgb[2])
    ratio = lum / max(wlum, 1.0)
    chroma = float(np.max(rgb) - np.min(rgb))
    return ratio, chroma


def force_low_if_neutral(param: str, raw_rgb: np.ndarray, white_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
    if param not in WHITE_LIKE_PARAMS:
        return None
    items = SWATCHES.get(param, [])
    if not items:
        return None
    ratio, chroma = relative_neutrality(raw_rgb, white_rgb)
    ordered = sorted(items, key=lambda x: x["numeric"])
    if ratio >= 0.90 and chroma <= 30:
        return ordered[0]
    return None


def choose_match(param: str, raw_rgb: np.ndarray, canonical_rgb: np.ndarray,
                 white_rgb: np.ndarray) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    forced = force_low_if_neutral(param, raw_rgb, white_rgb)
    raw_match = nearest_match(param, raw_rgb, "raw")
    can_match = nearest_match(param, canonical_rgb, "canonical")

    if forced is not None:
        return forced, "neutral-low", raw_match, can_match

    if raw_match["item"] is None:
        return can_match["item"], "canonical", raw_match, can_match
    if can_match["item"] is None:
        return raw_match["item"], "raw", raw_match, can_match

    raw_value = str(raw_match["item"]["value"])
    can_value = str(can_match["item"]["value"])
    if raw_value == can_value:
        return can_match["item"], "agree", raw_match, can_match

    # La decisión se basa principalmente en separación relativa frente al segundo swatch.
    # La calibración canónica solo gana si la evidencia es suficientemente clara.
    if can_match["ratio"] + 0.08 < raw_match["ratio"]:
        return can_match["item"], "canonical", raw_match, can_match
    if can_match["delta"] + 3.0 < raw_match["delta"] and can_match["ratio"] <= raw_match["ratio"] + 0.03:
        return can_match["item"], "canonical", raw_match, can_match
    return raw_match["item"], "raw", raw_match, can_match


def confidence_for(raw_match: Dict[str, Any], can_match: Dict[str, Any], mode: str) -> str:
    if mode == "neutral-low":
        return "medium"
    agree = (
        raw_match.get("item") is not None
        and can_match.get("item") is not None
        and str(raw_match["item"]["value"]) == str(can_match["item"]["value"])
    )
    best_ratio = min(float(raw_match.get("ratio", 1.0)), float(can_match.get("ratio", 1.0)))
    best_delta = min(float(raw_match.get("delta", 999.0)), float(can_match.get("delta", 999.0)))
    if agree and best_ratio <= 0.70 and best_delta <= 16:
        return "high"
    if agree or (best_ratio <= 0.78 and best_delta <= 22):
        return "medium"
    return "low"


def decode_image_payload(payload: Dict[str, Any]) -> np.ndarray:
    image_url = payload.get("image_url") or payload.get("imageUrl") or payload.get("url")
    image_b64 = payload.get("image_base64") or payload.get("imageBase64")

    if image_url:
        response = requests.get(str(image_url), timeout=25, verify=certifi.where())
        response.raise_for_status()
        data = np.frombuffer(response.content, dtype=np.uint8)
    elif image_b64:
        text = str(image_b64)
        if "," in text and text.lstrip().startswith("data:"):
            text = text.split(",", 1)[1]
        data = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    else:
        raise ValueError("Falta image_url/imageUrl o image_base64")

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se ha podido decodificar la imagen")
    return img


def analyze_image(img_bgr: np.ndarray) -> Dict[str, Any]:
    if not swatches_loaded():
        return {
            "ok": False,
            "version": VERSION,
            "foundBars": False,
            "error": "swatches.json no contiene todos los parámetros esperados",
        }

    geometry = detect_geometry(img_bgr)
    if geometry is None:
        return {
            "ok": True,
            "version": VERSION,
            "foundBars": False,
            "retake": True,
            "retakeReason": "No se han detectado con suficiente claridad las barras de referencia.",
            "warnings": [],
            "results": [],
        }

    centers_v, pitch, grid_score = detect_pad_centers(geometry)
    anchors = sample_reference_anchors(geometry)
    anchor_matrix = np.stack([anchors[name] for name in ANCHOR_NAMES])
    calibration = fit_diag_affine(anchor_matrix, CANONICAL_ANCHORS_RGB)

    corrected_anchors = apply_diag_affine(anchor_matrix, calibration)
    anchor_errors = np.array(
        [delta_e76(corrected_anchors[i], CANONICAL_ANCHORS_RGB[i]) for i in range(len(ANCHOR_NAMES))],
        dtype=np.float64,
    )
    cal_mean = float(np.mean(anchor_errors))
    cal_max = float(np.max(anchor_errors))

    raw_seq, glare, points = sample_pads(geometry, centers_v, pitch)
    canonical_seq = apply_diag_affine(raw_seq, calibration)

    raw_seq, canonical_seq, points, glare, orientation, orientation_score_used, orientation_score_other = choose_orientation(
        raw_seq, canonical_seq, points, glare
    )

    warnings = []
    if geometry["score"] > 1.0:
        warnings.append("La geometría de la tarjeta es menos nítida de lo habitual.")
    if cal_mean > 10.0 or cal_max > 20.0:
        warnings.append("La iluminación es irregular; se ha aplicado corrección con la tarjeta de referencia.")
    if abs(orientation_score_other - orientation_score_used) < 6.0:
        warnings.append("La orientación de la tira es poco concluyente.")

    # Solo rechazamos iluminación realmente extrema. En v0.4 evitamos descartar fotos útiles.
    white_luma = float(0.2126 * anchors["white"][0] + 0.7152 * anchors["white"][1] + 0.0722 * anchors["white"][2])
    if white_luma < 55 or white_luma > 252 or cal_mean > 22 or cal_max > 45:
        return {
            "ok": True,
            "version": VERSION,
            "foundBars": True,
            "retake": True,
            "retakeReason": "La iluminación de la foto es demasiado extrema para obtener una lectura fiable.",
            "warnings": warnings,
            "results": [],
            "diagnostics": {
                "calibrationMeanDeltaE": round(cal_mean, 2),
                "calibrationMaxDeltaE": round(cal_max, 2),
                "whiteLuma": round(white_luma, 1),
            },
        }

    results = []
    low_count = 0
    for i, param in enumerate(PARAM_ORDER):
        chosen, mode, raw_match, can_match = choose_match(param, raw_seq[i], canonical_seq[i], anchors["white"])
        if chosen is None:
            continue

        confidence = confidence_for(raw_match, can_match, mode)
        if confidence == "low":
            low_count += 1

        chosen_space_rgb = chosen["canonical_rgb"] if mode in ("canonical", "agree") else chosen["rgb"]
        used_rgb = canonical_seq[i] if mode in ("canonical", "agree") else raw_seq[i]
        d1 = delta_e76(used_rgb, chosen_space_rgb)

        # deltaE2 del espacio finalmente usado
        final_match = can_match if mode in ("canonical", "agree") else raw_match
        d2 = float(final_match.get("delta2", d1 + 20.0))

        results.append(
            {
                "parameter": param,
                "value": chosen["value"],
                "confidence": confidence,
                "mode": "global" if mode in ("canonical", "agree") else "raw",
                "decision": mode,
                "sample_rgb_raw": [int(round(x)) for x in raw_seq[i]],
                "sample_rgb_global": [int(round(x)) for x in canonical_seq[i]],
                "sample_rgb_used": [int(round(x)) for x in used_rgb],
                "reference_rgb": [int(round(x)) for x in chosen["rgb"]],
                "reference_rgb_canonical": [int(round(x)) for x in chosen["canonical_rgb"]],
                "deltaE": round(float(d1), 2),
                "deltaE2": round(float(d2), 2),
                "rawDeltaE": round(float(raw_match.get("delta", 999.0)), 2),
                "canonicalDeltaE": round(float(can_match.get("delta", 999.0)), 2),
                "sample_point": points[i],
                "glareRejectedPct": round(float(glare[i]), 1),
            }
        )

    if low_count >= 4:
        warnings.append("Varias lecturas tienen poca separación respecto al color vecino de la escala.")

    return {
        "ok": True,
        "version": VERSION,
        "foundBars": True,
        "retake": False,
        "warnings": warnings,
        "results": results,
        "diagnostics": {
            "geometryScore": round(float(geometry["score"]), 3),
            "padGridScore": round(float(grid_score), 2),
            "padPitchPx": round(float(pitch / geometry["scale"]), 2),
            "orientation": orientation,
            "orientationScore": round(float(orientation_score_used), 2),
            "orientationAlternativeScore": round(float(orientation_score_other), 2),
            "calibrationMeanDeltaE": round(cal_mean, 2),
            "calibrationMaxDeltaE": round(cal_max, 2),
            "anchorRgb": {name: [int(round(x)) for x in anchors[name]] for name in ANCHOR_NAMES},
            "whiteLuma": round(white_luma, 1),
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": VERSION,
        "swatchesLoaded": swatches_loaded(),
        "swatchesPath": os.path.abspath(SWATCHES_PATH),
        "swatchCounts": {p: len(SWATCHES.get(p, [])) for p in PARAM_ORDER},
        "parameters": PARAM_ORDER,
    }


@app.post("/analyze-strip")
def analyze_strip(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
        img = decode_image_payload(payload)
        return analyze_image(img)
    except Exception as exc:
        return {
            "ok": False,
            "version": VERSION,
            "foundBars": False,
            "error": str(exc),
        }
