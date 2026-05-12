# ColorScale backend — overwrite v2 calibration

Ficheros pensados para copiar encima del repo actual `qlioFactory/colorScale` sin cambiar Base44.

## Qué mantiene

- Endpoint principal: `POST /analyze-strip`
- Entrada compatible:

```json
{
  "image_url": "https://.../foto.jpg",
  "image_base64": "data:image/jpeg;base64,...",
  "debug": false,
  "client_id": "demo",
  "scan_id": "opcional"
}
```

## Qué añade

- Calibración cromática con las barras fijas de la plantilla:
  - gris `128,128,128`
  - azul `0,0,255`
  - verde `0,128,0`
  - rojo `255,0,0`
- Comparación en CIELAB con Delta E 2000.
- `quality_score`, `confidence`, diagnósticos y recomendaciones para repetir foto.
- Histórico en SQLite por defecto.
- Endpoints:
  - `GET /`
  - `GET /health`
  - `POST /analyze-strip`
  - `GET /history`
  - `GET /history/{analysis_id}`

## Variables de entorno

- `API_KEY`: si se define, exige header `x-api-key`.
- `SWATCHES_PATH`: ruta alternativa al fichero `swatches.json`.
- `HISTORY_DB_PATH`: ruta del SQLite. Por defecto `./colorscale_history.sqlite3`.
- `SAVE_HISTORY_DEFAULT`: `true`/`false`. Por defecto `true`.
- `MAX_DOWNLOAD_BYTES`: límite para imágenes descargadas por URL. Por defecto 12 MB.

## Ejecutar local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Probar con curl

```bash
curl -X POST http://localhost:8000/analyze-strip \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://.../foto.jpg","debug":true,"client_id":"demo"}'
```

## Subir al repo machacando ficheros

```bash
cp main.py swatches.json requirements.txt Dockerfile .dockerignore README.md /ruta/a/colorScale/
cd /ruta/a/colorScale
git status
git add main.py swatches.json requirements.txt Dockerfile .dockerignore README.md
git commit -m "Replace backend with calibrated v2 engine"
git push origin main
```

## Nota importante de precisión

El motor ya compensa parte de la luz usando las barras de referencia, pero para demos/ferias sigue siendo muy recomendable una caja de luz o soporte que mantenga iluminación y perpendicularidad constantes.
