# ColorScale hotfix 2.1.1 - Base44 compatibility

## Motivo
En algunos móviles Android/Redmi el backend analizaba la foto y devolvía los 10 resultados, pero marcaba la captura como `foto_no_fiable` por baja calidad (`quality_score` bajo). El frontend/proxy de Base44 interpretaba ese `ok:false` o `status:foto_no_fiable` como error de transporte y mostraba el JSON completo como `Error proxy`.

## Cambio
Para análisis completados con resultados:
- `ok` ahora indica éxito de transporte/análisis, no validez de foto.
- `status` se mantiene como `ok` para no romper Base44.
- La validez real de la foto pasa a `photo_status` (`ok` o `foto_no_fiable`).
- `quality_score`, `retake_reason`, `retake_tips` y `diagnostics.warnings` siguen informando de problemas de luz/enfoque/plantilla.

Para errores reales de análisis sin resultados:
- `ok:false`
- `status:error`
- `photo_status:foto_no_fiable`

## Fichero a sustituir
Sustituir `main.py` por el incluido en este paquete. No es necesario cambiar `swatches.json`, `requirements.txt` ni `Dockerfile`.

## Comprobación rápida
```bash
python -m py_compile main.py
git diff main.py
git add main.py
git commit -m "Fix Base44 proxy error for low-quality photos"
git push origin main
```
