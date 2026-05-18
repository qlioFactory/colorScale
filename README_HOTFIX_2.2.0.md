# Hotfix 2.2.0 - Conservative calibration

Este parche mantiene el endpoint `POST /analyze-strip` compatible con Base44.

## Motivo

En algunas fotos Android/Redmi la calibración RGB lineal usando las barras gris/azul/verde/roja era demasiado agresiva. Cuando las barras de referencia no encajan bien con una única corrección RGB, el algoritmo podía introducir una dominante magenta/azul en los pads claros y devolver valores altos para cloro, nitrato, cobre o hierro aunque visualmente fueran casi blancos.

## Cambios

- `APP_VERSION = 2.2.0-conservative-calibration`.
- Si el residuo de calibración de las barras es alto, el matching pasa a modo conservador `raw_conservative`.
- Para pads claros/white-like se reduce el peso de la luminosidad y se compara principalmente tono/croma.
- Los pads neutros o casi blancos se fuerzan al valor mínimo de su escala.
- Se evita aplicar la calibración RGB saturada cuando distorsiona los colores.
- Se mantiene la compatibilidad Base44: `status: ok`, `photo_status` separado y resultados siempre que haya lectura.

## Aplicación

Copiar `main.py` sobre el repo actual, comprobar sintaxis y subir:

```bash
python -m py_compile main.py
git add main.py README_HOTFIX_2.2.0.md
git commit -m "Add conservative calibration for Android photos"
git push origin main
```
