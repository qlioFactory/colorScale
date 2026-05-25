# ColorScale hotfix 2.3.0 - local blank + rotation

Versión: `2.3.0-local-blank-and-rotation`

Cambios incluidos:

1. Reintento automático de orientación:
   - analiza la imagen original;
   - si falla o hay una versión mejor, prueba rotaciones 90º/180º/270º;
   - ayuda con fotos subidas desde galería en horizontal o con metadatos de orientación inconsistentes.

2. Corrección para pads claros usando blanco local:
   - para cloro, nitrato, cobre, hierro y aluminio se muestrea el blanco de la tira entre pads;
   - si el pad tiene prácticamente el mismo color que el blanco local, se interpreta como valor bajo/0;
   - evita que el Redmi 9 convierta pads blancos/beige por iluminación en positivos falsos de cobre/hierro.

3. Diagnóstico ampliado:
   - `diagnostics.inputRotationApplied`;
   - `diagnostics.originalImageSize`;
   - por resultado: `local_blank_rgb` y `local_blank_deltaE`.

Aplicación:

```bash
cp main.py /ruta/a/colorScale/main.py
python -m py_compile main.py
git add main.py README_HOTFIX_2.3.0.md
git commit -m "Improve Redmi analysis with local blank correction and rotation retry"
git push origin main
```

Después de desplegar, comprobar:

```bash
curl https://TU_BACKEND/health
```

Debe devolver `2.3.0-local-blank-and-rotation`.
