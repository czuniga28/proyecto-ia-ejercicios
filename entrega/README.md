# Reconocimiento de Ejecución Biomecánica — CNN+LSTM

**Curso:** Inteligencia Artificial — Ciclo IV  
**Equipo:** Christopher Zúñiga (C28730) · Adrian Arrieta Orozco (B70734)

---

## Contenido de la entrega

```
entrega/
├── README.md                          ← este archivo
├── reporte_final.pdf                  ← paper en formato IEEE
├── reporte_uso_ia.md                  ← reporte de uso de IA
├── requirements.txt                   ← dependencias de Python
├── src/
│   ├── main.py                        ← script principal
│   ├── recognition_model.py           ← arquitectura CNN+LSTM
│   ├── data_loader.py                 ← DataLoaders de PyTorch
│   └── data_manager.py               ← extracción de landmarks
├── models/
│   └── recognition_model.pt          ← pesos del modelo entrenado
├── data/
│   └── pose_landmarks_normalized.csv ← dataset de landmarks (131 videos)
└── figures/
    ├── curva_aprendizaje.png
    ├── hist_landmark_y.png
    ├── heatmap_correlacion.png
    └── boxplot_proxies.png
```

---

## Requisitos

- Python 3.10 o superior
- pip

---

## Instalación

```bash
# 1. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install torch scikit-learn pandas numpy matplotlib
```

> Las dependencias de MediaPipe y OpenCV solo son necesarias si se desea
> re-extraer los landmarks desde los videos originales (no requerido para
> evaluar el modelo ya entrenado).

---

## Opción 1 — Evaluar el modelo entrenado (recomendado)

El modelo ya está entrenado y guardado en `models/recognition_model.pt`.
Solo se necesita el CSV de landmarks incluido en `data/`.

```bash
cd entrega
python3 src/main.py --eval-only
```

**Salida esperada:**

```
── Modo evaluación: cargando modelo guardado ───────────────────────
Split  →  train=93  val=19  test=19
Modelo en        : cpu
Parámetros totales: 310,209
Modelo cargado  ← models/recognition_model.pt

── Resultados en test set (umbral=0.3) ─────────────────────
  Accuracy  : 0.7368
  F1        : 0.8148
  Precision : 0.7333
  Recall    : 0.9167
  Confusion matrix:
              Pred Incorrecto  Pred Correcto
  Real Incorrecto         3              4
  Real Correcto           1             11
```

---

## Opción 2 — Re-entrenar el modelo

Re-entrena desde cero usando el CSV de landmarks incluido.
No requiere los videos originales ni MediaPipe.

```bash
cd entrega
python3 src/main.py --skip-extract
```

El entrenamiento usa semilla fija (seed=42) y early stopping (patience=10).
Duración aproximada: 1–3 minutos en CPU.

---

## Opción 3 — Pipeline completo (requiere videos y MediaPipe)

Solo si se dispone de los videos del dataset y el modelo BlazePose:

```bash
# Colocar los videos en dataset_limpio/ y el modelo en models/pose_landmarker_full.task
python3 src/main.py
```

---

## Hiperparámetros del modelo final

| Parámetro      | Valor | Descripción                          |
|----------------|-------|--------------------------------------|
| `conv_filters` | 64    | Filtros en primera capa CNN          |
| `lstm_units`   | 128   | Neuronas en capa oculta LSTM         |
| `lstm_layers`  | 2     | Capas LSTM apiladas                  |
| `dropout`      | 0.3   | Tasa de dropout                      |
| `lr`           | 1e-3  | Learning rate (Adam)                 |
| `epochs`       | 50    | Máximo de épocas (con early stopping)|
| `patience`     | 10    | Épocas sin mejora antes de detener   |
| `threshold`    | 0.30  | Umbral de decisión (≥ correcto)      |
| `seed`         | 42    | Semilla aleatoria (reproducibilidad) |
