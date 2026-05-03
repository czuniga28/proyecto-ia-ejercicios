# Reconocimiento de Ejercicios con CNN + LSTM

**Curso:** Inteligencia Artificial — Ciclo IV  
**Equipo:** Christopher Zúñiga (C28730) · Adrian Arrieta Orozco (B70734)

---

## Descripción del Problema

Clasificación binaria de la **calidad de ejecución biomecánica** en tres ejercicios de fuerza mediante una arquitectura híbrida CNN+LSTM y extracción de landmarks con MediaPipe BlazePose.

| Ejercicio | Clases |
|-----------|--------|
| Sentadilla (*Squat*) | Correcto / Incorrecto |
| Dominada (*Pull-up*) | Correcto / Incorrecto |
| Peso Muerto (*Deadlift*) | Correcto / Incorrecto |

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy | 73.68% |
| F1-score | 0.8148 |
| Precision | 0.7333 |
| Recall | 0.9167 |

Modelo final: `seed=42`, umbral de decisión `τ=0.30`, early stopping en epoch 12/22.

---

## Arquitectura

```
Video (30 frames) → MediaPipe BlazePose → 33 landmarks (x,y,z) + 9 ángulos
→ Tensor [30, 108] → CNN 1D (manual) → LSTM → Sigmoide → Correcto/Incorrecto
```

La capa CNN fue implementada manualmente con `nn.Parameter` + `tensor.unfold()` para comprender el mecanismo interno de la convolución.

---

## Estructura del Repositorio

```
proyecto_ia/
├── src/
│   ├── main.py                  # Pipeline completo (extraer → entrenar → evaluar)
│   ├── data_manager.py          # Extracción de landmarks con MediaPipe
│   ├── data_loader.py           # DataLoaders de PyTorch con split estratificado
│   └── recognition_model.py    # Arquitectura CNN+LSTM y entrenamiento
├── paper/
│   ├── main.tex                 # Paper IEEE (§1-9, figuras, resultados)
│   ├── references.bib           # 6 referencias bibliográficas
│   └── ai_usage_report.md      # Reporte de uso de IA
├── notebooks/
│   └── exploracion_datos.ipynb  # Análisis exploratorio del dataset
├── figures/                     # Gráficas de exploración + curva de aprendizaje
├── scripts/                     # Scripts auxiliares (descarga, normalización)
├── data/                        # CSV de landmarks normalizados (131 videos)
├── entrega/                     # Carpeta lista para entrega en Moodle
└── .scratch/experiments/        # Registro de 4 experimentos de ajuste
```

---

## Uso Rápido

### Evaluar el modelo entrenado

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch scikit-learn pandas numpy matplotlib
python3 src/main.py --eval-only
```

### Re-entrenar desde el CSV

```bash
python3 src/main.py --skip-extract
```

### Pipeline completo (requiere videos + MediaPipe)

```bash
python3 src/main.py
```

---

## Dataset

**Forma correcta:** [Workout/Exercises Video Dataset — Kaggle](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)  
**Forma incorrecta:** descargada con `scripts/download_bad_form.sh` (requiere `yt-dlp`)

> Los videos no están incluidos en el repositorio (~5 GB). El CSV de landmarks extraídos sí está incluido en `data/` y en `entrega/data/`.

---

## Tecnologías

| Herramienta | Uso |
|-------------|-----|
| `mediapipe` | Extracción de landmarks BlazePose (33 puntos) |
| `opencv-python` | Lectura de fotogramas |
| `torch` | CNN+LSTM, entrenamiento, evaluación |
| `scikit-learn` | StandardScaler, métricas, split estratificado |
| `numpy` / `pandas` | Operaciones tensoriales y manejo de datos |

---

## Progreso

| Etapa | Estado |
|-------|--------|
| Selección de dataset y definición del problema | ✅ Completo |
| Diseño preliminar, arquitectura y metodología | ✅ Completo |
| Selección de variables y características biomecánicas | ✅ Completo |
| Exploración inicial de datos (histogramas, heatmaps) | ✅ Completo |
| Preparación del dataset (normalización, split, missing frames) | ✅ Completo |
| Implementación (DataManager, DataLoader, RecognitionModel) | ✅ Completo |
| Entrenamiento y ajuste de hiperparámetros (4 experimentos) | ✅ Completo |
| Evaluación del modelo (Accuracy 73.68%, F1 0.815) | ✅ Completo |
| Informe final IEEE con figuras y resultados | ✅ Completo |
| Reporte de uso de IA | ✅ Completo |

---

## Licencia

Proyecto académico — Universidad de Costa Rica. No se permite redistribución comercial.
