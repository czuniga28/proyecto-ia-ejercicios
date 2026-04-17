# Reconocimiento de Ejercicios con CNN + LSTM

**Curso:** Inteligencia Artificial — Ciclo IV  
**Equipo:** Christopher Zúñiga (C28730) · Adrian Arrieta Orozco (B70734)

---

## Descripción del Problema

Clasificación binaria de la **calidad de ejecución biomecánica** en tres ejercicios:

| Ejercicio | Clases |
|-----------|--------|
| Sentadilla (*Squat*) | Correcto / Incorrecto |
| Dominada (*Pull-up*) | Correcto / Incorrecto |
| Peso Muerto (*Deadlift*) | Correcto / Incorrecto |

---

## Arquitectura del Modelo

Pipeline híbrido **CNN + LSTM** que combina extracción de características espaciales por fotograma con modelado de dinámicas temporales entre fotogramas.

```
Video → 30 fotogramas → MediaPipe BlazePose → 33 landmarks (x, y, z)
      → Ángulos articulares → CNN 1D → LSTM → Sigmoide → Correcto / Incorrecto
```

**Componentes:**
- **Spatial (CNN 1D):** extrae patrones de postura por fotograma
- **Temporal (LSTM):** modela la secuencia de 30 fotogramas
- **Clasificación:** capa densa + sigmoide → probabilidad binaria

---

## Estructura del Repositorio

```
proyecto_ia/
├── paper/
│   ├── main.tex            # Informe LaTeX del proyecto
│   └── references.bib      # Bibliografía
├── documento_tarea.pdf     # Enunciado y diseño preliminar
├── download_bad_form.sh    # Script para descargar videos de mala forma
└── README.md
```

> **Nota:** Los videos del dataset no están incluidos en el repositorio por su tamaño (≈ 5 GB). Ver instrucciones de obtención abajo.

---

## Dataset

**Fuente principal (forma correcta):** [Workout/Exercises Video Dataset — Kaggle](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)

**Forma incorrecta:** descargar con el script incluido (requiere `yt-dlp`):

```bash
# Instalar dependencia
pip install yt-dlp

# Descargar ~25 videos de mala forma por ejercicio
bash download_bad_form.sh
```

**Estructura esperada de `videos/` tras la descarga:**

```
videos/
├── deadlift/        # ~32 videos — forma correcta
├── deadlift_bad/    # ~25 videos — forma incorrecta
├── squat/           # ~29 videos — forma correcta
├── squat_bad/       # ~25 videos — forma incorrecta
├── pull Up/         # ~26 videos — forma correcta  ← directorio con espacio
└── pull_up_bad/     # ~25 videos — forma incorrecta
```

---

## Tecnologías

| Herramienta | Uso |
|-------------|-----|
| `mediapipe` | Extracción de landmarks BlazePose (33 puntos) |
| `opencv-python` | Lectura y muestreo de fotogramas |
| `numpy` | Operaciones tensoriales y cálculo de ángulos |
| `torch` / `keras` | Definición y entrenamiento del modelo CNN+LSTM |
| Jupyter Notebooks | Exploración de datos e informes |

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/czuniga28/proyecto-ia-ejercicios.git
cd proyecto-ia-ejercicios

# Instalar dependencias
pip install mediapipe opencv-python numpy torch yt-dlp

# Descargar dataset de mala forma
bash download_bad_form.sh
```

---

## Progreso

| Etapa | Estado |
|-------|--------|
| Selección de dataset y definición del problema | ✅ Completo |
| Diseño preliminar y arquitectura | ✅ Completo |
| Selección de variables y características | ✅ Completo |
| Exploración inicial de datos | ⏳ Pendiente |
| Preparación del dataset | ⏳ Pendiente |
| Entrenamiento y ajuste de hiperparámetros | ⏳ Pendiente |
| Evaluación del modelo | ⏳ Pendiente |
| Informe final | ⏳ Pendiente |

---

## Licencia

Proyecto académico — Universidad. No se permite redistribución comercial.
