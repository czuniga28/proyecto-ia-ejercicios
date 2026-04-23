# Reporte de Uso de IA — Proyecto CNN+LSTM Biomecánica

**Curso:** Inteligencia Artificial — Ciclo IV  
**Equipo:** Christopher Zúñiga (C28730) · Adrian Arrieta Orozco (B70734)  
**Herramienta:** Claude Code (claude-sonnet-4-6) via Anthropic CLI

---

## Declaración de Uso

El equipo utilizó herramientas de IA como **apoyo y no sustitución**. Todo diseño, decisión arquitectónica y comprensión del problema fue trabajo propio; la IA asistió en implementación técnica, revisión de código y sugerencias de algoritmos que fueron evaluadas y aceptadas/rechazadas por el equipo.

---

## Registro de Prompts y Conversaciones

### Sesión 1 — Diseño del proyecto y descarga de dataset

| # | Prompt | Propósito | Resultado usado |
|---|--------|-----------|-----------------|
| 1 | Diseño inicial del pipeline CNN+LSTM para clasificación de ejercicios | Validar arquitectura propuesta | Arquitectura confirmada: Conv1D → LSTM → Sigmoid |
| 2 | Script para descargar videos de mala forma de YouTube con yt-dlp | Automatizar descarga del dataset | `download_bad_form.sh` — revisado y aceptado |

---

### Sesión 2 — Normalización del dataset

| # | Prompt | Propósito | Resultado usado |
|---|--------|-----------|-----------------|
| 3 | Crear script de normalización de videos: 640×480, 30fps, 30 frames por video | Estandarizar el dataset | `normalize_dataset.py` v1 — base aceptada |
| 4 | Mejorar detección de segmento del ejercicio: ignorar intros y cortes de YouTube | Robustez contra ruido de YouTube | v2: ROI central, skip primer/último 10% |
| 5 | Reemplazar ventana deslizante por umbral adaptativo sobre señal suavizada | Los videos de mala forma no capturaban el movimiento con v2 | v3: `find_exercise_segment()` con percentil 25, promedio móvil 20 frames, mayor energía total |
| 6 | Recomendaciones para grabar videos de mala forma con iPhone (una toma, múltiples reps) | Estrategia de recolección de datos propia | Aceptado: 60fps, plano sagital, 28 reps con pausa 2s entre reps |

---

### Sesión 3 — Exploración de datos

| # | Prompt | Propósito | Resultado usado |
|---|--------|-----------|-----------------|
| 7 | Guía paso a paso para exploración de datos con MediaPipe y Jupyter | Planificar el trabajo de exploración | Estructura del notebook `exploracion_datos.ipynb` |

---

## Decisiones donde se rechazó la sugerencia de la IA

- Se evaluó usar ventana deslizante centrada en el clímax (pico de movimiento) para seleccionar frames — descartado porque en videos de YouTube el pico puede ser una transición de edición, no el clímax del ejercicio.
- Se evaluó usar percentil 50 como umbral de actividad — ajustado a percentil 25 para capturar el arranque y aterrizaje del movimiento completo.

---

## Notas

- Los chats exportados están disponibles bajo petición.
- Este documento se actualiza conforme avanza el proyecto.
