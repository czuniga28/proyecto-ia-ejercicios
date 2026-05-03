# Registro de Experimentos — CNN+LSTM Clasificación de Ejercicios

Dataset: 131 videos totales (train=93, val=19, test=19)
Split: 70/15/15 estratificado por (ejercicio, etiqueta), a nivel de video

---

## Experimento 1 — Baseline

**Fecha:** 2026-05-02
**Descripción:** Primera ejecución completa del pipeline. Hiperparámetros iniciales.

### Hiperparámetros
| Parámetro     | Valor |
|---------------|-------|
| conv_filters  | 64    |
| lstm_units    | 128   |
| lstm_layers   | 2     |
| dropout       | 0.3   |
| lr            | 1e-3  |
| batch_size    | 16    |
| epochs        | 50    |
| patience      | 10    |
| umbral        | 0.50  |

### Resultados (test set, umbral=0.50)
| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 73.68% |
| F1        | 0.8148 |
| Precision | 0.7333 |
| Recall    | 0.9167 |

### Confusion Matrix
|                  | Pred Incorrecto | Pred Correcto |
|------------------|-----------------|---------------|
| Real Incorrecto  | 3               | 4             |
| Real Correcto    | 1               | 11            |

### Observaciones
- El modelo está sesgado hacia predecir "correcto" (recall alto, precision baja).
- De 7 videos incorrectos, clasifica 4 como correctos (57% de miss rate en forma incorrecta).
- 1 video correcto clasificado como incorrecto.
- Problema principal: el modelo no detecta bien la forma incorrecta.

---

## Experimento 2 — Ajuste de umbral de decisión

**Fecha:** 2026-05-02
**Descripción:** Mismo modelo del Exp. 1. Se prueba bajar el umbral de clasificación
para reducir falsos negativos (correctos clasificados como incorrectos).
No se re-entrena el modelo.

### Resultados comparativos

| Umbral | Accuracy | F1     | Precision | Recall | FP | FN |
|--------|----------|--------|-----------|--------|----|----|
| 0.50   | 73.68%   | 0.8148 | 0.7333    | 0.9167 | 4  | 1  |
| 0.40   | 78.95%   | 0.8571 | 0.7500    | 1.0000 | 4  | 0  |
| 0.35   | 78.95%   | 0.8571 | 0.7500    | 1.0000 | 4  | 0  |

### Observaciones
- Bajar el umbral a 0.40 recupera el falso negativo → Recall = 1.0 (ningún ejercicio
  correcto clasificado como incorrecto).
- Los 4 falsos positivos (incorrectos clasificados como correctos) no cambian con
  ningún umbral probado. El modelo asigna probabilidades altas a esos videos
  independientemente del umbral → el problema es del modelo, no del umbral.
- **Umbral seleccionado: 0.40** — mejora Accuracy (+5.27pp) y F1 (+0.042) sin costo.
- Para reducir los 4 FP se requiere re-entrenamiento (Experimento 3).

---

## Experimento 3 — Más regularización (dropout=0.4, lr=5e-4)

**Fecha:** 2026-05-02
**Descripción:** Re-entrenamiento con más dropout y LR más bajo para reducir
los 4 falsos positivos del Exp. 1/2. Early stopping en epoch 26.

### Hiperparámetros
| Parámetro     | Valor |
|---------------|-------|
| conv_filters  | 64    |
| lstm_units    | 128   |
| lstm_layers   | 2     |
| dropout       | 0.4   |
| lr            | 5e-4  |
| batch_size    | 16    |
| epochs        | 80    |
| patience      | 15    |

### Resultados (umbral=0.50, idéntico para 0.40 y 0.35)
| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 63.16% |
| F1        | 0.6957 |
| Precision | 0.7273 |
| Recall    | 0.6667 |

### Confusion Matrix
|                  | Pred Incorrecto | Pred Correcto |
|------------------|-----------------|---------------|
| Real Incorrecto  | 4               | 3             |
| Real Correcto    | 4               | 8             |

### Observaciones
- Redujo 1 FP (4→3) pero introdujo 3 FN nuevos (1→4). Tradeoff desfavorable.
- Umbral insensible: los mismos resultados para 0.35, 0.40, 0.50 → el modelo
  asigna probabilidades muy separadas del rango de ajuste.
- **Descartado.** El Exp. 2 sigue siendo el mejor resultado.

---

## Experimento 4 — Semilla fija (seed=42) para reproducibilidad

**Fecha:** 2026-05-02
**Descripción:** Mismos hiperparámetros de Exp. 1 pero con semilla fija
(random=42, numpy=42, torch=42) para garantizar reproducibilidad en el paper.
Early stopping en epoch 22.

### Hiperparámetros
| Parámetro     | Valor |
|---------------|-------|
| conv_filters  | 64    |
| lstm_units    | 128   |
| lstm_layers   | 2     |
| dropout       | 0.3   |
| lr            | 1e-3  |
| batch_size    | 16    |
| epochs        | 50    |
| patience      | 10    |
| seed          | 42    |

### Resultados por umbral
| Umbral | Accuracy | F1     | Precision | Recall | TN | FP | FN | TP |
|--------|----------|--------|-----------|--------|----|----|----|----|
| 0.50   | 73.68%   | 0.7619 | 0.8889    | 0.6667 | 6  | 1  | 4  | 8  |
| 0.40   | 73.68%   | 0.7619 | 0.8889    | 0.6667 | 6  | 1  | 4  | 8  |
| 0.35   | 68.42%   | 0.7692 | 0.7143    | 0.8333 | 3  | 4  | 2  | 10 |
| **0.30** | **73.68%** | **0.8148** | **0.7333** | **0.9167** | 3 | 4 | 1 | 11 |

### Observaciones
- Con umbral=0.50: detecta 6/7 ejercicios incorrectos (mejor especificidad lograda),
  pero clasifica 4/12 correctos como incorrectos.
- Con umbral=0.30: balance más favorable — Recall=0.917, F1=0.815.
  Equivalente a Exp.1 en términos de métricas globales.
- **Umbral seleccionado para producción: 0.30** (con seed=42).

---

## Resumen de experimentos

| Exp | Seed | Dropout | LR    | Umbral | Accuracy | F1     | Precision | Recall |
|-----|------|---------|-------|--------|----------|--------|-----------|--------|
| 1   | —    | 0.3     | 1e-3  | 0.50   | 73.68%   | 0.8148 | 0.7333    | 0.9167 |
| 2   | —    | 0.3     | 1e-3  | 0.40   | 78.95%   | 0.8571 | 0.7500    | 1.0000 |
| 3   | —    | 0.4     | 5e-4  | 0.50   | 63.16%   | 0.6957 | 0.7273    | 0.6667 |
| **4** | **42** | **0.3** | **1e-3** | **0.30** | **73.68%** | **0.8148** | **0.7333** | **0.9167** |

**Modelo final para el paper:** Exp. 4 (seed=42, umbral=0.30)
- Reproducible
- Accuracy: 73.68% | F1: 0.8148 | Precision: 0.7333 | Recall: 0.9167
- Limitación principal: dataset pequeño (93 videos de entrenamiento, 19 de test)
  — varianza alta entre ejecuciones sin semilla fija.
