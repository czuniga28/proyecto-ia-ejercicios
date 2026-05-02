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

## Experimento 3 — (Pendiente)

**Descripción planeada:** Re-entrenamiento con hiperparámetros ajustados para
mejorar la detección de forma incorrecta (reducir falsos positivos).

Cambios planificados:
- dropout: 0.3 → 0.4 (más regularización)
- lr: 1e-3 → 5e-4 (convergencia más fina)
- epochs: 50 → 80, patience: 10 → 15
- umbral: 0.40 (fijado desde Exp. 2)
