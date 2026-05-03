# Reporte de Uso de IA — Proyecto CNN+LSTM Biomecánica

**Curso:** Inteligencia Artificial — Ciclo IV  
**Equipo:** Christopher Zúñiga (C28730) · Adrian Arrieta Orozco (B70734)  
**Herramienta:** Claude Code (claude-sonnet-4-6) via Anthropic CLI

---

## Declaración de Uso

El equipo utilizó herramientas de inteligencia artificial **exclusivamente como apoyo al aprendizaje**, de manera análoga a consultar a un profesor o tutor. La IA no tomó decisiones de diseño, no eligió la arquitectura del modelo, no definió los hiperparámetros ni interpretó los resultados: todas esas responsabilidades recayeron en los integrantes del equipo.

El uso se concentró en dos tipos de interacción:

1. **Explicación de conceptos:** cuando un integrante no comprendía un concepto técnico (ej. cómo funciona una convolución internamente, qué es `nn.Parameter`, qué significa `batch_first` en LSTM), se consultó a la IA para obtener una explicación detallada con ejemplos, del mismo modo que se haría con un profesor de oficina.

2. **Revisión de código:** el código escrito por los integrantes fue revisado por la IA para detectar errores de sintaxis, problemas estructurales o inconsistencias, del mismo modo que un profesor revisa un ejercicio de laboratorio y señala qué corregir sin reescribirlo.

---

## Decisiones tomadas por el equipo (no delegadas a la IA)

Las siguientes decisiones fueron tomadas de forma autónoma por el equipo, con base en el análisis del problema y los conocimientos del curso:

| Decisión | Justificación propia |
|---|---|
| Selección del dataset de Kaggle | Criterio de disponibilidad y variedad de ejercicios |
| Formulación como clasificación binaria | Simpleza del problema y naturaleza de las etiquetas |
| Selección de arquitectura CNN+LSTM | Revisión de trabajos relacionados y naturaleza espacio-temporal del problema |
| Selección de features: landmarks + ángulos articulares | Conocimiento biomecánico del equipo |
| Número de capas CNN (2) y filtros (64→128) | Decisión propia basada en tamaño del dataset |
| Hiperparámetros finales: dropout=0.3, lr=1e-3 | Evaluados experimentalmente y seleccionados por el equipo |
| Umbral de clasificación τ=0.30 | Seleccionado tras analizar la tabla de experimentos |
| Interpretación de resultados y análisis de limitaciones | Realizada íntegramente por el equipo |
| Redacción del paper | Escrita por el equipo; la IA no generó texto del informe |

---

## Tareas auxiliares apoyadas por IA

Las siguientes tareas son de carácter operacional o de infraestructura y no constituyen el núcleo académico del proyecto:

- **`download_bad_form.sh`:** script de descarga de videos con `yt-dlp`. No tiene relevancia conceptual para el proyecto.
- **`normalize_dataset.py`:** script de normalización de resolución y FPS de videos con OpenCV. Tarea de preprocesamiento mecánico, no relacionada con el diseño del modelo.
- Configuración del entorno de Python y dependencias.

---

## Registro Representativo de Prompts

Los prompts se agrupan por categoría. Se listan los más representativos de cada tipo de interacción.

### A — Explicación de conceptos

| # | Consulta | Concepto aprendido |
|---|---|---|
| 1 | ¿Qué es `nn.Parameter` y por qué PyTorch necesita saber qué tensores son aprendibles? | Autograd, grafo computacional, `requires_grad` |
| 2 | ¿Cómo funciona una convolución 1D internamente, a nivel de operaciones matemáticas? | Producto punto deslizante, ventanas, relación entre `in_channels`, `out_channels` y `kernel_size` |
| 3 | ¿Qué hace `tensor.unfold()` y cómo se usa para extraer ventanas deslizantes? | Operación de extracción de ventanas sin loops, forma del tensor resultante |
| 4 | ¿Qué significa `batch_first=True` en LSTM y por qué importa? | Convención de dimensiones en PyTorch, `[T, B, F]` vs `[B, T, F]` |
| 5 | ¿Por qué usamos 128 neuronas en el LSTM y 2 capas en vez de más? | Relación entre capacidad del modelo y tamaño del dataset, riesgo de overfitting |
| 6 | ¿Por qué el CNN solo tiene 2 capas? | Balance entre profundidad, dataset pequeño y longitud de secuencia de 30 frames |
| 7 | ¿Qué es BCELoss y por qué es la función de pérdida correcta para clasificación binaria con salida sigmoide? | Cross-entropy binaria, relación con probabilidades |
| 8 | ¿Qué hace `optimizer.zero_grad()` y por qué es necesario antes de cada batch? | Acumulación de gradientes en PyTorch, ciclo de entrenamiento |

### B — Revisión de código

| # | Qué se revisó | Error encontrado |
|---|---|---|
| 9 | Implementación de `CNNLSTMClassifier.__init__` | `nn.BatchNorm1` en vez de `nn.BatchNorm1d`; `self.lstm` definido dos veces; `self.head` conflictaba con `fc_w`/`fc_b` |
| 10 | Implementación de `_conv1d_manual` | Lógica correcta; se identificó que los tests de `_flip_raw` estaban mal planteados (no consideraban el swap de pares) |
| 11 | Estructura general de `recognition_model.py` | Docstring mal posicionado hacía que `__init__` quedara atrapado como string muerto dentro de `_conv1d_manual` |

### C — Consultas de PyTorch y NumPy

| # | Consulta | Uso |
|---|---|---|
| 12 | ¿Cómo usar `tensor.permute()` para reordenar dimensiones? | Necesario en `forward` para adaptar entre Conv1D y LSTM |
| 13 | ¿Qué hace `torch.no_grad()` y cuándo usarlo? | Evaluación sin cálculo de gradientes para ahorro de memoria |
| 14 | ¿Cómo funciona `WeightedRandomSampler` para balancear clases? | Implementación en `DataLoader` |

### D — Recomendaciones evaluadas por el equipo

| # | Recomendación recibida | Decisión del equipo |
|---|---|---|
| 15 | Probar dropout=0.4 y lr=5e-4 como Experimento 3 | **Evaluado y descartado:** los resultados fueron peores (Accuracy 63.16% vs 73.68%) |
| 16 | Bajar umbral de 0.5 a 0.4 como primer ajuste | **Aceptado parcialmente:** se bajó a 0.30 tras revisar la tabla comparativa |
| 17 | Usar split a nivel de video (no fotograma) para evitar data leakage | **Aceptado:** era el enfoque correcto una vez comprendido el riesgo |

---

## Notas

- Los chats completos exportados están disponibles bajo petición.
- Toda decisión final reflejada en el código y el paper fue tomada y comprendida por los integrantes del equipo.
