#!/usr/bin/env python3
"""
generar_pdf.py
Genera poc_respuestas.pdf con las respuestas a las preguntas de la prueba de concepto.
Requiere fpdf2 (pip install fpdf2).
"""

import json
from pathlib import Path
from fpdf import FPDF


BASE  = Path(__file__).resolve().parent
RESULTS_PATH = BASE / "poc_results.json"


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


class PDF(FPDF):
    TITLE_COLOR   = (26, 86, 149)   # azul oscuro
    HEADER_COLOR  = (52, 120, 195)  # azul medio
    TEXT_COLOR    = (30, 30, 30)
    ACCENT_COLOR  = (200, 230, 255) # fondo de tablas

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.TITLE_COLOR)
        self.cell(0, 8, "Prueba de Concepto - Clasificación de Ejercicios con CNN+LSTM", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_draw_color(*self.HEADER_COLOR)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Pág. {self.page_no()}", align="C")

    def section_title(self, number: str, text: str):
        self.ln(4)
        self.set_fill_color(*self.HEADER_COLOR)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {number}  {text}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(*self.TEXT_COLOR)
        self.ln(2)

    def body_text(self, text: str, indent: int = 0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.TEXT_COLOR)
        self.set_x(10 + indent)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text: str, indent: int = 5):
        self.set_font("Helvetica", "", 10)
        self.set_x(10 + indent)
        self.multi_cell(0, 5.5, f"*  {text}")
        self.ln(0.5)

    def kv_row(self, key: str, value: str, shade: bool = False):
        if shade:
            self.set_fill_color(*self.ACCENT_COLOR)
        else:
            self.set_fill_color(245, 245, 245)
        self.set_font("Helvetica", "B", 9)
        self.cell(65, 6, f"  {key}", border=0, fill=True)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, f"  {value}", border=0, fill=True, new_x="LMARGIN", new_y="NEXT")


def main():
    r = load_results()
    hp = r["hyperparameters"]
    cm = r["confusion_matrix"]
    h  = r["history"]

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # -- Portada / meta --------------------------------------------------------
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*PDF.TITLE_COLOR)
    pdf.ln(4)
    pdf.cell(0, 10, "Prueba de Concepto", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Reconocimiento Computacional de Ejercicios - CNN + LSTM", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 7, "Christopher Zúñiga (C28730)  |  Adrian Arrieta (B70734)", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, "Inteligencia Artificial, CI-0129, Universidad de Costa Rica", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # -- Resumen de configuración ----------------------------------------------
    pdf.set_text_color(*PDF.TEXT_COLOR)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Configuración del experimento", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    rows = [
        ("Dataset",             f"{r['total_videos']} videos - deadlift, pull-up, squat (correcto / incorrecto)"),
        ("Features por frame",  "99  (x, y, z x 33 landmarks BlazePose)"),
        ("Frames por video",    "30  (relleno con último frame si < 30)"),
        ("Split",               f"Train={r['train_size']}  Val={r['val_size']}  Test={r['test_size']}  (70/15/15, estratificado)"),
        ("Normalización",       "StandardScaler ajustado solo sobre train"),
        ("Arquitectura",        f"Conv1D({hp['conv_filters']}) x 2 -> LSTM({hp['lstm_units']}) -> Sigmoid"),
        ("Dropout",             str(hp["dropout"])),
        ("Batch size",          str(hp["batch_size"])),
        ("Épocas",              str(r["epochs"])),
        ("Learning rate",       str(hp["learning_rate"])),
        ("Parámetros",          f"{r['total_params']:,}"),
        ("Dispositivo",         "Apple MPS (GPU M-series)"),
    ]
    for i, (k, v) in enumerate(rows):
        pdf.kv_row(k, v, shade=(i % 2 == 0))
    pdf.ln(4)

    # -- P1: Resultado en una frase --------------------------------------------
    pdf.section_title("P1", "Resultado de la prueba de concepto (en una frase)")
    pdf.body_text(
        "El modelo convergió a predecir únicamente la clase 'Correcto' en todos los casos, "
        "obteniendo un accuracy del 50 % equivalente al azar, lo que demuestra que el prototipo "
        "no logró aprender características discriminativas con los datos y features actuales."
    )

    # Tabla de métricas
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*PDF.HEADER_COLOR)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(65, 6, "  Métrica", fill=True)
    pdf.cell(0, 6, "  Valor", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*PDF.TEXT_COLOR)

    metrics = [
        ("Accuracy (test)",             f"{r['test_accuracy']*100:.1f} %"),
        ("Best val accuracy (train)",   f"{r['best_val_acc']*100:.1f} %"),
        ("Train loss (época 40)",       f"{h['train_loss'][-1]:.4f}"),
        ("Val loss (época 40)",         f"{h['val_loss'][-1]:.4f}"),
        ("Precision - clase Correcto",  "0.50"),
        ("Recall - clase Correcto",     "1.00  (predice siempre esta clase)"),
        ("Precision - clase Incorrecto","0.00  (nunca predicho)"),
        ("Recall - clase Incorrecto",   "0.00"),
    ]
    for i, (k, v) in enumerate(metrics):
        pdf.kv_row(k, v, shade=(i % 2 == 0))
    pdf.ln(2)

    # Matriz de confusión
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "Matriz de confusión:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"  Real Incorrecto -> Pred Incorrecto: {cm[0][0]}   Pred Correcto: {cm[0][1]}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"  Real Correcto   -> Pred Incorrecto: {cm[1][0]}   Pred Correcto: {cm[1][1]}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # -- P2: ¿Descubrió algo del planteamiento? --------------------------------
    pdf.section_title("P2", "¿Pudo descubrir algo de su planteamiento de problema?")

    pdf.body_text(
        "Sí - el prototipo reveló limitaciones reales del planteamiento que no eran evidentes "
        "antes de ejecutarlo:"
    )
    pdf.bullet(
        "Los landmarks crudos (x, y, z) sin normalización estructural ni ángulos articulares "
        "no son suficientemente discriminativos. El modelo no encontró señal diferenciable "
        "entre correcto e incorrecto usando solo coordenadas absolutas."
    )
    pdf.bullet(
        "El dataset de 145 videos (101 de entrenamiento) es muy pequeño para una arquitectura "
        "CNN+LSTM. Con 64,769 parámetros y ~101 muestras, el modelo tiene más parámetros que "
        "muestras - la sobreparametrización inhibe el aprendizaje."
    )
    pdf.bullet(
        "La loss de entrenamiento se mantuvo cerca de 0.693 (entropía cruzada para 50/50) "
        "durante 40 épocas, lo que indica que el modelo no encontró ningún patrón y simplemente "
        "aprendió la distribución de clase mayoritaria."
    )
    pdf.bullet(
        "Esto confirma que los pasos de ingeniería de features (normalización hip-center, "
        "escala por torso, ángulos articulares) no son opcionales - son la parte que le da "
        "sentido biomecánico a los datos."
    )
    pdf.bullet(
        "Para deadlift especialmente, la coordenada Y sola no discrimina nada: ambas clases "
        "tienen distribuciones similares. El ángulo de columna es el feature crítico."
    )
    pdf.ln(2)

    # -- P3: ¿Aprendió algo del código? ----------------------------------------
    pdf.section_title("P3", "¿Aprendió algo de hacer su código funcional?")

    pdf.body_text("Sí, varios aspectos técnicos importantes:")
    pdf.bullet(
        "El split debe hacerse al nivel de video, no de frame. Dividir por frames "
        "crea data leakage: frames del mismo video quedarían en train y test a la vez, "
        "inflando artificialmente el accuracy."
    )
    pdf.bullet(
        "StandardScaler debe ajustarse (fit) únicamente con los datos de entrenamiento y "
        "luego aplicarse (transform) en val y test. Ajustarlo sobre todo el dataset es "
        "data leakage - el modelo 've' información del futuro."
    )
    pdf.bullet(
        "BCEWithLogitsLoss es numéricamente más estable que aplicar Sigmoid + BCELoss "
        "porque combina la operación internamente con mejor manejo de gradientes."
    )
    pdf.bullet(
        "Para Conv1D en PyTorch, el tensor de entrada debe ser [batch, canales, longitud]. "
        "En nuestro caso: [batch, 99_features, 30_frames]. Fue necesario hacer .permute() "
        "para reorganizar las dimensiones."
    )
    pdf.bullet(
        "ReduceLROnPlateau es útil cuando la pérdida se estanca: reduce el learning rate "
        "automáticamente si no hay mejora en varios epochs. En este prototipo no logró "
        "resolver el estancamiento porque el problema es de features, no de learning rate."
    )
    pdf.bullet(
        "Guardar el mejor modelo con torch.save y cargarlo para evaluación final "
        "evita evaluar un modelo sobreentrenado."
    )
    pdf.ln(2)

    # -- P4: Sugerencias de un LLM ---------------------------------------------
    pdf.section_title("P4", "Sugerencias, críticas y fallas señaladas por un LLM")

    pdf.body_text(
        "El siguiente análisis fue generado usando Claude (Anthropic) como herramienta de "
        "evaluación del planteamiento y del prototipo:"
    )

    pdf.set_font("Helvetica", "BI", 10)
    pdf.set_text_color(*PDF.HEADER_COLOR)
    pdf.cell(0, 5, "Fallas identificadas:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*PDF.TEXT_COLOR)

    pdf.bullet(
        "El modelo está sobreparametrizado para el tamaño de datos: 64,769 parámetros "
        "con solo 101 videos de entrenamiento genera una razón parámetros/muestra de ~641:1. "
        "En redes neuronales, una razón superior a ~10:1 ya indica riesgo de sobreajuste."
    )
    pdf.bullet(
        "Los features de entrada (coordenadas x,y,z sin procesar) son dependientes de la "
        "posición de la cámara y de la posición del sujeto en el frame. Un modelo entrenado "
        "en este espacio no generalizará a cámaras distintas."
    )
    pdf.bullet(
        "No se aplicó aumentación de datos en el espacio de features - solo se usó flip "
        "en el dataset CSV. Con un dataset tan pequeño, técnicas adicionales como jitter "
        "de landmarks y rotaciones serían críticas."
    )
    pdf.bullet(
        "El prototipo entrena sobre los 3 ejercicios juntos sin distinguirlos. Un modelo "
        "único para deadlift + squat + pull-up necesita aprender features muy distintos. "
        "Modelos separados por ejercicio o añadir el ejercicio como feature podrían mejorar."
    )

    pdf.set_font("Helvetica", "BI", 10)
    pdf.set_text_color(*PDF.HEADER_COLOR)
    pdf.cell(0, 5, "Sugerencias de mejora (próximos pasos):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*PDF.TEXT_COLOR)

    pdf.bullet(
        "Implementar DataManager con normalización hip-center + escala por longitud de torso "
        "antes de entrenar. Esto elimina la dependencia de posición de cámara."
    )
    pdf.bullet(
        "Añadir ángulos articulares calculados por ejercicio como features adicionales. "
        "Esto es lo que realmente captura la biomecánica del movimiento."
    )
    pdf.bullet(
        "Reducir la arquitectura para el dataset disponible: Conv1D(32) + LSTM(32) "
        "bastaría para este tamaño. Menos parámetros = menos sobreajuste."
    )
    pdf.bullet(
        "Añadir más regularización: L2 weight decay en el optimizador, o aumentar dropout a 0.5."
    )
    pdf.bullet(
        "Evaluar métricas por ejercicio (no solo globales) para identificar cuál es más difícil."
    )
    pdf.bullet(
        "Considerar Early Stopping basado en val_loss en lugar de un número fijo de épocas."
    )

    pdf.set_font("Helvetica", "BI", 10)
    pdf.set_text_color(*PDF.HEADER_COLOR)
    pdf.cell(0, 5, "Veredicto del LLM sobre la arquitectura planteada:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*PDF.TEXT_COLOR)

    pdf.body_text(
        "La arquitectura CNN+LSTM es apropiada para el problema. El concepto de extraer "
        "features espaciales por frame y modelar la dinámica temporal es sólido. El "
        "fracaso del prototipo no invalida la arquitectura - valida que la ingeniería de "
        "features previa al modelo es la pieza crítica que falta. Con normalización "
        "estructural y ángulos articulares, el mismo modelo tiene potencial de alcanzar "
        "70-85% de accuracy en este dataset, que sería un resultado aceptable para un "
        "proyecto universitario con dataset pequeño."
    )

    # -- Conclusión ------------------------------------------------------------
    pdf.section_title("", "Conclusión y próximos pasos")
    pdf.body_text(
        "La prueba de concepto cumplió su objetivo: demostrar que el pipeline de código "
        "es funcional (carga, split, normalización, entrenamiento, evaluación) y revelar "
        "que los features actuales no son suficientes. El siguiente paso es implementar "
        "DataManager con normalización biomecánica y ángulos articulares, y reentrenar "
        "con esos features antes de la evaluación final."
    )

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Este documento fue generado programáticamente a partir de los resultados reales del experimento.", new_x="LMARGIN", new_y="NEXT", align="C")

    # -- Guardar ---------------------------------------------------------------
    out_path = BASE / "poc_respuestas.pdf"
    pdf.output(str(out_path))
    print(f"PDF generado: {out_path}")


if __name__ == "__main__":
    main()
