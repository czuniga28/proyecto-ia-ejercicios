#!/usr/bin/env python3
"""
Pipeline completo: extracción de landmarks → entrenamiento → evaluación.

Uso:
    python src/main.py                  # extrae CSV si no existe, entrena y evalúa
    python src/main.py --skip-extract   # usa CSV existente, entrena y evalúa
    python src/main.py --eval-only      # carga modelo guardado y solo evalúa
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from data_loader import DataLoader
from data_manager import DataManager
from recognition_model import RecognitionModel

# Semilla fija para reproducibilidad completa entre ejecuciones
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_limpio"
MODEL_PATH  = BASE_DIR / "models" / "pose_landmarker_full.task"
CSV_PATH    = BASE_DIR / "data"   / "pose_landmarks_normalized.csv"
WEIGHTS_PATH = BASE_DIR / "models" / "recognition_model.pt"

# ── Hiperparámetros ────────────────────────────────────────────────────────────
HYPERPARAMS = {
    "conv_filters": 64,
    "lstm_units":   128,
    "lstm_layers":  2,
    "dropout":      0.3,
    "lr":           1e-3,
}
TRAIN_PARAMS = {
    "batch_size": 16,
    "epochs":     50,
    "patience":   10,
}


def step_extract(force: bool = False) -> None:
    """Extrae landmarks de los videos y guarda el CSV. Se salta si ya existe."""
    if CSV_PATH.exists() and not force:
        print(f"CSV ya existe → {CSV_PATH}  (usa --force-extract para regenerar)\n")
        return

    print("── Paso 1: Extrayendo landmarks de los videos ──────────────────────")
    dm = DataManager(
        dataset_dir = DATASET_DIR,
        model_path  = MODEL_PATH,
        output_csv  = CSV_PATH,
    )
    dm.extract_from_videos()
    print()


def step_train() -> tuple:
    """Construye DataLoaders, entrena el modelo y devuelve (model, loaders)."""
    print("── Paso 2: Construyendo DataLoaders ────────────────────────────────")
    dl = DataLoader(csv_path=CSV_PATH)
    train_loader, val_loader, test_loader, scaler = dl.build_loaders(
        batch_size = TRAIN_PARAMS["batch_size"],
    )
    print()

    print("── Paso 3: Iniciando modelo ────────────────────────────────────────")
    model = RecognitionModel(**HYPERPARAMS)
    print()

    print("── Paso 4: Entrenando ──────────────────────────────────────────────")
    history = model.train_model(
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = TRAIN_PARAMS["epochs"],
        patience     = TRAIN_PARAMS["patience"],
    )
    print()

    model.save(WEIGHTS_PATH)
    return model, test_loader, history


def step_evaluate(model: RecognitionModel, test_loader) -> None:
    """Evalúa el modelo sobre el test set."""
    print("── Paso 5: Evaluando en test set ───────────────────────────────────")
    model.evaluate(test_loader)


def step_eval_only() -> None:
    """Carga un modelo guardado y evalúa sin entrenar."""
    print("── Modo evaluación: cargando modelo guardado ───────────────────────")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {WEIGHTS_PATH}\n"
            "Entrenalo primero con: python src/main.py"
        )

    dl = DataLoader(csv_path=CSV_PATH)
    _, _, test_loader, _ = dl.build_loaders(batch_size=TRAIN_PARAMS["batch_size"])

    model = RecognitionModel(**HYPERPARAMS)
    model.load(WEIGHTS_PATH)
    model.evaluate(test_loader)


# ── Punto de entrada ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline CNN+LSTM — clasificación de ejercicios")
    parser.add_argument("--skip-extract", action="store_true",
                        help="No extraer landmarks; usar CSV existente")
    parser.add_argument("--force-extract", action="store_true",
                        help="Forzar re-extracción aunque el CSV ya exista")
    parser.add_argument("--eval-only", action="store_true",
                        help="Cargar modelo guardado y solo evaluar")
    args = parser.parse_args()

    if args.eval_only:
        step_eval_only()
        return

    if not args.skip_extract:
        step_extract(force=args.force_extract)

    model, test_loader, _ = step_train()
    step_evaluate(model, test_loader)


if __name__ == "__main__":
    main()
