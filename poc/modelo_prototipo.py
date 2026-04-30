#!/usr/bin/env python3
"""
modelo_prototipo.py
Prueba de concepto: CNN + LSTM para clasificación binaria de forma de ejercicio.

Dataset : data/pose_landmarks_dataset.csv
Input   : secuencia de 30 frames x 99 features (x,y,z de 33 landmarks BlazePose)
Output  : 0 = incorrecto, 1 = correcto
Split   : 70 % train / 15 % val / 15 % test  (estratificado por ejercicio+etiqueta)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from torch.utils.data import DataLoader, Dataset

# ── Reproducibilidad ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hiperparámetros ───────────────────────────────────────────────────────────
FRAMES       = 30          # frames por video (se rellena si < 30)
N_LANDMARKS  = 33
N_COORDS     = 3           # solo x, y, z  (sin visibility)
N_FEATURES   = N_LANDMARKS * N_COORDS   # 99

CONV_FILTERS = 64
LSTM_UNITS   = 64
DROPOUT      = 0.3
BATCH_SIZE   = 16
EPOCHS       = 40
LR           = 1e-3

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "pose_landmarks_dataset.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga y construcción de tensores por video
# ─────────────────────────────────────────────────────────────────────────────

def load_video_tensors(csv_path: Path):
    df = pd.read_csv(csv_path)

    coord_cols = [f"{c}{i}" for i in range(N_LANDMARKS) for c in ("x", "y", "z")]

    video_tensors, labels, strat_keys = [], [], []

    for vid_name, grp in df.groupby("video_name", sort=False):
        grp = grp.sort_values("frame_index")
        feats = grp[coord_cols].values.astype(np.float32)   # [n_frames, 99]

        # Rellenar hasta FRAMES repitiendo el último frame
        if len(feats) < FRAMES:
            pad = np.tile(feats[-1], (FRAMES - len(feats), 1))
            feats = np.vstack([feats, pad])
        feats = feats[:FRAMES]                               # [30, 99]

        label = 1 if grp["label"].iloc[0] == "correct" else 0
        exercise = grp["exercise"].iloc[0]

        video_tensors.append(feats)
        labels.append(label)
        strat_keys.append(f"{exercise}_{label}")

    return np.array(video_tensors), np.array(labels), strat_keys


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ExerciseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Modelo CNN + LSTM
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMClassifier(nn.Module):
    """
    Conv1D extrae patrones espaciales por frame (actúa sobre los 99 features
    como canales en la dimensión temporal de 30 frames).
    LSTM modela la dinámica temporal de la secuencia.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_FEATURES, CONV_FILTERS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(CONV_FILTERS, CONV_FILTERS, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm    = nn.LSTM(CONV_FILTERS, LSTM_UNITS, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(DROPOUT)
        self.head    = nn.Linear(LSTM_UNITS, 1)

    def forward(self, x):
        # x: [batch, 30, 99]
        x = x.permute(0, 2, 1)          # [batch, 99, 30]  ← canales para Conv1D
        x = self.conv(x)                 # [batch, 64, 30]
        x = x.permute(0, 2, 1)          # [batch, 30, 64]  ← secuencia para LSTM
        _, (h, _) = self.lstm(x)        # h: [1, batch, 64]
        x = self.dropout(h.squeeze(0))  # [batch, 64]
        return self.head(x).squeeze(1)  # [batch] logits


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds   = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / n, acc, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datos ─────────────────────────────────────────────────────────────────
    print("\n[1] Cargando dataset...")
    X, y, strat = load_video_tensors(CSV_PATH)
    print(f"    Videos: {len(X)}  |  Correctos: {y.sum()}  |  Incorrectos: {(y==0).sum()}")

    # ── Split estratificado al nivel de video ──────────────────────────────────
    X_tv, X_test, y_tv, y_test, s_tv, _ = train_test_split(
        X, y, strat, test_size=0.15, random_state=SEED, stratify=strat
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, random_state=SEED, stratify=s_tv
    )  # 0.176 ≈ 15/85 → da ~15% del total
    print(f"    Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # ── Normalización (StandardScaler ajustado solo en train) ─────────────────
    scaler = StandardScaler()
    n_train, T, F = X_train.shape
    X_train_2d = X_train.reshape(-1, F)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(n_train, T, F)
    X_val   = scaler.transform(X_val.reshape(-1, F)).reshape(len(X_val), T, F)
    X_test  = scaler.transform(X_test.reshape(-1, F)).reshape(len(X_test), T, F)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(ExerciseDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ExerciseDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(ExerciseDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model     = CNNLSTMClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2] Modelo: {total_params:,} parámetros entrenables")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    print("\n[3] Entrenando...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), Path(__file__).parent / "best_model.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}")

    # ── Evaluación final ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(Path(__file__).parent / "best_model.pt",
                                      map_location=device))
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*60)
    print("RESULTADOS EN TEST SET")
    print("="*60)
    print(f"Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print("\nClassification Report:")
    report = classification_report(test_labels, test_preds,
                                    target_names=["Incorrecto", "Correcto"])
    print(report)
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix (filas=real, cols=predicho):")
    print("              Pred Incorrecto  Pred Correcto")
    print(f"  Real Incorrecto       {cm[0,0]:>3}            {cm[0,1]:>3}")
    print(f"  Real Correcto         {cm[1,0]:>3}            {cm[1,1]:>3}")

    # ── Guardar resultados para el PDF ────────────────────────────────────────
    results = {
        "train_size":      int(len(X_train)),
        "val_size":        int(len(X_val)),
        "test_size":       int(len(X_test)),
        "total_videos":    int(len(X)),
        "total_params":    total_params,
        "epochs":          EPOCHS,
        "best_val_acc":    float(best_val_acc),
        "test_accuracy":   float(test_acc),
        "confusion_matrix": cm.tolist(),
        "history":         history,
        "hyperparameters": {
            "conv_filters": CONV_FILTERS,
            "lstm_units":   LSTM_UNITS,
            "dropout":      DROPOUT,
            "batch_size":   BATCH_SIZE,
            "learning_rate": LR,
        }
    }
    out_json = Path(__file__).parent / "poc_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados → {out_json}")


if __name__ == "__main__":
    main()
