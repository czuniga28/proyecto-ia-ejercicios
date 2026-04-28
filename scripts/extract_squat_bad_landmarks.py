#!/usr/bin/env python3
"""
extract_squat_bad_landmarks.py
Extrae landmarks de squat_bad directamente desde los videos originales,
sin pasar por videos normalizados intermedios (evita degradación por compresión).

Pipeline:
  1. Escanear TODOS los frames del original con MediaPipe.
  2. Conservar solo índices con detección exitosa.
  3. Calcular señal de movimiento sobre esos frames válidos.
  4. Seleccionar segmento de mayor movimiento.
  5. Muestrear 30 frames del segmento → extraer landmarks.
  6. Augmentar en espacio de landmarks:
       - Flip horizontal: x_i = 1 - x_i  (reflejo especular)
  7. Reemplazar filas squat/incorrect en pose_landmarks_dataset.csv.

Uso:
  python3 scripts/extract_squat_bad_landmarks.py
  python3 scripts/extract_squat_bad_landmarks.py --csv pose_landmarks_dataset.csv
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ─── Configuración ────────────────────────────────────────────────────────────

SRC_FOLDER       = "Dataset_Wrong Exercises/Dataset_Wrong Exercises/Squat_Wrong"
FRAMES_PER_VIDEO = 30
MIN_VALID_FRAMES = 15
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Pares de landmarks que se intercambian al hacer flip horizontal
# (MediaPipe BlazePose: izquierda ↔ derecha)
FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6),           # ojos y orejas
    (7, 8),                             # orejas
    (9, 10),                            # boca
    (11, 12),                           # hombros
    (13, 14), (15, 16),                 # codos y muñecas
    (17, 18), (19, 20), (21, 22),      # manos
    (23, 24),                           # caderas
    (25, 26), (27, 28),                 # rodillas y tobillos
    (29, 30), (31, 32),                 # talones y pies
]

# ─── MediaPipe ────────────────────────────────────────────────────────────────

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MP_OPTIONS = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
    running_mode=VisionRunningMode.IMAGE,
)


# ─── Scan de frames válidos ───────────────────────────────────────────────────

def scan_valid_frames(
    cap: cv2.VideoCapture,
    total: int,
    landmarker: PoseLandmarker,
) -> tuple[list[np.ndarray], list[list]]:
    """
    Retorna (valid_frames, valid_landmarks).
    valid_landmarks[i] es la lista de 33 landmarks del frame i.
    """
    valid_frames:    list[np.ndarray] = []
    valid_landmarks: list[list]       = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        if result.pose_landmarks:
            valid_frames.append(frame)
            valid_landmarks.append(result.pose_landmarks[0])

    return valid_frames, valid_landmarks


# ─── Señal de movimiento ──────────────────────────────────────────────────────

def motion_signal(frames: list[np.ndarray]) -> np.ndarray:
    SAMPLE_W, SAMPLE_H = 160, 120
    x0, x1 = SAMPLE_W // 5, SAMPLE_W - SAMPLE_W // 5
    y0, y1 = SAMPLE_H // 5, SAMPLE_H - SAMPLE_H // 5

    signal = np.zeros(len(frames), dtype=np.float32)
    prev   = None
    for i, f in enumerate(frames):
        small = cv2.resize(f, (SAMPLE_W, SAMPLE_H))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi   = gray[y0:y1, x0:x1]
        if prev is not None:
            signal[i] = np.mean(np.abs(roi - prev))
        prev = roi
    return signal


# ─── Segmento de mayor movimiento ─────────────────────────────────────────────

def find_segment(signal: np.ndarray) -> tuple[int, int]:
    n = len(signal)
    if n <= FRAMES_PER_VIDEO:
        return 0, n

    skip    = max(1, n // 10)
    s_start = skip
    s_end   = n - skip
    if s_end - s_start < FRAMES_PER_VIDEO:
        s_start, s_end = 0, n

    region   = signal[s_start:s_end]
    smoothed = np.convolve(region, np.ones(20) / 20, mode="same")
    thresh   = float(np.percentile(smoothed, 25))

    above    = smoothed >= thresh
    segments: list[tuple[int, int]] = []
    in_seg, seg_s = False, 0
    for i, active in enumerate(above):
        if active and not in_seg:
            seg_s, in_seg = i, True
        elif not active and in_seg:
            segments.append((seg_s, i))
            in_seg = False
    if in_seg:
        segments.append((seg_s, len(above)))

    if not segments:
        return s_start, s_end

    best      = max(segments, key=lambda s: float(np.sum(smoothed[s[0]:s[1]])))
    abs_start = s_start + best[0]
    abs_end   = s_start + best[1]

    if abs_end - abs_start < FRAMES_PER_VIDEO:
        center    = (abs_start + abs_end) // 2
        half      = FRAMES_PER_VIDEO // 2
        abs_start = max(s_start, center - half)
        abs_end   = min(s_end, abs_start + FRAMES_PER_VIDEO)

    return abs_start, abs_end


# ─── Muestreo uniforme ────────────────────────────────────────────────────────

def sample_indices(start: int, end: int, total: int) -> np.ndarray:
    n       = max(end - start, 1)
    indices = np.linspace(start, start + n - 1, FRAMES_PER_VIDEO, dtype=int)
    return np.clip(indices, 0, total - 1)


# ─── Landmarks → fila de DataFrame ───────────────────────────────────────────

def landmarks_to_row(video_name: str, frame_idx: int, lms: list) -> dict:
    row = {
        "video_name":  video_name,
        "exercise":    "squat",
        "label":       "incorrect",
        "frame_index": frame_idx,
    }
    for i, lm in enumerate(lms):
        row[f"x{i}"]   = lm.x
        row[f"y{i}"]   = lm.y
        row[f"z{i}"]   = lm.z
        row[f"vis{i}"] = lm.visibility
    return row


def flip_row(row: dict) -> dict:
    """Reflejo horizontal en espacio de landmarks: x_i = 1 - x_i + intercambio L↔R."""
    flipped = row.copy()
    flipped["video_name"] = row["video_name"].replace(".mp4", "_flip.mp4")

    # Invertir coordenada x de todos los landmarks
    for i in range(33):
        flipped[f"x{i}"] = 1.0 - row[f"x{i}"]

    # Intercambiar pares izquierda ↔ derecha
    for left, right in FLIP_PAIRS:
        for coord in ("x", "y", "z", "vis"):
            flipped[f"{coord}{left}"], flipped[f"{coord}{right}"] = (
                flipped[f"{coord}{right}"],
                flipped[f"{coord}{left}"],
            )
    return flipped


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=".", help="Raíz del proyecto")
    parser.add_argument("--csv", default="data/pose_landmarks_dataset.csv")
    args = parser.parse_args()

    src_dir  = Path(args.src) / SRC_FOLDER
    csv_path = Path(args.csv)

    videos = sorted(
        [f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.name,
    )
    print(f"Videos encontrados: {len(videos)}")

    new_rows: list[dict] = []

    with PoseLandmarker.create_from_options(MP_OPTIONS) as landmarker:
        for vid in videos:
            cap   = cv2.VideoCapture(str(vid))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"\n▶ {vid.name}  ({total} frames totales)")
            valid_frames, valid_lms = scan_valid_frames(cap, total, landmarker)
            cap.release()

            n_valid = len(valid_frames)
            print(f"  Frames válidos: {n_valid}/{total} ({n_valid/total*100:.0f}%)")

            if n_valid < MIN_VALID_FRAMES:
                print(f"  [DESCARTADO] < {MIN_VALID_FRAMES} frames válidos")
                continue

            # Segmento de mayor movimiento sobre frames válidos
            signal            = motion_signal(valid_frames)
            seg_s, seg_e      = find_segment(signal)
            sample_idx        = sample_indices(seg_s, seg_e, n_valid)

            # Extraer landmarks del segmento
            vid_name = vid.name
            for frame_i, idx in enumerate(sample_idx):
                lms = valid_lms[idx]
                new_rows.append(landmarks_to_row(vid_name, frame_i, lms))

            # Aumentación flip en espacio de landmarks
            flip_name = vid.stem + "_flip.mp4"
            for frame_i, idx in enumerate(sample_idx):
                lms = valid_lms[idx]
                row = landmarks_to_row(flip_name, frame_i, lms)
                new_rows.append(flip_row(row))

            print(f"  ✓ {FRAMES_PER_VIDEO} frames originales + {FRAMES_PER_VIDEO} flip extraídos")

    print(f"\nTotal filas nuevas de squat_bad: {len(new_rows)}")

    # Reemplazar filas squat/incorrect en el CSV existente
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[
            ~((df_existing["exercise"] == "squat") & (df_existing["label"] == "incorrect"))
        ]
        print(f"Filas squat/incorrect eliminadas del CSV anterior.")
    else:
        df_existing = pd.DataFrame()

    df_new    = pd.DataFrame(new_rows)
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged.to_csv(csv_path, index=False)

    print(f"\nCSV actualizado: {len(df_merged)} filas totales → '{csv_path}'")
    print(df_merged.groupby(["exercise", "label"]).size().to_string())


if __name__ == "__main__":
    main()
