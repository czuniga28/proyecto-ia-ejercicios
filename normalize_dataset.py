#!/usr/bin/env python3
"""
normalize_dataset.py
Normaliza el dataset de videos para el proyecto CNN+LSTM de biomecánica.

Pasos:
  1. Balance   — exactamente 25 videos por clase (descarta excedentes)
  2. Estándar  — 640x480, 30 FPS, .mp4 (codec mp4v)
  3. Segmento  — detecta el segmento del ejercicio via umbral adaptativo sobre
                  señal de movimiento suavizada; muestrea 30 frames uniformes
                  sobre ese segmento → captura el arco completo del movimiento
                  (inicio → clímax → final), robusto contra intros/cortes de YouTube
  4. Log       — resumen por clase al finalizar

Uso:
  python3 normalize_dataset.py
  python3 normalize_dataset.py --src videos --dst dataset_limpio
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ─── Configuración ────────────────────────────────────────────────────────────

TARGET_W = 640
TARGET_H = 480
TARGET_FPS = 30
FRAMES_PER_VIDEO = 30
VIDEOS_PER_CLASS = 25

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Estructura del dataset (carpetas fuente → carpetas destino)
FOLDERS = [
    "deadlift",
    "deadlift_bad",
    "squat",
    "squat_bad",
    "pull Up",
    "pull_up_bad",
]


# ─── Detección de clímax del movimiento ───────────────────────────────────────

def compute_motion_signal(cap: cv2.VideoCapture, total_frames: int) -> np.ndarray:
    """
    Calcula la energía de movimiento frame a frame a baja resolución.

    Mejoras sobre la versión anterior:
      - Solo se evalúa la región central (60% w × 60% h) para ignorar
        overlays, texto y movimiento de cámara en los bordes.
      - Retorna array de longitud `total_frames`.
    """
    SAMPLE_W, SAMPLE_H = 160, 120

    # Región central: ignora el 20% de cada borde
    x0 = SAMPLE_W // 5          # 32
    x1 = SAMPLE_W - SAMPLE_W // 5   # 128
    y0 = SAMPLE_H // 5          # 24
    y1 = SAMPLE_H - SAMPLE_H // 5   # 96

    motion = np.zeros(total_frames, dtype=np.float32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_roi = None
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (SAMPLE_W, SAMPLE_H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi = gray[y0:y1, x0:x1]

        if prev_roi is not None:
            motion[i] = np.mean(np.abs(roi - prev_roi))

        prev_roi = roi

    return motion


def find_exercise_segment(cap: cv2.VideoCapture, total_frames: int) -> tuple[int, int]:
    """
    Detecta el segmento del video que contiene el ejercicio.

    Estrategia:
      1. Calcula la señal de movimiento (ROI central, ignora primer/último 10%).
      2. Suaviza con media móvil de 20 frames para eliminar picos de cortes de edición.
      3. Umbral al percentil 25 — umbral bajo que incluye el rep completo:
         arranque, clímax y aterrizaje.
      4. Encuentra todos los segmentos contiguos sobre el umbral.
      5. Selecciona el de mayor energía total (el más activo = el ejercicio).
      6. Expande si es más corto que FRAMES_PER_VIDEO.

    Retorna (start, end) para muestreo uniforme posterior.
    """
    if total_frames <= FRAMES_PER_VIDEO:
        return 0, total_frames

    motion = compute_motion_signal(cap, total_frames)

    skip = max(1, total_frames // 10)
    search_start = skip
    search_end = total_frames - skip

    if search_end - search_start < FRAMES_PER_VIDEO:
        search_start, search_end = 0, total_frames

    region = motion[search_start:search_end]

    # Suavizado: elimina picos breves de cortes de YouTube
    SMOOTH_W = 20
    kernel = np.ones(SMOOTH_W) / SMOOTH_W
    smoothed = np.convolve(region, kernel, mode='same')

    # Umbral bajo: incluye arranque y aterrizaje del movimiento
    threshold = float(np.percentile(smoothed, 25))

    # Segmentos contiguos sobre el umbral
    above = smoothed >= threshold
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start_local = 0
    for i, active in enumerate(above):
        if active and not in_seg:
            seg_start_local = i
            in_seg = True
        elif not active and in_seg:
            segments.append((seg_start_local, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start_local, len(above)))

    if not segments:
        return search_start, search_end

    # Segmento con mayor energía total = el ejercicio
    best = max(segments, key=lambda s: float(np.sum(smoothed[s[0]:s[1]])))
    abs_start = search_start + best[0]
    abs_end = search_start + best[1]

    # Garantizar longitud mínima
    if abs_end - abs_start < FRAMES_PER_VIDEO:
        center = (abs_start + abs_end) // 2
        half = FRAMES_PER_VIDEO // 2
        abs_start = max(search_start, center - half)
        abs_end = min(search_end, abs_start + FRAMES_PER_VIDEO)

    return abs_start, abs_end


# ─── Procesamiento de video ───────────────────────────────────────────────────

def extract_frames_uniform(cap: cv2.VideoCapture, start: int, end: int, total: int) -> list[np.ndarray]:
    """
    Muestrea uniformemente FRAMES_PER_VIDEO frames del segmento [start, end).

    El acceso aleatorio por índice garantiza que el arco completo del movimiento
    quede representado independientemente de la duración del segmento detectado:
    un rep de 1 s y uno de 4 s producen los mismos 30 frames distribuidos de
    inicio a fin.
    """
    n = max(end - start, 1)
    indices = np.linspace(start, start + n - 1, FRAMES_PER_VIDEO, dtype=int)
    indices = np.clip(indices, 0, total - 1)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    while len(frames) < FRAMES_PER_VIDEO and frames:
        frames.append(frames[-1].copy())

    return frames[:FRAMES_PER_VIDEO]


def write_video(frames: list[np.ndarray], dst_path: Path) -> int:
    """Escribe la lista de frames como un video .mp4. Retorna frames escritos."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst_path), fourcc, TARGET_FPS, (TARGET_W, TARGET_H))

    if not out.isOpened():
        raise RuntimeError(f"No se pudo crear el archivo de salida: {dst_path}")

    written = 0
    for frame in frames:
        resized = cv2.resize(frame, (TARGET_W, TARGET_H))
        out.write(resized)
        written += 1

    out.release()
    return written


def process_video(src: Path, dst: Path) -> tuple[bool, int | str]:
    """
    Procesa un video: detecta clímax, extrae 30 frames, guarda a 640x480 30fps .mp4.
    Retorna (éxito, frames_escritos | mensaje_error).
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False, f"No se pudo abrir: {src.name}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False, f"Video vacío o sin frames legibles: {src.name}"

    try:
        start, end = find_exercise_segment(cap, total_frames)
        frames = extract_frames_uniform(cap, start, end, total_frames)
    finally:
        cap.release()

    if not frames:
        return False, f"No se pudieron extraer frames: {src.name}"

    written = write_video(frames, dst)
    return True, written


# ─── Lógica principal ─────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("normalize")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    return log


def collect_videos(src_dir: Path) -> list[Path]:
    """Retorna lista de videos ordenados por nombre (case-insensitive)."""
    return sorted(
        [f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def main():
    parser = argparse.ArgumentParser(description="Normaliza dataset de video biomecánico.")
    parser.add_argument("--src", default="videos", help="Carpeta raíz del dataset original")
    parser.add_argument("--dst", default="dataset_limpio", help="Carpeta raíz de salida")
    parser.add_argument(
        "--folders", nargs="+", default=None,
        help="Procesar solo estas carpetas (ej: deadlift_bad squat_bad). Por defecto: todas."
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    active_folders = args.folders if args.folders else FOLDERS

    log = setup_logger(dst_root / "normalization_log.txt")

    log.info("=" * 72)
    log.info("  NORMALIZACIÓN DE DATASET — Proyecto Biomecánica CNN+LSTM")
    log.info(f"  Fecha : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Origen: {src_root.resolve()}")
    log.info(f"  Salida: {dst_root.resolve()}")
    log.info(f"  Config: {TARGET_W}x{TARGET_H} | {TARGET_FPS} FPS | {FRAMES_PER_VIDEO} frames/video | {VIDEOS_PER_CLASS} videos/clase")
    log.info("=" * 72)

    summary = []

    for folder in active_folders:
        src_dir = src_root / folder
        dst_dir = dst_root / folder
        dst_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n▶ Clase: {folder}")

        if not src_dir.exists():
            log.warning(f"  [OMITIDA] Directorio no encontrado: {src_dir}")
            summary.append((folder, 0, f"{TARGET_W}x{TARGET_H}", 0, 0, 0))
            continue

        all_videos = collect_videos(src_dir)
        selected = all_videos[:VIDEOS_PER_CLASS]
        extras = all_videos[VIDEOS_PER_CLASS:]

        log.info(f"  Encontrados: {len(all_videos)} | Seleccionados: {len(selected)} | Descartados: {len(extras)}")
        if extras:
            log.info(f"  Excedentes omitidos: {[v.name for v in extras]}")

        processed = 0
        failed = 0
        total_frames = 0

        safe_name = folder.replace(" ", "_")

        for i, src_file in enumerate(selected, start=1):
            dst_name = f"{safe_name}_{i:02d}.mp4"
            dst_file = dst_dir / dst_name

            ok, result = process_video(src_file, dst_file)

            if ok:
                processed += 1
                total_frames += result
                log.info(f"  ✓ [{i:02d}/{len(selected)}] {src_file.name} → {dst_name}  ({result} frames)")
            else:
                failed += 1
                log.error(f"  ✗ [{i:02d}/{len(selected)}] {result}")

        summary.append((folder, processed, f"{TARGET_W}x{TARGET_H}", total_frames, failed, len(selected)))
        log.info(f"  → {processed}/{len(selected)} OK | {total_frames} frames totales | {failed} fallidos")

    # ── Tabla resumen final ────────────────────────────────────────────────────
    log.info("\n" + "=" * 72)
    log.info("  RESUMEN FINAL")
    log.info("=" * 72)
    log.info(f"  {'Clase':<18} {'Procesados':>10} {'Resolución':>11} {'Frames':>8} {'Fallidos':>9}")
    log.info(f"  {'-'*18} {'-'*10} {'-'*11} {'-'*8} {'-'*9}")

    total_proc = total_fr = total_fail = 0
    for clase, proc, res, fr, fail, _ in summary:
        log.info(f"  {clase:<18} {proc:>10} {res:>11} {fr:>8} {fail:>9}")
        total_proc += proc
        total_fr += fr
        total_fail += fail

    log.info(f"  {'-'*18} {'-'*10} {'-'*11} {'-'*8} {'-'*9}")
    log.info(f"  {'TOTAL':<18} {total_proc:>10} {'640x480':>11} {total_fr:>8} {total_fail:>9}")
    log.info("=" * 72)
    log.info(f"\n  Log guardado en: {dst_root / 'normalization_log.txt'}")


if __name__ == "__main__":
    main()
