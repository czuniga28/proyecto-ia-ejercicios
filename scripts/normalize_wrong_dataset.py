#!/usr/bin/env python3
"""
normalize_wrong_dataset.py
Normaliza y aumenta los videos de forma incorrecta.

Pipeline:
  1. Misma normalización que normalize_dataset.py
     (detección de segmento, 30 frames, 640x480, 30fps .mp4)
  2. Aumentación para llegar a 25 videos por clase (desde 10):
       - 10 originales
       - 10 espejo horizontal (flip)
       - 5 con jitter de brillo (primeros 5 originales)

Uso:
  python3 scripts/normalize_wrong_dataset.py
  python3 scripts/normalize_wrong_dataset.py --dst dataset_limpio
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ─── Configuración ────────────────────────────────────────────────────────────

TARGET_W, TARGET_H = 640, 480
TARGET_FPS = 30
FRAMES_PER_VIDEO = 30
TARGET_PER_CLASS = 25

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

WRONG_FOLDERS = [
    {
        "src": "Dataset_Wrong Exercises/Dataset_Wrong Exercises/DeadLift_Wrong",
        "dst": "deadlift_bad",
        "prefix": "deadlift_bad",
    },
    {
        "src": "Dataset_Wrong Exercises/Dataset_Wrong Exercises/PullUp_Wrong",
        "dst": "pull_up_bad",
        "prefix": "pull_up_bad",
    },
    {
        "src": "Dataset_Wrong Exercises/Dataset_Wrong Exercises/Squat_Wrong",
        "dst": "squat_bad",
        "prefix": "squat_bad",
    },
]


# ─── Detección de segmento (igual que normalize_dataset.py) ──────────────────

def compute_motion_signal(cap: cv2.VideoCapture, total_frames: int) -> np.ndarray:
    SAMPLE_W, SAMPLE_H = 160, 120
    x0, x1 = SAMPLE_W // 5, SAMPLE_W - SAMPLE_W // 5
    y0, y1 = SAMPLE_H // 5, SAMPLE_H - SAMPLE_H // 5

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
    if total_frames <= FRAMES_PER_VIDEO:
        return 0, total_frames

    motion = compute_motion_signal(cap, total_frames)

    skip = max(1, total_frames // 10)
    search_start = skip
    search_end = total_frames - skip
    if search_end - search_start < FRAMES_PER_VIDEO:
        search_start, search_end = 0, total_frames

    region = motion[search_start:search_end]
    smoothed = np.convolve(region, np.ones(20) / 20, mode="same")
    threshold = float(np.percentile(smoothed, 25))

    above = smoothed >= threshold
    segments: list[tuple[int, int]] = []
    in_seg, seg_start_local = False, 0
    for i, active in enumerate(above):
        if active and not in_seg:
            seg_start_local, in_seg = i, True
        elif not active and in_seg:
            segments.append((seg_start_local, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start_local, len(above)))

    if not segments:
        return search_start, search_end

    best = max(segments, key=lambda s: float(np.sum(smoothed[s[0]:s[1]])))
    abs_start = search_start + best[0]
    abs_end = search_start + best[1]

    if abs_end - abs_start < FRAMES_PER_VIDEO:
        center = (abs_start + abs_end) // 2
        half = FRAMES_PER_VIDEO // 2
        abs_start = max(search_start, center - half)
        abs_end = min(search_end, abs_start + FRAMES_PER_VIDEO)

    return abs_start, abs_end


def extract_frames_uniform(cap: cv2.VideoCapture, start: int, end: int, total: int) -> list[np.ndarray]:
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


# ─── Aumentación ─────────────────────────────────────────────────────────────

def flip_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.flip(f, 1) for f in frames]


def brightness_jitter(frames: list[np.ndarray], beta: float = 30.0) -> list[np.ndarray]:
    """Incrementa el brillo en beta (rango 0-255, clamp a uint8)."""
    result = []
    for f in frames:
        bright = np.clip(f.astype(np.int16) + int(beta), 0, 255).astype(np.uint8)
        result.append(bright)
    return result


# ─── Escritura de video ───────────────────────────────────────────────────────

def write_video(frames: list[np.ndarray], dst_path: Path) -> int:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst_path), fourcc, TARGET_FPS, (TARGET_W, TARGET_H))
    if not out.isOpened():
        raise RuntimeError(f"No se pudo crear: {dst_path}")
    written = 0
    for frame in frames:
        out.write(cv2.resize(frame, (TARGET_W, TARGET_H)))
        written += 1
    out.release()
    return written


def load_frames(src: Path) -> list[np.ndarray] | None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    try:
        start, end = find_exercise_segment(cap, total)
        frames = extract_frames_uniform(cap, start, end, total)
    finally:
        cap.release()
    return frames if frames else None


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("normalize_wrong")
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
    return sorted(
        [f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=".", help="Raíz del proyecto (donde está 'Dataset_Wrong Exercises/')")
    parser.add_argument("--dst", default="dataset_limpio", help="Carpeta raíz de salida")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    log = setup_logger(dst_root / "normalization_wrong_log.txt")
    log.info("=" * 72)
    log.info("  NORMALIZACIÓN + AUMENTACIÓN — Videos Forma Incorrecta")
    log.info(f"  Fecha  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Config : {TARGET_W}x{TARGET_H} | {TARGET_FPS} FPS | {FRAMES_PER_VIDEO} frames | target {TARGET_PER_CLASS}/clase")
    log.info("=" * 72)

    summary = []

    for entry in WRONG_FOLDERS:
        src_dir = src_root / entry["src"]
        dst_dir = dst_root / entry["dst"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        prefix = entry["prefix"]

        log.info(f"\n▶ Clase: {entry['dst']}")
        log.info(f"  Origen: {src_dir}")

        if not src_dir.exists():
            log.warning(f"  [OMITIDA] No encontrada: {src_dir}")
            summary.append((entry["dst"], 0, 0))
            continue

        all_videos = collect_videos(src_dir)
        log.info(f"  Videos encontrados: {len(all_videos)}")

        # ── Cargar todos los frames base ──────────────────────────────────────
        base_frames: list[list[np.ndarray]] = []
        for vid in all_videos:
            frames = load_frames(vid)
            if frames:
                base_frames.append(frames)
                log.info(f"  ✓ Cargado: {vid.name}")
            else:
                log.warning(f"  ✗ No se pudo cargar: {vid.name}")

        n_base = len(base_frames)
        if n_base == 0:
            log.error("  Sin videos válidos. Saltando.")
            summary.append((entry["dst"], 0, 0))
            continue

        # ── Construir lista de aumentaciones hasta TARGET_PER_CLASS ──────────
        # Orden: originales → flips → brightness de los primeros k
        augmented: list[tuple[list[np.ndarray], str]] = []

        # Pasada 1: originales
        for frames in base_frames:
            augmented.append((frames, "orig"))

        # Pasada 2: flip horizontal
        for frames in base_frames:
            if len(augmented) >= TARGET_PER_CLASS:
                break
            augmented.append((flip_frames(frames), "flip"))

        # Pasada 3: brightness jitter (beta=30) sobre los primeros k originales
        for frames in base_frames:
            if len(augmented) >= TARGET_PER_CLASS:
                break
            augmented.append((brightness_jitter(frames, beta=30), "bright"))

        # Pasada 4: flip + brightness (por si aún faltan)
        for frames in base_frames:
            if len(augmented) >= TARGET_PER_CLASS:
                break
            augmented.append((brightness_jitter(flip_frames(frames), beta=-20), "flip_dark"))

        augmented = augmented[:TARGET_PER_CLASS]

        # ── Escribir videos ───────────────────────────────────────────────────
        written_count = 0
        for i, (frames, aug_tag) in enumerate(augmented, start=1):
            dst_file = dst_dir / f"{prefix}_{i:02d}.mp4"
            try:
                n = write_video(frames, dst_file)
                written_count += 1
                log.info(f"  ✓ [{i:02d}/{TARGET_PER_CLASS}] {dst_file.name}  [{aug_tag}]  ({n} frames)")
            except Exception as e:
                log.error(f"  ✗ [{i:02d}] Error escribiendo {dst_file.name}: {e}")

        summary.append((entry["dst"], n_base, written_count))
        log.info(f"  → {written_count}/{TARGET_PER_CLASS} escritos | {n_base} originales | {written_count - n_base} aumentados")

    log.info("\n" + "=" * 72)
    log.info("  RESUMEN FINAL")
    log.info("=" * 72)
    log.info(f"  {'Clase':<18} {'Originales':>12} {'Total escritos':>15}")
    log.info(f"  {'-'*18} {'-'*12} {'-'*15}")
    for clase, orig, total in summary:
        log.info(f"  {clase:<18} {orig:>12} {total:>15}")
    log.info("=" * 72)
    log.info(f"\n  Log guardado en: {dst_root / 'normalization_wrong_log.txt'}")


if __name__ == "__main__":
    main()
