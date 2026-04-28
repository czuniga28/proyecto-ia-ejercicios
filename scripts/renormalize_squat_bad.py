#!/usr/bin/env python3
"""
renormalize_squat_bad.py
Re-normaliza los videos de sentadilla incorrecta con un pipeline
MediaPipe-first para garantizar que los 30 frames seleccionados
sean detectables y representen el segmento de mayor movimiento.

Pipeline:
  1. Escanear TODOS los frames del video original con MediaPipe.
  2. Conservar solo los índices donde hubo detección exitosa.
  3. Si el video tiene < MIN_VALID_FRAMES frames válidos, descartarlo.
  4. Calcular la señal de movimiento sobre los frames válidos.
  5. Encontrar el segmento de mayor movimiento dentro de los válidos.
  6. Muestrear 30 frames uniformes de ese segmento.
  7. Aumentar hasta TARGET_PER_CLASS videos:
       - originales → flip → brightness → flip+dark

Uso:
  python3 scripts/renormalize_squat_bad.py
  python3 scripts/renormalize_squat_bad.py --dst dataset_limpio
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ─── Configuración ────────────────────────────────────────────────────────────

TARGET_W, TARGET_H  = 640, 480
TARGET_FPS          = 30
FRAMES_PER_VIDEO    = 30
TARGET_PER_CLASS    = 25
MIN_VALID_FRAMES    = 15        # videos con menos frames válidos se descartan

SRC_FOLDER = "Dataset_Wrong Exercises/Dataset_Wrong Exercises/Squat_Wrong"
DST_FOLDER = "squat_bad"
PREFIX     = "squat_bad"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ─── MediaPipe ────────────────────────────────────────────────────────────────

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MP_OPTIONS = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
    running_mode=VisionRunningMode.IMAGE,
)


def scan_valid_frames(
    cap: cv2.VideoCapture,
    total: int,
    landmarker: PoseLandmarker,
) -> tuple[list[int], list[np.ndarray]]:
    """Devuelve (índices_válidos, frames_válidos) donde MediaPipe detectó pose."""
    valid_indices: list[int] = []
    valid_frames:  list[np.ndarray] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        if result.pose_landmarks:
            valid_indices.append(i)
            valid_frames.append(frame)

    return valid_indices, valid_frames


# ─── Señal de movimiento sobre frames válidos ─────────────────────────────────

def motion_on_valid(frames: list[np.ndarray]) -> np.ndarray:
    """Señal de movimiento calculada entre frames válidos consecutivos."""
    SAMPLE_W, SAMPLE_H = 160, 120
    x0, x1 = SAMPLE_W // 5, SAMPLE_W - SAMPLE_W // 5
    y0, y1 = SAMPLE_H // 5, SAMPLE_H - SAMPLE_H // 5

    n      = len(frames)
    signal = np.zeros(n, dtype=np.float32)
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

    skip         = max(1, n // 10)
    s_start      = skip
    s_end        = n - skip
    if s_end - s_start < FRAMES_PER_VIDEO:
        s_start, s_end = 0, n

    region   = signal[s_start:s_end]
    smoothed = np.convolve(region, np.ones(20) / 20, mode="same")
    thresh   = float(np.percentile(smoothed, 25))

    above    = smoothed >= thresh
    segments: list[tuple[int, int]] = []
    in_seg, seg_start = False, 0
    for i, active in enumerate(above):
        if active and not in_seg:
            seg_start, in_seg = i, True
        elif not active and in_seg:
            segments.append((seg_start, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(above)))

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


# ─── Muestreo uniforme del segmento ──────────────────────────────────────────

def sample_segment(
    frames: list[np.ndarray],
    start: int,
    end: int,
) -> list[np.ndarray]:
    n       = max(end - start, 1)
    indices = np.linspace(start, start + n - 1, FRAMES_PER_VIDEO, dtype=int)
    indices = np.clip(indices, 0, len(frames) - 1)
    sampled = [frames[i] for i in indices]
    while len(sampled) < FRAMES_PER_VIDEO and sampled:
        sampled.append(sampled[-1].copy())
    return sampled[:FRAMES_PER_VIDEO]


# ─── Aumentación ─────────────────────────────────────────────────────────────

def flip_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.flip(f, 1) for f in frames]


def brightness_jitter(
    frames: list[np.ndarray], beta: float
) -> list[np.ndarray]:
    return [
        np.clip(f.astype(np.int16) + int(beta), 0, 255).astype(np.uint8)
        for f in frames
    ]


# ─── Escritura de video ───────────────────────────────────────────────────────

def write_video(frames: list[np.ndarray], path: Path) -> int:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(path), fourcc, TARGET_FPS, (TARGET_W, TARGET_H))
    if not out.isOpened():
        raise RuntimeError(f"No se pudo crear: {path}")
    for f in frames:
        out.write(cv2.resize(f, (TARGET_W, TARGET_H)))
    out.release()
    return len(frames)


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("renorm_squat_bad")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    fh  = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch  = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=".", help="Raíz del proyecto")
    parser.add_argument("--dst", default="dataset_limpio", help="Carpeta raíz de salida")
    args = parser.parse_args()

    src_dir = Path(args.src) / SRC_FOLDER
    dst_dir = Path(args.dst) / DST_FOLDER
    dst_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(dst_dir.parent / "renorm_squat_bad_log.txt")
    log.info("=" * 72)
    log.info("  RE-NORMALIZACIÓN squat_bad — Pipeline MediaPipe-First")
    log.info(f"  Fecha  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Origen : {src_dir}")
    log.info(f"  Destino: {dst_dir}")
    log.info(f"  Config : {TARGET_W}x{TARGET_H} | {TARGET_FPS}fps | "
             f"{FRAMES_PER_VIDEO} frames | min_valid={MIN_VALID_FRAMES}")
    log.info("=" * 72)

    videos = sorted(
        [f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.name,
    )
    log.info(f"\nVideos encontrados: {len(videos)}")

    base_frames: list[list[np.ndarray]] = []
    scan_summary: list[tuple[str, int, int, str]] = []  # name, total, valid, status

    with PoseLandmarker.create_from_options(MP_OPTIONS) as landmarker:
        for vid in videos:
            cap   = cv2.VideoCapture(str(vid))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            log.info(f"\n▶ {vid.name}  ({total} frames totales)")
            valid_idx, valid_frms = scan_valid_frames(cap, total, landmarker)
            cap.release()

            n_valid = len(valid_frms)
            pct     = n_valid / total * 100 if total > 0 else 0
            log.info(f"  Frames detectados: {n_valid}/{total} ({pct:.0f}%)")

            if n_valid < MIN_VALID_FRAMES:
                log.warning(
                    f"  [DESCARTADO] Solo {n_valid} frames válidos "
                    f"(mínimo requerido: {MIN_VALID_FRAMES})"
                )
                scan_summary.append((vid.name, total, n_valid, "descartado"))
                continue

            # Segmento de mayor movimiento sobre frames válidos
            motion        = motion_on_valid(valid_frms)
            seg_s, seg_e  = find_segment(motion)
            sampled       = sample_segment(valid_frms, seg_s, seg_e)

            log.info(f"  Segmento seleccionado: índices válidos [{seg_s}, {seg_e})")
            base_frames.append(sampled)
            scan_summary.append((vid.name, total, n_valid, "ok"))

    log.info(f"\n{'─'*72}")
    log.info(f"Videos utilizables: {len(base_frames)} / {len(videos)}")

    if not base_frames:
        log.error("Sin videos válidos. Abortando.")
        return

    # ── Aumentación ──────────────────────────────────────────────────────────
    augmented: list[tuple[list[np.ndarray], str]] = []

    for frms in base_frames:
        augmented.append((frms, "orig"))

    for frms in base_frames:
        if len(augmented) >= TARGET_PER_CLASS:
            break
        augmented.append((flip_frames(frms), "flip"))

    for frms in base_frames:
        if len(augmented) >= TARGET_PER_CLASS:
            break
        augmented.append((brightness_jitter(frms, beta=30), "bright"))

    for frms in base_frames:
        if len(augmented) >= TARGET_PER_CLASS:
            break
        # beta=-10 (antes -20) para evitar oscurecer tanto que MediaPipe falle
        augmented.append((brightness_jitter(flip_frames(frms), beta=-10), "flip_dark"))

    augmented = augmented[:TARGET_PER_CLASS]

    # ── Eliminar archivos anteriores ─────────────────────────────────────────
    for old in dst_dir.glob(f"{PREFIX}_*.mp4"):
        old.unlink()
    log.info(f"Archivos anteriores eliminados de {dst_dir}")

    # ── Escribir videos ───────────────────────────────────────────────────────
    written = 0
    for i, (frms, tag) in enumerate(augmented, start=1):
        out_path = dst_dir / f"{PREFIX}_{i:02d}.mp4"
        try:
            write_video(frms, out_path)
            written += 1
            log.info(f"  ✓ [{i:02d}/{TARGET_PER_CLASS}] {out_path.name}  [{tag}]")
        except Exception as e:
            log.error(f"  ✗ [{i:02d}] Error: {e}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 72)
    log.info("  RESUMEN DE ESCANEO")
    log.info("=" * 72)
    log.info(f"  {'Video':<15} {'Total':>7} {'Válidos':>8} {'Estado':>12}")
    log.info(f"  {'-'*15} {'-'*7} {'-'*8} {'-'*12}")
    for name, total, valid, status in scan_summary:
        log.info(f"  {name:<15} {total:>7} {valid:>8} {status:>12}")
    log.info("=" * 72)
    log.info(f"\n  Videos escritos : {written}/{TARGET_PER_CLASS}")
    log.info(f"  Videos base     : {len(base_frames)}")
    log.info(f"  Videos augmentados: {written - len(base_frames)}")
    log.info(f"  Log guardado en : {dst_dir.parent / 'renorm_squat_bad_log.txt'}")


if __name__ == "__main__":
    main()
