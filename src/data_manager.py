#!/usr/bin/env python3
"""
DataManager — extracción de landmarks, normalización estructural,
ángulos articulares y construcción del CSV enriquecido.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Configuración BlazePose ────────────────────────────────────────────────────
_SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
_KEY_LMS     = [11, 12, 23, 24, 25, 26, 27, 28]   # hombros, caderas, rodillas, tobillos
_VIDEO_EXT   = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
_ANGLE_NAMES = [
    'left_knee', 'right_knee',
    'left_hip',  'right_hip',
    'left_elbow', 'right_elbow',
    'left_shoulder', 'right_shoulder',
    'spine',
]

# (folder_name, exercise_label, form_label)
_CLASSES = [
    ('deadlift',     'deadlift', 'correct'),
    ('deadlift_bad', 'deadlift', 'incorrect'),
    ('pull Up',      'pull_up',  'correct'),
    ('pull_up_bad',  'pull_up',  'incorrect'),
    ('squat',        'squat',    'correct'),
    ('squat_bad',    'squat',    'incorrect'),  # pipeline alternativo + aumentación
]


class DataManager:
    """
    Transforma videos crudos en el CSV enriquecido listo para entrenamiento.

    Uso típico:
        dm = DataManager(dataset_dir, model_path, output_csv)
        dm.extract_from_videos()   # corre una sola vez (~minutos)
        df = dm.load_from_csv()    # carga rápida para entrenamiento
    """

    N_LANDMARKS = 33
    FRAMES      = 30
    MIN_VALID   = 15

    def __init__(self, dataset_dir: Path, model_path: Path, output_csv: Path):
        self.dataset_dir = Path(dataset_dir)
        self.model_path  = str(model_path)
        self.output_csv  = Path(output_csv)

    # ──────────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────────

    def extract_from_videos(self) -> None:
        """Procesa todos los videos y guarda el CSV enriquecido en output_csv."""
        BaseOptions           = mp.tasks.BaseOptions
        PoseLandmarker        = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
        )

        all_rows: list[dict] = []
        with PoseLandmarker.create_from_options(options) as landmarker:
            for folder_name, exercise, label in _CLASSES:
                folder = self.dataset_dir / folder_name
                videos = sorted(
                    [f for f in folder.iterdir() if f.suffix.lower() in _VIDEO_EXT],
                    key=lambda p: p.name.lower(),
                )
                use_alt = folder_name == 'squat_bad'
                print(f"\n▶ {folder_name}  ({len(videos)} videos)  label={label}")

                for vid in videos:
                    if use_alt:
                        rows = self._process_squat_bad(vid, exercise, landmarker)
                    else:
                        rows = self._process_standard(vid, exercise, label, landmarker)

                    if rows:
                        aug = f"  +flip ({len(rows) // 2} orig + {len(rows) // 2} flipped)" if use_alt else ""
                        print(f"  ✓ {vid.name:40s}  {len(rows)} frames{aug}")
                        all_rows.extend(rows)
                    else:
                        print(f"  ✗ {vid.name}  (descartado — < {self.MIN_VALID} frames válidos)")

        df = pd.DataFrame(all_rows)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print(f"\nGuardado → {self.output_csv}")
        print(f"Total filas: {len(df)}")
        print(df.groupby(['exercise', 'label']).agg(
            videos=('video_name', 'nunique'), frames=('frame_index', 'count')
        ).to_string())

    def load_from_csv(self) -> pd.DataFrame:
        """Carga el CSV enriquecido. Lanza FileNotFoundError si no existe."""
        if not self.output_csv.exists():
            raise FileNotFoundError(
                f"CSV enriquecido no encontrado: {self.output_csv}\n"
                "Ejecuta extract_from_videos() primero."
            )
        return pd.read_csv(self.output_csv)

    # ──────────────────────────────────────────────────────────────────────────
    # Pipelines de extracción
    # ──────────────────────────────────────────────────────────────────────────

    def _process_standard(
        self, vid_path: Path, exercise: str, label: str, landmarker
    ) -> list[dict]:
        """Muestreo uniforme de FRAMES fotogramas; forward-fill en fallos de detección."""
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, self.FRAMES, dtype=int)
        idx_set = set(indices.tolist())
        frames_data: dict[int, np.ndarray] = {}

        fi = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fi in idx_set:
                lm = self._detect(frame, landmarker)
                if lm is not None:
                    frames_data[fi] = lm
            fi += 1
        cap.release()

        if len(frames_data) < self.MIN_VALID:
            return []

        sequence = self._forward_fill(indices, frames_data)
        return [
            self._build_row(vid_path.name, exercise, label, i, lm)
            for i, lm in enumerate(sequence)
        ]

    def _process_squat_bad(
        self, vid_path: Path, exercise: str, landmarker
    ) -> list[dict]:
        """
        Pipeline alternativo para squat_bad:
        1. Escanea todos los frames del video original con MediaPipe.
        2. Selecciona el segmento de mayor actividad biomecánica.
        3. Devuelve la secuencia original + su reflejo horizontal (aumentación).
        """
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            return []

        valid: list[np.ndarray] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            lm = self._detect(frame, landmarker)
            if lm is not None:
                valid.append(lm)
        cap.release()

        if len(valid) < self.MIN_VALID:
            return []

        segment = self._best_segment(valid, self.FRAMES)  # list de FRAMES arrays [33,3] raw

        rows_orig = [
            self._build_row(vid_path.name, exercise, 'incorrect', i, segment[i])
            for i in range(self.FRAMES)
        ]
        flip_name = vid_path.stem + '_flip' + vid_path.suffix
        rows_flip = [
            self._build_row(flip_name, exercise, 'incorrect', i, self._flip_raw(segment[i]))
            for i in range(self.FRAMES)
        ]
        return rows_orig + rows_flip

    # ──────────────────────────────────────────────────────────────────────────
    # Normalización estructural y ángulos
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(lm: np.ndarray) -> np.ndarray:
        """
        Traslación al centro de cadera + escalado por longitud de torso.
        lm: [33, 3] en espacio de imagen  →  [33, 3] normalizado.
        """
        hip_center = (lm[23] + lm[24]) / 2.0
        lm_t = lm - hip_center

        shoulder_center = (lm_t[11] + lm_t[12]) / 2.0
        torso_len = float(np.linalg.norm(shoulder_center))
        if torso_len < 1e-6:
            torso_len = 1.0

        return lm_t / torso_len

    @staticmethod
    def _compute_angles(lm: np.ndarray) -> np.ndarray:
        """
        Calcula 9 ángulos articulares (radianes) sobre landmarks normalizados.
        lm: [33, 3]  →  [9]
        """
        def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            ba = a - b
            bc = c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return float(np.arccos(np.clip(cos, -1.0, 1.0)))

        def vec_ang(v: np.ndarray, ref: np.ndarray) -> float:
            cos = np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-8)
            return float(np.arccos(np.clip(cos, -1.0, 1.0)))

        # En coords normalizadas la cadera está en el origen; el "arriba" es -y
        vertical        = np.array([0.0, -1.0, 0.0])
        shoulder_center = (lm[11] + lm[12]) / 2.0

        return np.array([
            ang(lm[23], lm[25], lm[27]),        # rodilla izquierda
            ang(lm[24], lm[26], lm[28]),        # rodilla derecha
            ang(lm[11], lm[23], lm[25]),        # cadera izquierda
            ang(lm[12], lm[24], lm[26]),        # cadera derecha
            ang(lm[11], lm[13], lm[15]),        # codo izquierdo
            ang(lm[12], lm[14], lm[16]),        # codo derecho
            ang(lm[23], lm[11], lm[13]),        # hombro izquierdo
            ang(lm[24], lm[12], lm[14]),        # hombro derecho
            vec_ang(shoulder_center, vertical), # inclinación de columna
        ], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades internas
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect(frame: np.ndarray, landmarker) -> np.ndarray | None:
        """Detecta pose en un frame BGR. Devuelve [33, 3] o None."""
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return None
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]],
            dtype=np.float32,
        )

    @classmethod
    def _forward_fill(
        cls, indices: np.ndarray, frames_data: dict[int, np.ndarray]
    ) -> list[np.ndarray]:
        """Rellena hacia adelante los índices sin detección."""
        result: list[np.ndarray] = []
        last: np.ndarray | None  = None
        for idx in indices:
            if idx in frames_data:
                last = frames_data[idx]
            result.append(
                last if last is not None
                else np.zeros((cls.N_LANDMARKS, 3), dtype=np.float32)
            )
        return result

    @staticmethod
    def _best_segment(lms_list: list[np.ndarray], n: int) -> list[np.ndarray]:
        """
        Devuelve la subsecuencia de n frames (de lms_list) con mayor
        actividad biomecánica, medida como variación en landmarks clave.
        """
        if len(lms_list) <= n:
            seq = lms_list[:]
            while len(seq) < n:
                seq.append(seq[-1].copy())
            return seq

        # Señal de movimiento entre frames consecutivos en landmarks clave
        motion = [
            float(np.sum((lms_list[i][_KEY_LMS] - lms_list[i - 1][_KEY_LMS]) ** 2))
            for i in range(1, len(lms_list))
        ]
        # Ventana deslizante: n frames tienen n-1 valores de movimiento
        best_start = max(
            range(len(lms_list) - n + 1),
            key=lambda s: sum(motion[s: s + n - 1]),
        )
        return lms_list[best_start: best_start + n]

    @staticmethod
    def _flip_raw(lm: np.ndarray) -> np.ndarray:
        """
        Reflejo horizontal en espacio de imagen (antes de normalizar):
        x → 1-x, intercambia pares izquierda↔derecha de BlazePose.
        """
        flipped = lm.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        for i, j in _SWAP_PAIRS:
            flipped[i], flipped[j] = flipped[j].copy(), flipped[i].copy()
        return flipped

    def _build_row(
        self, video_name: str, exercise: str, label: str,
        frame_index: int, lm_raw: np.ndarray,
    ) -> dict:
        """Normaliza estructuralmente, calcula ángulos y construye la fila del CSV."""
        lm_norm = self._normalize(lm_raw)
        angles  = self._compute_angles(lm_norm)

        row: dict = {
            'video_name':  video_name,
            'exercise':    exercise,
            'label':       label,
            'frame_index': frame_index,
        }
        for i in range(self.N_LANDMARKS):
            row[f'x{i}'] = float(lm_norm[i, 0])
            row[f'y{i}'] = float(lm_norm[i, 1])
            row[f'z{i}'] = float(lm_norm[i, 2])

        for name, val in zip(_ANGLE_NAMES, angles):
            row[f'angle_{name}'] = float(val)

        return row
