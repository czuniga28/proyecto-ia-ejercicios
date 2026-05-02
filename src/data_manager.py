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
            base_options=BaseOptions(model_asset_path=self.model_path), # Ruta del modelo
            running_mode=VisionRunningMode.IMAGE, # Modo de ejecución por imagen
        )

        all_rows: list[dict] = [] # Lista para almacenar las filas del CSV
        
        # Iniciar modelo de IA (Blazepose) con manejador de contexto
        with PoseLandmarker.create_from_options(options) as landmarker: 
            for folder_name, exercise, label in _CLASSES:
                folder = self.dataset_dir / folder_name
                # Ordenar videos para tener determinismo en el procesamiento
                videos = sorted(
                    # List comprehension: itera sobre todos los archivos en la carpeta y se queda con los que tengan extensión de video permitida
                    [f for f in folder.iterdir() if f.suffix.lower() in _VIDEO_EXT],
                    # Regla de ordenamiento: minusculas (lambda p es una funcion desechable que recibe como parametro p)
                    key=lambda p: p.name.lower(),
                )

                # flag para usar pipeline alternativo para squat_bad
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
        """
        Lee todos los frames del video (ya recortado a ~30 en dataset_limpio)
        y aplica MediaPipe a cada uno; forward-fill en fallos de detección.
        """
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            return []

        raw_frames: list[np.ndarray | None] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(self._detect(frame, landmarker))
        cap.release()

        detected = [lm for lm in raw_frames if lm is not None]
        if len(detected) < self.MIN_VALID:
            return []

        # Forward-fill y recortar/rellenar hasta exactamente FRAMES
        sequence: list[np.ndarray] = []
        last: np.ndarray | None = None
        for lm in raw_frames:
            if lm is not None:
                last = lm # se guarda el ultimo frame detectado
            # si no se detecta un frame, se guarda el ultimo detectado
            sequence.append(last if last is not None
                            else np.zeros((self.N_LANDMARKS, 3), dtype=np.float32))

        # Ajustar a FRAMES: recortar si sobra, repetir último si falta
        if len(sequence) > self.FRAMES:
            sequence = sequence[:self.FRAMES]
        while len(sequence) < self.FRAMES:
            sequence.append(sequence[-1].copy())

        return [
            self._build_row(vid_path.name, exercise, label, frame_index, lm)
            for frame_index, lm in enumerate(sequence)
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

        # Lista de arreglos numpy con keypoints
        valid: list[np.ndarray] = []
        while cap.isOpened():
            # ret = bool indicando si la lectura fue ok o no
            # frame = frame en BGR
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
        # dividir puntos de la cadera entre 2 para encontrar el centro de gravedad de la persona
        hip_center = (lm[23] + lm[24]) / 2.0
        lm_t = lm - hip_center

        # dividir puntos de los hombros entre 2 para encontrar el centro de los hombros
        shoulder_center = (lm_t[11] + lm_t[12]) / 2.0
        # calcular la distancia entre los hombros y la cadera
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
        def ang(punto_extremo_1: np.ndarray, vertice_central: np.ndarray, punto_extremo_2: np.ndarray) -> float:
            # Crear vectores desde la articulación central hacia los extremos (ej. de rodilla a cadera, y de rodilla a tobillo)
            vector_hacia_1 = punto_extremo_1 - vertice_central
            vector_hacia_2 = punto_extremo_2 - vertice_central
            
            # Formula del angulo entre dos vectores
            coseno_del_angulo = np.dot(vector_hacia_1, vector_hacia_2) / (np.linalg.norm(vector_hacia_1) * np.linalg.norm(vector_hacia_2) + 1e-8)

            # Clip asegura que angulo esté entre -1 y 1. Se retorna angulo en radianes
            return float(np.arccos(np.clip(coseno_del_angulo, -1.0, 1.0)))

        # Angulo entre un vector y la vertical
        def vec_ang(vector_postura: np.ndarray, vector_referencia: np.ndarray) -> float:
            # Formula del angulo entre un vector y la vertical
            coseno_del_angulo = np.dot(vector_postura, vector_referencia) / (np.linalg.norm(vector_postura) * np.linalg.norm(vector_referencia) + 1e-8)
            # Clip asegura que angulo esté entre -1 y 1. Se retorna angulo en radianes
            return float(np.arccos(np.clip(coseno_del_angulo, -1.0, 1.0)))

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
        # Convierte el frame de BGR a RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detecta pose
        result = landmarker.detect(
            # Convertir numpy array a mediapipe image
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        )
        
        if not result.pose_landmarks:
            return None
        return np.array(
            # list comprehension para extraer los landmarks y almacenarlas en un array de numpy
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]],
            dtype=np.float32,
        )

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
