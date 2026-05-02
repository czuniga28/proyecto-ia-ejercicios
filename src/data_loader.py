#!/usr/bin/env python3
"""
DataLoader — construcción de tensores por video, split estratificado
y DataLoaders de PyTorch con muestreo balanceado en entrenamiento.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, WeightedRandomSampler

_COORD_COLS  = [f'{c}{i}' for i in range(33) for c in ('x', 'y', 'z')]
_ANGLE_NAMES = [
    'left_knee', 'right_knee', 'left_hip', 'right_hip',
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 'spine',
]
_ANGLE_COLS   = [f'angle_{n}' for n in _ANGLE_NAMES]
_FEATURE_COLS = _COORD_COLS + _ANGLE_COLS   # 108 features por frame


class _ExerciseDataset(Dataset):
    """Dataset de PyTorch que envuelve tensores de video."""

    # X: np.ndarray que representa las características de los frames (p. ej., coordenadas y ángulos) de los videos.
    # y: np.ndarray que representa las etiquetas correspondientes (0 o 1, indicando si el ejercicio se hizo correctamente o no).
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Conversión de los datos de entrada X y y a tensores de PyTorch. 
        self.X = torch.tensor(X, dtype=torch.float32)   # [N, 30, 108]
        self.y = torch.tensor(y, dtype=torch.float32)   # [N]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class DataLoader:
    """
    Construye tensores por video, aplica split estratificado y devuelve
    DataLoaders de PyTorch con muestreo balanceado en entrenamiento.

    Uso típico:
        dl = DataLoader(csv_path)
        train_loader, val_loader, test_loader, scaler = dl.build_loaders()
    """

    FRAMES     = 30
    N_FEATURES = len(_FEATURE_COLS)   # 108

    def __init__(self, csv_path: Path, seed: int = 42):
        self.csv_path = Path(csv_path)
        self.seed     = seed

    def build_loaders(
        self,
        batch_size: int   = 16,
        test_size:  float = 0.15,
        val_size:   float = 0.15,
    ) -> tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, StandardScaler]:
        """
        Lee el CSV enriquecido y devuelve (train_loader, val_loader, test_loader, scaler).

        - Split a nivel de video, estratificado por (exercise, label).
        - Los videos flip (aumentación) solo van a train.
        - StandardScaler ajustado exclusivamente sobre train.
        - train_loader usa WeightedRandomSampler para balancear clases.
        """
        df     = pd.read_csv(self.csv_path)
        videos = self._build_video_tensors(df)

        # Separar originales y flips; los flips solo van a train
        originals = {k: v for k, v in videos.items() if not v['is_flip']}
        flips     = {k: v for k, v in videos.items() if v['is_flip']}

        # Split estratificado sobre originales (a nivel de video)
        names  = list(originals.keys())
        strata = [f"{originals[n]['exercise']}_{originals[n]['y']}" for n in names]

        tv_names, test_names = train_test_split(
            names, test_size=test_size, random_state=self.seed, stratify=strata,
        )
        strata_tv = [f"{originals[n]['exercise']}_{originals[n]['y']}" for n in tv_names]
        val_ratio = val_size / (1.0 - test_size)
        train_names, val_names = train_test_split(
            tv_names, test_size=val_ratio, random_state=self.seed, stratify=strata_tv,
        )

        # Agregar flips cuyo original cayó en train
        train_set  = set(train_names)
        train_all  = list(train_names)
        for flip_name in flips:
            if self._base_name(flip_name) in train_set:
                train_all.append(flip_name)

        # Construir arrays numpy
        X_train, y_train = self._collect(train_all, videos)
        X_val,   y_val   = self._collect(val_names,  originals)
        X_test,  y_test  = self._collect(test_names, originals)

        print(f"Split  →  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
        self._print_distribution(y_train, y_val, y_test)

        # StandardScaler ajustado solo en train
        scaler     = StandardScaler()
        N_tr, T, F = X_train.shape
        X_train    = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_tr, T, F)
        X_val      = scaler.transform(X_val.reshape(-1, F)).reshape(len(X_val),  T, F)
        X_test     = scaler.transform(X_test.reshape(-1, F)).reshape(len(X_test), T, F)

        train_loader = TorchDataLoader(
            _ExerciseDataset(X_train, y_train),
            batch_size=batch_size,
            sampler=self._balanced_sampler(y_train),
        )
        val_loader  = TorchDataLoader(
            _ExerciseDataset(X_val,  y_val),  batch_size=batch_size,
        )
        test_loader = TorchDataLoader(
            _ExerciseDataset(X_test, y_test), batch_size=batch_size,
        )

        return train_loader, val_loader, test_loader, scaler

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades internas
    # ──────────────────────────────────────────────────────────────────────────

    def _build_video_tensors(self, df: pd.DataFrame) -> dict[str, dict]:
        """Agrupa el DataFrame por video y construye un tensor [30, 108] por video."""
        result: dict[str, dict] = {}
        for vid_name, grp in df.groupby('video_name', sort=False):
            # Frames ordenados por frame_index
            grp   = grp.sort_values('frame_index')
            # Matrix 30x108 de datos numericos 
            X_vid = grp[_FEATURE_COLS].values.astype(np.float32)

            # Si hay mas de 30 frames, se toman los primeros 30
            if len(X_vid) > self.FRAMES:
                X_vid = X_vid[:self.FRAMES]
            # Si hay menos de 30 frames, se rellenan con el ultimo frame
            while len(X_vid) < self.FRAMES:
                X_vid = np.vstack([X_vid, X_vid[-1:]])

            result[vid_name] = {
                'X':        X_vid,
                'y':        1 if grp['label'].iloc[0] == 'correct' else 0,
                'exercise': grp['exercise'].iloc[0],
                'is_flip':  Path(str(vid_name)).stem.endswith('_flip'),
            }
        return result

    @staticmethod
    def _collect(
        names: list[str], source: dict[str, dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apila tensores y etiquetas de los videos indicados."""
        X = np.stack([source[n]['X'] for n in names])
        y = np.array([source[n]['y'] for n in names], dtype=np.float32)
        return X, y

    @staticmethod
    def _base_name(flip_name: str) -> str:
        """squat_bad_01_flip.mp4  →  squat_bad_01.mp4"""
        p = Path(flip_name)
        return p.stem.removesuffix('_flip') + p.suffix

    @staticmethod
    def _balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
        """Pesos inversos a la frecuencia de clase para balancear el muestreo."""
        n_pos   = int(y.sum())
        n_neg   = len(y) - n_pos
        weights = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg)
        return WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float64),
            num_samples=len(weights),
            replacement=True,
        )

    @staticmethod
    def _print_distribution(y_train, y_val, y_test) -> None:
        for split, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
            n_pos = int(y.sum())
            print(f"  {split:5s}  correct={n_pos}  incorrect={len(y)-n_pos}")
