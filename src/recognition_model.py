#!/usr/bin/env python3
"""
RecognitionModel — arquitectura CNN+LSTM, entrenamiento y evaluación.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader as TorchDataLoader

class CNNLSTMClassifier(nn.Module):
    """                                                   
      Entrada: [batch, 30, 108]
      Salida:  [batch, 1]  — probabilidad de ejecución correcta
    """

    def __init__(self,
      n_features: int = 108, # Número de características (dimensiones de landmarks formadas por X, Y, Z, Visibilidad)
      conv_filters: int = 64, # Numero de filtros (64 patrones de segmentos de landmarks a detectar; eje: 11-12 [hombros], 13-14 [cadera izq], 23-24 [rodilla izq], 25-26 [tobillo izq], etc) 
      lstm_units: int = 128, # Número de neuronas en la capa oculta del LSTM
      lstm_layers: int = 2, # Número de capas LSTM
      dropout: float = 0.3): # Dropout del 30%para evitar overfitting
        super().__init__()
        self.dropout_p = dropout

        # ──────── CONFIGURACION PARA PRIMER BLOQUE DE CAPAS CONVOLUCIONALES (CNN) ─────────────
        # Genera 64 filtros [108,3]. 64 canales de salida x 108 canales de entrada x 3 frames consecutivos x 0.01 para que los valores sean pequeños y mantener estabilidad
        self.conv1_w = nn.Parameter(torch.randn(conv_filters, n_features, 3) * 0.01)
        self.conv1_b = nn.Parameter(torch.zeros(conv_filters)) # Iniciar bias en 0

        # Normaliza la salida de la capa convolucional antes de pasarla a la siguiente. La normalizacion se hace canal por canal de forma independiente, no de forma global
        self.bn1     = nn.BatchNorm1d(conv_filters) # Normalizacion por canales de salida

        # ──────── CONFIGURACION PARA SEGUNDO BLOQUE DE CAPAS CONVOLUCIONALES (CNN) ─────────────
        # Segunda capa convolucional, se duplica el numero de filtros a 128 para capturar patrones mas complejos y las entradas son los canales de salida de la capa anterior
        self.conv2_w = nn.Parameter(torch.randn(conv_filters * 2, conv_filters, 3) * 0.01)
        self.conv2_b = nn.Parameter(torch.zeros(conv_filters * 2)) # bias con 128 valores (1 por filtro)
        self.bn2     = nn.BatchNorm1d(conv_filters * 2) # Normalizacion por canales de salida (128)


        # ──────── CONFIGURACION PARA LA CAPA LSTM ─────────────
        self.lstm = nn.LSTM(
            input_size  = conv_filters * 2,
            hidden_size = lstm_units, # Tamaño del vector que se actualiza en cada frame y resume la informacion vista
            num_layers  = lstm_layers,
            batch_first = True, # LSTM procesa por defecto tensores con secuencia como primera dimension, en nuestro caso (batch, sequence, features) = batch_first tiene que ser TRUE
            dropout     = dropout,
        )

        # Capa fully connected (clasificacion final)
        # 1 salida para clasificar
        self.fc_w = nn.Parameter(torch.randn(1, lstm_units) * 0.01)
        # sesgo
        self.fc_b = nn.Parameter(torch.zeros(1))

    def _conv1d_manual(
        self,
        tensor_in: torch.Tensor,  # [batch_size, channels_in, time_steps]
        weight:    nn.Parameter,  # [channels_out, channels_in, kernel_size]
        bias:      nn.Parameter,  # [channels_out]
    ) -> torch.Tensor:            # [batch_size, channels_out, time_steps]

        batch_size, channels_in, time_steps = tensor_in.shape
        channels_out, _, kernel_size        = weight.shape

        # Padding con ceros: 1 frame antes y 1 después para mantener time_steps fijo
        zero_pad = torch.zeros(batch_size, channels_in, kernel_size // 2, device=tensor_in.device)
        padded   = torch.cat([zero_pad, tensor_in, zero_pad], dim=2)  # [B, C_in, T+2]

        # Extraer todas las ventanas deslizantes a la vez
        # padded: [B, C_in, T+2] → sliding_windows: [B, C_in, T, kernel_size]
        sliding_windows = padded.unfold(2, kernel_size, 1)

        # Reorganizar para multiplicación de matrices: [B, T, C_in * kernel_size]
        sliding_windows = sliding_windows.permute(0, 2, 1, 3).reshape(
            batch_size, time_steps, channels_in * kernel_size
        )

        # Aplanar cada filtro: [channels_out, C_in * kernel_size]
        filters_flat = weight.reshape(channels_out, channels_in * kernel_size)

        # Producto punto entre ventanas y filtros: [B, T, C_out]
        conv_output = sliding_windows @ filters_flat.T + bias

        return conv_output.permute(0, 2, 1)  # [B, C_out, T]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [batch_size, time_steps, features] = [B, 30, 108]

        features_over_time = frames.permute(0, 2, 1)  # [B, 108, 30]

        # Bloque CNN 1: detecta patrones simples en segmentos de 3 frames
        cnn1_out = self._conv1d_manual(features_over_time, self.conv1_w, self.conv1_b)  # [B, 64, 30]
        cnn1_out = self.bn1(cnn1_out) # Aplicar normalizacion cada uno de los 64 canales de salida
        
        # ReLu para introducir no linealidad (max(0,x), pone en 0 todos los valores negativos y los positivos los deja igual)
        # Solo detecciones positivas pasan a la siguiente capa
        cnn1_out = torch.relu(cnn1_out)
        # Dropout: apaga neuronas aleatoriamente para evitar overfitting
        cnn1_out = F.dropout(cnn1_out, p=self.dropout_p, training=self.training)

        # Bloque CNN 2: detecta patrones sobre los patrones del bloque anterior
        cnn2_out = self._conv1d_manual(cnn1_out, self.conv2_w, self.conv2_b)  # [B, 128, 30]
        cnn2_out = self.bn2(cnn2_out) # Aplicar normalizacion cada uno de los 128 canales de salida
        cnn2_out = torch.relu(cnn2_out)
        cnn2_out = F.dropout(cnn2_out, p=self.dropout_p, training=self.training)

        # LSTM: modela la dinámica temporal sobre los 30 frames
        sequence         = cnn2_out.permute(0, 2, 1)   # [B, 30, 128]
        
        # Guarda el output de LSTM para los 30 frames
        # _ para ignorar y descartar los estados internos que no vamos a usar directamente.
        lstm_out, _      = self.lstm(sequence)         # [B, 30, 128]

        # Obtiene el ultimo estado del LSTM (el que resume la secuencia completa)
        last_frame_state = lstm_out[:, -1, :]          # [B, 128]

        # Aplicar dropout al estado final del LSTM para evitar overfitting
        # Evita que el modelo aprenda una "fórmula mágica" basada en una sola característica específica del último frame.
        last_frame_state = F.dropout(last_frame_state, p=self.dropout_p, training=self.training)

        # Multiplicar el vector de resumen de 128 datos por una matriz de pesos (fc_w) y sumar el sesgo (fc_b).
        logit       = last_frame_state @ self.fc_w.T + self.fc_b  # [B, 1]
        
        # Función sigmoide: convierte cualquier número real en una probabilidad (entre 0 y 1)
        probability = torch.sigmoid(logit)                         # [B, 1] ∈ (0, 1)

        # Devuelve la probabilidad de que la secuencia de frames sea correcta
        return probability


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal: envuelve CNNLSTMClassifier con entrenamiento y evaluación
# ──────────────────────────────────────────────────────────────────────────────

class RecognitionModel:
    """
    Envuelve CNNLSTMClassifier con entrenamiento, evaluación y persistencia.

    Uso típico:
        rm = RecognitionModel()
        rm.train_model(train_loader, val_loader, epochs=30)
        rm.evaluate(test_loader)
    """

    def __init__(
        self,
        conv_filters: int   = 64,
        lstm_units:   int   = 128,
        lstm_layers:  int   = 2,
        dropout:      float = 0.3,
        lr:           float = 1e-3,
    ):
        # Usar GPU si está disponible, si no CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instanciar el modelo y mover todos sus parámetros al dispositivo
        self.model = CNNLSTMClassifier(
            conv_filters = conv_filters,
            lstm_units   = lstm_units,
            lstm_layers  = lstm_layers,
            dropout      = dropout,
        ).to(self.device)

        # BCELoss: pérdida estándar para clasificación binaria con salida sigmoide
        self.criterion = nn.BCELoss()

        # Adam: optimizador adaptativo — ajusta el lr por parámetro durante el entrenamiento
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print(f"Modelo en        : {self.device}")
        print(f"Parámetros totales: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_model(
        self,
        train_loader: TorchDataLoader,
        val_loader:   TorchDataLoader,
        epochs:       int = 30,
        patience:     int = 7,
    ) -> dict:
        """
        Entrena el modelo con early stopping.
        Devuelve el historial de pérdidas por epoch: {'train_loss': [...], 'val_loss': [...]}.
        """
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss          = float('inf')
        epochs_without_improve = 0

        for epoch in range(1, epochs + 1):

            # ── Fase de entrenamiento ──────────────────────────────
            # model.train() activa dropout y batchnorm en modo entrenamiento
            self.model.train()
            running_train_loss = 0.0

            for batch_frames, batch_labels in train_loader:
                # Mover datos al mismo dispositivo que el modelo
                batch_frames = batch_frames.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)  # [B] → [B, 1]

                # 1. Limpiar gradientes del paso anterior
                self.optimizer.zero_grad()

                # 2. Paso hacia adelante: calcular predicciones
                predictions = self.model(batch_frames)           # [B, 1]

                # 3. Calcular pérdida entre predicciones y etiquetas reales
                loss = self.criterion(predictions, batch_labels)

                # 4. Paso hacia atrás: calcular gradientes
                loss.backward()

                # 5. Actualizar pesos en dirección que reduce la pérdida
                self.optimizer.step()

                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)

            # ── Fase de validación ─────────────────────────────────
            # model.eval() desactiva dropout y fija batchnorm — no se aprende aquí
            self.model.eval()
            running_val_loss = 0.0

            # torch.no_grad() evita calcular gradientes — ahorra memoria y tiempo
            with torch.no_grad():
                for batch_frames, batch_labels in val_loader:
                    batch_frames = batch_frames.to(self.device)
                    batch_labels = batch_labels.to(self.device).unsqueeze(1)
                    predictions  = self.model(batch_frames)
                    loss         = self.criterion(predictions, batch_labels)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

            # ── Early stopping ─────────────────────────────────────
            # Si la pérdida de validación mejoró, guardar los mejores pesos
            if avg_val_loss < best_val_loss:
                best_val_loss          = avg_val_loss
                epochs_without_improve = 0
                torch.save(self.model.state_dict(), '_best_weights.pt')
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"\nEarly stopping en epoch {epoch} "
                          f"(sin mejora por {patience} epochs consecutivos)")
                    break

        # Restaurar los mejores pesos encontrados durante el entrenamiento
        self.model.load_state_dict(torch.load('_best_weights.pt', weights_only=True))
        return history

    def evaluate(self, test_loader: TorchDataLoader) -> dict:
        """
        Evalúa el modelo sobre el test set.
        Devuelve métricas: accuracy, f1, precision, recall y confusion matrix.
        """
        self.model.eval()

        all_labels      = []
        all_predictions = []

        # Recorrer todos los batches del test set sin calcular gradientes
        with torch.no_grad():
            for batch_frames, batch_labels in test_loader:
                batch_frames = batch_frames.to(self.device)

                # Probabilidades de salida: [B, 1]
                probabilities = self.model(batch_frames)

                # Convertir probabilidades a clases binarias con umbral 0.5
                # squeeze(1): [B, 1] → [B] para comparar con batch_labels
                predicted_classes = (probabilities.squeeze(1) >= 0.5).long().cpu().numpy()
                true_labels       = batch_labels.long().numpy()

                all_predictions.extend(predicted_classes)
                all_labels.extend(true_labels)

        # Calcular métricas con sklearn
        metrics = {
            'accuracy':         accuracy_score(all_labels, all_predictions),
            'f1':               f1_score(all_labels, all_predictions, zero_division=0),
            'precision':        precision_score(all_labels, all_predictions, zero_division=0),
            'recall':           recall_score(all_labels, all_predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
        }

        print("\n── Resultados en test set ──────────────────────────")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  Confusion matrix:")
        print(f"              Pred Incorrecto  Pred Correcto")
        cm = metrics['confusion_matrix']
        print(f"  Real Incorrecto   {cm[0][0]:^14}  {cm[0][1]:^13}")
        print(f"  Real Correcto     {cm[1][0]:^14}  {cm[1][1]:^13}")

        return metrics

    def save(self, path: Path) -> None:
        """Guarda los pesos del modelo en disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Modelo guardado → {path}")

    def load(self, path: Path) -> None:
        """Carga pesos previamente guardados. El modelo debe tener la misma arquitectura."""
        self.model.load_state_dict(
            torch.load(Path(path), map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"Modelo cargado  ← {path}")