# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University AI project (Ciclo IV — Inteligencia Artificial).
**Team:** Christopher Zúñiga (C28730) & Adrian Arrieta Orozco (B70734)

**Problem:** Computational recognition of correct exercise execution — classify biomechanical quality (correct vs incorrect) for 3 exercises: squat, pull-up, deadlift.

**Dataset:** Workout/Exercises Video Dataset (Kaggle — hasyimabdillah/workoutfitness-video). ~652 videos across 25 exercise categories; 3 selected for this project.

**Architecture:** Hybrid CNN + LSTM
- CNN extracts spatial features (posture patterns) per frame via 1D conv layers
- LSTM models temporal dynamics across 30 frames per video
- Final dense layer outputs correct/incorrect probability

**Input:** 30-frame sequence per video → MediaPipe BlazePose → 33 landmarks (x,y,z) + biomechanical joint angles
**Output:** Binary classification — correct or incorrect execution

## Assignment Instructions (75% grade)

The task requires the following steps — see progress tracking below:

1. Select a Kaggle dataset and define a simple problem
2. Preliminary design (description, diagram/drawing)
3. Select applicable columns/variables
4. Initial exploration: data exploration, metadata/column descriptions, missing data, histograms and heatmaps — include in final report
5. Prepare dataset: decide on missing values (fill or drop), research and distinguish between scaling and normalization, split into train/validation/test. For images/compressible data: resize and normalize appropriately
6. Define model architecture including hyperparameters
7. Train and tune model and hyperparameters
8. Evaluate performance
9. Final document with report of all steps, results, decisions, and experiments

**Rules:** No code generators. AI tools allowed only as support (not substitution). Must include AI usage report and list of prompts used (or exported chats). May use Keras or PyTorch.

## Progress Tracking

| Step | Status | Notes |
|---|---|---|
| Dataset selection & justification | ✅ Done | Kaggle workout video dataset, spatio-temporal problem framing |
| Problem definition | ✅ Done | Binary classification of exercise execution quality |
| Neural network selection & rationale | ✅ Done | CNN + LSTM hybrid, documented in paper/main.tex |
| Preliminary design | ✅ Done | Processing → representation → model pipeline described in paper |
| Detailed OOP design | ✅ Done | 3 classes: DataManager, DataLoader, RecognitionModel — documented in paper §Design |
| Variable/feature selection | ✅ Done | 33 BlazePose landmarks (x,y,z) + joint angles per exercise |
| Bibliography | ✅ Done | 6 academic references in paper/references.bib |
| Dataset download (correct form) | ✅ Done | 25–26 videos per exercise from Kaggle |
| Dataset download (incorrect form) | ✅ Done | 25–26 videos per exercise via download_bad_form.sh |
| Related work section | ✅ Done | CNN, LSTM, CNN+LSTM, skeletal representation — in paper §Related |
| Methodology section | ✅ Done | Frame extraction, structural normalization, biomechanical angles, tensors — in paper §Methodology |
| Architecture section | ✅ Done | Conv1D + LSTM + sigmoid — in paper §Architecture |
| Initial data exploration (histograms, heatmaps, missing data) | ✅ Done | Notebook + paper §4.4: landmark distributions, correlation heatmaps, joint proxies, squat_bad quality analysis |
| Dataset preparation (missing values, scale vs normalization, train/val/test split) | ✅ Done | Forward-fill, StandardScaler train-only, split 70/15/15 a nivel de video — código + paper §7 |
| Implementation (DataManager, DataLoader, RecognitionModel) | ✅ Done | src/data_manager.py, src/data_loader.py, src/recognition_model.py |
| Model architecture with hyperparameters | ✅ Done | conv_filters=64, lstm_units=128, dropout=0.3, lr=1e-3, seed=42 — paper §8 |
| Generar CSV de landmarks | ✅ Done | data/pose_landmarks_normalized.csv (131 videos, 3930 filas) |
| Training and hyperparameter tuning | ✅ Done | 4 experimentos documentados — .scratch/experiments/experimentos.md + paper §9 |
| Performance evaluation | ✅ Done | Accuracy=73.68%, F1=0.815, Recall=0.917, umbral=0.30 — paper §9 |
| Final report document | ✅ Done | paper/main.tex §1–9 + conclusión + figuras de exploración + curva de aprendizaje |
| AI usage report + prompt log | ✅ Done | paper/ai_usage_report.md — 17 prompts representativos en 4 categorías |

## Technical Design (from documento_tarea.pdf)

### Data Pipeline
- Extract 30 frames per video (uniform sampling)
- MediaPipe BlazePose → 33 landmarks (x, y, z)
- Structural normalization: hip-center translation + torso-length scaling (camera/subject invariance)
- Feature engineering: joint angles via dot product — `θ = arccos((A·B) / (‖A‖‖B‖))`
  - Squat: knee angle; Pull-up: elbow angle; Deadlift: relevant joints

### Tensor Structure
- Shape: `[batch, 30_frames, features]` where features = 33×3 coords + biomechanical angles
- Balanced batches: equal correct/incorrect samples per batch

### Model Architecture
- **Spatial (CNN):** 1D conv layers over per-frame feature vector
- **Temporal (LSTM):** processes sequence of CNN output vectors across 30 frames
- **Classification:** dense layer → sigmoid → correct/incorrect probability

### OOP Structure
- `DataManager` — video → structured data (MediaPipe + NumPy)
- `DataLoader` — tensors + balanced batching
- `RecognitionModel` — CNN+LSTM definition, training, evaluation

## Dataset Structure

```
videos/
  deadlift/      — 25 videos — forma correcta (deadlift_1.mp4 … deadlift_25.mp4)
  deadlift_bad/  — 26 videos — forma incorrecta (deadlift_bad_1.mp4 …)
  squat/         — 25 videos — forma correcta (squat_1.mp4/MOV …)
  squat_bad/     — 26 videos — forma incorrecta (squat_bad_1.mp4 …)
  pull Up/       — 25 videos — forma correcta (pull up_1.mp4 …)  ← espacio en nombre
  pull_up_bad/   — 25 videos — forma incorrecta (pull_up_bad_1.mp4 …)
```

- Formats: `.mp4` and `.MOV` (mixed, especially in `squat/`)
- The `pull Up/` directory name contains a space — quote or escape it in shell commands and use care in Python `Path` or `glob` calls.
- Bad-form videos were downloaded via `download_bad_form.sh` using `yt-dlp` from YouTube searches.

## Key Paths

| Resource | Path |
|---|---|
| Deadlift videos | `videos/deadlift/` |
| Squat videos | `videos/squat/` |
| Pull-up videos | `videos/pull Up/` |

## Tech Stack

- **MediaPipe** (`mediapipe`) — BlazePose for landmark extraction
- **OpenCV** (`cv2`) — frame-level video processing
- **NumPy** — tensor operations, angle calculations
- **PyTorch** or **Keras** — CNN+LSTM model
- **Jupyter notebooks** (`.ipynb`) — exploration and reporting

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
