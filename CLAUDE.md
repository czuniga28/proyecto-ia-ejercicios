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
| Neural network selection & rationale | ✅ Done | CNN + LSTM hybrid, documented in documento_tarea.pdf |
| Preliminary design | ✅ Done | Processing → representation → model pipeline described |
| Detailed OOP design | ✅ Done | 3 classes: DataManager, DataLoader, RecognitionModel |
| Variable/feature selection | ✅ Done | 33 BlazePose landmarks (x,y,z) + joint angles per exercise |
| Bibliography | ✅ Done | 6 academic references |
| Initial data exploration (histograms, heatmaps, missing data) | ❌ Pending | Required for final report |
| Dataset preparation (missing values, scale vs normalization, train/val/test split) | ❌ Pending | Must also explain difference between scaling and normalization |
| Model architecture with hyperparameters | ⚠️ Partial | Architecture defined; specific hyperparameters not yet set |
| Training and hyperparameter tuning | ❌ Pending | |
| Performance evaluation | ❌ Pending | |
| Final report document | ❌ Pending | Must include all steps, decisions, and results |
| AI usage report + prompt log | ❌ Pending | Required per assignment rules |

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
  deadlift/   — 32 videos (deadlift_1.mp4 … deadlift_32.mp4)
  squat/      — 29 videos (squat_1.mp4/MOV … squat_29.mp4)
  pull Up/    — 26 videos (pull up_1.mp4 … pull up_26.mp4)  ← directory name has a space
```

- Formats: `.mp4` and `.MOV` (mixed, especially in `squat/`)
- The `pull Up/` directory name contains a space — quote or escape it in shell commands and use care in Python `Path` or `glob` calls.

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
