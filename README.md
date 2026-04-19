# Indian Sign Language Recognition System

An **AI-powered real-time Indian Sign Language (ISL) recognition and translation system**.  
Uses a webcam to detect hand signs, classify them with a MobileNetV2 deep learning model,
and display + speak the translated text — all through a modern desktop GUI.

---

Dataset used : https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure & File Descriptions](#-project-structure--file-descriptions)
- [Prerequisites](#-prerequisites)
- [Quick Start (All Platforms)](#-quick-start-all-platforms)
- [Detailed Setup](#%EF%B8%8F-detailed-setup--windows-macos--linux)
- [Usage](#-usage)
  - [Step 1 — Collect Training Data](#step-1--collect-training-data)
  - [Step 2 — Train the Model](#step-2--train-the-model)
  - [Step 3 — Run the Real-Time Translator](#step-3--run-the-real-time-translator)
- [Keyboard Shortcuts](#-keyboard-shortcuts)

---

## 🔎 Overview

This project enables **real-time ISL-to-text and speech translation**.  
It supports **35 sign classes** — digits **1–9** and letters **A–Z** — and ships with
tools for **dataset collection**, **model training**, and a **real-time translation GUI
with text-to-speech output**.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Real-Time Pipeline                     │
│                                                           │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Webcam  │───▶│  MediaPipe   │───▶│  Hand Region   │   │
│  │  (OpenCV)│    │  Hands API   │    │  Extraction    │   │
│  └──────────┘    └──────────────┘    └───────┬────────┘   │
│                                              │            │
│                                              ▼            │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Speech  │◀── │  Text Buffer │◀── │  MobileNetV2   │   │
│  │ (pyttsx3)│    │  (Tkinter)   │    │  Classifier    │   │
│  └──────────┘    └──────────────┘    └────────────────┘   │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
│                                                           │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Webcam  │───▶│  MediaPipe   │───▶│  Cropped Hand  │   │
│  │  Capture │    │  Detection   │    │  Images (224²) │   │
│  └──────────┘    └──────────────┘    └───────┬────────┘   │
│                                              │            │
│                                              ▼            │
│  ┌──────────────────┐    ┌───────────────────────────┐    │
│  │ .keras Model File│◀── │ MobileNetV2 Fine-Tuning   │    │
│  │  (≈ 10 MB)       │    │  + Data Augmentation      │    │
│  └──────────────────┘    └───────────────────────────┘    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**Key design decisions:**

| Component | Choice | Rationale |
|---|---|---|
| Hand detection | MediaPipe Hands | Real-time 21-landmark detection, works across platforms |
| Classifier | MobileNetV2 (transfer learning) | Lightweight (~25 MB), fast inference, high accuracy on small datasets |
| GUI framework | Tkinter | Ships with Python — zero extra dependencies |
| Speech engine | pyttsx3 | Offline TTS, uses native engines on every OS |
| Image pipeline | OpenCV + PIL | Industry standard, GPU-optimized where available |

---

## 📁 Project Structure & File Descriptions

```
Indian-Sign-Language-Recognition-System/
│
├── script.py                                 # Main application (real-time translator GUI)
├── collect_imgs.py                           # Webcam-based dataset collection tool
├── train_model.py                            # Standalone MobileNetV2 training script
├── Mobilenetv2_ISL_model.keras               # Pre-trained model file (≈ 10 MB)
├── Mobilenetv2_ISL_model.h5                  # Legacy model file (≈ 25 MB, fallback)
├── requirements.txt                          # Python dependencies (cross-platform)
│
├── Indian/                                   # Training image dataset
│   ├── 1/ ... 9/                             #   Digit classes (1–9)
│   └── A/ ... Z/                             #   Letter classes (A–Z)
│
├── ISL_Mobilenetv2.ipynb                     # Core training pipeline (optimized MobileNetV2)
├── ISL_resnet.ipynb                          # Alternative architecture (ResNet50)
├── ISL_classification.ipynb                  # Performance visualization & analysis
├── LICENSE                                   # MIT License
└── README.md                                 # Documentation
```

### File-by-file breakdown

| File | Purpose |
|---|---|
| **`script.py`** | The main application. Runs a pre-flight check for system dependencies, opens a Tkinter GUI with dual camera views (full frame + hand ROI), runs MediaPipe hand detection, feeds cropped hand images to the MobileNetV2 model for classification, appends predictions to a text buffer, and can speak the accumulated text via pyttsx3. Automatically loads `.keras` or `.h5` model, with cross-platform font and TTS fallbacks. |
| **`collect_imgs.py`** | Interactive data collection tool. Iterates through all 35 classes, prompts the user to show each sign, captures 100 images per class via webcam, crops hand regions using MediaPipe, resizes to 224×224, and saves both images (`.jpg`) and keypoints (`.npy`). |
| **`train_model.py`** | Standalone training script (Keras 3). Loads images from `Indian/`, uses Keras preprocessing layers for data augmentation (baked into the model), builds a MobileNetV2 with fine-tuned top layers, trains with early stopping & learning rate scheduling, evaluates per-class metrics, and saves the final model as `Mobilenetv2_ISL_model.keras`. |
| **`Mobilenetv2_ISL_model.keras`** | Pre-trained Keras 3 model file (≈ 10 MB). Can be used directly with `script.py` without retraining. |
| **`requirements.txt`** | Pinned minimum versions for all Python dependencies, with platform-specific notes for macOS, Linux, and optional GPU support. |

---

## ✅ Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 – 3.12 (3.12 recommended) |
| **Webcam** | Built-in or USB; must be accessible as device index `0` |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **GPU** *(optional)* | NVIDIA CUDA GPU for faster training; not needed for inference |
| **OS** | Windows 10/11, macOS 12+ (Intel & Apple Silicon), Ubuntu 20.04+ / Fedora 36+ |

---

## 🚀 Quick Start (All Platforms)

```bash
# 1. Clone
git clone https://github.com/<your-username>/Indian-Sign-Language-Recognition-System.git
cd Indian-Sign-Language-Recognition-System

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows (cmd)
# venv\Scripts\Activate.ps1      # Windows (PowerShell)

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the app (uses pre-trained model)
python script.py
```

That's it. If something is missing the script will tell you exactly what to install.

---

## 🛠️ Detailed Setup — Windows, macOS & Linux

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Indian-Sign-Language-Recognition-System.git
cd Indian-Sign-Language-Recognition-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Activate it:**

| OS | Command |
|---|---|
| **Windows** (cmd) | `venv\Scripts\activate` |
| **Windows** (PowerShell) | `venv\Scripts\Activate.ps1` |
| **macOS / Linux** | `source venv/bin/activate` |

### 3. Install system dependencies (if needed)

#### Windows

No extra system packages required — everything is handled by pip.

#### macOS

No extra system packages required. For Apple Silicon GPU acceleration (optional):

```bash
pip install tensorflow-metal
```

> **Camera:** The first run will trigger a macOS camera permission dialog.  
> If it doesn't appear, go to: **System Settings → Privacy & Security → Camera** and enable your terminal app.

#### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install python3-tk espeak
```

#### Linux (Fedora / RHEL)

```bash
sudo dnf install python3-tkinter espeak
```

> **Note:** `python3-tk` is required (GUI). `espeak` is optional (text-to-speech).  
> If you skip these, `script.py` will detect the missing packages at startup and print the exact install command.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -c "import cv2, mediapipe, tensorflow, pyttsx3; print('All dependencies OK')"
```

---

## 📖 Usage

### Step 1 — Collect Training Data

> **Skip this step** if you already have images in the `Indian/` directory or are using the pre-trained model.

```bash
python collect_imgs.py
```

1. A camera window opens. You'll be prompted class by class (1–9, A–Z).
2. Show the sign for the current class and press **Q** to start capturing.
3. The tool collects **100 images per class** (adjustable via `dataset_size` in the script).
4. Press **Esc** at any time to stop and move to the next class.

> **Tip:** For best accuracy, collect **300–500 images per class** with varied lighting, angles, and backgrounds.

### Step 2 — Train the Model

> **Skip this step** if using the included `Mobilenetv2_ISL_model.keras`.

```bash
python train_model.py
```

Training details:
- **Architecture:** MobileNetV2 with fine-tuned top 30% of layers
- **Input size:** 224 × 224 RGB
- **Augmentation:** Keras preprocessing layers (rotation, zoom, translation) — baked into the model
- **Callbacks:** Early stopping (patience=5), learning rate reduction
- **Output:** Saves `Mobilenetv2_ISL_model.keras` (~10 MB)

> **For GPU training on Google Colab:** Upload `train_model.py` and a zipped `Indian/` folder, then run in a GPU runtime.

### Step 3 — Run the Real-Time Translator

```bash
python script.py
```

The GUI launches with:
- **Left panel** — Live annotated camera feed with hand landmarks and bounding box
- **Right panel** — Zoomed-in hand ROI (region of interest)
- **Text area** — Accumulated detected characters
- **Control buttons** — Speak, Stop, Clear, Pause/Resume, Quit

> **How it works:** When a hand is detected with ≥90% confidence, the predicted character is appended to the text buffer. Use the **Speak** button to read the text aloud.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl + S` | Speak the detected text |
| `Ctrl + C` | Clear the text area |
| `Ctrl + P` | Pause / resume detection |
| `Esc` | Stop speech |
