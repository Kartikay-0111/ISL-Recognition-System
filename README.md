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
- [Setup — Windows, macOS & Linux](#-setup--windows-macos--linux)
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
│                    Real-Time Pipeline                      │
│                                                           │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Webcam   │───▶│  MediaPipe   │───▶│  Hand Region   │   │
│  │  (OpenCV) │    │  Hands API   │    │  Extraction    │   │
│  └──────────┘    └──────────────┘    └───────┬────────┘   │
│                                              │            │
│                                              ▼            │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Speech   │◀──│  Text Buffer  │◀──│  MobileNetV2   │   │
│  │  (pyttsx3)│    │  (Tkinter)   │    │  Classifier    │   │
│  └──────────┘    └──────────────┘    └────────────────┘   │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                    Training Pipeline                       │
│                                                           │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Webcam   │───▶│  MediaPipe   │───▶│  Cropped Hand  │   │
│  │  Capture  │    │  Detection   │    │  Images (224²)  │   │
│  └──────────┘    └──────────────┘    └───────┬────────┘   │
│                                              │            │
│                                              ▼            │
│  ┌──────────────────┐    ┌───────────────────────────┐    │
│  │  .h5 Model File  │◀──│  MobileNetV2 Fine-Tuning  │    │
│  │  (≈ 25 MB)       │    │  + Data Augmentation       │    │
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
├── Mobilenetv2_ISL_model.h5                  # Pre-trained model weights (≈ 25 MB)
├── requirements.txt                          # Python dependencies (cross-platform)
│
├── Indian/                                   # Training image dataset
│   ├── 1/ ... 9/                             #   Digit classes (1–9)
│   └── A/ ... Z/                             #   Letter classes (A–Z)
│
├── Indian_keypoints/                         # Unused keypoint data (can be ignored)
│
├── ISL_Mobilenetv2.ipynb                     # Core training pipeline (optimized MobileNetV2)
├── ISL_resnet.ipynb                          # Alternative architecture (ResNet50)
├── ISL_classification.ipynb                  # Performance visualization & analysis
├── Mobilenetv2_ISL_model.keras               # Optimized model file (≈ 10 MB)
├── LICENSE                                   # MIT License
└── README.md                                 # Documentation
```

### File-by-file breakdown

| File | Purpose |
|---|---|
| **`script.py`** | The main application. Opens a Tkinter GUI with dual camera views (full frame + hand ROI), runs MediaPipe hand detection, feeds cropped hand images to the MobileNetV2 model for classification, appends predictions to a text buffer, and can speak the accumulated text via pyttsx3. Cross-platform font and TTS fallbacks included. |
| **`collect_imgs.py`** | Interactive data collection tool. Iterates through all 35 classes, prompts the user to show each sign, captures 100 images per class via webcam, crops hand regions using MediaPipe, resizes to 224×224, and saves both images (`.jpg`) and keypoints (`.npy`). |
| **`train_model.py`** | Standalone training script. Loads images from `Indian/`, applies data augmentation, builds a MobileNetV2 model with fine-tuned top layers, trains with early stopping & learning rate scheduling, evaluates per-class metrics, and saves the final model as `Mobilenetv2_ISL_model.h5`. |
| **`Mobilenetv2_ISL_model.h5`** | Pre-trained Keras model file. Can be used directly with `script.py` without retraining. |
| **`requirements.txt`** | Pinned minimum versions for all Python dependencies, with platform-specific notes for macOS Apple Silicon, Linux espeak, and optional GPU support. |

---

## ✅ Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.8 – 3.11 (3.10 recommended) |
| **Webcam** | Built-in or USB; must be accessible as device index `0` |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **GPU** *(optional)* | NVIDIA CUDA GPU for faster training; not needed for inference |
| **OS** | Windows 10/11, macOS 12+ (Intel & Apple Silicon), Ubuntu 20.04+ / Fedora 36+ |

---

## 🚀 Setup — Windows, macOS & Linux

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Indian-Sign-Language-Recognition-System.git
cd Indian-Sign-Language-Recognition-System
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

**Activate it:**

| OS | Command |
|---|---|
| **Windows** (cmd) | `venv\Scripts\activate` |
| **Windows** (PowerShell) | `venv\Scripts\Activate.ps1` |
| **macOS / Linux** | `source venv/bin/activate` |

### 3. Install dependencies

#### Windows

```bash
pip install -r requirements.txt
```

> **Note:** Tkinter ships with the official Python installer on Windows. No extra steps.

#### macOS (Intel)

```bash
pip install -r requirements.txt
```

> **Note:** If you don't have Tkinter, install it via:
> ```bash
> brew install python-tk@3.10   # match your Python version
> ```

#### macOS (Apple Silicon — M1/M2/M3/M4)

For GPU-accelerated TensorFlow on Apple Silicon:

```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

> The `tensorflow` line in `requirements.txt` will be satisfied by `tensorflow-macos`.

#### Linux (Ubuntu / Debian)

```bash
# System dependencies
sudo apt update
sudo apt install python3-tk espeak libespeak-dev

# Python packages
pip install -r requirements.txt
```

#### Linux (Fedora / RHEL)

```bash
sudo dnf install python3-tkinter espeak espeak-devel
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import cv2, mediapipe, tensorflow, pyttsx3; print('All dependencies OK')"
```

### 5. Grant camera permissions (macOS only)

On macOS, the first run will trigger a camera permission dialog.  
If it doesn't appear, go to:  
**System Settings → Privacy & Security → Camera** and enable your terminal app.

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

> **Skip this step** if using the included `Mobilenetv2_ISL_model.h5`.

```bash
python train_model.py
```

Training details:
- **Architecture:** MobileNetV2 with fine-tuned top 30% of layers
- **Input size:** 224 × 224 RGB
- **Augmentation:** Rotation, zoom, shift, shear
- **Callbacks:** Early stopping (patience=5), learning rate reduction
- **Output:** Saves `Mobilenetv2_ISL_model.h5` (~25 MB)

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
