# 🔬 Skin Lesion & Cancer Classification with MobileNetV2

An end-to-end Deep Learning project for classifying skin lesions into **Benign** and **Malignant** categories using Transfer Learning with **MobileNetV2** and **Binary Focal Loss**. Includes model weights saved in native Keras (`.keras`), legacy HDF5 (`.h5`), and mobile-optimized TensorFlow Lite (`.tflite`) formats.

---

## 📌 Project Overview

Early detection of malignant skin lesions is critical for effective clinical intervention. This repository contains the complete training pipeline, model evaluation metrics, serialized model weights, and an inference script to classify dermoscopic skin images.

### Key Features
- **Architecture**: Transfer learning with pre-trained **MobileNetV2** backbone and custom classification head.
- **Handling Class Imbalance**: Implements **Binary Focal Crossentropy ($\gamma = 2.0$)** to penalize hard-to-classify samples and focus on malignant cases.
- **Comprehensive Augmentations**: Rotation, width/height shifts, shear, zoom, and horizontal/vertical flips.
- **Multi-Format Export**: Native Keras (`.keras`), HDF5 (`.h5`), and edge-ready TensorFlow Lite (`.tflite`).
- **Production-Ready Inference**: Lightweight inference script utilizing TFLite for fast predictions on CPU and mobile devices.

---

## 📁 Repository Structure

```text
├── SkinCare.ipynb               # Complete training, evaluation & conversion notebook
├── skin_analyzer_model.keras    # Native Keras format model weights (~27.2 MB)
├── skin_analyzer_model.h5       # HDF5 model weights (~27.1 MB)
├── skin_analyzer_model.tflite   # Quantized TFLite model (~11.5 MB)
├── predict.py                   # Standalone inference script for test images
├── requirements.txt             # Project dependencies
└── .gitignore                   # Ignored datasets, checkpoints & environments
```

---

## 📊 Model Architecture & Performance

### 1. Network Architecture
- **Backbone**: MobileNetV2 (ImageNet weights, feature extraction frozen during initial stage).
- **Head**:
  - `GlobalAveragePooling2D()`
  - `Dense(512, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(1, activation='sigmoid')`
- **Input Size**: `224 × 224 × 3`

### 2. Evaluation Results (Test Set: 2,094 images)
- **Overall Test Accuracy**: **83.52%**
- **Training Accuracy**: **87.28%**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 0.87 | 0.89 | 0.88 | 1404 |
| **Malignant** | 0.77 | 0.72 | 0.74 | 690 |
| **Accuracy** | | | **0.84** | **2094** |
| **Macro Avg** | 0.82 | 0.81 | 0.81 | 2094 |
| **Weighted Avg**| 0.83 | 0.84 | 0.83 | 2094 |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd <YOUR-REPO-NAME>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Inference on a Sample Image
You can test any skin lesion image using the provided `predict.py` script:
```bash
python predict.py path/to/sample_lesion.jpg
```

Sample Output:
```text
=============================================
 Skin Lesion Classification Result
=============================================
Prediction: BENIGN
Confidence: 96.38%
Raw Score:  0.0362
=============================================
```

### 4. Training & Fine-Tuning
Launch Jupyter Notebook to inspect or re-run the pipeline:
```bash
jupyter notebook SkinCare.ipynb
```

---

## 💾 Model Weights & GitHub Storage Note

All trained model weights in this repository are under GitHub's standard 50 MB / 100 MB limits:
- `skin_analyzer_model.keras`: **~27.2 MB**
- `skin_analyzer_model.h5`: **~27.1 MB**
- `skin_analyzer_model.tflite`: **~11.5 MB**

They can be tracked directly with Git or managed via Git LFS / GitHub Releases as desired.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
