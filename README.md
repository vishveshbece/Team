# 🔍 DeepDefect-AI

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg?style=for-the-badge\&logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/Optimization-ONNX-005CED.svg?style=for-the-badge\&logo=onnx)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-blue.svg?style=for-the-badge)]()

> **A high-throughput computer vision pipeline optimized for Industry 4.0 semiconductor defect detection, leveraging MobileNetV3 and super-convergence training strategies.**
> DATASET LINK: https://drive.google.com/file/d/1qOoRbfl7UOKl3At-ZgJiwWrNlG-Rgs-c/view?usp=sharing

---

## 📖 Abstract

**DeepDefect-AI** is a lightweight yet high-accuracy deep learning framework designed for **real-time semiconductor wafer and die defect classification**. The system targets **edge deployment** scenarios where low latency, minimal memory footprint, and portability are critical.

This repository contains an end-to-end pipeline (data, training, evaluation, and export) tuned for production-style inspections captured by optical microscopes, SEMs, and defect review stations.

---

## 🔑 Highlights

* MobileNetV3-based backbone with a custom **Dense-Neck** head for better defect feature representation.
* Training with **Super-Convergence** (OneCycleLR) and **Label Smoothing**.
* Mixed precision training (AMP) support for faster experiments on CUDA devices.
* ONNX export ready for edge runtime (NXP eIQ, TensorRT, OpenVINO).
* Validation reporting tools (confusion matrix, ROC, per-class metrics, confidence distribution).

---

## 📁 Project Structure

```
DeepDefect-AI/
├── data/                       # local dataset pointer (not stored in repo)
├── scripts/
│   ├── train.py                # training entrypoint
│   ├── eval_report.py          # validation + report generation (no retrain)
│   └── export_onnx.py          # ONNX export helper (if separate)
├── models/                     # trained artifacts (best_model.pth, .onnx)
├── notebooks/                  # optional analysis notebooks
├── README.md                   # this file (rendered)
└── LICENSE
```

---

## 📊 Dataset

### Expected dataset structure

```
Balanced_Dataset/
├── train/
│   ├── bridge/
│   ├── Clean/
│   ├── Crack/
│   └── ...
└── val/
    ├── bridge/
    ├── Clean/
    ├── Crack/
    └── ...
```
---

## 🛠 Installation

> Recommended: create and activate a Python virtual environment (PowerShell example below). If you're on an external HDD, the venv still works — but **activate it** before `pip install` to avoid user-site installs.

### Windows PowerShell quick setup

```powershell

python -m venv pack
# If execution policy blocks activation, run once:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\pack\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision onnx scikit-learn tqdm matplotlib seaborn opencv-python
```

### Linux / macOS (bash)

```bash
python3 -m venv pack
source pack/bin/activate
pip install --upgrade pip
pip install torch torchvision onnx scikit-learn tqdm matplotlib seaborn opencv-python
```

---

## 🚀 Usage

> All commands assume you are inside an activated virtual environment and the repository root is the current working directory. Adjust paths (e.g., `E:\IESA`) as needed.

### 1) Training (transfer learning + Dense-Neck + Super-Convergence)

`train.py` is the main script. Example usage:

```bash
python scripts/train.py \
  --data_root "E:\IESA\Balanced_Dataset" \
  --save_dir "E:\IESA\models" \
  --epochs 20 \
  --batch_size 32 \
  --img_size 224 \
  --lr 3e-4 \
  --num_classes 9
```

**What the training script does**

* Loads MobileNetV3-Small backbone (ImageNet pretrained)
* Replaces the classifier with a Dense-Neck head: `Linear(in,1024) -> HardSwish -> Dropout(0.3) -> Linear(1024, num_classes)`
* Uses `CrossEntropyLoss` with label smoothing, `AdamW` optimizer
* OneCycleLR or CosineAnnealing scheduler (configurable)
* Mixed precision with `torch.cuda.amp` (when CUDA available)
* Saves `best_model.pth` to the `--save_dir` when validation accuracy improves
* Exports ONNX automatically after training

### 2) Evaluation & Validation Report (no retraining)

Run the provided evaluation script to generate a validation report and graphs.

```bash
python scripts/eval_report.py \
  --data_root "E:\IESA\Balanced_Dataset" \
  --model_path "E:\IESA\models\best_model.pth" \
  --report_dir "E:\IESA\validation_report" \
  --img_size 224 \
  --batch_size 32
```

**Output files (in `report_dir`)**

* `classification_report.txt` (precision/recall/F1)
* `confusion_matrix.png`
* `per_class_accuracy.png`
* `roc_curves.png`
* `confidence_distribution.png`

These plots are publication-ready for a submission or demo.

### 3) ONNX Export (if not already exported)

If the training script didn't export ONNX automatically, run:

```bash
python scripts/export_onnx.py \
  --model_path "E:\IESA\models\best_model.pth" \
  --onnx_path "E:\IESA\models\defect_model_v1.onnx" \
  --img_size 224
```

**Export guidelines**

* Use `opset_version=13` for broad compatibility
* Enable `do_constant_folding=True`
* Provide dynamic axes for batch size if you need variable batch inference

### 4) Quick inference (example snippet)

```python
import onnx
import onnxruntime as ort
import cv2
import numpy as np

onnx_model_path = "E:\\IESA\\models\\defect_model_v1.onnx"
ort_sess = ort.InferenceSession(onnx_model_path)

img = cv2.imread('path_to_sample.jpg')
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
img = np.transpose(img, (2,0,1)).astype(np.float32)
img = np.expand_dims(img, 0)

outputs = ort_sess.run(None, {'input': img})
probs = softmax(outputs[0], axis=1)
pred = np.argmax(probs, axis=1)
```

---

## 📈 Evaluation Metrics

* Overall accuracy
* Per-class precision, recall, F1
* Confusion matrix
* ROC-AUC (one-vs-rest)
* Confidence histogram

Use the `scripts/eval_report.py` produced plots for your submission slides.

---

## 🔧 Tips & Troubleshooting

* **`pip` installed to user site**: This happens if you didn't activate the venv. Activate the venv before `pip install` or use the full path to venv `.
  pack\Scripts\pip.exe install package`.
* **PowerShell ExecutionPolicy**: If activation fails, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.
* **Windows path length**: Avoid extremely long paths — move project to `E:\IESA` as you already do.
* **GPU missing or OutOfMemory**: Reduce `batch_size`, enable mixed precision, or train on CPU with smaller batches.

---

## 🧪 Reproducibility

* Set random seeds for `torch`, `numpy`, and `random` in training scripts.
* Log hyperparameters and the exact code commit used for the final artifact.
* Save `class_names` mapping alongside the model: `classes.json`.

---

## 🔮 Roadmap

* Grad-CAM visualizations for explainability
* Post-Training Quantization (INT8) and latency benchmarks
* Docker + ONNX Runtime container for deployment

---

## 📜 License

MIT License — see `LICENSE` file.

---

## 🤝 Contributors

* **Deep Learning Engineer:** Jaiwant D
* **Data Pipeline:** Vishvesh B

Built with ❤️ using PyTorch and ONNX.

---


