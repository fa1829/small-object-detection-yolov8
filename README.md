# 🚁 Small Object Detection — Aerial Drone Vision
### Fine-tuned YOLOv8 for detecting persons and vehicles in aerial drone footage

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![SAHI](https://img.shields.io/badge/SAHI-Sliced%20Inference-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)
![mAP50](https://img.shields.io/badge/mAP50-91.57%25-brightgreen)

---

## 📌 Project Overview

Standard object detection models fail on aerial drone imagery because objects — people, vehicles — occupy only **4–8 pixels** when a high-resolution image is resized to 640×640 for inference. This project solves that through:

1. **Domain-specific fine-tuning** of YOLOv8 on 4,821 labeled aerial drone images
2. **SAHI (Slicing Aided Hyper Inference)** — a technique that slices images into overlapping tiles before inference, preserving resolution for small objects
3. **End-to-end ML pipeline** from raw dataset to evaluated, deployable model

This project was built as part of a practical AI/ML learning journey to demonstrate computer vision competency for DevSecOps and AI research roles.

---

## 📊 Results

| Metric | Base YOLOv8n (pretrained) | Fine-tuned (30 epochs) |
|--------|--------------------------|------------------------|
| mAP50 | ~0.30 | **0.9157** |
| mAP50-95 | ~0.15 | **0.5402** |
| Precision | low | **0.929** |
| Recall | low | **0.865** |
| box_loss | 2.521 | **1.239** |

> Training was done on **CPU only** (AMD Ryzen 7 5700U) over ~48 hours. On Google Colab T4 GPU, equivalent training takes ~30 minutes.

---

## 🗂️ Repository Structure

```
small-object-detection-yolov8/
│
├── notebooks/
│   ├── week1_first_inference.ipynb     # Run YOLOv8 out of the box
│   ├── week2_dataset_exploration.ipynb # Understand YOLO dataset format
│   ├── week3_finetune.ipynb            # Fine-tune on aerial drone data
│   └── week4_sahi.ipynb               # SAHI sliced inference comparison
│
├── results/
│   ├── loss_curve.png                  # Training loss over 30 epochs
│   ├── before_after_sahi.png           # Side-by-side inference comparison
│   ├── sahi_result.png                 # SAHI detection output
│   └── training_metrics.md            # Final epoch metrics summary
│
├── docs/
│   ├── architecture.md                 # How the pipeline works
│   ├── use_cases.md                    # Real-world applications
│   ├── ethics.md                       # AI governance and misuse prevention
│   └── interview_qa.md                # Technical Q&A deep dive
│
├── models/
│   └── README.md                       # Instructions to obtain best.pt
│
├── .env.example                        # API key template (safe to commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/fa1829/small-object-detection-yolov8.git
cd small-object-detection-yolov8
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and add your Roboflow API key
```

### 3. Run inference with the pretrained model (no training needed)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads automatically
results = model("your_aerial_image.jpg")
results[0].show()
```

### 4. Run notebooks in order

```
Week 1 → Week 2 → Week 3 → Week 4
```

> **Week 3 (training) recommendation:** Use Google Colab with T4 GPU.
> Runtime → Change runtime type → T4 GPU → Save

---

## 🧠 How It Works

### The Small Object Problem

When a 4K drone image is resized to 640×640 pixels for YOLOv8 inference, a person who occupied 40×40 pixels becomes just **4×4 pixels** — barely enough visual information for the model to detect. This is the core challenge of small object detection.

### Transfer Learning Pipeline

```
pretrained yolov8n.pt          fine-tuned best.pt
(trained on COCO, 80 classes)  (specialized: aerial view)
         |                              |
         ↓                              ↓
  General objects              Tiny aerial objects
  Street-level view            Drone perspective
  mAP50 ~0.30 on aerial        mAP50 0.9157 on aerial
```

The model starts with weights learned from COCO (a dataset of 330K everyday images). Fine-tuning shifts those 3.2 million parameters from general object recognition to aerial small object specialist — exactly like a general doctor becoming a radiologist through specialized training.

### What Happens During Training (Each Epoch)

1. **Forward pass** — model sees an aerial image, predicts bounding boxes
2. **Loss calculation** — compares predictions to ground truth labels:
   - `box_loss` — how wrong the box position/size is
   - `cls_loss` — how wrong the class prediction is
   - `dfl_loss` — how precise the box edges are
3. **Backpropagation** — adjusts weights to reduce loss
4. **Repeat** for all 4,821 images, 30 times

Loss curve shows steady descent from 2.521 → 1.239, confirming learning.

### SAHI — Slicing Aided Hyper Inference

SAHI addresses small object detection at inference time (no retraining needed):

```
Full image (e.g., 1920×1080)
         |
         ↓
Slice into overlapping tiles (e.g., 512×512 with 20% overlap)
         |
         ↓
Run YOLOv8 inference on EACH tile separately
         |
         ↓
Merge and deduplicate detections (Non-Maximum Suppression)
         |
         ↓
Final result with more detected small objects
```

In this project, the fine-tuned model achieved such high accuracy (92% mAP50) that SAHI provided minimal additional improvement — demonstrating that **domain-specific fine-tuning can be as powerful as inference-time augmentation techniques**.

---

## 📁 Dataset

**Source:** [Roboflow Universe — drone-aerial dataset](https://universe.roboflow.com)

| Property | Value |
|----------|-------|
| Total images | 4,821 |
| Train split | 87% (4,194 images) |
| Validation split | 8% (386 images) |
| Test split | 4% (193 images) |
| Classes | 1 (person/vehicle from aerial view) |
| Format | YOLOv8 (images + .txt labels + data.yaml) |
| Perspective | Aerial drone, varying altitudes |

### Dataset Format (YOLO)

```
drone-aerial-1/
├── train/
│   ├── images/   ← .jpg files
│   └── labels/   ← .txt files (one per image)
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml     ← paths + class names config
```

Each label file contains one line per object:
```
class_id  x_center  y_center  width  height
0         0.512     0.341     0.023  0.041
```
All values normalized 0–1 relative to image dimensions. Small objects have width/height values like `0.02 × 0.02` — that's the challenge.

---

## 🔬 Technical Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Model architecture | YOLOv8n (Ultralytics) | Object detection |
| Deep learning framework | PyTorch 2.12 | Training engine |
| Dataset management | Roboflow | Labeled dataset download |
| Inference augmentation | SAHI | Small object improvement |
| Data analysis | Pandas, Matplotlib | Results visualization |
| Environment | Python 3.11, venv | Dependency management |
| Version control | Git/GitHub | Portfolio showcase |

---

## 🌍 Real-World Use Cases

### Security & Surveillance
- **Perimeter monitoring** — detect unauthorized persons in restricted aerial zones
- **Physical SOC integration** — combine with SIEM systems for automated alerts when unknown persons detected
- **Border surveillance** — aerial detection of unauthorized crossings

### Search & Rescue
- **Disaster response** — locate survivors in collapsed buildings or remote terrain from drone footage
- **Missing persons** — scan large areas faster than human observers

### Smart Cities & Traffic
- **Aerial traffic management** — count and classify vehicles at intersections
- **Crowd density analysis** — monitor public gatherings from above
- **Accident detection** — flag unusual vehicle behavior in real time

### Environmental & Agriculture
- **Wildlife conservation** — count animals in reserves, detect poachers
- **Crop monitoring** — detect disease patches or irrigation issues from drone surveys
- **Deforestation tracking** — compare aerial imagery over time

### Healthcare (analogous application)
The small object detection pipeline directly maps to medical imaging:
- Aerial drone person (~4px) ≡ Early-stage tumor in MRI (~small region)
- SAHI tiling ≡ Pathology slide scanning in overlapping windows
- Fine-tuning on domain data ≡ Training on labeled medical scans

---

## ⚖️ Ethics & Responsible AI

### The Dual-Use Problem
A model that detects people from aerial drones can:
- ✅ Save lives in search and rescue
- ✅ Monitor wildlife without human intrusion
- ❌ Enable mass surveillance
- ❌ Target individuals in conflict zones

### Misuse Prevention Framework

**Access Control (Zero-Trust for Models)**
```python
# Never expose model file directly
# Always gate behind authenticated API

if api_key not in authorized_keys:
    return 403  # Forbidden
if request.use_case not in ALLOWED_USE_CASES:
    return 403
if rate_limit_exceeded(api_key):
    return 429  # Too Many Requests
result = model(image)
log_inference(api_key, timestamp, use_case)
return result
```

**Regulatory Compliance**
- **EU AI Act** — people detection from aerial systems classified as "high risk"; requires registration, human oversight, audit trails
- **NIST AI RMF** — risk identification, measurement, and management framework (maps to cybersecurity risk frameworks)
- **GDPR** — images of identifiable persons require consent; anonymization/blurring of non-target individuals recommended

**Responsible Deployment Checklist**
- [ ] Define and document intended use cases
- [ ] Publish model card with known limitations
- [ ] Implement rate limiting and audit logging
- [ ] Require human review for high-stakes decisions
- [ ] Regular bias and performance audits
- [ ] Incident response plan for model failures

---

## 💰 Monetization & Deployment

### Option 1 — Roboflow Hosted API
Upload `best.pt` to Roboflow workspace → Deploy as cloud endpoint → Gate access with API keys.

Users call:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="USER_KEY")
model = rf.workspace("khandoker-faisal").model("aerial-detection", 1)
result = model.predict("drone_image.jpg").json()
```

### Option 2 — FastAPI Self-Hosted
```python
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best.pt")

@app.post("/detect")
async def detect(file: UploadFile):
    image = await file.read()
    results = model(image)
    return {"detections": len(results[0].boxes)}
```

### Option 3 — Edge Deployment
Export to ONNX for edge devices (Raspberry Pi, Jetson Nano):
```python
model.export(format="onnx")
# Deploy best.onnx to edge hardware
```

---

## 📈 Training Details

### Configuration

```python
model = YOLO("yolov8n.pt")  # nano variant — fastest, lightest
results = model.train(
    data="drone-aerial-1/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    name="small_object_run1",
    exist_ok=True
)
```

### Training Progress

| Epoch | box_loss | cls_loss | dfl_loss | mAP50 |
|-------|----------|----------|----------|-------|
| 1 | 2.521 | 4.789 | 1.374 | ~0.10 |
| 5 | 1.930 | 1.177 | 1.078 | 0.721 |
| 19 | 1.538 | 0.768 | 0.936 | 0.887 |
| 30 | 1.239 | 0.580 | 0.900 | **0.922** |

Loss curves show consistent descent — no overfitting, no plateau — indicating the dataset size and epoch count were well-matched.

### Hardware

| Property | Value |
|----------|-------|
| CPU | AMD Ryzen 7 5700U |
| GPU | None (CPU-only training) |
| RAM | 16GB |
| Training time | ~48 hours |
| Equivalent GPU time | ~30 min (Google Colab T4) |

---

## 🔑 Environment Setup

```bash
# .env.example — copy to .env and fill in values
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

Usage in notebooks:
```python
from dotenv import load_dotenv
import os

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
```

**Never commit `.env` to version control.** See `.gitignore`.

---

## 📦 Model Weights

The fine-tuned `best.pt` (~23MB) is not stored in this repository.

**To obtain:**

**Option A — Train yourself (recommended for learning):**
Follow `notebooks/week3_finetune.ipynb`. Use Google Colab T4 GPU for ~30 minute training.

**Option B — Request access:**
Contact [faisaleeeb@gmail.com](mailto:faisaleeeb@gmail.com) for research or evaluation access.

**Option C — Roboflow hosted model:**
Coming soon — link will be added here upon deployment.

**Final model metrics:**
```
mAP50:    0.9157
mAP50-95: 0.5402
Precision: 0.929
Recall:    0.865
Parameters: 3.2M
Model size: ~23MB
Inference speed: ~85ms/image (CPU)
```

---

## 🧪 Weekly Learning Journey

This project was built as a structured 4-week deep dive:

| Week | Notebook | Goal | Key Learning |
|------|----------|------|--------------|
| 1 | `week1_first_inference.ipynb` | Run YOLOv8 out of the box | Confidence scores, bounding boxes, COCO classes |
| 2 | `week2_dataset_exploration.ipynb` | Understand YOLO data format | images/, labels/, data.yaml, annotation structure |
| 3 | `week3_finetune.ipynb` | Fine-tune on aerial data | Transfer learning, loss curves, mAP metrics |
| 4 | `week4_sahi.ipynb` | SAHI sliced inference | Before/after comparison, inference-time augmentation |

---

## 🔗 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [SAHI — Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [Roboflow Universe](https://universe.roboflow.com)
- [EU AI Act — High Risk AI Systems](https://artificialintelligenceact.eu)
- [NIST AI Risk Management Framework](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf)

---

## 👤 Author

**Khandoker Faisal**
Master's student — Information Systems Security, Concordia University, Montreal

- GitHub: [@fa1829](https://github.com/fa1829)
- Email: faisaleeeb@gmail.com
- Target role: DevSecOps / AI Security Engineer

*This project demonstrates practical ML competency — from raw dataset to fine-tuned model to deployment-ready inference pipeline — with security and ethics considerations integrated throughout.*

---

## 📄 License

MIT License — free to use for research and educational purposes.
For commercial use or deployment in security-sensitive contexts, contact the author.
