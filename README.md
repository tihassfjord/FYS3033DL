# 🧠 FYS-3033 Home Exam – Deep Learning Image Classifier

This project contains my submission for the home exam in FYS-3033 at UiT, Spring 2025.

I build a **VGG-11 with BatchNorm**, trained **from scratch**, to classify 96×96 color images of **planes**, **ships**, and **trucks** using PyTorch.

---

## 📁 Project Structure

```
FYS3033DL/
├── data/                # Training and evaluation datasets (problem2, problem3)
├── doc/                 # LaTeX-written report (report.tex + report.pdf)
├── logs/                # CSV files with training metrics (loss, accuracy)
├── models/              # Saved PyTorch models (.pth)
├── plots/               # PNG figures (training curves, confusion matrices)
├── src/                 # All core Python code
│   ├── train.py         # Training loop with logging, tqdm progress bar
│   ├── vgg11bn.py       # VGG-11 model with optional Dropout
│   ├── utils.py         # Dataset class, entropy function, seed control
│   └── evaluation.py    # Confusion matrix, classification report, entropy hist
├── home_exam.ipynb      # Main notebook: analysis, training, visualizations
├── FYS3033_homexam.pdf  # Original exam task description
```

---

## 📦 Contents

- **Problem 2**: Train VGG-11 from scratch on a labeled dataset (planes/ships/trucks).
- **Problem 3**: Train a ResNet-18 and perform backdoor detection using saliency maps & occlusion.

All models are trained without using pretrained weights or torchvision shortcuts — in full compliance with the exam requirements.

---

## 🚀 Running the Code

1. Clone the repo and set up your Python env (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook:
   ```bash
   jupyter notebook home_exam.ipynb
   ```

3. Or use the Python API:
   ```python
   from src.train import train_model
   from src.vgg11bn import VGG11BN
   ```

---

## ✍️ Author

**Tor-Ivar Hassfjord**  
Candidate number XX · UiT — The Arctic University of Norway  
Course: FYS-3033 — Deep Learning, Spring 2025
