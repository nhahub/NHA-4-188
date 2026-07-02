# Skin Disease Detection

Data pipeline for the skin diseases classification project.

## Structure

```
Skin-Disease-Detection/
│
├── data/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   └── augmentation.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── outputs/
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

Download the processed dataset from **[Kaggle](https://www.kaggle.com/datasets/dinvrvslvn/preprocessed-balanced-skin-diseases)**.

Place the `train/`, `val/`, and `test/` folders inside `data/processed/`.

Each split folder contains one subfolder per class (PyTorch `ImageFolder` format).

## Usage

```python
import sys
sys.path.append("src")

from dataset import get_train_loader, get_val_loader, get_test_loader, get_class_names

train_loader = get_train_loader()
val_loader = get_val_loader()
test_loader = get_test_loader()

print(get_class_names())
```

## Notes

- The training set is already balanced (imbalance ratio reduced from **6.34** to **2.57**) via targeted augmentation on underrepresented classes only.
- Validation and test sets are intentionally left unbalanced to reflect real-world class distribution during evaluation.
- Resizing (**224×224**) and normalization (ImageNet statistics) are applied inside `src/transforms.py`, not on the stored images.
- `src/augmentation.py` is kept for reference and reproducibility if the balancing step needs to be re-run on new raw data.