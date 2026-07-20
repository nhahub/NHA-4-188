# evaluation/evaluation.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import sys
sys.path.append(r"C:\Users\jessi\Downloads\skin_project\fusion\fusion")

import config
from predict import FusionPredictor

# ─── Paths ───
VAL_CSV_PATH = r"C:\Users\jessi\Downloads\skin_project\fusion\fusion\real_val.csv"
DATA_ROOT = r"C:\Users\jessi\Downloads\skin_project\data"                 

CLASS_NAMES = config.CLASS_NAMES


def resolve_image_path(colab_path):
    """
   
   
    """
    parts = colab_path.replace("\\", "/").split("/")
    
    folder_name, file_name = parts[-2], parts[-1]
    return os.path.join(DATA_ROOT, folder_name, file_name)


def run_predictions(val_df, predictor):
    y_true, y_pred = [], []
    skipped = 0
    total = len(val_df)

    for i, (_, row) in enumerate(val_df.iterrows(), start=1):
        image_path = resolve_image_path(row["image_path"])
        symptom_text = row["text"]
        true_label = row["condition"]

        if not os.path.exists(image_path):
            print(f"[{i}/{total}] صورة مفقودة: {image_path}")
            skipped += 1
            continue

        try:
            result = predictor.predict(image_path, symptom_text)
            y_true.append(true_label)
            y_pred.append(result["condition"])
            print(f"[{i}/{total}] OK -> true={true_label} | pred={result['condition']}")
        except Exception as e:
            print(f"[{i}/{total}]  failure {image_path}: {e}")
            skipped += 1

    if skipped:
        print(f"\n تم تخطي {skipped} صف بسبب صور مفقودة أو أخطاء.")

    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Confusion Matrix saved → {save_path}")


def save_metrics(metrics, save_path):
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f" Metrics saved → {save_path}")


if __name__ == "__main__":

    
    val_df = pd.read_csv(VAL_CSV_PATH)
    
    predictor = FusionPredictor(FUSION_MODEL_PATH)

    y_true, y_pred = run_predictions(val_df, predictor)

    if len(y_true) == 0:
        raise RuntimeError("No sample has been predicted successfully — check the pipelines.")

    metrics = compute_metrics(y_true, y_pred)

    print("\n Evaluation Results:")
    print(f"  Accuracy  : {metrics['accuracy']  * 100:.2f}%")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall']    * 100:.2f}%")
    print(f"  F1 Score  : {metrics['f1_score']  * 100:.2f}%")

    os.makedirs("outputs", exist_ok=True)
    save_metrics(metrics, "outputs/metrics.json")
    plot_confusion_matrix(y_true, y_pred, "outputs/confusion_matrix.png")

#  MLflow 
mlflow.set_tracking_uri("sqlite:///mlflow.db") 
mlflow.set_experiment("Skin Condition Detection")

with mlflow.start_run(run_name="Evaluation Run - Real Data"):
    mlflow.log_metrics(metrics) 
    mlflow.log_artifact("outputs/confusion_matrix.png")
    mlflow.log_artifact("outputs/metrics.json")
    print("\n MLflow run logged successfully!")
    
    