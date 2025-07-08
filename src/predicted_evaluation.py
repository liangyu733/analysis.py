import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def safe_divide(a, b):
    return a / b if b != 0 else 0

def evaluate_model(y_true, y_prob, threshold=0.5):
    # 將機率轉為分類結果
    y_pred = (y_prob >= threshold).astype(int)

    # 計算 TP, FP, TN, FN
    TP = ((y_true == 1) & (y_pred == 1)).sum() # True Positives
    FP = ((y_true == 0) & (y_pred == 1)).sum() # False Positives
    TN = ((y_true == 0) & (y_pred == 0)).sum() # True Negatives
    FN = ((y_true == 1) & (y_pred == 0)).sum() # False Negatives

    # 基本指標
    accuracy = safe_divide(TP + TN, TP + TN + FP + FN)
    precision = safe_divide(TP, TP + FP)
    recall = safe_divide(TP, TP + FN)
    specificity = safe_divide(TN, TN + FP)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    # ROC 曲線與 AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # 畫圖
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 輸出
    return {
        "conf_matrix": {
            "TP": int(TP), "FP": int(FP),
            "TN": int(TN), "FN": int(FN)
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1_score
        },
        "auc": roc_auc
    }
