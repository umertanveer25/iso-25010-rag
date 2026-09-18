"""
Main Evaluation Pipeline for Angle 1: ISO/IEC 25010 Knowledge-Augmented RAG Classification.
Runs baseline vs. ISO-Augmented classifier and generates comprehensive comparative performance metrics.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report
from dataset_loader import load_fnfc_dataset, FNFC_CLASSES
from augmented_classifier import FNFCBaselineClassifier, ISOAugmentedFusionClassifier

def evaluate_model(y_true, y_pred, encoder):
    """
    Computes Accuracy, Macro F1, Weighted F1, MCC, and per-class metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cls_report = classification_report(
        y_true, y_pred, 
        target_names=encoder.classes_, 
        output_dict=True, 
        zero_division=0
    )
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc),
        "class_report": cls_report
    }

def main():
    print("=" * 70)
    print(" ANGLE 1: ISO/IEC 25010 KNOWLEDGE-AUGMENTED RAG CLASSIFICATION")
    print(" dataset: FNFC.csv (7,060 samples, 14 classes)")
    print("=" * 70)

    # 1. Load Dataset
    train_df, test_df, encoder, _ = load_fnfc_dataset()
    
    train_texts, train_labels = train_df['clean_text'].values, train_df['label'].values
    test_texts, test_labels = test_df['clean_text'].values, test_df['label'].values

    # 2. Train Standard Baseline
    print("\n[1/2] Training Standard Baseline Classifier (TF-IDF + Balanced Classifier)...")
    baseline_model = FNFCBaselineClassifier()
    baseline_model.fit(train_texts, train_labels)
    baseline_preds = baseline_model.predict(test_texts)
    baseline_metrics = evaluate_model(test_labels, baseline_preds, encoder)

    # 3. Train Angle 1 ISO-Augmented Fusion Model
    print("\n[2/2] Training Angle 1 ISO 25010 Knowledge-Augmented Fusion Classifier...")
    iso_model = ISOAugmentedFusionClassifier(fnfc_classes=encoder.classes_)
    iso_model.fit(train_texts, train_labels)
    iso_preds = iso_model.predict(test_texts)
    iso_metrics = evaluate_model(test_labels, iso_preds, encoder)

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 70)
    print(" EXPERIMENTAL RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Standard Baseline':<20} | {'Angle 1 (ISO 25010 RAG)':<25} | {'Improvement':<12}")
    print("-" * 88)

    metrics_list = [
        ("Accuracy", "accuracy"),
        ("Macro F1-Score", "macro_f1"),
        ("Weighted F1-Score", "weighted_f1"),
        ("MCC Correlation", "mcc")
    ]

    summary_diffs = {}
    for label, key in metrics_list:
        b_val = baseline_metrics[key]
        i_val = iso_metrics[key]
        diff = i_val - b_val
        summary_diffs[key] = diff
        diff_str = f"+{diff*100:.2f}%" if diff >= 0 else f"{diff*100:.2f}%"
        print(f"{label:<25} | {b_val*100:19.2f}% | {i_val*100:24.2f}% | {diff_str:<12}")

    print("-" * 88)

    # 5. Per-Class F1 Breakdown
    print("\nPER-CLASS F1-SCORE BREAKDOWN:")
    print(f"{'Class':<6} | {'Class Name':<30} | {'Baseline F1':<12} | {'Angle 1 F1':<12} | {'Diff':<10}")
    print("-" * 78)
    
    for cls in encoder.classes_:
        b_f1 = baseline_metrics["class_report"][cls]["f1-score"]
        i_f1 = iso_metrics["class_report"][cls]["f1-score"]
        diff = i_f1 - b_f1
        diff_str = f"+{diff*100:.2f}%" if diff >= 0 else f"{diff*100:.2f}%"
        cls_name = encoder.classes_[encoder.transform([cls])[0]]
        print(f"{cls:<6} | {cls_name:<30} | {b_f1*100:11.2f}% | {i_f1*100:11.2f}% | {diff_str:<10}")

    # 6. Save Results to Output Folder
    results_dir = r"C:\Users\umert\Downloads\FNFC_Angle1_ISO25010\results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    
    with open(results_path, "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "iso_augmented": iso_metrics,
            "improvements": summary_diffs
        }, f, indent=4)

    print(f"\n[SUCCESS] Execution complete! Detailed results saved to: {results_path}")

if __name__ == "__main__":
    main()
