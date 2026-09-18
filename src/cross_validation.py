"""
Ultra-Fast Stratified Cross-Validation Engine for FNFC Dataset Benchmarking.
Precomputes TF-IDF and ISO RAG features once across the dataset to execute 5-fold CV in seconds.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from dataset_loader import load_fnfc_dataset
from iso_rag_retriever import ISORagRetriever
from benchmark_baselines import get_precomputed_models

def run_cross_validation(n_splits=5, random_state=42):
    print("=" * 80)
    print(f" CONTROLLED APPLE-TO-APPLE BENCHMARK: {n_splits}-FOLD STRATIFIED CV")
    print(" dataset: FNFC.csv (7,060 samples, 14 classes)")
    print("=" * 80)

    # 1. Load Dataset
    _, _, encoder, full_df = load_fnfc_dataset()
    texts = full_df['clean_text'].values
    labels = full_df['label'].values

    # 2. Precompute TF-IDF (4000D) & ISO RAG Features (28D) Once
    print("\nPrecomputing text TF-IDF & ISO 25010 RAG context features...")
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words='english')
    X_text_all = tfidf.fit_transform(texts)

    retriever = ISORagRetriever(encoder.classes_)
    X_iso_all = retriever.transform_all_features(texts)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model_names = get_precomputed_models().keys()
    results = {name: {"accuracy": [], "macro_f1": [], "weighted_f1": [], "mcc": []} for name in model_names}

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"Running Fold {fold}/{n_splits}...")
        X_text_tr, X_text_val = X_text_all[train_idx], X_text_all[val_idx]
        X_iso_tr, X_iso_val = X_iso_all[train_idx], X_iso_all[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]

        models = get_precomputed_models()
        for name, model in models.items():
            model.fit(X_text_tr, X_iso_tr, y_tr)
            preds = model.predict(X_text_val, X_iso_val)

            acc = accuracy_score(y_val, preds)
            m_f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            w_f1 = f1_score(y_val, preds, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_val, preds)

            results[name]["accuracy"].append(acc)
            results[name]["macro_f1"].append(m_f1)
            results[name]["weighted_f1"].append(w_f1)
            results[name]["mcc"].append(mcc)

    # Summary Output Table
    summary = {}
    print("\n" + "=" * 92)
    print(f" {n_splits}-FOLD STRATIFIED CROSS-VALIDATION SUMMARY (MEAN ± STD)")
    print("=" * 92)
    print(f"{'Model Name':<32} | {'Accuracy (%)':<16} | {'Macro F1 (%)':<16} | {'MCC (%)':<16}")
    print("-" * 92)

    for name, metrics in results.items():
        acc_m, acc_s = np.mean(metrics["accuracy"]) * 100, np.std(metrics["accuracy"]) * 100
        mf1_m, mf1_s = np.mean(metrics["macro_f1"]) * 100, np.std(metrics["macro_f1"]) * 100
        wf1_m, wf1_s = np.mean(metrics["weighted_f1"]) * 100, np.std(metrics["weighted_f1"]) * 100
        mcc_m, mcc_s = np.mean(metrics["mcc"]) * 100, np.std(metrics["mcc"]) * 100

        summary[name] = {
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
            "macro_f1_mean": mf1_m, "macro_f1_std": mf1_s,
            "weighted_f1_mean": wf1_m, "weighted_f1_std": wf1_s,
            "mcc_mean": mcc_m, "mcc_std": mcc_s,
            "raw_metrics": metrics
        }
        
        print(f"{name:<32} | {acc_m:5.2f} ± {acc_s:4.2f}%   | {mf1_m:5.2f} ± {mf1_s:4.2f}%   | {mcc_m:5.2f} ± {mcc_s:4.2f}%")

    print("-" * 92)

    # Save JSON Results
    results_dir = r"C:\Users\umert\Downloads\FNFC_Angle1_ISO25010\results"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "10fold_cv_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[SUCCESS] Cross-validation completed! Results saved to: {out_path}")
    return summary

if __name__ == "__main__":
    run_cross_validation()
