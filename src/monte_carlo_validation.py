"""
Ultra-Fast 30-Run Randomized Train/Test Split Validation Engine for FNFC Dataset Benchmarking.
Evaluates model stability across 30 distinct random seeds (seeds 0..29) using precomputed features.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from dataset_loader import load_fnfc_dataset
from iso_rag_retriever import ISORagRetriever
from benchmark_baselines import get_precomputed_models

def run_30_random_splits(n_runs=30):
    print("=" * 80)
    print(f" RIGOROUS ACADEMIC VALIDATION: {n_runs} RANDOMIZED TRAIN/TEST RUNS")
    print(" dataset: FNFC.csv (7,060 samples, 14 classes)")
    print("=" * 80)

    # 1. Load Dataset
    _, _, encoder, full_df = load_fnfc_dataset()
    texts = full_df['clean_text'].values
    labels = full_df['label'].values

    # 2. Precompute Features
    print("\nPrecomputing text TF-IDF & ISO 25010 RAG features for 30 runs...")
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words='english')
    X_text_all = tfidf.fit_transform(texts)

    retriever = ISORagRetriever(encoder.classes_)
    X_iso_all = retriever.transform_all_features(texts)

    model_names = get_precomputed_models().keys()
    results = {name: {"accuracy": [], "macro_f1": [], "weighted_f1": [], "mcc": []} for name in model_names}

    indices = np.arange(len(labels))

    for seed in range(n_runs):
        tr_idx, te_idx, train_y, test_y = train_test_split(
            indices, labels, test_size=0.2, random_state=seed, stratify=labels
        )

        X_text_tr, X_text_te = X_text_all[tr_idx], X_text_all[te_idx]
        X_iso_tr, X_iso_te = X_iso_all[tr_idx], X_iso_all[te_idx]

        models = get_precomputed_models()
        for name, model in models.items():
            model.fit(X_text_tr, X_iso_tr, train_y)
            preds = model.predict(X_text_te, X_iso_te)

            acc = accuracy_score(test_y, preds)
            m_f1 = f1_score(test_y, preds, average='macro', zero_division=0)
            w_f1 = f1_score(test_y, preds, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(test_y, preds)

            results[name]["accuracy"].append(acc)
            results[name]["macro_f1"].append(m_f1)
            results[name]["weighted_f1"].append(w_f1)
            results[name]["mcc"].append(mcc)

        if (seed + 1) % 5 == 0 or seed == n_runs - 1:
            print(f"Completed {seed + 1}/{n_runs} randomized runs.")

    # Aggregate Mean and Std
    summary = {}
    print("\n" + "=" * 92)
    print(f" {n_runs}-RUN RANDOMIZED SPLITS SUMMARY (MEAN ± STD)")
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

    # Save Results
    results_dir = r"C:\Users\umert\Downloads\FNFC_Angle1_ISO25010\results"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "30run_random_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[SUCCESS] {n_runs}-Run Randomized Splits completed! Results saved to: {out_path}")
    return summary

if __name__ == "__main__":
    run_30_random_splits(n_runs=30)
