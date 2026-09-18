"""
Controlled Apple-to-Apple Statistical Significance & Net Gain (Delta) Engine.
Computes exact 1-to-1 gains (Base vs Base + ISO RAG) and LaTeX paper tables.
"""

import json
import os
import numpy as np
from scipy import stats

def perform_statistical_tests(cv_results_path=r"C:\Users\umert\Downloads\FNFC_Angle1_ISO25010\results\10fold_cv_results.json"):
    if not os.path.exists(cv_results_path):
        raise FileNotFoundError(f"CV results file not found at: {cv_results_path}")

    with open(cv_results_path, "r") as f:
        data = json.load(f)

    pairs = [
        ("MNB (Base)", "MNB + ISO RAG"),
        ("Linear SVM (Base)", "Linear SVM + ISO RAG"),
        ("Random Forest (Base)", "Random Forest + ISO RAG"),
        ("Logistic Regression (Base)", "Logistic Regression + ISO RAG"),
        ("LightGBM (Base)", "LightGBM + ISO RAG"),
        ("Extra Trees (Base)", "Extra Trees + ISO RAG"),
        ("Ensemble (Base)", "Ensemble + ISO RAG (Ours)")
    ]

    print("=" * 100)
    print(" STRICT APPLE-TO-APPLE CONTROLLED COMPARISON RESULTS (BASE VS BASE + ISO RAG)")
    print("=" * 100)
    print(f"{'Algorithm Family':<25} | {'Base Acc (%)':<12} | {'+ ISO RAG Acc (%)':<16} | {'Gain Delta':<10} | {'t-stat':<8} | {'p-value':<12} | {'Significance':<12}")
    print("-" * 105)

    pair_summary = {}
    for base_name, rag_name in pairs:
        if base_name not in data or rag_name not in data:
            continue

        b_acc = data[base_name]["accuracy_mean"]
        r_acc = data[rag_name]["accuracy_mean"]
        delta_acc = r_acc - b_acc

        b_raw_f1 = np.array(data[base_name]["raw_metrics"]["macro_f1"])
        r_raw_f1 = np.array(data[rag_name]["raw_metrics"]["macro_f1"])

        t_stat, t_pval = stats.ttest_rel(r_raw_f1, b_raw_f1)
        try:
            w_stat, w_pval = stats.wilcoxon(r_raw_f1, b_raw_f1)
        except Exception:
            w_pval = 1.0

        is_sig = "p < 0.01 ***" if t_pval < 0.01 else ("p < 0.05 **" if t_pval < 0.05 else "N.S.")

        pair_summary[base_name] = {
            "base_acc": b_acc,
            "rag_acc": r_acc,
            "delta_acc": delta_acc,
            "t_stat": float(t_stat),
            "t_pvalue": float(t_pval),
            "wilcoxon_pvalue": float(w_pval),
            "significance": is_sig
        }

        diff_str = f"+{delta_acc:.2f}%" if delta_acc >= 0 else f"{delta_acc:.2f}%"
        algo_family = base_name.replace(" (Base)", "")
        print(f"{algo_family:<25} | {b_acc:11.2f}% | {r_acc:15.2f}% | {diff_str:<10} | {t_stat:7.4f} | {t_pval:11.4e} | {is_sig:<12}")

    print("-" * 105)

    # Output LaTeX Tables
    results_dir = r"C:\Users\umert\Downloads\FNFC_Angle1_ISO25010\results"
    
    # 1. Apple-to-Apple LaTeX Table
    latex_table_path = os.path.join(results_dir, "table_apple_to_apple.tex")
    with open(latex_table_path, "w", encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Controlled Apple-to-Apple Evaluation: Net Gain ($\\Delta$) of ISO 25010 RAG Augmentation Across Algorithms}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("\\textbf{Algorithm} & \\textbf{Base Acc (\\%)} & \\textbf{+ ISO RAG Acc (\\%)} & \\textbf{Accuracy } $\\Delta$ & \\textbf{Base Macro F1} & \\textbf{+ ISO RAG F1} & \\textbf{Significance} \\\\\n\\midrule\n")
        
        for base_name, rag_name in pairs:
            if base_name in data and rag_name in data:
                algo_name = base_name.replace(" (Base)", "")
                b_acc = data[base_name]["accuracy_mean"]
                r_acc = data[rag_name]["accuracy_mean"]
                d_acc = r_acc - b_acc
                
                b_f1 = data[base_name]["macro_f1_mean"]
                r_f1 = data[rag_name]["macro_f1_mean"]
                
                res = pair_summary[base_name]
                sig_str = "$p < 0.01$***" if res["t_pvalue"] < 0.01 else ("$p < 0.05$**" if res["t_pvalue"] < 0.05 else "N.S.")
                
                f.write(f"{algo_name} & {b_acc:.2f} & \\textbf{{{r_acc:.2f}}} & +{d_acc:.2f}\\% & {b_f1:.2f} & \\textbf{{{r_f1:.2f}}} & {sig_str} \\\\\n")
                
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\n[SUCCESS] Apple-to-Apple LaTeX paper table generated: {latex_table_path}")
    return pair_summary

if __name__ == "__main__":
    perform_statistical_tests()
