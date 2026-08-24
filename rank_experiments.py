import os
import sys
import glob
import pandas as pd

# Ensure UTF-8 output encoding on Windows terminals to prevent UnicodeEncodeError
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass

def rank_experiments():
    results_dir = os.path.join("experiments", "results")
    eval_dir = os.path.join(results_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Locate main experiment summary CSV
    exp_summary_paths = [
        os.path.join(results_dir, "experiment_results.csv"),
        os.path.join(results_dir, "experiment_results_lora_dashboard_plus_latest_qlora.csv"),
    ]
    
    exp_summary_df = None
    for path in exp_summary_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty and "experiment" in df.columns:
                    exp_summary_df = df
                    break
            except Exception:
                continue

    methods = ["lora", "qlora", "qklora", "qalora"]
    ranked_rows = []

    for method in methods:
        ranking_csv = os.path.join(eval_dir, f"{method}_checkpoint_ranking.csv")
        row_data = {"method": method.upper()}
        
        # Pull aggregate training stats if present
        if exp_summary_df is not None:
            method_exp = exp_summary_df[exp_summary_df["experiment"].str.lower() == method]
            if not method_exp.empty:
                exp_row = method_exp.iloc[0]
                row_data["trainable_parameters"] = int(exp_row.get("trainable_parameters", 0))
                row_data["training_time_s"] = float(exp_row.get("training_time", 0.0))
                row_data["peak_gpu_memory_gb"] = float(exp_row.get("peak_gpu_memory", 0.0))

        # Pull detailed checkpoint evaluation metrics
        if os.path.exists(ranking_csv):
            try:
                eval_df = pd.read_csv(ranking_csv)
                if not eval_df.empty:
                    # Filter for best checkpoint if column rank exists or pick top row
                    if "rank" in eval_df.columns:
                        best_chk = eval_df.sort_values(by="rank").iloc[0]
                    else:
                        best_chk = eval_df.sort_values(by="accuracy", ascending=False).iloc[0]
                    
                    row_data["best_checkpoint"] = os.path.basename(str(best_chk.get("checkpoint", f"{method}_best.pth")))
                    row_data["test_accuracy"] = float(best_chk.get("accuracy", 0.0))
                    row_data["f1_macro"] = float(best_chk.get("f1_macro", 0.0))
                    row_data["binary_accuracy"] = float(best_chk.get("binary_accuracy", 0.0))
                    row_data["binary_f1"] = float(best_chk.get("binary_f1", 0.0))
                    row_data["binary_roc_auc"] = float(best_chk.get("binary_roc_auc", 0.0))
                    row_data["both_correct_pct"] = float(best_chk.get("both_correct_pct", 0.0))
                    row_data["checkpoint_size_mb"] = float(best_chk.get("size_mb", 0.0))
            except Exception as e:
                print(f"Warning parsing {ranking_csv}: {e}")

        # Fallback / Default filling if evaluation CSV missing
        if "test_accuracy" not in row_data or row_data["test_accuracy"] == 0.0:
            if exp_summary_df is not None:
                method_exp = exp_summary_df[exp_summary_df["experiment"].str.lower() == method]
                if not method_exp.empty:
                    exp_row = method_exp.iloc[0]
                    row_data["test_accuracy"] = float(exp_row.get("test_accuracy", 0.0))
                    row_data["f1_macro"] = float(exp_row.get("test_f1_macro", 0.0))
                    row_data["best_checkpoint"] = f"{method}_best.pth"
        
        if "test_accuracy" in row_data:
            ranked_rows.append(row_data)

    if not ranked_rows:
        print("No method evaluation data found.")
        return

    df_out = pd.DataFrame(ranked_rows)

    # Sort cross-method leaderboard by test_accuracy (desc), f1_macro (desc), peak_gpu_memory_gb (asc)
    sort_cols = [col for col in ["test_accuracy", "f1_macro", "binary_f1"] if col in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(by=sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
    
    df_out.insert(0, "overall_rank", df_out.index + 1)
    
    out_path = os.path.join(eval_dir, "cross_method_ranking.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Cross-method ranking complete! Scorecard saved to: {out_path}")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    rank_experiments()
