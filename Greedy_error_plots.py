import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILE_TEMPLATE = "test_{fold}_full_summary.txt"
FOLDS = range(6)


def clean_numpy_wrappers(text: str) -> str:
    return re.sub(r"np\.float64\(([^\)]+)\)", r"\1", text)


def extract_scalar(block: str, key: str):
    m = re.search(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)", block, re.MULTILINE)
    return float(m.group(1)) if m else np.nan


def extract_list(block: str, key: str):
    m = re.search(rf"^{re.escape(key)}:\s*(\[[^\n]*\])", block, re.MULTILINE)
    if not m:
        return []
    cleaned = clean_numpy_wrappers(m.group(1))
    return ast.literal_eval(cleaned)


def parse_summary_file(path: str | Path) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8")
    shot_blocks = re.split(r"\n(?=Shot:\s)", text)
    shot_blocks = [b for b in shot_blocks if b.strip().startswith("Shot:")]

    rows = []
    for i, block in enumerate(shot_blocks, start=1):
        shot_match = re.search(r"Shot:\s*(.+)", block)
        shot_name = shot_match.group(1).strip() if shot_match else f"shot_{i}"

        current_error = extract_scalar(block, "current_error")
        z_error = extract_scalar(block, "Z_error")
        r_error = extract_scalar(block, "R_error")
        xpf_error = extract_scalar(block, "XpointFlux_error")
        gap_errors = extract_list(block, "gap_error")

        avg_gap_error = float(np.mean(gap_errors)) if gap_errors else np.nan
        avg_gap_error_cm = avg_gap_error * 100

        rows.append(
            {
                "shot_index": i,
                "shot_name": shot_name,
                "current_error": current_error,
                "Z_error": z_error,
                "R_error": r_error,
                "XpointFlux_error": xpf_error,
                "avg_gap_error_cm": avg_gap_error_cm,
            }
        )

    df = pd.DataFrame(rows)
    df["abs_current_error"] = np.abs(df["current_error"])
    df["abs_Z_error"] = np.abs(df["Z_error"])
    df["abs_R_error"] = np.abs(df["R_error"])
    return df


def plot_single_fold(df: pd.DataFrame, fold_id: int):
    metrics = [
        ("abs_current_error", "Current Error (%)"),
        ("abs_Z_error", "Z Error (cm)"),
        ("abs_R_error", "R Error (cm)"),
        ("XpointFlux_error", "X-point Flux Error"),
        ("avg_gap_error_cm", "Wall Gap Error (cm)"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 16), sharex=True)
    x = df["shot_index"]

    for ax, (metric, label) in zip(axes, metrics):
        ax.bar(x, df[metric], width=0.8)
        ax.axhline(df[metric].mean(), linestyle="--", linewidth=1,
                   label=f"mean={df[metric].mean():.4g}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Shot Number (1-30)")
    plt.suptitle(f"Fold {fold_id}: Tokamak Coil Placement Errors Across 30 Shots", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_fold_mean_summary(df: pd.DataFrame, fold_id: int):
    summary_metrics = {
        "Current Error (%)": df["abs_current_error"].mean(),
        "Z Error (cm)": df["abs_Z_error"].mean(),
        "R Error (cm)": df["abs_R_error"].mean(),
        "X-point Flux Error": df["XpointFlux_error"].mean(),
        "Wall Gap Error (cm)": df["avg_gap_error_cm"].mean(),
    }

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    plt.figure(figsize=(12, 6))
    plt.bar(list(summary_metrics.keys()), list(summary_metrics.values()), color=colors)
    plt.ylabel("Mean Error")
    plt.title(f"Fold {fold_id}: Mean Error Summary Across 30 Shots")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_fold_means(df: pd.DataFrame, fold_id: int) -> dict:
    return {
        "fold": fold_id,
        "Current Error (%)": df["abs_current_error"].mean(),
        "Z Error (cm)": df["abs_Z_error"].mean(),
        "R Error (cm)": df["abs_R_error"].mean(),
        "X-point Flux Error": df["XpointFlux_error"].mean(),
        "Wall Gap Error (cm)": df["avg_gap_error_cm"].mean(),
    }


def plot_overall_fold_comparison(summary_df: pd.DataFrame):
    metrics = [
        "Current Error (%)",
        "Z Error (cm)",
        "R Error (cm)",
        "X-point Flux Error",
        "Wall Gap Error (cm)",
    ]

    x = np.arange(len(summary_df))
    width = 0.15

    plt.figure(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, summary_df[metric], width=width, label=metric)

    plt.xticks(x + width * 2, [f"Fold {f}" for f in summary_df["fold"]])
    plt.ylabel("Mean Error")
    plt.title("Comparison of Mean Errors Across All 6 Folds")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grand_mean_summary(summary_df: pd.DataFrame):
    metrics = [
        "Current Error (%)",
        "Z Error (cm)",
        "R Error (cm)",
        "X-point Flux Error",
        "Wall Gap Error (cm)",
    ]

    grand_means = {metric: summary_df[metric].mean() for metric in metrics}
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    plt.figure(figsize=(12, 6))
    plt.bar(list(grand_means.keys()), list(grand_means.values()), color=colors)
    plt.ylabel("Mean Error")
    plt.title("Grand Mean Error Summary Across All 180 Shots")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_grouped_fold_comparison(summary_df: pd.DataFrame):
    group_a_folds = [1, 3]
    group_b_folds = [f for f in summary_df["fold"] if f not in group_a_folds]

    group_a = summary_df[summary_df["fold"].isin(group_a_folds)]
    group_b = summary_df[summary_df["fold"].isin(group_b_folds)]

    metrics = [
        "Current Error (%)",
        "Z Error (cm)",
        "R Error (cm)",
        "X-point Flux Error",
        "Wall Gap Error (cm)",
    ]

    group_a_means = [group_a[m].mean() for m in metrics]
    group_b_means = [group_b[m].mean() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, group_a_means, width, label="Folds 1 & 3")
    plt.bar(x + width/2, group_b_means, width, label="Other 4 Folds")

    plt.xticks(x, metrics, rotation=20)
    plt.ylabel("Mean Error")
    plt.title("Comparison: Folds (1,3) vs Other 4 Folds")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    all_fold_summaries = []

    for fold_id in FOLDS:
        file_name = FILE_TEMPLATE.format(fold=fold_id)
        path = Path(file_name)

        if not path.exists():
            print(f"Skipping missing file: {file_name}")
            continue

        print(f"\nProcessing {file_name}")
        df = parse_summary_file(path)

        plot_single_fold(df, fold_id)
        plot_fold_mean_summary(df, fold_id)

        fold_summary = compute_fold_means(df, fold_id)
        all_fold_summaries.append(fold_summary)

    summary_df = pd.DataFrame(all_fold_summaries)

    print("\nFold mean comparison table:")
    print(summary_df.to_string(index=False))

    plot_overall_fold_comparison(summary_df)
    plot_grand_mean_summary(summary_df)

    plot_grouped_fold_comparison(summary_df)

    summary_df.to_csv("all_fold_mean_comparison.csv", index=False)
    print("\nSaved fold comparison summary to all_fold_mean_comparison.csv")


if __name__ == "__main__":
    main()
