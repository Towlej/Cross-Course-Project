import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# MASTER STYLE (FORCES CONSISTENCY)
# =========================
BASE_FONTSIZE = 26
TITLE_SIZE = 30
LABEL_SIZE = 28
TICK_SIZE = 24
LEGEND_SIZE = 28

plt.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})


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


# =========================
# RMSE (CLEAN + ROBUST)
# =========================
def compute_rmse(values):
    """
    True RMSE over signed residuals:
    sqrt(mean(x^2)), ignoring NaNs properly.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]  # IMPORTANT FIX

    if values.size == 0:
        return np.nan

    return np.sqrt(np.mean(np.square(values)))


# =========================
# PARSE FILE
# =========================
def parse_summary_file(path: str | Path) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8")
    shot_blocks = re.split(r"\n(?=Shot:\s)", text)
    shot_blocks = [b for b in shot_blocks if b.strip().startswith("Shot:")]

    rows = []

    for i, block in enumerate(shot_blocks, start=1):

        current_error = extract_scalar(block, "current_error")
        z_error = extract_scalar(block, "Z_error")
        r_error = extract_scalar(block, "R_error")
        xpf_sigma = extract_scalar(block, "XpointFlux_error")

        gap_errors = extract_list(block, "gap_error")
        gap_errors_cm = [g * 100 for g in gap_errors] if gap_errors else []

        rows.append({
            "shot_index": i,
            "current_error": current_error,
            "Z_error": z_error,
            "R_error": r_error,
            "XpointFlux_sigma": xpf_sigma,
            "gap_errors_cm": gap_errors_cm,
        })

    df = pd.DataFrame(rows)

    # =========================
    # RAW SIGNALS ONLY (NO ARTIFICIAL RMS HERE)
    # =========================
    df["current_error"] = df["current_error"]
    df["Z_error"] = df["Z_error"]
    df["R_error"] = df["R_error"]

    for g in range(6):
        df[f"gap_{g+1}_cm"] = df["gap_errors_cm"].apply(
            lambda x: x[g] if isinstance(x, list) and len(x) > g else np.nan
        )

    return df


# =========================
# SINGLE FOLD PLOT
# =========================
def plot_single_fold(df: pd.DataFrame, fold_id: int):

    metrics = [
        ("current_error", "Current Error (%)"),
        ("Z_error", "Z Error (cm)"),
        ("R_error", "R Error (cm)"),
        ("XpointFlux_sigma", "X-point Flux σ (Wb)"),
    ]

    fig, axes = plt.subplots(10, 1, figsize=(16, 24), sharex=True)
    x = df["shot_index"]

    for ax, (m, label) in zip(axes[:4], metrics):
        ax.bar(x, df[m], color="tab:blue")
        ax.set_ylabel(label)
        ax.legend([f"RMSE={compute_rmse(df[m]):.3g}"])

    colors = plt.cm.tab10.colors

    for i in range(6):
        col = f"gap_{i+1}_cm"
        axes[4 + i].bar(x, df[col], color=colors[i])
        axes[4 + i].set_ylabel(f"Gap {i+1} (cm)")
        axes[4 + i].legend([f"RMSE={compute_rmse(df[col]):.3g}"])

    axes[-1].set_xlabel("Shot Number (1–30)")
    plt.suptitle(f"Fold {fold_id}: Raw Errors", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.show()


# =========================
# FOLD SUMMARY (TRUE RMSE ONLY HERE)
# =========================
def compute_fold_means(df: pd.DataFrame, fold_id: int):

    summary = {
        "fold": fold_id,
        "Current RMSE (%)": compute_rmse(df["current_error"]),
        "Z RMSE (cm)": compute_rmse(df["Z_error"]),
        "R RMSE (cm)": compute_rmse(df["R_error"]),
        "X-point Flux σ (Wb)": df["XpointFlux_sigma"].mean(),
    }

    for i in range(6):
        summary[f"Gap {i+1} RMSE (cm)"] = compute_rmse(df[f"gap_{i+1}_cm"])

    return summary


# =========================
# NON-GAP FOLD COMPARISON
# =========================
def plot_overall_fold_comparison_non_gaps(summary_df):

    metrics = [
        "Current RMSE (%)",
        "Z RMSE (cm)",
        "R RMSE (cm)",
        "X-point Flux σ (Wb)",
    ]

    x = np.arange(len(summary_df))
    width = 0.18

    plt.figure(figsize=(16, 7))

    for i, m in enumerate(metrics):
        plt.bar(
            x + i * width,
            summary_df[m],
            width=width,
            label=m,
            color=plt.cm.Set2(i)
        )

    plt.xticks(x + width * 1.5, [f"Fold {f}" for f in summary_df["fold"]])
    plt.ylabel("RMSE / σ")
    plt.title("Fold Comparison (Non-Gap RMSE Metrics)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =========================
# GAP FOLD COMPARISON
# =========================
def plot_overall_fold_comparison_gaps(summary_df):

    metrics = [f"Gap {i+1} RMSE (cm)" for i in range(6)]

    x = np.arange(len(summary_df))
    width = 0.12

    plt.figure(figsize=(18, 7))

    for i, m in enumerate(metrics):
        plt.bar(
            x + i * width,
            summary_df[m],
            width=width,
            label=m,
            color=plt.cm.tab10(i)
        )

    plt.xticks(x + width * 2.5, [f"Fold {f}" for f in summary_df["fold"]])
    plt.ylabel("RMSE (cm)")
    plt.title("Fold Comparison (6 Gap RMSE Metrics)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right", fontsize=22)

    plt.tight_layout()
    plt.show()


# =========================
# GRAND MEAN
# =========================
def plot_grand_mean_summary(summary_df):

    metrics = [
        "Current RMSE (%)",
        "Z RMSE (cm)",
        "R RMSE (cm)",
        "X-point Flux σ (Wb)",
    ] + [f"Gap {i+1} RMSE (cm)" for i in range(6)]

    grand = {m: summary_df[m].mean() for m in metrics}

    plt.figure(figsize=(18, 7))

    plt.bar(
        list(grand.keys()),
        list(grand.values()),
        color=plt.cm.tab20.colors[:len(grand)]
    )

    plt.ylabel("RMSE / σ")
    plt.title("Grand Mean Across All 180 Shots")
    plt.xticks(rotation=25)
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================
# GROUP COMPARISON
# =========================
def plot_grouped_fold_comparison(summary_df):

    group_a = summary_df[summary_df["fold"].isin([1, 3])]
    group_b = summary_df[~summary_df["fold"].isin([1, 3])]

    metrics = [
        "Current RMSE (%)",
        "Z RMSE (cm)",
        "R RMSE (cm)",
        "X-point Flux σ (Wb)",
    ] + [f"Gap {i+1} RMSE (cm)" for i in range(6)]

    a_vals = [group_a[m].mean() for m in metrics]
    b_vals = [group_b[m].mean() for m in metrics]

    x = np.arange(len(metrics))

    plt.figure(figsize=(20, 7))

    plt.bar(x - 0.2, a_vals, 0.4, label="17 Normal Coil Configuration")
    plt.bar(x + 0.2, b_vals, 0.4, label="18 Normal Coil Configuration")

    plt.xticks(x, metrics, rotation=25)
    plt.ylabel("RMSE / σ")
    plt.title("17 vs 18 Normal Coil Configuration")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():

    all_summaries = []

    for f in FOLDS:

        path = Path(FILE_TEMPLATE.format(fold=f))
        if not path.exists():
            continue

        df = parse_summary_file(path)
        plot_single_fold(df, f)

        all_summaries.append(compute_fold_means(df, f))

    summary_df = pd.DataFrame(all_summaries)

    print(summary_df)

    plot_overall_fold_comparison_non_gaps(summary_df)
    plot_overall_fold_comparison_gaps(summary_df)
    plot_grand_mean_summary(summary_df)
    plot_grouped_fold_comparison(summary_df)


if __name__ == "__main__":
    main()
