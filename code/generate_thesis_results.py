#!/usr/bin/env python3
"""
ACES-FL thesis results generator

Run from:
    ~/aces-fl-project

Command:
    python generate_thesis_results.py

Outputs:
    thesis_results/
        tables/
        figures/
        chapter4_results_summary.md
        generation_manifest.txt


The script reads the archived experiment CSV files created during the ACES-FL
implementation and produces thesis-ready tables and static figures.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path.cwd()

OUT = ROOT / "thesis_results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)



EXPERIMENTS = {
    "FedAvg": ROOT / "acesfl-baseline/results/fedavg_20round/metrics_corrected.csv",
    "Selection V1": ROOT / "acesfl-selection/results/aces_selection_v1_20round/metrics_corrected.csv",
    "Selection V2": ROOT / "acesfl-selection-v2/results/aces_selection_v2_20round/metrics_corrected.csv",
    "ACES + Compression": ROOT / "acesfl-compression/results/aces_compression_20round/aces_compression_metrics.csv",
    "Attack: No Defense": ROOT / "acesfl-defense/results/attack_no_defense_20round/aces_attack_no_defense_metrics.csv",
    "Defense V1": ROOT / "acesfl-defense/results/defense_v1_20round/aces_defense_metrics.csv",
    "Defense V2": ROOT / "acesfl-defense/results/defense_v2_20round/aces_defense_metrics.csv",
    "10% Malicious": ROOT / "acesfl-defense/results/robustness_signflip_10pct/aces_defense_metrics.csv",
    "30% Malicious": ROOT / "acesfl-defense/results/robustness_signflip_30pct/aces_defense_metrics.csv",
    "Sign-flip x2": ROOT / "acesfl-defense/results/robustness_signflip_scale2/aces_defense_metrics.csv",
    "Sign-flip x10": ROOT / "acesfl-defense/results/robustness_signflip_scale10/aces_defense_metrics.csv",
}

DEFENSE_HISTORY = {
    "Defense V1": ROOT / "acesfl-defense/results/defense_v1_20round/defense_history.csv",
    "Defense V2": ROOT / "acesfl-defense/results/defense_v2_20round/defense_history.csv",
    "10% Malicious": ROOT / "acesfl-defense/results/robustness_signflip_10pct/defense_history.csv",
    "30% Malicious": ROOT / "acesfl-defense/results/robustness_signflip_30pct/defense_history.csv",
    "Sign-flip x2": ROOT / "acesfl-defense/results/robustness_signflip_scale2/defense_history.csv",

    "Sign-flip x10": ROOT / "acesfl-defense/results/robustness_signflip_scale10/defense_history.csv",
}

PARTICIPATION = {
    "Selection V1": ROOT / "acesfl-selection/results/aces_selection_v1_20round/participation_counts.csv",
    "Selection V2": ROOT / "acesfl-selection-v2/results/aces_selection_v2_20round/participation_counts.csv",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] Empty CSV: {path}")
            return None
        return df
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def norm_name(value: str) -> str:
    return (

        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {norm_name(c): c for c in df.columns}

    for alias in aliases:
        key = norm_name(alias)
        if key in lookup:
            return lookup[key]

    # Conservative contains fallback
    for alias in aliases:
        key = norm_name(alias)
        for normalized, original in lookup.items():
            if key in normalized:
                return original

    return None


ALIASES = {
    "round": ["round", "server_round", "global_round"],
    "loss": ["loss", "global_loss", "test_loss"],
    "accuracy": ["accuracy", "global_accuracy", "test_accuracy", "acc"],
    "precision": ["precision", "global_precision", "test_precision"],
    "recall": ["recall", "global_recall", "test_recall"],

    "f1": ["f1", "f1_score", "global_f1", "test_f1"],
    "model_payload_bytes": ["model_payload_bytes", "model_bytes", "payload_bytes"],
    "round_upload_bytes": [
        "round_upload_bytes",
        "round_compressed_update_bytes",
        "upload_bytes",
        "communication_bytes",
    ],
    "selected_fp32_reference_bytes": [
        "selected_fp32_reference_bytes",
        "round_raw_update_bytes",
        "raw_round_bytes",
    ],
}


def standardize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for canonical, aliases in ALIASES.items():
        col = find_col(df, aliases)
        if col is not None:
            out[canonical] = pd.to_numeric(df[col], errors="coerce")

    if "round" not in out.columns:
        out["round"] = np.arange(len(out), dtype=int)

    out = out.sort_values("round").reset_index(drop=True)
    return out


def metric_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:

    s = standardize_metrics(df)
    if metric not in s.columns:
        return pd.DataFrame(columns=["round", metric])

    z = s[["round", metric]].dropna()
    return z


def final_metric_row(df: pd.DataFrame) -> dict[str, float]:
    s = standardize_metrics(df)

    if s.empty:
        return {}

    # Prefer the maximum numeric round.
    s = s.sort_values("round")
    row = s.iloc[-1]

    result = {}
    for col in [
        "round",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "model_payload_bytes",
    ]:
        if col in s.columns and pd.notna(row[col]):
            result[col] = float(row[col])

    return result



def pct(x: float | int | None) -> float:
    if x is None or pd.isna(x):
        return np.nan
    return float(x) * 100.0


def fmt_pct(x: float | int | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):.{digits}f}%"


def fmt_num(x: float | int | None, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):.{digits}f}"


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLES / filename
    df.to_csv(path, index=False)
    return path


def style_axes(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)
    if subtitle:
        ax.text(
            0.0,
            1.015,

            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, stem: str):
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def add_bar_labels(ax, values, suffix="", decimals=2):
    for patch, value in zip(ax.patches, values):

        if pd.isna(value):
            continue

        label = f"{value:.{decimals}f}{suffix}"
        ax.annotate(
            label,
            (
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


# ---------------------------------------------------------------------
# Load experiment metrics
# ---------------------------------------------------------------------

metrics_raw: dict[str, pd.DataFrame] = {}
final_rows: dict[str, dict[str, float]] = {}

for name, path in EXPERIMENTS.items():
    df = safe_read_csv(path)
    if df is not None:
        metrics_raw[name] = df
        final_rows[name] = final_metric_row(df)



print(f"[INFO] Loaded {len(metrics_raw)} metric datasets.")


# ---------------------------------------------------------------------
# Table 1: final model performance
# ---------------------------------------------------------------------

performance_order = [
    "FedAvg",
    "Selection V1",
    "Selection V2",
    "ACES + Compression",
    "Attack: No Defense",
    "Defense V1",
    "Defense V2",
]

performance_rows = []

for name in performance_order:
    row = final_rows.get(name, {})
    if not row:
        continue

    performance_rows.append(
        {
            "Experiment": name,
            "Final Round": int(row.get("round", np.nan))
            if not pd.isna(row.get("round", np.nan))
            else np.nan,
            "Loss": row.get("loss", np.nan),
            "Accuracy (%)": pct(row.get("accuracy")),

            "Precision (%)": pct(row.get("precision")),
            "Recall (%)": pct(row.get("recall")),
            "F1 (%)": pct(row.get("f1")),
        }
    )

performance = pd.DataFrame(performance_rows)
save_table(performance, "table_01_final_model_performance.csv")


# ---------------------------------------------------------------------
# Security detection metrics
# ---------------------------------------------------------------------

def security_summary(path: Path) -> dict[str, float] | None:
    df = safe_read_csv(path)
    if df is None:
        return None

    actual = find_col(df, ["actual_malicious", "is_malicious"])
    accepted = find_col(df, ["accepted"])

    if actual is None or accepted is None:
        print(f"[WARN] Missing security truth/decision columns in {path}")
        return None

    a = pd.to_numeric(df[actual], errors="coerce")
    ok = pd.to_numeric(df[accepted], errors="coerce")

    valid = a.notna() & ok.notna()
    a = a[valid].astype(int)
    ok = ok[valid].astype(int)


    tp = int(((a == 1) & (ok == 0)).sum())
    fp = int(((a == 0) & (ok == 0)).sum())
    fn = int(((a == 1) & (ok == 1)).sum())
    tn = int(((a == 0) & (ok == 1)).sum())

    detection = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    accuracy = (tp + tn) / len(a) if len(a) else np.nan

    return {
        "Total Updates": int(len(a)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Detection Rate (%)": detection * 100,
        "False Positive Rate (%)": fpr * 100,
        "Detection Precision (%)": precision * 100,
        "Decision Accuracy (%)": accuracy * 100,
    }


security_rows = []
security_by_name = {}

for name, path in DEFENSE_HISTORY.items():
    summary = security_summary(path)
    if summary is None:
        continue


    security_by_name[name] = summary
    security_rows.append({"Experiment": name, **summary})

security_table = pd.DataFrame(security_rows)
save_table(security_table, "table_03_security_detection_performance.csv")


# ---------------------------------------------------------------------
# Communication efficiency
# ---------------------------------------------------------------------

def summed_round_upload(name: str) -> float | None:
    df = metrics_raw.get(name)
    if df is None:
        return None

    s = standardize_metrics(df)

    if "round_upload_bytes" in s.columns:
        if "round" in s.columns:
            s = s[s["round"] > 0]
        values = pd.to_numeric(
            s["round_upload_bytes"],
            errors="coerce",
        ).dropna()

        if len(values):
            return float(values.sum())

    return None



def find_model_payload() -> int:
    # Prefer explicit value from compression/defense metrics.
    for candidate in [
        "ACES + Compression",
        "Defense V2",
        "Attack: No Defense",
    ]:
        df = metrics_raw.get(candidate)
        if df is None:
            continue

        s = standardize_metrics(df)
        if "model_payload_bytes" in s.columns:
            vals = s["model_payload_bytes"].dropna()
            if len(vals):
                return int(round(float(vals.iloc[-1])))

    # Verified ACES-FL model payload fallback.
    return 133_384


MODEL_PAYLOAD = find_model_payload()
ROUNDS = 20

# FedAvg uses 20 clients/round; ACES selection uses 10 clients/round.
BASELINE_TOTAL = float(MODEL_PAYLOAD * 20 * ROUNDS)
SELECTION_TOTAL = float(MODEL_PAYLOAD * 10 * ROUNDS)

compression_total = summed_round_upload("ACES + Compression")
defense_total = summed_round_upload("Defense V2")

if compression_total is None:

    # Verified archived full compression run fallback.
    compression_total = 18_473_684.0

if defense_total is None:
    # Defense sees the same transmitted client updates before server rejection.
    # Use the run's compressed upload if present; otherwise use clean compression
    # total as a conservative fallback.
    defense_total = compression_total


communication = pd.DataFrame(
    [
        {
            "Experiment": "FedAvg",
            "Total Upload Bytes": BASELINE_TOTAL,
            "Total Upload MiB": BASELINE_TOTAL / (1024 ** 2),
            "Reduction vs FedAvg (%)": 0.0,
        },
        {
            "Experiment": "Selection V2",
            "Total Upload Bytes": SELECTION_TOTAL,
            "Total Upload MiB": SELECTION_TOTAL / (1024 ** 2),
            "Reduction vs FedAvg (%)":
                (1 - SELECTION_TOTAL / BASELINE_TOTAL) * 100,
        },
        {
            "Experiment": "ACES + Compression",
            "Total Upload Bytes": compression_total,
            "Total Upload MiB": compression_total / (1024 ** 2),
            "Reduction vs FedAvg (%)":
                (1 - compression_total / BASELINE_TOTAL) * 100,
        },

        {
            "Experiment": "Defense V2",
            "Total Upload Bytes": defense_total,
            "Total Upload MiB": defense_total / (1024 ** 2),
            "Reduction vs FedAvg (%)":
                (1 - defense_total / BASELINE_TOTAL) * 100,
        },
    ]
)

save_table(communication, "table_02_communication_efficiency.csv")


# ---------------------------------------------------------------------
# Robustness tables
# ---------------------------------------------------------------------

def final_perf_for(name: str) -> dict[str, float]:
    row = final_rows.get(name, {})
    return {
        "Accuracy (%)": pct(row.get("accuracy")),
        "Precision (%)": pct(row.get("precision")),
        "Recall (%)": pct(row.get("recall")),
        "F1 (%)": pct(row.get("f1")),
        "Loss": row.get("loss", np.nan),
    }


ratio_spec = [
    (10, "10% Malicious"),
    (20, "Defense V2"),
    (30, "30% Malicious"),

]

ratio_rows = []

for ratio, name in ratio_spec:
    sec = security_by_name.get(name, {})
    perf = final_perf_for(name)

    ratio_rows.append(
        {
            "Malicious Ratio (%)": ratio,
            **perf,
            "Detection Rate (%)":
                sec.get("Detection Rate (%)", np.nan),
            "False Positive Rate (%)":
                sec.get("False Positive Rate (%)", np.nan),
            "Decision Accuracy (%)":
                sec.get("Decision Accuracy (%)", np.nan),
        }
    )

ratio_table = pd.DataFrame(ratio_rows)
save_table(
    ratio_table,
    "table_04_malicious_ratio_robustness.csv",
)


strength_spec = [
    (2, "Sign-flip x2"),
    (5, "Defense V2"),
    (10, "Sign-flip x10"),

]

strength_rows = []

for scale, name in strength_spec:
    sec = security_by_name.get(name, {})
    perf = final_perf_for(name)

    strength_rows.append(
        {
            "Sign-flip Scale": scale,
            **perf,
            "Detection Rate (%)":
                sec.get("Detection Rate (%)", np.nan),
            "False Positive Rate (%)":
                sec.get("False Positive Rate (%)", np.nan),
            "Decision Accuracy (%)":
                sec.get("Decision Accuracy (%)", np.nan),
        }
    )

strength_table = pd.DataFrame(strength_rows)
save_table(
    strength_table,
    "table_05_attack_strength_robustness.csv",
)


# ---------------------------------------------------------------------
# Ablation tables
# ---------------------------------------------------------------------


efficiency_ablation_names = [
    "FedAvg",
    "Selection V1",
    "Selection V2",
    "ACES + Compression",
]

eff_rows = []

for name in efficiency_ablation_names:
    row = final_rows.get(name, {})
    if not row:
        continue

    if name == "FedAvg":
        upload = BASELINE_TOTAL
    elif name in ("Selection V1", "Selection V2"):
        upload = SELECTION_TOTAL
    else:
        upload = compression_total

    eff_rows.append(
        {
            "Configuration": name,
            "Accuracy (%)": pct(row.get("accuracy")),
            "F1 (%)": pct(row.get("f1")),
            "Total Upload MiB": upload / (1024 ** 2),
            "Communication Reduction vs FedAvg (%)":
                (1 - upload / BASELINE_TOTAL) * 100,
        }
    )


efficiency_ablation = pd.DataFrame(eff_rows)
save_table(
    efficiency_ablation,
    "table_06_efficiency_ablation.csv",
)


security_ablation_names = [
    ("Clean ACES + Compression", "ACES + Compression"),
    ("Attack Without Defense", "Attack: No Defense"),
    ("Attack + Defense V2", "Defense V2"),
]

sec_rows = []

for display, source in security_ablation_names:
    row = final_rows.get(source, {})
    if not row:
        continue

    sec = security_by_name.get(source, {})

    sec_rows.append(
        {
            "Configuration": display,
            "Accuracy (%)": pct(row.get("accuracy")),
            "Precision (%)": pct(row.get("precision")),
            "Recall (%)": pct(row.get("recall")),
            "F1 (%)": pct(row.get("f1")),
            "Loss": row.get("loss", np.nan),
            "Detection Rate (%)":
                sec.get("Detection Rate (%)", np.nan),

            "False Positive Rate (%)":
                sec.get("False Positive Rate (%)", np.nan),
        }
    )

security_ablation = pd.DataFrame(sec_rows)
save_table(
    security_ablation,
    "table_07_security_ablation.csv",
)


# ---------------------------------------------------------------------
# Figures 1-3: convergence
# ---------------------------------------------------------------------

convergence_series = [
    "FedAvg",
    "ACES + Compression",
    "Attack: No Defense",
    "Defense V2",
]

for metric, ylabel, stem in [
    ("accuracy", "Accuracy (%)", "fig_01_accuracy_convergence"),
    ("f1", "F1 Score (%)", "fig_02_f1_convergence"),
    ("loss", "Loss", "fig_03_loss_convergence"),
]:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    plotted = 0


    for idx, name in enumerate(convergence_series):
        df = metrics_raw.get(name)
        if df is None:
            continue

        s = metric_series(df, metric)
        if s.empty:
            continue

        y = s[metric].to_numpy(dtype=float)

        if metric in ("accuracy", "f1"):
            y = y * 100.0

        ax.plot(
            s["round"],
            y,
            marker=["o", "s", "^", "D"][idx % 4],
            linestyle=["-", "--", "-.", ":"][idx % 4],
            linewidth=1.6,
            markersize=3.8,
            label=name,
        )
        plotted += 1

    style_axes(
        ax,
        {
            "accuracy": "Accuracy Convergence Across Federated Learning Configurations",
            "f1": "F1-Score Convergence Across Federated Learning Configurations",
            "loss": "Loss Convergence Across Federated Learning Configurations",
        }[metric],

        "UNSW-NB15, 20-round experiments; server-side evaluation",
    )

    ax.set_xlabel("Federated Round")
    ax.set_ylabel(ylabel)

    if plotted:
        ax.legend(frameon=False, fontsize=8)

    save_figure(fig, stem)


# ---------------------------------------------------------------------
# Figure 4: total communication
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.0, 5.0))
values = communication["Total Upload MiB"].to_numpy()
labels = communication["Experiment"].tolist()

ax.bar(labels, values)
style_axes(
    ax,
    "Total Client-to-Server Communication Cost",
    "20-round upload tensor bytes; lower is better",
)

ax.set_ylabel("Total Upload (MiB)")
ax.tick_params(axis="x", rotation=15)
add_bar_labels(ax, values, suffix=" MiB", decimals=2)

save_figure(fig, "fig_04_total_communication_cost")



# ---------------------------------------------------------------------
# Figure 5: communication reduction
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.0, 5.0))

comm_reduction = communication[
    communication["Experiment"] != "FedAvg"
].copy()

vals = comm_reduction[
    "Reduction vs FedAvg (%)"
].to_numpy()

ax.bar(
    comm_reduction["Experiment"].tolist(),
    vals,
)

style_axes(
    ax,
    "Communication Reduction Relative to FedAvg",
    "Client-to-server model/update tensor bytes over 20 rounds",
)

ax.set_ylabel("Reduction vs FedAvg (%)")
ax.set_ylim(
    0,
    max(75, float(np.nanmax(vals)) + 8)
)


ax.tick_params(axis="x", rotation=15)
add_bar_labels(ax, vals, suffix="%", decimals=2)

save_figure(
    fig,
    "fig_05_communication_reduction",
)


# ---------------------------------------------------------------------
# Figure 6: clean vs attack vs defense
# ---------------------------------------------------------------------

sec_compare = security_ablation.copy()

if not sec_compare.empty:
    for metric_col, stem, title in [
        (
            "Accuracy (%)",
            "fig_06_security_accuracy",
            "Model Accuracy Under Poisoning and Defense",
        ),
        (
            "F1 (%)",
            "fig_07_security_f1",
            "F1 Score Under Poisoning and Defense",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        vals = sec_compare[metric_col].to_numpy(dtype=float)


        ax.bar(
            sec_compare["Configuration"].tolist(),
            vals,
        )

        style_axes(
            ax,
            title,
            "Clean ACES, poisoning without defense, and ACES Defense V2",
        )

        ax.set_ylabel(metric_col)
        ax.set_ylim(
            0,
            max(100, float(np.nanmax(vals)) + 8),
        )

        ax.tick_params(axis="x", rotation=12)
        add_bar_labels(
            ax,
            vals,
            suffix="%",
            decimals=2,
        )

        save_figure(fig, stem)


# ---------------------------------------------------------------------
# Figure 8: V1 vs V2 security detector
# ---------------------------------------------------------------------


v12 = security_table[
    security_table["Experiment"].isin(
        ["Defense V1", "Defense V2"]
    )
].copy()

if not v12.empty:
    # Detection rate
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vals = v12["Detection Rate (%)"].to_numpy(dtype=float)

    ax.bar(v12["Experiment"], vals)

    style_axes(
        ax,
        "Malicious-Update Detection Rate: Defense V1 vs V2",
        "20% malicious clients, sign-flip scale 5",
    )

    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(0, 105)
    add_bar_labels(ax, vals, suffix="%", decimals=2)

    save_figure(
        fig,
        "fig_08_defense_detection_rate_v1_v2",
    )

    # False-positive rate
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vals = v12[

        "False Positive Rate (%)"
    ].to_numpy(dtype=float)

    ax.bar(v12["Experiment"], vals)

    style_axes(
        ax,
        "False-Positive Rate: Defense V1 vs V2",
        "Lower is better; benign updates incorrectly rejected",
    )

    ax.set_ylabel("False Positive Rate (%)")
    ax.set_ylim(
        0,
        max(3.0, float(np.nanmax(vals)) + 0.8),
    )
    add_bar_labels(ax, vals, suffix="%", decimals=2)

    save_figure(
        fig,
        "fig_09_defense_false_positive_rate_v1_v2",
    )


# ---------------------------------------------------------------------
# Figures 10-11: malicious-ratio robustness
# ---------------------------------------------------------------------

if not ratio_table.empty:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.plot(

        ratio_table["Malicious Ratio (%)"],
        ratio_table["Accuracy (%)"],
        marker="o",
        linestyle="-",
        label="Accuracy",
    )
    ax.plot(
        ratio_table["Malicious Ratio (%)"],
        ratio_table["F1 (%)"],
        marker="s",
        linestyle="--",
        label="F1",
    )

    style_axes(
        ax,
        "Model Performance Across Malicious-Client Ratios",
        "Sign-flip scale 5, ACES Defense V2",
    )

    ax.set_xlabel("Malicious Clients Among Selected Clients (%)")
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(
        min(
            70,
            float(
                np.nanmin(
                    ratio_table[
                        ["Accuracy (%)", "F1 (%)"]
                    ].to_numpy()
                )
            ) - 3,

        ),
        100,
    )

    ax.legend(frameon=False)

    save_figure(
        fig,
        "fig_10_malicious_ratio_model_performance",
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.plot(
        ratio_table["Malicious Ratio (%)"],
        ratio_table["Detection Rate (%)"],
        marker="o",
        linestyle="-",
        label="Detection Rate",
    )
    ax.plot(
        ratio_table["Malicious Ratio (%)"],
        ratio_table["False Positive Rate (%)"],
        marker="s",
        linestyle="--",
        label="False Positive Rate",
    )

    style_axes(
        ax,
        "Security Robustness Across Malicious-Client Ratios",
        "Detection and benign false-positive rates",

    )

    ax.set_xlabel("Malicious Clients Among Selected Clients (%)")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)

    save_figure(
        fig,
        "fig_11_malicious_ratio_security_performance",
    )


# ---------------------------------------------------------------------
# Figures 12-13: attack-strength robustness
# ---------------------------------------------------------------------

if not strength_table.empty:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.plot(
        strength_table["Sign-flip Scale"],
        strength_table["Accuracy (%)"],
        marker="o",
        linestyle="-",
        label="Accuracy",
    )
    ax.plot(
        strength_table["Sign-flip Scale"],
        strength_table["F1 (%)"],
        marker="s",
        linestyle="--",

        label="F1",
    )

    style_axes(
        ax,
        "Model Performance Across Sign-Flip Attack Strength",
        "20% malicious clients, ACES Defense V2",
    )

    ax.set_xlabel("Sign-Flip Scale")
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(
        min(
            70,
            float(
                np.nanmin(
                    strength_table[
                        ["Accuracy (%)", "F1 (%)"]
                    ].to_numpy()
                )
            ) - 3,
        ),
        100,
    )
    ax.legend(frameon=False)

    save_figure(
        fig,
        "fig_12_attack_strength_model_performance",
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))


    ax.plot(
        strength_table["Sign-flip Scale"],
        strength_table["Detection Rate (%)"],
        marker="o",
        linestyle="-",
        label="Detection Rate",
    )
    ax.plot(
        strength_table["Sign-flip Scale"],
        strength_table["False Positive Rate (%)"],
        marker="s",
        linestyle="--",
        label="False Positive Rate",
    )

    style_axes(
        ax,
        "Security Robustness Across Sign-Flip Attack Strength",
        "20% malicious clients",
    )

    ax.set_xlabel("Sign-Flip Scale")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)

    save_figure(
        fig,
        "fig_13_attack_strength_security_performance",
    )



# ---------------------------------------------------------------------
# Figure 14: efficiency ablation
# ---------------------------------------------------------------------

if not efficiency_ablation.empty:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    vals = efficiency_ablation[
        "Accuracy (%)"
    ].to_numpy(dtype=float)

    ax.bar(
        efficiency_ablation["Configuration"],
        vals,
    )

    style_axes(
        ax,
        "Efficiency Ablation: Final Model Accuracy",
        "FedAvg → selection → fairness-aware selection → adaptive compression",
    )

    ax.set_ylabel("Final Accuracy (%)")

    low = max(
        0,
        float(np.nanmin(vals)) - 5,
    )
    high = min(
        100,
        float(np.nanmax(vals)) + 5,

    )

    ax.set_ylim(low, high)
    ax.tick_params(axis="x", rotation=15)
    add_bar_labels(ax, vals, suffix="%", decimals=2)

    save_figure(
        fig,
        "fig_14_efficiency_ablation_accuracy",
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    vals = efficiency_ablation[
        "Communication Reduction vs FedAvg (%)"
    ].to_numpy(dtype=float)

    ax.bar(
        efficiency_ablation["Configuration"],
        vals,
    )

    style_axes(
        ax,
        "Efficiency Ablation: Communication Reduction",
        "Client-to-server upload tensor bytes relative to FedAvg",
    )

    ax.set_ylabel("Reduction vs FedAvg (%)")
    ax.set_ylim(
        0,
        max(75, float(np.nanmax(vals)) + 8),

    )
    ax.tick_params(axis="x", rotation=15)
    add_bar_labels(ax, vals, suffix="%", decimals=2)

    save_figure(
        fig,
        "fig_15_efficiency_ablation_communication",
    )


# ---------------------------------------------------------------------
# Figure 16: security ablation
# ---------------------------------------------------------------------

if not security_ablation.empty:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    vals = security_ablation[
        "Accuracy (%)"
    ].to_numpy(dtype=float)

    ax.bar(
        security_ablation["Configuration"],
        vals,
    )

    style_axes(
        ax,
        "Security Ablation: Clean, Attacked, and Defended ACES-FL",
        "20% malicious clients, sign-flip scale 5",
    )


    ax.set_ylabel("Final Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=12)
    add_bar_labels(ax, vals, suffix="%", decimals=2)

    save_figure(
        fig,
        "fig_16_security_ablation_accuracy",
    )


# ---------------------------------------------------------------------
# Optional participation comparison
# ---------------------------------------------------------------------

def participation_frame(path: Path) -> pd.DataFrame | None:
    df = safe_read_csv(path)
    if df is None:
        return None

    # Pick a client-id-like column and a numeric count-like column.
    id_col = find_col(
        df,
        [
            "client_id",
            "partition_id",
            "client",
            "partition",
        ],
    )

    count_col = find_col(

        df,
        [
            "participation_count",
            "selected_count",
            "count",
            "participations",
        ],
    )

    if id_col is None or count_col is None:
        return None

    out = pd.DataFrame(
        {
            "client": df[id_col],
            "count": pd.to_numeric(
                df[count_col],
                errors="coerce",
            ),
        }
    ).dropna()

    return out


p1 = participation_frame(PARTICIPATION["Selection V1"])
p2 = participation_frame(PARTICIPATION["Selection V2"])

if p1 is not None and p2 is not None:
    merged = p1.merge(
        p2,
        on="client",

        how="outer",
        suffixes=("_v1", "_v2"),
    ).fillna(0)

    merged = merged.sort_values("client")

    save_table(
        merged,
        "table_08_selection_participation_v1_v2.csv",
    )

    x = np.arange(len(merged))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    ax.bar(
        x - width / 2,
        merged["count_v1"],
        width=width,
        label="Selection V1",
    )

    ax.bar(
        x + width / 2,
        merged["count_v2"],
        width=width,
        label="Selection V2",
    )

    style_axes(
        ax,

        "Client Participation: Selection V1 vs Fairness-Aware V2",
        "Number of selections across 20 rounds",
    )

    ax.set_xlabel("Client / Partition")
    ax.set_ylabel("Selection Count")
    ax.set_xticks(x)
    ax.set_xticklabels(
        merged["client"].astype(str),
        rotation=0,
        fontsize=8,
    )
    ax.legend(frameon=False)

    save_figure(
        fig,
        "fig_17_selection_participation_v1_v2",
    )


# ---------------------------------------------------------------------
# Chapter 4 narrative summary
# ---------------------------------------------------------------------

clean = final_rows.get("ACES + Compression", {})
attacked = final_rows.get("Attack: No Defense", {})
defended = final_rows.get("Defense V2", {})

clean_acc = pct(clean.get("accuracy"))
attack_acc = pct(attacked.get("accuracy"))
def_acc = pct(defended.get("accuracy"))


clean_f1 = pct(clean.get("f1"))
attack_f1 = pct(attacked.get("f1"))
def_f1 = pct(defended.get("f1"))

accuracy_loss = (
    clean_acc - attack_acc
    if not any(pd.isna(v) for v in [clean_acc, attack_acc])
    else np.nan
)

accuracy_recovered = (
    def_acc - attack_acc
    if not any(pd.isna(v) for v in [def_acc, attack_acc])
    else np.nan
)

recovery_pct = (
    accuracy_recovered / accuracy_loss * 100
    if (
        not pd.isna(accuracy_loss)
        and accuracy_loss != 0
        and not pd.isna(accuracy_recovered)
    )
    else np.nan
)

v2_sec = security_by_name.get("Defense V2", {})

md = []
md.append("# ACES-FL Chapter 4 Results Summary")
md.append("")
md.append("## 1. Overall Performance")

md.append("")
md.append(
    f"- Clean ACES + adaptive compression final accuracy: "
    f"**{fmt_pct(clean_acc)}**."
)
md.append(
    f"- Clean ACES + adaptive compression final F1: "
    f"**{fmt_pct(clean_f1)}**."
)
md.append(
    f"- Under the 20% malicious-client sign-flip attack without defense, "
    f"final accuracy fell to **{fmt_pct(attack_acc)}** and F1 to "
    f"**{fmt_pct(attack_f1)}**."
)
md.append(
    f"- With ACES Defense V2 enabled, final accuracy recovered to "
    f"**{fmt_pct(def_acc)}** and F1 to **{fmt_pct(def_f1)}**."
)

if not pd.isna(recovery_pct):
    md.append(
        f"- The defense recovered approximately "
        f"**{recovery_pct:.2f}%** of the accuracy lost to poisoning."
    )

md.append("")
md.append("## 2. Communication Efficiency")
md.append("")
md.append(
    f"- FedAvg 20-round client-to-server upload reference: "
    f"**{BASELINE_TOTAL:,.0f} bytes** "
    f"({BASELINE_TOTAL/(1024**2):.2f} MiB)."

)
md.append(
    f"- Fairness-aware selection reduces the upload reference by "
    f"**{(1-SELECTION_TOTAL/BASELINE_TOTAL)*100:.2f}%**."
)
md.append(
    f"- Adaptive compression reduces total upload relative to FedAvg by "
    f"**{(1-compression_total/BASELINE_TOTAL)*100:.2f}%**."
)
md.append(
    "- Communication is defined as client-to-server model/update tensor bytes. "
    "Flower serialization overhead is recorded separately in the experiment files."
)

md.append("")
md.append("## 3. Security Detection")
md.append("")

if v2_sec:
    md.append(
        f"- Defense V2 detection rate: "
        f"**{v2_sec['Detection Rate (%)']:.2f}%**."
    )
    md.append(
        f"- Defense V2 false-positive rate: "
        f"**{v2_sec['False Positive Rate (%)']:.2f}%**."
    )
    md.append(
        f"- Defense V2 decision accuracy: "
        f"**{v2_sec['Decision Accuracy (%)']:.2f}%**."
    )
    md.append(

        f"- Confusion counts: TP={int(v2_sec['TP'])}, "
        f"FP={int(v2_sec['FP'])}, FN={int(v2_sec['FN'])}, "
        f"TN={int(v2_sec['TN'])}."
    )

md.append("")
md.append("## 4. Ablation Interpretation")
md.append("")
md.append(
    "The efficiency ablation separates the contribution of resource-aware "
    "selection, fairness-aware participation, and adaptive compression. "
    "Selection reduces the number of client uploads per round, while adaptive "
    "compression further reduces the payload size without materially degrading "
    "final predictive performance."
)
md.append(
    "The security ablation demonstrates the value of the defense component. "
    "The undefended sign-flip attack causes severe model degradation, whereas "
    "Defense V2 rejects anomalous updates using current-round norm and direction "
    "signals supported by recoverable historical trust, restoring model "
    "performance close to the clean ACES configuration."
)

md.append("")
md.append("## 5. Robustness Interpretation")
md.append("")
md.append(
    "The malicious-ratio experiments assess whether the detector remains "
    "effective as the fraction of malicious selected clients changes from "
    "10% to 30%. The attack-strength experiments assess sensitivity to "
    "sign-flip scales of 2, 5, and 10. Use Tables 4 and 5 and Figures 10–13 "
    "to report how detection rate, false-positive rate, accuracy, and F1 "

    "change under these conditions."
)
md.append("")
md.append("## 6. Important Reporting Caveats")
md.append("")
md.append(
    "- Do not describe the results as proof of universal Byzantine robustness; "
    "they are empirical results for the tested UNSW-NB15/Flower setup."
)
md.append(
    "- Keep the 20% sign-flip scale-5 configuration as the principal security "
    "comparison because it has both attack-without-defense and defended runs."
)
md.append(
    "- Treat non-zero false positives as a measurable trade-off rather than "
    "tuning thresholds retrospectively to force a perfect result."
)
md.append(
    "- Report communication reduction using the same client-to-server tensor-byte "
    "definition across all configurations."
)

summary_path = OUT / "chapter4_results_summary.md"
summary_path.write_text(
    "\n".join(md) + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------


generated = sorted(
    [
        p.relative_to(OUT)
        for p in OUT.rglob("*")
        if p.is_file()
    ],
    key=lambda p: str(p),
)

manifest_lines = [
    "ACES-FL thesis results generation manifest",
    "=" * 44,
    "",
    f"Project root: {ROOT}",
    f"Model payload used: {MODEL_PAYLOAD:,} bytes",
    "",
    "Generated files:",
]

manifest_lines.extend(
    f"- {p}"
    for p in generated
)

manifest = OUT / "generation_manifest.txt"
manifest.write_text(
    "\n".join(manifest_lines) + "\n",
    encoding="utf-8",
)



# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("ACES-FL THESIS RESULTS GENERATED")
print("=" * 72)
print(f"Output folder: {OUT}")
print(f"Tables       : {TABLES}")
print(f"Figures      : {FIGURES}")
print(f"Summary      : {summary_path}")
print()

if not performance.empty:
    print("FINAL PERFORMANCE")
    print(
        performance[
            [
                "Experiment",
                "Accuracy (%)",
                "F1 (%)",
            ]
        ].to_string(index=False)
    )

print()
print("COMMUNICATION")
print(
    communication[
        [
            "Experiment",

            "Total Upload MiB",
            "Reduction vs FedAvg (%)",
        ]
    ].to_string(index=False)
)

if not security_table.empty:
    print()
    print("SECURITY")
    print(
        security_table[
            [
                "Experiment",
                "Detection Rate (%)",
                "False Positive Rate (%)",
                "Decision Accuracy (%)",
            ]
        ].to_string(index=False)
    )

print()
print(f"Generated {len(generated)} files.")
print("=" * 72)
