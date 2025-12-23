#!/usr/bin/env python3
"""
Read evaluation_result_*.json from two best_shared directories,
aggregate cost_improvement_percent by dataset(matrix_name),
plot three charts:
1) Improvement for each dataset in directory A
2) Improvement for each dataset in directory B
3) Maximum improvement per dataset from both directories (colored by source), to compare which performs better
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

# Ensure the script can be run directly with `python3 scripts/xxx.py` (add project root to sys.path)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Note: Don't import utils.plot_utils directly, because utils/__init__.py imports torch,
# which may fail in some environments due to missing libcudnn. Load plot_utils.py directly instead.
_PLOT_UTILS_PATH = os.path.join(_PROJECT_ROOT, "utils", "plot_utils.py")
_spec = importlib.util.spec_from_file_location("_rlcg_plot_utils", _PLOT_UTILS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load plot_utils: {_PLOT_UTILS_PATH}")
_plot_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plot_utils)
configure_matplotlib_chinese = getattr(_plot_utils, "configure_matplotlib_chinese")


DEFAULT_DIR_A = "/home/bingxing2/home/scx7axu/program/rlcg/log/backup/251221/mesh3e1_tilesize64_20251221_174200/best_shared"
DEFAULT_DIR_B = "/home/bingxing2/home/scx7axu/program/rlcg/log/backup/251221/mesh2e1_tilesize64_20251221_174355/best_shared"


@dataclass(frozen=True)
class Record:
    dataset: str
    improvement_percent: float
    source_file: str


@dataclass(frozen=True)
class ConvergenceRecord:
    dataset: str
    double_precision_iterations: int
    double_precision_residual: float
    model_guided_iterations: int
    model_guided_residual: float
    source_file: str


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _extract_dataset_name(data: dict, fallback_from_filename: str) -> str:
    if isinstance(data, dict):
        name = data.get("matrix_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    # fallback: evaluation_result_xxx.json -> xxx
    base = os.path.basename(fallback_from_filename)
    if base.startswith("evaluation_result_") and base.endswith(".json"):
        return base[len("evaluation_result_") : -len(".json")]
    return os.path.splitext(base)[0]


def _extract_cost_improvement_percent(data: dict) -> Optional[float]:
    """
    Prefer comparison.cost_improvement_percent,
    fallback: some result files may use performance_improvement_percent
    """
    if not isinstance(data, dict):
        return None
    comp = data.get("comparison")
    if isinstance(comp, dict) and "cost_improvement_percent" in comp:
        v = _safe_float(comp.get("cost_improvement_percent"))
        if v is not None:
            return v
    if "cost_improvement_percent" in data:
        v = _safe_float(data.get("cost_improvement_percent"))
        if v is not None:
            return v
    if "performance_improvement_percent" in data:
        v = _safe_float(data.get("performance_improvement_percent"))
        if v is not None:
            return v
    return None


def _extract_convergence_data(data: dict) -> Optional[Tuple[int, float, int, float]]:
    """
    Extract iterations and final_residual from double_precision and model_guided sections.
    Returns: (dp_iterations, dp_residual, mg_iterations, mg_residual)
    """
    if not isinstance(data, dict):
        return None

    dp_data = data.get("double_precision", {})
    mg_data = data.get("model_guided", {})

    try:
        dp_iterations = int(dp_data.get("iterations", 0))
        dp_residual = float(dp_data.get("final_residual", 0.0))
        mg_iterations = int(mg_data.get("iterations", 0))
        mg_residual = float(mg_data.get("final_residual", 0.0))

        # Skip if any value is missing or invalid
        if dp_iterations <= 0 or mg_iterations <= 0 or dp_residual <= 0 or mg_residual <= 0:
            return None

        return (dp_iterations, dp_residual, mg_iterations, mg_residual)
    except (ValueError, TypeError):
        return None


def load_folder_convergence(folder: str) -> Dict[str, ConvergenceRecord]:
    """
    Load convergence data (iterations and final_residual) from evaluation_result_*.json files in folder.
    Returns dataset -> ConvergenceRecord
    """
    results: Dict[str, ConvergenceRecord] = {}
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    files = sorted(
        f for f in os.listdir(folder)
        if f.startswith("evaluation_result_") and f.endswith(".json")
    )
    if not files:
        raise FileNotFoundError(f"No evaluation_result_*.json found in directory: {folder}")

    for fp in files:
        full_path = os.path.join(folder, fp)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read, skipping: {fp} ({e})")
            continue

        dataset = _extract_dataset_name(data, fp)
        conv_data = _extract_convergence_data(data)
        if conv_data is None:
            print(f"[WARN] Convergence data not found or invalid, skipping: {fp}")
            continue

        dp_iter, dp_res, mg_iter, mg_res = conv_data
        record = ConvergenceRecord(
            dataset=dataset,
            double_precision_iterations=dp_iter,
            double_precision_residual=dp_res,
            model_guided_iterations=mg_iter,
            model_guided_residual=mg_res,
            source_file=full_path,
        )

        # If same dataset appears multiple times, keep the one with better convergence (lower final residual for model_guided)
        if dataset not in results or record.model_guided_residual < results[dataset].model_guided_residual:
            results[dataset] = record

    return results


def load_folder(folder: str) -> Dict[str, Record]:
    """
    Return dataset -> Record (if same dataset appears multiple times, take the one with max improvement_percent)
    """
    results: Dict[str, Record] = {}
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    files = sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".json") and f.startswith("evaluation_result_")
        ]
    )
    if not files:
        raise FileNotFoundError(f"No evaluation_result_*.json found in directory: {folder}")

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read, skipping: {fp} ({e})")
            continue

        dataset = _extract_dataset_name(data, fp)
        imp = _extract_cost_improvement_percent(data)
        if imp is None:
            print(f"[WARN] cost_improvement_percent not found, skipping: {fp}")
            continue

        rec = Record(dataset=dataset, improvement_percent=imp, source_file=fp)
        prev = results.get(dataset)
        if prev is None or rec.improvement_percent > prev.improvement_percent:
            results[dataset] = rec

    return results


def _sorted_datasets(*maps: Dict[str, Record]) -> List[str]:
    all_ds = set()
    for m in maps:
        all_ds.update(m.keys())
    return sorted(all_ds)


def plot_single(
    dataset_to_record: Dict[str, Record],
    datasets: List[str],
    title: str,
    out_path: str,
    bar_color: str,
):
    xs = list(range(len(datasets)))
    ys: List[float] = []
    for d in datasets:
        rec = dataset_to_record.get(d)
        ys.append(rec.improvement_percent if rec else float("nan"))

    plt.figure(figsize=(max(10, len(datasets) * 0.6), 5))
    bars = plt.bar(xs, ys, color=bar_color, alpha=0.85)
    plt.title(title)
    plt.ylabel("cost_improvement_percent (%)")
    plt.xticks(xs, datasets, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # 标注数值
    for i, b in enumerate(bars):
        v = ys[i]
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            continue
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_improvement_comparison(
    a: Dict[str, Record],
    b: Dict[str, Record],
    datasets: List[str],
    title: str,
    out_path: str,
    label_a: str,
    label_b: str,
):
    """
    Plot cost improvement comparison between two directories using grouped bars.
    """
    xs = list(range(len(datasets)))
    ys_a: List[float] = []
    ys_b: List[float] = []

    for d in datasets:
        rec_a = a.get(d)
        rec_b = b.get(d)
        ys_a.append(rec_a.improvement_percent if rec_a else float("nan"))
        ys_b.append(rec_b.improvement_percent if rec_b else float("nan"))

    plt.figure(figsize=(max(12, len(datasets) * 0.8), 6))
    width = 0.35

    bars_a = plt.bar([x - width/2 for x in xs], ys_a, width, label=label_a,
                     color="#1f77b4", alpha=0.8)
    bars_b = plt.bar([x + width/2 for x in xs], ys_b, width, label=label_b,
                     color="#ff7f0e", alpha=0.8)

    plt.title(title)
    plt.ylabel("Cost Improvement Percent (%)")
    plt.xticks(xs, datasets, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate values
    for bars, ys in [(bars_a, ys_a), (bars_b, ys_b)]:
        for bar, v in zip(bars, ys):
            if not (math.isnan(v) or math.isinf(v)):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_iterations_and_improvement_comparison(
    conv_a: Dict[str, ConvergenceRecord],
    conv_b: Dict[str, ConvergenceRecord],
    imp_a: Dict[str, Record],
    imp_b: Dict[str, Record],
    datasets: List[str],
    title: str,
    out_path: str,
    label_a: str,
    label_b: str,
):
    """
    Plot combined iterations (bars, left y-axis) and cost improvement percent (line, right y-axis).
    """
    xs = list(range(len(datasets)))

    # Prepare iterations data (same as before)
    dp_iterations = []  # Same for both directories
    mg_a_iterations = []
    mg_b_iterations = []

    # Prepare improvement data
    imp_a_values = []
    imp_b_values = []

    for d in datasets:
        # Iterations data
        rec_a = conv_a.get(d)
        rec_b = conv_b.get(d)
        dp_iter = (rec_a.double_precision_iterations if rec_a else 0) or (rec_b.double_precision_iterations if rec_b else 0)
        dp_iterations.append(dp_iter)
        mg_a_iterations.append(rec_a.model_guided_iterations if rec_a else 0)
        mg_b_iterations.append(rec_b.model_guided_iterations if rec_b else 0)

        # Improvement data
        imp_rec_a = imp_a.get(d)
        imp_rec_b = imp_b.get(d)
        imp_a_values.append(imp_rec_a.improvement_percent if imp_rec_a else float('nan'))
        imp_b_values.append(imp_rec_b.improvement_percent if imp_rec_b else float('nan'))

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(max(14, len(datasets) * 1.2), 7))

    # Bar positions and width for iterations
    width = 0.2
    positions = []
    for i in xs:
        positions.extend([i - width, i, i + width])

    # Iterations data: dp, mg_a, mg_b
    iterations_data = []
    colors = []
    for i, d in enumerate(datasets):
        iterations_data.extend([dp_iterations[i], mg_a_iterations[i], mg_b_iterations[i]])
        colors.extend(['#808080', '#1f77b4', '#ff7f0e'])

    # Plot iterations bars (left y-axis)
    bars = ax1.bar(positions, iterations_data, width, color=colors, alpha=0.7)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Iterations', color='#333333')
    ax1.tick_params(axis='y', labelcolor='#333333')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value annotations for bars
    for bar, value in zip(bars, iterations_data):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value}', ha='center', va='bottom', fontsize=7)

    # Right y-axis for improvement percentages
    ax2 = ax1.twinx()

    # Plot improvement lines
    line_a = ax2.plot(xs, imp_a_values, 'o-', color='#1f77b4', linewidth=2, markersize=6,
                     label=f'{label_a} Improvement (%)', alpha=0.8)
    line_b = ax2.plot(xs, imp_b_values, 's-', color='#ff7f0e', linewidth=2, markersize=6,
                     label=f'{label_b} Improvement (%)', alpha=0.8)

    ax2.set_ylabel('Cost Improvement Percent (%)', color='#666666')
    ax2.tick_params(axis='y', labelcolor='#666666')

    # Add value annotations for improvement lines
    for i, (x_pos, val_a, val_b) in enumerate(zip(xs, imp_a_values, imp_b_values)):
        if not math.isnan(val_a):
            ax2.text(x_pos, val_a, f'{val_a:.1f}%', ha='center', va='bottom', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        if not math.isnan(val_b):
            ax2.text(x_pos, val_b, f'{val_b:.1f}%', ha='center', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Create combined legend
    # Bar legend
    bar_legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#808080', alpha=0.7, label='Double Precision Iterations'),
        plt.Rectangle((0,0),1,1, facecolor='#1f77b4', alpha=0.7, label=f'{label_a} Model Guided Iterations'),
        plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', alpha=0.7, label=f'{label_b} Model Guided Iterations'),
    ]

    # Combine legends
    ax1.legend(handles=bar_legend_elements, loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    plt.title(title)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_iterations_comparison(
    conv_a: Dict[str, ConvergenceRecord],
    conv_b: Dict[str, ConvergenceRecord],
    datasets: List[str],
    title: str,
    out_path: str,
    label_a: str,
    label_b: str,
):
    """
    Plot iterations comparison between two directories using grouped bars.
    Shows Double Precision (same for both) vs Model Guided iterations for each dataset.
    """
    xs = list(range(len(datasets)))
    dp_iterations = []  # Same for both directories
    mg_a_iterations = []
    mg_b_iterations = []

    for d in datasets:
        rec_a = conv_a.get(d)
        rec_b = conv_b.get(d)

        # Use DP iterations from either record (they should be the same)
        dp_iter = (rec_a.double_precision_iterations if rec_a else 0) or (rec_b.double_precision_iterations if rec_b else 0)
        dp_iterations.append(dp_iter)

        mg_a_iterations.append(rec_a.model_guided_iterations if rec_a else 0)
        mg_b_iterations.append(rec_b.model_guided_iterations if rec_b else 0)

    plt.figure(figsize=(max(12, len(datasets) * 1.0), 6))

    # Bar positions and width
    width = 0.25
    positions = []
    for i in xs:
        positions.extend([i - width, i, i + width])

    # Data in order: dp, mg_a, mg_b for each dataset
    all_data = []
    colors = []
    for i, d in enumerate(datasets):
        all_data.extend([dp_iterations[i], mg_a_iterations[i], mg_b_iterations[i]])
        colors.extend(['#808080', '#1f77b4', '#ff7f0e'])  # Gray for DP, Blue for A MG, Orange for B MG

    bars = plt.bar(positions, all_data, width, color=colors, alpha=0.8)

    # Create custom legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#808080', alpha=0.8, label='Double Precision'),
        plt.Rectangle((0,0),1,1, facecolor='#1f77b4', alpha=0.8, label=f'{label_a} Model Guided'),
        plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', alpha=0.8, label=f'{label_b} Model Guided'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.title(title)
    plt.ylabel('Iterations')
    plt.xticks(xs, datasets, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value annotations
    for bar, value in zip(bars, all_data):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value}', ha='center', va='bottom', fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_max(
    a: Dict[str, Record],
    b: Dict[str, Record],
    datasets: List[str],
    title: str,
    out_path: str,
    label_a: str,
    label_b: str,
):
    xs = list(range(len(datasets)))
    ys: List[float] = []
    colors: List[str] = []
    winners: List[str] = []
    for d in datasets:
        va = a.get(d).improvement_percent if d in a else float("-inf")
        vb = b.get(d).improvement_percent if d in b else float("-inf")
        if va >= vb:
            ys.append(va if va != float("-inf") else float("nan"))
            colors.append("#1f77b4")  # Blue: A
            winners.append(label_a)
        else:
            ys.append(vb if vb != float("-inf") else float("nan"))
            colors.append("#ff7f0e")  # Orange: B
            winners.append(label_b)

    plt.figure(figsize=(max(10, len(datasets) * 0.6), 5))
    bars = plt.bar(xs, ys, color=colors, alpha=0.9)
    plt.title(title)
    plt.ylabel("max(cost_improvement_percent) (%)")
    plt.xticks(xs, datasets, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # annotate: max value + winner
    for i, b0 in enumerate(bars):
        v = ys[i]
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            continue
        plt.text(
            b0.get_x() + b0.get_width() / 2,
            b0.get_height(),
            f"{v:.2f}\n({winners[i]})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # legend
    import matplotlib.patches as mpatches

    patch_a = mpatches.Patch(color="#1f77b4", label=f"From {label_a}")
    patch_b = mpatches.Patch(color="#ff7f0e", label=f"From {label_b}")
    plt.legend(handles=[patch_a, patch_b], loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize(a: Dict[str, Record], b: Dict[str, Record], datasets: List[str], label_a: str, label_b: str):
    win_a = 0
    win_b = 0
    tie = 0
    miss_a = 0
    miss_b = 0

    def _mean(vals: Iterable[float]) -> float:
        xs = [v for v in vals if v is not None and not (math.isnan(v) or math.isinf(v))]
        return sum(xs) / len(xs) if xs else float("nan")

    vals_a: List[float] = []
    vals_b: List[float] = []

    print("\n=== per-dataset comparison (cost_improvement_percent, higher is better) ===")
    for d in datasets:
        ra = a.get(d)
        rb = b.get(d)
        va = ra.improvement_percent if ra else None
        vb = rb.improvement_percent if rb else None
        if va is None:
            miss_a += 1
        else:
            vals_a.append(va)
        if vb is None:
            miss_b += 1
        else:
            vals_b.append(vb)

        if va is None or vb is None:
            who = label_a if vb is None else label_b
            print(f"- {d}: {label_a}={va} , {label_b}={vb}  -> Only {who} has result")
            continue
        if abs(va - vb) < 1e-12:
            tie += 1
            who = "tie"
        elif va > vb:
            win_a += 1
            who = label_a
        else:
            win_b += 1
            who = label_b
        print(f"- {d}: {label_a}={va:.4f} , {label_b}={vb:.4f}  -> Winner: {who}")

    print("\n=== Summary ===")
    print(f"- {label_a} wins: {win_a}")
    print(f"- {label_b} wins: {win_b}")
    print(f"- Ties: {tie}")
    print(f"- {label_a} missing datasets: {miss_a}")
    print(f"- {label_b} missing datasets: {miss_b}")
    print(f"- {label_a} average improvement (%): {_mean(vals_a):.4f}")
    print(f"- {label_b} average improvement (%): {_mean(vals_b):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", type=str, default=DEFAULT_DIR_A, help="First best_shared directory")
    parser.add_argument("--dir_b", type=str, default=DEFAULT_DIR_B, help="Second best_shared directory")
    parser.add_argument("--label_a", type=str, default="mesh3e1(best_shared)")
    parser.add_argument("--label_b", type=str, default="mesh2e1(best_shared)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/bingxing2/home/scx7axu/program/rlcg/log/backup/251221/cost_improvement_plots",
        help="Output directory (saves three charts)",
    )
    args = parser.parse_args()

    configure_matplotlib_chinese()

    a = load_folder(args.dir_a)
    b = load_folder(args.dir_b)
    datasets = _sorted_datasets(a, b)

    # Max chart + combined iterations and improvement chart
    out1 = os.path.join(args.out_dir, "cost_improvement_percent_max_per_dataset.png")
    out2 = os.path.join(args.out_dir, "iterations_and_improvement_comparison.png")

    plot_max(
        a,
        b,
        datasets,
        title="Max Cost Improvement Percent per Dataset (Color indicates source directory)",
        out_path=out1,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    # Load convergence data and create combined iterations and improvement comparison plot
    try:
        conv_a = load_folder_convergence(args.dir_a)
        conv_b = load_folder_convergence(args.dir_b)

        plot_iterations_and_improvement_comparison(
            conv_a,
            conv_b,
            a,
            b,
            datasets,
            title="Iterations and Cost Improvement Percent Comparison",
            out_path=out2,
            label_a=args.label_a,
            label_b=args.label_b,
        )

        print("\n=== Saved images ===")
        print(f"- {out1}")
        print(f"- {out2}")

    except Exception as e:
        print(f"[WARN] Failed to create combined comparison plot: {e}")
        print("\n=== Saved images ===")
        print(f"- {out1}")


    summarize(a, b, datasets, args.label_a, args.label_b)


if __name__ == "__main__":
    main()


