#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 nsys profile 运行 aCG 的 CUDA 版本 CG，并统计 CUDA kernel 时间在三类算子中的占比：
  - gemv: 主要对应 cuSPARSE SpMV（在 aCG 里实现为稀疏矩阵-向量乘）
  - dot_product/reduce: 主要对应 cuBLAS dot 以及相关 reduction kernel
  - daxpy: 主要对应 axpy/daxpy（aCG 自定义 kernel 或 cuBLAS）

默认从仓库根目录的 cg_results.csv 读取矩阵 name 字段，拼成 <matrix_dir>/<name>.mtx，
对每个矩阵执行：
  nsys profile ... <acg_bin> <matrix_path>
随后用 nsys stats 导出 cuda_kernel_summary 的 CSV，按 kernel 名称规则归类并画图。

示例：
  python3 scripts/profile_acg_cuda_ops.py --name thermomech_dM --matrix-dir ~/data/matrix
  python3 scripts/profile_acg_cuda_ops.py --max-matrices 5
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from glob import glob


# ---------------------------
# 配置：kernel 名称归类规则
# ---------------------------

_CATEGORY_ORDER = ["gemv", "dot_reduce", "daxpy", "other"]

_CATEGORY_PATTERNS: Dict[str, List[re.Pattern]] = {
    # gemv 在 CG 里基本就是 SpMV（cusparseSpMV -> 内部若干 kernel）
    "gemv": [
        re.compile(r"spmv", re.IGNORECASE),
        re.compile(r"csrmv", re.IGNORECASE),
        re.compile(r"csr.*mv", re.IGNORECASE),
        re.compile(r"cusparse", re.IGNORECASE),
        re.compile(r"sparse", re.IGNORECASE),
    ],
    # dot + reduction：cublas dot、以及常见 reduce/cub reduction kernel
    "dot_reduce": [
        re.compile(r"\bdot\b", re.IGNORECASE),
        re.compile(r"cublas.*dot", re.IGNORECASE),
        re.compile(r"reduce", re.IGNORECASE),
        re.compile(r"reduction", re.IGNORECASE),
        re.compile(r"\bcub::.*reduce", re.IGNORECASE),
    ],
    # daxpy/axpy：aCG 自定义 kernel 函数名里就含 daxpy
    "daxpy": [
        re.compile(r"daxpy", re.IGNORECASE),
        re.compile(r"\baxpy\b", re.IGNORECASE),
        re.compile(r"acgsolvercuda_.*axpy", re.IGNORECASE),
    ],
}


@dataclass(frozen=True)
class KernelRow:
    name: str
    total_time_s: float


@dataclass(frozen=True)
class BreakdownRow:
    matrix: str
    category: str
    time_s: float
    percent: float


def _repo_root() -> Path:
    # scripts/ 目录下运行也能找到仓库根目录
    return Path(__file__).resolve().parents[1]


def _which(exe: str) -> Optional[str]:
    from shutil import which

    return which(exe)


def _resolve_acg_bin(user_path: str) -> Path:
    p = Path(os.path.expanduser(user_path)).resolve()
    if p.exists() and os.access(p, os.X_OK):
        return p

    # 用户习惯写 ./acg-cuda，但本仓库里常见位置是 third_party/aCG/build/acg-cuda
    fallback = (_repo_root() / "third_party" / "aCG" / "build" / "acg-cuda").resolve()
    if fallback.exists() and os.access(fallback, os.X_OK):
        return fallback

    raise FileNotFoundError(
        "找不到可执行文件 acg-cuda。\n"
        f"- 你传入的路径：{str(p)}\n"
        f"- 也未在默认回退位置找到：{str(fallback)}\n"
        "请先编译 aCG，或用 --acg-bin 显式指定可执行文件路径。"
    )


def _ensure_nsys_available(nsys_bin: str) -> str:
    path = _which(nsys_bin)
    if path:
        return path
    raise FileNotFoundError(
        "找不到 nsys（NVIDIA Nsight Systems）。\n"
        "请确认已安装并在 PATH 中。例如：\n"
        "- 集群环境：`module load nsys` 或 `module load nvhpc`（以实际环境为准）\n"
        "- 本机安装：确保 `nsys` 可执行文件在 PATH 里\n"
        f"也可以用 --nsys-bin 指定 nsys 路径（当前：{nsys_bin}）。"
    )


def _read_matrix_names_from_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 cg_results.csv：{str(csv_path)}")
    names: List[str] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "name" not in (reader.fieldnames or []):
            raise ValueError(f"{str(csv_path)} 缺少 'name' 字段（实际字段：{reader.fieldnames}）")
        for row in reader:
            n = (row.get("name") or "").strip()
            if n:
                names.append(n)

    # 去重但保持顺序
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _read_converged_iterations_from_csv(csv_path: Path) -> Dict[str, int]:
    """
    从 cg_results.csv 读取 {name -> converged_iterations}。
    仅用于自动设置 --max-iterations 以避免大量“not converged”导致 exit code=1。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 cg_results.csv：{str(csv_path)}")
    out: Dict[str, int] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        if "name" not in reader.fieldnames or "converged_iterations" not in reader.fieldnames:
            return out
        for row in reader:
            n = (row.get("name") or "").strip()
            it = (row.get("converged_iterations") or "").strip()
            if not n or not it:
                continue
            try:
                out[n] = int(float(it))
            except ValueError:
                continue
    return out


def _run(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    print("+", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _run_capture(cmd: List[str], *, cwd: Path) -> str:
    print("+", " ".join(shlex.quote(x) for x in cmd))
    p = subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout


def _list_nsys_reports(nsys: str) -> List[str]:
    """
    返回 nsys stats 支持的 report 名称列表（不同版本差异很大）。
    尝试若干常见命令格式，尽量兼容。
    """
    candidates = [
        [nsys, "stats", "--list-reports"],
        [nsys, "stats", "--list", "reports"],
        [nsys, "stats", "--help"],
    ]
    out = ""
    for cmd in candidates:
        try:
            out = _run_capture(cmd, cwd=_repo_root())
            if out:
                break
        except subprocess.CalledProcessError:
            continue
    if not out:
        return []

    reports: List[str] = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        # 常见格式：
        # - "cuda_gpu_kern_sum"
        # - "  cuda_gpu_kern_sum  : CUDA GPU Kernel Summary"
        # - "cuda_gpu_kern_sum - CUDA GPU Kernel Summary"
        m = re.match(r"^([A-Za-z0-9_]+)\b", s)
        if not m:
            continue
        name = m.group(1)
        # 排除一些 help 噪声行
        if name in {"Usage", "Options", "Examples", "Report", "Reports"}:
            continue
        # 只收集看起来像 report 名的 token
        if "_" in name or name.startswith("cuda") or name.startswith("gpu"):
            reports.append(name)

    # 去重保持顺序
    seen = set()
    out_reports = []
    for r in reports:
        if r not in seen:
            seen.add(r)
            out_reports.append(r)
    return out_reports


def _pick_kernel_report(nsys: str, preferred: Optional[str]) -> str:
    """
    选择一个“CUDA kernel summary”类的 report。
    - preferred 非空：直接使用（如果不存在会在 stats 阶段报错并给出可选列表）
    - preferred 为空：自动从 list-reports 中挑一个最可能的
    """
    if preferred:
        return preferred

    reports = _list_nsys_reports(nsys)
    # 常见候选（不同版本命名不同）
    preferred_candidates = [
        "cuda_gpu_kern_sum",
        "cuda_kern_sum",
        "cuda_gpu_kernel_sum",
        "cuda_gpu_kern_summary",
        "cuda_kernels_sum",
        "cuda_kernel_sum",
    ]
    for c in preferred_candidates:
        if c in reports:
            return c

    # 再用关键词匹配
    for r in reports:
        low = r.lower()
        if low.startswith("cuda") and ("kern" in low or "kernel" in low) and ("sum" in low or "summary" in low):
            return r

    # 最后兜底：如果任何 cuda report 都没有，就直接用一个最像的
    if reports:
        return reports[0]
    # 完全探测不到：返回一个常见默认名（让后续报错给出提示）
    return "cuda_gpu_kern_sum"


def _nsys_profile_one(
    *,
    nsys: str,
    acg_bin: Path,
    matrix_path: Path,
    out_dir: Path,
    run_tag: str,
    extra_acg_args: List[str],
    dry_run: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{matrix_path.stem}.{run_tag}"
    # 注意：Path.with_suffix() 会把最后一个 '.' 后面当成“扩展名”整体替换，
    # 对于 base=xxx.<timestamp> 会导致 timestamp 被吞掉。
    # nsys 实际输出的 rep 文件名是：<base>.nsys-rep
    rep_path = Path(str(base) + ".nsys-rep")

    # 说明：
    # - --trace=cuda：抓 CUDA kernel / memcpy 等
    # - --sample=none：降低 CPU sampling 干扰
    # - --stats=false：profile 阶段不输出统计（后面统一用 nsys stats 处理）
    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--sample=none",
        "--trace=cuda,osrt",
        "--stats=false",
        "-o",
        str(base),
        str(acg_bin),
        str(matrix_path),
        *extra_acg_args,
    ]

    if dry_run:
        print("[dry-run] 将生成:", rep_path)
        print("+", " ".join(shlex.quote(x) for x in cmd))
        return rep_path

    # 重要：acg-cuda 在“not converged”时会返回非 0（通常是 1）。
    # nsys profile 会把这个退出码原样返回，但 rep 文件仍会生成、统计也仍然有意义。
    print("+", " ".join(shlex.quote(x) for x in cmd))
    p = subprocess.run(cmd, cwd=str(_repo_root()), check=False)
    if p.returncode != 0:
        print(f"[warn] nsys profile 退出码非 0（{p.returncode}）；若 rep 已生成将继续统计。")
    if not rep_path.exists():
        raise RuntimeError(f"nsys profile 结束后未找到 .nsys-rep：{str(rep_path)}（returncode={p.returncode}）")
    return rep_path


def _nsys_stats_to_csv(
    *,
    nsys: str,
    rep_path: Path,
    csv_out: Path,
    report: str,
    dry_run: bool,
) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("[dry-run] 将生成:", csv_out)
        cmd = [nsys, "stats", "--report", report, "--format", "csv", str(rep_path)]
        print("+", " ".join(shlex.quote(x) for x in cmd))
        return

    # 直接抓 stdout 写入，避免 --output 在不同版本里自动追加文件名后缀导致路径难以预测
    cmd = [nsys, "stats", "--report", report, "--format", "csv", str(rep_path)]
    out = _run_capture(cmd, cwd=_repo_root())
    csv_out.write_text(out, encoding="utf-8")
    if csv_out.stat().st_size == 0:
        raise RuntimeError("nsys stats 输出为空，请检查 nsys 版本/权限/rep 文件是否有效。")


def _parse_time_to_seconds(value: str) -> float:
    s = value.strip()
    if not s:
        return 0.0

    # 兼容 "123.4" / "123.4 ns" / "123.4us" / "1.2ms" / "0.01s"
    m = re.match(r"^\s*([0-9.+-eE]+)\s*([a-zA-Z]*)\s*$", s)
    if not m:
        raise ValueError(f"无法解析时间字段：{value!r}")

    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("", "s", "sec", "secs", "second", "seconds"):
        return num
    if unit in ("ns", "nsec"):
        return num * 1e-9
    if unit in ("us", "usec", "µs"):
        return num * 1e-6
    if unit in ("ms", "msec"):
        return num * 1e-3
    if unit in ("ps",):
        return num * 1e-12
    # nsys 有时会给 "nanoseconds" 之类
    if unit.startswith("nano"):
        return num * 1e-9
    if unit.startswith("micro"):
        return num * 1e-6
    if unit.startswith("milli"):
        return num * 1e-3
    raise ValueError(f"未知时间单位：{unit!r} (原字段={value!r})")


def _read_kernel_rows(kernel_csv: Path) -> List[KernelRow]:
    # nsys stats 的 csv 可能在前面带一些说明行；我们做“找 header 的行”的解析。
    lines = kernel_csv.read_text(encoding="utf-8", errors="replace").splitlines()
    # 找到第一行看起来像 CSV header 且包含 Name/Kernel 与 Total Time
    header_idx = None
    for i, line in enumerate(lines):
        if "," not in line:
            continue
        low = line.lower()
        if ("time" in low) and (("kernel" in low) or ("name" in low)) and ("total" in low):
            header_idx = i
            break
    if header_idx is None:
        # 直接当成纯 CSV
        header_idx = 0

    content = "\n".join(lines[header_idx:])
    reader = csv.DictReader(content.splitlines())
    if not reader.fieldnames:
        raise ValueError(f"{str(kernel_csv)} 解析失败：未找到 CSV header。")

    # 常见字段名（不同版本可能不同）
    name_key_candidates = [
        "Kernel Name",
        "Name",
        "Kernel",
        "CUDA Kernel Name",
    ]
    time_key_candidates = [
        "Total Time (ns)",
        "Total Time (us)",
        "Total Time (ms)",
        "Total Time",
        "Total Time (s)",
    ]

    def _pick_key(cands: List[str]) -> Optional[str]:
        # 精确匹配优先，其次大小写不敏感匹配
        fields = reader.fieldnames or []
        for k in cands:
            if k in fields:
                return k
        low_map = {f.lower(): f for f in fields}
        for k in cands:
            if k.lower() in low_map:
                return low_map[k.lower()]
        return None

    name_key = _pick_key(name_key_candidates)
    time_key = _pick_key(time_key_candidates)
    if not name_key or not time_key:
        raise ValueError(
            f"{str(kernel_csv)} 字段不符合预期。\n"
            f"- fieldnames={reader.fieldnames}\n"
            f"- 需要包含 kernel 名称 与 total time 字段。"
        )

    out: List[KernelRow] = []
    for row in reader:
        name = (row.get(name_key) or "").strip()
        t = (row.get(time_key) or "").strip()
        if not name:
            continue
        try:
            # 如果 time_key 自带单位（例如 Total Time (ns)），但值里常常是裸数字
            if time_key.lower().endswith("(ns)"):
                total_s = float(t) * 1e-9 if t else 0.0
            elif time_key.lower().endswith("(us)"):
                total_s = float(t) * 1e-6 if t else 0.0
            elif time_key.lower().endswith("(ms)"):
                total_s = float(t) * 1e-3 if t else 0.0
            elif time_key.lower().endswith("(s)"):
                total_s = float(t) if t else 0.0
            else:
                # 值里可能带单位
                total_s = _parse_time_to_seconds(t)
        except Exception as e:
            raise ValueError(f"解析 Total Time 失败：row[{time_key}]={t!r}, error={e}") from e
        out.append(KernelRow(name=name, total_time_s=total_s))
    return out


def load_breakdown_csv(breakdown_csv: Path) -> List[BreakdownRow]:
    """
    读取由本脚本生成的 breakdown CSV（ops_breakdown_all.*.csv 或 summary/*.ops_breakdown.csv）。
    字段：matrix, category, time_s, percent
    """
    p = Path(breakdown_csv).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"未找到 breakdown CSV：{str(p)}")
    rows: List[BreakdownRow] = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"matrix", "category", "time_s", "percent"}
        if not required.issubset(set(fields)):
            raise ValueError(f"breakdown CSV 字段不匹配：{fields}，需要至少 {sorted(required)}")
        for r in reader:
            m = (r.get("matrix") or "").strip()
            c = (r.get("category") or "").strip()
            if not m or not c:
                continue
            try:
                t = float((r.get("time_s") or "0").strip())
            except ValueError:
                t = 0.0
            try:
                pct = float((r.get("percent") or "0").strip())
            except ValueError:
                pct = 0.0
            rows.append(BreakdownRow(matrix=m, category=c, time_s=t, percent=pct))
    return rows


def resolve_breakdown_csv_input(user_input: Path) -> Path:
    """
    解析 --plot-from-breakdown 的输入：
    - 传入 CSV 文件：直接返回
    - 传入目录：自动选择最新的 ops_breakdown_all.*.csv（优先）或任意 .csv
    - 传入带通配符的 pattern：glob 后选择最新文件
    """
    s = str(user_input)
    p = Path(user_input).expanduser()

    # 通配符：优先 glob
    if any(ch in s for ch in ["*", "?", "["]):
        matches = [Path(x) for x in glob(os.path.expanduser(s))]
        matches = [m for m in matches if m.is_file()]
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {s}")
        return max(matches, key=lambda x: x.stat().st_mtime)

    p = p.resolve()
    if p.is_file():
        return p

    if p.is_dir():
        # 优先 ops_breakdown_all.*.csv
        cands = sorted(p.glob("ops_breakdown_all.*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
        # 兜底：任意 csv
        cands = sorted(p.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
        raise FileNotFoundError(f"No .csv found under directory: {str(p)}")

    raise FileNotFoundError(f"Breakdown CSV path not found: {str(p)}")


def breakdown_rows_to_per_matrix(rows: Iterable[BreakdownRow]) -> List[Tuple[str, Dict[str, float]]]:
    """
    将 breakdown rows 归并成 _plot 所需的 per_matrix 结构：
      [(matrix_name, {category: time_s, ...}), ...]
    """
    per: Dict[str, Dict[str, float]] = {}
    order: List[str] = []
    seen = set()
    for r in rows:
        if r.matrix not in seen:
            seen.add(r.matrix)
            order.append(r.matrix)
        if r.matrix not in per:
            per[r.matrix] = {k: 0.0 for k in _CATEGORY_ORDER}
        cat = r.category if r.category in per[r.matrix] else "other"
        per[r.matrix][cat] += float(r.time_s)
    return [(m, per[m]) for m in order if m in per]


def plot_from_breakdown_csv(*, breakdown_csv: Path, out_png: Path, title: str) -> None:
    """
    对外接口：直接从 breakdown CSV 画图，不需要重跑 nsys。
    """
    rows = load_breakdown_csv(breakdown_csv)
    per_matrix = breakdown_rows_to_per_matrix(rows)
    if not per_matrix:
        raise RuntimeError("breakdown CSV 中没有可用于绘图的数据（per_matrix 为空）。")
    _plot(Path(out_png).expanduser().resolve(), per_matrix, title=title)


def _categorize_kernel(name: str) -> str:
    for cat in ("gemv", "dot_reduce", "daxpy"):
        for pat in _CATEGORY_PATTERNS[cat]:
            if pat.search(name):
                return cat
    return "other"


def _aggregate_categories(rows: Iterable[KernelRow]) -> Dict[str, float]:
    totals = {k: 0.0 for k in _CATEGORY_ORDER}
    for r in rows:
        cat = _categorize_kernel(r.name)
        totals[cat] += r.total_time_s
    return totals


def _write_summary_csv(path: Path, matrix_name: str, totals: Dict[str, float]) -> None:
    total_all = sum(totals.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "category", "time_s", "percent"])
        for cat in _CATEGORY_ORDER:
            t = totals.get(cat, 0.0)
            pct = (t / total_all * 100.0) if total_all > 0 else 0.0
            w.writerow([matrix_name, cat, f"{t:.9f}", f"{pct:.6f}"])


def _plot(
    out_png: Path,
    per_matrix: List[Tuple[str, Dict[str, float]]],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    names = [n for n, _ in per_matrix]
    times = [d for _, d in per_matrix]
    totals = [sum(d.values()) for d in times]

    # 归一化为百分比
    pct_by_cat = {cat: [] for cat in _CATEGORY_ORDER}
    for d, total in zip(times, totals):
        for cat in _CATEGORY_ORDER:
            pct_by_cat[cat].append((d.get(cat, 0.0) / total * 100.0) if total > 0 else 0.0)

    fig_w = max(8.0, 0.55 * len(names))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    bottom = [0.0] * len(names)
    colors = {
        "gemv": "#4C78A8",
        "dot_reduce": "#F58518",
        "daxpy": "#54A24B",
        "other": "#B0B0B0",
    }
    for cat in _CATEGORY_ORDER:
        vals = pct_by_cat[cat]
        ax.bar(names, vals, bottom=bottom, label=cat, color=colors.get(cat))
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_ylabel("CUDA kernel time share (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="nsys profile aCG-cuda 并统计 gemv/dot(reduce)/daxpy 时间占比")
    ap = argparse.ArgumentParser(description="profile + plot 用法： python scripts/plot_profile_acg_cuda_ops.py --dry-run \
                                                --matrix-dir ~/data/matrix \
                                                --acg-bin ./third_party/aCG/build/acg-cuda \
                                                --out-dir ./nsys_profiles_all \
                                                --skip-missing \
                                                --continue-on-error \
                                                --delete-rep \
                                                --max-iterations 1000")
    ap = argparse.ArgumentParser(description="plot 用法： python scripts/plot_profile_acg_cuda_ops.py \
                                                --plot-from-breakdown ./nsys_profiles_all \
                                                --plot ./nsys_profiles_all/ops_time_percent.from_csv.png")
    ap.add_argument("--csv", default=str(_repo_root() / "cg_results.csv"), help="cg_results.csv 路径")
    ap.add_argument("--matrix-dir", default="~/data/matrix", help="矩阵 .mtx 文件目录（默认 ~/data/matrix）")
    ap.add_argument("--name", action="append", default=[], help="指定矩阵 name（可多次给出）；不指定则从 cg_results.csv 读取全部")
    ap.add_argument("--max-matrices", type=int, default=0, help="最多处理多少个矩阵（0=不限制）")
    ap.add_argument("--acg-bin", default="./acg-cuda", help="acg-cuda 可执行文件路径（找不到会回退到 third_party/aCG/build/acg-cuda）")
    ap.add_argument("--nsys-bin", default="nsys", help="nsys 可执行文件名或路径")
    ap.add_argument("--out-dir", default=str(_repo_root() / "nsys_profiles"), help="输出目录（rep/csv/png）")
    ap.add_argument(
        "--report",
        default="",
        help="nsys stats 的 report 名；留空则自动探测并选择一个 CUDA kernel summary 类 report（推荐）",
    )
    ap.add_argument("--run-tag", default="", help="输出文件 tag（默认自动用时间戳）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不实际运行")
    ap.add_argument("--extra-acg-args", default="", help="额外传给 acg-cuda 的参数（整串字符串，会按 shell 规则拆分）")
    ap.add_argument("--plot", default="", help="输出图路径（默认 out-dir/ops_time_percent.png）")
    ap.add_argument(
        "--plot-from-breakdown",
        default="",
        help="只绘图：给一个已生成的 breakdown CSV（例如 nsys_profiles_all/ops_breakdown_all.*.csv）则直接读它画图，不再重跑 nsys。",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="传给 acg-cuda 的 --max-iterations；0 表示自动（优先用 cg_results.csv 的 converged_iterations+margin，否则用 acg-cuda 默认 100）",
    )
    ap.add_argument(
        "--max-iterations-margin",
        type=int,
        default=10,
        help="自动 max-iterations 时的冗余：max = converged_iterations + margin（默认 10）",
    )
    ap.add_argument("--skip-missing", action="store_true", help="若矩阵文件不存在则跳过（默认遇到缺失会报错退出）")
    ap.add_argument("--continue-on-error", action="store_true", help="单个矩阵失败则继续跑后续矩阵（默认失败即退出）")
    ap.add_argument(
        "--delete-rep",
        action="store_true",
        help="在导出 stats CSV 后删除 .nsys-rep/.sqlite 以节省空间（默认保留）",
    )
    args = ap.parse_args(argv)

    # plot-only：直接用已有 breakdown CSV 画图（不触发 nsys/profile）
    if args.plot_from_breakdown:
        in_csv = resolve_breakdown_csv_input(Path(args.plot_from_breakdown))
        out_png = Path(args.plot).expanduser().resolve() if args.plot else in_csv.with_suffix(".png")
        plot_from_breakdown_csv(
            breakdown_csv=in_csv,
            out_png=out_png,
            title="aCG-cuda: CUDA kernel time breakdown (gemv / dot_reduce / daxpy / other)",
        )
        print("Done (plot-only).")
        print("breakdown_csv:", in_csv)
        print("plot_png:", out_png)
        return 0

    # 解析/检查
    acg_bin = _resolve_acg_bin(args.acg_bin)
    # dry-run 允许在没有 nsys 的环境下先把命令与输出路径打印出来
    nsys = args.nsys_bin if args.dry_run else _ensure_nsys_available(args.nsys_bin)

    csv_path = Path(os.path.expanduser(args.csv)).resolve()
    matrix_dir = Path(os.path.expanduser(args.matrix_dir)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    extra_acg_args = shlex.split(args.extra_acg_args) if args.extra_acg_args else []
    report_name = _pick_kernel_report(nsys, args.report.strip() or None) if not args.dry_run else (args.report.strip() or "auto")

    if args.name:
        names = args.name
    else:
        names = _read_matrix_names_from_csv(csv_path)

    conv_iters = {} if args.dry_run else _read_converged_iterations_from_csv(csv_path)

    if args.max_matrices and args.max_matrices > 0:
        names = names[: args.max_matrices]

    if not names:
        raise SystemExit("未找到任何矩阵 name。请用 --name 指定，或检查 cg_results.csv。")

    per_matrix: List[Tuple[str, Dict[str, float]]] = []
    summary_rows: List[List[str]] = []

    for name in names:
        matrix_path = (matrix_dir / f"{name}.mtx").resolve()
        if (not args.dry_run) and (not matrix_path.exists()):
            msg = f"未找到矩阵文件：{str(matrix_path)}（name={name}）"
            if args.skip_missing:
                print("[skip-missing]", msg)
                continue
            raise FileNotFoundError(msg)

        try:
            # 自动设置 max-iterations：避免很多矩阵在默认 100 次内不收敛，从而 exit=1
            auto_maxits = 0
            if args.max_iterations and args.max_iterations > 0:
                auto_maxits = args.max_iterations
            elif name in conv_iters and conv_iters[name] > 0:
                auto_maxits = max(100, conv_iters[name] + max(0, args.max_iterations_margin))

            run_extra_args = list(extra_acg_args)
            if auto_maxits > 0:
                run_extra_args = ["--max-iterations", str(auto_maxits), *run_extra_args]

            rep = _nsys_profile_one(
                nsys=nsys,
                acg_bin=acg_bin,
                matrix_path=matrix_path,
                out_dir=out_dir / "rep",
                run_tag=run_tag,
                extra_acg_args=run_extra_args,
                dry_run=args.dry_run,
            )

            kernel_csv = (out_dir / "stats" / f"{matrix_path.stem}.{run_tag}.{report_name}.csv").resolve()
            _nsys_stats_to_csv(nsys=nsys, rep_path=rep, csv_out=kernel_csv, report=report_name, dry_run=args.dry_run)

            if args.dry_run:
                per_matrix.append((name, {k: 0.0 for k in _CATEGORY_ORDER}))
                continue

            rows = _read_kernel_rows(kernel_csv)
            totals = _aggregate_categories(rows)
            per_matrix.append((name, totals))

            total_all = sum(totals.values())
            for cat in _CATEGORY_ORDER:
                t = totals.get(cat, 0.0)
                pct = (t / total_all * 100.0) if total_all > 0 else 0.0
                summary_rows.append([name, cat, f"{t:.9f}", f"{pct:.6f}"])

            # 每个矩阵单独写一个 summary
            _write_summary_csv(out_dir / "summary" / f"{name}.{run_tag}.ops_breakdown.csv", name, totals)

            if args.delete_rep:
                try:
                    if rep.exists():
                        rep.unlink()
                    sqlite_path = rep.with_suffix(".sqlite")
                    if sqlite_path.exists():
                        sqlite_path.unlink()
                except Exception as e:
                    print("[warn] 删除 rep/sqlite 失败：", e)
        except Exception as e:
            if args.continue_on_error:
                print(f"[continue-on-error] matrix={name} 失败：{e}")
                continue
            raise

    # 写总表
    summary_path = out_dir / f"ops_breakdown_all.{run_tag}.csv"
    if not args.dry_run:
        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["matrix", "category", "time_s", "percent"])
            w.writerows(summary_rows)

    # 画图
    out_png = Path(args.plot).expanduser().resolve() if args.plot else (out_dir / f"ops_time_percent.{run_tag}.png")
    if not args.dry_run:
        if not per_matrix:
            raise RuntimeError("没有任何矩阵成功生成统计结果（per_matrix 为空）。")
        _plot(out_png, per_matrix, title="aCG-cuda: CUDA kernel time breakdown (gemv / dot_reduce / daxpy / other)")

    print("完成。")
    print("acg_bin:", acg_bin)
    print("out_dir:", out_dir)
    if not args.dry_run:
        print("summary_csv:", summary_path)
        print("plot_png:", out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


