import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


RUN_SCRIPT = Path(__file__).parent / "run.py"


@dataclass
class ExperimentConfig:
    name: str
    display_name: str
    extra_args: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare XGBLoRA+ZO vs LoRA+ZO on the same task."
    )
    parser.add_argument(
        "--model_name",
        default="facebook/opt-350m",
        help="Backbone model name (HuggingFace identifier).",
    )
    parser.add_argument("--task_name", default="SST2", help="Task to evaluate.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--zo_num_perturbations", type=int, default=1, help="Number of perturbation directions per ZO step (default 1)")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_dev", type=int, default=500)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--xgblora_steps_per_iteration", type=int, default=1000)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to compare (also reused for train_set_seed).",
    )
    parser.add_argument(
        "--output_root",
        default="comparison_runs",
        help="Root directory where method/seed outputs will be stored.",
    )
    parser.add_argument(
        "--figure_path",
        default="comparison_runs/xgblora_vs_lora.png",
        help="Where to save the summary plot.",
    )
    parser.add_argument(
        "--loss_figure_path",
        default="comparison_runs/loss_vs_steps.png",
        help="Where to save the loss-vs-steps plot.",
    )
    parser.add_argument(
        "--merge_eval_figure_path",
        default="comparison_runs/merge_eval_loss.png",
        help="Where to save the evaluation-at-merge plot.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip runs where the result file already exists.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="Optional extra args appended to every call to run.py",
    )
    return parser.parse_args()


def build_base_args(args: argparse.Namespace) -> List[str]:
    base = [
        "--model_name",
        args.model_name,
        "--task_name",
        args.task_name,
        "--trainer",
        "zo",
        "--learning_rate",
        str(args.learning_rate),
        "--zo_eps",
        str(args.zo_eps),
        "--zo_num_perturbations",
        str(args.zo_num_perturbations),
        "--lora_alpha",
        str(args.lora_alpha),
        "--no_auto_device",
        "--max_steps",
        str(args.max_steps),
        "--num_train",
        str(args.num_train),
        "--num_dev",
        str(args.num_dev),
        "--num_eval",
        str(args.num_eval),
        "--evaluation_strategy",
        "steps",
        "--eval_steps",
        str(args.eval_steps),
        "--save_strategy",
        "steps",
        "--save_steps",
        "1000",
        "--save_total_limit",
        "1",
        "--logging_steps",
        str(args.logging_steps),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--load_best_model_at_end",
        "--overwrite_output_dir",
    ]
    if args.extra_args:
        base.extend(args.extra_args)
    return base


def run_single_experiment(
    base_args: List[str],
    exp: ExperimentConfig,
    seed: int,
    output_root: Path,
    skip_completed: bool,
    max_steps: int,
    merge_steps: List[int],
) -> Dict:
    run_output_dir = output_root / f"{exp.name}_seed{seed}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_output_dir / "metrics.json"

    if result_file.exists() and skip_completed:
        print(f"[Skip] {exp.name} seed {seed} (result file exists).")
        metrics = json.load(open(result_file))
        loss_curve, loss_curve_path = ensure_loss_curve(run_output_dir, max_steps)
        eval_curve, eval_curve_path = ensure_eval_curve(run_output_dir, max_steps)
        merge_eval_curve, merge_eval_path = ensure_merge_eval_curve(
            run_output_dir, merge_steps, eval_curve
        )
        return {
            "method": exp.display_name,
            "seed": seed,
            "metrics": metrics,
            "result_file": str(result_file),
            "output_dir": str(run_output_dir),
            "train_time_sec": None,
            "loss_curve": loss_curve,
            "loss_curve_file": loss_curve_path,
            "eval_curve": eval_curve,
            "eval_curve_file": eval_curve_path,
            "merge_eval_curve": merge_eval_curve,
            "merge_eval_curve_file": merge_eval_path,
        }

    cmd = (
        ["python", str(RUN_SCRIPT)]
        + base_args
        + exp.extra_args
        + [
            "--output_dir",
            str(run_output_dir),
            "--result_file",
            str(result_file),
            "--seed",
            str(seed),
            "--train_set_seed",
            str(seed),
        ]
    )

    print(f"[Run] {' '.join(cmd)}")
    start_time = time.time()
    subprocess.run(cmd, check=True)
    duration = time.time() - start_time

    metrics = json.load(open(result_file))
    loss_curve, loss_curve_path = ensure_loss_curve(run_output_dir, max_steps)
    eval_curve, eval_curve_path = ensure_eval_curve(run_output_dir, max_steps)
    merge_eval_curve, merge_eval_path = ensure_merge_eval_curve(
        run_output_dir, merge_steps, eval_curve
    )
    return {
        "method": exp.display_name,
        "seed": seed,
        "metrics": metrics,
        "result_file": str(result_file),
        "output_dir": str(run_output_dir),
        "train_time_sec": duration,
        "loss_curve": loss_curve,
        "loss_curve_file": loss_curve_path,
        "eval_curve": eval_curve,
        "eval_curve_file": eval_curve_path,
        "merge_eval_curve": merge_eval_curve,
        "merge_eval_curve_file": merge_eval_path,
    }


def ensure_loss_curve(run_dir: Path, max_steps: int) -> Tuple[List[Dict[str, float]], str]:
    """
    Ensure there is a saved loss curve (step vs loss) for the run.
    Returns the curve list and the path to the json file.
    """
    curve_file = run_dir / "loss_curve.json"
    
    # First, try to read from the custom training_loss.jsonl file
    training_loss_jsonl = run_dir / "training_loss.jsonl"
    curve: List[Dict[str, float]] = []
    
    print(f"[Debug] Looking for training_loss.jsonl at: {training_loss_jsonl}")
    print(f"[Debug] training_loss.jsonl exists: {training_loss_jsonl.exists()}")
    
    if training_loss_jsonl.exists():
        try:
            with open(training_loss_jsonl, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        curve.append({
                            "step": int(entry["step"]),
                            "loss": float(entry["loss"]),
                            "time": None  # Will be filled later if needed
                        })
            print(f"[Debug] Loaded {len(curve)} loss points from training_loss.jsonl")
        except Exception as e:
            print(f"[Warn] Error reading training_loss.jsonl: {e}")
    
    # Fallback to trainer_state.json if jsonl file doesn't exist or is empty
    if not curve:
        trainer_state_path = run_dir / "trainer_state.json"
        print(f"[Debug] Falling back to trainer_state.json at: {trainer_state_path}")
        print(f"[Debug] trainer_state.json exists: {trainer_state_path.exists()}")
        
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                
                print(f"[Debug] Loaded trainer_state.json, has log_history: {'log_history' in state}")
                if 'log_history' in state:
                    print(f"[Debug] log_history length: {len(state['log_history'])}")
                
                for entry in state.get("log_history", []):
                    if "loss" in entry and "step" in entry:
                        curve.append({
                            "step": int(entry["step"]), 
                            "loss": float(entry["loss"]), 
                            "time": None
                        })
                
                print(f"[Debug] Extracted {len(curve)} loss points from trainer_state.json")
                            
            except Exception as e:
                print(f"[Warn] Error processing {trainer_state_path}: {e}")
        else:
            print(f"[Warn] trainer_state.json not found")
            # List what files DO exist in the directory
            if run_dir.exists():
                files = [f.name for f in run_dir.iterdir()]
                print(f"[Debug] Files in {run_dir}: {files}")
            
    if not curve and curve_file.exists():
        # Fallback to existing curve file
        print(f"[Debug] Using existing curve file: {curve_file}")
        return json.load(open(curve_file)), str(curve_file)

    if curve:
        curve.sort(key=lambda x: x["step"])
        json.dump(curve, open(curve_file, "w"), indent=2)
        print(f"[Debug] Saved {len(curve)} points to {curve_file}")
    else:
        print(f"[Warn] No loss data found for {run_dir}")
    
    return curve, str(curve_file)


def ensure_eval_curve(run_dir: Path, max_steps: int) -> Tuple[List[Dict[str, float]], str]:
    """
    Save evaluation loss curve (step vs eval_loss) with approximate time.
    """
    curve_file = run_dir / "eval_curve.json"
    
    # First, try to read from the custom eval_loss.jsonl file
    eval_loss_jsonl = run_dir / "eval_loss.jsonl"
    curve: List[Dict[str, float]] = []
    
    print(f"[Debug] Looking for eval_loss.jsonl at: {eval_loss_jsonl}")
    
    if eval_loss_jsonl.exists():
        try:
            with open(eval_loss_jsonl, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        curve.append({
                            "step": int(entry["step"]),
                            "eval_loss": float(entry["eval_loss"]),
                            "time": None
                        })
            print(f"[Debug] Loaded {len(curve)} eval points from eval_loss.jsonl")
        except Exception as e:
            print(f"[Warn] Error reading eval_loss.jsonl: {e}")
    
    # Fallback to trainer_state.json
    if not curve:
        trainer_state_path = run_dir / "trainer_state.json"
        
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                
                for entry in state.get("log_history", []):
                    if "eval_loss" in entry and "step" in entry:
                        curve.append({
                            "step": int(entry["step"]),
                            "eval_loss": float(entry["eval_loss"]),
                            "time": None,
                        })
                
                print(f"[Debug] Extracted {len(curve)} eval points from trainer_state.json")
            except Exception as e:
                print(f"[Warn] Error processing eval curve from {trainer_state_path}: {e}")
    
    if not curve and curve_file.exists():
        return json.load(open(curve_file)), str(curve_file)
    
    if curve:
        curve.sort(key=lambda x: x["step"])
        json.dump(curve, open(curve_file, "w"), indent=2)
        print(f"[Debug] Saved {len(curve)} eval points to {curve_file}")
    
    return curve, str(curve_file)


def ensure_merge_eval_curve(
    run_dir: Path, merge_steps: List[int], eval_curve: List[Dict[str, float]]
) -> Tuple[List[Dict[str, float]], str]:
    file_path = run_dir / "merge_eval_curve.json"
    if file_path.exists():
        data = json.load(open(file_path))
        return data, str(file_path)

    merge_set = set(merge_steps)
    data = [point for point in eval_curve if point["step"] in merge_set]
    json.dump(data, open(file_path, "w"), indent=2)
    return data, str(file_path)


def extract_train_runtime(state: Dict) -> Optional[float]:
    runtime = state.get("train_runtime")
    if runtime is not None:
        return runtime
    for entry in reversed(state.get("log_history", [])):
        if "train_runtime" in entry:
            return entry["train_runtime"]
    return None


def extract_main_metric(metrics: Dict) -> float:
    priority_keys = ["accuracy", "f1", "macro_f1", "micro_f1"]
    for key in priority_keys:
        if key in metrics:
            return metrics[key]
    # Fallback to the first float value
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"Could not infer main metric from metrics: {metrics}")


def summarize_results(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for method in {r["method"] for r in records}:
        method_records = [r for r in records if r["method"] == method]
        metrics = [extract_main_metric(r["metrics"]) for r in method_records]
        times = [r["train_time_sec"] for r in method_records if r["train_time_sec"] is not None]
        summary[method] = {
            "metric_mean": float(np.mean(metrics)),
            "metric_std": float(np.std(metrics)),
            "time_mean": float(np.mean(times)) if times else float("nan"),
            "time_std": float(np.std(times)) if times else float("nan"),
        }
    return summary


def plot_summary(summary: Dict[str, Dict[str, float]], figure_path: Path) -> None:
    methods = list(summary.keys())
    metric_means = [summary[m]["metric_mean"] for m in methods]
    metric_stds = [summary[m]["metric_std"] for m in methods]
    time_means = [summary[m]["time_mean"] for m in methods]
    time_stds = [summary[m]["time_std"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(methods, metric_means, yerr=metric_stds, capsize=5, color=["#4c72b0", "#dd8452"])
    axes[0].set_ylabel("Dev Metric")
    axes[0].set_title("Performance (mean ± std)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(methods, time_means, yerr=time_stds, capsize=5, color=["#55a868", "#c44e52"])
    axes[1].set_ylabel("Training Time (s)")
    axes[1].set_title("Training Time (mean ± std)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("XGBLoRA+ZO vs LoRA+ZO Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    print(f"[Info] Saved comparison figure to {figure_path}")


def plot_loss_trends(records: List[Dict], figure_path: Path) -> None:
    print(f"[Debug] plot_loss_trends called with {len(records)} records")
    
    method_points: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        method = rec["method"]
        loss_curve = rec.get("loss_curve", [])
        print(f"[Debug] {method}: {len(loss_curve)} loss points")
        
        for point in loss_curve:
            step = int(point["step"])
            loss = float(point["loss"])
            method_points[method][step].append(loss)

    if not method_points:
        print("[Warn] No loss curves found; skipping loss plot.")
        return

    print(f"[Debug] Plotting loss trends for methods: {list(method_points.keys())}")
    
    plt.figure(figsize=(8, 5))
    for method, step_dict in method_points.items():
        steps = sorted(step_dict.keys())
        losses = [np.mean(step_dict[step]) for step in steps]
        print(f"[Debug] {method}: plotting {len(steps)} points")
        plt.plot(steps, losses, marker='o' if len(steps) < 50 else None, label=method, markersize=3)

    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Steps")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print(f"[Info] Saved loss curve figure to {figure_path}")


def plot_merge_eval_trends(
    records: List[Dict], merge_steps: List[int], figure_path: Path
) -> None:
    print(f"[Debug] plot_merge_eval_trends called with {len(records)} records")
    print(f"[Debug] Expected merge steps: {merge_steps}")
    
    method_points: Dict[str, Tuple[List[int], List[float]]] = {}
    merge_set = set(merge_steps)

    for rec in records:
        method = rec["method"]
        curve = rec.get("merge_eval_curve", [])
        print(f"[Debug] {method}: {len(curve)} merge eval points")
        
        if not curve:
            continue
        steps = [pt["step"] for pt in curve if pt["step"] in merge_set]
        losses = [pt["eval_loss"] for pt in curve if pt["step"] in merge_set]
        if steps:
            method_points[method] = (steps, losses)
            print(f"[Debug] {method}: keeping {len(steps)} points at merge boundaries")

    if not method_points:
        print("[Warn] No evaluation-at-merge data found; skipping merge plot.")
        return

    print(f"[Debug] Plotting merge eval trends for methods: {list(method_points.keys())}")
    
    plt.figure(figsize=(8, 5))
    for method, (steps, losses) in method_points.items():
        plt.plot(steps, losses, marker="o", label=method, markersize=6)

    plt.xlabel("Merge Step")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss at Merge Points")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print(f"[Info] Saved merge evaluation figure to {figure_path}")


if __name__ == "__main__":
    args = parse_args()
    base_args = build_base_args(args)
    output_root = Path(args.output_root).expanduser().resolve()

    experiments = [
        ExperimentConfig(
            name="xgblora_zo",
            display_name="XGBLoRA+ZO",
            extra_args=[
                "--xgblora",
                "--xgblora_steps_per_iteration",
                str(args.xgblora_steps_per_iteration),
                "--xgblora_merge_frequency",
                "1",
            ],
        ),
        ExperimentConfig(
            name="lora_zo",
            display_name="LoRA+ZO",
            extra_args=[
                "--lora",
                "--lora_r",
                str(args.lora_rank),
            ],
        ),
    ]

    all_records = []
    merge_steps = (
        list(range(args.xgblora_steps_per_iteration, args.max_steps + 1, args.xgblora_steps_per_iteration))
        if args.xgblora_steps_per_iteration > 0
        else []
    )
    for exp in experiments:
        for seed in args.seeds:
            record = run_single_experiment(
                base_args,
                exp,
                seed,
                output_root,
                args.skip_completed,
                args.max_steps,
                merge_steps,
            )
            all_records.append(record)

    summary = summarize_results(all_records)
    print("\n=== Summary ===")
    for method, stats in summary.items():
        metric_mean = stats["metric_mean"]
        metric_std = stats["metric_std"]
        time_mean = stats["time_mean"]
        time_std = stats["time_std"]
        print(
            f"{method}: metric={metric_mean:.4f} ± {metric_std:.4f}, "
            f"time={time_mean:.1f}s ± {time_std:.1f}s"
        )

    plot_summary(summary, Path(args.figure_path).expanduser())
    plot_loss_trends(all_records, Path(args.loss_figure_path).expanduser())
    if merge_steps:
        plot_merge_eval_trends(
            all_records, merge_steps, Path(args.merge_eval_figure_path).expanduser()
        )


