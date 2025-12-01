#!/usr/bin/env python3
"""
Run XGBLoRA experiments and generate comprehensive results including:
- Training and evaluation loss curves
- Evaluation metrics at merge points
- Summary statistics across multiple seeds
- Visualization plots

This is a simplified version of compare_xgblora_lora.py that focuses solely on XGBLoRA.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RUN_SCRIPT = Path(__file__).parent / "run.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XGBLoRA+ZO experiments with comprehensive logging and visualization."
    )
    parser.add_argument(
        "--model_name",
        default="facebook/opt-350m",
        help="Backbone model name (HuggingFace identifier).",
    )
    parser.add_argument("--task_name", default="SST2", help="Task to evaluate.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_dev", type=int, default=500)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=4000)
    parser.add_argument("--save_steps", type=int, default=4000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--xgblora_steps_per_iteration", 
        type=int, 
        default=1000,
        help="Number of steps per XGBLoRA boosting iteration"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16,
        help="LoRA alpha scaling parameter (XGBLoRA uses rank-1 automatically)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to run experiments with (also used for train_set_seed).",
    )
    parser.add_argument(
        "--output_root",
        default="xgblora_runs",
        help="Root directory where seed outputs will be stored.",
    )
    parser.add_argument(
        "--summary_figure",
        default="xgblora_runs/summary.png",
        help="Where to save the summary metrics plot.",
    )
    parser.add_argument(
        "--loss_figure",
        default="xgblora_runs/training_loss.png",
        help="Where to save the training loss-vs-steps plot.",
    )
    parser.add_argument(
        "--eval_figure",
        default="xgblora_runs/eval_loss.png",
        help="Where to save the evaluation loss-vs-steps plot.",
    )
    parser.add_argument(
        "--merge_eval_figure",
        default="xgblora_runs/merge_eval_loss.png",
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
    """Build the common arguments for all runs."""
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
        "--save_steps",
        str(args.save_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--overwrite_output_dir",
    ]
    if args.extra_args:
        base.extend(args.extra_args)
    return base


def run_single_experiment(
    base_args: List[str],
    seed: int,
    output_root: Path,
    skip_completed: bool,
    max_steps: int,
    merge_steps: List[int],
    xgblora_steps: int,
    lora_alpha: int,
) -> Dict:
    """Run a single XGBLoRA experiment for a given seed."""
    run_output_dir = output_root / f"seed_{seed}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_output_dir / "metrics.json"

    if result_file.exists() and skip_completed:
        print(f"[Skip] Seed {seed} (result file exists).")
        metrics = json.load(open(result_file))
        loss_curve, loss_curve_path = ensure_loss_curve(run_output_dir, max_steps)
        eval_curve, eval_curve_path = ensure_eval_curve(run_output_dir, max_steps)
        merge_eval_curve, merge_eval_path = ensure_merge_eval_curve(
            run_output_dir, merge_steps, eval_curve
        )
        return {
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

    # XGBLoRA-specific args (placed after base_args to ensure they take precedence)
    xgblora_args = [
        "--xgblora",
        "--xgblora_steps_per_iteration",
        str(xgblora_steps),
        "--lora_alpha",
        str(lora_alpha),
    ]
    
    # Only set merge_frequency for epoch-based merging (when step-based is disabled)
    if xgblora_steps == 0:
        xgblora_args.extend(["--xgblora_merge_frequency", "1"])
    else:
        # Explicitly set to 0 to disable epoch-based merging when using step-based
        xgblora_args.extend(["--xgblora_merge_frequency", "0"])
    
    cmd = (
        ["python", str(RUN_SCRIPT)]
        + base_args
        + xgblora_args
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

    print(f"\n[Run] Seed {seed}")
    print(f"[Cmd] {' '.join(cmd)}")
    start_time = time.time()
    subprocess.run(cmd, check=True)
    duration = time.time() - start_time

    metrics = json.load(open(result_file))
    loss_curve, loss_curve_path = ensure_loss_curve(run_output_dir, max_steps)
    eval_curve, eval_curve_path = ensure_eval_curve(run_output_dir, max_steps)
    merge_eval_curve, merge_eval_path = ensure_merge_eval_curve(
        run_output_dir, merge_steps, eval_curve
    )
    
    print(f"[Done] Seed {seed} completed in {duration:.1f}s")
    
    return {
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
    
    if training_loss_jsonl.exists():
        try:
            with open(training_loss_jsonl, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        curve.append({
                            "step": int(entry["step"]),
                            "loss": float(entry["loss"]),
                            "time": entry.get("time")
                        })
            print(f"  [Info] Loaded {len(curve)} training loss points from training_loss.jsonl")
        except Exception as e:
            print(f"  [Warn] Error reading training_loss.jsonl: {e}")
    
    # Fallback to trainer_state.json if jsonl file doesn't exist or is empty
    if not curve:
        trainer_state_path = run_dir / "trainer_state.json"
        
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                
                if 'log_history' in state:
                    log_history = state['log_history']
                    train_runtime = state.get('train_runtime', max_steps * 10)  # fallback estimate
                    
                    for entry in log_history:
                        if "loss" in entry and "step" in entry:
                            try:
                                step = int(entry["step"])
                                loss = float(entry["loss"])
                                time_sec = (step / max_steps) * train_runtime if train_runtime else None
                                curve.append({
                                    "step": step,
                                    "loss": loss,
                                    "time": time_sec
                                })
                            except (ValueError, TypeError) as e:
                                continue
                    
                    print(f"  [Info] Loaded {len(curve)} training loss points from trainer_state.json")
            except Exception as e:
                print(f"  [Warn] Error reading trainer_state.json: {e}")
    
    # Save to curve_file
    if curve:
        with open(curve_file, 'w') as f:
            json.dump(curve, f, indent=2)
    
    return curve, str(curve_file)


def ensure_eval_curve(run_dir: Path, max_steps: int) -> Tuple[List[Dict[str, float]], str]:
    """
    Ensure there is a saved eval curve (step vs eval_loss) for the run.
    Returns the curve list and the path to the json file.
    """
    curve_file = run_dir / "eval_curve.json"
    
    # First, try to read from the custom eval_loss.jsonl file
    eval_loss_jsonl = run_dir / "eval_loss.jsonl"
    curve: List[Dict[str, float]] = []
    
    if eval_loss_jsonl.exists():
        try:
            with open(eval_loss_jsonl, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        curve.append({
                            "step": int(entry["step"]),
                            "eval_loss": float(entry["eval_loss"]),
                            "eval_accuracy": entry.get("eval_accuracy"),
                            "time": entry.get("time")
                        })
            print(f"  [Info] Loaded {len(curve)} eval loss points from eval_loss.jsonl")
        except Exception as e:
            print(f"  [Warn] Error reading eval_loss.jsonl: {e}")
    
    # Fallback to trainer_state.json
    if not curve:
        trainer_state_path = run_dir / "trainer_state.json"
        
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                
                if 'log_history' in state:
                    log_history = state['log_history']
                    train_runtime = state.get('train_runtime', max_steps * 10)
                    
                    for entry in log_history:
                        if "eval_loss" in entry and "step" in entry:
                            try:
                                step = int(entry["step"])
                                eval_loss = float(entry["eval_loss"])
                                eval_accuracy = entry.get("eval_accuracy")
                                time_sec = (step / max_steps) * train_runtime if train_runtime else None
                                curve.append({
                                    "step": step,
                                    "eval_loss": eval_loss,
                                    "eval_accuracy": eval_accuracy,
                                    "time": time_sec
                                })
                            except (ValueError, TypeError):
                                continue
                    
                    print(f"  [Info] Loaded {len(curve)} eval loss points from trainer_state.json")
            except Exception as e:
                print(f"  [Warn] Error reading trainer_state.json: {e}")
    
    # Save to curve_file
    if curve:
        with open(curve_file, 'w') as f:
            json.dump(curve, f, indent=2)
    
    return curve, str(curve_file)


def ensure_merge_eval_curve(
    run_dir: Path, merge_steps: List[int], eval_curve: List[Dict]
) -> Tuple[List[Dict[str, float]], str]:
    """
    Extract evaluation metrics at merge points from the eval curve.
    Returns the merge eval curve and the path to the json file.
    """
    curve_file = run_dir / "merge_eval_curve.json"
    merge_set = set(merge_steps)
    
    merge_curve = [pt for pt in eval_curve if pt["step"] in merge_set]
    
    if merge_curve:
        print(f"  [Info] Extracted {len(merge_curve)} eval points at merge steps")
        with open(curve_file, 'w') as f:
            json.dump(merge_curve, f, indent=2)
    
    return merge_curve, str(curve_file)


def summarize_results(records: List[Dict]) -> Dict:
    """Compute summary statistics across all seeds."""
    metrics = []
    times = []
    
    for rec in records:
        # Extract the main metric (assume it's the first numeric value in metrics dict)
        metric_dict = rec["metrics"]
        if metric_dict:
            # Try to get 'accuracy' or the first numeric value
            if "accuracy" in metric_dict:
                metrics.append(metric_dict["accuracy"])
            else:
                # Take first numeric value
                for v in metric_dict.values():
                    if isinstance(v, (int, float)):
                        metrics.append(v)
                        break
        
        if rec["train_time_sec"] is not None:
            times.append(rec["train_time_sec"])
    
    summary = {
        "metric_mean": np.mean(metrics) if metrics else 0.0,
        "metric_std": np.std(metrics) if metrics else 0.0,
        "metric_values": metrics,
        "time_mean": np.mean(times) if times else 0.0,
        "time_std": np.std(times) if times else 0.0,
        "time_values": times,
        "num_seeds": len(records)
    }
    
    return summary


def plot_summary(summary: Dict, figure_path: Path) -> None:
    """Plot summary bar chart with mean metrics and training times."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Metric (e.g., accuracy)
    ax1.bar([0], [summary["metric_mean"]], color='skyblue', width=0.5)
    ax1.errorbar([0], [summary["metric_mean"]], 
                 yerr=[summary["metric_std"]], 
                 fmt='none', color='black', capsize=10)
    ax1.set_ylabel("Task Metric (Accuracy)")
    ax1.set_title(f"XGBLoRA Performance\n({summary['num_seeds']} seeds)")
    ax1.set_xticks([0])
    ax1.set_xticklabels(["XGBLoRA+ZO"])
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Plot 2: Training time
    ax2.bar([0], [summary["time_mean"]], color='lightcoral', width=0.5)
    ax2.errorbar([0], [summary["time_mean"]], 
                 yerr=[summary["time_std"]], 
                 fmt='none', color='black', capsize=10)
    ax2.set_ylabel("Training Time (seconds)")
    ax2.set_title(f"XGBLoRA Training Time\n({summary['num_seeds']} seeds)")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["XGBLoRA+ZO"])
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print(f"\n[Info] Saved summary figure to {figure_path}")


def plot_loss_trends(records: List[Dict], figure_path: Path) -> None:
    """Plot training loss vs steps for all seeds."""
    if not records:
        print("[Warn] No records to plot.")
        return
    
    has_data = False
    plt.figure(figsize=(10, 6))
    
    for rec in records:
        curve = rec.get("loss_curve", [])
        if not curve:
            continue
        
        has_data = True
        steps = [pt["step"] for pt in curve]
        losses = [pt["loss"] for pt in curve]
        seed = rec["seed"]
        plt.plot(steps, losses, marker='o', markersize=3, label=f"Seed {seed}", alpha=0.7)
    
    if not has_data:
        print("[Warn] No loss curves found; skipping loss plot.")
        return
    
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("XGBLoRA: Training Loss vs Steps")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print(f"[Info] Saved training loss figure to {figure_path}")


def plot_eval_trends(records: List[Dict], figure_path: Path) -> None:
    """Plot evaluation loss vs steps for all seeds."""
    if not records:
        print("[Warn] No records to plot.")
        return
    
    has_data = False
    plt.figure(figsize=(10, 6))
    
    for rec in records:
        curve = rec.get("eval_curve", [])
        if not curve:
            continue
        
        has_data = True
        steps = [pt["step"] for pt in curve]
        losses = [pt["eval_loss"] for pt in curve]
        seed = rec["seed"]
        plt.plot(steps, losses, marker='s', markersize=5, label=f"Seed {seed}", alpha=0.7)
    
    if not has_data:
        print("[Warn] No eval curves found; skipping eval plot.")
        return
    
    plt.xlabel("Step")
    plt.ylabel("Evaluation Loss")
    plt.title("XGBLoRA: Evaluation Loss vs Steps")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print(f"[Info] Saved evaluation loss figure to {figure_path}")


def plot_merge_eval_trends(records: List[Dict], merge_steps: List[int], figure_path: Path) -> None:
    """Plot evaluation loss at merge points for all seeds."""
    if not records or not merge_steps:
        print("[Warn] No merge steps defined; skipping merge eval plot.")
        return
    
    has_data = False
    merge_set = set(merge_steps)
    plt.figure(figsize=(10, 6))
    
    for rec in records:
        curve = rec.get("merge_eval_curve", [])
        if not curve:
            continue
        
        steps = [pt["step"] for pt in curve if pt["step"] in merge_set]
        losses = [pt["eval_loss"] for pt in curve if pt["step"] in merge_set]
        
        if steps:
            has_data = True
            seed = rec["seed"]
            plt.plot(steps, losses, marker="o", markersize=6, label=f"Seed {seed}", alpha=0.7)
    
    if not has_data:
        print("[Warn] No evaluation-at-merge data found; skipping merge plot.")
        return
    
    plt.xlabel("Merge Step (Boosting Iteration)")
    plt.ylabel("Eval Loss")
    plt.title("XGBLoRA: Eval Loss at Merge Points")
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

    # Calculate merge steps for XGBLoRA
    merge_steps = (
        list(range(args.xgblora_steps_per_iteration, args.max_steps + 1, args.xgblora_steps_per_iteration))
        if args.xgblora_steps_per_iteration > 0
        else []
    )
    
    print("=" * 80)
    print("XGBLoRA Experiment Runner")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task_name}")
    print(f"Max Steps: {args.max_steps}")
    print(f"XGBLoRA Steps per Iteration: {args.xgblora_steps_per_iteration}")
    print(f"LoRA Alpha: {args.lora_alpha}")
    print(f"Seeds: {args.seeds}")
    print(f"Output Root: {output_root}")
    print(f"Merge Steps: {merge_steps}")
    print("=" * 80)

    # Run experiments for all seeds
    all_records = []
    for seed in args.seeds:
        record = run_single_experiment(
            base_args,
            seed,
            output_root,
            args.skip_completed,
            args.max_steps,
            merge_steps,
            args.xgblora_steps_per_iteration,
            args.lora_alpha,
        )
        all_records.append(record)

    # Summarize results
    summary = summarize_results(all_records)
    
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"Task Metric: {summary['metric_mean']:.4f} ± {summary['metric_std']:.4f}")
    print(f"Training Time: {summary['time_mean']:.1f}s ± {summary['time_std']:.1f}s")
    print(f"Number of Seeds: {summary['num_seeds']}")
    print("=" * 80)

    # Generate all plots
    plot_summary(summary, Path(args.summary_figure).expanduser())
    plot_loss_trends(all_records, Path(args.loss_figure).expanduser())
    plot_eval_trends(all_records, Path(args.eval_figure).expanduser())
    
    if merge_steps:
        plot_merge_eval_trends(
            all_records, merge_steps, Path(args.merge_eval_figure).expanduser()
        )

    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print("=" * 80)

