import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_json_step_loss(path: Path):
    """Load step/loss pairs from a JSON file with objects containing 'step' and 'loss'."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure sorted by step
    data = sorted(data, key=lambda x: x["step"])
    steps = [item["step"] for item in data]
    losses = [item["loss"] for item in data]
    return steps, losses


def load_csv_step_loss(path: Path):
    """Load step/loss pairs from a CSV file with header: step,loss."""
    steps = []
    losses = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows if any
            if not row.get("step") or not row.get("loss"):
                continue
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


def main():
    base_dir = Path(__file__).resolve().parent / "experiment_results"

    # =========================
    # Line plot: loss vs. steps
    # =========================

    # 1. XGBLoRA + ZO, alpha = 16 (JSON)
    boosting_json = base_dir / "step_loss_boosting_a16_10.json"
    boosting_steps, boosting_losses = load_json_step_loss(boosting_json)

    # 1. XGBLoRA + ZO, alpha = 16 (JSON)
    boosting_json_2 = base_dir / "step_loss_boosting_a16_100.json"
    boosting_steps_2, boosting_losses_2 = load_json_step_loss(boosting_json_2)

    # 1. XGBLoRA + ZO, alpha = 16 (JSON)
    boosting_json_3 = base_dir / "step_loss_boosting_a16_300.json"
    boosting_steps_3, boosting_losses_3 = load_json_step_loss(boosting_json_3)

    # 1. XGBLoRA + ZO, alpha = 16 (JSON)
    boosting_json_4 = base_dir / "step_loss_boosting_a16_500.json"
    boosting_steps_4, boosting_losses_4 = load_json_step_loss(boosting_json_4)

    # 1. XGBLoRA + ZO, alpha = 16 (JSON)
    boosting_json_5 = base_dir / "step_loss_boosting_a16_2000.json"
    boosting_steps_5, boosting_losses_5 = load_json_step_loss(boosting_json_5)


    plt.figure(figsize=(10, 5))

    plt.plot(
        boosting_steps,
        boosting_losses,
        label="XGBLoRA + ZO (α=16, 10 steps merge)",
        linewidth=2,
    )
    
    plt.plot(
        boosting_steps_2,
        boosting_losses_2,
        label="XGBLoRA + ZO (α=16, 100 steps merge)",
        linewidth=2,
    )

    plt.plot(
        boosting_steps_3,
        boosting_losses_3,
        label="XGBLoRA + ZO (α=16, 300 steps merge)",
        linewidth=2,
    )

    plt.plot(
        boosting_steps_4,
        boosting_losses_4,
        label="XGBLoRA + ZO (α=16, 500 steps merge)",
        linewidth=2,
    )

    plt.plot(
        boosting_steps_5,
        boosting_losses_5,
        label="XGBLoRA + ZO (α=16, 2000 steps merge)",
        linewidth=2,
    )

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training loss per step (α=16)")

    # Vertical red lines every 100 steps
    max_step = max(
        boosting_steps[-1],
        boosting_steps_2[-1],
    )
    \
    plt.axvline(x=10, color="blue", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(x=100, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(x=300, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(x=500, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(x=2000, color="purple", linestyle="--", linewidth=0.8, alpha=0.6)
    # for step in range(100, max_step + 1, 100):
    #     plt.axvline(x=step, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    # for step in range(500, max_step + 1, 100):
    #     plt.axvline(x=step, color="green", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = base_dir / "step_loss_comparison_a16.png"
    plt.savefig(output_path, dpi=150)

    # =========================
    # Bar plot: accuracy
    # =========================

    # Metrics JSON files
    metrics_boosting = base_dir / "metrics_boosting_a16_10.json"
    metrics_boosting_2 = base_dir / "metrics_boosting_a16_100.json"
    metrics_boosting_3 = base_dir / "metrics_boosting_a16_300.json"
    metrics_boosting_4 = base_dir / "metrics_boosting_a16_500.json"
    metrics_boosting_5 = base_dir / "metrics_boosting_a16_2000.json"

    def load_accuracy(path: Path) -> float:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["accuracy"])

    accuracies = [
        load_accuracy(metrics_boosting),
        load_accuracy(metrics_boosting_2),
        load_accuracy(metrics_boosting_3),
        load_accuracy(metrics_boosting_4),
        load_accuracy(metrics_boosting_5),
    ]

    labels = [
        "XGBLoRA + ZO (α=16, 10 steps merge)",
        "XGBLoRA + ZO (α=16, 100 steps merge)",
        "XGBLoRA + ZO (α=16, 300 steps merge)",
        "XGBLoRA + ZO (α=16, 500 steps merge)",
        "XGBLoRA + ZO (α=16, 2000 steps merge)",
    ]

    plt.figure(figsize=(8, 5))
    x = range(len(labels))
    plt.bar(x, accuracies)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison (α=16)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    acc_output_path = base_dir / "accuracy_comparison_a16.png"
    plt.savefig(acc_output_path, dpi=150)

    # Also show the figures when run interactively
    plt.show()


if __name__ == "__main__":
    main()


