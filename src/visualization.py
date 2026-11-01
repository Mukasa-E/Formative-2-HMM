
# run with `python -m src.visualization` to test visualization functions

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ACTIVITIES = ["standing", "walking", "jumping", "still"]


def plot_transition_matrix(transmat, activities=ACTIVITIES, save_path=None):
    """Heatmap of state transition probabilities."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=activities,
        yticklabels=activities,
        ax=ax,
        cbar_kws={"label": "Transition probability"},
    )
    ax.set_xlabel("Next state (activity)")
    ax.set_ylabel("Current state (activity)")
    ax.set_title("Learned Transition Probability Matrix (A)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_emission_matrix(emissionprob, activities=ACTIVITIES, save_path=None):
    """Heatmap of emission probabilities (hidden state -> observation symbol)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        emissionprob,
        annot=False,
        cmap="viridis",
        xticklabels=[f"Obs {i}" for i in range(emissionprob.shape[1])],
        yticklabels=activities,
        ax=ax,
        cbar_kws={"label": "Emission probability"},
    )
    ax.set_xlabel("Observation symbol")
    ax.set_ylabel("Hidden state (activity)")
    ax.set_title("Learned Emission Probability Matrix (B)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_decoded_sequence(true_states, pred_states, activities=ACTIVITIES, save_path=None):
    """Plot decoded hidden state sequence alongside ground truth."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    time = np.arange(len(true_states))

    axes[0].step(time, true_states, where="mid", color="steelblue", linewidth=2)
    axes[0].set_yticks(range(len(activities)))
    axes[0].set_yticklabels(activities)
    axes[0].set_ylabel("True activity")
    axes[0].set_title("Ground Truth Activity Sequence")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(time, pred_states, where="mid", color="darkorange", linewidth=2)
    axes[1].set_yticks(range(len(activities)))
    axes[1].set_yticklabels(activities)
    axes[1].set_xlabel("Time window")
    axes[1].set_ylabel("Predicted activity")
    axes[1].set_title("Viterbi-Decoded Activity Sequence")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, activities=ACTIVITIES, save_path=None, title="Confusion Matrix"):
    """Confusion matrix heatmap. Returns (fig, cm)."""
    labels = list(activities)
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    label_to_idx = {a: i for i, a in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig, cm


def plot_sample_signals(data_dir, activities=ACTIVITIES, save_path=None, fs=50):
    """Plot one sample raw accelerometer signal per activity."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, activity in zip(axes, activities):
        activity_dir = os.path.join(data_dir, activity)
        files = sorted([f for f in os.listdir(activity_dir) if f.endswith(".csv")])
        if not files:
            continue
        filepath = os.path.join(activity_dir, files[0])

        timestamps = []
        accel = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line.startswith("timestamp"):
                    continue
                parts = line.split(",")
                timestamps.append(float(parts[0]))
                accel.append([float(parts[1]), float(parts[2]), float(parts[3])])

        timestamps = np.array(timestamps)
        accel = np.array(accel)

        for i, axis in enumerate(["x", "y", "z"]):
            ax.plot(timestamps, accel[:, i], label=f"accel_{axis}", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.set_title(activity.capitalize())
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Sample Raw Accelerometer Signals per Activity", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_log_likelihood(log_likelihoods, save_path=None):
    """Plot Baum-Welch log-likelihood convergence curve."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(log_likelihoods) + 1), log_likelihoods,
            marker="o", color="steelblue")
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Baum-Welch Convergence (Log-Likelihood vs Iteration)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def compute_metrics(y_true, y_pred, activities=ACTIVITIES):
    """
    Compute per-class sensitivity, specificity, and overall accuracy.
    Returns a dict with per-class metrics and overall accuracy + confusion matrix.
    """
    label_to_idx = {a: i for i, a in enumerate(activities)}
    n = len(activities)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    results = {}
    for i, activity in enumerate(activities):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[activity] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "support": int(cm[i, :].sum()),
        }

    overall_accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    results["overall"] = {"accuracy": float(overall_accuracy), "cm": cm}
    return results