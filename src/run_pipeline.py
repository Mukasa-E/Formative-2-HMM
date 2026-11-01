#usage: python -m src.run_pipeline

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
SAMPLING_RATE_HZ = 50
ACTIVITIES = ["standing", "walking", "jumping", "still"]

from features import extract_all_features_from_dir, z_score_normalize
from hmm import DiscreteHMM
from visualization import (
    plot_transition_matrix,
    plot_emission_matrix,
    plot_decoded_sequence,
    plot_confusion_matrix,
    plot_sample_signals,
    plot_log_likelihood,
    compute_metrics,
)

FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

WINDOW_SEC = 2.0  
N_OBS = 20               
N_HIDDEN = 4              


def main():
    print("=" * 60)
    print("HMM Human Activity Recognition - End-to-End Pipeline")
    print("=" * 60)

    print("\n[Step 0] Using existing real data in data/train/ and data/test/")

   
    print("\n[Step 1] Extracting features from training data...")
    X_train_raw, y_train, train_info, feature_names = extract_all_features_from_dir(
        TRAIN_DIR, window_sec=WINDOW_SEC
    )
    print(f"  Training windows: {X_train_raw.shape[0]}")
    print(f"  Features: {feature_names}")

   
    X_train, mean, std = z_score_normalize(X_train_raw)
    print(f"  Z-score normalization applied (fit on training data).")

    
    print(f"\n[Step 2] Discretising observations into {N_OBS} symbols...")
    X_train_symbols, bin_edges = DiscreteHMM.encode_observations(
        X_train, n_obs=N_OBS, random_state=42
    )

    
    train_lengths = [info["n_windows"] for info in train_info]
    print(f"  Number of training sequences (files): {len(train_lengths)}")

    print(f"\n[Step 3] Training Discrete HMM (Baum-Welch)...")
    print(f"  Hidden states: {N_HIDDEN}, Observation symbols: {N_OBS}")
    model = DiscreteHMM(n_hidden=N_HIDDEN, n_obs=N_OBS, random_state=42)
    model.fit(
        X_train_symbols,
        lengths=train_lengths,
        n_iter=500,
        tol=1e-4,
        verbose=True,
    )
    print(f"  Training complete. Converged: {model.converged_}, iterations: {model.n_iter_}")

    print("\n[Step 4] Evaluating on unseen test data...")
    X_test_raw, y_test, test_info, _ = extract_all_features_from_dir(
        TEST_DIR, window_sec=WINDOW_SEC
    )

    X_test, _, _ = z_score_normalize(X_test_raw, mean=mean, std=std)
    X_test_symbols, _ = DiscreteHMM.encode_observations(X_test, n_obs=N_OBS, random_state=42)

    test_lengths = [info["n_windows"] for info in test_info]

    train_pred_states = model.predict(X_train_symbols)

    from collections import Counter
    state_to_activity = {}
    for s in range(N_HIDDEN):
        mask = train_pred_states == s
        if mask.sum() > 0:
            most_common = Counter(y_train[mask]).most_common(1)[0][0]
            state_to_activity[s] = most_common
        else:
            state_to_activity[s] = ACTIVITIES[s]  # fallback

    activity_to_state = {a: s for s, a in state_to_activity.items()}

    y_test_pred = []
    y_test_true = []
    seq_idx = 0
    for info in test_info:
        n_win = info["n_windows"]
        seq_symbols = X_test_symbols[seq_idx:seq_idx + n_win]
        pred_states = model.predict(seq_symbols)
        pred_activities = [state_to_activity[s] for s in pred_states]
        y_test_pred.extend(pred_activities)
        y_test_true.extend([info["activity"]] * n_win)
        seq_idx += n_win

    y_test_pred = np.array(y_test_pred)
    y_test_true = np.array(y_test_true)

    
    metrics = compute_metrics(y_test_true, y_test_pred, activities=ACTIVITIES)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Unseen Test Data)")
    print("=" * 60)
    print(f"{'Activity':<12} {'Sensitivity':>12} {'Specificity':>12} {'Support':>10}")
    print("-" * 60)
    for activity in ACTIVITIES:
        m = metrics[activity]
        print(f"{activity:<12} {m['sensitivity']:>12.3f} {m['specificity']:>12.3f} {m['support']:>10}")
    print("-" * 60)
    print(f"{'OVERALL ACCURACY':<12} {metrics['overall']['accuracy']:>12.3f}")
    print("=" * 60)

    
    print("\n[Step 6] Generating visualizations...")

    plot_transition_matrix(
        model.transmat_,
        activities=ACTIVITIES,
        save_path=os.path.join(FIG_DIR, "transition_matrix.png"),
    )

    plot_emission_matrix(
        model.emissionprob_,
        activities=ACTIVITIES,
        save_path=os.path.join(FIG_DIR, "emission_matrix.png"),
    )

    plot_log_likelihood(
        model.log_likelihoods_,
        save_path=os.path.join(FIG_DIR, "convergence.png"),
    )

    plot_confusion_matrix(
        y_test_true,
        y_test_pred,
        activities=ACTIVITIES,
        save_path=os.path.join(FIG_DIR, "confusion_matrix.png"),
        title="Confusion Matrix - Unseen Test Data",
    )

    first_test = test_info[0]
    n_win = first_test["n_windows"]
    seq_symbols = X_test_symbols[:n_win]
    pred_states = model.predict(seq_symbols)
    true_state_indices = [activity_to_state[a] for a in [first_test["activity"]] * n_win]
    plot_decoded_sequence(
        np.array(true_state_indices),
        pred_states,
        activities=ACTIVITIES,
        save_path=os.path.join(FIG_DIR, "decoded_sequence.png"),
    )

    plot_sample_signals(
        TRAIN_DIR,
        activities=ACTIVITIES,
        save_path=os.path.join(FIG_DIR, "sample_signals.png"),
        fs=SAMPLING_RATE_HZ,
    )

    print(f"  Figures saved to: {FIG_DIR}/")
    print(f"    transition_matrix.png, emission_matrix.png, convergence.png,")
    print(f"    confusion_matrix.png, decoded_sequence.png, sample_signals.png")

    
    print("\n[Pipeline Complete]")
    print(f"  Training files: {len(train_lengths)}  ({N_HIDDEN} activities x 10 files)")
    print(f"  Test (unseen) files: {len(test_lengths)}  ({N_HIDDEN} activities x 2 files)")
    print(f"  Overall accuracy on unseen data: {metrics['overall']['accuracy']:.3f}")
    print(f"  HMM converged: {model.converged_} after {model.n_iter_} EM iterations")


if __name__ == "__main__":
    main()