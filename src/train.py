"""
End-to-end pipeline:
  1. Load + preprocess the Elec2 stream
  2. Warm-up train the MLP, then run it prequentially over the rest of the
     stream, collecting confidence/entropy/error signals
  3. Run ADWIN and DDM on the error stream (statistical baselines)
  4. Build sliding-window sequence datasets and train the LSTM and GRU
     degradation forecasters
  5. Evaluate everything and save a comparison plot + metrics table

Run from the `project/` directory:  python -m src.train
"""
import os
import numpy as np

from src.preprocessing import build_stream
from src.mlp import MLP, run_stream
from src.drift_detectors import run_adwin, run_ddm
from src.sequence_dataset import make_windows
from src.rnn_layers import RNNDriftDetector
from src.evaluate import (classification_metrics, detection_delay, false_alarm_rate,
                           plot_signal_and_alarms, print_comparison_table)


def main(seed: int = 0):
    os.makedirs("outputs", exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. Data
    X, y, warmup_idx, scaler = build_stream("data/electricity.csv", warmup_frac=0.1)
    print(f"Stream length: {len(X)}, warmup: {warmup_idx}")

    # 2. MLP warm-up + prequential streaming run
    mlp = MLP(n_in=X.shape[1], n_hidden=32, n_classes=2, lr=0.003, dropout=0.2, seed=seed)
    mlp.fit_batch(X[:warmup_idx], y[:warmup_idx], epochs=20, batch_size=32)

    preds, errors, confs, ents = run_stream(mlp, X, y, start=warmup_idx, online_update=True)
    print(f"Streaming accuracy after warm-up: {1 - errors.mean():.4f}")

    # 3. Statistical baselines
    adwin_flags = run_adwin(errors)
    ddm_flags, ddm_warn = run_ddm(errors)
    print(f"ADWIN alarms: {adwin_flags.sum()}, DDM alarms: {ddm_flags.sum()}")

    # 4. Sequence dataset + LSTM / GRU forecasters
    Xw, yw, centers = make_windows(confs, ents, errors, window_len=20, horizon=20,
                                    degradation_delta=0.15)
    print(f"Sequence windows: {Xw.shape}, positive rate: {yw.mean():.3f}")

    split = int(len(Xw) * 0.7)
    Xw_train, yw_train = Xw[:split], yw[:split]
    Xw_test, yw_test, centers_test = Xw[split:], yw[split:], centers[split:]

    results = {}
    forecasts = {}
    for cell_type in ("lstm", "gru"):
        model = RNNDriftDetector(cell_type=cell_type, n_in=2, n_hidden=16, lr=0.01, seed=seed)
        idx = rng.permutation(len(Xw_train))
        for epoch in range(5):
            losses = []
            for i in idx:
                loss = model.train_step(Xw_train[i], yw_train[i])
                losses.append(loss)
            print(f"[{cell_type.upper()}] epoch {epoch+1}/5 loss={np.mean(losses):.4f}")

        probs = np.array([model.predict(w) for w in Xw_test])
        results[cell_type.upper()] = classification_metrics(yw_test, probs)
        forecasts[cell_type] = probs

    print_comparison_table(results)

    # 5. Detection delay / false alarm rate, if a clear injected drift exists
    #    (works out of the box on synthetic streams; on raw Elec2, which has
    #    natural recurring drift, treat this section as illustrative).
    true_drift_index = warmup_idx + len(errors) // 2  # placeholder marker
    true_windows = [(true_drift_index - 50, true_drift_index + 200)]

    for name, flags in [("ADWIN", np.where(adwin_flags)[0]),
                         ("DDM", np.where(ddm_flags)[0])]:
        delay = detection_delay(flags, true_drift_index - warmup_idx)
        far = false_alarm_rate(flags, [(s - warmup_idx, e - warmup_idx) for s, e in true_windows],
                                len(errors))
        print(f"{name}: detection delay={delay}, false alarm rate={far:.3f}")

    lstm_alarm_idx = centers_test[forecasts["lstm"] >= 0.5]
    gru_alarm_idx = centers_test[forecasts["gru"] >= 0.5]
    for name, flags in [("LSTM", lstm_alarm_idx), ("GRU", gru_alarm_idx)]:
        delay = detection_delay(np.array(flags), true_drift_index - warmup_idx)
        print(f"{name}: detection delay={delay}")

    # 6. Plot
    out_path = plot_signal_and_alarms(
        confs, ents, errors, adwin_flags, ddm_flags,
        lstm_prob=forecasts["lstm"], lstm_centers=centers_test,
        gru_prob=forecasts["gru"], gru_centers=centers_test,
        true_drift_index=true_drift_index - warmup_idx,
        out_path="outputs/signals.png",
    )
    print(f"Saved plot to {out_path}")

    return results


if __name__ == "__main__":
    main()
