from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from api.schemas import STAGE_LABELS
from training.model import build_cnn_lstm_model, save_model


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset = np.load(path)
    required = {"X_train", "y_train", "X_val", "y_val"}
    missing = sorted(required.difference(dataset.files))
    if missing:
        raise ValueError(f"Dataset is missing required arrays: {', '.join(missing)}")
    return {name: dataset[name] for name in dataset.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN-LSTM sleep stage classifier.")
    parser.add_argument("--dataset", required=True, help="Path to .npz file with train/val arrays.")
    parser.add_argument("--output", default="models/sleep_model.keras", help="Saved model path.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.dataset)

    model = build_cnn_lstm_model(
        input_shape=data["X_train"].shape[1:],
        num_classes=len(STAGE_LABELS),
        learning_rate=args.learning_rate,
    )
    model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if {"X_test", "y_test"}.issubset(data):
        predictions = model.predict(data["X_test"]).argmax(axis=1)
        print(classification_report(data["y_test"], predictions, target_names=STAGE_LABELS))

    save_model(model, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

