from __future__ import annotations

from pathlib import Path


def build_cnn_lstm_model(
    input_shape: tuple[int, int] = (3000, 1),
    num_classes: int = 5,
    learning_rate: float = 1e-3,
):
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "TensorFlow is required for training. Install it with "
            "`pip install -r requirements-ml.txt`."
        ) from exc

    inputs = keras.Input(shape=input_shape, name="eeg_window")
    x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="sleep_stage")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="sleep_stage_cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model(model, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

