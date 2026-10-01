from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

data = np.load("data/processed/fashion_mnist_processed.npz")

x_train = data["x_train"]
y_train = data["y_train"]
x_val = data["x_val"]
y_val = data["y_val"]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(params["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(params["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=params["learning_rate"]
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"]
)

Path("models").mkdir(exist_ok=True)

model.save("models/model.h5")
pd.DataFrame(history.history).to_csv(
    "models/history.csv",
    index=False
)

print("Training complete.")
print("Model saved to models/model.h5")
print("History saved to models/history.csv")