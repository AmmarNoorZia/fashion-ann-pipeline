from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["preprocess"]

raw = np.load("data/raw/fashion_mnist.npz")

x_train = raw["x_train"].astype("float32") / 255.0
y_train = raw["y_train"]
x_test = raw["x_test"].astype("float32") / 255.0
y_test = raw["y_test"]

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=y_train
)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output_dir / "fashion_mnist_processed.npz",
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test
)

print("Preprocessing complete.")
print("Train:", x_train.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)
print("Pixel range:", x_train.min(), "to", x_train.max())