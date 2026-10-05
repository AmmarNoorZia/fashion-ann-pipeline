from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

np.savez_compressed(
    output_dir / "fashion_mnist.npz",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

print("Raw Fashion-MNIST saved successfully.")
print("Training:", x_train.shape, y_train.shape)
print("Testing :", x_test.shape, y_test.shape)