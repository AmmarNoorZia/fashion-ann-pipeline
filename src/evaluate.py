import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = np.load("data/processed/fashion_mnist_processed.npz")

x_test = data["x_test"]
y_test = data["y_test"]

model = tf.keras.models.load_model("models/model.h5")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(cm)
display.plot(cmap="Blues")
plt.title("Fashion-MNIST Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("metrics.json created.")
print("confusion_matrix.png created.")