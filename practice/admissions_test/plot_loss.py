import numpy as np
import matplotlib.pyplot as plt

data = np.load("data_output.npy")

iterations = data[:, 0]
train_loss = data[:, 1]
val_loss = data[:, 2]

# https://discuss.pytorch.org/t/how-to-plot-train-and-validation-accuracy-graph/105524
plt.figure(figsize=(6, 4))
plt.title("Training and Validation Loss")
plt.plot(train_loss, label="train")
plt.plot(val_loss, label="val")
plt.xlabel("iterations (per 500)")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss-admissions.png", dpi=300)
