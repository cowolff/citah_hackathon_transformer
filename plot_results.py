import json
import matplotlib.pyplot as plt
import pandas as pd

data = json.load(open("results/history.json"))

plt.figure(figsize=(12, 6))

loss_data = data['train_loss']
accuracy_data = data['train_acc']

# Smooth loss using pandas
loss_data = pd.Series(loss_data).rolling(5).mean().tolist()

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Train Loss", color="tab:blue")
ax1.plot(loss_data, label="Smoothed Train Loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Train Accuracy
ax2 = ax1.twinx()
ax2.set_ylabel("Train Accuracy", color="tab:orange")
ax2.plot(accuracy_data, label="Train Accuracy", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

# Title and legend
fig.suptitle("Training Loss and Accuracy")
fig.tight_layout()
plt.savefig("results/loss_accuracy.png")