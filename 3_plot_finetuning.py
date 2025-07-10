import timeit
start_time = timeit.default_timer()
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Load and parse the log file ---
log_file = 'finetune_no2025.log'  # <-- Replace with your actual log file path
output_filename = 'results/Galileo_training_without2025.png'

pattern = re.compile(
    r"\[Epoch (\d+)\] LR: ([0-9.]+), Train Loss = ([0-9.]+), Train Acc = ([0-9.]+), Val Acc = ([0-9.]+), Val mIoU = ([0-9.]+)"
)

data = []

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch, lr, train_loss, train_acc, val_acc, val_miou = match.groups()
            data.append({
                'Epoch': int(epoch),
                'LR': float(lr),
                'Train Loss': float(train_loss),
                'Train Acc': float(train_acc),
                'Val Acc': float(val_acc),
                'Val mIoU': float(val_miou)
            })

df = pd.DataFrame(data)
print('Training data log', df.head(), flush=True)

# --- Plotting ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Galileo Model Finetuning Performance over Epochs', fontsize=16)

# Plot Training and Validation Accuracy
axs[0, 0].plot(df['Epoch'], df['Train Acc'], label='Train Acc', color='blue')
axs[0, 0].plot(df['Epoch'], df['Val Acc'], label='Val Acc', color='green')
axs[0, 0].set_title('Accuracy')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()

# Plot mIoU
axs[0, 1].plot(df['Epoch'], df['Val mIoU'], label='Val mIoU', color='purple')
axs[0, 1].set_title('Mean Intersection over Union (mIoU)')
axs[0, 1].set_ylabel('mIoU')
axs[0, 1].legend()

# Plot Training Loss
axs[1, 0].plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='red')
axs[1, 0].set_title('Training Loss')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

# Plot Learning Rate
axs[1, 1].plot(df['Epoch'], df['LR'], label='Learning Rate', color='orange')
axs[1, 1].set_title('Learning Rate Schedule')
axs[1, 1].set_ylabel('LR')
axs[1, 1].legend()

for ax in axs.flat:
    ax.set_xlabel('Epoch')
    ax.set_xlim(1, df['Epoch'].max())

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_filename, dpi=300)
plt.show()
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)
