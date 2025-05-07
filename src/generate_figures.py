import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create directories if not exist
import os
os.makedirs('results/figures', exist_ok=True)

# Load your results (replace these with your actual data loading methods)
# Example synthetic data for demonstration purposes
np.random.seed(42)
actual_outcomes = np.random.normal(5, 1, 100)
predicted_outcomes = actual_outcomes + np.random.normal(0, 0.5, 100)
ite = np.random.normal(1.0, 0.5, 1000)
epochs = np.arange(1, 51)
train_loss = np.exp(-epochs/10) + 0.1*np.random.rand(50)
val_loss = np.exp(-epochs/9) + 0.15*np.random.rand(50)

# Ablation scores (example data)
metrics = ['PEHE', 'ATE']
mtal_scores = [0.22, 1.30]
ablation_scores = [0.35, 1.95]

# Figure 1: Predicted vs. Actual Outcomes
plt.figure()
plt.scatter(actual_outcomes, predicted_outcomes, alpha=0.7)
plt.plot([actual_outcomes.min(), actual_outcomes.max()],
         [actual_outcomes.min(), actual_outcomes.max()], 'r--', linewidth=2)
plt.xlabel('Actual Outcomes')
plt.ylabel('Predicted Outcomes')
plt.title('Predicted vs. Actual Outcomes')
plt.grid(True)
plt.savefig('results/figures/predicted_vs_actual.png')

# Figure 2: Distribution of Treatment Effects (ITE)
plt.figure()
sns.histplot(ite, kde=True)
plt.title('Distribution of Individual Treatment Effects (ITE)')
plt.xlabel('Individual Treatment Effect')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('results/figures/treatment_effect_distribution.png')

# Figure 3: Training and Validation Loss Curves
plt.figure()
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/loss_curves.png')

# Figure 4: Ablation Study Comparison
plt.figure()
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mtal_scores, width, label='MTAL')
rects2 = ax.bar(x + width/2, ablation_scores, width, label='No Discriminator')

ax.set_ylabel('Metric Scores')
ax.set_title('Performance Comparison (Ablation Study)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
fig.tight_layout()
plt.grid(True, axis='y')
plt.savefig('results/figures/ablation_comparison.png')

plt.show()