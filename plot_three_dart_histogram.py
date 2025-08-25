import numpy as np
import matplotlib.pyplot as plt

# Load the saved totals
filename = "/Users/a_fin/Documents/Stat_of_darts/three_dart_totals.npy"
totals = np.load(filename)

mean_score = np.mean(totals)

plt.figure(figsize=(14, 8))
bins = range(0, int(totals.max()) + 5, 5)
plt.hist(totals, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
plt.xlabel('Total Score (3 darts)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title(f'Distribution of Three-Dart Total Scores\nMean: {mean_score:.2f}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/a_fin/Documents/Stat_of_darts/three_dart_histogram.png")
plt.show()