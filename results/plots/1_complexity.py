import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use seaborn's set_theme to set the theme for the plot
sns.set_theme(style="whitegrid", palette="pastel")
# Update rcParams for LaTeX and font settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Amiri"
})

# Data
methods = ['GNN', 'LM', 'GLEM', 'TAPE (Ours)']
times = [4, 104, 551, 192]
metrics = [70.83, 73.61, 76.57, 77.50]
categories = ['Heuristic', 'Embeddings', 'GCNs', 'GCN2LP', 'LLM', 'Fine-Tuning']
colors = np.random.rand(len(methods), 3)  # Generate random colors
params = [10, 106, 1008, 10]

# Create the plot
fig, ax = plt.subplots(facecolor='white')
# ax.set_facecolor('white')


# Plot each method
for method, time, metric, category, color, param in zip(methods, times, metrics, categories[:len(methods)], colors, params):
    ax.text(time, metric, f'{method}\n{time}min, {metric}%', fontsize=12, ha='center', va='center')
    ax.scatter(time, metric, s=param, color=color, label=category, alpha=0.5, edgecolors='w', linewidth=2)
    
# Annotation for computation time reduction
# ax.annotate('2.88× lower computation time',
#             xy=(551, 76.57), xycoords='data',
#             xytext=(192, 77.5), textcoords='data',
#             arrowprops=dict(arrowstyle="->", lw=2, color='red'),
#             fontsize=15, color='red')
ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
ax.yaxis.set_minor_formatter(plt.ScalarFormatter())

# Customize the plot
ax.set_xlabel('Total Time (min)', fontsize=14)
ax.set_ylabel('Accuracy (\%)', fontsize=14)
ax.grid(True, which="both", ls="--")

# Set log scale for both axes

# Adjust y-axis limits to prevent cutoff

# Adjust axis limits
ax.set_xlim([1, 600])  # Extend x-axis limits
ax.set_ylim([70, 80])  # Set y-axis limits

#ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
ax.yaxis.set_minor_formatter(plt.ScalarFormatter())

# Ensure the ticks and labels are visible
ax.tick_params(axis='both', which='both', direction='in', length=6)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=8)

# Title and legend
ax.set_title('Comparison of Different Methods', fontsize=16)
ax.legend(title='Methods', fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))

# Save and show the plot
plt.savefig('comparison.png')
