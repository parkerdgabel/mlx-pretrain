import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import numpy as np

# Read and parse the leaderboard file
with open("leaderboard.txt", "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    match = re.match(r"(.+)\s+-\s+Accuracy:\s+([0-9.]+)%", line.strip())
    if match:
        model = match.group(1).strip()
        accuracy = float(match.group(2))
        data.append((model, accuracy))

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Model', 'Accuracy'])

# Calculate error margins using sqrt(p(1-p)/n) where n = 2048
n = 2048
df['Error'] = np.sqrt((df['Accuracy']/100) * (1 - df['Accuracy']/100) / n) * 100

# Sort by accuracy (highest first)
df = df.sort_values('Accuracy', ascending=False)

# Plot
colors = cm.viridis((df['Accuracy'] - df['Accuracy'].min()) / (df['Accuracy'].max() - df['Accuracy'].min()))  # Normalize for color mapping

plt.figure(figsize=(12, 8))  # Increased width for more padding
bars = plt.barh(df['Model'], df['Accuracy'], color=colors, xerr=df['Error'], 
        error_kw={'ecolor': '0.3', 'capsize': 3, 'elinewidth': 1})
plt.xlabel("Accuracy (%)")
plt.title("Model Performance on OEIS Next-Term Prediction")
plt.gca().invert_yaxis()  # Highest accuracy on top
plt.grid(axis='both', linestyle='--', alpha=0.5)

# Add text labels showing the accuracy values with error margins
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + df['Error'].iloc[i] + 0.5, bar.get_y() + bar.get_height()/2, 
             f"{df['Accuracy'].iloc[i]:.1f}%", 
             va='center')

# Set the x-axis limit to include space for the text labels and error bars
max_accuracy = df['Accuracy'].max()
max_error = df['Error'].max()
plt.xlim(0, max_accuracy * 1.25)  # Add 25% padding to the right to accommodate error bars and text

plt.tight_layout()
plt.savefig("graphs/leaderboard.png", dpi=300)  # Save the plot to a file