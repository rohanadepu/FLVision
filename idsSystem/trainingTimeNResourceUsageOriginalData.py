import matplotlib.pyplot as plt
import numpy as np

# Training data: 9 points divided into 3 training methods (3 points each)
# Flight Training (red circles)
category_a_x = [205.82, 196.17, 195.85]
category_a_y = [284.15, 278.35, 279.20]

# FLWR Training (blue squares)
category_b_x = [179.81, 160.78, 165.30]
category_b_y = [172.2, 176.3, 178.7]

# Centralized Training (green triangles)
category_c_x = [119.55, 146.18, 146.12]
category_c_y = [174.3, 176.5, 171.4]

# Calculate data ranges for proper axis limits
all_x = category_a_x + category_b_x + category_c_x
all_y = category_a_y + category_b_y + category_c_y
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)

# Add 5% padding around the data
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

# Create the plot
plt.figure(figsize=(12, 8))

# Plot each training method with enhanced visibility for overlapping points
# Use larger markers with contrasting borders and different transparency levels
plt.scatter(category_a_x, category_a_y,
            color='red', marker='o', s=150, label='Flight Training',
            alpha=0.8, edgecolors='darkred', linewidth=2)
plt.scatter(category_b_x, category_b_y,
            color='blue', marker='s', s=150, label='FLWR Training',
            alpha=0.8, edgecolors='darkblue', linewidth=2)
plt.scatter(category_c_x, category_c_y,
            color='green', marker='^', s=150, label='Centralized Training',
            alpha=0.8, edgecolors='darkgreen', linewidth=2)

# Add jittering for overlapping points (small random offsets)
np.random.seed(42)  # For reproducible results
jitter_strength = 0.5

# Add slightly jittered versions with lighter colors for better visibility
for i, (x, y) in enumerate(zip(category_a_x, category_a_y)):
    jitter_x = x + np.random.uniform(-jitter_strength, jitter_strength)
    jitter_y = y + np.random.uniform(-jitter_strength, jitter_strength)

for i, (x, y) in enumerate(zip(category_b_x, category_b_y)):
    jitter_x = x + np.random.uniform(-jitter_strength, jitter_strength)
    jitter_y = y + np.random.uniform(-jitter_strength, jitter_strength)

for i, (x, y) in enumerate(zip(category_c_x, category_c_y)):
    jitter_x = x + np.random.uniform(-jitter_strength, jitter_strength)
    jitter_y = y + np.random.uniform(-jitter_strength, jitter_strength)

# Add connecting lines within each category to show relationships
plt.plot(category_a_x, category_a_y, 'r--', alpha=0.3, linewidth=1, zorder=1)
plt.plot(category_b_x, category_b_y, 'b--', alpha=0.3, linewidth=1, zorder=1)
plt.plot(category_c_x, category_c_y, 'g--', alpha=0.3, linewidth=1, zorder=1)

# Customize the plot
plt.xlabel('Time (Seconds)', fontsize=14)
plt.ylabel('CPU Usage (%)', fontsize=14)
plt.title('', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)

# Set axis limits based on actual data with padding
plt.xlim(x_min - x_padding, x_max + x_padding)
plt.ylim(y_min - y_padding, y_max + y_padding)

# Add annotations for each point with better positioning and backgrounds
annotation_offset = [(10, 10), (-15, 10), (10, -15)]  # Different offsets for each point

for i, (x, y) in enumerate(zip(category_a_x, category_a_y)):
    plt.annotate(f'F{i + 1}', (x, y), xytext=annotation_offset[i], textcoords='offset points',
                 fontsize=10, fontweight='bold', color='darkred',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkred', alpha=0.8))

for i, (x, y) in enumerate(zip(category_b_x, category_b_y)):
    plt.annotate(f'FL{i + 1}', (x, y), xytext=annotation_offset[i], textcoords='offset points',
                 fontsize=10, fontweight='bold', color='darkblue',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkblue', alpha=0.8))

for i, (x, y) in enumerate(zip(category_c_x, category_c_y)):
    plt.annotate(f'C{i + 1}', (x, y), xytext=annotation_offset[i], textcoords='offset points',
                 fontsize=10, fontweight='bold', color='darkgreen',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkgreen', alpha=0.8))

# Add subtle background highlighting for each category region
from matplotlib.patches import Ellipse

# Create elliptical regions to highlight each category cluster
ellipse_a = Ellipse((np.mean(category_a_x), np.mean(category_a_y)),
                    width=max(category_a_x) - min(category_a_x) + 10,
                    height=max(category_a_y) - min(category_a_y) + 20,
                    facecolor='red', alpha=0.1, zorder=0)
ellipse_b = Ellipse((np.mean(category_b_x), np.mean(category_b_y)),
                    width=max(category_b_x) - min(category_b_x) + 10,
                    height=max(category_b_y) - min(category_b_y) + 10,
                    facecolor='blue', alpha=0.1, zorder=0)
ellipse_c = Ellipse((np.mean(category_c_x), np.mean(category_c_y)),
                    width=max(category_c_x) - min(category_c_x) + 15,
                    height=max(category_c_y) - min(category_c_y) + 10,
                    facecolor='green', alpha=0.1, zorder=0)

plt.gca().add_patch(ellipse_a)
plt.gca().add_patch(ellipse_b)
plt.gca().add_patch(ellipse_c)

# Add some styling improvements
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)

# Display the plot
plt.tight_layout()
plt.show()

# Print summary statistics
print("Training Methods Performance Summary:")
print("=" * 50)

print(f"\nFlight Training (Red Circles):")
print(f"  Time Range: {min(category_a_x):.2f} - {max(category_a_x):.2f} seconds")
print(f"  CPU Usage Range: {min(category_a_y):.1f} - {max(category_a_y):.1f}%")
print(f"  Average Time: {np.mean(category_a_x):.2f} seconds")
print(f"  Average CPU Usage: {np.mean(category_a_y):.1f}%")

print(f"\nFLWR Training (Blue Squares):")
print(f"  Time Range: {min(category_b_x):.2f} - {max(category_b_x):.2f} seconds")
print(f"  CPU Usage Range: {min(category_b_y):.1f} - {max(category_b_y):.1f}%")
print(f"  Average Time: {np.mean(category_b_x):.2f} seconds")
print(f"  Average CPU Usage: {np.mean(category_b_y):.1f}%")

print(f"\nCentralized Training (Green Triangles):")
print(f"  Time Range: {min(category_c_x):.2f} - {max(category_c_x):.2f} seconds")
print(f"  CPU Usage Range: {min(category_c_y):.1f} - {max(category_c_y):.1f}%")
print(f"  Average Time: {np.mean(category_c_x):.2f} seconds")
print(f"  Average CPU Usage: {np.mean(category_c_y):.1f}%")