#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# Create the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=z, cmap='viridis', marker='o', s=40)

# Set labels and title
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.title('Mountain Elevation')

# Create a colorbar
colorbar = plt.colorbar(scatter)
colorbar.set_label('Elevation (m)')

# Show the plot
plt.show()
