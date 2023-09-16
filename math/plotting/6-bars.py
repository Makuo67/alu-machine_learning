#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Define the fruit labels, colors, and legend labels
fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
legend_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

# Create a stacked bar graph
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(fruit_labels)):
    ax.bar(np.arange(3), fruit[i], width=0.5, label=legend_labels[i],
           color=colors[i], bottom=np.sum(fruit[:i], axis=0))

# Set labels and title
ax.set_xlabel('Person')
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')

# Set the y-axis range and ticks
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))

# Set x-axis labels and ticks for each person
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'])

# Add a legend
ax.legend(loc='upper right')

# Show the plot
plt.show()
