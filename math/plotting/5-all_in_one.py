import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axs = plt.subplots(3, 2, figsize=(10, 10))
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2])
# Plot y0 in the first subplot
axs[0, 0].plot(y0, color='red')
axs[0, 0].tick_params(labelsize='x-small')

# Plot x1/y1 in the second subplot
axs[0, 1].scatter(x1, y1, color='magenta')
axs[0, 1].set_title("Men's Height vs Weight")
axs[0, 1].set_xlabel('Height (in)')
axs[0, 1].set_ylabel('Weight (lbs)')
axs[0, 1].tick_params(labelsize='x-small')

# Plot x2/y2 in the third subplot
axs[1, 0].plot(x2, y2, color='b',)
axs[1, 0].set_title('Exponential Decay')
axs[1, 0].set_xlabel('Time (years)')
axs[1, 0].set_ylabel('Fraction Remaining')
axs[1, 0].set_xlim(0, 28650)
axs[1, 0].set_yscale('log')
axs[1, 0].tick_params(labelsize='x-small')

# Plot x3/y31 in the fourth subplot
axs[1, 1].plot(x3, y31, linestyle='--', color='red', label='C-14')
axs[1, 1].plot(x3, y32, linestyle='-', color='green', label='Ra-226')
axs[1, 1].set_xlabel('Time (years)')
axs[1, 1].set_ylabel("Fraction Remaining")
axs[1, 1].set_title('Exponential Decay of Radioactive Elements')
axs[1, 1].set_xlim(0, 20000)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].legend(loc="upper right")
axs[1, 1].tick_params(labelsize='x-small')

# Plot student_grades as a histogram in the sixth subplot
axs[2, 0].hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
axs[2, 0].set_title('Student Grades')
axs[2, 0].set_xlabel('Grades')
axs[2, 0].set_ylabel('Number of Students')
axs[2, 0].tick_params(labelsize='x-small')

# Set the overall title for the entire figure
fig.suptitle('All in One', fontsize='x-small')

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
