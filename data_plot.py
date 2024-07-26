import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression

num_data = 75

# Plot in the requested style
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Generate synthetic data for two classes
# Source Data Plot
np.random.seed(0)
s_class_1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], num_data)
s_class_2 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], num_data)
np.random.seed(42)
t_class_1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], num_data) + [6, 1]
t_class_2 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], num_data) + [8, -2]

axes[0, 0].scatter(s_class_1[:, 0], s_class_1[:, 1], color='red', marker='.', label='Class 1 (Source)')
axes[0, 0].scatter(s_class_2[:, 0], s_class_2[:, 1], color='blue', marker='.', label='Class 2 (Source)')
axes[0, 0].scatter(t_class_1[:, 0], t_class_1[:, 1], color='red', marker='*', label='Class 1 (Target)')
axes[0, 0].scatter(t_class_2[:, 0], t_class_2[:, 1], color='blue', marker='*', label='Class 2 (Target)')
# Plot Decision Boundary
x = np.vstack((s_class_1, s_class_2))
y = np.hstack((np.ones(num_data), np.zeros(num_data)))
clf = LogisticRegression()
clf.fit(x, y)
all_data = np.vstack((s_class_1, s_class_2, t_class_1, t_class_2))
x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0, 0].contourf(xx, yy, z, alpha=0.3, cmap=plt.cm.Paired)
axes[0, 0].set_title('Original Data')
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Class 1'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Class 2'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='black', markersize=10, label='Source'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='Target'),
]
axes[0, 0].legend(handles=legend_elements, loc='lower right')

# Covariate Shift Plot
# Simulate covariate shift by adding some noise
s_class_1_covariate = s_class_1 + np.random.normal(0, 1, s_class_1.shape)
s_class_2_covariate = s_class_2 + np.random.normal(0, 1, s_class_2.shape)
axes[0, 1].scatter(s_class_1[:, 0], s_class_1[:, 1], color='red', marker='.', label='Class 1 (Source)')
axes[0, 1].scatter(s_class_2[:, 0], s_class_2[:, 1], color='blue', marker='.', label='Class 2 (Source)')
axes[0, 1].scatter(t_class_1[:, 0], t_class_1[:, 1], color='red', marker='*', label='Class 1 (Target)')
axes[0, 1].scatter(t_class_2[:, 0], t_class_2[:, 1], color='blue', marker='*', label='Class 2 (Target)')
axes[0, 1].scatter(s_class_1_covariate[:, 0], s_class_1_covariate[:, 1], marker='x', color='red', label='Class 1 (Covariate Shift)')
axes[0, 1].scatter(s_class_2_covariate[:, 0], s_class_2_covariate[:, 1], marker='x', color='blue', label='Class 2 (Covariate Shift)')
axes[0, 1].set_title('Covariate Shift Augmentation')
axes[0, 1].legend(handles=legend_elements, loc='lower right')

# Conditional Distribution Shift Plot
# Simulate conditional distribution shift by shifting means
# class_1_cond_shift = class_1 + [1.5, 1.5]
# class_2_cond_shift = class_2 + [-1.5, -1.5]
axes[1, 0].scatter(s_class_1[:, 0], s_class_1[:, 1], color='red', marker='.', label='Class 1 (Source)')
axes[1, 0].scatter(s_class_2[:, 0], s_class_2[:, 1], color='blue', marker='.', label='Class 2 (Source)')
axes[1, 0].scatter(t_class_1[:, 0], t_class_1[:, 1], color='red', marker='*', label='Class 1 (Target)')
axes[1, 0].scatter(t_class_2[:, 0], t_class_2[:, 1], color='blue', marker='*', label='Class 2 (Target)')
# axes[2].scatter(class_1_cond_shift[:, 0], class_1_cond_shift[:, 1], color='pink', marker='x', label='Class 1 (Conditional Shift)')
# axes[2].scatter(class_2_cond_shift[:, 0], class_2_cond_shift[:, 1], color='purple', marker='x', label='Class 2 (Conditional Shift)')
axes[1, 0].set_title('Conditional Distribution Shift Augmentation')
axes[1, 0].legend(handles=legend_elements, loc='lower right')

# Covariate and Conditional Distribution Shift
axes[1, 1].scatter(s_class_1[:, 0], s_class_1[:, 1], color='red', marker='.', label='Class 1 (Source)')
axes[1, 1].scatter(s_class_2[:, 0], s_class_2[:, 1], color='blue', marker='.', label='Class 2 (Source)')
axes[1, 1].scatter(t_class_1[:, 0], t_class_1[:, 1], color='red', marker='*', label='Class 1 (Target)')
axes[1, 1].scatter(t_class_2[:, 0], t_class_2[:, 1], color='blue', marker='*', label='Class 2 (Target)')
axes[1, 1].set_title('Dual Augmentation')
axes[1, 1].legend(handles=legend_elements, loc='lower right')

plt.show()
plt.savefig("plot.png")
