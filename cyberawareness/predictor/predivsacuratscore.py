# Count occurrences of each class
actual_counts = Counter(y_test)
predicted_counts = Counter(y_pred)

# Get unique class labels
labels = list(set(y_test) | set(y_pred))

# Get counts in same order
actual_values = [actual_counts[label] for label in labels]
predicted_values = [predicted_counts[label] for label in labels]

# Set bar width
bar_width = 0.35
x = np.arange(len(labels))

# Create bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bar_width/2, actual_values, bar_width, label="Actual", color='blue')
ax.bar(x + bar_width/2, predicted_values, bar_width, label="Predicted", color='orange')

# Labels and title
ax.set_xlabel("Cybersecurity Awareness Levels")
ax.set_ylabel("Number of Students")
ax.set_title("Comparison of Actual vs. Predicted Results")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Show the chart
plt.show()