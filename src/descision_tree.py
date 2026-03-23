import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300
load = np.random.randint(100, 1001, n)
inspection = np.random.randint(1, 91, n)
sensors = np.random.randint(1, 6, n)
floor_age = np.random.randint(1, 31, n)
true_risk = ((load > 500) | (inspection > 45)).astype(float)
flip = np.random.random(n) < 0.20
high_risk = true_risk.copy()
high_risk[flip] = 1 - high_risk[flip]
high_risk = high_risk.astype(int)

X = np.column_stack([load, inspection, sensors, floor_age])
y = high_risk
feature_names = ["load_kg", "inspection_days", "sensors", "floor_age_years"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {len(X_train)} examples "
      f"({y_train.sum()}/{len(y_train) - y_train.sum()} high/low-risk)")
print(f"Testing:     {len(X_test)} examples "
      f"({y_test.sum()}/{len(y_test) - y_test.sum()} high/low-risk)")

# Train DecisionTreeClassifier with entropy criterion
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

# Print the tree
tree_rules = export_text(dt, feature_names=feature_names)
print("\nDecision Tree Rules:")
print(tree_rules)

# Compute training accuracy
train_accuracy = dt.score(X_train, y_train)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

# Loop over max_depth values
max_depths = [1, 2, 3, 4, 5, 6, None]
train_accuracies = []
cv_scores_mean = []
cv_scores_std = []

for depth in max_depths:
    # Create and train the model
    dt_depth = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
    dt_depth.fit(X_train, y_train)
    
    # Compute training accuracy
    train_acc = dt_depth.score(X_train, y_train)
    train_accuracies.append(train_acc)
    
    # Compute cross-validation scores
    cv_scores = cross_val_score(dt_depth, X_train, y_train, cv=5)
    cv_scores_mean.append(cv_scores.mean())
    cv_scores_std.append(cv_scores.std())

# Plot both curves
depth_labels = [str(d) if d is not None else 'None' for d in max_depths]
x = np.arange(len(max_depths))

plt.figure(figsize=(10, 6))
plt.plot(x, train_accuracies, marker='o', label='Training Accuracy', linewidth=2)
plt.errorbar(x, cv_scores_mean, yerr=cv_scores_std, marker='s', label='Cross-Val Score (mean ± std)', linewidth=2, capsize=5)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Training Accuracy vs Cross-Validation Score by Max Depth')
plt.xticks(x, depth_labels)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Select the best max_depth based on cross-validation scores
best_idx = np.argmax(cv_scores_mean)
best_depth = max_depths[best_idx]
best_cv_score = cv_scores_mean[best_idx]

print(f"\n--- Best Model Selection ---")
print(f"Best max_depth: {best_depth}")
print(f"Best CV Score: {best_cv_score:.4f} (± {cv_scores_std[best_idx]:.4f})")

# Retrain on the full training set with the best depth
best_dt = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, random_state=42)
best_dt.fit(X_train, y_train)

# Report the final tree structure
print(f"\n--- Final Tree Structure (max_depth={best_depth}) ---")
best_tree_rules = export_text(best_dt, feature_names=feature_names)
print(best_tree_rules)

# Report training and test accuracies
train_acc_final = best_dt.score(X_train, y_train)
test_acc_final = best_dt.score(X_test, y_test)
print(f"\nFinal Training Accuracy: {train_acc_final:.4f}")
print(f"Final Test Accuracy: {test_acc_final:.4f}")

