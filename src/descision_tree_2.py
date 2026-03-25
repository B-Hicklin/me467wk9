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
print(f"Training set: {len(X_train)} examples "
      f"({y_train.sum()} high-risk, {len(y_train) - y_train.sum()} low-risk)")
print(f"Test set:     {len(X_test)} examples "
      f"({y_test.sum()} high-risk, {len(y_test) - y_test.sum()} low-risk)")

tree_full = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_full.fit(X_train, y_train)
class_names = ["low-risk", "high-risk"]
print(export_text(tree_full, feature_names=feature_names, class_names=class_names))
print(f"Training accuracy: {tree_full.score(X_train, y_train):.3f}")

depths = [1, 2, 3, 4, 5, 6, None]
depth_labels = [str(d) if d is not None else "None" for d in depths]
train_accs = []
cv_means = []
cv_stds = []
for depth in depths:
    tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=depth, random_state=42
    )
    tree.fit(X_train, y_train)
    train_accs.append(tree.score(X_train, y_train))
    cv = cross_val_score(tree, X_train, y_train, cv=5, scoring="accuracy")
    cv_means.append(cv.mean())
    cv_stds.append(cv.std())
    print(f"max_depth={str(depth):>4s}  "
          f"train acc: {tree.score(X_train, y_train):.3f}  "
          f"CV acc: {cv.mean():.3f} ± {cv.std():.3f}")
    
# Plot both curves
depth_labels = [str(d) if d is not None else 'None' for d in depths]
x = np.arange(len(depths))

plt.figure(figsize=(10, 6))
plt.plot(x, train_accs, marker='o', label='Training Accuracy', linewidth=2)
plt.errorbar(x, cv_means, yerr=cv_stds, marker='s', label='Cross-Val Score (mean ± std)', linewidth=2, capsize=5)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Training Accuracy vs Cross-Validation Score by Max Depth')
plt.xticks(x, depth_labels)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_idx = int(np.argmax(cv_means))
best_depth = depths[best_idx]
print(f"Best depth by CV: {best_depth} "
      f"(CV accuracy: {cv_means[best_idx]:.3f})")
# Retrain on the full training set with the selected depth
tree_best = DecisionTreeClassifier(
    criterion="entropy", max_depth=best_depth, random_state=42
)
tree_best.fit(X_train, y_train)
print(f"\nSelected tree (max_depth={best_depth}):")
print(export_text(tree_best, feature_names=feature_names, class_names=class_names))
print(f"Training accuracy: {tree_best.score(X_train, y_train):.3f}")
# Final evaluation on the held-out test set (done once)
test_acc = tree_best.score(X_test, y_test)
print(f"Test accuracy:     {test_acc:.3f}  "
      f"({int(test_acc * len(y_test))}/{len(y_test)} correct)")
# Show feature importances
print("\nFeature importances:")
for name, imp in sorted(
    zip(feature_names, tree_best.feature_importances_),
    key=lambda x: -x[1],
):
    print(f"  {name}: {imp:.3f}")