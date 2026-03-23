import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


#dataset generation
np.random.seed(0)
n = 50
distance = np.random.uniform(5, 40, n)        # meters
load = np.random.uniform(10, 100, n)           # kg
congestion = np.random.randint(0, 5, n)        # number of nearby robots
# True relationship (unknown to the model)
time = 1.8 * distance + 0.3 * load + 5.0 * congestion + 10 + np.random.normal(0, 5, n)
X = np.column_stack([distance, load, congestion])
y = time

#test/train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} examples")
print(f"Test set:     {len(X_test)} examples")


def gradient_descent_linear_regression(X, y, alpha=0.1, iterations=1000, batch_size=None, reg_lambda=0.0, verbose=False):
    """Fit linear regression by gradient descent from scratch with optional L2 regularization.

    Arguments:
    - X: np.ndarray of shape (n_samples, n_features)
    - y: np.ndarray of shape (n_samples,)
    - alpha: float, learning rate
    - iterations: int, number of full passes over training data
    - batch_size: int or None for full-batch. If provided, use mini-batch.
    - reg_lambda: float, L2 regularization strength (lambda). 0.0 for no regularization.
    - verbose: bool, if True print loss every 100 iterations

    Returns:
    - w: np.ndarray shape (n_features,), learned weights
    - b: float, learned bias
    - losses: list of float, MSE loss during training (without regularization term)
    - X_mean: np.ndarray shape (n_features,), means for normalization
    - X_std: np.ndarray shape (n_features,), std devs for normalization
    """
    n_samples, n_features = X.shape

    # Feature normalization (standard scale)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    w = np.zeros(n_features)
    b = 0.0
    losses = []

    if batch_size is None or batch_size > n_samples:
        batch_size = n_samples

    for iteration in range(1, iterations + 1):
        perm = np.random.permutation(n_samples)
        X_shuffled = X_norm[perm]
        y_shuffled = y[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            preds = X_batch.dot(w) + b
            errors = preds - y_batch
            loss = np.mean(errors ** 2)

            grad_w = (2 / len(y_batch)) * X_batch.T.dot(errors) + 2 * reg_lambda * w
            grad_b = (2 / len(y_batch)) * np.sum(errors)

            w -= alpha * grad_w
            b -= alpha * grad_b

        preds_all = X_norm.dot(w) + b
        loss_all = np.mean((preds_all - y) ** 2)
        losses.append(loss_all)

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration:4d}/{iterations} - loss: {loss_all:.6f}")

    return w, b, losses, X_mean, X_std


def gradient_descent_logistic_regression(X, y, alpha=0.1, iterations=1000, batch_size=None, reg_lambda=0.0, verbose=False):
    """Fit logistic regression by gradient descent with sigmoid activation and cross-entropy loss.

    Arguments:
    - X: np.ndarray of shape (n_samples, n_features)
    - y: np.ndarray of shape (n_samples,), binary labels (0 or 1)
    - alpha: float, learning rate
    - iterations: int, number of full passes over training data
    - batch_size: int or None for full-batch. If provided, use mini-batch.
    - reg_lambda: float, L2 regularization strength (lambda). 0.0 for no regularization.
    - verbose: bool, if True print loss every 100 iterations

    Returns:
    - w: np.ndarray shape (n_features,), learned weights
    - b: float, learned bias
    - losses: list of float, cross-entropy loss during training
    - accuracies: list of float, training accuracy per iteration
    - X_mean: np.ndarray shape (n_features,), means for normalization
    - X_std: np.ndarray shape (n_features,), std devs for normalization
    """
    n_samples, n_features = X.shape

    # Feature normalization (standard scale)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    w = np.zeros(n_features)
    b = 0.0
    losses = []
    accuracies = []

    if batch_size is None or batch_size > n_samples:
        batch_size = n_samples

    for iteration in range(1, iterations + 1):
        perm = np.random.permutation(n_samples)
        X_shuffled = X_norm[perm]
        y_shuffled = y[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            z = X_batch.dot(w) + b
            preds_proba = sigmoid(z)
            errors = preds_proba - y_batch

            grad_w = (1 / len(y_batch)) * X_batch.T.dot(errors) + 2 * reg_lambda * w
            grad_b = (1 / len(y_batch)) * np.sum(errors)

            w -= alpha * grad_w
            b -= alpha * grad_b

        # Compute cross-entropy loss on full training set
        z_all = X_norm.dot(w) + b
        preds_proba_all = sigmoid(z_all)
        cross_entropy = -np.mean(y * np.log(preds_proba_all + 1e-15) + (1 - y) * np.log(1 - preds_proba_all + 1e-15))
        losses.append(cross_entropy)

        # Compute accuracy on full training set
        predictions = (preds_proba_all >= 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration:4d}/{iterations} - loss: {cross_entropy:.6f} - accuracy: {accuracy:.4f}")

    return w, b, losses, accuracies, X_mean, X_std


if __name__ == "__main__":
    w, b, losses, X_mean, X_std = gradient_descent_linear_regression(
        X_train, y_train, alpha=0.1, iterations=1000, batch_size=None, verbose=True
    )

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(losses) + 1), losses, label='Training MSE')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Gradient Descent Linear Regression: Loss vs Iteration')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    iterations = 500

    plt.figure(figsize=(10, 6))
    for alpha in alpha_values:
        _, _, losses, _, _ = gradient_descent_linear_regression(
            X_train, y_train, alpha=alpha, iterations=iterations, batch_size=None, verbose=False
        )
        plt.plot(np.arange(1, iterations + 1), losses, label=f"alpha={alpha}")

    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Gradient Descent: Loss vs Iteration for Different Learning Rates')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    lambda_values = [0, 0.01, 0.1, 1, 10]
    alpha = 0.1
    iterations = 1000

    print("Running gradient descent with different lambda values (L2 regularization):")
    print(f"Alpha (learning rate): {alpha}")
    print(f"Iterations: {iterations}")
    print()

    for reg_lambda in lambda_values:
        w, b, losses, _, _ = gradient_descent_linear_regression(
            X_train, y_train, alpha=alpha, iterations=iterations, reg_lambda=reg_lambda, verbose=False
        )
        final_mse = losses[-1]
        print(f"Lambda: {reg_lambda}")
        print(f"  Final weights: {w}")
        print(f"  Final bias: {b:.6f}")
        print(f"  Final MSE: {final_mse:.6f}")
        print()
    
    # Logistic Regression: Convert continuous y to binary labels
    y_median = np.median(y)
    y_binary = (y >= y_median).astype(int)
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    print("=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)
    w_log, b_log, losses_log, accuracies_log, _, _ = gradient_descent_logistic_regression(
        X_train_log, y_train_log, alpha=0.1, iterations=1000, batch_size=None, verbose=True
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(losses_log) + 1), losses_log, label='Training Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Logistic Regression: Loss vs Iteration')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(accuracies_log) + 1), accuracies_log, label='Training Accuracy', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression: Training Accuracy vs Iteration')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
