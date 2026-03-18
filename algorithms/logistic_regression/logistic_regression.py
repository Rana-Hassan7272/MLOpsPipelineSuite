"""
╔══════════════════════════════════════════════════════════════════════════╗
║         LOGISTIC REGRESSION FROM SCRATCH — Built by Hassan              ║
║         Class: Hassan | Full gradient descent implementation             ║
╚══════════════════════════════════════════════════════════════════════════╝

Implements binary logistic regression using:
  - Sigmoid activation
  - Binary Cross-Entropy loss
  - Batch Gradient Descent
  - Full evaluation pipeline
  - Comparison against sklearn's LogisticRegression
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
import time
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Hassan — Logistic Regression from Scratch
# ─────────────────────────────────────────────────────────────────────────────

class Hassan:
    """
    Logistic Regression built from scratch by Hassan.

    Uses binary cross-entropy loss minimized via full-batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent (default: 0.1)
    epochs : int
        Number of training iterations (default: 1000)
    verbose : bool
        Print loss every 100 epochs if True (default: True)
    tolerance : float
        Early stopping threshold on loss change (default: 1e-6)
    """

    def __init__(self, learning_rate: float = 0.1,
                 epochs: int = 1000,
                 verbose: bool = True,
                 tolerance: float = 1e-6,
                 regularization: float = 0.01,
                 random_state: int = None,
                 adaptive_lr: bool = True,
                 lr_decay: float = 0.95,
                 patience: int = 100):

        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.epochs        = epochs
        self.verbose       = verbose
        self.tolerance     = tolerance
        self.regularization = regularization
        self.random_state  = random_state
        self.adaptive_lr  = adaptive_lr
        self.lr_decay      = lr_decay
        self.patience      = patience

        # Parameters — initialized in fit()
        self.weights: np.ndarray = None
        self.bias: float         = 0.0

        # Training history
        self.loss_history: list  = []
        self.n_features_in_: int = None
        self.training_time: float = 0.0
        self.best_loss: float = float('inf')
        self.patience_counter: int = 0

    # ── Core Math ────────────────────────────────────────────────────────────

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid / logistic function (highly optimized).
            σ(z) = 1 / (1 + e^{-z})

        Numerically stable: clips z to [-20, 20] to avoid overflow.
        Output range: (0, 1)
        Optimized: Uses faster computation methods.
        """
        # Use faster clipping with smaller range
        z_clipped = np.clip(z, -20, 20)
        # Optimized: Direct exp computation
        exp_neg_z = np.exp(-z_clipped)
        # Vectorized division
        return 1.0 / (1.0 + exp_neg_z)

    def compute_loss(self, y_true: np.ndarray,
                     y_pred: np.ndarray, 
                     class_weights: np.ndarray = None) -> float:
        """
        Binary Cross-Entropy (Log Loss) with optional class weighting.

            L = -(1/n) Σ [ w_y·y·log(p) + w_(1-y)·(1-y)·log(1-p) ]

        Parameters
        ----------
        y_true : (n,) array of true binary labels
        y_pred : (n,) array of predicted probabilities
        class_weights : (2,) array of weights for classes [0, 1] (optional)

        Returns
        -------
        float : mean binary cross-entropy
        """
        n = len(y_true)
        # Clip predictions to avoid log(0)
        eps = 1e-15
        p = np.clip(y_pred, eps, 1.0 - eps)
        
        if class_weights is not None:
            # Weighted loss for imbalanced datasets
            w0, w1 = class_weights[0], class_weights[1]
            loss = -(1.0 / n) * np.sum(
                w1 * y_true * np.log(p) + w0 * (1.0 - y_true) * np.log(1.0 - p)
            )
        else:
            loss = -(1.0 / n) * np.sum(
                y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)
            )
        return float(loss)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Hassan":
        """
        Train the model using full-batch gradient descent.

        Algorithm per epoch:
          1. Forward pass  →  z = X·W + b ,  p = σ(z)
          2. Loss          →  BCE(y, p)
          3. Gradients     →  dW = (1/n) Xᵀ(p-y) ,  db = mean(p-y)
          4. Update        →  W -= lr·dW ,  b -= lr·db
          5. Early stop if |Δloss| < tolerance

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        y : (n_samples,) binary target vector  {0, 1}

        Returns
        -------
        self
        """
        # Handle sparse matrices (for TF-IDF features)
        is_sparse = hasattr(X, 'toarray')
        if is_sparse:
            n_samples, n_features = X.shape
            X_dense = None  # Keep sparse for efficiency
        else:
            X = np.array(X, dtype=np.float64)
            n_samples, n_features = X.shape
        
        y = np.array(y, dtype=np.float64).ravel()

        self.n_features_in_ = n_features

        # ── Compute class weights for imbalanced datasets ─────────────────
        # Optimized: Use bincount for faster counting
        class_counts = np.bincount(y.astype(int))
        n_class_0 = class_counts[0] if len(class_counts) > 0 else 0
        n_class_1 = class_counts[1] if len(class_counts) > 1 else 0
        total = n_class_0 + n_class_1
        if n_class_0 > 0 and n_class_1 > 0:
            # Balanced class weights (less extreme than inverse frequency)
            weight_0 = total / (2.0 * n_class_0)
            weight_1 = total / (2.0 * n_class_1)
            # Cap weights to prevent extreme imbalance that causes model collapse
            max_weight = max(weight_0, weight_1)
            if max_weight > 5.0:  # Cap at 5x to prevent extreme weights
                scale = 5.0 / max_weight
                weight_0 *= scale
                weight_1 *= scale
            class_weights = np.array([weight_0, weight_1], dtype=np.float64)
        else:
            class_weights = np.array([1.0, 1.0], dtype=np.float64)

        # ── Parameter initialisation ──────────────────────────────────────
        # Xavier/Glorot initialization for better gradient flow
        if self.random_state is not None:
            np.random.seed(self.random_state)
        limit = np.sqrt(2.0 / (n_features + 1))
        self.weights = np.random.normal(0, limit, n_features).astype(np.float64)
        # Initialize bias to log(positive_class_ratio) for better starting point
        if n_class_1 > 0 and n_class_0 > 0:
            pos_ratio = n_class_1 / n_class_0
            self.bias = np.log(pos_ratio + 1e-10)  # Add epsilon for numerical stability
        else:
            self.bias = 0.0
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.learning_rate = self.initial_lr

        # ── Pre-compute constants for speed ──────────────────────────────
        inv_n_samples = 1.0 / n_samples
        max_grad_norm = 1.0
        
        # Pre-compute class weight arrays for vectorized operations
        class_weight_array = np.where(y == 1, class_weights[1], class_weights[0])
        
        # ── Training loop ─────────────────────────────────────────────────
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):

            # ① Forward pass (optimized)
            if is_sparse:
                z = X.dot(self.weights) + self.bias  # sparse matrix dot product
            else:
                z = np.dot(X, self.weights) + self.bias  # Use dot instead of @ for speed
            p = self.sigmoid(z)  # predicted probabilities

            # ② Loss (with class weights) - compute every 10 epochs for speed
            if epoch % 10 == 0 or epoch == 1:
                loss = self.compute_loss(y, p, class_weights)
                if self.regularization > 0:
                    reg_loss = 0.5 * self.regularization * np.sum(self.weights ** 2)
                    loss += reg_loss
                self.loss_history.append(loss)
            else:
                # Use previous loss for early stopping checks
                loss = self.loss_history[-1] if self.loss_history else 1.0

            # ③ Early stopping with patience (check every 10 epochs)
            if epoch % 10 == 0:
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 10  # Increment by 10 since we check every 10 epochs
                    if self.patience_counter >= self.patience:
                        if self.verbose:
                            print(f"\n  ✓ Early stop at epoch {epoch} "
                                  f"(no improvement for {self.patience} epochs)")
                        break

            # ④ Adaptive learning rate decay (check less frequently)
            if self.adaptive_lr and epoch > 1 and epoch % 200 == 0:
                # Only decay if loss hasn't improved significantly over last 5 loss checks
                if len(self.loss_history) > 5:
                    recent_improvement = self.loss_history[-5] - self.loss_history[-1]
                    if recent_improvement < 0.001:  # Very small improvement threshold
                        self.learning_rate *= self.lr_decay
                        if self.verbose:
                            print(f"  Learning rate decayed to {self.learning_rate:.6f}")

            # ⑤ Gradients (optimized vectorized computation)
            error = p - y  # (n,)
            # Vectorized weighted error computation
            weighted_error = class_weight_array * error
            
            if is_sparse:
                # Optimized sparse matrix multiplication
                dw = inv_n_samples * X.T.dot(weighted_error)
            else:
                # Use dot product for better performance
                dw = inv_n_samples * np.dot(X.T, weighted_error)
            
            db = inv_n_samples * np.sum(weighted_error)  # scalar
            
            # Add L2 regularization (vectorized)
            if self.regularization > 0:
                dw += self.regularization * self.weights

            # ⑥ Parameter update with gradient clipping (optimized)
            # Only compute norm if needed (skip if gradient is small)
            grad_norm_sq = np.dot(dw, dw)
            if grad_norm_sq > max_grad_norm * max_grad_norm:
                dw *= max_grad_norm / np.sqrt(grad_norm_sq)
            
            # Vectorized parameter update
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db
            
            # Fast convergence check using gradient norm (every 20 epochs)
            if epoch % 20 == 0 and grad_norm_sq < 1e-7:
                if self.verbose:
                    print(f"\n  ✓ Early stop at epoch {epoch} (gradient converged)")
                break

            # ⑦ Logging (less frequent)
            if self.verbose and epoch % 200 == 0:
                print(f"  Epoch {epoch:>5}/{self.epochs}  │  Loss: {loss:.6f}  │  LR: {self.learning_rate:.6f}")

            # ⑧ Tolerance-based early stopping (check every 10 epochs)
            if epoch % 10 == 0 and epoch > 10 and len(self.loss_history) > 1:
                if abs(self.loss_history[-2] - loss) < self.tolerance:
                    if self.verbose:
                        print(f"\n  ✓ Early stop at epoch {epoch} "
                              f"(Δloss < {self.tolerance})")
                    break
        
        self.training_time = time.time() - start_time
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities P(y=1 | X) (optimized).

        Parameters
        ----------
        X : (n_samples, n_features) - can be sparse or dense

        Returns
        -------
        (n_samples,) array of probabilities in (0, 1)
        """
        # Optimized: Use sparse operations directly without conversion
        if hasattr(X, 'dot'):
            z = X.dot(self.weights) + self.bias  # Keep sparse for speed
        else:
            X_dense = np.asarray(X, dtype=np.float64)  # Use asarray (faster than array)
            z = np.dot(X_dense, self.weights) + self.bias  # Use dot instead of @
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert probabilities to hard class predictions.

            ŷ = 1  if P(y=1|X) ≥ threshold  else  0

        Parameters
        ----------
        X         : (n_samples, n_features)
        threshold : decision boundary (default 0.5)

        Returns
        -------
        (n_samples,) array of integer predictions {0, 1}
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    # ── Convenience ──────────────────────────────────────────────────────────

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y)."""
        return accuracy_score(y, self.predict(X))

    def __repr__(self) -> str:
        return (f"Hassan(learning_rate={self.learning_rate}, "
                f"epochs={self.epochs}, tolerance={self.tolerance})")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation & Comparison Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name: str, y_true: np.ndarray,
                   y_pred: np.ndarray) -> dict:
    """Compute and print a full classification report."""
    metrics = {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1 Score" : f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1 Score  : {metrics['F1 Score']:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    return metrics


def plot_results(hassan_model: Hassan,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 hassan_metrics: dict,
                 sklearn_metrics: dict) -> None:
    """
    4-panel diagnostic figure:
      Panel 1 — Training loss curve
      Panel 2 — Predicted probability distribution
      Panel 3 — Confusion matrices (side-by-side)
      Panel 4 — Metrics bar chart comparison
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT  = "#00e5ff"
    ORANGE  = "#ff6b35"
    GREEN   = "#39ff14"
    BG      = "#0f0f1a"
    PANEL   = "#1a1a2e"
    WHITE   = "#e8e8f0"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=WHITE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    # ── Panel 1: Loss curve ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "📉  Training Loss — Hassan Model")
    epochs_ran = range(1, len(hassan_model.loss_history) + 1)
    ax1.plot(epochs_ran, hassan_model.loss_history,
             color=ACCENT, linewidth=2, label="BCE Loss")
    ax1.fill_between(epochs_ran, hassan_model.loss_history,
                     alpha=0.15, color=ACCENT)
    ax1.set_xlabel("Epoch", color=WHITE, fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy", color=WHITE, fontsize=10)
    ax1.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
    ax1.grid(True, color="#222244", linewidth=0.5)

    # ── Panel 2: Probability distribution ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "🎯  Predicted Probability Distribution")
    proba = hassan_model.predict_proba(X_test)
    ax2.hist(proba[y_test == 0], bins=30, alpha=0.7,
             color=ORANGE, label="True Class 0", edgecolor="none")
    ax2.hist(proba[y_test == 1], bins=30, alpha=0.7,
             color=GREEN, label="True Class 1", edgecolor="none")
    ax2.axvline(0.5, color=WHITE, linestyle="--", linewidth=1.2,
                label="Threshold = 0.5")
    ax2.set_xlabel("P(y=1 | X)", color=WHITE, fontsize=10)
    ax2.set_ylabel("Count", color=WHITE, fontsize=10)
    ax2.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
    ax2.grid(True, color="#222244", linewidth=0.5)

    # ── Panel 3: Confusion matrices ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "🔢  Confusion Matrices")
    hassan_cm  = confusion_matrix(y_test, hassan_model.predict(X_test))
    # sklearn predictions stored earlier
    display_data = np.zeros((4, 4))
    # Left half = Hassan, Right half = sklearn (visualised as text table)
    ax3.axis("off")
    labels = ["Hassan", "sklearn"]
    cms    = [hassan_cm, confusion_matrix(
                  y_test,
                  SklearnLR(random_state=42, max_iter=1000)
                  .fit(X_train_scaled, y_train)
                  .predict(X_test_scaled)
              )]
    for idx, (lbl, cm) in enumerate(zip(labels, cms)):
        x_off = idx * 0.5
        ax3.text(x_off + 0.15, 0.93, lbl,
                 transform=ax3.transAxes,
                 color=ACCENT if idx == 0 else ORANGE,
                 fontsize=11, fontweight="bold", ha="center")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = GREEN if i == j else "#ff4444"
                ax3.text(x_off + 0.07 + j*0.13, 0.65 - i*0.22,
                         str(val), transform=ax3.transAxes,
                         color=color, fontsize=16, fontweight="bold",
                         ha="center")
        ax3.text(x_off + 0.07, 0.85, "Pred 0", transform=ax3.transAxes,
                 color=WHITE, fontsize=8, ha="center")
        ax3.text(x_off + 0.20, 0.85, "Pred 1", transform=ax3.transAxes,
                 color=WHITE, fontsize=8, ha="center")
        ax3.text(x_off + 0.02, 0.65, "Act 0", transform=ax3.transAxes,
                 color=WHITE, fontsize=8, va="center")
        ax3.text(x_off + 0.02, 0.43, "Act 1", transform=ax3.transAxes,
                 color=WHITE, fontsize=8, va="center")

    # ── Panel 4: Metrics bar chart ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "📊  Hassan vs sklearn — Metric Comparison")
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    hassan_vals  = [hassan_metrics[m]  for m in metric_names]
    sklearn_vals = [sklearn_metrics[m] for m in metric_names]
    x = np.arange(len(metric_names))
    w = 0.35
    bars1 = ax4.bar(x - w/2, hassan_vals,  w, color=ACCENT,  label="Hassan",  alpha=0.85)
    bars2 = ax4.bar(x + w/2, sklearn_vals, w, color=ORANGE, label="sklearn", alpha=0.85)
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 color=WHITE, fontsize=8)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 color=WHITE, fontsize=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names, color=WHITE, fontsize=9)
    ax4.set_ylim(0, 1.12)
    ax4.set_ylabel("Score", color=WHITE, fontsize=10)
    ax4.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=9)
    ax4.grid(True, axis="y", color="#222244", linewidth=0.5)

    fig.suptitle(
        "Hassan — Logistic Regression from Scratch  vs  sklearn",
        color=WHITE, fontsize=16, fontweight="bold", y=0.98
    )

    plt.savefig("/mnt/user-data/outputs/hassan_results.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    print("\n  ✓ Plot saved → hassan_results.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main — Run Everything
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("═" * 60)
    print("  HASSAN — Logistic Regression from Scratch")
    print("═" * 60)

    # ── 1. Generate Dataset ───────────────────────────────────────────────
    print("\n[1] Generating dataset...")
    X, y = make_classification(
        n_samples    = 1000,
        n_features   = 10,
        n_informative= 6,
        n_redundant  = 2,
        n_clusters_per_class=1,
        flip_y       = 0.03,     # 3 % label noise — realistic
        random_state = 42
    )
    print(f"    Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"    Class balance: {np.bincount(y)}")

    # ── 2. Split & Scale ──────────────────────────────────────────────────
    print("\n[2] Splitting (80/20) and scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 3. Train Hassan model ─────────────────────────────────────────────
    print("\n[3] Training Hassan model...")
    hassan = Hassan(learning_rate=0.1, epochs=2000, verbose=True)
    hassan.fit(X_train_scaled, y_train)

    # ── 4. Evaluate Hassan ────────────────────────────────────────────────
    print("\n[4] Evaluating Hassan model...")
    y_pred_hassan  = hassan.predict(X_test_scaled)
    hassan_metrics = evaluate_model("Hassan (Scratch)", y_test, y_pred_hassan)

    # ── 5. Train & Evaluate sklearn ───────────────────────────────────────
    print("\n[5] Training sklearn LogisticRegression...")
    sk_model = SklearnLR(random_state=42, max_iter=1000)
    sk_model.fit(X_train_scaled, y_train)
    y_pred_sk      = sk_model.predict(X_test_scaled)
    sklearn_metrics = evaluate_model("sklearn LogisticRegression", y_test, y_pred_sk)

    # ── 6. Summary Table ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  COMPARISON SUMMARY")
    print("═" * 60)
    header = f"  {'Metric':<15} {'Hassan':>12} {'sklearn':>12} {'Δ':>10}"
    print(header)
    print("  " + "─" * 52)
    for m in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        h_val  = hassan_metrics[m]
        sk_val = sklearn_metrics[m]
        delta  = h_val - sk_val
        sign   = "+" if delta >= 0 else ""
        print(f"  {m:<15} {h_val:>12.4f} {sk_val:>12.4f} "
              f"  {sign}{delta:.4f}")

    # ── 7. Plot ───────────────────────────────────────────────────────────
    print("\n[6] Generating diagnostic plots...")
    plot_results(hassan, X_test_scaled, y_test, hassan_metrics, sklearn_metrics)

    # ── 8. Model Parameters ───────────────────────────────────────────────
    print("\n[7] Learned Parameters")
    print(f"  Weights : {np.round(hassan.weights, 4)}")
    print(f"  Bias    : {hassan.bias:.6f}")
    print(f"  Epochs ran  : {len(hassan.loss_history)}")
    print(f"  Final loss  : {hassan.loss_history[-1]:.6f}")
    print(f"  Initial loss: {hassan.loss_history[0]:.6f}")
    print(f"  Loss drop   : {hassan.loss_history[0] - hassan.loss_history[-1]:.6f}")

    print("\n" + "═" * 60)
    print("  Done. Hassan model fully trained and compared. ✓")
    print("═" * 60)
