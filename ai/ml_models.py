"""
Classical ML Predictors for Perovskite Solar Cell Performance
==============================================================
Implements Random Forest, XGBoost (gradient boosting), and ANN
for predicting PCE, Voc, Jsc, FF from material/device parameters.

Addresses PhD Gap 5 (classical ML from synopsis Table 1) and
Gap 2 (training on experimental-scale data).

Models can be trained on:
  (a) Simulation-generated data (from the physics solver)
  (b) Real experimental data (from Perovskite Database Project CSV)

All models report: R², MAPE, RMSE, and feature importance.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION: Simulation-based dataset
# ═══════════════════════════════════════════════════════════════════════════════
def generate_simulation_dataset(n_samples=2000, seed=42):
    """
    Generate a dataset by running the physics solver across the parameter space.
    Returns features X and targets Y with proper column names.
    """
    np.random.seed(seed)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physics.materials import HTL_DB, PEROVSKITE_DB, ETL_DB
    from physics.device import fast_simulate

    htl_names = list(HTL_DB.keys())
    abs_names = list(PEROVSKITE_DB.keys())
    etl_names = list(ETL_DB.keys())

    feature_names = [
        "Eg_abs", "chi_abs", "eps_abs", "mu_e_abs", "mu_h_abs",
        "Eg_htl", "chi_htl", "Eg_etl", "chi_etl",
        "d_htl", "d_abs", "d_etl", "log_Nt", "temperature",
        "CBO", "VBO", "doping_abs_log",
    ]
    target_names = ["PCE", "Voc", "Jsc", "FF"]

    X_list, Y_list = [], []

    for _ in range(n_samples):
        h_name = np.random.choice(htl_names)
        a_name = np.random.choice(abs_names)
        e_name = np.random.choice(etl_names)
        h, a, e = HTL_DB[h_name], PEROVSKITE_DB[a_name], ETL_DB[e_name]

        d_htl = np.random.uniform(20, 400)
        d_abs = np.random.uniform(100, 1200)
        d_etl = np.random.uniform(20, 300)
        log_nt = np.random.uniform(10, 17)
        T = np.random.uniform(280, 380)

        try:
            r = fast_simulate(h, a, e, d_htl, d_abs, d_etl, 10**log_nt, T)
            if r["PCE"] <= 0 or not np.isfinite(r["PCE"]):
                continue

            CBO = e.chi - a.chi
            VBO = (h.chi + h.Eg) - (a.chi + a.Eg)

            features = [
                a.Eg, a.chi, a.eps, np.log10(max(a.mu_e, 1e-6)),
                np.log10(max(a.mu_h, 1e-6)),
                h.Eg, h.chi, e.Eg, e.chi,
                d_htl, d_abs, d_etl, log_nt, T,
                CBO, VBO, np.log10(max(a.doping, 1e6)),
            ]
            targets = [r["PCE"], r["Voc"], r["Jsc"], r["FF"]]

            X_list.append(features)
            Y_list.append(targets)
        except Exception:
            continue

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, feature_names, target_names


def load_experimental_csv(filepath):
    """
    Load real experimental data from Perovskite Database Project CSV.
    Extracts relevant features and targets.

    Expected columns (flexible matching):
      - Perovskite composition / Eg / bandgap
      - ETL, HTL material names
      - Layer thicknesses
      - PCE, Voc, Jsc, FF
    """
    import pandas as pd

    df = pd.read_csv(filepath, low_memory=False)

    # Try to find relevant columns by fuzzy matching
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "pce" in cl or "efficiency" in cl:
            col_map["PCE"] = col
        elif "voc" in cl or "open" in cl:
            col_map["Voc"] = col
        elif "jsc" in cl or "short" in cl:
            col_map["Jsc"] = col
        elif "fill" in cl or " ff" in cl:
            col_map["FF"] = col
        elif "bandgap" in cl or "eg" in cl or "band_gap" in cl:
            col_map["Eg"] = col
        elif "perovskite" in cl and "thickness" in cl:
            col_map["d_abs"] = col

    if "PCE" not in col_map:
        raise ValueError(f"Could not find PCE column. Available: {list(df.columns[:20])}")

    # Extract numeric targets
    targets = []
    for key in ["PCE", "Voc", "Jsc", "FF"]:
        if key in col_map:
            targets.append(pd.to_numeric(df[col_map[key]], errors="coerce"))
        else:
            targets.append(pd.Series(np.nan, index=df.index))

    Y = np.column_stack([t.values for t in targets])

    # Extract features (use whatever is available)
    feature_cols = []
    for key in ["Eg", "d_abs"]:
        if key in col_map:
            feature_cols.append(pd.to_numeric(df[col_map[key]], errors="coerce").values)

    if len(feature_cols) == 0:
        raise ValueError("No usable feature columns found")

    X = np.column_stack(feature_cols)

    # Drop rows with NaN
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y[:, 0])
    X, Y = X[mask], Y[mask]

    feature_names = [k for k in ["Eg", "d_abs"] if k in col_map]
    target_names = ["PCE", "Voc", "Jsc", "FF"]

    return X, Y, feature_names, target_names


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT AND METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def train_test_split(X, Y, test_fraction=0.2, seed=42):
    """Stratified-like split for regression."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_fraction))
    return X[idx[:split]], X[idx[split:]], Y[idx[:split]], Y[idx[split:]]


def compute_metrics(y_true, y_pred):
    """Compute R², MAPE, RMSE, MAE."""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 2:
        return {"R2": 0, "MAPE": 100, "RMSE": 0, "MAE": 0}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": float(r2), "MAPE": float(mape), "RMSE": float(rmse), "MAE": float(mae)}


def cross_validate(model_class, X, Y, n_folds=5, **model_kwargs):
    """K-fold cross-validation returning per-fold metrics."""
    n = len(X)
    fold_size = n // n_folds
    idx = np.random.permutation(n)
    all_metrics = []

    for fold in range(n_folds):
        test_idx = idx[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])

        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        model = model_class(**model_kwargs)
        model.fit(X_tr, Y_tr[:, 0])  # Predict PCE
        y_pred = model.predict(X_te)

        metrics = compute_metrics(Y_te[:, 0], y_pred)
        all_metrics.append(metrics)

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    std = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return avg, std, all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: RANDOM FOREST (from scratch — no sklearn dependency)
# ═══════════════════════════════════════════════════════════════════════════════
class DecisionTreeRegressor:
    """Simple decision tree for regression."""

    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)

    def _best_split(self, X, y):
        best_feat, best_thresh, best_gain = None, None, -np.inf
        n = len(y)
        current_mse = self._mse(y)

        n_features = X.shape[1]
        # Random feature subset for random forest
        features_to_try = np.random.choice(
            n_features, max(1, int(np.sqrt(n_features))), replace=False)

        for feat in features_to_try:
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feat], np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                gain = current_mse - self._mse(y[left_mask]) - self._mse(y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {"value": float(np.mean(y))}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {"value": float(np.mean(y))}

        left_mask = X[:, feat] <= thresh
        return {
            "feat": feat, "thresh": thresh,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _predict_one(self, x, node):
        if "value" in node:
            return node["value"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class RandomForestRegressor:
    """Random Forest ensemble of decision trees."""

    def __init__(self, n_estimators=100, max_depth=12, min_samples_split=5,
                 min_samples_leaf=2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for i in range(self.n_estimators):
            np.random.seed(i + 42)
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(boot_idx))

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)

            # OOB predictions for feature importance
            if len(oob_idx) > 0:
                oob_predictions[oob_idx] += tree.predict(X[oob_idx])
                oob_counts[oob_idx] += 1

        # Feature importance via permutation
        self._compute_importance(X, y)

    def _compute_importance(self, X, y):
        baseline = compute_metrics(y, self.predict(X))["R2"]
        importances = np.zeros(X.shape[1])

        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            perm_r2 = compute_metrics(y, self.predict(X_perm))["R2"]
            importances[j] = max(0, baseline - perm_r2)

        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return predictions.mean(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: GRADIENT BOOSTING (XGBoost-style)
# ═══════════════════════════════════════════════════════════════════════════════
class GradientBoostingRegressor:
    """
    Gradient Boosting regression (XGBoost-style).
    Uses decision stumps/shallow trees as weak learners.
    """

    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.1,
                 subsample=0.8, min_samples_leaf=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.base_prediction = 0
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples = len(X)
        self.base_prediction = float(np.mean(y))
        residuals = y - self.base_prediction
        self.trees = []
        split_counts = np.zeros(X.shape[1])

        for i in range(self.n_estimators):
            np.random.seed(i + 100)
            # Subsample
            n_sub = max(int(n_samples * self.subsample), 10)
            sub_idx = np.random.choice(n_samples, n_sub, replace=False)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=4,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[sub_idx], residuals[sub_idx])
            self.trees.append(tree)

            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions

            # Track which features are used
            self._count_splits(tree.tree, split_counts)

        total = split_counts.sum()
        self.feature_importances_ = split_counts / total if total > 0 else split_counts

    def _count_splits(self, node, counts):
        if node is None or "value" in node:
            return
        counts[node["feat"]] += 1
        self._count_splits(node.get("left"), counts)
        self._count_splits(node.get("right"), counts)

    def predict(self, X):
        pred = np.full(len(X), self.base_prediction)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: ARTIFICIAL NEURAL NETWORK (Multi-layer Perceptron)
# ═══════════════════════════════════════════════════════════════════════════════
class ANNRegressor:
    """
    Feed-forward neural network for regression.
    Architecture: input → 64 → 32 → 16 → 1 with ReLU activation.
    Trained with Adam-like optimizer (momentum + RMSprop).
    """

    def __init__(self, hidden_layers=(64, 32, 16), learning_rate=0.001,
                 epochs=500, batch_size=32, l2_reg=1e-4):
        self.hidden_layers = hidden_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.weights = []
        self.biases = []
        self.x_mean = None
        self.x_std = None
        self.y_mean = 0
        self.y_std = 1
        self.training_losses = []
        self.feature_importances_ = None

    def _init_weights(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            w = np.random.randn(layer_sizes[i + 1], fan_in) * np.sqrt(2.0 / fan_in)
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        activations = [X.T]  # Each column is a sample
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ activations[-1] + b[:, None]
            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = z  # Linear output
            activations.append(a)
        return activations

    def fit(self, X, y):
        np.random.seed(42)
        # Normalize inputs
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8

        X_norm = (X - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std

        n_features = X.shape[1]
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self._init_weights(layer_sizes)

        # Adam optimizer state
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        n_samples = len(X_norm)
        self.training_losses = []

        for epoch in range(self.epochs):
            # Mini-batch SGD
            perm = np.random.permutation(n_samples)
            epoch_loss = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = perm[start:end]
                X_batch = X_norm[batch_idx]
                y_batch = y_norm[batch_idx]
                bs = len(batch_idx)

                # Forward
                activations = self._forward(X_batch)
                pred = activations[-1].ravel()
                error = pred - y_batch
                loss = np.mean(error ** 2)
                epoch_loss += loss * bs

                # Backward
                delta = (error * 2 / bs).reshape(1, -1)

                for l in range(len(self.weights) - 1, -1, -1):
                    a_prev = activations[l]
                    dw = delta @ a_prev.T / bs + self.l2_reg * self.weights[l]
                    db = delta.mean(axis=1)

                    # Adam update
                    t = epoch * (n_samples // self.batch_size) + start // self.batch_size + 1
                    m_w[l] = beta1 * m_w[l] + (1 - beta1) * dw
                    v_w[l] = beta2 * v_w[l] + (1 - beta2) * dw ** 2
                    m_b[l] = beta1 * m_b[l] + (1 - beta1) * db
                    v_b[l] = beta2 * v_b[l] + (1 - beta2) * db ** 2

                    mw_hat = m_w[l] / (1 - beta1 ** t)
                    vw_hat = v_w[l] / (1 - beta2 ** t)
                    mb_hat = m_b[l] / (1 - beta1 ** t)
                    vb_hat = v_b[l] / (1 - beta2 ** t)

                    self.weights[l] -= self.lr * mw_hat / (np.sqrt(vw_hat) + eps)
                    self.biases[l] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

                    if l > 0:
                        delta = (self.weights[l].T @ delta) * self._relu_grad(activations[l])

            self.training_losses.append(epoch_loss / n_samples)

        # Feature importance via permutation
        self._compute_importance(X, y)

    def _compute_importance(self, X, y):
        baseline = compute_metrics(y, self.predict(X))["R2"]
        importances = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            r2_perm = compute_metrics(y, self.predict(X_perm))["R2"]
            importances[j] = max(0, baseline - r2_perm)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

    def predict(self, X):
        X_norm = (X - self.x_mean) / self.x_std
        activations = self._forward(X_norm)
        pred_norm = activations[-1].ravel()
        return pred_norm * self.y_std + self.y_mean


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelResult:
    name: str
    train_metrics: Dict
    test_metrics: Dict
    feature_importances: Optional[np.ndarray]
    feature_names: List[str]
    predictions_test: np.ndarray
    y_test: np.ndarray
    training_time: float


def compare_all_models(X, Y, feature_names, target_idx=0, test_fraction=0.2):
    """
    Train and compare RF, XGBoost, ANN on the same dataset.

    Args:
        X: feature matrix
        Y: target matrix (columns: PCE, Voc, Jsc, FF)
        feature_names: list of feature names
        target_idx: which target to predict (0=PCE)

    Returns:
        list of ModelResult
    """
    import time

    y = Y[:, target_idx]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_fraction)
    y_train, y_test = Y_train[:, target_idx], Y_test[:, target_idx]

    results = []
    n_est_rf = min(50, max(10, len(X_train) // 20))
    n_est_gb = min(150, max(30, len(X_train) // 10))
    n_epochs = min(300, max(100, len(X_train) // 5))

    models = [
        ("Random Forest", RandomForestRegressor(
            n_estimators=n_est_rf, max_depth=10, min_samples_split=5)),
        ("Gradient Boosting (XGBoost-style)", GradientBoostingRegressor(
            n_estimators=n_est_gb, max_depth=5, learning_rate=0.1)),
        ("Neural Network (ANN)", ANNRegressor(
            hidden_layers=(64, 32, 16), epochs=n_epochs, learning_rate=0.001)),
    ]

    for name, model in models:
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_met = compute_metrics(y_train, pred_train)
        test_met = compute_metrics(y_test, pred_test)

        fi = getattr(model, "feature_importances_", None)

        results.append(ModelResult(
            name=name,
            train_metrics=train_met,
            test_metrics=test_met,
            feature_importances=fi,
            feature_names=feature_names,
            predictions_test=pred_test,
            y_test=y_test,
            training_time=train_time,
        ))

    return results
