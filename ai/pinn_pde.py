"""
Feed-Forward J(V) Surrogate with Physics-Informed Regularization
=================================================================
Small neural network learning J(V) from sampled data, with soft constraints:
    - Monotonicity of J(V)
    - Smoothness penalty
    - Data MSE

HONEST LABELLING
----------------
This is NOT a PINN in the Raissi-et-al. sense. A true PINN evaluates a PDE
residual via automatic differentiation. We do not have autodiff in pure
NumPy, and a J(V) network (1-D voltage → 1-D current) has no spatial
coordinate, so it could never satisfy a spatial PDE residual. The earlier
version of this file claimed to compute Poisson and continuity residuals
from a J(V) network — that was dimensionally incoherent. Removed.

If you need a ground-truth comparison, use `compare_against_dd()` which
reports RMS error vs a real drift-diffusion J(V) solution — an honest
diagnostic rather than a fictional PDE residual.
"""
import numpy as np
from typing import Dict


class JVSurrogate:
    """Feed-forward MLP learning J(V). Backprop via chain rule on tanh layers."""

    def __init__(self, layers=(1, 64, 48, 32, 1), seed=42):
        np.random.seed(seed)
        self.weights, self.biases = [], []
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            w = np.random.randn(layers[i + 1], fan_in) * np.sqrt(2.0 / fan_in)
            self.weights.append(w)
            self.biases.append(np.zeros(layers[i + 1]))
        self.loss_history = {"total": [], "data": [], "physics": []}
        self._v_max = 1.0
        self._j_scale = 1.0

    def forward(self, x):
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ a + (b[:, None] if a.ndim > 1 else b)
            a = np.tanh(np.clip(z, -10, 10)) if i < len(self.weights) - 1 else z
        return a

    def train(self, V_data, J_data, epochs=500, lr=2e-3,
              w_mono=0.1, w_smooth=0.01, verbose=False):
        jsc = abs(J_data[0]) if len(J_data) > 0 else 1.0
        mask = J_data > -jsc * 1.5
        if mask.sum() < 20:
            mask = np.ones(len(J_data), dtype=bool)
        v_train, j_train = V_data[mask], J_data[mask]
        self._v_max = max(v_train.max(), 1e-6)
        self._j_scale = max(abs(j_train).max(), 1e-6)
        sort_idx = np.argsort(v_train)
        X = (v_train[sort_idx] / self._v_max).reshape(1, -1)
        Y = (j_train[sort_idx] / self._j_scale).reshape(1, -1)
        self.loss_history = {"total": [], "data": [], "physics": []}

        for epoch in range(epochs):
            activations = [X]
            a = X
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = w @ a + b[:, None]
                a = np.tanh(np.clip(z, -10, 10)) if i < len(self.weights) - 1 else z
                activations.append(a)
            Y_pred = activations[-1]
            err = Y_pred - Y
            L_data = float(np.mean(err ** 2))
            dY = np.diff(Y_pred, axis=1)
            L_mono = float(np.mean(np.maximum(dY, 0) ** 2))
            L_smooth = float(np.mean(np.diff(Y_pred, 2, axis=1) ** 2)) if Y_pred.shape[1] > 2 else 0.0
            L_total = L_data + w_mono * L_mono + w_smooth * L_smooth
            self.loss_history["total"].append(L_total)
            self.loss_history["data"].append(L_data)
            self.loss_history["physics"].append(w_mono * L_mono + w_smooth * L_smooth)

            delta_out = 2.0 * err / X.shape[1]
            violates = np.maximum(dY, 0)
            if violates.size > 0:
                grad_mono = np.zeros_like(Y_pred)
                grad_mono[:, 1:]  += 2 * violates / max(violates.size, 1)
                grad_mono[:, :-1] -= 2 * violates / max(violates.size, 1)
                delta_out += w_mono * grad_mono

            delta = delta_out
            for l in range(len(self.weights) - 1, -1, -1):
                a_prev = activations[l]
                if l < len(self.weights) - 1:
                    delta = delta * (1 - activations[l + 1] ** 2)
                dw = np.clip(delta @ a_prev.T, -1, 1)
                db = np.clip(np.mean(delta, axis=1), -1, 1)
                self.weights[l] -= lr * dw
                self.biases[l]  -= lr * db
                if l > 0:
                    delta = self.weights[l].T @ delta
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"  epoch {epoch:4d}  L={L_total:.5f}  data={L_data:.5f}")
        return self.loss_history

    def predict(self, V):
        X = np.clip((np.asarray(V) / self._v_max).reshape(1, -1), -0.1, 1.1)
        return self.forward(X).ravel() * self._j_scale

    def predict_with_uncertainty(self, V, n_samples=20, noise_scale=0.01):
        """Weight-perturbation ensemble (NOT MC dropout). Heuristic spread,
        not a Bayesian posterior."""
        predictions = []
        rng = np.random.default_rng(0)
        for _ in range(n_samples):
            nw = [w + rng.normal(0, 1, w.shape) * noise_scale * np.std(w)
                  for w in self.weights]
            X = np.clip((np.asarray(V) / self._v_max).reshape(1, -1), -0.1, 1.1)
            a = X
            for i, (w, b) in enumerate(zip(nw, self.biases)):
                z = w @ a + b[:, None]
                a = np.tanh(np.clip(z, -10, 10)) if i < len(nw) - 1 else z
            predictions.append(a.ravel() * self._j_scale)
        preds = np.array(predictions)
        return np.mean(preds, axis=0), np.std(preds, axis=0)


def compare_against_dd(surrogate: JVSurrogate,
                       V_ref: np.ndarray, J_ref_mA_cm2: np.ndarray) -> Dict:
    """Honest diagnostic: how closely does the surrogate match a real
    drift-diffusion J(V) curve? Replaces the old fictional 'PDE residual'."""
    J_pred = surrogate.predict(V_ref)
    err = J_pred - J_ref_mA_cm2
    rms = float(np.sqrt(np.mean(err ** 2)))
    mx = float(np.max(np.abs(err)))
    dJ = np.diff(J_pred)
    mono_viol = float(np.mean(np.maximum(dJ, 0) ** 2))
    rel_err = float(np.mean(np.abs(err) / np.maximum(np.abs(J_ref_mA_cm2), 1e-3)))
    return {
        "rms_error_mA_cm2":       rms,
        "max_error_mA_cm2":       mx,
        "mean_rel_error":         rel_err,
        "monotonicity_violation": mono_viol,
        "n_points":               len(V_ref),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Backwards-compat alias (deprecated)
# ═════════════════════════════════════════════════════════════════════════════
class PhysicsPINN(JVSurrogate):
    """DEPRECATED: was mis-named. This is a J(V) surrogate, not a PINN.
    Use JVSurrogate + compare_against_dd() instead."""

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "PhysicsPINN has been renamed to JVSurrogate. It is NOT a PINN; "
            "it is a feed-forward network with monotonicity regularization. "
            "See the module docstring.", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def get_pde_residual_report(self):
        return {
            "status": ("PDE residuals are NOT computed. This module is a J(V) "
                       "surrogate, not a PINN. Use compare_against_dd("
                       "surrogate, V_ref, J_ref) to measure agreement against "
                       "a drift-diffusion reference instead."),
        }
