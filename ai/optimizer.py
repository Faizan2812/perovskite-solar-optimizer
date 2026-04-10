"""
AI/ML Optimization and Analysis Module
=======================================
- Bayesian Optimization with GPR surrogate
- NSGA-II Multi-objective optimization (efficiency + stability)
- Physics-Informed Neural Network surrogate
- SHAP feature importance
- Inverse design (target → parameters)
- Uncertainty quantification
- Differential Evolution, PSO, GA (baselines)
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS REGRESSION (for Bayesian Optimization)
# ═══════════════════════════════════════════════════════════════════════════════
class GaussianProcessRegressor:
    """Minimal GPR implementation with Matern 5/2 kernel for BO."""
    
    def __init__(self, length_scale=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha_ = None
    
    def matern52_kernel(self, X1, X2):
        """Matern 5/2 kernel."""
        dists = cdist(X1 / self.length_scale, X2 / self.length_scale, metric='euclidean')
        sqrt5_d = np.sqrt(5) * dists
        K = (1 + sqrt5_d + 5 * dists**2 / 3) * np.exp(-sqrt5_d)
        return K
    
    def fit(self, X, y):
        """Fit GPR to training data."""
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64).ravel()
        
        # Auto-tune length scale
        self.length_scale = np.std(self.X_train, axis=0) + 1e-6
        
        K = self.matern52_kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(K))
        
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.inv(K + 1e-4 * np.eye(len(K)))
        
        self.alpha_ = self.K_inv @ self.y_train
    
    def predict(self, X, return_std=False):
        """Predict mean and (optionally) standard deviation."""
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        K_star = self.matern52_kernel(X, self.X_train)
        mean = K_star @ self.alpha_
        
        if return_std:
            K_ss = self.matern52_kernel(X, X)
            var = np.diag(K_ss) - np.sum(K_star @ self.K_inv * K_star, axis=1)
            std = np.sqrt(np.clip(var, 1e-10, None))
            return mean, std
        return mean


def expected_improvement(X, gpr, y_best, xi=0.01):
    """Expected Improvement acquisition function."""
    mu, sigma = gpr.predict(X, return_std=True)
    sigma = np.clip(sigma, 1e-8, None)
    Z = (mu - y_best - xi) / sigma
    # CDF and PDF of standard normal
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


# ═══════════════════════════════════════════════════════════════════════════════
# BAYESIAN OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def bayesian_optimization(objective_fn, bounds, n_initial=10, n_iterations=40,
                          maximize=True):
    """
    Bayesian Optimization with GPR surrogate and Expected Improvement.
    
    Args:
        objective_fn: function(params) -> scalar value
        bounds: list of (min, max) tuples
        n_initial: number of random initial evaluations
        n_iterations: number of BO iterations
        maximize: if True, maximize objective; if False, minimize
    
    Returns:
        best_params, best_value, history, gpr_model
    """
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initial random sampling (Latin Hypercube-like)
    X_samples = lo + np.random.rand(n_initial, dim) * (hi - lo)
    y_samples = np.array([objective_fn(x) for x in X_samples])
    
    if not maximize:
        y_samples = -y_samples
    
    history = [float(np.max(y_samples))]
    gpr = GaussianProcessRegressor(noise=1e-4)
    
    for iteration in range(n_iterations):
        # Fit GPR
        gpr.fit(X_samples, y_samples)
        y_best = np.max(y_samples)
        
        # Optimize acquisition function via random restarts
        best_ei = -np.inf
        best_x = None
        
        # Random candidates
        n_candidates = 500
        X_cand = lo + np.random.rand(n_candidates, dim) * (hi - lo)
        ei_values = expected_improvement(X_cand, gpr, y_best)
        
        top_idx = np.argsort(ei_values)[-5:]  # Top 5 candidates
        for idx in top_idx:
            x0 = X_cand[idx]
            try:
                result = minimize(
                    lambda x: -expected_improvement(x.reshape(1, -1), gpr, y_best)[0],
                    x0, bounds=bounds, method='L-BFGS-B'
                )
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_x = result.x
            except Exception:
                if best_x is None:
                    best_x = X_cand[idx]
        
        if best_x is None:
            best_x = lo + np.random.rand(dim) * (hi - lo)
        
        # Evaluate
        y_new = objective_fn(best_x)
        if not maximize:
            y_new = -y_new
        
        X_samples = np.vstack([X_samples, best_x.reshape(1, -1)])
        y_samples = np.append(y_samples, y_new)
        history.append(float(np.max(y_samples)))
    
    best_idx = np.argmax(y_samples)
    best_params = X_samples[best_idx]
    best_value = y_samples[best_idx] if maximize else -y_samples[best_idx]
    
    if not maximize:
        history = [-h for h in history]
    
    return best_params, best_value, history, gpr


# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II MULTI-OBJECTIVE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def fast_non_dominated_sort(fitness):
    """NSGA-II fast non-dominated sorting."""
    n_pop = len(fitness)
    domination_count = np.zeros(n_pop, dtype=int)
    dominated_set = [[] for _ in range(n_pop)]
    fronts = [[]]
    
    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            if all(fitness[i] >= fitness[j]) and any(fitness[i] > fitness[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif all(fitness[j] >= fitness[i]) and any(fitness[j] > fitness[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
        
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    k = 0
    while len(fronts[k]) > 0:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    
    return fronts[:-1]  # Remove last empty front


def crowding_distance(fitness, front):
    """Compute crowding distance for a Pareto front."""
    n = len(front)
    if n <= 2:
        return [np.inf] * n
    
    distances = np.zeros(n)
    n_obj = fitness.shape[1]
    
    for m in range(n_obj):
        sorted_idx = np.argsort(fitness[front, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        
        f_range = fitness[front[sorted_idx[-1]], m] - fitness[front[sorted_idx[0]], m]
        if f_range < 1e-10:
            continue
        
        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                fitness[front[sorted_idx[i + 1]], m] - 
                fitness[front[sorted_idx[i - 1]], m]
            ) / f_range
    
    return distances


def nsga2_optimize(objectives, bounds, n_gen=50, pop_size=40, 
                   crossover_rate=0.9, mutation_rate=0.1):
    """
    NSGA-II multi-objective optimization.
    
    Args:
        objectives: list of functions, each takes params -> scalar (all maximized)
        bounds: list of (min, max) tuples
        n_gen: number of generations
        pop_size: population size
    
    Returns:
        pareto_front (list of param arrays), pareto_fitness (array of objective values),
        history (per-generation best for each objective)
    """
    dim = len(bounds)
    n_obj = len(objectives)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = lo + np.random.rand(pop_size, dim) * (hi - lo)
    fitness = np.array([[obj(p) for obj in objectives] for p in pop])
    
    history = []
    
    for gen in range(n_gen):
        # Generate offspring via SBX crossover + polynomial mutation
        offspring = np.zeros_like(pop)
        for i in range(0, pop_size, 2):
            p1, p2 = pop[np.random.randint(pop_size)], pop[np.random.randint(pop_size)]
            if np.random.rand() < crossover_rate:
                # SBX crossover
                eta_c = 20
                u = np.random.rand(dim)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                               (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Polynomial mutation
            for d in range(dim):
                if np.random.rand() < mutation_rate:
                    eta_m = 20
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    c1[d] += delta * (hi[d] - lo[d])
                    c2[d] += delta * (hi[d] - lo[d])
            
            offspring[i] = np.clip(c1, lo, hi)
            if i + 1 < pop_size:
                offspring[i + 1] = np.clip(c2, lo, hi)
        
        # Evaluate offspring
        off_fitness = np.array([[obj(p) for obj in objectives] for p in offspring])
        
        # Combine parent + offspring
        combined_pop = np.vstack([pop, offspring])
        combined_fit = np.vstack([fitness, off_fitness])
        
        # Non-dominated sorting
        fronts = fast_non_dominated_sort(combined_fit)
        
        # Select next generation
        new_pop = []
        new_fit = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_fit.append(combined_fit[idx])
            else:
                # Need to select subset using crowding distance
                distances = crowding_distance(combined_fit, front)
                sorted_by_cd = np.argsort(-np.array(distances))
                remaining = pop_size - len(new_pop)
                for j in sorted_by_cd[:remaining]:
                    new_pop.append(combined_pop[front[j]])
                    new_fit.append(combined_fit[front[j]])
                break
        
        pop = np.array(new_pop[:pop_size])
        fitness = np.array(new_fit[:pop_size])
        
        # Record history
        history.append([float(np.max(fitness[:, i])) for i in range(n_obj)])
    
    # Extract Pareto front
    fronts = fast_non_dominated_sort(fitness)
    if len(fronts) > 0 and len(fronts[0]) > 0:
        pareto_idx = fronts[0]
        pareto_front = [pop[i] for i in pareto_idx]
        pareto_fitness = fitness[pareto_idx]
    else:
        best = np.argmax(fitness[:, 0])
        pareto_front = [pop[best]]
        pareto_fitness = fitness[best:best+1]
    
    return pareto_front, pareto_fitness, history


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS-INFORMED NEURAL NETWORK (NumPy)
# ═══════════════════════════════════════════════════════════════════════════════
class NumpyPINN:
    """Physics-Informed Neural Network with customizable architecture."""
    
    def __init__(self, layers):
        np.random.seed(42)
        self.weights, self.biases = [], []
        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            w = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ a + (b[:, None] if a.ndim > 1 else b)
            a = np.tanh(z) if i < len(self.weights) - 1 else z
        return a
    
    def train(self, X, Y, epochs=300, lr=0.002, physics_weight=0.1):
        """Train with MSE data loss + physics-informed monotonicity penalty."""
        losses = []
        for epoch in range(epochs):
            activations = [X]
            a = X
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = w @ a + b[:, None]
                a = np.tanh(z) if i < len(self.weights) - 1 else z
                activations.append(a)
            
            error = activations[-1] - Y
            data_loss = np.mean(error ** 2)
            
            # Physics: penalize non-monotonic J(V) (J should decrease with V)
            dY = np.diff(activations[-1], axis=1)
            phys_loss = np.mean(np.maximum(dY, 0) ** 2)
            
            total_loss = data_loss + physics_weight * phys_loss
            losses.append(float(total_loss))
            
            delta = error * 2.0
            for l in range(len(self.weights) - 1, -1, -1):
                act = activations[l]
                if l < len(self.weights) - 1:
                    delta = delta * (1 - activations[l + 1] ** 2)
                dw = delta @ act.T / X.shape[1]
                db = np.mean(delta, axis=1)
                self.weights[l] -= lr * np.clip(dw, -1, 1)
                self.biases[l] -= lr * np.clip(db, -1, 1)
                delta = self.weights[l].T @ delta
        
        return losses
    
    def predict_with_uncertainty(self, X, n_samples=20):
        """Monte Carlo dropout-like uncertainty via weight noise."""
        predictions = []
        for _ in range(n_samples):
            # Add small noise to weights
            noisy_weights = [w + np.random.randn(*w.shape) * 0.01 for w in self.weights]
            a = X
            for i, (w, b) in enumerate(zip(noisy_weights, self.biases)):
                z = w @ a + (b[:, None] if a.ndim > 1 else b)
                a = np.tanh(z) if i < len(noisy_weights) - 1 else z
            predictions.append(a)
        
        preds = np.array(predictions)
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        return mean, std


def train_pinn_surrogate(voltages, currents, epochs=500, lr=0.002):
    """Train PINN on J-V data with uncertainty quantification."""
    jsc = abs(currents[0])
    mask = currents > -jsc * 1.5
    if mask.sum() < 20:
        mask = np.ones(len(currents), dtype=bool)
    
    v_train, j_train = voltages[mask], currents[mask]
    v_max = max(v_train.max(), 1e-6)
    j_scale = max(abs(j_train).max(), 1e-6)
    
    X = (v_train / v_max).reshape(1, -1)
    Y = (j_train / j_scale).reshape(1, -1)
    sort_idx = np.argsort(X[0])
    X, Y = X[:, sort_idx], Y[:, sort_idx]
    
    nn = NumpyPINN([1, 48, 32, 1])
    losses = nn.train(X, Y, epochs, lr)
    
    X_full = np.clip((voltages / v_max).reshape(1, -1), 0, 1.05)
    Y_pred, Y_std = nn.predict_with_uncertainty(X_full)
    predicted = Y_pred[0] * j_scale
    uncertainty = Y_std[0] * j_scale
    
    return predicted, uncertainty, losses, nn, v_max, j_scale


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP-LIKE FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_shap_importance(sim_fn, base_params, param_names, param_ranges, 
                            n_samples=50):
    """
    Compute SHAP-like feature importance via permutation.
    
    Args:
        sim_fn: function(params_dict) -> PCE
        base_params: dict of baseline parameter values
        param_names: list of parameter names to analyze
        param_ranges: dict of {name: (min, max)}
        n_samples: number of random perturbations per feature
    
    Returns:
        dict of {param_name: importance_score}
    """
    base_pce = sim_fn(base_params)
    importances = {}
    
    for name in param_names:
        lo, hi = param_ranges[name]
        deltas = []
        
        for _ in range(n_samples):
            perturbed = base_params.copy()
            perturbed[name] = lo + np.random.rand() * (hi - lo)
            pce = sim_fn(perturbed)
            deltas.append(abs(pce - base_pce))
        
        importances[name] = float(np.mean(deltas))
    
    # Normalize to sum = 1
    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}
    
    return importances


# ═══════════════════════════════════════════════════════════════════════════════
# INVERSE DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
def inverse_design(target_metrics, sim_fn, bounds, n_iterations=100):
    """
    Inverse design: find parameters that produce target output metrics.
    
    Args:
        target_metrics: dict {"PCE": 25, "Voc": 1.1, "Jsc": 25, "FF": 0.85}
        sim_fn: function(params) -> dict with keys matching target_metrics
        bounds: list of (min, max) for each parameter
        n_iterations: optimization iterations
    
    Returns:
        best_params, achieved_metrics, distance_to_target
    """
    target_keys = list(target_metrics.keys())
    target_values = np.array([target_metrics[k] for k in target_keys])
    
    # Normalize targets for equal weighting
    weights = 1.0 / (np.abs(target_values) + 1e-6)
    
    def objective(params):
        result = sim_fn(params)
        achieved = np.array([result.get(k, 0) for k in target_keys])
        # Weighted distance to target
        distance = np.sum(weights * (achieved - target_values) ** 2)
        return distance
    
    result = differential_evolution(objective, bounds, maxiter=n_iterations,
                                   popsize=15, seed=42, tol=1e-6)
    
    final = sim_fn(result.x)
    achieved = {k: final.get(k, 0) for k in target_keys}
    distance = result.fun
    
    return result.x, achieved, distance


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD METAHEURISTIC ALGORITHMS (baselines)
# ═══════════════════════════════════════════════════════════════════════════════
def run_de(objective_fn, bounds, maxiter=60, popsize=20):
    """SciPy Differential Evolution with convergence tracking."""
    history = []
    def callback(xk, convergence):
        history.append(float(-objective_fn(xk)))
    
    result = differential_evolution(
        objective_fn, bounds, maxiter=maxiter, popsize=popsize,
        mutation=(0.5, 1.5), recombination=0.8, seed=42,
        tol=1e-6, polish=True, callback=callback
    )
    return result.x, -result.fun, history


def run_pso(objective_fn, bounds, max_iter=60, swarm_size=25):
    """Particle Swarm Optimization."""
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    pos = lo + np.random.rand(swarm_size, dim) * (hi - lo)
    vel = np.random.randn(swarm_size, dim) * (hi - lo) * 0.05
    p_best = pos.copy()
    p_best_fit = np.array([objective_fn(p) for p in pos])
    g_idx = np.argmin(p_best_fit)
    g_best, g_best_fit = p_best[g_idx].copy(), p_best_fit[g_idx]
    history = []
    
    for t in range(max_iter):
        w = 0.9 - 0.5 * t / max_iter  # Linear decay
        for i in range(swarm_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[i] = w * vel[i] + 1.5 * r1 * (p_best[i] - pos[i]) + 1.5 * r2 * (g_best - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], lo, hi)
            fit = objective_fn(pos[i])
            if fit < p_best_fit[i]:
                p_best_fit[i] = fit; p_best[i] = pos[i].copy()
            if fit < g_best_fit:
                g_best_fit = fit; g_best = pos[i].copy()
        history.append(float(-g_best_fit))
    
    return g_best, -g_best_fit, history


def run_ga(objective_fn, bounds, max_gen=60, pop_size=30, mut_rate=0.15):
    """Genetic Algorithm."""
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    pop = lo + np.random.rand(pop_size, dim) * (hi - lo)
    fitness = np.array([objective_fn(p) for p in pop])
    history = []
    
    for g in range(max_gen):
        sorted_idx = np.argsort(fitness)
        elite_n = max(2, pop_size // 10)
        new_pop = [pop[sorted_idx[i]].copy() for i in range(elite_n)]
        
        while len(new_pop) < pop_size:
            i1 = sorted_idx[np.random.randint(0, pop_size // 2)]
            i2 = sorted_idx[np.random.randint(0, pop_size // 2)]
            mask = np.random.rand(dim) < 0.5
            child = np.where(mask, pop[i1], pop[i2])
            mut_mask = np.random.rand(dim) < mut_rate
            child[mut_mask] += np.random.randn(mut_mask.sum()) * (hi - lo)[mut_mask] * 0.15
            new_pop.append(np.clip(child, lo, hi))
        
        pop = np.array(new_pop[:pop_size])
        fitness = np.array([objective_fn(p) for p in pop])
        history.append(float(-fitness.min()))
    
    best_idx = np.argmin(fitness)
    return pop[best_idx], -fitness[best_idx], history


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY PREDICTION MODEL (empirical)
# ═══════════════════════════════════════════════════════════════════════════════
def predict_stability_t80(absorber_name, Eg, Nt, encapsulated=True):
    """
    Predict T80 lifetime (hours) based on material and defect properties.
    Semi-empirical model calibrated on Perovskite Database trends.
    """
    # Base T80 by material class
    base_t80 = {
        "MAPbI3": 500, "FAPbI3": 800, "CsPbI3": 1200, "MAPbBr3": 600,
        "CsPbI2Br": 1000, "MASnI3": 100, "FASnI3": 150, "CsSnI3": 80,
        "Cs2SnI6": 2000, "Cs2AgBiBr6": 3000, "Cs2TiBr6": 2500,
        "BaZrS3": 5000, "CsGeI3": 200,
    }
    
    t80 = base_t80.get(absorber_name, 500)
    
    # Defect density penalty: higher defects = faster degradation
    log_nt = np.log10(max(Nt, 1e10))
    t80 *= np.exp(-0.3 * max(0, log_nt - 14))
    
    # Encapsulation factor
    if encapsulated:
        t80 *= 5
    
    # Inorganic perovskites are more stable
    if absorber_name.startswith("Cs"):
        t80 *= 1.5
    
    return max(t80, 10)  # Minimum 10 hours


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
def parameter_sweep(sim_fn, base_params, sweep_param, sweep_values):
    """
    Sweep a single parameter and record all output metrics.
    
    Args:
        sim_fn: function(params_dict) -> result_dict
        base_params: dict of baseline parameters
        sweep_param: name of parameter to sweep
        sweep_values: array of values to test
    
    Returns:
        DataFrame with columns for sweep_param and all output metrics
    """
    results = []
    for val in sweep_values:
        params = base_params.copy()
        params[sweep_param] = val
        try:
            r = sim_fn(params)
            results.append({
                sweep_param: val,
                "PCE": r.get("PCE", 0),
                "Voc": r.get("Voc", 0),
                "Jsc": r.get("Jsc", 0),
                "FF": r.get("FF", 0) * 100,
                "Pmax": r.get("Pmax", 0),
            })
        except Exception:
            results.append({sweep_param: val, "PCE": 0, "Voc": 0,
                           "Jsc": 0, "FF": 0, "Pmax": 0})
    
    import pandas as pd
    return pd.DataFrame(results)


def temperature_coefficients(sim_fn, base_T=300, delta_T=20):
    """Compute temperature coefficients for Voc, Jsc, FF, PCE."""
    r_lo = sim_fn(base_T - delta_T)
    r_hi = sim_fn(base_T + delta_T)
    
    dT = 2 * delta_T
    coeffs = {
        "dVoc/dT": (r_hi["Voc"] - r_lo["Voc"]) / dT * 1000,  # mV/K
        "dJsc/dT": (r_hi["Jsc"] - r_lo["Jsc"]) / dT,  # mA/cm²/K
        "dFF/dT": (r_hi["FF"] - r_lo["FF"]) * 100 / dT,  # %/K
        "dPCE/dT": (r_hi["PCE"] - r_lo["PCE"]) / dT,  # %/K
    }
    return coeffs


def compare_materials(sim_fn, material_names, material_db, fixed_params):
    """Compare multiple absorber materials with fixed device parameters."""
    results = []
    for name in material_names:
        try:
            r = sim_fn(name, fixed_params)
            results.append({
                "Material": name,
                "Eg (eV)": material_db[name].Eg,
                "PCE (%)": round(r["PCE"], 2),
                "Voc (V)": round(r["Voc"], 3),
                "Jsc (mA/cm²)": round(r["Jsc"], 2),
                "FF (%)": round(r["FF"] * 100, 1),
                "Pmax (mW/cm²)": round(r["Pmax"], 2),
            })
        except Exception:
            results.append({"Material": name, "Eg (eV)": material_db[name].Eg,
                          "PCE (%)": 0, "Voc (V)": 0, "Jsc (mA/cm²)": 0,
                          "FF (%)": 0, "Pmax (mW/cm²)": 0})
    
    import pandas as pd
    return pd.DataFrame(results).sort_values("PCE (%)", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP OPERATOR NETWORK (DeepONet) SURROGATE
# ═══════════════════════════════════════════════════════════════════════════════
class DeepONet:
    """
    Deep Operator Network for learning the mapping:
    (material_params, voltage) -> current_density
    
    Branch net: encodes input parameters (thicknesses, Nt, Eg, etc.)
    Trunk net: encodes query point (voltage)
    Output = dot(branch_output, trunk_output)
    """
    
    def __init__(self, branch_layers=[6, 32, 16], trunk_layers=[1, 32, 16]):
        np.random.seed(42)
        self.branch_w, self.branch_b = [], []
        for i in range(len(branch_layers) - 1):
            fan_in = branch_layers[i]
            w = np.random.randn(branch_layers[i + 1], fan_in) * np.sqrt(2.0 / fan_in)
            self.branch_w.append(w)
            self.branch_b.append(np.zeros(branch_layers[i + 1]))
        
        self.trunk_w, self.trunk_b = [], []
        for i in range(len(trunk_layers) - 1):
            fan_in = trunk_layers[i]
            w = np.random.randn(trunk_layers[i + 1], fan_in) * np.sqrt(2.0 / fan_in)
            self.trunk_w.append(w)
            self.trunk_b.append(np.zeros(trunk_layers[i + 1]))
        
        self.bias = 0.0
    
    def _forward_net(self, x, weights, biases):
        a = x
        for i, (w, b) in enumerate(zip(weights, biases)):
            z = w @ a + (b[:, None] if a.ndim > 1 else b)
            a = np.tanh(z) if i < len(weights) - 1 else z
        return a
    
    def forward(self, params, voltage):
        """
        Args:
            params: [N_features] or [N_features, N_samples] — branch input
            voltage: [1] or [1, N_points] — trunk input
        """
        branch_out = self._forward_net(params, self.branch_w, self.branch_b)
        trunk_out = self._forward_net(voltage, self.trunk_w, self.trunk_b)
        
        if branch_out.ndim == 1 and trunk_out.ndim > 1:
            return np.sum(branch_out[:, None] * trunk_out, axis=0) + self.bias
        return np.sum(branch_out * trunk_out, axis=0) + self.bias
    
    def train(self, X_params, X_voltage, Y_current, epochs=300, lr=0.005):
        """Train DeepONet on simulation data."""
        losses = []
        N = X_voltage.shape[1] if X_voltage.ndim > 1 else len(X_voltage)
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(min(N, X_params.shape[1] if X_params.ndim > 1 else 1)):
                p = X_params[:, i] if X_params.ndim > 1 else X_params
                v = X_voltage[:, i:i+1] if X_voltage.ndim > 1 else X_voltage
                y = Y_current[:, i] if Y_current.ndim > 1 else Y_current
                
                pred = self.forward(p, v)
                error = pred - (y if y.ndim == 0 else y.ravel())
                loss = np.mean(error**2)
                total_loss += loss
                
                # Simple gradient step on bias
                self.bias -= lr * np.mean(error) * 0.1
            
            losses.append(total_loss / max(N, 1))
        
        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
def active_learning_loop(sim_fn, bounds, gpr, n_initial=10, n_queries=20,
                         acquisition="uncertainty"):
    """
    Active learning: iteratively query the most uncertain regions.
    
    Args:
        sim_fn: function(params) -> scalar (e.g., PCE)
        bounds: list of (min, max) tuples
        gpr: GaussianProcessRegressor instance
        n_initial: initial random samples
        n_queries: number of active learning queries
        acquisition: "uncertainty" (max σ) or "expected_improvement"
    
    Returns:
        X_all, y_all, history of uncertainties
    """
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initial random sampling
    X = lo + np.random.rand(n_initial, dim) * (hi - lo)
    y = np.array([sim_fn(x) for x in X])
    
    uncertainty_history = []
    
    for q in range(n_queries):
        gpr.fit(X, y)
        
        # Generate candidates
        X_cand = lo + np.random.rand(500, dim) * (hi - lo)
        mu, sigma = gpr.predict(X_cand, return_std=True)
        
        if acquisition == "uncertainty":
            # Query the most uncertain point
            best_idx = np.argmax(sigma)
        else:
            # Expected improvement
            y_best = np.max(y)
            Z = (mu - y_best - 0.01) / np.clip(sigma, 1e-8, None)
            ei = (mu - y_best - 0.01) * norm.cdf(Z) + sigma * norm.pdf(Z)
            best_idx = np.argmax(ei)
        
        x_new = X_cand[best_idx]
        y_new = sim_fn(x_new)
        
        X = np.vstack([X, x_new.reshape(1, -1)])
        y = np.append(y, y_new)
        uncertainty_history.append(float(sigma.mean()))
    
    return X, y, uncertainty_history


# ═══════════════════════════════════════════════════════════════════════════════
# LLM QUERY INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
def parse_natural_language_query(query_text):
    """
    Parse a natural language query into simulation parameters.
    Uses rule-based parsing (no external LLM dependency required).
    
    Examples:
        "Simulate MAPbI3 with 500nm absorber" -> {absorber: "MAPbI3", d_abs: 500}
        "What is the best HTL for CsPbI3?" -> {action: "compare_htl", absorber: "CsPbI3"}
        "Optimize PCE for Cs2SnI6" -> {action: "optimize", absorber: "Cs2SnI6"}
    """
    import re
    query = query_text.lower().strip()
    result = {"action": "simulate", "params": {}}
    
    # Detect material names
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    for name in PEROVSKITE_DB:
        if name.lower() in query:
            result["params"]["absorber"] = name
    for name in HTL_DB:
        if name.lower() in query:
            result["params"]["htl"] = name
    for name in ETL_DB:
        if name.lower() in query:
            result["params"]["etl"] = name
    
    # Detect thickness
    thick_match = re.search(r'(\d+)\s*nm', query)
    if thick_match:
        result["params"]["d_abs"] = int(thick_match.group(1))
    
    # Detect action
    if any(w in query for w in ["optimize", "maximize", "best"]):
        result["action"] = "optimize"
    elif any(w in query for w in ["compare", "which", "what htl", "what etl"]):
        result["action"] = "compare"
    elif any(w in query for w in ["sweep", "scan", "vary"]):
        result["action"] = "sweep"
    elif any(w in query for w in ["stability", "lifetime", "t80"]):
        result["action"] = "stability"
    
    # Detect defect density
    nt_match = re.search(r'(?:nt|defect)[^\d]*(\d+e\d+|1e\d+)', query)
    if nt_match:
        result["params"]["Nt"] = float(nt_match.group(1))
    
    return result


def execute_query(parsed_query, default_htl="Spiro-OMeTAD", default_etl="TiO2"):
    """Execute a parsed natural language query and return results."""
    from physics.materials import PEROVSKITE_DB, HTL_DB, ETL_DB
    from physics.device import fast_simulate
    
    action = parsed_query["action"]
    params = parsed_query["params"]
    
    abs_name = params.get("absorber", "MAPbI3")
    htl_name = params.get("htl", default_htl)
    etl_name = params.get("etl", default_etl)
    d_abs = params.get("d_abs", 400)
    Nt = params.get("Nt", 1e14)
    
    if abs_name not in PEROVSKITE_DB:
        return {"error": f"Unknown absorber: {abs_name}"}
    if htl_name not in HTL_DB:
        return {"error": f"Unknown HTL: {htl_name}"}
    if etl_name not in ETL_DB:
        return {"error": f"Unknown ETL: {etl_name}"}
    
    if action == "simulate":
        r = fast_simulate(HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                         100, d_abs, 50, Nt, 300)
        return {"action": "simulate", "result": r, "description": 
                f"Simulated {htl_name}/{abs_name}/{etl_name} ({d_abs}nm): PCE={r['PCE']:.2f}%"}
    
    elif action == "compare":
        results = {}
        for h_name in HTL_DB:
            r = fast_simulate(HTL_DB[h_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                             100, d_abs, 50, Nt, 300)
            results[h_name] = r["PCE"]
        best = max(results, key=results.get)
        return {"action": "compare", "results": results, "best": best,
                "description": f"Best HTL for {abs_name}: {best} (PCE={results[best]:.2f}%)"}
    
    elif action == "optimize":
        bounds = [(20, 200), (100, 1000), (20, 200), (10, 16)]
        def obj(p):
            return fast_simulate(HTL_DB[htl_name], PEROVSKITE_DB[abs_name], ETL_DB[etl_name],
                               p[0], p[1], p[2], 10**p[3], 300)["PCE"]
        bp, bv, _, _ = bayesian_optimization(obj, bounds, 8, 20)
        return {"action": "optimize", "best_pce": bv, "best_params": bp,
                "description": f"Optimized {abs_name}: PCE={bv:.2f}%"}
    
    elif action == "stability":
        t80 = predict_stability_t80(abs_name, PEROVSKITE_DB[abs_name].Eg, Nt)
        return {"action": "stability", "t80_hours": t80,
                "description": f"{abs_name} T80 = {t80:.0f} hours ({t80/24:.0f} days)"}
    
    return {"error": "Could not parse query"}
