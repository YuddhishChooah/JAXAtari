"""
Optimizers for parameter search.

Implements CMA-ES, Bayesian Optimization, and Random Search.
"""

import numpy as np
from typing import Dict, Any, Tuple, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve


def params_to_vector(params: Dict) -> Tuple[np.ndarray, Any]:
    """Flatten a parameter dict to a 1D vector."""
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    flat_arrays = [np.atleast_1d(np.asarray(p, dtype=np.float32)) for p in flat_params]
    vector = np.concatenate([a.ravel() for a in flat_arrays])
    shapes = [a.shape for a in flat_arrays]
    return vector, (tree_def, shapes)


def vector_to_params(vector: np.ndarray, structure: Tuple) -> Dict:
    """Reconstruct parameter dict from 1D vector."""
    tree_def, shapes = structure
    flat_arrays = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(np.array(shape)))
        arr = vector[idx:idx + size].reshape(shape)
        if shape == (1,):
            arr = arr[0]
        flat_arrays.append(arr)
        idx += size
    return jax.tree_util.tree_unflatten(tree_def, flat_arrays)


class CMAESOptimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    A powerful derivative-free optimizer that maintains a multivariate
    Gaussian distribution over parameters and adapts it based on fitness.
    """
    
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
    
    def optimize(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any], list]:
        """Run CMA-ES optimization."""
        
        base_params = init_params_fn()
        x0, structure = params_to_vector(base_params)
        n = len(x0)
        
        sigma = self.config.param_perturbation_scale
        lambda_ = self.config.num_param_samples
        mu = lambda_ // 2
        
        # CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        
        mean = x0.copy()
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_params = base_params
        best_score = float('-inf')
        best_metrics = None
        history = []
        
        if self.config.verbose:
            print(f"  CMA-ES: n={n} params, λ={lambda_} population, "
                  f"{self.config.cma_es_generations} generations")
        
        for gen in range(self.config.cma_es_generations):
            key, sample_key = jrandom.split(key)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sqrt_C = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            
            # Sample offspring
            z = np.random.randn(lambda_, n)
            offspring = mean + sigma * (z @ sqrt_C.T)
            
            # Evaluate
            fitness_scores = []
            for i in range(lambda_):
                candidate_params = vector_to_params(offspring[i], structure)
                key, eval_key = jrandom.split(key)
                
                try:
                    metrics = self.evaluator.evaluate(policy_fn, candidate_params, eval_key)
                    score = metrics['avg_return']
                except Exception as e:
                    if i == 0 and gen == 0:
                        print(f"    WARNING: Evaluation failed: {e}")
                    score = float('-inf')
                    metrics = None
                
                fitness_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_params = candidate_params
                    best_metrics = metrics
            
            # Sort and select
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_offspring = offspring[sorted_indices[:mu]]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * selected_offspring, axis=0)
            
            # Update paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (mean - old_mean) / sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi_n < (1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
            
            # Update covariance
            artmp = (selected_offspring - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = float(np.clip(sigma, 0.01, 2.0))
            
            gen_best = max(fitness_scores)
            gen_mean = np.mean([s for s in fitness_scores if s > float('-inf')])
            
            history.append({
                'generation': gen + 1,
                'best': gen_best,
                'mean': gen_mean,
                'sigma': sigma,
                'overall_best': best_score,
            })
            
            if self.config.verbose:
                print(f"    Gen {gen+1}: best={gen_best:.2f}, mean={gen_mean:.2f}, "
                      f"σ={sigma:.3f}, overall_best={best_score:.2f}")
        
        return best_params, best_metrics, history


class GaussianProcess:
    """Gaussian Process regressor for Bayesian Optimization."""
    
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dists = cdist(X1 / self.length_scale, X2 / self.length_scale, metric='sqeuclidean')
        return np.exp(-0.5 * dists)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.array(X)
        self.y_train = np.array(y).ravel()
        
        self.y_mean = np.mean(self.y_train)
        self.y_std = np.std(self.y_train) + 1e-8
        y_normalized = (self.y_train - self.y_mean) / self.y_std
        
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(self.X_train))
        
        try:
            L = cholesky(K, lower=True)
            self.alpha = cho_solve((L, True), y_normalized)
            self.L = L
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(len(self.X_train))
            L = cholesky(K, lower=True)
            self.alpha = cho_solve((L, True), y_normalized)
            self.L = L
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        K_star = self._rbf_kernel(X, self.X_train)
        y_mean = K_star @ self.alpha
        y_mean = y_mean * self.y_std + self.y_mean
        
        if return_std:
            K_star_star = self._rbf_kernel(X, X)
            v = cho_solve((self.L, True), K_star.T)
            y_var = np.diag(K_star_star) - np.sum(K_star.T * v, axis=0)
            y_var = np.maximum(y_var, 0)
            y_std = np.sqrt(y_var) * self.y_std
            return y_mean, y_std
        
        return y_mean


class BayesianOptimizer:
    """
    Gaussian Process-based Bayesian Optimization with UCB acquisition.
    """

    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator

    def optimize(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any], list]:
        """Run Bayesian optimization."""
        
        base_params = init_params_fn()
        x0, structure = params_to_vector(base_params)
        x0 = np.array(x0)
        n = len(x0)

        best_params = base_params
        best_score = float('-inf')
        best_metrics = None
        
        X_train = []
        y_train = []
        history = []

        # Evaluate initial point
        key, eval_key = jrandom.split(key)
        metrics = self.evaluator.evaluate(policy_fn, base_params, eval_key)
        score = float(metrics["avg_return"])
        X_train.append(x0.copy())
        y_train.append(score)
        
        if score > best_score:
            best_score = score
            best_params = base_params
            best_metrics = metrics

        if self.config.verbose:
            print(f"  Bayesian Opt: n={n} params, {self.config.cma_es_generations} iterations")
            print(f"    Initial: return={score:.2f}, win_rate={metrics['win_rate']:.2%}")

        # Initial random exploration
        sigma = self.config.param_perturbation_scale
        n_initial = min(5, self.config.num_param_samples)
        
        for i in range(n_initial):
            key, noise_key, eval_key = jrandom.split(key, 3)
            noise = np.array(jrandom.normal(noise_key, shape=(n,))) * sigma * 2.0
            x_candidate = x0 + noise
            candidate_params = vector_to_params(x_candidate, structure)
            
            try:
                metrics = self.evaluator.evaluate(policy_fn, candidate_params, eval_key)
                score = float(metrics["avg_return"])
            except Exception:
                score = float('-inf')
                metrics = None
            
            X_train.append(x_candidate.copy())
            y_train.append(score)
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
                best_metrics = metrics

        # GP-UCB loop
        gp = GaussianProcess(length_scale=sigma, noise=1e-4)
        total_budget = self.config.cma_es_generations * self.config.num_param_samples - len(X_train)
        
        for iteration in range(total_budget):
            X = np.array(X_train)
            y = np.array(y_train)
            
            valid_mask = y > float('-inf')
            if np.sum(valid_mask) < 2:
                key, noise_key = jrandom.split(key)
                x_next = x0 + np.array(jrandom.normal(noise_key, shape=(n,))) * sigma
            else:
                gp.fit(X[valid_mask], y[valid_mask])
                
                # Anneal beta
                progress = iteration / max(total_budget - 1, 1)
                beta = 2.5 * (1 - progress) + 0.5 * progress
                
                # Generate candidates
                best_idx = np.argmax(y)
                n_local = 700
                n_global = 300
                
                local_candidates = X[best_idx] + np.random.randn(n_local, n) * sigma * 0.5
                global_candidates = x0 + np.random.randn(n_global, n) * sigma * 2.0
                candidates = np.vstack([local_candidates, global_candidates])
                
                # UCB acquisition
                mean, std = gp.predict(candidates, return_std=True)
                ucb = mean + beta * std
                x_next = candidates[np.argmax(ucb)]
            
            # Evaluate
            key, eval_key = jrandom.split(key)
            candidate_params = vector_to_params(x_next, structure)
            
            try:
                metrics = self.evaluator.evaluate(policy_fn, candidate_params, eval_key)
                score = float(metrics["avg_return"])
            except Exception:
                score = float('-inf')
                metrics = None
            
            X_train.append(x_next.copy())
            y_train.append(score)
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
                best_metrics = metrics
            
            if (iteration + 1) % max(1, total_budget // 10) == 0:
                history.append({
                    'iteration': iteration + 1,
                    'best': best_score,
                    'current': score,
                })
                if self.config.verbose:
                    print(f"    Iter {iteration + 1}/{total_budget}: best={best_score:.2f}")
        
        return best_params, best_metrics, history


class RandomSearchOptimizer:
    """Simple random search optimizer (baseline)."""
    
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
    
    def optimize(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any], list]:
        """Run random search optimization."""
        
        base_params = init_params_fn()
        x0, structure = params_to_vector(base_params)
        n = len(x0)
        
        best_params = base_params
        best_score = float('-inf')
        best_metrics = None
        history = []
        
        sigma = self.config.param_perturbation_scale
        total_samples = self.config.cma_es_generations * self.config.num_param_samples
        
        if self.config.verbose:
            print(f"  Random Search: n={n} params, {total_samples} samples")
        
        for i in range(total_samples):
            key, noise_key, eval_key = jrandom.split(key, 3)
            
            if i == 0:
                candidate_params = base_params
            else:
                noise = np.array(jrandom.normal(noise_key, shape=(n,))) * sigma
                x_candidate = x0 + noise
                candidate_params = vector_to_params(x_candidate, structure)
            
            try:
                metrics = self.evaluator.evaluate(policy_fn, candidate_params, eval_key)
                score = metrics['avg_return']
            except Exception:
                score = float('-inf')
                metrics = None
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
                best_metrics = metrics
                
                if self.config.verbose and i > 0:
                    print(f"    Sample {i+1}: NEW BEST return={score:.2f}")
            
            if (i + 1) % 50 == 0:
                history.append({'sample': i + 1, 'best': best_score})
        
        return best_params, best_metrics, history


def get_optimizer(name: str, config, evaluator):
    """Factory function to get optimizer by name."""
    optimizers = {
        'cma-es': CMAESOptimizer,
        'bayes': BayesianOptimizer,
        'random': RandomSearchOptimizer,
    }
    
    if name not in optimizers:
        available = ", ".join(optimizers.keys())
        raise ValueError(f"Unknown optimizer: {name}. Available: {available}")
    
    return optimizers[name](config, evaluator)
