# Four configurations:
# 1) Config A (LayerMixHeads): 4 layers; each layer has 4 heads with *different* kernels; per-layer kernel is the average of the four.
# 2) Config B (LayerWiseDifferent): 4 layers; each layer has 4 heads, but all 4 heads use the *same* kernel; across layers kernels differ (linear→relu→exp→softmax-like).
# 3) Config C (SingleHeadPerLayer): 4 layers; 1 head per layer; across layers kernels differ (linear→relu→exp→softmax-like).
# 4) Config D (SingleLayerMultiHead): 1 layer; 4 heads in a single layer, each a different kernel (average them).


import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple
import math
import random

rng = np.random.default_rng(42)

# ----------------------------
# Kernel definitions
# ----------------------------

def kernel_linear(X, Y):

    return X.T @ Y

def kernel_relu(X, Y):

    return np.maximum(0.0, X.T @ Y)

def kernel_exp(X, Y, sigma=1.0):

    return np.exp((X.T @ Y) / (sigma**2))

def kernel_softmax_like(X, Y, sigma=1.0):

    return np.exp((X.T @ Y) / (sigma**2))

KERNELS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "linear": kernel_linear,
    "relu": kernel_relu,
    "exp": kernel_exp,
    "softmax": kernel_softmax_like,
}

# ----------------------------
# GP data generator (K-GP)
# ----------------------------

def sample_episode(d: int, n: int, kernel_name: str, sigma: float = 1.0):

    X = rng.normal(size=(d, n+1))
    X = X / np.linalg.norm(X, axis=0, keepdims=True)

    Kfun = KERNELS[kernel_name]
    K = Kfun(X, X)  # (n+1) x (n+1)

    D, U = np.linalg.eigh((K + K.T) / 2.0)
    K_plus = U @ np.diag(np.abs(D)) @ U.T

    y = rng.multivariate_normal(mean=np.zeros(n+1), cov=K_plus)
    y = y.reshape(-1)

    X_ctx = X[:, :n]       # d x n
    y_ctx = y[:n]          # n
    x_q = X[:, n:n+1]      # d x 1
    y_q = y[n]             # scalar
    return X_ctx, y_ctx, x_q, y_q

# ----------------------------
# Functional Gradient Descent Predictor
# ----------------------------

def fgd_predict(X_ctx: np.ndarray, y_ctx: np.ndarray, x_q: np.ndarray,
                layer_kernels: List[Callable[[np.ndarray, np.ndarray], np.ndarray]],
                lr: float = 0.5):

    n = X_ctx.shape[1]
    f_vals = np.zeros(n)
    for Kfun in layer_kernels:
        K_ctx_ctx = Kfun(X_ctx, X_ctx)
        res = y_ctx - f_vals
        f_vals = f_vals + lr * (K_ctx_ctx @ res)

    a = np.zeros(n)
    for Kfun in layer_kernels:
        K_ctx_ctx = Kfun(X_ctx, X_ctx)  # n x n
        res = y_ctx - (K_ctx_ctx @ a)   # since f_vals = K a
        a = a + lr * res

    if len(layer_kernels) == 0:
        return 0.0
    K_mats = [Kfun(x_q, X_ctx) for Kfun in layer_kernels]  # 1 x n each
    K_mix = sum(K_mats) / len(K_mats)

    y_pred = float(K_mix @ a)
    return y_pred

# ----------------------------
# Build per-configuration kernel schedules
# ----------------------------

def make_heads(klist: List[str]) -> List[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    return [KERNELS[name] for name in klist]

def composite_kernel_avg(heads: List[Callable]) -> Callable:
    def K(X, Y):
        S = None
        for h in heads:
            Ki = h(X, Y)
            S = Ki if S is None else S + Ki
        return S / len(heads)
    return K

def config_A_layer_kernels() -> List[Callable]:
    # 4 layers; each layer has 4 different heads mixed
    heads = make_heads(["linear", "relu", "exp", "softmax"])
    K_comp = composite_kernel_avg(heads)
    return [K_comp, K_comp, K_comp, K_comp]

def config_B_layer_kernels() -> List[Callable]:
    # 4 layers; each layer's 4 heads are the same activation; layers differ
    order = ["linear", "relu", "exp", "softmax"]
    return [composite_kernel_avg(make_heads([name]*4)) for name in order]

def config_C_layer_kernels() -> List[Callable]:
    # 4 layers; 1 head per layer with different activations
    order = ["linear", "relu", "exp", "softmax"]
    return [KERNELS[name] for name in order]

def config_D_layer_kernels() -> List[Callable]:
    # 1 layer; 4 heads, all different
    heads = make_heads(["linear", "relu", "exp", "softmax"])
    return [composite_kernel_avg(heads)]  # single layer

CONFIG_BUILDERS = {
    "A_layerMixHeads": config_A_layer_kernels,
    "B_layerWiseDifferent": config_B_layer_kernels,
    "C_singleHeadPerLayer": config_C_layer_kernels,
    "D_singleLayerMultiHead": config_D_layer_kernels,
}

# ----------------------------
# Evaluation loop
# ----------------------------

def run_eval(d=8, ns=(4, 8, 12, 16), episodes=200, lr=0.5, sigma=1.0, seed=123):
    rng = np.random.default_rng(seed)
    results = []  # list of (gen_kernel, config_name, n, mse)
    gen_kernels = ["linear", "relu", "exp", "softmax"]
    for gen in gen_kernels:
        for n in ns:
            # accumulate MSE per config
            mse_sums = {cfg: 0.0 for cfg in CONFIG_BUILDERS.keys()}
            for _ in range(episodes):
                X_ctx, y_ctx, x_q, y_q = sample_episode(d, n, gen, sigma=sigma)
                for cfg_name, builder in CONFIG_BUILDERS.items():
                    layer_kerns = builder()
                    y_hat = fgd_predict(X_ctx, y_ctx, x_q, layer_kerns, lr=lr, normalize_softmax=False)
                    mse_sums[cfg_name] += (y_hat - y_q)**2
            for cfg_name in CONFIG_BUILDERS.keys():
                mse = mse_sums[cfg_name] / episodes
                results.append((gen, cfg_name, n, mse))
    return results

results = run_eval(d=8, ns=(4, 8, 12, 16), episodes=150, lr=0.4, sigma=1.0, seed=2025)

# ----------------------------
# Plotting
# ----------------------------

def plot_results(results):
    gen_kernels = ["linear", "relu", "exp", "softmax"]
    cfgs = list(CONFIG_BUILDERS.keys())
    ns_sorted = sorted(list({n for (_, _, n, _) in results}))
    n2idx = {n:i for i,n in enumerate(ns_sorted)}
    for gen in gen_kernels:
        # matrix: len(ns) x len(cfgs)
        M = np.zeros((len(ns_sorted), len(cfgs)))
        for (g, cfg, n, mse) in results:
            if g == gen:
                M[n2idx[n], cfgs.index(cfg)] = mse
        plt.figure()
        for j, cfg in enumerate(cfgs):
            plt.plot(ns_sorted, M[:, j], marker='o', label=cfg)
        plt.xlabel("Context length n")
        plt.ylabel("MSE")
        plt.title(f"Test MSE vs. n (Ground-truth kernel: {gen})")
        plt.legend()
        plt.show()

plot_results(results)

# Also print a compact table for quick comparison at n=12
def summarize_table(results, target_n=12):
    gen_kernels = ["linear", "relu", "exp", "softmax"]
    cfgs = list(CONFIG_BUILDERS.keys())
    lines = []
    header = ["gen_kernel"] + cfgs
    lines.append(header)
    for gen in gen_kernels:
        row = [gen]
        for cfg in cfgs:
            val = next(mse for (g, c, n, mse) in results if g==gen and c==cfg and n==target_n)
            row.append(f"{val:.4f}")
        lines.append(row)
    return lines

table = summarize_table(results, target_n=12)
for row in table:
    print("\t".join(row))