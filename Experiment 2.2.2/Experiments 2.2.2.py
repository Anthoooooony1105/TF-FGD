import numpy as np
import matplotlib.pyplot as plt
import csv

# ----------------------------
# Kernels
# ----------------------------
def kernel_linear(X, Y): return X.T @ Y
def kernel_relu(X, Y): return np.maximum(0.0, X.T @ Y)
def kernel_exp(X, Y, sigma=1.0): return np.exp((X.T @ Y) / (sigma**2))
def kernel_softmax_like(X, Y, sigma=1.0): return np.exp((X.T @ Y) / (sigma**2))

KERNELS = {
    "linear": kernel_linear,
    "relu": kernel_relu,
    "exp": kernel_exp,
    "softmax": kernel_softmax_like,
}

# ----------------------------
# Helpers: PSD-fix, norm, spectral radius, adaptive step
# ----------------------------
def make_psd(K: np.ndarray) -> np.ndarray:
    K_sym = (K + K.T) / 2.0
    D, U = np.linalg.eigh(K_sym)
    return U @ np.diag(np.abs(D)) @ U.T

def fro_normalize(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.linalg.norm(K, ord='fro')
    return K / max(s, eps)

def power_iteration_lambda_max(A: np.ndarray, iters: int = 30) -> float:
    n = A.shape[0]
    v = np.random.default_rng(0).normal(size=n)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        v = A @ v
        v /= (np.linalg.norm(v) + 1e-12)
    return float(v @ (A @ v))

# ----------------------------
# GP data generator
# ----------------------------
def sample_episode(d: int, n: int, kernel_name: str, sigma: float = 1.0):
    rng = np.random.default_rng()
    X = rng.normal(size=(d, n+1))
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
    Kfun = KERNELS[kernel_name]
    K = make_psd(Kfun(X, X))
    y = rng.multivariate_normal(mean=np.zeros(n+1), cov=K)
    return X[:, :n], y[:n], X[:, n:n+1], y[n]

# ----------------------------
# Multi-head / layer configs
# ----------------------------
def make_heads(klist): return [KERNELS[name] for name in klist]

def composite_kernel_avg(heads):
    def K(X, Y):
        S = None
        for h in heads:
            Ki = h(X, Y)
            S = Ki if S is None else S + Ki
        return S / len(heads)
    return K

def config_A():
    heads = make_heads(["linear","relu","exp","softmax"])
    Kc = composite_kernel_avg(heads)
    return [Kc]*4

def config_B():
    order = ["linear","relu","exp","softmax"]
    return [composite_kernel_avg(make_heads([name]*4)) for name in order]

def config_C():
    order = ["linear","relu","exp","softmax"]
    return [KERNELS[name] for name in order]

def config_D():
    heads = make_heads(["linear","relu","exp","softmax"])
    return [composite_kernel_avg(heads)]

CONFIG_BUILDERS = {
    "A_layerMixHeads": config_A,
    "B_layerWiseDifferent": config_B,
    "C_singleHeadPerLayer": config_C,
    "D_singleLayerMultiHead": config_D,
}

# ----------------------------
# Stable FGD predictor (adaptive Landweber + ridge + normalization)
# ----------------------------
def fgd_predict_stable(
    X_ctx, y_ctx, x_q, layer_kernels,
    ridge: float = 1e-2,
    safety: float = 0.8,
    normalize: bool = True
):
    n = X_ctx.shape[1]
    a = np.zeros(n, dtype=float)

    Ks_ctx = []
    Ks_q = []
    for Kfun in layer_kernels:
        Kc = Kfun(X_ctx, X_ctx)
        Kq = Kfun(x_q, X_ctx)      # shape (1, n)
        if normalize:
            Kc = fro_normalize(Kc)
            Kq = Kq / (np.linalg.norm(Kq, ord='fro') + 1e-12)
        Ks_ctx.append(Kc)
        Ks_q.append(Kq)

    for Kc in Ks_ctx:
        K_reg = Kc + ridge * np.eye(n)
        lam_max = power_iteration_lambda_max(K_reg)
        delta = safety / max(lam_max, 1e-8)
        res = y_ctx - (K_reg @ a)
        a = a + delta * res
    K_mix_q = sum(Ks_q) / len(Ks_q)
    return float(K_mix_q @ a)

# ----------------------------
# Run evaluation
# ----------------------------
def run_eval(
    d=8, ns=(32,64,128,256), episodes=40, ridge=1e-2, safety=0.8, normalize=True, sigma=1.0
):
    results = []
    for gen in ["linear","relu","exp","softmax"]:
        for n in ns:
            mse_sums = {cfg:0.0 for cfg in CONFIG_BUILDERS}
            for _ in range(episodes):
                Xc, yc, xq, yq = sample_episode(d,n,gen,sigma)
                for cfg,builder in CONFIG_BUILDERS.items():
                    y_hat = fgd_predict_stable(
                        Xc, yc, xq, builder(),
                        ridge=ridge, safety=safety, normalize=normalize
                    )
                    mse_sums[cfg] += (y_hat-yq)**2
            for cfg in CONFIG_BUILDERS:
                results.append((gen,cfg,n,mse_sums[cfg]/episodes))
    return results

# ----------------------------
# Plot & Save
# ----------------------------
def plot_and_save(results):
    cfgs = list(CONFIG_BUILDERS.keys())
    ns_sorted = sorted({n for _,_,n,_ in results})
    for gen in ["linear","relu","exp","softmax"]:
        M = {cfg:[] for cfg in cfgs}
        for n in ns_sorted:
            for cfg in cfgs:
                val = next(mse for g,c,n0,mse in results if g==gen and c==cfg and n0==n)
                M[cfg].append(val)
        plt.figure()
        for cfg in cfgs:
            plt.plot(ns_sorted, M[cfg], marker='o', label=cfg)
        plt.xlabel("Context length n")
        plt.ylabel("MSE")
        plt.title(f"Stable FGD (adaptive) - gen kernel: {gen}")
        plt.legend()
        fname = f"fgd_{gen}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {fname}")
        plt.close()

# ----------------------------
# Run everything
# ----------------------------
if __name__=="__main__":
    results = run_eval(d=8, ns=(16,32,64,128,256,512,1024), episodes=40, ridge=1e-2, safety=0.8, normalize=True)
    plot_and_save(results)
    target_n = max({n for _,_,n,_ in results})
    with open("../Experiments 2.2.2/results_n_max.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["gen_kernel"]+list(CONFIG_BUILDERS.keys())
        writer.writerow(header)
        for gen in ["linear","relu","exp","softmax"]:
            row=[gen]
            for cfg in CONFIG_BUILDERS:
                val = next(mse for g,c,n,mse in results if g==gen and c==cfg and n==target_n)
                row.append(f"{val:.6f}")
            writer.writerow(row)
    print("Saved results_n_max.csv")
