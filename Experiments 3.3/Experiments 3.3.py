import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from typing import Callable, Tuple

rng = np.random.default_rng(1234)

def sample_unit_sphere(d: int, m: int) -> np.ndarray:
    X = rng.normal(size=(d, m))
    norms = LA.norm(X, axis=0, keepdims=True) + 1e-12
    return X / norms

def K_linear(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return X.T @ Z

def K_relu(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, X.T @ Z)

def K_exp(X: np.ndarray, Z: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.exp((X.T @ Z) / (sigma**2))

def K_plus(K: np.ndarray) -> np.ndarray:
    # K_plus = U |D| U^T
    D, U = LA.eigh((K + K.T) / 2.0)  # ensure symmetry
    return (U * np.abs(D)) @ U.T

def sample_K_GP(X: np.ndarray, Kfun: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    K = Kfun(X, X)
    Kp = K_plus(K)
    D, U = LA.eigh((Kp + Kp.T) / 2.0)
    D = np.clip(D, 0, None)
    Y = U @ (np.sqrt(D) * rng.normal(size=U.shape[0]))
    return Y  # shape (n+1,)

def functional_gd_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray,
    Kfun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    L: int, step: float, variant: str
) -> np.ndarray:
    n = X_train.shape[1]
    m = X_eval.shape[1]
    if variant == "softmax":
        K_tt = K_exp(X_train, X_train)
        K_te = K_exp(X_train, X_eval)
        tau_eval = 1.0 / (np.sum(K_exp(X_eval, X_train), axis=1) + 1e-12)  # (m,)
        tau_train = 1.0 / (np.sum(K_tt, axis=1) + 1e-12)  # (n,)
    else:
        K_tt = Kfun(X_train, X_train)
        K_te = Kfun(X_train, X_eval)
        tau_eval = None
        tau_train = None

    f_train = np.zeros(n)
    f_eval  = np.zeros(m)

    for _ in range(L):
        resid = y_train - f_train  # (n,)
        if variant == "softmax":
            f_eval  = f_eval  + step * (tau_eval  * (K_te.T @ resid))
            f_train = f_train + step * (tau_train * (K_tt @ resid))
        else:
            f_eval  = f_eval  + step * (K_te.T @ resid)
            f_train = f_train + step * (K_tt @ resid)

    return f_eval  # (m,)

def bayes_estimator(
    X: np.ndarray, Kfun: Callable[[np.ndarray, np.ndarray], np.ndarray], ridge: float = 1e-6
) -> float:

    K = Kfun(X, X)
    Kp = K_plus(K)
    n1 = X.shape[1]
    n = n1 - 1
    K_hat = Kp[:n, :n]
    nu = Kp[:n, n]
    def predict(Y: np.ndarray) -> float:
        Y_hat = Y[:n]
        A = K_hat + ridge * np.eye(n)
        alpha = LA.solve(A, Y_hat)
        return float(nu @ alpha)
    return predict

# Experiment parameters
d = 5
n_trials = 60
sigma = 1.0
step = 0.1
ridge = 1e-6

# Define kernel sets
kernels = {
    "linear": K_linear,
    "relu": K_relu,
    "exp": lambda X, Z: K_exp(X, Z, sigma=sigma),
}

variants = ["linear", "relu", "exp", "softmax"]

def run_curve_vs_n(L_layers=3, ns=(2,4,6,8,10,12,14)):
    results = {Kname: {v: [] for v in variants} | {"bayes": []} for Kname in kernels.keys()}
    for n in ns:
        for Kname, Kfun in kernels.items():
            mse_accum = {v: 0.0 for v in variants}
            bayes_accum = 0.0
            for _ in range(n_trials):
                X = sample_unit_sphere(d, n+1)
                Y = sample_K_GP(X, Kfun)
                X_train, X_q = X[:, :n], X[:, n:]
                y_train, y_q = Y[:n], Y[n:][0]
                for v in variants:
                    if v == "linear":
                        f_q = functional_gd_predict(X_train, y_train, X_q, K_linear, L_layers, step, "linear")[0]
                    elif v == "relu":
                        f_q = functional_gd_predict(X_train, y_train, X_q, K_relu, L_layers, step, "relu")[0]
                    elif v == "exp":
                        f_q = functional_gd_predict(X_train, y_train, X_q, kernels["exp"], L_layers, step, "exp")[0]
                    elif v == "softmax":
                        f_q = functional_gd_predict(X_train, y_train, X_q, kernels["exp"], L_layers, step, "softmax")[0]
                    mse_accum[v] += (f_q - y_q)**2
                f_bayes_fn = bayes_estimator(X, Kfun, ridge=ridge)
                f_b = f_bayes_fn(Y)
                bayes_accum += (f_b - y_q)**2
            for v in variants:
                results[Kname][v].append(mse_accum[v] / n_trials)
            results[Kname]["bayes"].append(bayes_accum / n_trials)
    return ns, results

def run_curve_vs_layers(Kname="relu", n_ctx=14, Ls=range(1,8)):
    Kfun = kernels[Kname]
    results = {v: [] for v in variants if v != "linear" or Kname != "linear"}  # keep all, match paper figs
    results["bayes"] = []
    for L in Ls:
        mse_accum = {v: 0.0 for v in results if v != "bayes"}
        bayes_accum = 0.0
        for _ in range(n_trials):
            X = sample_unit_sphere(d, n_ctx+1)
            Y = sample_K_GP(X, Kfun)
            X_train, X_q = X[:, :n_ctx], X[:, n_ctx:]
            y_train, y_q = Y[:n_ctx], Y[n_ctx:][0]
            for v in mse_accum.keys():
                if v == "linear":
                    f_q = functional_gd_predict(X_train, y_train, X_q, K_linear, L, step, "linear")[0]
                elif v == "relu":
                    f_q = functional_gd_predict(X_train, y_train, X_q, K_relu, L, step, "relu")[0]
                elif v == "exp":
                    f_q = functional_gd_predict(X_train, y_train, X_q, kernels["exp"], L, step, "exp")[0]
                elif v == "softmax":
                    f_q = functional_gd_predict(X_train, y_train, X_q, kernels["exp"], L, step, "softmax")[0]
                mse_accum[v] += (f_q - y_q)**2
            f_bayes_fn = bayes_estimator(X, Kfun, ridge=ridge)
            f_b = f_bayes_fn(Y)
            bayes_accum += (f_b - y_q)**2
        for v in mse_accum.keys():
            results[v].append(mse_accum[v] / n_trials)
        results["bayes"].append(bayes_accum / n_trials)
    return list(Ls), results

# ---- Run experiments & plot ----
ns, res_vs_n = run_curve_vs_n(L_layers=3, ns=(2,4,6,8,10,12,14))

# Plot Fig 1-style: three subplots for Klinear, Krelu, Kexp
fig1, axes = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)
for ax, (Kname, title) in zip(axes, [("linear","Klinear"), ("relu","Krelu"), ("exp","Kexp")]):
    ax.plot(ns, np.log(res_vs_n[Kname]["relu"]), marker='o', label="relu")
    ax.plot(ns, np.log(res_vs_n[Kname]["linear"]), marker='o', label="linear")
    ax.plot(ns, np.log(res_vs_n[Kname]["exp"]), marker='o', label="exp")
    ax.plot(ns, np.log(res_vs_n[Kname]["softmax"]), marker='o', label="softmax")
    ax.plot(ns, np.log(res_vs_n[Kname]["bayes"]), marker='o', label="bayes")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("log(Loss)")
    ax.set_title(title)
    ax.legend()
fig1.suptitle("log(test ICL loss) vs Context Length (L=3)")
plt.show()

# Fig 2-style: 4 subplots: (Krelu, n=14), (Kexp, n=14), (Krelu, n=6), (Kexp, n=6)
Ls = list(range(1,8))
Ls1, r_relu_14 = run_curve_vs_layers(Kname="relu", n_ctx=14, Ls=Ls)
Ls2, r_exp_14  = run_curve_vs_layers(Kname="exp",  n_ctx=14, Ls=Ls)
Ls3, r_relu_6  = run_curve_vs_layers(Kname="relu", n_ctx=6,  Ls=Ls)
Ls4, r_exp_6   = run_curve_vs_layers(Kname="exp",  n_ctx=6,  Ls=Ls)

fig2, axes2 = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
# (a) Krelu, n=14
ax = axes2[0,0]
for key in ["relu","exp","softmax","bayes"]:
    ax.plot(Ls1, np.log(r_relu_14[key]), marker='o', label=key)
ax.set_xlabel("Layer Depth")
ax.set_ylabel("log(Loss)")
ax.set_title("Krelu, n=14")
ax.legend()

# (b) Kexp, n=14
ax = axes2[0,1]
for key in ["relu","exp","softmax","bayes"]:
    ax.plot(Ls2, np.log(r_exp_14[key]), marker='o', label=key)
ax.set_xlabel("Layer Depth")
ax.set_ylabel("log(Loss)")
ax.set_title("Kexp, n=14")
ax.legend()

# (c) Krelu, n=6
ax = axes2[1,0]
for key in ["relu","exp","softmax","bayes"]:
    ax.plot(Ls3, np.log(r_relu_6[key]), marker='o', label=key)
ax.set_xlabel("Layer Depth")
ax.set_ylabel("log(Loss)")
ax.set_title("Krelu, n=6")
ax.legend()

# (d) Kexp, n=6
ax = axes2[1,1]
for key in ["relu","exp","softmax","bayes"]:
    ax.plot(Ls4, np.log(r_exp_6[key]), marker='o', label=key)
ax.set_xlabel("Layer Depth")
ax.set_ylabel("log(Loss)")
ax.set_title("Kexp, n=6")
ax.legend()

fig2.suptitle("log(test ICL loss) vs Layer Depth")
plt.show()

