import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from typing import Callable

rng = np.random.default_rng(2025)

def sample_unit_sphere(d: int, m: int) -> np.ndarray:
    X = rng.normal(size=(d, m))
    X /= (LA.norm(X, axis=0, keepdims=True) + 1e-12)
    return X

def K_linear(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return X.T @ Z

def K_exp(X: np.ndarray, Z: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.exp(scale * (X.T @ Z))

def K_plus(K: np.ndarray) -> np.ndarray:
    S = (K + K.T) / 2.0
    D, U = LA.eigh(S)
    return (U * np.abs(D)) @ U.T

def sample_K_GP(X: np.ndarray, Kfun: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    K = Kfun(X, X)
    Kp = K_plus(K)
    D, U = LA.eigh((Kp + Kp.T) / 2.0)
    D = np.clip(D, 0, None)
    return U @ (np.sqrt(D) * rng.normal(size=U.shape[0]))

def functional_gd_single(Xtr, ytr, Xev, Kfun, L, step):
    n = Xtr.shape[1]
    m = Xev.shape[1]
    K_tt = Kfun(Xtr, Xtr)
    K_te = Kfun(Xtr, Xev)
    f_tr = np.zeros(n)
    f_ev = np.zeros(m)
    for _ in range(L):
        resid = ytr - f_tr
        f_ev  = f_ev  + step * (K_te.T @ resid)
        f_tr  = f_tr  + step * (K_tt @ resid)
    return f_ev

def functional_gd_twohead(Xtr, ytr, Xev, K1, K2, beta1, beta2, L, step):
    n = Xtr.shape[1]
    m = Xev.shape[1]
    K1_tt = K1(Xtr, Xtr); K1_te = K1(Xtr, Xev)
    K2_tt = K2(Xtr, Xtr); K2_te = K2(Xtr, Xev)
    f_tr = np.zeros(n)
    f_ev = np.zeros(m)
    for _ in range(L):
        resid = ytr - f_tr
        f_ev  = f_ev  + step * (beta1 * (K1_te.T @ resid) + beta2 * (K2_te.T @ resid))
        f_tr  = f_tr  + step * (beta1 * (K1_tt @ resid) + beta2 * (K2_tt @ resid))
    return f_ev

# G1 keeps first two dims, G2 keeps last three dims
def apply_G1(X):
    G = np.zeros_like(X)
    G[0,:] = X[0,:]; G[1,:] = X[1,:]
    return G
def apply_G2(X):
    G = np.zeros_like(X)
    G[2:,:] = X[2:,:]
    return G
def K_linear_G1(X,Z): return apply_G1(X).T @ apply_G1(Z)
def K_exp_G2_scaled(X,Z): return np.exp(0.5 * (apply_G2(X).T @ apply_G2(Z)))
def K_composite(X,Z): return 0.5 * K_linear_G1(X,Z) + 0.5 * K_exp_G2_scaled(X,Z)

def bayes_mse(X, Y, Kfun, ridge=1e-6):
    n = X.shape[1] - 1
    K = Kfun(X,X); Kp = K_plus(K)
    K_hat = Kp[:n,:n]; nu = Kp[:n,n]
    alpha = LA.solve(K_hat + ridge*np.eye(n), Y[:n])
    pred = float(nu @ alpha)
    return (pred - Y[n])**2

# Parameters
d = 5; n_ctx = 14; Ls = list(range(1,8)); n_trials = 60; step = 0.1

def run_dataset(kind: str):
    curves = {"linear":[], "exp":[], "twohead":[], "bayes":[]}
    for L in Ls:
        acc = {k:0.0 for k in curves}
        for _ in range(n_trials):
            X = sample_unit_sphere(d, n_ctx+1)
            if kind == "linear":
                Kgen = K_linear; b1,b2 = 1.0,0.0; K1 = K_linear; K2 = lambda A,B: K_exp(A,B,1.0)
            elif kind == "exp":
                Kgen = lambda A,B: K_exp(A,B,1.0); b1,b2 = 0.0,1.0; K1 = K_linear; K2 = lambda A,B: K_exp(A,B,1.0)
            else:
                Kgen = K_composite; b1,b2 = 0.5,0.5; K1 = K_linear_G1; K2 = K_exp_G2_scaled
            Y = sample_K_GP(X, Kgen)
            Xtr, Xq = X[:,:n_ctx], X[:,n_ctx:]
            ytr, yq = Y[:n_ctx], Y[n_ctx]
            pred_lin = functional_gd_single(Xtr, ytr, Xq, K_linear, L, step)[0]
            pred_exp = functional_gd_single(Xtr, ytr, Xq, lambda A,B: K_exp(A,B,1.0), L, step)[0]
            pred_two = functional_gd_twohead(Xtr, ytr, Xq, K1, K2, b1, b2, L, step)[0]
            acc["linear"] += (pred_lin - yq)**2
            acc["exp"]    += (pred_exp - yq)**2
            acc["twohead"]+= (pred_two - yq)**2
            acc["bayes"]  += bayes_mse(X, Y, Kgen)
        for k in curves:
            curves[k].append(acc[k]/n_trials)
    return curves

curves_linear    = run_dataset("linear")
curves_exp       = run_dataset("exp")
curves_composite = run_dataset("composite")

# Plot
fig, axes = plt.subplots(1,3, figsize=(15,4), constrained_layout=True)
titles = ["Klinear", "Kexp", "K⋄ (composite)"]
datasets = [curves_linear, curves_exp, curves_composite]
for ax, title, curves in zip(axes, titles, datasets):
    ax.plot(Ls, np.log(curves["linear"]), marker='o', label="linear")
    ax.plot(Ls, np.log(curves["exp"]), marker='o', label="exp")
    ax.plot(Ls, np.log(curves["twohead"]), marker='o', label="2-head (linear+exp)")
    ax.plot(Ls, np.log(curves["bayes"]), marker='o', label="bayes")
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("log(Loss)")
    ax.set_title(title)
    ax.legend()

fig.suptitle("Section 3.4 reproduction: log(test ICL loss) vs Layer Depth (n=14, d=5)")
plt.show()
