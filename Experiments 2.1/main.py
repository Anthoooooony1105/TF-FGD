import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve
import pandas as pd

# ---------- synthetic data & kernel ----------
rng = np.random.default_rng(42)
n, d = 80, 5
X = rng.normal(size=(n, d))
x_query = rng.normal(size=(d,))

def rbf_kernel(x, y, sigma=1.2):
    return np.exp(-norm(x - y) ** 2 / (2.0 * sigma ** 2))

sigma = 1.2
K = np.empty((n, n))
for i in range(n):
    for j in range(n):
        K[i, j] = rbf_kernel(X[i], X[j], sigma=sigma)

ridge = 1e-8
K = K + ridge * np.eye(n)

nu = np.array([rbf_kernel(X[i], x_query, sigma=sigma) for i in range(n)])

Y = rng.normal(size=(n,))

# ---------- Bayes predictor ----------
# y* = nu^T K^{-1} Y
y_star = nu @ solve(K, Y)

eigvals, Q = np.linalg.eigh(K)
lam_min, lam_max = eigvals[0], eigvals[-1]
kappa = lam_max / lam_min

delta_opt  = 2.0 / (lam_max + lam_min)
delta_safe = 1.0 / lam_max

def rho_of_delta(delta):
    return np.max(np.abs(1.0 - delta * eigvals))

rho_opt  = rho_of_delta(delta_opt)
rho_safe = rho_of_delta(delta_safe)

C = (norm(nu) * norm(Y)) / lam_min

L_max = 200
Ls = np.arange(1, L_max + 1)

Kinvy   = solve(K, Y)
Qt_Kinvy = Q.T @ Kinvy
Qt_nu    = Q.T @ nu

def errors_and_bounds(delta, rho):
    one_minus = 1.0 - delta * eigvals
    errs = np.empty_like(Ls, dtype=float)
    bnds = np.empty_like(Ls, dtype=float)
    for idx, L in enumerate(Ls):
        powvec = one_minus ** L
        vec = Q @ (powvec * Qt_Kinvy)
        err = abs(nu @ vec)
        bound = C * (rho ** L)
        errs[idx] = err
        bnds[idx] = bound
    return errs, bnds

errs_opt,  bnds_opt  = errors_and_bounds(delta_opt,  rho_opt)
errs_safe, bnds_safe = errors_and_bounds(delta_safe, rho_safe)

print("lambda_min =", lam_min)
print("lambda_max =", lam_max)
print("kappa      =", kappa)
print("delta_opt  =", delta_opt,  "  rho_opt  =", rho_opt)
print("delta_safe =", delta_safe, "  rho_safe =", rho_safe)
print("Inequality holds (opt): ",  np.all(errs_opt  <= bnds_opt  + 1e-12))
print("Inequality holds (safe): ", np.all(errs_safe <= bnds_safe + 1e-12))

# ---------- plot ----------
plt.figure(figsize=(7,5))
plt.semilogy(Ls, errs_opt,  label="Error (optimal step)")
plt.semilogy(Ls, bnds_opt,  linestyle="--", label="Bound (optimal step)")
plt.semilogy(Ls, errs_safe, label="Error (safe step)")
plt.semilogy(Ls, bnds_safe, linestyle="--", label="Bound (safe step)")
plt.xlabel("Depth L")
plt.ylabel("Absolute error |y* - y_L|")
plt.title("Finite-depth error vs. non-asymptotic bound")
plt.legend()
plt.tight_layout()
plt.show()


