"""Gradientes Conjugados (GC) y Gradientes Conjugados Precondicionado (GCP).

Implementa el Algoritmo 2 de Suñagua (2020), "algoritmo Mz":
en cada iteración se resuelve M z_k = r_k y se usa z_k en lugar de r_k.
Con M = I se obtiene el GC simple (Algoritmo 1).
"""
import numpy as np
from scipy.linalg import solve_triangular, eigh

PRECONDICIONADORES = ["Jacobi", "SSOR", "Ninguno"]


class Preconditioner:
    def __init__(self, name, apply, matrix):
        self.name, self.apply, self.matrix = name, apply, matrix


def make_preconditioner(A, kind="Jacobi", omega=1.0):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if kind == "Ninguno":
        return Preconditioner("Ninguno (GC simple)", lambda r: r.copy(), np.eye(n))
    d = np.diag(A)
    if kind == "Jacobi":
        return Preconditioner("Jacobi", lambda r: r / d, np.diag(d))
    if kind == "SSOR":
        D = np.diag(d)
        T = D + omega * np.tril(A, -1)          # D + ωL
        c = omega * (2 - omega)
        M = T @ np.diag(1 / d) @ T.T / c

        def apply(r):
            u = solve_triangular(T, r, lower=True)
            w = solve_triangular(T.T, d * u, lower=False)
            return c * w
        return Preconditioner(f"SSOR (ω = {omega:g})", apply, M)
    raise ValueError(f"Precondicionador desconocido: {kind}")


def pcg(A, b, x0=None, precond=None, tol=1e-6, max_iter=1000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    if precond is None:
        precond = make_preconditioner(A, "Ninguno")
    nb = np.linalg.norm(b) or 1.0
    r = b - A @ x
    hist = [np.linalg.norm(r) / nb]
    if hist[0] < tol:
        return {"x": x, "iterations": 0, "converged": True, "history": hist, "residual": hist[0]}
    z = precond.apply(r)
    p = z.copy()
    rz = r @ z
    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rel = np.linalg.norm(r) / nb
        hist.append(rel)
        if rel < tol:
            converged = True
            break
        z = precond.apply(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return {"x": x, "iterations": k, "converged": converged, "history": hist, "residual": hist[-1]}


def condition_precond(A, precond):
    """κ(M⁻¹A) a partir de los autovalores del problema generalizado A v = λ M v."""
    w = eigh(np.asarray(A, float), precond.matrix, eigvals_only=True)
    return float(w.max() / w.min())
