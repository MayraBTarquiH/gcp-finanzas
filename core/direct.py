"""Método directo de referencia: factorización de Cholesky A = L Lᵀ (programada a mano)."""
import numpy as np


def cholesky(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                v = A[i, i] - s
                if v <= 0:
                    raise ValueError("A no es definida positiva: Cholesky no se puede completar.")
                L[i, j] = np.sqrt(v)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_cholesky(A, b):
    """Devuelve (x, L, y) con L y = b y Lᵀ x = y."""
    L = cholesky(A)
    y = forward_substitution(L, np.asarray(b, dtype=float))
    x = backward_substitution(L.T, y)
    return x, L, y
