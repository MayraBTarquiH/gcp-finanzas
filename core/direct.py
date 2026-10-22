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


def cholesky_steps(A, L):
    """Fórmulas (LaTeX) con números de cada l_ij, en el mismo orden en que se calculan."""
    A = np.asarray(A, dtype=float)
    out = []
    for i in range(A.shape[0]):
        for j in range(i + 1):
            s = float(sum(L[i, k] * L[j, k] for k in range(j)))
            ii, jj = i + 1, j + 1
            if i == j:
                out.append(rf"l_{{{ii}{ii}}} = \sqrt{{a_{{{ii}{ii}}} - \sum_{{k<{ii}}} l_{{{ii}k}}^2}}"
                           rf" = \sqrt{{{A[i, i]:g} - ({s:.4f})}} = {L[i, i]:.4f}")
            else:
                out.append(rf"l_{{{ii}{jj}}} = \frac{{a_{{{ii}{jj}}} - \sum_{{k<{jj}}} l_{{{ii}k}}\,l_{{{jj}k}}}}{{l_{{{jj}{jj}}}}}"
                           rf" = \frac{{{A[i, j]:g} - ({s:.4f})}}{{{L[j, j]:.4f}}} = {L[i, j]:.4f}")
    return out


def forward_steps(L, b, y):
    out = []
    for i in range(len(b)):
        s = float(L[i, :i] @ y[:i])
        ii = i + 1
        out.append(rf"y_{{{ii}}} = \frac{{b_{{{ii}}} - \sum_{{k<{ii}}} l_{{{ii}k}}\,y_k}}{{l_{{{ii}{ii}}}}}"
                   rf" = \frac{{{b[i]:g} - ({s:.4f})}}{{{L[i, i]:.4f}}} = {y[i]:.4f}")
    return out


def backward_steps(L, y, x):
    out = []
    for i in range(len(y) - 1, -1, -1):
        s = float(L.T[i, i + 1:] @ x[i + 1:])
        ii = i + 1
        out.append(rf"x_{{{ii}}} = \frac{{y_{{{ii}}} - \sum_{{k>{ii}}} l_{{k{ii}}}\,x_k}}{{l_{{{ii}{ii}}}}}"
                   rf" = \frac{{{y[i]:.4f} - ({s:.4f})}}{{{L[i, i]:.4f}}} = {x[i]:.4f}")
    return out
