"""Validación de que A es cuadrada, simétrica y definida positiva."""
import numpy as np


def validate_spd(A, tol=1e-9):
    out = {"ok": False, "errors": [], "symmetric": False, "positive_definite": False,
           "eigenvalues": None, "minors": None}
    try:
        A = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        out["errors"].append("La matriz contiene valores que no son números.")
        return out
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        out["errors"].append("A debe ser cuadrada (n × n).")
        return out
    if not np.all(np.isfinite(A)):
        out["errors"].append("A tiene celdas vacías o no numéricas. Completa todos los coeficientes.")
        return out
    diff = np.abs(A - A.T)
    if diff.max() > tol:
        i, j = np.unravel_index(np.argmax(diff), diff.shape)
        out["errors"].append(
            f"A no es simétrica: a[{i+1},{j+1}] = {A[i, j]:g} pero a[{j+1},{i+1}] = {A[j, i]:g}. "
            "Haz que ambos coeficientes sean iguales.")
        return out
    out["symmetric"] = True
    n = A.shape[0]
    minors = [float(np.linalg.det(A[:k, :k])) for k in range(1, n + 1)]
    eig = np.linalg.eigvalsh(A)
    out["minors"], out["eigenvalues"] = minors, eig
    if min(minors) > tol and eig.min() > tol:
        out["positive_definite"] = True
        out["ok"] = True
    else:
        out["errors"].append(
            f"A es simétrica pero no es definida positiva (autovalor mínimo = {eig.min():.4g}). "
            "Aumenta los valores de la diagonal o reduce los coeficientes fuera de ella.")
    return out
