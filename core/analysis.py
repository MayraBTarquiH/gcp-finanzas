"""Orquesta todo el cálculo de un escenario: validación, referencia, GC, GCP y métricas."""
import numpy as np
from .validation import validate_spd
from .direct import solve_cholesky
from .cg import pcg, make_preconditioner, condition_precond
from .metrics import error_table, relative_residual, cond2
from .scenarios import NOMBRES


def analyze(A, b, x0=None, tol=1e-6, max_iter=1000, precond="Jacobi", omega=1.0, names=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    v = validate_spd(A)
    if not v["ok"]:
        return {"valid": False, "validation": v}
    if b.shape != (A.shape[0],) or not np.all(np.isfinite(b)):
        v["errors"].append("b debe tener un valor numérico por cada ecuación.")
        return {"valid": False, "validation": v}
    names = names or NOMBRES
    x_ref, L, y = solve_cholesky(A, b)
    M = make_preconditioner(A, precond, omega)
    gc = pcg(A, b, x0, make_preconditioner(A, "Ninguno"), tol, max_iter)
    gcp = pcg(A, b, x0, M, tol, max_iter)
    return {
        "valid": True, "validation": v, "x_ref": x_ref, "L": L, "y": y,
        "gc": gc, "gcp": gcp, "precond": M,
        "kappa_A": cond2(A), "kappa_M": condition_precond(A, M),
        "errors": error_table(names, x_ref, gcp["x"]),
        "resid_gcp": relative_residual(A, b, gcp["x"]),
        "resid_gc": relative_residual(A, b, gc["x"]),
    }
