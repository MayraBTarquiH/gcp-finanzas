"""Orquesta todo el cálculo de un escenario: validación, referencia, GC, GCP y métricas."""
import numpy as np
from .validation import validate_spd
from .direct import solve_cholesky
from .cg import pcg, make_preconditioner, condition_precond
from .metrics import error_table, relative_residual, cond2
import pandas as pd
from .scenarios import NOMBRES, SIMBOLOS


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


def compare_methods(A, b, x0=None, tol=1e-6, max_iter=1000, precond="Jacobi", omega=1.0):
    """Tabla con los resultados de los tres métodos: Cholesky, GC y GCP."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if not validate_spd(A)["ok"]:
        return None
    x_ref, _, _ = solve_cholesky(A, b)
    kA = cond2(A)

    def fila(nombre, x, it, conv, kappa, referencia=False):
        ea = np.abs(x_ref - x)
        with np.errstate(divide="ignore", invalid="ignore"):
            ep = np.where(x_ref != 0, ea / np.abs(x_ref) * 100, np.nan)
        d = {"Método": nombre}
        d.update({SIMBOLOS[i]: float(x[i]) for i in range(len(x))})
        d.update({"Iteraciones": it, "Residuo relativo": relative_residual(A, b, x),
                  "Error absoluto máx.": np.nan if referencia else float(ea.max()),
                  "Error % máx.": np.nan if referencia else float(np.nanmax(ep)),
                  "Número de condición": kappa, "¿Convergió?": conv})
        return d

    filas = [fila("Cholesky (referencia)", x_ref, np.nan, "Sí (método directo)", kA, referencia=True)]
    gc = pcg(A, b, x0, make_preconditioner(A, "Ninguno"), tol, max_iter)
    filas.append(fila("Gradientes Conjugados (GC)", gc["x"], gc["iterations"], "Sí" if gc["converged"] else "No", kA))
    M = make_preconditioner(A, precond, omega)
    r = pcg(A, b, x0, M, tol, max_iter)
    filas.append(fila(f"GC Precondicionado (GCP, {M.name})", r["x"], r["iterations"],
                      "Sí" if r["converged"] else "No", condition_precond(A, M)))
    return pd.DataFrame(filas)
