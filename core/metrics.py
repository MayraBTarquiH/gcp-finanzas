"""Errores por componente, residuo relativo y números de condición."""
import numpy as np
import pandas as pd


def error_table(names, x_ref, x_app):
    x_ref, x_app = np.asarray(x_ref, float), np.asarray(x_app, float)
    ea = np.abs(x_ref - x_app)
    er = np.where(x_ref != 0, ea / np.where(x_ref != 0, np.abs(x_ref), 1), np.nan)
    return pd.DataFrame({
        "Variable": names, "x referencia": x_ref, "x GCP": x_app,
        "Error absoluto": ea, "Error relativo": er, "Error porcentual (%)": er * 100,
    })


def relative_residual(A, b, x):
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(b - A @ x) / nb) if nb != 0 else float("nan")


def cond2(A):
    return float(np.linalg.cond(np.asarray(A, float), 2))
