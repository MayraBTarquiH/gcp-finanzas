import numpy as np
import pandas as pd
import streamlit as st
from core.scenarios import cargar, SCENARIOS, NOMBRES, SIMBOLOS
from core.analysis import analyze


def init_state():
    if "init" not in st.session_state:
        A, b = cargar("Ideal")
        st.session_state.update(init=True, scenario="Ideal", A=A, b=b, A_base=A.copy(), b_base=b.copy(),
                                x0=[0.0] * 4, tol=1e-6, max_iter=1000, precond="Jacobi", omega=1.0, editor_v=0)


def load_scenario(name):
    A, b = cargar(name)
    st.session_state.update(scenario=name, A=A, b=b, A_base=A.copy(), b_base=b.copy())
    st.session_state.editor_v += 1


def run_current():
    s = st.session_state
    return analyze(s.A, s.b, np.array(s.x0, float), s.tol, s.max_iter, s.precond, s.omega)


def run_scenario(name):
    s = st.session_state
    A, b = cargar(name)
    return analyze(A, b, np.array(s.x0, float), s.tol, s.max_iter, s.precond, s.omega)


def show_invalid(res):
    st.error("No se puede calcular porque la matriz A no es válida:\n\n" +
             "\n".join(f"- {e}" for e in res["validation"]["errors"]))
    st.info("Corrígela en «Entrada de datos» o carga uno de los tres escenarios.")


def nota(texto, aviso=False):
    st.markdown(f'<div class="nota{" aviso" if aviso else ""}">{texto}</div>', unsafe_allow_html=True)


def latex_matrix(M, vertical=False):
    M = np.asarray(M)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    filas = " \\\\ ".join(" & ".join(f"{v:g}" for v in fila) for fila in M)
    return "\\begin{bmatrix} " + filas + " \\end{bmatrix}"


def fmt_vec(v, sci=False):
    if sci:
        return "[" + ", ".join(f"{x:.2e}" for x in v) + "]"
    return "[" + ", ".join(f"{x:.5f}" for x in v) + "]"


def df_matrix(A):
    return pd.DataFrame(A, index=NOMBRES, columns=SIMBOLOS)
