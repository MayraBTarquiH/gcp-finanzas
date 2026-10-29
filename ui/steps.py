"""Procedimientos paso a paso: cada cálculo con sus fórmulas y números."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.linalg import eigh

from core.analysis import analyze, compare_methods
from core.direct import cholesky_steps, forward_steps, backward_steps
from core.scenarios import SCENARIOS, NOMBRES, cargar
from ui.common import latex_matrix, latex_vec, latex_mat, tabla, nota, show_invalid


def _casos():
    s = st.session_state
    casos = [(n, *cargar(n)) for n in SCENARIOS]
    if s.scenario == "Personalizado":
        casos.append(("Personalizado", s.A, s.b))
    return casos


def _analizar(A, b):
    s = st.session_state
    return analyze(A, b, np.array(s.x0, float), s.tol, s.max_iter, s.precond, s.omega)


def _por_escenario(funcion):
    """Ejecuta `funcion(A, b, res)` dentro de una pestaña por escenario."""
    casos = _casos()
    for tab, (nombre, A, b) in zip(st.tabs([c[0] for c in casos]), casos):
        with tab:
            res = _analizar(A, b)
            if not res["valid"]:
                show_invalid(res)
            else:
                funcion(A, b, res)


# ───────────────────────── Formulación: simetría y definida positiva
def pasos_spd(A, v):
    with st.expander("Ver procedimiento paso a paso"):
        st.markdown("**Paso 1. Simetría.** Se compara cada coeficiente con su simétrico.")
        st.latex(rf"\max_{{i,j}} |a_{{ij}} - a_{{ji}}| = {np.abs(A - A.T).max():g} \;\Rightarrow\; A = A^{{T}}")
        st.markdown("**Paso 2. Menores principales (criterio de Sylvester).** Todos deben ser positivos.")
        for k, d in enumerate(v["minors"], start=1):
            st.latex(rf"\Delta_{k} = \det" + latex_matrix(A[:k, :k]) + rf" = {d:.4f} \;{'>' if d > 0 else r'\leq'}\; 0")
        st.markdown("**Paso 3. Autovalores.** Todos deben ser positivos.")
        st.latex(r"\lambda = \{" + ",\\; ".join(f"{x:.4f}" for x in v["eigenvalues"]) + r"\}")
        st.markdown("**Conclusión.** " + ("Los cuatro menores y los cuatro autovalores son positivos, por lo tanto "
                    "A es simétrica y definida positiva." if v["ok"] else "No se cumple alguna condición."))


# ───────────────────────── Cómo se construyen los escenarios
def pasos_escenarios():
    st.subheader("Procedimiento: cómo se construyen los escenarios")
    Ai, bi = cargar("Ideal")
    Ae, be = cargar("Bajo estrés")
    st.markdown("**Bajo estrés a partir del Ideal.** A no cambia, solo b.")
    st.latex(rf"\max |A_{{estrés}} - A_{{ideal}}| = {np.abs(Ae - Ai).max():g}")
    tabla(pd.DataFrame({"Unidad": NOMBRES, "b ideal": bi, "b bajo estrés": be, "Incremento (b₂ − b₁)": be - bi,
                        "Variación (%)": (be - bi) / bi * 100}),
          fmt={"b ideal": "{:g}", "b bajo estrés": "{:g}", "Incremento (b₂ − b₁)": "{:+g}", "Variación (%)": "{:+.1f}"})
    Am, bm = cargar("Mal condicionado")
    d = np.diag(Am)
    D = np.diag(1 / np.sqrt(d))
    Ah = D @ Am @ D
    st.markdown("**Mal condicionado: el escalamiento diagonal.** Se divide cada coeficiente entre √(aᵢᵢ·aⱼⱼ). "
                "Es lo que hace el precondicionador de Jacobi.")
    st.latex(r"\hat a_{ij} = \dfrac{a_{ij}}{\sqrt{a_{ii}\,a_{jj}}}, \qquad "
             rf"\hat a_{{21}} = \dfrac{{{Am[1, 0]:g}}}{{\sqrt{{{d[1]:g}\cdot {d[0]:g}}}}} = {Ah[1, 0]:.4f}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Diagonal de A")
        tabla(pd.DataFrame({"Unidad": NOMBRES, "aᵢᵢ": d, "√aᵢᵢ": np.sqrt(d)}))
    with c2:
        st.markdown("Matriz reescalada Â")
        tabla(pd.DataFrame(Ah, index=NOMBRES, columns=["x₁", "x₂", "x₃", "x₄"]), hide_index=False,
              fmt={c: "{:.4f}" for c in ["x₁", "x₂", "x₃", "x₄"]})
    st.latex(rf"\kappa(A) = {np.linalg.cond(Am):.2f} \;\longrightarrow\; \kappa(\hat A) = {np.linalg.cond(Ah):.4f}")


# ───────────────────────── Cholesky
def _bloque_cholesky(A, b, res):
    st.markdown("**Paso 1. Idea.** Se busca L triangular inferior con A = L Lᵀ.")
    st.latex(r"l_{ii} = \sqrt{a_{ii} - \sum_{k<i} l_{ik}^2}, \qquad "
             r"l_{ij} = \dfrac{a_{ij} - \sum_{k<j} l_{ik}\,l_{jk}}{l_{jj}} \;\; (i>j)")
    st.markdown("**Paso 2. Cálculo de cada elemento de L** (en el orden en que se obtienen).")
    for linea in cholesky_steps(A, res["L"]):
        st.latex(linea)
    st.markdown("**Paso 3. Factor L resultante.**")
    st.latex("L = " + latex_mat(res["L"]))
    st.markdown("**Paso 4. Verificación.** Se multiplica L Lᵀ y se compara con A.")
    st.latex(rf"\max |L L^{{T}} - A| = {np.abs(res['L'] @ res['L'].T - A).max():.2e}")
    st.markdown("**Paso 5. Sustitución hacia adelante:** resolver L y = b.")
    for linea in forward_steps(res["L"], b, res["y"]):
        st.latex(linea)
    st.latex("y = " + latex_vec(res["y"]))
    st.markdown("**Paso 6. Sustitución hacia atrás:** resolver Lᵀ x = y (de x₄ a x₁).")
    for linea in backward_steps(res["L"], res["y"], res["x_ref"]):
        st.latex(linea)
    st.latex(r"x_{ref} = " + latex_vec(res["x_ref"], 5))


def pasos_cholesky():
    st.subheader("Procedimiento paso a paso")
    _por_escenario(_bloque_cholesky)


# ───────────────────────── GCP y GC
def _bloque_gcp(A, b, res):
    s = st.session_state
    M = res["precond"]
    tr = res["gcp"]["trace"]
    ini = tr["init"]
    st.markdown("**Paso 0. Datos de partida**")
    st.latex(r"x_0 = " + latex_vec(ini["x0"]))
    st.latex(r"r_0 = b - A x_0 = " + latex_vec(b) + " - " + latex_vec(ini["Ax0"]) + " = " + latex_vec(ini["r0"]))
    st.latex(rf"\|b\|_2 = {ini['nb']:.4f}, \qquad \dfrac{{\|r_0\|_2}}{{\|b\|_2}} = {ini['rel0']:.4e}")
    st.markdown(f"**Precondicionador:** {M.name}.")
    if s.precond == "Jacobi":
        st.latex(r"M = \operatorname{diag}(A) = " + latex_mat(M.matrix, 4) +
                 r" \;\Rightarrow\; z_i = r_i / a_{ii}")
    else:
        st.latex(r"M = " + latex_mat(M.matrix, 4))
    if "z0" in ini:
        st.latex(r"M z_0 = r_0 \;\Rightarrow\; z_0 = " + latex_vec(ini["z0"]) +
                 rf", \qquad p_1 = z_0, \qquad r_0^{{T}} z_0 = {ini['rz0']:.4f}")
    else:
        nota("El residuo inicial ya cumple la tolerancia: no hizo falta iterar.")
    for it in tr["iters"]:
        k = it["k"]
        with st.expander(f"Iteración {k}  ·  residuo relativo = {it['rel']:.3e}", expanded=(k == 1)):
            st.latex(rf"p_{{{k}}} = " + latex_vec(it["p"]))
            st.latex(rf"A p_{{{k}}} = " + latex_vec(it["Ap"]) +
                     rf", \qquad p_{{{k}}}^{{T}} A p_{{{k}}} = {it['pAp']:.4f}")
            st.latex(rf"\alpha_{{{k}}} = \dfrac{{r_{{{k-1}}}^{{T}} z_{{{k-1}}}}}{{p_{{{k}}}^{{T}} A p_{{{k}}}}}"
                     rf" = \dfrac{{{it['rz']:.4f}}}{{{it['pAp']:.4f}}} = {it['alpha']:.6f}")
            st.latex(rf"x_{{{k}}} = x_{{{k-1}}} + \alpha_{{{k}}} p_{{{k}}} = " + latex_vec(it["x_prev"]) +
                     rf" + {it['alpha']:.4f}" + latex_vec(it["p"]) + " = " + latex_vec(it["x"]))
            st.latex(rf"r_{{{k}}} = r_{{{k-1}}} - \alpha_{{{k}}} A p_{{{k}}} = " + latex_vec(it["r"]))
            ok = it["rel"] < s.tol
            st.latex(rf"\dfrac{{\|r_{{{k}}}\|_2}}{{\|b\|_2}} = {it['rel']:.3e} \;{'<' if ok else r'\geq'}\; {s.tol:.0e}")
            if ok:
                st.success("Se cumple la tolerancia: el algoritmo se detiene aquí.")
            else:
                st.latex(rf"M z_{{{k}}} = r_{{{k}}} \;\Rightarrow\; z_{{{k}}} = " + latex_vec(it["z"]))
                st.latex(rf"\beta_{{{k+1}}} = \dfrac{{r_{{{k}}}^{{T}} z_{{{k}}}}}{{r_{{{k-1}}}^{{T}} z_{{{k-1}}}}}"
                         rf" = \dfrac{{{it['rz_new']:.4f}}}{{{it['rz']:.4f}}} = {it['beta']:.6f}")
                st.latex(rf"p_{{{k+1}}} = z_{{{k}}} + \beta_{{{k+1}}} p_{{{k}}} = " + latex_vec(it["p_next"]))
    st.markdown("**Resumen por iteración: GC (contraste) y GCP**")
    gc_it, gcp_it = res["gc"]["trace"]["iters"], tr["iters"]
    filas = []
    for k in range(max(len(gc_it), len(gcp_it))):
        g = gc_it[k] if k < len(gc_it) else None
        p = gcp_it[k] if k < len(gcp_it) else None
        filas.append({"Iteración": k + 1,
                      "α GC": f"{g['alpha']:.5f}" if g else "—", "Residuo GC": f"{g['rel']:.2e}" if g else "—",
                      "α GCP": f"{p['alpha']:.5f}" if p else "—", "Residuo GCP": f"{p['rel']:.2e}" if p else "—"})
    tabla(pd.DataFrame(filas))


def pasos_gcp():
    st.subheader("Procedimiento paso a paso del GCP")
    st.caption("Cada pestaña recorre las iteraciones de un escenario con la tolerancia y el precondicionador "
               "elegidos en «Entrada de datos».")
    _por_escenario(_bloque_gcp)


# ───────────────────────── Errores, residuo y condicionamiento
def _bloque_errores(A, b, res):
    xr, xg = res["x_ref"], res["gcp"]["x"]
    st.markdown("**Paso 1. Errores por variable.**")
    st.latex(r"E_a = |x_{ref} - x_{GCP}|, \qquad E_r = \dfrac{E_a}{|x_{ref}|}, \qquad E_p = E_r \times 100\,\%")
    filas = []
    for i, n in enumerate(NOMBRES):
        ea = abs(xr[i] - xg[i])
        er = ea / abs(xr[i]) if xr[i] != 0 else None
        filas.append({"Variable": n,
                      "Eₐ = |x_ref − x_GCP|": f"|{xr[i]:.6f} − {xg[i]:.6f}| = {ea:.2e}",
                      "Eᵣ = Eₐ / |x_ref|": f"{ea:.2e} / {abs(xr[i]):.6f} = {er:.2e}" if er is not None else "no definido",
                      "Eₚ = Eᵣ × 100": f"{er * 100:.2e} %" if er is not None else "no definido"})
    tabla(pd.DataFrame(filas))
    st.markdown("**Paso 2. Residuo relativo global.**")
    Ax = A @ xg
    rv = b - Ax
    st.latex(r"A x_{GCP} = " + latex_vec(Ax, 6))
    st.latex(r"b - A x_{GCP} = " + latex_vec(b) + " - " + latex_vec(Ax, 6) + " = " + latex_vec(rv, sci=True))
    st.latex(rf"\dfrac{{\|b - A x_{{GCP}}\|_2}}{{\|b\|_2}} = \dfrac{{{np.linalg.norm(rv):.3e}}}"
             rf"{{{np.linalg.norm(b):.4f}}} = {res['resid_gcp']:.3e}")
    st.markdown("**Paso 3. Número de condición de A.** Para A simétrica definida positiva, κ(A) = λmáx / λmín.")
    w = np.linalg.eigvalsh(A)
    st.latex(r"\lambda(A) = \{" + ",\\; ".join(f"{x:.4f}" for x in w) + r"\}")
    st.latex(rf"\kappa(A) = \dfrac{{{w[-1]:.4f}}}{{{w[0]:.4f}}} = {res['kappa_A']:.4f}")
    st.markdown("**Paso 4. Número de condición de la matriz precondicionada.** Se usan los autovalores de M⁻¹A "
                "(es decir, A v = λ M v).")
    M = res["precond"]
    wm = eigh(A, M.matrix, eigvals_only=True)
    st.latex(rf"M = " + latex_mat(M.matrix, 4))
    st.latex(r"\lambda(M^{-1}A) = \{" + ",\\; ".join(f"{x:.4f}" for x in wm) + r"\}")
    st.latex(rf"\kappa(M^{{-1}}A) = \dfrac{{{wm[-1]:.4f}}}{{{wm[0]:.4f}}} = {res['kappa_M']:.4f}")
    st.markdown("**Conclusión.** " + (
        f"El precondicionador redujo el número de condición de {res['kappa_A']:.2f} a {res['kappa_M']:.2f}."
        if res["kappa_M"] < res["kappa_A"] else
        f"El precondicionador no mejoró el número de condición ({res['kappa_A']:.2f} → {res['kappa_M']:.2f})."))


def pasos_errores():
    st.subheader("Procedimiento paso a paso")
    _por_escenario(_bloque_errores)


# ───────────────────────── Comparación de todos los métodos
def _bloque_comparacion(A, b, res):
    s = st.session_state
    df = compare_methods(A, b, np.array(s.x0, float), s.tol, s.max_iter, s.omega)
    fmt = {c: "{:.6f}" for c in ["x₁", "x₂", "x₃", "x₄"]}
    fmt.update({"Iteraciones": "{:.0f}", "Residuo relativo": "{:.2e}", "Error absoluto máx.": "{:.2e}",
                "Error % máx.": "{:.2e}", "Número de condición": "{:.4g}"})
    tabla(df, fmt=fmt, na_rep="—")
    st.caption("Los errores se miden contra Cholesky. El número de condición es κ(A) para Cholesky, máximo "
               "descenso y GC, y κ(M⁻¹A) para los GCP. El máximo descenso se deja iterar hasta 20 000 pasos.")


def comparacion_metodos():
    st.subheader("Tabla de comparación de todos los métodos")
    _por_escenario(_bloque_comparacion)
