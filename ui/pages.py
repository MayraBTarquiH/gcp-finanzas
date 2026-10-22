import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import qa
from core.cg import PRECONDICIONADORES
from core.scenarios import SCENARIOS, VARIABLES, NOMBRES, SIMBOLOS, UNIDAD, cargar
from core.validation import validate_spd
from ui.config import ESTUDIANTE
from ui.common import (run_current, run_scenario, show_invalid, nota, latex_matrix, fmt_vec,
                       load_scenario, df_matrix)

ESCENARIOS = list(SCENARIOS)


# 1 ─────────────────────────────────────────────────────────────
def inicio():
    st.markdown("""<div class="hero">
      <h1>Presupuesto de TI con gradientes conjugados precondicionados</h1>
      <p>Reparto de recursos entre cuatro unidades de una empresa, resuelto con el método de
      Suñagua (2020) y analizado en tres escenarios: ideal, bajo estrés y mal condicionado.</p></div>""",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Datos del estudiante")
        st.table(pd.DataFrame({"": list(ESTUDIANTE.values())}, index=list(ESTUDIANTE.keys())))
    with c2:
        st.subheader("Síntesis del caso")
        st.write(
            "Una empresa debe repartir su presupuesto anual de tecnología entre desarrollo de software, "
            "infraestructura y nube, ciberseguridad y soporte. El reparto óptimo resulta de un sistema "
            "lineal **Ax = b** con A simétrica y definida positiva de 4 × 4. Se resuelve con Cholesky "
            "(solución de referencia) y con gradientes conjugados, con y sin precondicionador, para "
            "comparar precisión, número de condición y velocidad de convergencia.")
        st.write("**Cómo recorrer la página:** carga un escenario en «Tres escenarios», revisa su "
                 "formulación y resultados en los módulos siguientes, y edita los coeficientes en "
                 "«Entrada de datos» para ver cómo se recalcula todo.")


# 2 ─────────────────────────────────────────────────────────────
def caso():
    st.title("Caso de estudio")
    st.subheader("Contexto financiero")
    st.write(
        "Una empresa mediana de servicios en Bolivia debe decidir cuánto del presupuesto anual de TI "
        f"asigna a cada una de cuatro unidades. Las asignaciones se miden en {UNIDAD}. Cada unidad "
        "tiene un costo creciente (cada Bs adicional rinde menos que el anterior) y comparte recursos "
        "con las demás, de modo que invertir en una afecta el costo de las otras.")
    st.subheader("Por qué el modelo lleva a Ax = b")
    st.latex(r"\min_x \; f(x) = \tfrac{1}{2}\,x^{T}Ax - b^{T}x \;\;\Rightarrow\;\; \nabla f(x) = Ax - b = 0")
    st.write(
        "f es el costo total menos el beneficio esperado. Como A es simétrica y definida positiva, f es "
        "estrictamente convexa y tiene un único mínimo, que es la solución de Ax = b. Por eso el sistema "
        "no es arbitrario: es la condición de óptimo de un problema de asignación de recursos.")
    st.subheader("Procedencia de los datos")
    nota("Los coeficientes son datos <b>ilustrativos de carácter académico</b>, definidos con criterio "
         "financiero, no provienen de una empresa real. Los órdenes de magnitud buscan representar costos "
         "propios similares (escenario ideal) o de escalas muy distintas (escenario mal condicionado).",
         aviso=True)
    st.subheader("Justificación de los tres escenarios")
    st.dataframe(pd.DataFrame({
        "Escenario": ESCENARIOS,
        "Qué cambia": [SCENARIOS[e]["resumen"] for e in ESCENARIOS],
        "Justificación": [SCENARIOS[e]["justificacion"] for e in ESCENARIOS]}),
        hide_index=True, width="stretch")


# 3 ─────────────────────────────────────────────────────────────
def formulacion():
    st.title("Formulación")
    st.subheader("Variables")
    st.dataframe(pd.DataFrame({
        "Variable": SIMBOLOS, "Unidad": NOMBRES,
        "Significado financiero": [v["significado"] for v in VARIABLES],
        "Medida en": UNIDAD}), hide_index=True, width="stretch")
    st.subheader("Sistema lineal del escenario activo")
    s = st.session_state
    st.caption(f"Escenario activo: {s.scenario}")
    xvec = r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix}"
    st.latex(r"A\,x = b \quad\Rightarrow\quad " + latex_matrix(s.A) + xvec + " = " + latex_matrix(s.b))
    st.subheader("Interpretación de los coeficientes")
    st.markdown(
        "- **aᵢᵢ (diagonal):** costo propio de la unidad *i* con rendimientos decrecientes. "
        "Un valor mayor significa que cada Bs adicional en esa unidad es más costoso.\n"
        "- **aᵢⱼ (fuera de la diagonal):** interacción entre las unidades *i* y *j* (competencia por "
        "recursos compartidos o costos de coordinación). Es simétrica: la interacción de *i* con *j* es "
        "la misma que la de *j* con *i*.\n"
        "- **bᵢ:** beneficio marginal esperado de invertir en la unidad *i*.")
    st.subheader("Demostración: A es simétrica y definida positiva")
    st.write("Se verifica con tres criterios: **A = Aᵀ**, **menores principales positivos** (criterio de "
             "Sylvester) y **autovalores positivos**. Además, Cholesky se completa solo si A es definida positiva.")
    tabs = st.tabs(ESCENARIOS)
    for tab, nombre in zip(tabs, ESCENARIOS):
        with tab:
            A, _ = cargar(nombre)
            v = validate_spd(A)
            c1, c2, c3 = st.columns(3)
            c1.metric("A = Aᵀ", "Sí" if v["symmetric"] else "No")
            c2.metric("Menor autovalor", f"{v['eigenvalues'].min():.4g}")
            c3.metric("Definida positiva", "Sí" if v["positive_definite"] else "No")
            st.dataframe(pd.DataFrame({
                "Menor principal": [f"Δ{k}" for k in range(1, 5)], "Valor": v["minors"],
                "Autovalor": v["eigenvalues"]}), hide_index=True, width="stretch")
            st.latex("A = " + latex_matrix(A))


# 4 ─────────────────────────────────────────────────────────────
def entrada():
    st.title("Entrada de datos")
    s = st.session_state
    st.write("Edita los coeficientes y todo se recalcula en los demás módulos. Al editar, el escenario "
             "pasa a «Personalizado».")
    dfA = pd.DataFrame(s.A_base, index=NOMBRES, columns=SIMBOLOS)
    dfb = pd.DataFrame({"b": s.b_base}, index=NOMBRES)
    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown("**Matriz A**")
        eA = st.data_editor(dfA, key=f"edA_{s.editor_v}", width="stretch")
    with c2:
        st.markdown("**Vector b**")
        eb = st.data_editor(dfb, key=f"edb_{s.editor_v}", width="stretch")
    A_new, b_new = eA.to_numpy(dtype=float), eb["b"].to_numpy(dtype=float)
    if not (np.array_equal(A_new, s.A, equal_nan=True) and np.array_equal(b_new, s.b, equal_nan=True)):
        s.A, s.b = A_new, b_new
        if not (np.array_equal(A_new, s.A_base, equal_nan=True) and np.array_equal(b_new, s.b_base, equal_nan=True)):
            s.scenario = "Personalizado"
    v = validate_spd(s.A)
    if v["ok"]:
        st.success("A es simétrica y definida positiva. Los métodos se pueden ejecutar.")
    else:
        for e in v["errors"]:
            st.error(e)
    st.markdown("**Parámetros de ejecución**")
    st.write("Vector inicial x₀ (por defecto cero):")
    cols = st.columns(4)
    for i, col in enumerate(cols):
        s.x0[i] = col.number_input(SIMBOLOS[i], value=float(s.x0[i]), key=f"x0w_{i}", format="%.4f")
    p1, p2, p3, p4 = st.columns(4)
    s.tol = p1.select_slider("Tolerancia ‖rₖ‖/‖b‖", options=[1e-4, 1e-6, 1e-8, 1e-10], value=s.tol,
                             format_func=lambda v: f"{v:.0e}")
    s.max_iter = p2.number_input("Iteraciones máximas", 1, 10000, int(s.max_iter))
    s.precond = p3.selectbox("Precondicionador", PRECONDICIONADORES,
                             index=PRECONDICIONADORES.index(s.precond))
    if s.precond == "SSOR":
        s.omega = p4.slider("ω (SSOR)", 0.1, 1.9, float(s.omega), 0.1)
    st.button("Restablecer datos del escenario cargado", on_click=lambda: load_scenario(
        s.scenario if s.scenario in SCENARIOS else "Ideal"))


# 5 ─────────────────────────────────────────────────────────────
def escenarios():
    st.title("Tres escenarios")
    st.write("Los tres comparten las mismas cuatro variables y la estructura del modelo. Cambia A, b o ambos.")
    cols = st.columns(3)
    for col, nombre in zip(cols, ESCENARIOS):
        activo = st.session_state.scenario == nombre
        with col:
            st.markdown(f'<div class="ficha{" activa" if activo else ""}"><h4>{nombre}</h4>'
                        f'<p>{SCENARIOS[nombre]["resumen"]}</p></div>', unsafe_allow_html=True)
            st.button("Cargar y ejecutar" if not activo else "Escenario activo", key=f"btn_{nombre}",
                      on_click=load_scenario, args=(nombre,), disabled=activo, width="stretch")
    st.divider()
    s = st.session_state
    st.subheader(f"Escenario activo: {s.scenario}")
    if s.scenario in SCENARIOS:
        nota(SCENARIOS[s.scenario]["justificacion"])
    c1, c2 = st.columns([2.2, 1])
    c1.dataframe(df_matrix(s.A), width="stretch")
    c2.dataframe(pd.DataFrame({"b": s.b}, index=NOMBRES), width="stretch")
    res = run_current()
    if not res["valid"]:
        return show_invalid(res)
    m = st.columns(4)
    m[0].metric("κ(A)", f"{res['kappa_A']:.4g}")
    m[1].metric("κ(M⁻¹A)", f"{res['kappa_M']:.4g}")
    m[2].metric("Iteraciones GC", res["gc"]["iterations"])
    m[3].metric("Iteraciones GCP", res["gcp"]["iterations"])
    st.write("**Solución (GCP):** " + ", ".join(f"{n}: {x:.4f}" for n, x in zip(NOMBRES, res["gcp"]["x"])))


# 6 ─────────────────────────────────────────────────────────────
def referencia():
    st.title("Solución de referencia: Cholesky")
    res = run_current()
    if not res["valid"]:
        return show_invalid(res)
    s = st.session_state
    st.write("Se factoriza **A = L Lᵀ** (programado a mano, sin librerías) y se resuelven dos sistemas "
             "triangulares: **L y = b** por sustitución hacia adelante y **Lᵀ x = y** hacia atrás.")
    st.caption(f"Escenario activo: {s.scenario}")
    c1, c2 = st.columns(2)
    c1.markdown("**Factor L**")
    c1.dataframe(pd.DataFrame(res["L"], index=NOMBRES, columns=SIMBOLOS), width="stretch")
    c2.markdown("**Vector intermedio y y solución x**")
    c2.dataframe(pd.DataFrame({"y": res["y"], f"x ({UNIDAD})": res["x_ref"]}, index=NOMBRES),
                 width="stretch")
    err_fact = np.max(np.abs(res["L"] @ res["L"].T - s.A))
    err_lib = np.max(np.abs(res["x_ref"] - np.linalg.solve(s.A, s.b)))
    m = st.columns(2)
    m[0].metric("máx |L Lᵀ − A|", f"{err_fact:.1e}")
    m[1].metric("máx |x − numpy.linalg.solve|", f"{err_lib:.1e}")
    nota("Si Cholesky se completa sin errores, A es definida positiva. Esta solución es la "
         "«solución numérica de referencia» contra la que se miden los errores del GCP.")


# 7 ─────────────────────────────────────────────────────────────
def gcp_vs_gc():
    st.title("GCP vs GC")
    res = run_current()
    if not res["valid"]:
        return show_invalid(res)
    s = st.session_state
    st.caption(f"Escenario activo: {s.scenario} · Precondicionador: {res['precond'].name} · "
               f"Tolerancia: {s.tol:.0e}")
    st.write("El **GC** es el método sin precondicionar y solo sirve de contraste. El **GCP** resuelve "
             "**M zₖ = rₖ** en cada iteración (Algoritmo 2 de Suñagua, 2020).")
    gc, gcp = res["gc"], res["gcp"]
    st.dataframe(pd.DataFrame({
        "Método": ["Solución de referencia (Cholesky)", "GC (sin precondicionar)", f"GCP ({res['precond'].name})"],
        "Vector solución": [fmt_vec(res["x_ref"]), fmt_vec(gc["x"]), fmt_vec(gcp["x"])],
        "Iteraciones": ["—", str(gc["iterations"]), str(gcp["iterations"])],
        "Residuo relativo final": ["—", f"{res['resid_gc']:.2e}", f"{res['resid_gcp']:.2e}"],
        "¿Convergió?": ["—", "Sí" if gc["converged"] else "No", "Sí" if gcp["converged"] else "No"],
    }), hide_index=True, width="stretch")
    m = st.columns(2)
    m[0].metric("κ(A)", f"{res['kappa_A']:.4g}")
    m[1].metric("κ(M⁻¹A)", f"{res['kappa_M']:.4g}",
                delta=f"{res['kappa_M'] / res['kappa_A'] - 1:+.1%} respecto a κ(A)", delta_color="inverse")
    tol_line = pd.DataFrame({"y": [s.tol]})
    hist = pd.concat([
        pd.DataFrame({"Iteración": range(len(gc["history"])), "Residuo relativo": np.maximum(gc["history"], 1e-18), "Método": "GC"}),
        pd.DataFrame({"Iteración": range(len(gcp["history"])), "Residuo relativo": np.maximum(gcp["history"], 1e-18), "Método": "GCP"})])
    base = alt.Chart(hist).mark_line(point=True).encode(
        x=alt.X("Iteración:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Residuo relativo:Q", scale=alt.Scale(type="log")),
        color=alt.Color("Método:N", scale=alt.Scale(range=["#C98A1B", "#0F8B8D"])))
    rule = alt.Chart(tol_line).mark_rule(strokeDash=[5, 4], color="#B3261E").encode(y="y:Q")
    st.altair_chart(base + rule, width="stretch")
    st.caption("Línea roja: tolerancia. Eje vertical en escala logarítmica.")
    nota("En un sistema de solo n = 4 variables, el GC termina en a lo sumo 4 iteraciones por teoría, "
         "por eso la diferencia en iteraciones puede ser pequeña. La ganancia del precondicionador se ve "
         "con claridad en el número de condición κ(M⁻¹A).")


# 8 ─────────────────────────────────────────────────────────────
def errores():
    st.title("Análisis de errores")
    res = run_current()
    if not res["valid"]:
        return show_invalid(res)
    st.subheader("Errores por variable (GCP vs referencia)")
    df = res["errors"]
    fmt = {"x referencia": "{:.6f}", "x GCP": "{:.6f}", "Error absoluto": "{:.2e}",
           "Error relativo": "{:.2e}", "Error porcentual (%)": "{:.2e}"}
    st.dataframe(df.style.format(fmt, na_rep="no definido"), hide_index=True, width="stretch")
    if df["Error relativo"].isna().any():
        st.caption("Si xref,i = 0, el error relativo no está definido para esa componente.")
    m = st.columns(3)
    m[0].metric("Residuo relativo ‖b − Ax‖/‖b‖ (GCP)", f"{res['resid_gcp']:.2e}")
    m[1].metric("Máximo error absoluto", f"{df['Error absoluto'].max():.2e}")
    m[2].metric("¿Convergió el GCP?", "Sí" if res["gcp"]["converged"] else "No")
    st.subheader("Tabla comparativa de los tres escenarios")
    st.caption("Calculada en vivo con los parámetros actuales (x₀, tolerancia y precondicionador).")
    filas = {k: [] for k in ["Solución numérica de referencia", "Solución GCP", "Error absoluto (Ea)",
                              "Error porcentual (Ep)", "Iteraciones GC (sin precondicionar)",
                              "Iteraciones GCP (precondicionado)", "Número de condición κ(A)",
                              "Número de condición κ(M⁻¹A)", "Residuo relativo final GCP",
                              "¿Convergió el algoritmo?", "Interpretación financiera"]}
    for nombre in ESCENARIOS:
        r = run_scenario(nombre)
        e = r["errors"]
        filas["Solución numérica de referencia"].append(fmt_vec(r["x_ref"]))
        filas["Solución GCP"].append(fmt_vec(r["gcp"]["x"]))
        filas["Error absoluto (Ea)"].append(fmt_vec(e["Error absoluto"], sci=True))
        filas["Error porcentual (Ep)"].append(fmt_vec(e["Error porcentual (%)"], sci=True))
        filas["Iteraciones GC (sin precondicionar)"].append(r["gc"]["iterations"])
        filas["Iteraciones GCP (precondicionado)"].append(r["gcp"]["iterations"])
        filas["Número de condición κ(A)"].append(f"{r['kappa_A']:.4g}")
        filas["Número de condición κ(M⁻¹A)"].append(f"{r['kappa_M']:.4g}")
        filas["Residuo relativo final GCP"].append(f"{r['resid_gcp']:.2e}")
        filas["¿Convergió el algoritmo?"].append("Sí" if r["gcp"]["converged"] else "No")
        filas["Interpretación financiera"].append(SCENARIOS[nombre]["interpretacion"])
    tabla = pd.DataFrame(filas, index=ESCENARIOS).T
    tabla.index.name = "Indicador"
    tabla = tabla.astype(str)
    st.table(tabla)


# 9 ─────────────────────────────────────────────────────────────
def qa_page():
    st.title("Módulo de pruebas (QA)")
    st.write("Cada prueba se ejecuta de verdad sobre el código de la página, no son resultados escritos a mano.")
    if st.button("Ejecutar todas las pruebas" if "qa" not in st.session_state else "Volver a ejecutar",
                 type="primary"):
        st.session_state.qa = qa.run_all()
    if "qa" in st.session_state:
        rows = st.session_state.qa
        ok = sum(r["Conclusión"] == "Aprobada" for r in rows)
        (st.success if ok == len(rows) else st.error)(f"{ok} de {len(rows)} pruebas aprobadas")
        st.table(pd.DataFrame(rows).set_index("Prueba"))
    else:
        st.info("Presiona el botón para ejecutar las pruebas.")
    st.caption("Las pruebas 1 a 4 son las obligatorias del desafío. La 5 verifica el Cholesky propio.")


# 10 ────────────────────────────────────────────────────────────
def conclusiones():
    st.title("Conclusiones y referencias")
    r = {n: run_scenario(n) for n in ESCENARIOS}
    mal = r["Mal condicionado"]
    if all(x["valid"] for x in r.values()):
        st.subheader("Interpretación técnica")
        st.markdown(
            f"- En los tres escenarios el GCP convergió y coincidió con la solución de referencia de "
            f"Cholesky (error absoluto máximo del orden de "
            f"{max(x['errors']['Error absoluto'].max() for x in r.values()):.0e}).\n"
            f"- En el escenario mal condicionado, el precondicionador redujo el número de condición de "
            f"κ(A) = {mal['kappa_A']:.1f} a κ(M⁻¹A) = {mal['kappa_M']:.2f}. Eso confirma lo que dice "
            f"Suñagua (2020): la velocidad de convergencia depende de κ y el precondicionador la mejora.\n"
            f"- Iteraciones en ese escenario: GC = {mal['gc']['iterations']}, GCP = {mal['gcp']['iterations']}. "
            f"Con n = 4, el GC termina en a lo sumo 4 iteraciones, así que la ventaja en iteraciones "
            f"crece con la dimensión del sistema y con κ.\n"
            f"- El escenario de estrés no altera κ(A), porque solo cambia b: desplaza el equilibrio, no la "
            f"dificultad numérica.")
    st.subheader("Interpretación financiera")
    st.markdown(
        "- Con costos equilibrados (Ideal), el presupuesto se reparte de forma pareja.\n"
        "- Bajo estrés, ciberseguridad y desarrollo exigen mucho más presupuesto, que es lo esperable "
        "con inflación salarial y más incidentes.\n"
        "- Con escalas de costo muy distintas (Mal condicionado), el sistema es sensible a pequeños cambios "
        "en los datos: conviene reescalar o precondicionar antes de confiar en la asignación.")
    st.subheader("Referencias")
    st.markdown(
        "- Suñagua S., P. (2020). Método de Gradientes Conjugados Precondicionado. *Revista Boliviana de "
        "Matemática – UMSA*, 04, 2–7.\n"
        "- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.\n"
        "- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.\n"
        "- Datos: coeficientes ilustrativos definidos para este trabajo académico (ver «Caso de estudio»).")
