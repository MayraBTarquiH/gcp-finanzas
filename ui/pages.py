from datetime import datetime, timedelta, timezone

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
                       load_scenario, df_matrix, tabla, latex_system)
from ui import steps

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

    st.subheader("Planteamiento del problema")

    st.write(
        "Una empresa mediana de servicios en Bolivia dispone de un "
        "presupuesto anual de tecnología que debe distribuir entre "
        "cuatro unidades:"
    )

    tabla(pd.DataFrame({
        "Variable": ["x₁", "x₂", "x₃", "x₄"],
        "Unidad": [
            "Desarrollo de software",
            "Infraestructura y nube",
            "Ciberseguridad",
            "Soporte y capacitación"
        ],
        "Descripción": [
            "Desarrollo y mantenimiento de sistemas.",
            "Servidores, almacenamiento y servicios en la nube.",
            "Protección de sistemas y datos.",
            "Asistencia técnica y formación del personal."
        ]
    }))

    st.info(
        f"El objetivo es determinar la distribución óptima del presupuesto "
        f"entre las cuatro unidades, considerando sus costos e interacciones. "
        f"Los valores se expresan en {UNIDAD}."
    )

    st.subheader("Objetivo del estudio")

    st.write(
        "Resolver el problema de asignación de recursos mediante "
        "métodos numéricos, comparar su precisión y analizar cómo "
        "influye el precondicionamiento en la velocidad de convergencia."
    )

    st.subheader("Escenarios de evaluación")

    st.write(
        "Se utilizan tres escenarios para estudiar el comportamiento "
        "de los métodos ante diferentes condiciones del sistema."
    )

    tabla(pd.DataFrame({
        "Escenario": ESCENARIOS,
        "Características": [
            SCENARIOS[e]["resumen"] for e in ESCENARIOS
        ],
        "Justificación": [
            SCENARIOS[e]["justificacion"] for e in ESCENARIOS
        ]
    }))

    st.subheader("Procedencia de los datos")

    nota(
        "Los coeficientes utilizados son datos ilustrativos de carácter "
        "académico. No proceden de una empresa real; fueron definidos "
        "para representar distintas condiciones financieras y numéricas.",
        aviso=True
    )



# 3 ─────────────────────────────────────────────────────────────

def formulacion():
    st.title("Formulación matemática")

    # 1. Variables del modelo
    st.subheader("1. Variables de decisión")

    tabla(pd.DataFrame({
        "Variable": SIMBOLOS,
        "Unidad": NOMBRES,
        "Significado financiero": [v["significado"] for v in VARIABLES],
        "Medida en": UNIDAD
    }))

    # 2. Modelo matemático
    st.subheader("2. Modelo de optimización")

    st.write(
        "El problema consiste en encontrar la distribución del presupuesto "
        "que minimice el costo total menos el beneficio esperado."
    )

    st.latex(
        r"\min_x \; f(x) = \frac{1}{2}x^TAx - b^Tx"
    )

    st.markdown("**Condición de optimalidad:**")

    st.latex(
        r"\nabla f(x) = Ax-b=0"
    )

    st.info(
        "Al igualar el gradiente a cero se obtiene el sistema lineal "
        "Ax = b. Como A es simétrica y definida positiva, existe "
        "una única solución que minimiza la función."
    )

    # 3. Sistema lineal
    st.subheader("3. Sistema lineal del escenario activo")

    s = st.session_state

    st.caption(f"Escenario activo: {s.scenario}")

    st.markdown("**Sistema de ecuaciones:**")

    st.latex(latex_system(s.A, s.b))

    st.markdown("**Representación matricial:**")

    xvec = r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix}"

    st.latex(
        r"A\,x = b \quad\Rightarrow\quad "
        + latex_matrix(s.A)
        + xvec
        + " = "
        + latex_matrix(s.b)
    )

    # 4. Interpretación
    st.subheader("4. Interpretación de los coeficientes")

    st.markdown(
        "- **aᵢᵢ (diagonal):** representa el costo propio de cada unidad.\n"
        "- **aᵢⱼ (fuera de la diagonal):** representa la interacción "
        "entre dos unidades por recursos compartidos o coordinación.\n"
        "- **bᵢ:** representa el beneficio marginal esperado de invertir "
        "en cada unidad."
    )

    # 5. Validación matemática
    st.subheader("5. Verificación de la matriz")

    st.write(
        "Para aplicar los métodos de Gradientes Conjugados y Cholesky, "
        "se verifica que A sea simétrica y definida positiva."
    )

    st.caption(
        "La comprobación utiliza la simetría, los menores principales "
        "(criterio de Sylvester) y los autovalores."
    )

    tabs = st.tabs(ESCENARIOS)

    for tab, nombre in zip(tabs, ESCENARIOS):
        with tab:
            A, b = cargar(nombre)
            v = validate_spd(A)

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "A = Aᵀ",
                "Sí" if v["symmetric"] else "No"
            )

            c2.metric(
                "Menor autovalor",
                f"{v['eigenvalues'].min():.4g}"
            )

            c3.metric(
                "Definida positiva",
                "Sí" if v["positive_definite"] else "No"
            )

            tabla(pd.DataFrame({
                "Menor principal": [f"Δ{k}" for k in range(1, 5)],
                "Valor": v["minors"],
                "Autovalor": v["eigenvalues"]
            }), fmt={
                "Valor": "{:.4f}",
                "Autovalor": "{:.4f}"
            })

            st.markdown("**Sistema de ecuaciones**")
            st.latex(latex_system(A, b))

            st.markdown("**Matriz A y vector b**")
            st.latex(
                "A = " + latex_matrix(A)
                + r", \qquad b = " + latex_matrix(b)
            )

            steps.pasos_spd(A, v)
          

# 4 ─────────────────────────────────────────────────────────────
CORTOS = ["Desarrollo", "Infraestructura", "Ciberseguridad", "Soporte"]
COLS = [f"{SIMBOLOS[i]}" for i in range(4)]
FILAS = [f"Ec. {i + 1} · {NOMBRES[i]}" for i in range(4)]


def _hacer_simetrica():
    s = st.session_state
    A = np.triu(s.A) + np.triu(s.A, 1).T       # copia el triángulo superior al inferior
    s.A, s.A_base = A, A.copy()
    s.b_base = s.b.copy()
    s.editor_v += 1


CONFIG_PREDETERMINADA = {"tol": 1e-6, "max_iter": 1000, "precond": "Jacobi", "omega": 1.0}


def _restablecer_configuracion():
    """Devuelve x₀, tolerancia, iteraciones máximas y precondicionador a sus valores predeterminados."""
    s = st.session_state
    s.x0 = [0.0] * 4
    for i in range(4):
        s[f"x0w_{i}"] = 0.0            # estado de las casillas de x₀
    for k, v in CONFIG_PREDETERMINADA.items():
        s[k] = v
    s["tol_w"], s["maxit_w"], s["precond_w"], s["omega_w"] = 1e-6, 1000, "Jacobi", 1.0  # estado de los controles


def _firma_datos():
    """Resume todo lo que el usuario puede cambiar, para detectar cuándo hay que recalcular."""
    s = st.session_state
    return (s.A.tobytes(), s.b.tobytes(), tuple(s.x0), s.tol, int(s.max_iter), s.precond, float(s.omega))


def _registrar_calculo(clic):
    """Guarda la hora del último cálculo y el motivo (botón, cambio de datos o cálculo inicial)."""
    s = st.session_state
    ahora = datetime.now(timezone(timedelta(hours=-4)))          # hora de Bolivia (UTC-4)
    firma = _firma_datos()
    if clic:
        s.calculo = {"hora": ahora, "motivo": "Recalculado con el botón «Calcular con estos datos»."}
    elif s.get("firma") is None:
        s.calculo = {"hora": ahora, "motivo": "Cálculo inicial con los datos cargados."}
    elif firma != s.firma:
        s.calculo = {"hora": ahora, "motivo": "Se detectaron cambios en tus datos y se recalculó automáticamente."}
    s.firma = firma


def _caja_confirmacion(res, con_nota=True):
    """Cuadro verde que confirma con qué datos y a qué hora se calcularon los resultados."""
    s = st.session_state
    c = s.calculo
    x0 = ", ".join(f"{v:g}" for v in s.x0)
    st.success(
        f"**Resultados actualizados a las {c['hora']:%H:%M:%S}** (hora de Bolivia). {c['motivo']}\n\n"
        f"**Datos usados:** escenario **{s.scenario}** · tolerancia {s.tol:.0e} · precondicionador "
        f"{res['precond'].name} · máximo {int(s.max_iter)} iteraciones · x₀ = [{x0}]\n\n"
        f"**Resultado:** " + " · ".join(f"{SIMBOLOS[i]} = {res['gcp']['x'][i]:.4f}" for i in range(4)) +
        f"\n\n**GC:** {res['gc']['iterations']} iteraciones · **GCP:** {res['gcp']['iterations']} iteraciones · "
        f"**Residuo final:** {res['resid_gcp']:.2e} · "
        f"**¿Convergió?** {'Sí' if res['gcp']['converged'] else 'No'}")
    if con_nota:
        st.caption("x₀, la tolerancia y el precondicionador cambian el camino que sigue el algoritmo, no la solución. "
                   "Para cambiar la solución modifica A o b (o pon iteraciones máximas muy bajas).")


def entrada():
    st.title("Entrada de datos")
    s = st.session_state

    
    st.caption("Modifica los valores de las tablas para actualizar los resultados.")
    
    st.info(
        "1. Selecciona un escenario en el módulo 5. (Tres escenarios)\n\n"
        "2. Edita las casillas de las tablas. (doble click para editar las casillas)\n\n"
        "3. Los resultados se actualizarán automáticamente."
    )
    
    st.caption(
        f"Escenario activo: **{s.scenario}**. "
        "Al modificar los datos, el escenario pasará a ser Personalizado."
    )

    arriba = st.container()
    with arriba:
        clic_arriba = st.button("Calcular con estos datos", key="calc_arriba", type="primary")
    
    st.subheader("Datos del sistema Ax = b")
    
    st.caption(
        f"Las variables_ **x₁·Desarrollo, x₂·Infraestructura, x₃·Ciberseguridad y x₄·Soporte** _representan los presupuestos "
        f"asignados a cada unidad, expresados en {UNIDAD}."
    )

    st.write("Las casillas **editables** son todas las de las dos tablas de abajo (doble click para editar las casillas). "
             "Cada **fila** es la ecuación de una unidad.")
    dfA = pd.DataFrame(s.A_base, index=FILAS, columns=COLS)
    dfb = pd.DataFrame({"b": s.b_base}, index=FILAS)
    cfgA = {COLS[i]: st.column_config.NumberColumn(
        COLS[i], format="%g", step=0.1,
        help=f"Coeficientes de {NOMBRES[i]} en cada ecuación. En la diagonal es su costo propio; "
             "fuera de la diagonal, su interacción con otra unidad.") for i in range(4)}
    cfgb = {"b · Beneficio": st.column_config.NumberColumn(
        "b · Beneficio", format="%g", step=1.0,
        help="Beneficio marginal esperado de invertir en esa unidad. Súbelo para simular que esa unidad "
             "necesita más presupuesto.")}
    c1, c2 = st.columns([2.6, 1])
    with c1:
        st.markdown("**Matriz A** · costos propios e interacciones")
        eA = st.data_editor(dfA, key=f"edA_{s.editor_v}", column_config=cfgA, width="stretch")
    with c2:
        st.markdown("**Vector b** · beneficio esperado")
        eb = st.data_editor(dfb, key=f"edb_{s.editor_v}", column_config=cfgb, width="stretch")
    A_new, b_new = eA.to_numpy(dtype=float), eb.iloc[:, 0].to_numpy(dtype=float)
    if not (np.array_equal(A_new, s.A, equal_nan=True) and np.array_equal(b_new, s.b, equal_nan=True)):
        s.A, s.b = A_new, b_new
        if not (np.array_equal(A_new, s.A_base, equal_nan=True) and np.array_equal(b_new, s.b_base, equal_nan=True)):
            s.scenario = "Personalizado"

    v = validate_spd(s.A)
    if v["ok"]:
        st.success("Los datos son válidos: A es simétrica y definida positiva. Los métodos se pueden ejecutar.")
    else:
        for e in v["errors"]:
            st.error(e)
        if not v["symmetric"] and np.all(np.isfinite(s.A)):
            st.button("Hacer A simétrica automáticamente (copiar la parte superior hacia abajo)",
                      on_click=_hacer_simetrica)
    st.button("Restablecer los datos del escenario cargado", on_click=lambda: load_scenario(
        s.scenario if s.scenario in SCENARIOS else "Ideal"))


   
    # Configuración del método
    st.subheader("Configuración del método")

    st.caption(
        "Modifica los parámetros que utilizarán los algoritmos "
        "para calcular los presupuestos."
    )

    st.markdown("**Punto inicial (x₀)**")

    cols = st.columns(4)

    for i, col in enumerate(cols):
        s.setdefault(f"x0w_{i}", float(s.x0[i]))
        s.x0[i] = col.number_input(
            f"{SIMBOLOS[i]} · {CORTOS[i]}",
            key=f"x0w_{i}",
            format="%.4f",
            help="Valor inicial de esta variable. Normalmente se utiliza 0."
        )

    st.markdown("**Parámetros de ejecución**")

    p1, p2, p3, p4 = st.columns(4)

    s.setdefault("tol_w", s.tol)
    s.tol = p1.select_slider(
        "Tolerancia",
        options=[1e-4, 1e-6, 1e-8, 1e-10],
        key="tol_w",
        format_func=lambda v: f"{v:.0e}",
        help="Precisión exigida para detener el algoritmo."
    )

    s.setdefault("maxit_w", int(s.max_iter))
    s.max_iter = p2.number_input(
        "Iteraciones máximas",
        min_value=1,
        max_value=10000,
        key="maxit_w"
    )

    s.setdefault("precond_w", s.precond)
    s.precond = p3.selectbox(
        "Precondicionador",
        PRECONDICIONADORES,
        key="precond_w",
        help="Permite mejorar las propiedades del sistema."
    )

    if s.precond == "SSOR":
        s.setdefault("omega_w", float(s.omega))
        s.omega = p4.slider(
            "ω (SSOR)",
            0.1,
            1.9,
            step=0.1,
            key="omega_w",
            help="Parámetro de relajación del precondicionador."
        )

    st.button("Restablecer configuración predeterminada", key="reset_config",
              on_click=_restablecer_configuracion,
              help="Vuelve x₀ = 0, tolerancia 10⁻⁶, 1000 iteraciones máximas y precondicionador Jacobi. "
                   "No modifica las tablas de A y b.")
    st.caption("Valores predeterminados: x₀ = 0 · tolerancia 10⁻⁶ · 1000 iteraciones máximas · precondicionador Jacobi.")

    st.subheader("Resultado con tus datos")
    res = run_current()
    if not res["valid"]:
        with arriba:
            st.warning("No se pudo recalcular porque los datos no son válidos. Revisa el aviso en rojo más abajo.")
        return show_invalid(res)
    m = st.columns(4)
    for i, col in enumerate(m):
        col.metric(f"{SIMBOLOS[i]} · {CORTOS[i]}", f"{res['gcp']['x'][i]:.4f}")
    st.caption(f"Presupuesto óptimo calculado con el GCP ({res['precond'].name}), en {UNIDAD}. "
               f"Iteraciones: GC = {res['gc']['iterations']}, GCP = {res['gcp']['iterations']}. "
               "Mira los demás módulos del menú para ver el procedimiento, los errores y la comparación de métodos.")
    clic_abajo = st.button("Calcular con estos datos", key="calc_abajo", type="primary")
    _registrar_calculo(clic_arriba or clic_abajo)
    with arriba:
        _caja_confirmacion(res, con_nota=False)
    _caja_confirmacion(res)


# 5 ─────────────────────────────────────────────────────────────
def nombre_visible(nombre):
    texto = nombre.lower()

    if "ideal" in texto:
        return "Escenario ideal"
    elif "estres" in texto or "estrés" in texto:
        return "Escenario bajo estrés"
    elif "condicionado" in texto:
        return "Escenario mal condicionado"

    return nombre


def escenarios():
    st.title("Análisis de escenarios")

    st.write(
        "Evalúa el comportamiento de los métodos numéricos "
        "en tres condiciones diferentes del sistema."
    )

    # Selección de escenarios
    cols = st.columns(3)

    for col, nombre in zip(cols, ESCENARIOS):
        activo = st.session_state.scenario == nombre
        nombre_mostrar = nombre_visible(nombre)

        with col:
            st.markdown(
                f'<div class="ficha{" activa" if activo else ""}>'
                f'<h4>{nombre_mostrar}</h4>'
                f'<p>{SCENARIOS[nombre]["resumen"]}</p></div>',
                unsafe_allow_html=True
            )

            st.button(
                nombre_mostrar,
                key=f"btn_{nombre}",
                on_click=load_scenario,
                args=(nombre,),
                disabled=activo,
                width="stretch"
            )

    st.divider()

    # Datos del escenario seleccionado
    s = st.session_state

    st.subheader(f"Datos del escenario: {nombre_visible(s.scenario)}")

    if s.scenario in SCENARIOS:
        nota(SCENARIOS[s.scenario]["justificacion"])

    c1, c2 = st.columns([2.2, 1])

    with c1:
        st.markdown("**Matriz A**")
        tabla(df_matrix(s.A), hide_index=False)

    with c2:
        st.markdown("**Vector b**")
        tabla(
            pd.DataFrame({"Beneficio": s.b}, index=NOMBRES),
            hide_index=False
        )

    # Resultados del escenario
    res = run_current()

    if not res["valid"]:
        return show_invalid(res)

    st.subheader("Resultados del escenario")

    m = st.columns(4)

    m[0].metric("Condición de A · κ(A)", f"{res['kappa_A']:.4g}")
    m[1].metric("Condición de M⁻¹A", f"{res['kappa_M']:.4g}")
    m[2].metric("Iteraciones GC", res["gc"]["iterations"])
    m[3].metric("Iteraciones GCP", res["gcp"]["iterations"])

    st.markdown("**Solución calculada mediante GCP**")

    tabla(pd.DataFrame({
        "Unidad": NOMBRES,
        "Presupuesto óptimo": res["gcp"]["x"]
    }), hide_index=True)

    st.markdown("**Interpretación financiera**")
    st.info(SCENARIOS[s.scenario]["interpretacion"])

    st.divider()

    st.subheader("Análisis de los escenarios")

    steps.pasos_escenarios()

    
    st.divider()

    st.subheader("Comparación de resultados de los tres escenarios")

    st.write(
        "Se comparan los resultados obtenidos mediante los métodos "
        "de Gradientes Conjugados (GC) y Gradientes Conjugados "
        "Precondicionados (GCP) en cada uno de los tres escenarios."
    )

    resultados_comparacion = []

    for nombre in ESCENARIOS:
        resultado = run_scenario(nombre)

        if not resultado["valid"]:
            st.warning(
                f"No fue posible calcular el escenario: {nombre}"
            )
            continue

        resultados_comparacion.append({
            "Escenario": nombre_visible(nombre),
            "κ(A)": resultado["kappa_A"],
            "κ(M⁻¹A)": resultado["kappa_M"],
            "Iteraciones GC": resultado["gc"]["iterations"],
            "Iteraciones GCP": resultado["gcp"]["iterations"]
        })

    if resultados_comparacion:
        df_comparacion = pd.DataFrame(resultados_comparacion)

        tabla(
            df_comparacion,
            hide_index=True,
            fmt={
                "κ(A)": "{:.4g}",
                "κ(M⁻¹A)": "{:.4g}"
            }
        )




# 6 ─────────────────────────────────────────────────────────────
def referencia():
    st.title("Solución de referencia: método de Cholesky")

    st.write(
        "Se obtiene una solución de referencia del sistema Ax = b "
        "mediante la factorización de Cholesky. Esta solución permitirá "
        "evaluar la precisión de los métodos GC y GCP."
    )

    res = run_current()

    if not res["valid"]:
        return show_invalid(res)

    s = st.session_state

    st.caption(f"Escenario activo: {s.scenario}")

    st.subheader("1. Factorización de Cholesky")

    st.markdown(
        "La matriz A se descompone en el producto de una matriz "
        "triangular inferior L y su transpuesta:"
    )

    st.latex(r"A = LL^T")

    st.markdown(
        "Después se resuelven dos sistemas triangulares: "
        r"$Ly=b$ mediante sustitución hacia adelante y "
        r"$L^Tx=y$ mediante sustitución hacia atrás."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Matriz triangular inferior L**")
        tabla(
            pd.DataFrame(res["L"], index=NOMBRES, columns=SIMBOLOS),
            hide_index=False
        )

    with c2:
        st.markdown("**Vector intermedio y solución de referencia**")
        tabla(
            pd.DataFrame({
                "Vector y": res["y"],
                f"Solución x ({UNIDAD})": res["x_ref"]
            }, index=NOMBRES),
            hide_index=False
        )

    st.divider()

    st.subheader("2. Verificación de los resultados")

    err_fact = np.max(np.abs(res["L"] @ res["L"].T - s.A))

    err_lib = np.max(
        np.abs(res["x_ref"] - np.linalg.solve(s.A, s.b))
    )

    m = st.columns(2)

    m[0].metric(
        "Error de factorización",
        f"{err_fact:.2e}"
    )

    m[1].metric(
        "Diferencia frente a NumPy",
        f"{err_lib:.2e}"
    )

    nota(
        "Un error de factorización cercano a cero indica que "
        "L · Lᵀ reproduce correctamente la matriz A. La solución "
        "obtenida mediante Cholesky se utiliza como referencia "
        "para comparar la precisión de GC y GCP."
    )

    st.divider()

    st.subheader("Desarrollo del método de Cholesky")
    steps.pasos_cholesky()


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
    tabla(pd.DataFrame({
        "Método": ["Solución de referencia (Cholesky)", "GC (sin precondicionar)", f"GCP ({res['precond'].name})"],
        "Vector solución": [fmt_vec(res["x_ref"]), fmt_vec(gc["x"]), fmt_vec(gcp["x"])],
        "Iteraciones": ["—", str(gc["iterations"]), str(gcp["iterations"])],
        "Residuo relativo final": ["—", f"{res['resid_gc']:.2e}", f"{res['resid_gcp']:.2e}"],
        "¿Convergió?": ["—", "Sí" if gc["converged"] else "No", "Sí" if gcp["converged"] else "No"],
    }))
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
    st.divider()
    steps.pasos_gcp()


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
    tabla(df, fmt=fmt)
    if df["Error relativo"].isna().any():
        st.caption("Si xref,i = 0, el error relativo no está definido para esa componente.")
    m = st.columns(3)
    m[0].metric("Residuo relativo ‖b − Ax‖/‖b‖ (GCP)", f"{res['resid_gcp']:.2e}")
    m[1].metric("Máximo error absoluto", f"{df['Error absoluto'].max():.2e}")
    m[2].metric("¿Convergió el GCP?", "Sí" if res["gcp"]["converged"] else "No")
    steps.comparacion_metodos()
    st.divider()
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
    comparativa = pd.DataFrame(filas, index=ESCENARIOS).T
    comparativa.index.name = "Indicador"
    st.table(comparativa.astype(str))
    st.divider()
    steps.pasos_errores()


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
