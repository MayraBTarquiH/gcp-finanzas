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
      <h1>Presupuesto de Tecnologías de la Información (TI) con gradientes conjugados precondicionados</h1>
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

    if s.scenario in SCENARIOS:
        st.info(SCENARIOS[s.scenario]["interpretacion"])
    else:
        st.info(
            "Estás trabajando con un escenario personalizado. "
            "Los resultados se calculan utilizando los valores "
            "introducidos manualmente en la matriz A y el vector b."
        )

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

    st.subheader("3. Desarrollo del método de Cholesky")
    steps.pasos_cholesky()



# 7 ─────────────────────────────────────────────────────────────
def gcp_vs_gc():
    st.title("Comparación: GCP vs GC")

    st.write(
        "En esta sección se comparan los métodos de Gradientes "
        "Conjugados (GC) y Gradientes Conjugados Precondicionados (GCP) "
        "para analizar su precisión y velocidad de convergencia."
    )

    res = run_current()

    if not res["valid"]:
        return show_invalid(res)

    s = st.session_state
    gc, gcp = res["gc"], res["gcp"]

    st.caption(
        f"Escenario activo: {s.scenario} · "
        f"Precondicionador: {res['precond'].name} · "
        f"Tolerancia: {s.tol:.0e}"
    )

    # 1. Descripción de los métodos
    st.subheader("1. ¿Cómo trabajan los métodos?")

    st.markdown(
        "- **GC:** resuelve el sistema lineal sin utilizar un precondicionador.\n"
        "- **GCP:** utiliza un precondicionador para mejorar las propiedades "
        "del sistema y favorecer la convergencia."
    )

    st.latex(r"Mz_k = r_k")

    st.caption(
        "En GCP, el sistema anterior se resuelve en cada iteración para "
        "obtener el vector auxiliar zₖ."
    )

    # 2. Comparación de resultados
    st.divider()
    st.subheader("2. Comparación de resultados")

    tabla(pd.DataFrame({
        "Método": [
            "Referencia (Cholesky)",
            "Gradientes Conjugados (GC)",
            f"Gradientes Conjugados Precondicionados (GCP)"
        ],
        "Vector solución": [
            fmt_vec(res["x_ref"]),
            fmt_vec(gc["x"]),
            fmt_vec(gcp["x"])
        ],
        "Iteraciones": [
            "—",
            str(gc["iterations"]),
            str(gcp["iterations"])
        ],
        "Residuo relativo": [
            "—",
            f"{res['resid_gc']:.2e}",
            f"{res['resid_gcp']:.2e}"
        ],
        "Convergencia": [
            "Referencia",
            "Sí" if gc["converged"] else "No",
            "Sí" if gcp["converged"] else "No"
        ]
    }))

    # 3. Números de condición
    st.divider()
    st.subheader("3. Comparación de los números de condición")

    m = st.columns(2)

    m[0].metric(
        "Condición de A · κ(A)",
        f"{res['kappa_A']:.4g}"
    )

    m[1].metric(
        "Condición de M⁻¹A · κ(M⁻¹A)",
        f"{res['kappa_M']:.4g}",
        delta=f"{res['kappa_M'] / res['kappa_A'] - 1:+.1%} respecto a κ(A)",
        delta_color="inverse"
    )

    st.write(
        "Un número de condición menor indica que el sistema está "
        "mejor condicionado y, en general, es menos sensible a "
        "pequeñas perturbaciones."
    )

    # 4. Gráfica de convergencia
    st.divider()
    st.subheader("4. Comportamiento de convergencia")

    tol_line = pd.DataFrame({"y": [s.tol]})

    hist = pd.concat([
        pd.DataFrame({
            "Iteración": range(len(gc["history"])),
            "Residuo relativo": np.maximum(gc["history"], 1e-18),
            "Método": "GC"
        }),
        pd.DataFrame({
            "Iteración": range(len(gcp["history"])),
            "Residuo relativo": np.maximum(gcp["history"], 1e-18),
            "Método": "GCP"
        })
    ])

    base = alt.Chart(hist).mark_line(point=True).encode(
        x=alt.X(
            "Iteración:Q",
            axis=alt.Axis(tickMinStep=1),
            title="Número de iteraciones"
        ),
        y=alt.Y(
            "Residuo relativo:Q",
            scale=alt.Scale(type="log"),
            title="Residuo relativo (escala logarítmica)"
        ),
        color=alt.Color(
            "Método:N",
            scale=alt.Scale(range=["#C98A1B", "#0F8B8D"])
        ),
        tooltip=["Método", "Iteración", "Residuo relativo"]
    )

    rule = alt.Chart(tol_line).mark_rule(
        strokeDash=[5, 4],
        color="#B3261E"
    ).encode(y="y:Q")

    st.altair_chart(base + rule, width="stretch")

    st.caption(
        "La línea roja representa la tolerancia establecida. "
        "El eje vertical utiliza una escala logarítmica para facilitar "
        "la observación de la disminución del residuo."
    )

    # 5. Interpretación
    st.divider()
    st.subheader("5. Interpretación de los resultados")

    nota(
        "En un sistema de n = 4 variables, el método de Gradientes "
        "Conjugados puede alcanzar la solución en un máximo de 4 "
        "iteraciones en aritmética exacta. En la práctica, los errores "
        "de redondeo y la tolerancia pueden influir en las iteraciones "
        "observadas. El precondicionador busca mejorar el condicionamiento "
        "del sistema y favorecer la convergencia."
    )

    # 6. Procedimiento matemático
    st.divider()
    st.subheader("Desarrollo del método GCP")

    steps.pasos_gcp()



# 8 ─────────────────────────────────────────────────────────────
def errores():
    st.title("Análisis de errores")

    st.write(
        "En esta sección se evalúa la precisión de la solución obtenida "
        "mediante Gradientes Conjugados Precondicionados (GCP), "
        "comparándola con la solución numérica de referencia calculada "
        "mediante el método de Cholesky."
    )

    res = run_current()
    if not res["valid"]:
        return show_invalid(res)

    # 1. Errores por variable del escenario activo
    st.subheader("1. Errores por variable")

    st.write(
        "Se compara el presupuesto calculado por GCP con el presupuesto "
        "obtenido mediante Cholesky para cada unidad de TI."
    )

    df = res["errors"]

    fmt = {
        "por referencia": "{:.6f}",
        "por GCP": "{:.6f}",
        "Error absoluto": "{:.2e}",
        "Error relativo": "{:.2e}",
        "Error porcentual (%)": "{:.2e}"
    }

    tabla(df, fmt=fmt)

    st.caption(
        "La solución de referencia corresponde al método de Cholesky "
        "y la solución GCP es el resultado del método precondicionado."
    )

    if df["Error relativo"].isna().any():
        st.caption(
            "Cuando una componente de la solución de referencia es cero, "
            "el error relativo no está definido para esa componente."
        )

    st.divider()

    # 2. Indicadores de precisión y convergencia
    st.subheader("2. Indicadores de precisión y convergencia")

    m = st.columns(3)

    m[0].metric(
        "Residuo relativo final (GCP)",
        f"{res['resid_gcp']:.2e}"
    )

    m[1].metric(
        "Máximo error absoluto",
        f"{df['Error absoluto'].max():.2e}"
    )

    m[2].metric(
        "Convergencia del GCP",
        "Sí" if res["gcp"]["converged"] else "No"
    )

    st.markdown("**¿Cómo se interpretan estos indicadores?**")

    st.markdown(
        "- **Error absoluto:** mide la diferencia entre la solución "
        "de referencia y la solución calculada por GCP.\n"
        "- **Error relativo:** expresa el error absoluto en relación "
        "con el valor de referencia.\n"
        "- **Error porcentual:** representa el error relativo en porcentaje.\n"
        "- **Residuo relativo:** mide qué tan bien la solución calculada "
        "satisface el sistema original Ax = b."
    )

    st.latex(
        r"E_a = |x_{\text{referencia}}-x_{\text{GCP}}|"
    )
    st.latex(
        r"E_r = \frac{E_a}{|x_{\text{referencia}}|}"
    )
    st.latex(
        r"E_p = E_r \times 100"
    )
    st.latex(
        r"\text{Residuo relativo} = "
        r"\frac{\|b-Ax_{\text{GCP}}\|}{\|b\|}"
    )

    st.divider()

    # 3. Comparación del comportamiento de los métodos
    st.subheader("3. Comparación de los métodos numéricos")
    steps.comparacion_metodos()

    st.divider()

    # 4. Comparación de los tres escenarios
    st.subheader("4. Comparación de resultados de los tres escenarios")

    st.write(
        "La siguiente tabla reúne los resultados de los tres escenarios "
        "para comparar la precisión, el número de iteraciones y el "
        "condicionamiento del sistema."
    )

    st.caption(
        "Los resultados se calculan con los parámetros actuales: "
        "vector inicial, tolerancia y precondicionador."
    )

    filas = {
        "Solución de referencia": [],
        "Solución GCP": [],
        "Error absoluto (Ea)": [],
        "Error porcentual (Ep)": [],
        "Iteraciones GC": [],
        "Iteraciones GCP": [],
        "Número de condición κ(A)": [],
        "Número de condición κ(M⁻¹A)": [],
        "Residuo relativo final GCP": [],
        "Convergencia GCP": []
    }

    interpretaciones = []

    for nombre in ESCENARIOS:
        r = run_scenario(nombre)

        if not r["valid"]:
            st.warning(
                f"No fue posible calcular los resultados del escenario: {nombre}."
            )
            continue

        e = r["errors"]

        filas["Solución de referencia"].append(
            fmt_vec(r["x_ref"])
        )
        filas["Solución GCP"].append(
            fmt_vec(r["gcp"]["x"])
        )
        filas["Error absoluto (Ea)"].append(
            fmt_vec(e["Error absoluto"], sci=True)
        )
        filas["Error porcentual (Ep)"].append(
            fmt_vec(e["Error porcentual (%)"], sci=True)
        )
        filas["Iteraciones GC"].append(
            r["gc"]["iterations"]
        )
        filas["Iteraciones GCP"].append(
            r["gcp"]["iterations"]
        )
        filas["Número de condición κ(A)"].append(
            f"{r['kappa_A']:.4g}"
        )
        filas["Número de condición κ(M⁻¹A)"].append(
            f"{r['kappa_M']:.4g}"
        )
        filas["Residuo relativo final GCP"].append(
            f"{r['resid_gcp']:.2e}"
        )
        filas["Convergencia GCP"].append(
            "Sí" if r["gcp"]["converged"] else "No"
        )

        interpretaciones.append({
            "Escenario": nombre,
            "Interpretación": SCENARIOS[nombre]["interpretacion"]
        })

    if interpretaciones:
        # Tabla de resultados numéricos
        comparativa = pd.DataFrame(
            filas,
            index=[
                item["Escenario"] for item in interpretaciones
            ]
        ).T

        comparativa.index.name = "Indicador"

        tabla(comparativa.astype(str), hide_index=False)

        st.divider()

        # Interpretación financiera de cada escenario
        st.subheader("5. Interpretación de los escenarios")

        for item in interpretaciones:
            with st.expander(item["Escenario"]):
                st.write(item["Interpretación"])

    st.divider()

    # 6. Desarrollo del procedimiento
    st.subheader("Desarrollo del análisis de errores")
    steps.pasos_errores()


# 9 ─────────────────────────────────────────────────────────────
def qa_page():
    st.title("Pruebas de calidad del sistema (QA)")

    st.write(
        "En este módulo se ejecutan pruebas para verificar el "
        "funcionamiento de los métodos numéricos y comprobar que "
        "los resultados obtenidos sean correctos."
    )

    st.info(
        "Las pruebas se ejecutan directamente sobre el código de la "
        "aplicación. Los resultados no son datos escritos manualmente."
    )

    # Ejecución de las pruebas
    if st.button(
        "Ejecutar todas las pruebas" if "qa" not in st.session_state
        else "Volver a ejecutar las pruebas",
        type="primary",
        width="stretch"
    ):
        st.session_state.qa = qa.run_all()

    # Presentación de resultados
    if "qa" in st.session_state:

        rows = st.session_state.qa

        st.divider()
        st.subheader("Resultados de las pruebas")

        if not rows:
            st.warning("No se encontraron resultados de las pruebas.")
            return

        total = len(rows)
        aprobadas = sum(
            r["Conclusión"] == "Aprobada" for r in rows
        )
        no_aprobadas = total - aprobadas

        # Indicadores generales
        m = st.columns(3)

        m[0].metric(
            "Total de pruebas",
            total
        )

        m[1].metric(
            "Pruebas aprobadas",
            aprobadas
        )

        m[2].metric(
            "Pruebas no aprobadas",
            no_aprobadas
        )

        # Estado general
        if aprobadas == total:
            st.success(
                f"Todas las pruebas fueron aprobadas: {aprobadas} de {total}."
            )
        elif aprobadas > 0:
            st.warning(
                f"Se aprobaron {aprobadas} de {total} pruebas. "
                "Es necesario revisar las pruebas restantes."
            )
        else:
            st.error(
                "Ninguna prueba fue aprobada. "
                "Es necesario revisar los resultados."
            )

        # Tabla de resultados
        st.markdown("**Detalle de las pruebas ejecutadas**")

        df_resultados = pd.DataFrame(rows).set_index("Prueba")
        tabla(df_resultados, hide_index=False)

    else:
        st.divider()
        st.info(
            "Todavía no se ejecutaron las pruebas. "
            "Presiona el botón para comenzar."
        )

    st.divider()

    # Información complementaria
    st.subheader("Información sobre las pruebas")

    st.write(
        "Las pruebas permiten comprobar que los procedimientos "
        "implementados en la aplicación produzcan resultados correctos "
        "y que los métodos numéricos funcionen según lo esperado."
    )



# 10 ────────────────────────────────────────────────────────────
def conclusiones():
    st.title("Conclusiones y referencias")

    st.write(
        "En este módulo se presentan las conclusiones obtenidas a partir "
        "de la aplicación de los métodos de Gradientes Conjugados (GC) "
        "y Gradientes Conjugados Precondicionados (GCP) al problema "
        "de distribución del presupuesto de TI."
    )

    # Cálculo de resultados para los tres escenarios
    resultados = {
        nombre: run_scenario(nombre)
        for nombre in ESCENARIOS
    }

    todos_validos = all(
        resultado["valid"]
        for resultado in resultados.values()
    )

    st.divider()

    # 1. Conclusiones técnicas
    st.subheader("1. Conclusiones técnicas")

    if todos_validos:
        mal = resultados["Mal condicionado"]

        error_maximo = max(
            resultado["errors"]["Error absoluto"].max()
            for resultado in resultados.values()
        )

        st.markdown(
            f"- **Precisión de los métodos:** en los tres escenarios, "
            f"el GCP logró converger y obtuvo soluciones muy próximas "
            f"a las calculadas mediante Cholesky. El error absoluto "
            f"máximo fue de aproximadamente {error_maximo:.0e}.\n\n"
            f"- **Efecto del precondicionador:** en el escenario mal "
            f"condicionado, el número de condición disminuyó de "
            f"κ(A) = {mal['kappa_A']:.1f} a "
            f"κ(M⁻¹A) = {mal['kappa_M']:.3f}. Esto demuestra que "
            f"el precondicionador de Jacobi mejora considerablemente "
            f"el condicionamiento del sistema.\n\n"
            f"- **Número de iteraciones:** tanto GC como GCP necesitaron "
            f"{mal['gc']['iterations']} iteraciones en el escenario mal "
            f"condicionado. En este problema de cuatro variables, "
            f"la mejora del precondicionador se observa principalmente "
            f"en el número de condición y no en una reducción de "
            f"iteraciones.\n\n"
            f"- **Escenario bajo estrés:** el número de condición "
            f"permanece igual al del escenario ideal porque la matriz A "
            f"no cambia; únicamente se modifica el vector b."
        )
    else:
        st.warning(
            "No fue posible generar todas las conclusiones técnicas "
            "porque uno o más escenarios presentan resultados no válidos. "
            "Revisa los datos y vuelve a ejecutar los cálculos."
        )

    st.divider()

    # 2. Conclusiones financieras
    st.subheader("2. Conclusiones financieras")

    st.markdown(
        "- **Escenario ideal:** representa una situación estable, "
        "en la que los costos e interacciones entre las unidades de TI "
        "se mantienen equilibrados.\n\n"
        "- **Escenario bajo estrés:** el incremento de los valores de "
        "b representa mayores necesidades de presupuesto, especialmente "
        "en desarrollo y ciberseguridad. Esto modifica la asignación "
        "óptima sin alterar el condicionamiento de la matriz.\n\n"
        "- **Escenario mal condicionado:** las diferencias de escala "
        "entre las unidades generan un sistema numéricamente sensible. "
        "Por ello, aplicar un precondicionador permite mejorar el "
        "tratamiento numérico antes de calcular la asignación."
    )

    st.divider()

    # 3. Conclusión general
    st.subheader("3. Conclusión general")

    st.write(
        "El desarrollo de este trabajo permitió aplicar los métodos "
        "de Gradientes Conjugados y Gradientes Conjugados Precondicionados "
        "a un problema de distribución de presupuesto tecnológico. "
        "Los resultados demostraron que ambos métodos pueden obtener "
        "soluciones precisas para los escenarios planteados. Además, "
        "se comprobó que el precondicionamiento de Jacobi resulta "
        "especialmente útil cuando el sistema presenta un mal "
        "condicionamiento, ya que reduce considerablemente su número "
        "de condición. Por tanto, el uso de métodos numéricos y técnicas "
        "de precondicionamiento contribuye a resolver sistemas lineales "
        "y analizar problemas de asignación de recursos."
    )

    st.divider()

    # 4. Referencias bibliográficas
    st.subheader("4. Referencias bibliográficas")

    st.markdown(
        "- Suñagua S., P. (2020). *Método de Gradientes Conjugados "
        "Precondicionado*. Revista Boliviana de Matemática – UMSA, "
        "04, 2–7.\n\n"
        "- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* "
        "(4th ed.). Johns Hopkins University Press.\n\n"
        "- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* "
        "(2nd ed.). SIAM.\n\n"
        "- Datos: coeficientes ilustrativos definidos para este trabajo "
        "académico (ver módulo «Caso de estudio»)."
    )
