import streamlit as st

from ui.style import CSS
from ui.common import init_state
from ui import pages

st.set_page_config(page_title="GCP en finanzas", page_icon="📈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
init_state()

MODULOS = {
    "1. Inicio": pages.inicio,
    "2. Caso de estudio": pages.caso,
    "3. Formulación": pages.formulacion,
    "4. Entrada de datos": pages.entrada,
    "5. Tres escenarios": pages.escenarios,
    "6. Solución de referencia": pages.referencia,
    "7. GCP vs GC": pages.gcp_vs_gc,
    "8. Análisis de errores": pages.errores,
    "9. Módulo QA": pages.qa_page,
    "10. Conclusiones y referencias": pages.conclusiones,
}

with st.sidebar:
    st.markdown('<div class="marca">Presupuesto de TI</div>'
                '<div class="marca-sub">Gradientes conjugados precondicionados</div>', unsafe_allow_html=True)
    eleccion = st.radio("Módulos", list(MODULOS), label_visibility="collapsed")
    st.markdown(f'<div class="estado">Escenario activo<br><b>{st.session_state.scenario}</b></div>',
                unsafe_allow_html=True)

MODULOS[eleccion]()
