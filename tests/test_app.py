from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
MODULOS = ["1. Inicio", "2. Caso de estudio", "3. Formulación", "4. Entrada de datos",
           "5. Tres escenarios", "6. Solución de referencia", "7. GCP vs GC",
           "8. Análisis de errores", "9. Módulo QA", "10. Conclusiones y referencias"]


def test_todos_los_modulos_cargan_sin_errores():
    at = AppTest.from_file(APP, default_timeout=60).run()
    for m in MODULOS:
        at.sidebar.radio[0].set_value(m).run()
        assert not at.exception, (m, [e.value for e in at.exception])


def test_qa_se_ejecuta_desde_la_pagina():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("9. Módulo QA").run()
    at.button[0].click().run()
    assert not at.exception
    assert "5 de 5" in at.success[0].value


def test_matriz_invalida_no_colapsa():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.session_state.A[0, 1] = 9.0
    at.sidebar.radio[0].set_value("7. GCP vs GC").run()
    assert not at.exception
    assert at.error
