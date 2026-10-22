import numpy as np
from core import qa
from core.cg import make_preconditioner, pcg
from core.scenarios import cargar


def test_qa_todas_aprobadas():
    for fila in qa.run_all():
        assert fila["Conclusión"] == "Aprobada", fila


def test_gc_converge_en_a_lo_sumo_n_iteraciones():
    A, b = cargar("Ideal")
    r = pcg(A, b, precond=make_preconditioner(A, "Ninguno"))
    assert r["converged"] and r["iterations"] <= 4


def test_ssor_converge():
    A, b = cargar("Mal condicionado")
    r = pcg(A, b, precond=make_preconditioner(A, "SSOR", 1.0), tol=1e-10)
    assert r["converged"]
    assert np.allclose(r["x"], np.linalg.solve(A, b), atol=1e-5)
