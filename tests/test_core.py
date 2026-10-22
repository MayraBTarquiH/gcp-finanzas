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


def test_traza_gcp_es_coherente():
    from core.analysis import analyze
    A, b = cargar("Ideal")
    r = analyze(A, b)
    it = r["gcp"]["trace"]["iters"]
    assert len(it) == r["gcp"]["iterations"]
    assert np.allclose(it[-1]["x"], r["gcp"]["x"])
    assert np.isclose(it[0]["alpha"], it[0]["rz"] / it[0]["pAp"])


def test_pasos_de_cholesky_tienen_una_linea_por_elemento():
    from core.analysis import analyze
    from core.direct import cholesky_steps, forward_steps, backward_steps
    A, b = cargar("Ideal")
    r = analyze(A, b)
    assert len(cholesky_steps(A, r["L"])) == 10          # 4 + 3 + 2 + 1 elementos de L
    assert len(forward_steps(r["L"], b, r["y"])) == 4
    assert len(backward_steps(r["L"], r["y"], r["x_ref"])) == 4
