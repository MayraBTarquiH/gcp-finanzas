"""Pruebas de software del módulo QA (también las usa pytest)."""
import numpy as np
from .scenarios import SCENARIOS, cargar
from .analysis import analyze
from .validation import validate_spd
from .direct import solve_cholesky


def _fila(nombre, entrada, esperado, obtenido, ok):
    return {"Prueba": nombre, "Entrada": entrada, "Resultado esperado": esperado,
            "Resultado obtenido": obtenido, "Conclusión": "Aprobada" if ok else "Fallida"}


def prueba_carga_escenarios():
    partes, ok = [], True
    for nombre in SCENARIOS:
        A, b = cargar(nombre)
        r = analyze(A, b)
        bueno = (r["valid"] and r["gcp"]["converged"] and r["resid_gcp"] < 1e-6
                 and np.max(r["errors"]["Error absoluto"]) < 1e-4)
        ok &= bool(bueno)
        if r["valid"]:
            partes.append(f"{nombre}: {r['gcp']['iterations']} it., residuo {r['resid_gcp']:.1e}")
        else:
            partes.append(f"{nombre}: inválido")
    return _fila("1. Carga de escenarios", "Ideal, Bajo estrés y Mal condicionado",
                 "Los tres son simétricos y definidos positivos, GCP converge con residuo < 1e-6.",
                 "; ".join(partes), ok)


def prueba_matriz_invalida():
    A, b = cargar("Ideal")
    nosim = A.copy(); nosim[0, 1] = 5.0
    nodef = np.array([[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]], float)
    try:
        r1, r2 = analyze(nosim, b), analyze(nodef, b)
        v1, v2 = validate_spd(nosim), validate_spd(nodef)
        ok = (not r1["valid"]) and (not r2["valid"]) and bool(v1["errors"]) and bool(v2["errors"])
        obt = f"Sin colapso. No simétrica → «{v1['errors'][0][:60]}…». No definida positiva → «{v2['errors'][0][:60]}…»."
    except Exception as e:  # pragma: no cover
        ok, obt = False, f"Excepción no controlada: {e}"
    return _fila("2. Matriz no válida", "a₁₂ = 5 con a₂₁ = 2 (no simétrica); matriz simétrica con autovalor −1",
                 "El sistema detecta el error, avisa al usuario y no calcula.", obt, ok)


def prueba_edicion_dinamica():
    A, b = cargar("Ideal")
    r0 = analyze(A, b)
    A2, b2 = A.copy(), b.copy()
    A2[0, 0], b2[0] = 12.0, 65.0
    r1 = analyze(A2, b2)
    x_np = np.linalg.solve(A2, b2)
    cambio = abs(r1["x_ref"][0] - r0["x_ref"][0])
    err = np.max(np.abs(r1["gcp"]["x"] - x_np))
    ok = r1["valid"] and cambio > 1e-3 and err < 1e-5
    return _fila("3. Edición dinámica", "a₁₁: 10 → 12 y b₁: 60 → 65",
                 "La referencia, el GCP y los errores se recalculan y coinciden con una solución independiente.",
                 f"x₁ pasó de {r0['x_ref'][0]:.4f} a {r1['x_ref'][0]:.4f}; diferencia GCP vs numpy = {err:.1e}.", ok)


def prueba_precondicionamiento():
    A, b = cargar("Mal condicionado")
    r = analyze(A, b, precond="Jacobi")
    ig, ip = r["gc"]["iterations"], r["gcp"]["iterations"]
    ok = ip <= ig and r["kappa_M"] < r["kappa_A"]
    return _fila("4. Comparación de precondicionamiento", "Escenario Mal condicionado, Jacobi, tolerancia 1e-6",
                 "Iteraciones GCP ≤ iteraciones GC, y κ(M⁻¹A) < κ(A).",
                 f"GC: {ig} it.; GCP: {ip} it.; κ(A) = {r['kappa_A']:.1f}, κ(M⁻¹A) = {r['kappa_M']:.2f}.", ok)


def prueba_cholesky_propio():
    dif = 0.0
    for nombre in SCENARIOS:
        A, b = cargar(nombre)
        x, _, _ = solve_cholesky(A, b)
        dif = max(dif, np.max(np.abs(x - np.linalg.solve(A, b))))
    return _fila("5. Cholesky propio vs librería", "Los tres escenarios",
                 "Diferencia máxima < 1e-10 respecto a numpy.linalg.solve.",
                 f"Diferencia máxima = {dif:.1e}.", dif < 1e-10)


def run_all():
    return [prueba_carga_escenarios(), prueba_matriz_invalida(), prueba_edicion_dinamica(),
            prueba_precondicionamiento(), prueba_cholesky_propio()]
