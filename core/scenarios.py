"""Datos del caso: presupuesto de TI repartido entre 4 unidades.

Las variables xᵢ son montos asignados, en decenas de miles de Bs.
El sistema Ax = b es la condición de óptimo de minimizar
f(x) = ½ xᵀAx − bᵀx  (costo total menos beneficio esperado).
"""
import numpy as np

UNIDAD = "decenas de miles de Bs"

VARIABLES = [
    {"simbolo": "x₁", "nombre": "Desarrollo de software",
     "significado": "Presupuesto asignado al equipo que construye y mantiene los sistemas propios."},
    {"simbolo": "x₂", "nombre": "Infraestructura y nube",
     "significado": "Presupuesto para servidores, redes y servicios en la nube."},
    {"simbolo": "x₃", "nombre": "Ciberseguridad",
     "significado": "Presupuesto para protección, monitoreo y respuesta a incidentes."},
    {"simbolo": "x₄", "nombre": "Soporte y capacitación",
     "significado": "Presupuesto para mesa de ayuda y formación del personal."},
]
NOMBRES = [v["nombre"] for v in VARIABLES]
SIMBOLOS = [v["simbolo"] for v in VARIABLES]

A_BASE = [[10, 2, 1, 1],
          [2, 8, 1, 2],
          [1, 1, 6, 1],
          [1, 2, 1, 7]]

A_MAL = [[1, 0.9, 0.5, 0.3],
         [0.9, 10, 4, 2],
         [0.5, 4, 100, 30],
         [0.3, 2, 30, 1000]]

SCENARIOS = {
    "Ideal": {
        "A": A_BASE, "b": [60, 50, 30, 35],
        "resumen": "Situación estable: costos propios parecidos e interacciones débiles.",
        "justificacion": (
            "Los costos propios de las cuatro unidades están en el mismo orden (6 a 10) y las "
            "interacciones entre ellas son pequeñas (1 a 2). La matriz es casi diagonal dominante, "
            "así que κ(A) resulta bajo."),
        "interpretacion": (
            "Con presupuesto estable y costos equilibrados, la asignación óptima es razonablemente "
            "pareja. El método converge rápido y el precondicionador casi no cambia el resultado "
            "porque no hay nada grave que corregir."),
    },
    "Bajo estrés": {
        "A": A_BASE, "b": [72, 55, 45, 38],
        "resumen": "Inflación y más incidentes de seguridad elevan el beneficio marginal requerido.",
        "justificacion": (
            "La estructura de costos e interacciones (A) no cambia, pero b sube: salarios de desarrollo "
            "+20 % (60 → 72), tarifas de nube +10 % (50 → 55), incidentes de seguridad +50 % (30 → 45) "
            "y soporte +8,6 % (35 → 38)."),
        "interpretacion": (
            "El shock solo desplaza el punto de equilibrio: ciberseguridad y desarrollo reciben "
            "bastante más presupuesto. El condicionamiento no cambia, porque A es la misma; "
            "el algoritmo responde igual de bien."),
    },
    "Mal condicionado": {
        "A": A_MAL, "b": [6, 40, 300, 2000],
        "resumen": "Costos de escalas muy distintas entre unidades: mal escalamiento diagonal.",
        "justificacion": (
            "Las unidades tienen estructuras de costo de magnitudes muy dispares (de 1 a 1000), "
            "por ejemplo un equipo pequeño frente a infraestructura de gran escala. Esto produce un "
            "mal escalamiento diagonal, por lo que corresponde el precondicionador de Jacobi."),
        "interpretacion": (
            "Cuando las escalas de costo son tan distintas, el sistema es muy sensible y el GC simple "
            "sufre. El precondicionador de Jacobi reescala cada unidad y baja κ de cientos a casi 1, "
            "aunque en un sistema de solo 4 variables la ventaja se nota más en κ que en iteraciones."),
    },
}


def cargar(nombre):
    s = SCENARIOS[nombre]
    return np.array(s["A"], dtype=float), np.array(s["b"], dtype=float)
