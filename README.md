# Presupuesto de TI con Gradientes Conjugados Precondicionados

Página web interactiva en Streamlit que resuelve un problema de asignación de presupuesto de TI entre
4 unidades con el método de **Gradientes Conjugados Precondicionado (GCP)** de Suñagua (2020), en tres
escenarios: ideal, bajo estrés y mal condicionado.

- **Estudiante:** Belen [completa tus apellidos]
- **Mención:** Ingeniería de Sistemas
- **Enlace a la página:** [pega aquí tu enlace de Streamlit Cloud]

## El problema

Cuatro unidades (desarrollo de software, infraestructura y nube, ciberseguridad, soporte y capacitación)
reciben un presupuesto xᵢ (en decenas de miles de Bs). El reparto óptimo minimiza
f(x) = ½ xᵀAx − bᵀx, cuya condición de óptimo es el sistema **Ax = b** con A simétrica y definida positiva.

| Escenario | Qué cambia |
|---|---|
| Ideal | Costos equilibrados, κ(A) bajo |
| Bajo estrés | Sube b (inflación, incidentes de seguridad); A no cambia |
| Mal condicionado | Costos de escalas muy distintas; precondicionador de Jacobi |

## Qué calcula

- Validación de que A es simétrica y definida positiva (Sylvester, autovalores, Cholesky).
- Solución de referencia por **Cholesky** programado a mano.
- **GC** (contraste) y **GCP** (Algoritmo 2 de Suñagua, «Mz») con Jacobi o SSOR.
- Errores absoluto, relativo y porcentual por variable, residuo relativo, κ(A) y κ(M⁻¹A).
- Módulo QA con pruebas que se ejecutan en la propia página.

## Instalación y ejecución local

```bash
git clone https://github.com/TU_USUARIO/gcp-finanzas.git
cd gcp-finanzas
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Pruebas automáticas: `python -m pytest`

## Arquitectura

```
app.py            Punto de entrada y menú de navegación
core/             Matemática, sin dependencia de Streamlit
  scenarios.py    Variables, A y b de los tres escenarios
  validation.py   Simetría y definida positiva
  direct.py       Cholesky y sustituciones (solución de referencia)
  cg.py           GC, GCP y precondicionadores (Jacobi, SSOR)
  metrics.py      Errores, residuo y números de condición
  analysis.py     Orquesta el cálculo de un escenario
  qa.py           Pruebas del módulo QA
ui/               Interfaz: un módulo de la página por función en pages.py
tests/            Pruebas con pytest
```

## Referencias

- Suñagua S., P. (2020). Método de Gradientes Conjugados Precondicionado. *Revista Boliviana de Matemática – UMSA*, 04, 2–7.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
