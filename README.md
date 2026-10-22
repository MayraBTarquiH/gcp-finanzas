# Presupuesto de TI con Gradientes Conjugados Precondicionados

- **Estudiante:** Tarqui Humerez Mayra Belen
- **Mención:** Ingeniería de Sistemas
- **Enlace a la página:** https://gcp-finanzas-ieukzzzemjwnv2dlofrvhy.streamlit.app/

## De qué trata este trabajo?

Esta es una página web interactiva que resuelve un problema de distribución de presupuesto. Una empresa tiene que repartir su
presupuesto de tecnología entre 4 áreas: desarrollo de software, infraestructura y nube, ciberseguridad
y soporte y capacitación.

El reparto se puede plantear como un sistema de ecuaciones **Ax = b**, donde A es una matriz simétrica
y definida positiva. Este problema se resolvió con el método de Gradientes Conjugados Precondicionado del artículo de
Suñagua (2020), y lo comparé con el Gradiente Conjugado simple y con una solución de referencia
calculada con Cholesky.

Se trabajó en tres escenarios con las mismas variables:

- **Ideal:** los costos de las áreas son parecidos y el sistema está bien condicionado.
- **Bajo estrés:** sube el vector b por inflación y más incidentes de seguridad.
- **Mal condicionado:** los costos tienen escalas muy distintas, y ahí uso el precondicionador de Jacobi.

## Qué hace la pagina

- Revisa que la matriz A sea simétrica y definida positiva antes de calcular.
- Calcula la solución de referencia con Cholesky (se programó sin usar librerías).
- Resuelve con Gradientes Conjugados y con Gradientes Conjugados Precondicionado.
- Calcula los errores por variable, el residuo relativo y los números de condición κ(A) y κ(M⁻¹A).
- Tiene un módulo de pruebas que se ejecuta dentro de la misma página.

## Instalación y ejecución local

```bash
git clone https://github.com/MayraBTarquiH/gcp-finanzas.git
cd gcp-finanzas
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Pruebas automáticas: `python -m pytest`

## Cómo está organizado el código

```
app.py            archivo principal y menú de la página
core/             toda la parte matemática
  scenarios.py    las matrices A y los vectores b de los 3 escenarios
  validation.py   revisa si A es simétrica y definida positiva
  direct.py       Cholesky (solución de referencia)
  cg.py           Gradientes Conjugados y precondicionadores (Jacobi y SSOR)
  metrics.py      errores, residuo y números de condición
  analysis.py     junta todos los cálculos de un escenario
  qa.py           las pruebas del módulo QA
ui/               las pantallas de la página
tests/            pruebas automáticas
```

## Referencias

- Suñagua S., P. (2020). Método de Gradientes Conjugados Precondicionado. *Revista Boliviana de Matemática – UMSA*, 04, 2–7.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
