
# Presupuesto de TI con Gradientes Conjugados Precondicionados

- **Estudiante:** Tarqui Humerez Mayra Belen
- **Mención:** Ingeniería de Sistemas
- **Aplicación web:** https://gcp-finanzas-ieukzzzemjwnv2dlofrvhy.streamlit.app/
- **Repositorio:** https://github.com/MayraBTarquiH/gcp-finanzas

## ¿De qué trata este proyecto?

Este proyecto consiste en una aplicación web interactiva que permite modelar y resolver un problema de distribución del presupuesto de tecnología de una empresa entre cuatro áreas:

- Desarrollo de software.
- Infraestructura y nube.
- Ciberseguridad.
- Soporte y capacitación.

El problema se representa mediante un sistema de ecuaciones lineales **Ax = b**, donde A es una matriz simétrica y definida positiva, x representa los recursos asignados a cada área y b contiene los valores del modelo.

Para resolver el sistema se implementó el método de **Gradientes Conjugados Precondicionado (GCP)**, tomando como referencia el artículo de Suñagua (2020). Sus resultados se comparan con el método de Gradientes Conjugados (GC) y con una solución de referencia obtenida mediante la factorización de Cholesky.

El proyecto contempla tres escenarios:

- **Ideal:** representa una situación estable, donde los costos propios de las áreas son similares y las interacciones son relativamente bajas.
- **Bajo estrés:** representa un escenario en el que aumenta el vector b debido a factores como la inflación y los incidentes de seguridad. La matriz A se mantiene igual que en el escenario ideal.
- **Mal condicionado:** representa un sistema cuyos coeficientes tienen escalas muy diferentes, permitiendo analizar el efecto del precondicionamiento de Jacobi.

Los valores del modelo se expresan en decenas de miles de bolivianos (Bs).

## Funcionalidades principales

- Verifica que la matriz A sea simétrica y definida positiva antes de realizar los cálculos.
- Calcula una solución de referencia mediante una implementación propia del método de Cholesky.
- Resuelve el sistema mediante Gradientes Conjugados y Gradientes Conjugados Precondicionados.
- Permite trabajar con tres escenarios y modificar los datos de entrada.
- Calcula los errores por variable, el residuo relativo y los números de condición κ(A) y κ(M⁻¹A).
- Compara el comportamiento de los métodos GC y GCP.
- Incluye un módulo de pruebas de calidad (QA) para verificar el funcionamiento de los procedimientos implementados.
- Presenta conclusiones técnicas y financieras de los resultados obtenidos.

## Instalación y ejecución local

### Requisitos

- Python instalado.
- Git.
- Visual Studio Code (opcional).

### Pasos de instalación

Clonar el repositorio:

```bash
git clone https://github.com/MayraBTarquiH/gcp-finanzas.git
```

Ingresar a la carpeta del proyecto:

```bash
cd gcp-finanzas
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno virtual:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

### Ejecución de pruebas automáticas

Para ejecutar las pruebas automatizadas del proyecto:

```bash
python -m pytest
```

Además, la aplicación cuenta con un módulo QA que permite ejecutar pruebas directamente desde su interfaz web.

## Organización del código

```text
gcp-finanzas/
│
├── app.py                  # Archivo principal y menú de navegación
│
├── core/                   # Implementación matemática
│   ├── scenarios.py        # Matrices A y vectores b de los escenarios
│   ├── validation.py       # Validación de la matriz
│   ├── direct.py           # Método de Cholesky
│   ├── cg.py               # GC y métodos de precondicionamiento
│   ├── metrics.py          # Errores, residuos y números de condición
│   ├── analysis.py         # Cálculos y análisis de escenarios
│   └── qa.py               # Pruebas del módulo QA
│
├── ui/                     # Interfaz de usuario
│   ├── pages.py            # Pantallas y módulos de la aplicación
│   └── steps.py            # Procedimientos paso a paso
│
├── tests/                  # Pruebas automatizadas
│
├── requirements.txt        # Dependencias del proyecto
│
└── README.md               # Documentación del proyecto
```

## Referencias bibliográficas

- Suñagua S., P. (2020). *Método de Gradientes Conjugados Precondicionado*. Revista Boliviana de Matemática – UMSA, 04, 2–7.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
