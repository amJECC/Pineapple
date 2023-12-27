# PEP 257: Docstring conventions
"""
Este script contiene configuraciones y parámetros para un algoritmo genético.
"""

# PEP 8: Imports should usually be on separate lines
# Importar módulos necesarios

# PEP 8: Imports should be grouped in the following order:
# Standard library imports.
# Related third-party imports.
# Local application/library specific imports.

population_size = 140  # PEP 8: Limit all lines to a maximum of 79 characters
generations = 1000   # Número de generaciones

dmin = 0
dmax = 1

n = 3                      # Numero de variables
m = 3                      # Numero de funciones

distribution_index = 20  # Tamaño de la población para la selección
mutation_rate = 1/n      # Tasa de mutación
crossover_rate = 0.9

min_values = [0, 0, 0]  # Valores mínimos para las variables de decisión
max_values = [1, 1, 1]  # Valores máximos para las variables de decisión

nmig = 1               # Número de migraciones

TM = [
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0]
]  # Matriz de topología que define las conexiones entre islas

list_indicators = ["HV", "R2", "IGD", "e", "D"]  # Lista de indicadores