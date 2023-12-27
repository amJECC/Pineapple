import numpy as np
import Config
import Funciones
import Operadores

from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import Hypervolume

# ------------------Parámetros---------------------------------
# Definición de parámetros desde el módulo Config
population_size = Config.population_size
distribution_index = Config.distribution_index
mutation_rate = Config.mutation_rate
crossover_rate = Config.crossover_rate
list_indicators = Config.list_indicators

dmin = Config.dmin
dmax = Config.dmax

n = Config.n

TM = Config.TM

nmig = Config.nmig
# -------------------------------------------------------------

# -------------Selección para el indicador HV--------------------
def C_HV(a, A):
    """
    Función para calcular la contribución al indicador Hypervolume (HV) de una solución 'a' en un conjunto 'A'.

    Parameters:
    - a: Solución individual
    - A: Conjunto de soluciones

    Returns:
    - Contribución al indicador HV
    """
    Z = np.array([2, 2, 2])
    I = Hypervolume(ref_point=Z)
    I_A = I(A)

    # Encontrar la ubicación de 'a' en 'A'
    indices = np.where((A == a).all(axis=1))

    # Eliminar 'a' de 'A'
    A_a = np.delete(A, indices, axis=0)

    I_A_a = I(A_a)

    C = abs(I_A - I_A_a)
    return C
# ---------------------------------------------------------------

# -------------Selección para el indicador R2--------------------
def C_R2(a, A, W):
    """
    Función para calcular la contribución al indicador R2 de una solución 'a' en un conjunto 'A'.

    Parameters:
    - a: Solución individual
    - A: Conjunto de soluciones
    - W: Pesos aleatorios

    Returns:
    - Contribución al indicador R2
    """
    I_A = R2(A, W)

    indices = np.where((A == a).all(axis=1))

    A_a = np.delete(A, indices, axis=0)

    I_A_a = R2(A_a, W)

    C = abs(I_A - I_A_a)

    return C

def R2(A, W):
    """
    Función para calcular el indicador R2.

    Parameters:
    - A: Conjunto de soluciones
    - W: Pesos aleatorios

    Returns:
    - Valor del indicador R2
    """
    tamaño_W = len(W)
    suma_utilidades_maximas = 0

    for w in W:
        # Utilizar una list comprehension para recoger las utilidades para cada vector de diseño
        utilities = [np.dot(a, w) for a in A]

        # Verificar si la lista de utilidades no está vacía
        if utilities:
            max_utilidad = max(utilities)
            suma_utilidades_maximas += max_utilidad

    R2 = -(1/tamaño_W) * suma_utilidades_maximas
    return R2
# ---------------------------------------------------------------

# -------------Selección para el indicador IGD-------------------
def C_IGD(a, A, pf):
    """
    Función para calcular la contribución al indicador IGD de una solución 'a' en un conjunto 'A'.

    Parameters:
    - a: Solución individual
    - A: Conjunto de soluciones
    - pf: Frontera de Pareto verdadera

    Returns:
    - Contribución al indicador IGD
    """
    I_A = IGD_C(A, pf)

    # Encontrar la ubicación de 'a' en 'A'
    indices = np.where((A == a).all(axis=1))

    # Eliminar 'a' de 'A'
    A_a = np.delete(A, indices, axis=0)

    I_A_a = IGD_C(A_a, pf)

    C = abs(I_A - I_A_a)
    return C

def calcular_distancia_plus(z, A):
    """
    Función para calcular la distancia plus entre un punto 'z' y un conjunto 'A'.

    Parameters:
    - z: Punto
    - A: Conjunto de puntos

    Returns:
    - Distancia plus
    """
    return np.sqrt(np.sum(np.maximum(A - z, 0)**2, axis=1))

def IGD_C(A, Z):
    """
    Función para calcular el indicador IGD.

    Parameters:
    - A: Conjunto de soluciones
    - Z: Frontera de Pareto verdadera

    Returns:
    - Valor del indicador IGD
    """
    if len(A) == 0:
        return np.nan  # Otra opción podría ser devolver 0 o algún valor predeterminado

    # Broadcasting para calcular distancias de manera eficiente
    distances = np.array([np.min(calcular_distancia_plus(z, A)) for z in Z])

    igd_plus_unario = np.sum(distances) / len(Z)
    return igd_plus_unario
# ---------------------------------------------------------------

# -------------Selección para el indicador E+--------------------
def C_Ep(a, A, pf):
    """
    Función para calcular la contribución al indicador E+ de una solución 'a' en un conjunto 'A'.

    Parameters:
    - a: Solución individual
    - A: Conjunto de soluciones
    - pf: Frontera de Pareto verdadera

    Returns:
    - Contribución al indicador E+
    """
    I_A = e_p(A, pf)

    # Encontrar la ubicación de 'a' en 'A'
    indices = np.where((A == a).all(axis=1))

    # Eliminar 'a' de 'A'
    A_a = np.delete(A, indices, axis=0)

    I_A_a = e_p(A_a, pf)

    C = abs(I_A - I_A_a)
    return C

def e_p(A, Z):
    """
    Función para calcular el indicador E+.

    Parameters:
    - A: Conjunto de soluciones
    - Z: Frontera de Pareto verdadera

    Returns:
    - Valor del indicador E+
    """
    max_distances = np.zeros(Z.shape[0])

    for i in range(Z.shape[0]):
        distances = np.maximum(A - Z[i, :], 0)

        # Verificar si distances no está vacío antes de calcular el máximo
        if distances.size > 0:
            min_distances = np.max(distances, axis=1)
            max_distances[i] = np.min(min_distances)

    return np.max(max_distances)
# ---------------------------------------------------------------

# -------------Selección para el indicador DP--------------------
def C_Dp(a, A, Z):
    """
    Función para calcular la contribución al indicador Dp de una solución 'a' en un conjunto 'A'.

    Parameters:
    - a: Solución individual
    - A: Conjunto de soluciones
    - Z: Frontera de Pareto verdadera

    Returns:
    - Contribución al indicador Dp
    """
    I_GD = GD(Z)
    I_IGD = IGD(Z)

    gd = I_GD(A)
    igd = I_IGD(A)

    Dp = max(gd, igd)

    # Encontrar la ubicación de 'a' en 'A'
    indices = np.where((A == a).all(axis=1))

    # Eliminar 'a' de 'A'
    A_a = np.delete(A, indices, axis=0)

    gd_a = I_GD(A_a)

    igd_a = I_IGD(A_a)

    Dp_a = max(gd_a, igd_a)

    C = abs(Dp - Dp_a)

    return C
# ---------------------------------------------------------------

# -------------Selección para el indicador RIESZ ENERGY----------
def Es(A, s, epsilon=1e-10):
    """
    Función para calcular la Riesz energy de un conjunto de soluciones.

    Parameters:
    - A: Conjunto de soluciones
    - s: Exponente

    Returns:
    - Valor de la Riesz energy
    """
    A = np.array(A)
    n = len(A)

    # Replicar A para comparaciones element-wise
    A_replicado = np.tile(A, (n, 1, 1))

    # Calcular las normas para todas las combinaciones de a y b
    normas = np.linalg.norm(A_replicado - np.expand_dims(A, axis=1), axis=2)

    # Ignorar diagonales (a - a)
    np.fill_diagonal(normas, 1)

    # Calcular el término (-s) y sumar para obtener el resultado
    total = np.sum((normas + epsilon) ** (-s + epsilon))

    return total

def C_Es(a, A, s):
    """
    Función para calcular la contribución al indicador RIESZ ENERGY de una solución 'a' en un conjunto 'A'.

    Parameters:
    - a: Solución individual
    - A: Conjunto de soluciones
    - s: Exponente

    Returns:
    - Contribución al indicador RIESZ ENERGY
    """
    Es_A = Es(A, s)

    # Encontrar la ubicación de 'a' en 'A'
    indice = np.argmax((A == a).all(axis=1))

    # Eliminar 'a' de 'A'
    A_a = np.delete(A, indice, axis=0)

    Es_A_sin_a = Es(A_a, s)

    if np.isnan(Es_A) or np.isnan(Es_A_sin_a):
        contribucion = np.nan
    else:
        contribucion = 0.5 * (Es_A - Es_A_sin_a)

    return contribucion
# ---------------------------------------------------------------
