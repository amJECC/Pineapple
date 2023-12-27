import numpy as np
import math
import Config  # Importar módulo Config

# ------------------Parametros---------------------------------
m = Config.m 
n = Config.n
# -------------------------------------------------------------

def sumatoriaFon(x, n, f):
    """
    Calcula la sumatoria para las funciones fonseca f1 y f2.

    Parameters:
    - x: matriz de entrada
    - n: número de variables
    - f: función ('f1' o 'f2')

    Returns:
    - s: resultado de la sumatoria
    """
    s = 0
    for i in range(n):
        if f == "f1":
            s += pow((x[:, i] - (1 - np.sqrt(n))), 2)
        elif f == "f2":
            s += pow((x[:, i] + (1 - np.sqrt(n))), 2)
    return s

def fonseca(x, n):
    """
    Calcula las funciones fonseca f1 y f2.

    Parameters:
    - x: matriz de entrada
    - n: número de variables

    Returns:
    - Matriz con los resultados de f1 y f2
    """
    f1 = 1 - np.exp(-sumatoriaFon(x, n, "f1"))
    f2 = 1 - np.exp(-sumatoriaFon(x, n, "f2"))
    return np.stack([f1, f2], axis=1)

def seno(f):
    """
    Aplica la función seno a cada elemento de la matriz.

    Parameters:
    - f: matriz de entrada

    Returns:
    - Matriz resultante
    """
    for i, g in enumerate(f):
        f[i] = math.sin(10 * math.pi * f[i])
    return f

def g(x, n):
    """
    Calcula la función g.

    Parameters:
    - x: matriz de entrada
    - n: número de variables

    Returns:
    - Resultado de la función g
    """
    g = 0
    for i in range(n):
        g += 1 + (9 / 29) * (x[:, i])
    return g

def h1(f, g):
    """
    Calcula la función h1.

    Parameters:
    - f: matriz de entrada
    - g: resultado de la función g

    Returns:
    - Resultado de la función h1
    """
    h = 1 - np.sqrt(f / g)
    return h

def h2(f, g):
    """
    Calcula la función h2.

    Parameters:
    - f: matriz de entrada
    - g: resultado de la función g

    Returns:
    - Resultado de la función h2
    """
    h = 1 - pow((f / g), 2)
    return h

def h3(f, g):
    """
    Calcula la función h3.

    Parameters:
    - f: matriz de entrada
    - g: resultado de la función g

    Returns:
    - Resultado de la función h3
    """
    h = 1 - np.sqrt(f / g) - ((f / g) * seno(f))
    return h

def f(x):
    """
    Retorna la segunda columna de la matriz x.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Segunda columna de la matriz x
    """
    f = x[:, 1]
    return f
    
def ZDT1(x, n):
    """
    Calcula las funciones objetivo ZDT1.

    Parameters:
    - x: matriz de entrada
    - n: número de variables

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    f1 = f(x)
    f2 = g(x, n) * h1(f(x), g(x, n))
    return np.stack([f1, f2], axis=1)

def ZDT2(x, n):
    """
    Calcula las funciones objetivo ZDT2.

    Parameters:
    - x: matriz de entrada
    - n: número de variables

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    f1 = f(x)
    f2 = g(x, n) * h2(f(x), g(x, n))
    return np.stack([f1, f2], axis=1)

def ZDT3(x, n):
    """
    Calcula las funciones objetivo ZDT3.

    Parameters:
    - x: matriz de entrada
    - n: número de variables

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    f1 = f(x)
    f2 = g(x, n) * h3(f(x), g(x, n))
    return np.stack([f1, f2], axis=1)

def omega(gx, xi):
    """
    Calcula la función omega.

    Parameters:
    - gx: valor de gx
    - xi: valor de xi

    Returns:
    - Resultado de la función omega
    """
    return np.pi / (4 * (1 + gx)) * (1 + 2 * gx * xi)

def DTLZ1(x):
    """
    Calcula las funciones objetivo DTLZ1.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    funciones = []
    gx = 100 * (np.sum(np.square(x[:, m:] - 0.5) - np.cos(20 * np.pi * (x[:, m:] - 0.5)), axis=1))

    for f in range(m):
        if f == m - 1:
            xi = (1 - x[:, 0])  # (1 - x1)
        else:
            xi = x[:, 0]  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * x[:, (i + 1)]
            if f == 0:
                xi = xi
            else:
                xi = xi * (1 - x[:, ((m - 1) - f)])  # (1 - Xm-1)
        fi = 0.5 * (xi * (1 + gx))  # fm(x)
        funciones.append(fi)

    return np.stack(funciones, axis=1)

def DTLZ2(x):
    """
    Calcula las funciones objetivo DTLZ2.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    funciones = []
    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    for f in range(m):
        if f == m - 1:
            xi = (np.sin(x[:, 0] * (np.pi) / 2))  # (sen(x1) pi/2)
        else:
            xi = np.cos(x[:, 0] * (np.pi) / 2)  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(x[:, i + 1] * (np.pi) / 2))
            if f == 0:
                xi = xi * (np.cos(x[:, ((m - 2) - f)] * (np.pi) / 2))
            else:
                xi = xi * (np.sin(x[:, ((m - 1) - f)] * (np.pi) / 2))  # (1 - Xm-1)
        fi = (1 + gx) * xi  # fm(x)
        funciones.append(fi)

    return np.stack(funciones, axis=1)

def DTLZ3(x):
    """
    Calcula las funciones objetivo DTLZ3.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    funciones = []
    gx = 100 * (np.sum(np.square(x[:, m:] - 0.5) - np.cos(20 * np.pi * (x[:, m:] - 0.5)), axis=1))

    for f in range(m):
        if f == m - 1:
            xi = (np.sin(x[:, 0] * (np.pi) / 2))  # (sen(x1) pi/2)
        else:
            xi = np.cos(x[:, 0] * (np.pi) / 2)  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(x[:, i + 1] * (np.pi) / 2))
            if f == 0:
                xi = xi * (np.cos(x[:, ((m - 2) - f)] * (np.pi) / 2))
            else:
                xi = xi * (np.sin(x[:, ((m - 1) - f)] * (np.pi) / 2))  # (1 - Xm-1)
        fi = (1 + gx) * xi  # fm(x)
        funciones.append(fi)

    return np.stack(funciones, axis=1)

def DTLZ4(x):
    """
    Calcula las funciones objetivo DTLZ4.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    funciones = []
    a = 10
    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    for f in range(m):
        if f == m - 1:
            xi = (np.sin(np.power(x[:, 0], a) * (np.pi) / 2))  # (sen(x1) pi/2)
        else:
            xi = np.cos(np.power(x[:, 0], a) * (np.pi) / 2)  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(np.power(x[:, i + 1], a) * (np.pi) / 2))
            if f == 0:
                xi = xi * (np.cos(np.power(x[:, ((m - 2) - f)], a) * (np.pi) / 2))
            else:
                xi = xi * (np.sin(np.power(x[:, ((m - 1) - f)], a) * (np.pi) / 2))  # (1 - Xm-1)
        fi = (1 + gx) * xi  # fm(x)
        funciones.append(fi)

    return np.stack(funciones, axis=1)

def DTLZ5(x):
    """
    Calcula las funciones objetivo DTLZ5.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    funciones = []
    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    for f in range(m):
        if f == m - 1:
            xi = (np.sin(omega(gx, x[:, 0]) * (np.pi) / 2))  # (sen(x1) pi/2)
        else:
            xi = np.cos(omega(gx, x[:, 0]) * (np.pi) / 2)  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(omega(gx, x[:, i + 1]) * (np.pi) / 2))
            if f == 0:
                xi = xi * (np.cos(omega(gx, x[:, ((m - 2) - f)]) * (np.pi) / 2))
            else:
                xi = xi * (np.sin(omega(gx, x[:, ((m - 1) - f)]) * (np.pi) / 2))  # (1 - Xm-1)
        fi = (1 + gx) * xi  # fm(x)
        funciones.append(fi)

    return np.stack(funciones, axis=1)


def DTLZ6(x):
    """
    Calcula las funciones objetivo DTLZ6.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    funciones = []
    gx = np.sum(np.power(x[:, m:], 0.1), axis=1)

    for f in range(m):
        if f == m - 1:
            xi = (np.sin(omega(gx, x[:, 0]) * (np.pi) / 2))  # (sen(x1) pi/2)
        else:
            xi = np.cos(omega(gx, x[:, 0]) * (np.pi) / 2)  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(omega(gx, x[:, i + 1]) * (np.pi) / 2))
            if f == 0:
                xi = xi * (np.cos(omega(gx, x[:, ((m - 2) - f)]) * (np.pi) / 2))
            else:
                xi = xi * (np.sin(omega(gx, x[:, ((m - 1) - f)]) * (np.pi) / 2))  # (1 - Xm-1)
        fi = (1 + gx) * xi  # fm(x)
        funciones.append(fi)

    return np.stack(funciones, axis=1)

def DTLZ7(x):
    print("DTLZ7")

def kursawe(x):
    """
    Calcula las funciones objetivo para el problema Kursawe.

    Parameters:
    - x: matriz de entrada

    Returns:
    - Matriz con los resultados de las funciones objetivo
    """
    sq_x = x**2
    objetivo_1 = -10 * np.exp(-0.2 * np.sqrt(np.sum(sq_x[:, :2], axis=1))) - 10 * np.exp(-0.2 * np.sqrt(np.sum(sq_x[:, 1:], axis=1)))
    objetivo_2 = np.sum(np.power(np.abs(x), 0.8) + 5 * np.sin(x**3), axis=1)
    return np.stack([objetivo_1, objetivo_2], axis=1)