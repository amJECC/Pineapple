import numpy as np
import pandas as pd
import random
import os
import Funciones
import Config
import matplotlib.pyplot as plt
import functools
import copy
import Indicadores
from pymoo.indicators.hv import Hypervolume

# ------------------Parámetros---------------------------------
population_size = Config.population_size
distribution_index = Config.distribution_index
mutation_rate = Config.mutation_rate
crossover_rate = Config.crossover_rate
list_indicators = Config.list_indicators

dmin = Config.dmin
dmax = Config.dmax

n = Config.n
m = Config.m

TM = Config.TM

nmig = Config.nmig

n_i = len(list_indicators)
# -------------------------------------------------------------

# ----------------------Iniciar Población----------------------
def initial_population():
    """
    Función para inicializar la población.

    Returns:
    - Población inicial
    """
    population = np.random.uniform(dmin, dmax, (int(population_size/5), n))
    population_f = Funciones.DTLZ2(population)
    return population_f
# -------------------------------------------------------------

# -----------Evaluar dominancia de dos soluciones--------------
def domina(x, y):
    """
    Función para verificar si una solución x domina a otra y en términos de Pareto.

    Parameters:
    - x: Solución x
    - y: Solución y

    Returns:
    - True si x domina a y, False en caso contrario
    """
    return np.all(x <= y) and np.any(x < y)
# -------------------------------------------------------------

# ------------------------Offspring----------------------------
def offspring(P):
    """
    Función para generar un descendiente.

    Parameters:
    - P: Población actual

    Returns:
    - Descendiente
    """
    p1 = Seleccion(P)
    p2 = Seleccion(P)

    h1, h2 = SBX(p1, p2)

    h1 = PBM(h1)
    return h1
# -------------------------------------------------------------

def cruza(individuos):
    """
    Función para realizar la cruza entre individuos.

    Parameters:
    - individuos: Población actual

    Returns:
    - Descendencia después de la cruza
    """
    descendencia = individuos.copy()
    for i in range(int(crossover_rate/2)):
        x = np.random.randint(0, individuos.shape[0])
        y = np.random.randint(0, individuos.shape[0])
        while x == y:
            x = np.random.randint(0, individuos.shape[0])
            y = np.random.randint(0, individuos.shape[0])
        punto_cruza = np.random.randint(1, individuos.shape[1])
        descendencia[2*i, 0:punto_cruza] = individuos[x, 0:punto_cruza]  # hijo 1
        descendencia[2*i, punto_cruza:] = individuos[y, punto_cruza:]
        descendencia[2*i+1, 0:punto_cruza] = individuos[y, 0:punto_cruza]  # hijo_2
        descendencia[2*i+1, punto_cruza:] = individuos[x, punto_cruza:]
    return descendencia

def mutacion(individuo):
    """
    Función para realizar la mutación de un individuo.

    Parameters:
    - individuo: Individuo a mutar

    Returns:
    - Individuo mutado
    """
    mutacion_p = (dmax - dmin) * mutation_rate
    descendencia = individuo.copy()
    descendencia += np.random.normal(0, mutacion_p, size=descendencia.shape)
    descendencia = np.clip(descendencia, dmin, dmax)
    return descendencia

# ----------------Selección de padres--------------------------
def Seleccion(population):
    """
    Función para realizar la selección de padres.

    Parameters:
    - population: Población actual

    Returns:
    - Padre seleccionado
    """
    indices = np.random.choice(len(population), 2, replace=False)
    winner_index = indices[np.argmax([domina(population[i], population[j]) for i, j in zip(indices[:-1], indices[1:])])]

    return population[winner_index]
# -------------------------------------------------------------

# -----------------------------SBX-----------------------------
def SBX(padre1, padre2):
    """
    Función para realizar la cruza SBX entre dos padres.

    Parameters:
    - padre1: Primer padre
    - padre2: Segundo padre

    Returns:
    - Dos descendientes después de la cruza
    """
    child1 = np.copy(padre1)
    child2 = np.copy(padre2)

    if random.uniform(0.0, 1.0) <= crossover_rate:
        for i in range(len(child1)):
            if random.uniform(0.0, 1.0) <= 0.5:
                x1, x2 = sbx_crossover(child1[i], child2[i])

                child1[i] = x1
                child2[i] = x2

    return child1, child2

def sbx_crossover(x1, x2):
    """
    Función para realizar la cruza SBX en una variable entre dos valores.

    Parameters:
    - x1: Valor de la primera variable del primer padre
    - x2: Valor de la primera variable del segundo padre

    Returns:
    - Dos valores después de la cruza SBX
    """
    dx = x2 - x1

    if dx > np.finfo(float).eps:
        if x2 > x1:
            y2, y1 = x2, x1
        else:
            y2, y1 = x1, x2

        beta = 1.0 / (1.0 + (2.0 * (y1 - dmin) / (y2 - y1)))
        alpha = 2.0 - pow(beta, distribution_index + 1.0)
        rand = random.uniform(0.0, 1.0)

        if rand <= 1.0 / alpha:
            alpha = alpha * rand
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))
        else:
            alpha = alpha * rand
            alpha = 1.0 / (2.0 - alpha)
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))

        x1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
        beta = 1.0 / (1.0 + (2.0 * (dmax - y2) / (y2 - y1)))
        alpha = 2.0 - pow(beta, distribution_index + 1.0)

        if rand <= 1.0 / alpha:
            alpha = alpha * rand
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))
        else:
            alpha = alpha * rand
            alpha = 1.0 / (2.0 - alpha)
            betaq = pow(alpha, 1.0 / (distribution_index + 1.0))

        x2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        # randomly swap the values
        if bool(random.getrandbits(1)):
            x1, x2 = x2, x1

        x1 = np.clip(x1, dmin, dmax)
        x2 = np.clip(x2, dmin, dmax)

    return x1, x2
# -------------------------------------------------------------

# -----------------------------PBM-----------------------------
def PBM(individuo):
    """
    Función para realizar la mutación PBM de un individuo.

    Parameters:
    - individuo: Individuo a mutar

    Returns:
    - Individuo mutado
    """
    mutado = copy.deepcopy(individuo)

    for i in range(len(mutado)):
        if random.uniform(0.0, 1.0) <= mutation_rate:
            mutado[i] = pm_mutation(mutado[i])

    return mutado

def pm_mutation(x):
    """
    Función para realizar la mutación PBM en una variable de un individuo.

    Parameters:
    - x: Valor de la variable a mutar

    Returns:
    - Valor de la variable mutado
    """
    u = random.uniform(0, 1)
    dx = dmax - dmin

    if u < 0.5:
        bl = (x - dmin) / dx
        b = 2.0 * u + (1.0 - 2.0 * u) * pow(1.0 - bl, distribution_index + 1.0)
        delta = pow(b, 1.0 / (distribution_index + 1.0)) - 1.0
    else:
        bu = (dmax - x) / dx
        b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * pow(1.0 - bu, distribution_index + 1.0)
        delta = 1.0 - pow(b, 1.0 / (distribution_index + 1.0))

    x = x + delta * dx
    x = np.clip(x, dmin, dmax)

    return x
# -------------------------------------------------------------

# ------------------------------Migration----------------------
def Migration():
    """
    Función para realizar la migración entre islas.
    """
    for isla_origen in range(n_i):
        for isla_destino in range(n_i):
            # Verificamos que las islas tienen conexión
            if TM[isla_origen][isla_destino] == 1:

                for _ in range(nmig):
                    print(f"Enviando individuo de {list_indicators[isla_origen]} a {list_indicators[isla_destino]}")

                    df_o = pd.read_csv(f"{list_indicators[isla_origen]}_p.csv")
                    df_d = pd.read_csv(f"{list_indicators[isla_destino]}_p.csv")

                    indice = random.randint(0, len(df_o) - 1)
                    individuo = df_o.iloc[indice]
                    ind = individuo["Indicador"]

                    while ind == list_indicators[isla_destino]:
                        indice = random.randint(0, len(df_o) - 1)
                        individuo = df_o.iloc[indice]
                        ind = individuo["Indicador"]

                    df_o.drop(df_o.index[indice], inplace=True)

                    # Agrega el individuo al DataFrame de la isla de destino
                    df_d = pd.concat([df_d, pd.DataFrame([individuo])], ignore_index=True)

                    # Guarda el DataFrame actualizado en el archivo CSV de la isla de destino
                    df_d.to_csv(f"{list_indicators[isla_destino]}_p.csv", index=False)
# -------------------------------------------------------------

#------------------------------Insert-------------------------
def Insert(I, r):
    """
    Inserta una nueva solución en el conjunto de soluciones de un indicador.

    Parameters:
    - I: Nombre del indicador
    - r: Nueva solución a insertar en el conjunto
    
    Returns:
    - None
    """
    df = pd.read_csv(f"{I}_a.csv")
    
    A = np.array(df["Solucion"])
    
    sol = [] 
    for a in A:
        sol.append(np.fromstring(a[1:-1], sep=' '))

    A = np.array(sol)
    
    indices_eliminar = []
    
    for ind, a in enumerate(A):
        if domina(r, a):
            indices_eliminar.append(ind)
            A = quitar(A, a)
        elif domina(a, r):
            return
        
    if len(indices_eliminar) > 0:
        df.drop(indices_eliminar)
    
    r_df = pd.DataFrame({"Indicador": [I], "Solucion": [np.array2string(r).replace('\n', '')]})
    A = np.vstack([A, r])
    
    df = pd.concat([df, r_df], ignore_index=True)
    
    while len(A) > int(population_size / 5):
        i_aw = np.argmax([Indicadores.C_Es(a, A, (m-1)) for a in A])
        aw = A[i_aw]
        A = quitar(A, aw)
        df.drop(df.index[i_aw], inplace=True)
    
    df.to_csv(f"{I}_a.csv", index=False)   
#-------------------------------------------------------------

#----------------Probar con Grafica---------------------------
def graficar(P):
    """
    Genera un gráfico tridimensional a partir de un conjunto de soluciones.

    Parameters:
    - P: Conjunto de soluciones
    
    Returns:
    - None
    """
    # Extrae las coordenadas X, Y y Z de tus datos
    x = P[:, 0]  # Primera columna
    y = P[:, 1]  # Segunda columna
    z = P[:, 2]  # Tercera columna

    # Crea una figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotea los puntos en el espacio 3D
    ax.scatter(x, y, z, c='b', marker='o')

    # Configura las etiquetas de los ejes
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')

    # Muestra el gráfico
    plt.show()  
#-------------------------------------------------------------

#----------------Matriz Dominancia----------------------------
def matriz_dominancia(datos):    
    """
    Calcula la matriz de dominancia a partir de un conjunto de datos.

    Parameters:
    - datos: Conjunto de datos
    
    Returns:
    - Matriz de dominancia
    """
    p_t = datos.shape[0]
    n = datos.shape[1]
    
    x = np.zeros([p_t, p_t, n])
    y = np.zeros([p_t, p_t, n])
    
    for i in range(p_t):
        x[i, :, :] = datos[i]
        y[:, i, :] = datos[i]
    
    condicion_1 = x <= y
    condicion_2 = x < y
    
    return np.logical_and(np.all(condicion_1, axis=2), np.any(condicion_2, axis=2))  
#-------------------------------------------------------------

#------------------------------Pareto Front-------------------
def frentes_pareto(datos):
    """
    Identifica los frentes de Pareto a partir de un conjunto de datos.

    Parameters:
    - datos: Conjunto de datos
    
    Returns:
    - Lista de frentes de Pareto
    """
    conjunto_d = []
    cuenta = []
    
    matriz = matriz_dominancia(datos)
    pop_size = datos.shape[0]
    
    for i in range(pop_size):
        dominante_actual = set()
        cuenta.append(0)
        for j in range(pop_size):
            if matriz[i, j]:
                dominante_actual.add(j)
            elif matriz[j, i]:
                cuenta[-1] += 1
                
        conjunto_d.append(dominante_actual)

    cuenta = np.array(cuenta)
    frentes = []
    while True:
        frente_actual = np.where(cuenta == 0)[0]
        if len(frente_actual) == 0:
            break
        
        frentes.append(frente_actual)

        for individual in frente_actual:
            cuenta[individual] = -1 
            dominado_actual_c = conjunto_d[individual]
            for dominado_aux in dominado_actual_c:
                cuenta[dominado_aux] -= 1
            
    return frentes
#-------------------------------------------------------------

#-----------------------Rango de dominancia-------------------
def rangos(frentes): 
    """
    Asigna un rango a cada solución basado en los frentes de Pareto.

    Parameters:
    - frentes: Lista de frentes de Pareto
    
    Returns:
    - Diccionario con el índice de cada solución y su rango
    """
    rank_indice = {}
    for i, front in enumerate(frentes):
        for x in front:   
            rank_indice[x] = i
            
    return rank_indice
#-------------------------------------------------------------

#-----------------------Quitar un elemnto---------------------
def quitar(Q, rw):
    """
    Elimina un elemento de un conjunto.

    Parameters:
    - Q: Conjunto
    - rw: Elemento a eliminar
    
    Returns:
    - Conjunto actualizado
    """
    # Encontrar el primer índice del valor en el conjunto
    indice_eliminar = next((i for i, solucion in enumerate(Q) if np.array_equal(solucion, rw)), None)

    # Si se encontró el índice, eliminar el elemento
    if indice_eliminar is not None:
        Q = np.delete(Q, indice_eliminar, axis=0)
    
    return Q
#-------------------------------------------------------------

#-------------Generar vector de pesos aleatorios--------------
def random_weights(population_size):
    """
    Genera un conjunto de vectores de pesos aleatorios para utilizar en algoritmos de optimización multicriterio.

    Parameters:
    - population_size: Tamaño del conjunto
    
    Returns:
    - Conjunto de vectores de pesos aleatorios
    """
    weights = []

    if n == 2:
        weights = np.array([[1, 0], [0, 1]])
        remaining_weights = np.column_stack([(np.arange(1, population_size-1) / (population_size-1.0)), (1.0 - np.arange(1, population_size-1) / (population_size-1.0))])
        weights = np.vstack([weights, remaining_weights])
    else:
        # generate candidate weights
        candidate_weights = np.random.rand(population_size*50, n)
        candidate_weights /= candidate_weights.sum(axis=1, keepdims=True)

        # add weights for the corners
        weights.extend(np.eye(n))

        # iteratively fill in the remaining weights by finding the candidate
        # weight with the largest distance from the assigned weights
        while len(weights) < population_size:
            distances = np.min(np.linalg.norm(candidate_weights[:, np.newaxis, :] - np.array(weights), axis=2), axis=1)
            max_index = np.argmax(distances)
            weights.append(candidate_weights[max_index])
            candidate_weights = np.delete(candidate_weights, max_index, axis=0)

    return np.array(weights)
#-------------------------------------------------------------

#------------------Elementos no dominados---------------------
def no_dominados(population):
    """
    Identifica y devuelve las soluciones no dominadas en un conjunto.

    Parameters:
    - population: Conjunto de soluciones
    
    Returns:
    - Soluciones no dominadas
    """
    num_solutions = population.shape[0]
    indices_no_dominantes = np.ones(num_solutions, dtype=bool)

    for i in range(num_solutions):
        for j in range(num_solutions):
            if i != j and domina(population[j], population[i]):
                indices_no_dominantes[i] = False
                break

    no_dominantes = population[indices_no_dominantes]
    return no_dominantes
#-------------------------------------------------------------
