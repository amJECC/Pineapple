import numpy as np
import pandas as pd
import Operadores
import Config
import Funciones
import Indicadores
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume

# ------------------Parametros---------------------------------
# Definición de parámetros desde el módulo Config
population_size = Config.population_size
distribution_index = Config.distribution_index
mutation_rate = Config.mutation_rate
crossover_rate = Config.crossover_rate
list_indicators = Config.list_indicators
generations = Config.generations

dmin = Config.dmin
dmax = Config.dmax

n = Config.n
epsilon = 1e-10

# M = len(list_of_functions)

TM = Config.TM
# -------------------------------------------------------------

# -------------Agregar elementos al csv (A o P)----------------
def archivo(population, filename, Ind):
    """
    Guarda las soluciones de la población en un archivo CSV.

    Parameters:
    - population: población de soluciones
    - filename: nombre del archivo CSV
    - Ind: indicador utilizado

    Returns:
    - None
    """
    data = []

    cleaned_population = np.array([np.array2string(solution).replace('\n', '') for solution in population])

    # Almacenar soluciones y el indicador utilizado
    data.extend([(Ind, solution) for solution in cleaned_population])

    # Crear un DataFrame a partir de los datos
    df = pd.DataFrame(data, columns=['Indicador', 'Solucion'])

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(filename, index=False)


# -------------------------------------------------------------

# -------Ciclo Generic steady-state IB_MOEA--------------------
def IBEA(Ind):
    """
    Implementación del algoritmo IBEA con el indicador especificado.

    Parameters:
    - Ind: indicador a utilizar

    Returns:
    - Población final después de las generaciones
    """
    filename = f"{Ind}_p" + ".csv"
    file_a = f"{Ind}_a" + ".csv"

    print(f"usando el indicador: {Ind}")
    count = 1
    population = Operadores.initial_population()

    A = Operadores.no_dominados(population)

    archivo(A, file_a, Ind)

    while count <= generations:

        print(f"Indicador {Ind} en la generacion {count}")

        # population = Funciones.DTLZ2(population)

        offspring = Operadores.offspring(population)
        offspring = np.array([np.array(offspring)])
        offspring = Funciones.DTLZ2(offspring)

        # print(offspring)
        Q = np.vstack([population, offspring])

        # Punto Nadir
        nadir_point = np.max(Q, axis=0)

        # Punto Ideal
        ideal_point = np.min(Q, axis=0)

        Q = (Q - ideal_point) / (nadir_point - ideal_point + epsilon)

        # Q = Funciones.DTLZ2(Q)

        fronts = Operadores.frentes_pareto(Q)

        posicion_ranking = Operadores.rangos(fronts)

        max_rango = max(posicion_ranking.values())

        # Filtrar las soluciones con el peor rango
        i_Rt = [solucion for solucion, rango in posicion_ranking.items() if rango == max_rango]

        Rt = Q[i_Rt]

        if len(Rt) > 1:
            if Ind == "HV":
                i_rw = np.argmin([Indicadores.C_HV(r, Rt) for r in Rt])
            elif Ind == "R2":
                W = Operadores.random_weights(len(Rt))
                i_rw = np.argmin([Indicadores.C_R2(r, Rt, W) for r in Rt])
            elif Ind == "IGD":
                i_rw = np.argmin([Indicadores.C_IGD(r, Rt, Q[fronts[0]]) for r in Rt])
            elif Ind == "e":
                i_rw = np.argmin([Indicadores.C_Ep(r, Rt, Q[fronts[0]]) for r in Rt])
            elif Ind == "D":
                i_rw = np.argmin([Indicadores.C_Dp(r, Rt, Q[fronts[0]]) for r in Rt])
            rw = Rt[i_rw]
        else:
            rw = Rt[0]

        if not np.array_equal(offspring, rw):
            Operadores.Insert(Ind, offspring[0])

        population = Operadores.quitar(Q, rw)

        archivo(population, filename, Ind)

        count += 1

    return population


# -------------------------------------------------------------