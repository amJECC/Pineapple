import concurrent.futures
import pandas as pd 
import numpy as np
import IB_MOEA
import matplotlib.pyplot as plt
import Config
import Operadores
import Funciones
import Indicadores

#------------------Parametros---------------------------------
population_size = Config.population_size

min_values = Config.min_values
max_values = Config.max_values
list_indicators = Config.list_indicators

m = Config.m

epsilon = 1e-10

#-------------------------------------------------------------

#----------Funcion para graficar los individuos---------------
def graficar(fig,matriz,titulo,posicion):

    # Extrae las columnas de la matriz
    columna1 = matriz[:, 0]  # Reemplaza el índice 0 con el número de columna que desees para DTLZ1
    columna2 = matriz[:, 1]  # Reemplaza el índice 1 con el número de columna que desees para DTLZ2
    columna3 = matriz[:, 2]  # Reemplaza el índice 2 con el número de columna que desees para DTLZ3

    ax = fig.add_subplot(posicion, projection='3d')

    # Grafica cada fila como un punto en el espacio tridimensional
    ax.scatter(columna1, columna2, columna3, c='b', marker='+')

    # Etiqueta los ejes
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F3')
    
    # Agrega un título a la figura
    ax.set_title(titulo)
    ax.view_init(elev=20, azim=25)
#-------------------------------------------------------------

#---------Funcion para ejecutar el IBEA paralelamente---------
def ejecutar_IBEA(ind):
    P = IB_MOEA.IBEA(ind)
    return P, ind
#-------------------------------------------------------------

#--------Obtener las soluciones de los archivos---------------
def obtenes_soluciones(a):

    # Crear un diccionario para almacenar los DataFrames
    dataframes = {}

    # Cargar los archivos CSV y crear DataFrames
    for indicator in list_indicators:
        file_name = f"{indicator}_{a}.csv"
        dataframes[indicator] = pd.read_csv(file_name)

    # Combinar todos los DataFrames en uno solo
    combined_df = pd.concat([df[["Indicador", "Solucion"]] for df in dataframes.values()], ignore_index=True)

    soluciones = np.array(combined_df["Solucion"])

    sol = [] 
    for solucion in soluciones:
        sol.append(np.fromstring(solucion[1:-1], sep=' '))

    soluciones = np.array(sol)
    
    #Filtramos las soluciones que se pasen de los valores minimos y maximos 
    mascara = np.all((soluciones[:, -(len(min_values)):] >= min_values) & (soluciones[:, -(len(min_values)):] <= max_values), axis=1)
    
    # Obtener las soluciones que cumplen con el rango
    soluciones_filtradas = soluciones[mascara]

    return soluciones_filtradas
#-------------------------------------------------------------

#------------Paralelizacion de los indicadores----------------  
def IMIA():
    tasks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for ind in list_indicators:
            task = executor.submit(ejecutar_IBEA, ind)
            tasks.append(task)

    results = [task.result() for task in tasks]

    Operadores.Migration()
    
    fig = plt.figure()
    for i, (P, ind) in enumerate(results):
        posicion = 151 + i  # Aumenta la posición en cada iteración
        graficar(fig,P, ind, posicion)
    
    print("Obteniendo soluciones")
    P = obtenes_soluciones("p")
    A = obtenes_soluciones("a")
    
    A = np.vstack([P,A])
    return A
#-------------------------------------------------------------

#--------------------Ejecucion principal----------------------   
def main():

    A = IMIA()
    
    A = Operadores.no_dominados(A)

    if(len(A) > population_size):
        print(f"longitud A - - - {len(A)}")
        # Punto Nadir
        nadir_point = np.max(A, axis=0)

        # Punto Ideal
        ideal_point = np.min(A, axis=0)
        
        #Normalizacion de las soluciones
        A = (A - ideal_point) / (nadir_point - ideal_point + epsilon)
        
    while(len(A) > population_size):
        print(f"longitud A - - - {len(A)}")
        i_aw = np.argmax([Indicadores.C_Es(a,A,(m-1)) for a in A])
        aw = A[i_aw]
        A = Operadores.quitar(A,aw)
    
    filename = f"IMIA"+".csv"
    
    IB_MOEA.archivo(A,filename,"IMIA")
    
    fig = plt.figure()
    graficar(fig,A,"IMIA",111)
    
    # Muestra el gráfico
    plt.show()

main()
#-------------------------------------------------------------