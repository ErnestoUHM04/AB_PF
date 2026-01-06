# Cornejo Morales Paola
# Hernández Martínez Ernesto Ulises

import random
import pandas as pd
from predict import predecir

# Usar el algoritmo de la colonia de abejas para encontrar el mejor parley posible para una jornada de la Liga MX.

# Cada jornada es un conjunto de partidos que se juegan en una misma semana.
# Usaremos varias abejas () para cada jornada, y posteriormente se repetirá el proceso para abarcar todas las jornadas de la temporada.

# Cada abeja se verá de esta manera:
# [x1, x2, x3, x4, x5, x6, x7, x8, x9].   <-- cada xi representa un partido de la jornada
# Cada xi es un vector que puede tomar los valores:
# xi = [xi1, xi2]
# xi1 = 1 si se incluye el partido en el parley, 0 si no se incluye.
# xi2 = 0 si se predice victoria local, 1 si se predice empate, 2 si se predice victoria visitante.

###### Parameters ###### <----- para un BeeHive Algorithm
# Número de variables
n = 9  #.  <--- PUEDE VARIAR DENTRO DEL PROGRAMA
# Tamaño del enjambre = 40
beehive_size = 40
# abeja obreras = 20
worker_bees_count = beehive_size // 2
#print("Número de abejas obreras:", worker_bees_count)
# abejas Observadoras = 20
observer_bees_count = beehive_size // 2
#print("Número de abejas observadoras:", observer_bees_count)
# límite = 5
# limit = 5
limit = (beehive_size * n) // 2
# iteraciones = 50
max_iterations = 50
# Capacidad de la mochila : 30 lb.     <----- ESTE VALOR SE VA A CAMBIAR O ELIMINAR
max_capacity = 30
# Una abeja no debe tener todos los partidos de la jornada obligatoriamente, puede tener solo algunos.
partidos_min_jornada = 2 # Mínimo de partidos que debe tener una abeja para considerarse válida.  <-- CAMBIAR SI ESTOY MAL

########################

def leer_historial_liga():
    # Cargar datos históricos de partidos de la Liga MX desde un archivo CSV
    df = pd.read_csv('data/datos_normalizados.csv')
    return df

def define_parley_games(df, umbral_diferencia = 0.15, print_progress = False):
    # Ahora usaremos estos vectores para guardar los partidos que usaremos en el parley para cada jornada
    jornada1 = [] # 0 si no se incluye el partido, 1 si se incluye
    jornada2 = []
    jornada3 = []
    jornada4 = []
    jornada5 = []
    jornada6 = []
    jornada7 = []
    jornada8 = []
    jornada9 = []
    jornada10 = []
    jornada11 = []
    jornada12 = []
    jornada13 = []
    jornada14 = []
    jornada15 = []
    jornada16 = []
    jornada17 = []
    play_in = []
    quarter_finals = []
    semi_finals = []

    umbral_diferencia = umbral_diferencia  # Umbral para considerar una diferencia significativa  <--- CAMBIAR ESTE VALOR SI ES NECESARIO

    for i in range(len(df)):
        # Conseguir los nombres de los equipos y las probabilidades históricas
        local = df.iloc[i]['Local']
        visitante = df.iloc[i]['Visitante']
        prob_local_histo = df.iloc[i]['PG_L']
        prob_empate_histo = df.iloc[i]['PG_E']
        prob_visita_histo = df.iloc[i]['PG_V']
        # Usar el modelo para predecir las probabilidades
        prob_local_model, prob_empate_model, prob_visita_model = predecir(local, visitante)

        # Comparamos las probabilidaddes
        prob_local_diff = abs(prob_local_histo - prob_local_model)
        prob_empate_diff = abs(prob_empate_histo - prob_empate_model)
        prob_visita_diff = abs(prob_visita_histo - prob_visita_model)

        prob_local_promedio = float((prob_local_histo + prob_local_model) / 2)
        prob_empate_promedio = float((prob_empate_histo + prob_empate_model) / 2)
        prob_visita_promedio = float((prob_visita_histo + prob_visita_model) / 2)

        # Guardamos los momios de los partidos elegidos    <--- ESTO SERÁ PARA LA FUNCIÓN FITNESS
        momio_L = float(df.iloc[i]['Momio_L_norm'])
        momio_V = float(df.iloc[i]['Momio_V_norm'])
        momio_E = float(df.iloc[i]['Momio_E_norm'])


        # Si la diferencia entre las probabilidades es menor al umbral, entonces incluimos el partido en el parley
        if prob_local_diff < umbral_diferencia and prob_empate_diff < umbral_diferencia and prob_visita_diff < umbral_diferencia:
            # Si la diferencia es pequeña, entonces incluimos el partido en el parley
            jornada = df.iloc[i]['Jornada']
            if jornada == '1':
                jornada1.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '2':
                jornada2.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '3':
                jornada3.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '4':
                jornada4.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '5':
                jornada5.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '6':
                jornada6.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '7':
                jornada7.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '8':
                jornada8.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '9':
                jornada9.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '10':
                jornada10.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '11':
                jornada11.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '12':
                jornada12.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '13':
                jornada13.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '14':
                jornada14.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '15':
                jornada15.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '16':
                jornada16.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '17':
                jornada17.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == 'PI1':
                play_in.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '1/4F':
                quarter_finals.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == 'SF':
                semi_finals.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            else:
                pass  # Jornada no válida, no hacemos nada
        else: # ponemos un 0 indicando que no se incluye el partido en el parley
            if jornada == '1':
                jornada1.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '2':
                jornada2.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '3':
                jornada3.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '4':
                jornada4.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '5':
                jornada5.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '6':
                jornada6.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '7':
                jornada7.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '8':
                jornada8.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '9':
                jornada9.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '10':
                jornada10.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '11':
                jornada11.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '12':
                jornada12.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '13':
                jornada13.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '14':
                jornada14.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '15':
                jornada15.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '16':
                jornada16.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == '17':
                jornada17.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == 'PI1':
                play_in.append((1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V)) # CONTAR SIEMPRE (SOLO SON 2 JUEGOS)
            elif jornada == '1/4F':
                quarter_finals.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            elif jornada == 'SF':
                semi_finals.append((0, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V))
            else:
                pass  # Jornadas no válidas

    def format_floats_in_tuple(tup):
        return tuple(f"{x:.3f}" if isinstance(x, float) else x for x in tup)

    def format_jornada(jornada):
        return [format_floats_in_tuple(t) for t in jornada]

    if print_progress:
        # Imprimimos los partidos seleccionados para cada jornada
        print(f"Jornada 1: {format_jornada(jornada1)}\n")
        print(f"Jornada 2: {format_jornada(jornada2)}\n")
        print(f"Jornada 3: {format_jornada(jornada3)}\n")
        print(f"Jornada 4: {format_jornada(jornada4)}\n")
        print(f"Jornada 5: {format_jornada(jornada5)}\n")
        print(f"Jornada 6: {format_jornada(jornada6)}\n")
        print(f"Jornada 7: {format_jornada(jornada7)}\n")
        print(f"Jornada 8: {format_jornada(jornada8)}\n")
        print(f"Jornada 9: {format_jornada(jornada9)}\n")
        print(f"Jornada 10: {format_jornada(jornada10)}\n")
        print(f"Jornada 11: {format_jornada(jornada11)}\n")
        print(f"Jornada 12: {format_jornada(jornada12)}\n")
        print(f"Jornada 13: {format_jornada(jornada13)}\n")
        print(f"Jornada 14: {format_jornada(jornada14)}\n")
        print(f"Jornada 15: {format_jornada(jornada15)}\n")
        print(f"Jornada 16: {format_jornada(jornada16)}\n")
        print(f"Jornada 17: {format_jornada(jornada17)}\n")
        print(f"Play In's: {format_jornada(play_in)}\n")
        print(f"Quarter Finals: {format_jornada(quarter_finals)}\n")
        print(f"Semi Finals: {format_jornada(semi_finals)}\n")

    return [jornada1, jornada2, jornada3, jornada4, jornada5, jornada6, jornada7, jornada8, jornada9, jornada10, jornada11, jornada12, jornada13, jornada14, jornada15, jornada16, jornada17, play_in, quarter_finals, semi_finals]

def create_worker_bees(lower_bounds, upper_bounds, jornada, print_progress=True):
    n = len(jornada)
    # Extraer los valores de la jornada
    # (partido1, partido2, ..., partido9)                   <-- estructura de la jornada   (no siempre son 9 partidos)
    # (1, local, visitante, prob_local_promedio, prob_visita_promedio, prob_empate_promedio, momio_L, momio_E, momio_V) <-- estructura de cada partido
    #. 0.    1.     2.             3.                      4.                 5.               6.        7.       8.

    # Create worker bees
    worker_bees = []
    for i in range(worker_bees_count):
        worker_bee = create_worker_bee(lower_bounds, upper_bounds, jornada, n) # Indiviual bee.    devuelve una lista de => 0, 1, o 2     (local, empate, o visitante)
        # (0, 1, -1, 0, -1, 2, 0, 1, 2), donde los numeros -1 significan que no se incluye en el parley al individuo
        momios_combinados = get_momios(worker_bee, jornada, n)

        prob_acertar = get_prob_acertar(worker_bee, jornada, n)

        fitness_value = fitness(momios_combinados, prob_acertar) # CONSEGUIR ESOS DOS VALORES

        if print_progress:
            print("Worker Bee:", worker_bee, "\tValue:", fitness_value)
        # Remember initialization of the limit counter for each worker bee
        worker_bees.append((worker_bee, fitness_value, 0)) # (bee, fitness, limit_counter)
    return worker_bees

def create_worker_bee(lower_bounds, upper_bounds, jornada, n):
    bee = []
    for i in range(n):
        if jornada[i][0] == 0:  # No se incluye
            bee.append(-1)
        else:
            r1 = random.random()  # Número aleatorio entre 0 y 1
            xi = lower_bounds[i] + r1 * (upper_bounds[i] - lower_bounds[i])
            xi = round(xi)  # Redondear al entero más cercano
            bee.append(xi)

    # Contar partidos incluidos
    included = sum(1 for x in bee if x != -1)

    # Si no hay suficientes partidos incluidos, forzar algunos -1 a ser incluidos
    if included < partidos_min_jornada:
        excluded_indices = [i for i, x in enumerate(bee) if x == -1]
        if excluded_indices:
            # Elegir aleatoriamente cuántos incluir para llegar al mínimo
            to_include = min(len(excluded_indices), partidos_min_jornada - included)
            selected = random.sample(excluded_indices, to_include)
            for idx in selected:
                r1 = random.random()
                xi = lower_bounds[idx] + r1 * (upper_bounds[idx] - lower_bounds[idx])
                bee[idx] = round(xi)

    return bee

def fitness(momios_combinados, prob_acertar): # BUSCAMOS MAXIMIZAR LA FUNCIÓN FITNESS
    # La función Fitness de una abeja será:
    # f = M(momios de la apuesta) / P(asertar todos los partidos del parley)
    fitness_value = momios_combinados / prob_acertar # Caso para MAXIMIZACIÓN
    return fitness_value

def get_momios(bee, jornada, n): # M(momios de la apuesta) = M(p1) * M(p2) * ... * M(p9)
    momios_combinados = 1.0
    for i in range(n):
        apuesta = bee[i] # esta puede ser 0, 1, 2, o -1
        if apuesta == 0:
            # apuesta por local
            momio = jornada[i][6]
        elif apuesta == 1:
            # apuesta por empate
            momio = jornada[i][7]
        elif apuesta == 2:
            # apuesta por visitante
            momio = jornada[i][8]
        else:
            pass
        momios_combinados *= momio
    return momios_combinados

def get_prob_acertar(bee, jornada, n): # P(asertar todos los partidos del parley) = P(p1) * P(p2) * ... * P(p9) <-- Usar el modelo entrenado previamente para obtener esta probabilidad.
    prob_acertar = 1.0
    for i in range(n):
        apuesta = bee[i] # esta puede ser 0, 1, 2, o -1
        if apuesta == 0:
            # apuesta por local
            proba = jornada[i][3]
        elif apuesta == 1:
            # apuesta por empate
            proba = jornada[i][5]
        elif apuesta == 2:
            # apuesta por visitante
            proba = jornada[i][4]
        else:
            pass
        prob_acertar *= proba
    return prob_acertar

def enforce_min_partidos(bee, lower_bounds, upper_bounds, min_partidos):
    included = sum(1 for x in bee if x != -1)

    if included >= min_partidos:
        return bee

    excluded = [i for i, x in enumerate(bee) if x == -1]
    needed = min(min_partidos - included, len(excluded))

    selected = random.sample(excluded, needed)
    for idx in selected:
        r1 = random.random()
        xi = lower_bounds[idx] + r1 * (upper_bounds[idx] - lower_bounds[idx])
        bee[idx] = round(xi)

    return bee

# CAMBIAR ESTA FUNCIÓN
def worker_bee_search(i, bee, lower_bounds, upper_bounds):
    # First, we select a dimension to modify
    j = random.randint(0, n - 1)
    # Now, we select another bee to interact with
    while True:
        k = random.randint(0, worker_bees_count - 1)
        if k != i: # Ensure we don't select the same bee
            break
    r2 = random.uniform(-1, 1) # Número aleatorio entre -1 y 1

    # Modify the selected dimension
    bee[j] = bee[j] + r2 * (bee[j] - worker_bees[k][0][j])

    if bee[j] < lower_bounds[j]:
        bee[j] = lower_bounds[j]
    elif bee[j] > upper_bounds[j]:
        bee[j] = upper_bounds[j]

    bee[j] = round(bee[j]) # Redondear al entero más cercano <- NO FRACTIONS
    
    bee = enforce_min_partidos(bee, lower_bounds, upper_bounds, partidos_min_jornada) #Asegurar mínimo de partidos incluidos

    return bee

def beehive_algorithm(worker_bees, lower_bounds, upper_bounds, jornada, print_progress=True):
    HoF = [] # Hall of Fame
    iteration = 0
    n = len(jornada)
    while True: # We will run this for max_iterations
        iteration += 1
        if print_progress:
            print("\n--- Iteration", iteration, "---")
        # We need them to work now :)
        for i in range(worker_bees_count):
            bee, fitness_value, limit_counter = worker_bees[i]

            new_bee = worker_bee_search(i, bee.copy(), lower_bounds, upper_bounds)

            momios_combinados = get_momios(new_bee, jornada, n)

            prob_acertar = get_prob_acertar(new_bee, jornada, n)

            new_fitness_value = fitness(momios_combinados, prob_acertar) # CONSEGUIR ESOS DOS VALORES

            # Check if the new bee is better
            if new_fitness_value > fitness_value: # If it is, we update the bee, and reset the counter
                worker_bees[i] = (new_bee, new_fitness_value, 0) # Reset limit counter
            else: # If not, we DO NOT update the bee, and we increment the counter by one
                limit_counter += 1
                worker_bees[i] = (bee, fitness_value, limit_counter)

            if print_progress:
                print("Worker Bee:", worker_bees[i][0], "\tFitness:", worker_bees[i][1])

        # After they finish working, we evaluate their fitness and create probabilities for observer bees
        # After the worker bee, create observer bees based on the best solutions found by worker bees
        sum_fitness = sum(bee[1] for bee in worker_bees)
        probabilities = [bee[1] / sum_fitness for bee in worker_bees]

        acumulated_probabilities = calculate_acumulated_probabilities(probabilities)

        # Create observer bees
        observer_bees = [] # We select the best solutions found by observer_bees via the waggle dance
        for i in range(observer_bees_count):

            observer_bee, followed_bee_index = create_observer_bee(acumulated_probabilities, worker_bees, lower_bounds, upper_bounds)
                # We evaluate the observer bee
            momios_combinados = get_momios(observer_bee, jornada, n)

            prob_acertar = get_prob_acertar(observer_bee, jornada, n)

            fitness_value = fitness(momios_combinados, prob_acertar)

            # We still check if the observer bee is better than the followed worker bee
            followed_bee = worker_bees[followed_bee_index]
            # if they are better than the worker bees, then we replace the worker bees with them
            if fitness_value > followed_bee[1]: # If it is, we replace the worker bee with the observer bee
                worker_bees[followed_bee_index] = (observer_bee, fitness_value, 0) # Reset limit counter
            else:
                # We do not replace the worker bee, just add one to the limit counter
                worker_bees[followed_bee_index] = (followed_bee[0], followed_bee[1], followed_bee[2] + 1) # Increment limit counter +1

            if print_progress:
                print("Observer Bee:", observer_bee, "\tFitness:", fitness_value)
            observer_bees.append((observer_bee, fitness_value))

        # Now we check for any bee that has exceeded the limit and reinitialize it
        for i in range(worker_bees_count):
            bee, fitness_value, limit_counter = worker_bees[i]

            # THIS IS THE EXPLORER BEE PHASE
            if limit_counter >= limit:
                # Reinitialize the bee
                new_bee = create_worker_bee(lower_bounds, upper_bounds, jornada, n)

                momios_combinados = get_momios(observer_bee, jornada, n)

                prob_acertar = get_prob_acertar(observer_bee, jornada, n)

                new_fitness_value = fitness(momios_combinados, prob_acertar)

                worker_bees[i] = (new_bee, new_fitness_value, 0) # Reset limit counter
                if print_progress:
                    print("Reinitialized Bee:", new_bee, "\tFitness:", new_fitness_value)

        # We can print the best solution found
        best_bee = max(worker_bees, key=lambda x: x[1])
        if print_progress:
            print("Best Bee Found:", best_bee[0], "\tFitness:", best_bee[1])
        HoF.append(best_bee)
        if iteration >= max_iterations:
            break
    return worker_bees, HoF

def calculate_acumulated_probabilities(probabilities):
    # Create acumulated probabilities
    acumulated_probabilities = []
    acum_sum = 0
    for p in probabilities:
        acum_sum += p
        acumulated_probabilities.append(acum_sum)
    return acumulated_probabilities

def create_observer_bee(acumulated_probabilities, worker_bees, lower_bounds, upper_bounds):
    # We select a worker bee based on the accumulated probabilities
    roullete_wheel_index = roullete_wheel(acumulated_probabilities)
    selected_bee = worker_bees[roullete_wheel_index][0].copy() # this is our 'i' # AQUI ESTABA EL PEDOOOOOO

    # Now that we have selected a bee, we can create a new observer bee based on it
    j = random.randint(0, n - 1)
    while True:
        k = random.randint(0, worker_bees_count - 1)
        if k != roullete_wheel_index: # Ensure we don't select the same bee
            break
    r2 = random.uniform(-1, 1) # Número aleatorio entre -1 y 1

    # Modify the selected dimension
    selected_bee[j] = selected_bee[j] + r2 * (selected_bee[j] - worker_bees[k][0][j])

    if selected_bee[j] < lower_bounds[j]:
        selected_bee[j] = lower_bounds[j]
    elif selected_bee[j] > upper_bounds[j]:
        selected_bee[j] = upper_bounds[j]

    selected_bee[j] = round(selected_bee[j]) # Get the closest integer <- NO FRACTIONS

    selected_bee = enforce_min_partidos(selected_bee, lower_bounds, upper_bounds, partidos_min_jornada) #Asegurar mínimo de partidos incluidos

    # We return both the bee and the index of the selected worker bee for reference
    return ((selected_bee, roullete_wheel_index)) # (observer bee, index of the selected worker bee)

def roullete_wheel(acumulated_probabilities):
    r = random.random()
    for i, p in enumerate(acumulated_probabilities):
        if r <= p:
            return i
    return len(acumulated_probabilities) - 1

def print_hall_of_fame(HoF):
    print("\n=== Final Best Solutions in Hall of Fame ===")
    for i, bee in enumerate(HoF):
        print("Iteration", i + 1, "-> Bee:", bee[0], "\tFitness:", bee[1])

# ===================================== MAIN PROGRAM ==========================================
# Primero cargamos los datos históricos de la Liga MX y usaremos nuestro modelo para poder conseguir la probalidad de cada partido individualmente
df = leer_historial_liga() # Luego compararemos las dos predicciones
# print(df.head()) # Verificamos que se haya cargado correctamente <-- Imprimer las primeras 5 lineas

jornadas_partidos = define_parley_games(df) # jornadas_partidos[0] = jornada 1, jornadas_partidos[1] = jornada 2, etc.

i = 1
for jornada in jornadas_partidos:
    print("Jornada ", i)
    i += 1
    n = len(jornada)
    lower_bounds = [0] * n # Se definen límites para cada uno de las jornadas, pues pueden tener más o menos partidos
    upper_bounds = [2] * n
    HoF = [] # Hall of Fame
    worker_bees = create_worker_bees(lower_bounds, upper_bounds, jornada, print_progress = False)
    final_worker_bees, HoF = beehive_algorithm(worker_bees, lower_bounds, upper_bounds, jornada, print_progress = False)
    print_hall_of_fame(HoF)
    print("\n")