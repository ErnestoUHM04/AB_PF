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

# Una abeja no debe tener todos los partidos de la jornada obligatoriamente, puede tener solo algunos.
partidos_min_jornada = 2 # Mínimo de partidos que debe tener una abeja para considerarse válida.  <-- CAMBIAR SI ESTOY MAL

# Hay 17 jornadas en la temporada regular de la Liga MX.
jornadas = 17

# P(asertar todos los partidos del parley) = P(p1) * P(p2) * ... * P(p9) <-- Usar el modelo entrenado previamente para obtener esta probabilidad.

# M(momios de la apuesta) = M(p1) * M(p2) * ... * M(p9)

# La función Fitness de una abeja será:
# f = M(momios de la apuesta) / P(asertar todos los partidos del parley)

# BUSCAMOS MAXIMIZAR LA FUNCIÓN FITNESS

###### Parameters ###### <----- para un BeeHive Algorithm
# Número de variables
n = 9
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

########################

def leer_historial_liga():
    # Cargar datos históricos de partidos de la Liga MX desde un archivo CSV
    df = pd.read_csv('data/datos_normalizados.csv')
    return df

# Primero cargamos los datos históricos de la Liga MX y usaremos nuestro modelo para poder conseguir la probalidad de cada partido individualmente
# Luego compararemos las dos predicciones
df = leer_historial_liga()
# print(df.head()) # Verificamos que se haya cargado correctamente <-- Imprimer las primeras 5 lineas

def define_parley_games(df, umbral_diferencia = 0.15):
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

        # También conseguimos el mejor resultado según el modelo    <---- ESTO SERÁ A LO QUE APOSTAREMOS
        mejor_resultado_model = max(prob_local_model, prob_empate_model, prob_visita_model)
        if mejor_resultado_model == prob_local_model:
            mejor_resultado_model_str = 'Local'
        elif mejor_resultado_model == prob_empate_model:
            mejor_resultado_model_str = 'Empate'
        else:
            mejor_resultado_model_str = 'Visita'

        # Guardamos los momios de los partidos elegidos    <--- ESTO SERÁ PARA LA FUNCIÓN FITNESS
        momio_L = df.iloc[i]['Momio_L_norm']
        momio_V = df.iloc[i]['Momio_V_norm']
        momio_E = df.iloc[i]['Momio_E_norm']

        # Si la diferencia entre las probabilidades es menor al umbral, entonces incluimos el partido en el parley
        if prob_local_diff < umbral_diferencia and prob_empate_diff < umbral_diferencia and prob_visita_diff < umbral_diferencia:
            # Si la diferencia es pequeña, entonces incluimos el partido en el parley
            jornada = df.iloc[i]['Jornada']
            if jornada == '1':
                jornada1.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '2':
                jornada2.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '3':
                jornada3.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '4':
                jornada4.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '5':
                jornada5.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '6':
                jornada6.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '7':
                jornada7.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '8':
                jornada8.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '9':
                jornada9.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '10':
                jornada10.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '11':
                jornada11.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '12':
                jornada12.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '13':
                jornada13.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '14':
                jornada14.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '15':
                jornada15.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '16':
                jornada16.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            elif jornada == '17':
                jornada17.append((1, local, visitante, mejor_resultado_model_str, momio_L, momio_V, momio_E))
            else:
                pass  # Jornada no válida, no hacemos nada
        else: # ponemos un 0 indicando que no se incluye el partido en el parley
            if jornada == '1':
                jornada1.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '2':
                jornada2.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '3':
                jornada3.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '4':
                jornada4.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '5':
                jornada5.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '6':
                jornada6.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '7':
                jornada7.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '8':
                jornada8.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '9':
                jornada9.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '10':
                jornada10.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '11':
                jornada11.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '12':
                jornada12.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '13':
                jornada13.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '14':
                jornada14.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '15':
                jornada15.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            elif jornada == '16':
                jornada16.append((0, local, visitante, 'No incluido',momio_L, momio_V, momio_E))
            else:
                pass  # Jornadas no válidas
    """
    # Imprimimos los partidos seleccionados para cada jornada
    print("\nJornada 1:", jornada1)
    print("Jornada 2:", jornada2)
    print("Jornada 3:", jornada3)
    print("Jornada 4:", jornada4)
    print("Jornada 5:", jornada5)
    print("Jornada 6:", jornada6)
    print("Jornada 7:", jornada7)
    print("Jornada 8:", jornada8)
    print("Jornada 9:", jornada9)
    print("Jornada 10:", jornada10)
    print("Jornada 11:", jornada11)
    print("Jornada 12:", jornada12)
    print("Jornada 13:", jornada13)
    print("Jornada 14:", jornada14)
    print("Jornada 15:", jornada15)
    print("Jornada 16:", jornada16)
    print("Jornada 17:", jornada17)
    """
    return [jornada1, jornada2, jornada3, jornada4, jornada5, jornada6, jornada7, jornada8, jornada9, jornada10, jornada11, jornada12, jornada13, jornada14, jornada15, jornada16, jornada17]

jornadas_partidos = define_parley_games(df) # jornadas_partidos[0] = jornada 1, jornadas_partidos[1] = jornada 2, etc.


"""""
# CAMBIAR ESTA FUNCIÓN
def create_worker_bee(lower_bounds, upper_bounds):
    bee = []
    for i in range(n):
        r1 = random.random() # Número aleatorio entre 0 y 1
        xi = lower_bounds[i] + r1 * (upper_bounds[i] - lower_bounds[i])
        xi = round(xi) # Redondear al entero más cercano
        bee.append(xi)
    return bee

# CAMBIAR ESTA FUNCIÓN
def create_worker_bee_new(lower_bounds, upper_bounds):
    bee = []
    for i in range(n):
        xi = random.randint(lower_bounds[i], upper_bounds[i])
        bee.append(xi)
    return bee

# CAMBIAR ESTA FUNCIÓN
def create_worker_bees(lower_bounds, upper_bounds, values, weights, print_progress=True):
    # Create worker bees
    worker_bees = []
    for i in range(worker_bees_count):
        while True:
            worker_bee = create_worker_bee(lower_bounds, upper_bounds)
            #worker_bee = create_worker_bee_new(lower_bounds, upper_bounds)

            fitness_value = fitness(worker_bee, values)
            weight_value = weight(worker_bee, weights)

            if weight_value <= max_capacity:
                break

        if print_progress:
            print("Worker Bee:", worker_bee, "\tValue:", fitness_value, "\tWeight:", weight_value)
        # Remember initialization of the limit counter for each worker bee
        worker_bees.append((worker_bee, fitness_value, weight_value, 0)) # (bee, fitness, weight, limit_counter)
    return worker_bees

# CAMBIAR ESTA FUNCIÓN
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

    # We return both the bee and the index of the selected worker bee for reference
    return ((selected_bee, roullete_wheel_index)) # (observer bee, index of the selected worker bee)

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

    return bee

def roullete_wheel(acumulated_probabilities):
    r = random.random()
    for i, p in enumerate(acumulated_probabilities):
        if r <= p:
            return i
    return len(acumulated_probabilities) - 1

# CAMBIAR ESTA FUNCIÓN
def fitness(bee, values):
    total_value = sum(bee[i] * values[i] for i in range(n))
    return total_value

# CAMBIAR ESTA FUNCIÓN.    <--- Posbilemente eliminar
def weight(bee, weights):
    total_weight = sum(bee[i] * weights[i] for i in range(n))
    return total_weight

# CAMBIAR ESTA FUNCIÓN
def calculate_acumulated_probabilities(probabilities):
    # Create acumulated probabilities
    acumulated_probabilities = []
    acum_sum = 0
    for p in probabilities:
        acum_sum += p
        acumulated_probabilities.append(acum_sum)
    return acumulated_probabilities

# CAMBIAR ESTA FUNCIÓN
def beehive_algorithm(worker_bees, lower_bounds, upper_bounds, values, weights, print_progress=True):
    HoF = [] # Hall of Fame
    iteration = 0
    while True: # We will run this for max_iterations
        iteration += 1
        if print_progress:
            print("\n--- Iteration", iteration, "---")
        # We need them to work now :)
        for i in range(worker_bees_count):
            bee, fitness_value, weight_value, limit_counter = worker_bees[i]
            while True:
                new_bee = worker_bee_search(i, bee.copy(), lower_bounds, upper_bounds)

                new_fitness_value = fitness(new_bee, values)
                new_weight_value = weight(new_bee, weights)

                if new_weight_value <= max_capacity:
                    break

            # Check if the new bee is better
            if new_fitness_value > fitness_value: # If it is, we update the bee, and reset the counter
                worker_bees[i] = (new_bee, new_fitness_value, new_weight_value, 0) # Reset limit counter
            else: # If not, we DO NOT update the bee, and we increment the counter by one
                limit_counter += 1
                worker_bees[i] = (bee, fitness_value, weight_value, limit_counter)

            if print_progress:
                print("Worker Bee:", worker_bees[i][0], "\tValue:", worker_bees[i][1], "\tWeight:", worker_bees[i][2])

        # After they finish working, we evaluate their fitness and create probabilities for observer bees
        # After the worker bee, create observer bees based on the best solutions found by worker bees
        sum_fitness = sum(bee[1] for bee in worker_bees)
        probabilities = [bee[1] / sum_fitness for bee in worker_bees]

        acumulated_probabilities = calculate_acumulated_probabilities(probabilities)

        # Create observer bees
        observer_bees = [] # We select the best solutions found by observer_bees via the waggle dance
        for i in range(observer_bees_count):
            while True:
                observer_bee, followed_bee_index = create_observer_bee(acumulated_probabilities, worker_bees, lower_bounds, upper_bounds)
                # We evaluate the observer bee
                fitness_value = fitness(observer_bee, values)
                weight_value = weight(observer_bee, weights)

                if weight_value <= max_capacity:
                    break

            # We still check if the observer bee is better than the followed worker bee
            followed_bee = worker_bees[followed_bee_index]
            # if they are better than the worker bees, then we replace the worker bees with them
            if fitness_value > followed_bee[1]: # If it is, we replace the worker bee with the observer bee
                worker_bees[followed_bee_index] = (observer_bee, fitness_value, weight_value, 0) # Reset limit counter
            else:
                # We do not replace the worker bee, just add one to the limit counter
                worker_bees[followed_bee_index] = (followed_bee[0], followed_bee[1], followed_bee[2], followed_bee[3] + 1) # Increment limit counter +1

            if print_progress:
                print("Observer Bee:", observer_bee, "\tValue:", fitness_value, "\tWeight:", weight_value)
            observer_bees.append((observer_bee, fitness_value, weight_value))

        # Now we check for any bee that has exceeded the limit and reinitialize it
        for i in range(worker_bees_count):
            bee, fitness_value, weight_value, limit_counter = worker_bees[i]

            # THIS IS THE EXPLORER BEE PHASE
            if limit_counter >= limit:
                # Reinitialize the bee
                while True:
                    new_bee = create_worker_bee(lower_bounds, upper_bounds)
                    #new_bee = create_worker_bee_new(lower_bounds, upper_bounds)

                    new_fitness_value = fitness(new_bee, values)
                    new_weight_value = weight(new_bee, weights)

                    if new_weight_value <= max_capacity:
                        break

                worker_bees[i] = (new_bee, new_fitness_value, new_weight_value, 0) # Reset limit counter
                if print_progress:
                    print("Reinitialized Bee:", new_bee, "\tValue:", new_fitness_value, "\tWeight:", new_weight_value)

        # We can print the best solution found
        best_bee = max(worker_bees, key=lambda x: x[1])
        if print_progress:
            print("Best Bee Found:", best_bee[0], "\tValue:", best_bee[1], "\tWeight:", best_bee[2])
        HoF.append(best_bee)
        if iteration >= max_iterations:
            break
    return worker_bees, HoF

# CAMBIAR ESTA FUNCIÓN
def print_hall_of_fame(HoF):
    print("\n=== Final Best Solutions in Hall of Fame ===")
    for i, bee in enumerate(HoF):
        print("Iteration", i + 1, "-> Bee:", bee[0], "\tValue:", bee[1], "\tWeight:", bee[2])


lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0] # <------------ Para este problema, CAMBIAR ESTOS VALORES
upper_bounds = [2, 2, 2, 2, 2, 2, 2, 2, 2] # Irán del 0 al 2 (0 = gana local, 1 = empate, 2 = gana visitante)


worker_bees = create_worker_bees(lower_bounds, upper_bounds, values, weights, print_progress = False)

final_worker_bees, HoF = beehive_algorithm(worker_bees, lower_bounds, upper_bounds, values, weights, print_progress = False)

print_hall_of_fame(HoF)
"""