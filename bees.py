# Cornejo Morales Paola
# Hernándz Martínez Ernesto Ulises

# Usar el algoritmo de colonia de abejas para resolver el problema de la mochila que se asignó en la práctica 1.
# Recuerden utilizar los límites de las variables de acuerdo con problema, respetando los límites de 3 Love Potions y 2 Skiving Snackbox.
# Para el resto de productos los límites están entre (0,10).

import random

###### Parameters ######
# Número de variables
n = 7
# Tamaño del enjambre = 40
beehive_size = 40
# abeja obreras = 20
worker_bees_count = beehive_size // 2
#print("Número de abejas obreras:", worker_bees_count)
# abejas Observadoras = 20
observer_bees_count = beehive_size // 2
#print("Número de abejas observadoras:", observer_bees_count)
# límite = 5
limit = 5
#limit = (beehive_size * n) // 2
# iteraciones = 50
max_iterations = 50
# Capacidad de la mochila : 30 lb
max_capacity = 30

# x1 = Decoy Detonators (0-10)  - 4 lb      - $ 10
#x1_lower_bound = 0
#x1_upper_bound = 10
# x2 = Love Potions (3-10)      - 2 lb      - $ 8
#x2_lower_bound = 3
#x2_upper_bound = 10
# x3 = Extendable Ears (0-10)   - 5 lb      - $ 12
#x3_lower_bound = 0
#x3_upper_bound = 10
# x4 = Skiving Snackbox (2-10)  - 5 lb      - $ 6
#x4_lower_bound = 2
#x4_upper_bound = 10
# x5 = Fever Fudge (0-10)       - 2 lb      - $ 3
#x5_lower_bound = 0
#x5_upper_bound = 10
# x6 = Puking Pastilles (0-10)  - 1.5 lb    - $ 2
#x6_lower_bound = 0
#x6_upper_bound = 10
# x7 = Nosebleed Nougat (0-10)  - 1 lb      - $ 2
#x7_lower_bound = 0
#x7_upper_bound = 10

def create_worker_bee(lower_bounds, upper_bounds):
    bee = []
    for i in range(n):
        r1 = random.random() # Número aleatorio entre 0 y 1
        xi = lower_bounds[i] + r1 * (upper_bounds[i] - lower_bounds[i])
        xi = round(xi) # Redondear al entero más cercano
        bee.append(xi)
    return bee

def create_worker_bee_new(lower_bounds, upper_bounds):
    bee = []
    for i in range(n):
        xi = random.randint(lower_bounds[i], upper_bounds[i])
        bee.append(xi)
    return bee

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

def fitness(bee, values):
    total_value = sum(bee[i] * values[i] for i in range(n))
    return total_value

def weight(bee, weights):
    total_weight = sum(bee[i] * weights[i] for i in range(n))
    return total_weight

def calculate_acumulated_probabilities(probabilities):
    # Create acumulated probabilities
    acumulated_probabilities = []
    acum_sum = 0
    for p in probabilities:
        acum_sum += p
        acumulated_probabilities.append(acum_sum)
    return acumulated_probabilities

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

def print_hall_of_fame(HoF):
    print("\n=== Final Best Solutions in Hall of Fame ===")
    for i, bee in enumerate(HoF):
        print("Iteration", i + 1, "-> Bee:", bee[0], "\tValue:", bee[1], "\tWeight:", bee[2])


lower_bounds = [0, 3, 0, 2, 0, 0, 0]
upper_bounds = [10, 10, 10, 10, 10, 10, 10]
#upper_bounds = [9, 9, 9, 9, 9, 9, 9]

values = [10, 8, 12, 6, 3, 2, 2]
weights = [4, 2, 5, 5, 2, 1.5, 1]

worker_bees = create_worker_bees(lower_bounds, upper_bounds, values, weights, print_progress = False)

final_worker_bees, HoF = beehive_algorithm(worker_bees, lower_bounds, upper_bounds, values, weights, print_progress = False)

print_hall_of_fame(HoF)