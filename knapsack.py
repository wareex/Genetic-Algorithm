import numpy as np

# Function to evaluate the fitness of a solution (total value of selected items)
def evaluate_fitness(solution, values, weights, max_weight):
    total_value = np.sum(solution * values)
    total_weight = np.sum(solution * weights)
    if total_weight > max_weight:
        return 0  # Penalize solutions that exceed the weight constraint
    else:
        return total_value

# Function to generate an initial population of solutions
def initialize_population(population_size, num_items):
    return np.random.randint(2, size=(population_size, num_items))

# Function to perform selection (tournament selection)
def select_parents(population, values, weights, max_weight, tournament_size):
    selected_parents = []
    for _ in range(2):
        tournament_indices = np.random.choice(range(len(population)), size=tournament_size, replace=False)
        tournament = population[tournament_indices]
        tournament_fitness = np.array([evaluate_fitness(individual, values, weights, max_weight) for individual in tournament])
        best_parent_index = np.argmax(tournament_fitness)
        selected_parents.append(tournament[best_parent_index])
    return selected_parents

# Function to perform crossover (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Function to perform mutation (bit flip mutation)
def mutate(solution, mutation_rate):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if np.random.rand() < mutation_rate:
            mutated_solution[i] = 1 - mutated_solution[i]  # Flip the bit
    return mutated_solution

# Function to perform genetic algorithm
def genetic_algorithm(values, weights, max_weight, population_size, num_generations, tournament_size, mutation_rate):
    num_items = len(values)
    population = initialize_population(population_size, num_items)

    for generation in range(num_generations):
        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, values, weights, max_weight, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = np.array(new_population)

    best_solution_index = np.argmax([evaluate_fitness(individual, values, weights, max_weight) for individual in population])
    best_solution = population[best_solution_index]
    best_fitness = evaluate_fitness(best_solution, values, weights, max_weight)
    return best_solution, best_fitness

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
max_weight = 50
population_size = 100
num_generations = 1000
tournament_size = 5
mutation_rate = 0.1

best_solution, best_fitness = genetic_algorithm(values, weights, max_weight, population_size, num_generations, tournament_size, mutation_rate)
print("Best solution (selected items):", best_solution)
print("Best fitness (total value):", best_fitness)
