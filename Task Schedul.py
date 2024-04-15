import random

# Define task scheduling problem parameters
num_tasks = 10
num_processors = 3
task_duration_range = (1, 10)

# Generate random tasks with durations
tasks = [(i, random.randint(*task_duration_range)) for i in range(num_tasks)]

def generate_initial_population(population_size):
    population = []
    for _ in range(population_size):
        schedule = [random.randint(0, num_processors - 1) for _ in range(num_tasks)]
        population.append(schedule)
    return population

def fitness(schedule):
    processor_times = [0] * num_processors
    for task, processor in enumerate(schedule):
        processor_times[processor] += tasks[task][1]
    return max(processor_times)

def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_tasks - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(schedule, mutation_rate):
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            schedule[i] = random.randint(0, num_processors - 1)
    return schedule

def select_parents(population, tournament_size):
    selected_parents = []
    for _ in range(2):
        tournament = random.sample(population, tournament_size)
        best_parent = min(tournament, key=lambda x: fitness(x))
        selected_parents.append(best_parent)
    return selected_parents

def genetic_algorithm(population_size, generations, tournament_size, mutation_rate):
    population = generate_initial_population(population_size)
    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
    best_schedule = min(population, key=lambda x: fitness(x))
    return best_schedule, fitness(best_schedule)

best_schedule, min_time = genetic_algorithm(population_size=100, generations=100, tournament_size=5, mutation_rate=0.1)
print("Best Schedule:", best_schedule)
print("Minimum Time:", min_time)
