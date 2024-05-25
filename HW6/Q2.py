# Baktash Ansari
# 99521082
import random

# Define the fitness function
def fitness(root):
    # print("root: ",root)
    # Compute the value of the polynomial for this root
    y = sum(c * root**i for i, c in enumerate(coefficients))
    # print("value: ", y)
    
    # The fitness is the absolute value of y (since we want to minimize y)
    return abs(y)

def crossOver(population):
    new_population = []
    for i in range(10):
        samples = random.sample(population, 2)
        new_population.append((samples[0] + samples[1]) / 2)
    return population + new_population

def mutation(population):
    new_population = []
    for i in range(10):
        samples = random.sample(population, 1)
        new_population.append(samples[0] + random.uniform(-1, 1))  
    return population + new_population

# Define the polynomial coefficients
coefficients = list(map(float, input("Enter the Coefficients of your polynomial equation: ").split()))
coefficients = coefficients[::-1]

# Define the population size and the range of the roots
pop_size = 200
root_range = (-10, 10)

# Initialize the population with random values
population = [random.uniform(*root_range) for _ in range(pop_size)]


# Run the genetic algorithm
for generation in range(300):

    # Evaluate the fitness of the population
    fitnessValues = []
    for root in population:
        fitValue = fitness(root)
        fitnessValues.append((root, fitValue))
    
    # Create a new population
    new_population = []

    # sort the chromozones based on their fitness values
    sorted_tuples = sorted(fitnessValues, key=lambda x: x[1])

    # select the 30 percent of the top chromozones 
    threshold_index = int(len(sorted_tuples) * 0.3)
    children = [t[0] for t in sorted_tuples[:threshold_index]]

    # add them to old population
    new_population = children

    # Perform crossover
    new_population = crossOver(new_population)
    
    # Perform mutation
    new_population = mutation(new_population)
        
    # Replace the old population with the new population
    population = new_population

# Print the best solution
best_root = min(population, key=fitness)
print("Best solution: x = " + str(best_root))
