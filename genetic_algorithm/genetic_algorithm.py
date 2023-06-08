import random


def randomGenome(length):
    """
    Generates a random genome of a given length
    """

    genome = []
    for _ in range(length):
        genome.append(random.randint(0, 1))

    return genome


def makePopulation(size, length):
    """
    Returns a new randomly created population of the specified size,
    represented as a list of genomes of the specified length.
    """

    population = []
    for _ in range(size):
        population.append(randomGenome(length))

    return population


def fitness(genome):
    """
    Evaluates the fitness of a genome: f(x) = number of ones in x, where x is a genome of length 10
    """

    return sum(genome)


def evaluateFitness(population):
    """
    Evaluate the average fitness of a population and the fitness of the best individual
    """

    fitnesses = []
    for individual in population:
        fitnesses.append(fitness(individual))

    avg_fitness = sum(fitnesses) / len(fitnesses)
    best_fitness = max(fitnesses)

    return avg_fitness, best_fitness


def selectPair(population):
    """
    Select and return two genomes from the given population using fitness-proportionate selection
    """

    weights = []
    for individual in population:
        weights.append(fitness(individual))

    return random.choices(population, weights=weights, k=2)


def crossover(genome1, genome2):
    """
    Return two new genomes produced by crossing over the given genomes at a random crossover point
    """

    if random.random() < 0.7:  # crossover rate
        point = random.randint(1, len(genome1) - 1)
        genome1[point:], genome2[point:] = genome2[point:], genome1[point:]

    return genome1, genome2


def mutate(genome, mutationRate):
    """
    Returns a new mutated version of the given genome
    """

    for i in range(len(genome)):
        if random.random() < mutationRate:  # mutation rate
            genome[i] = 1 if genome[i] == 0 else 0
    return genome


def runGA(populationSize, genomeLength, crossoverRate, mutationRate, maxRuns):
    """
    The Genetic Algorithm
    """

    # Initial population
    population = makePopulation(populationSize, genomeLength)

    # Run the genetic algorithm for a specified number of runs (generations)
    for run in range(maxRuns):

        # At each generation, compute the average and best fitness in the current population
        avg_fitness, best_fitness = evaluateFitness(population)
        print(f"Generation {run}: average fitness {avg_fitness:.2f}, best fitness {best_fitness:.2f}")

        # If we find a genome with with all 1s, return the current run (generation)
        if best_fitness == genomeLength:  # we found the best individual
            return run

        # Create a new population for the next generation
        new_population = []
        for _ in range(populationSize // 2):
            # Get two individuals from the current population based on their fitness
            genome1, genome2 = selectPair(population)

            # If the random number is less than crossoverRate, perform crossover between the selected individuals
            if random.random() < crossoverRate:
                genome1, genome2 = crossover(genome1, genome2)

            # Execute mutation on the genomes
            genome1 = mutate(genome1, mutationRate)
            genome2 = mutate(genome2, mutationRate)

            # Add the mutated/crossed-over genomes to the new population
            new_population.extend([genome1, genome2])

        # Replace the old population with the new population for the next run (generation)
        population = new_population

    # If we reach the maxRuns and did not found the best solution (all 1s), terminate and return maxRuns
    return maxRuns


output = runGA(50, 10, 0.7, 0.001, 30)
print(output)