import os
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


def runGA(populationSize, genomeLength, crossoverRate, mutationRate, statistics):
    """
    The Genetic Algorithm
    """

    # Initial population
    maxRuns = 30  # Maximum number of generations that population will go through in a single run of the algorithm
    population = makePopulation(populationSize, genomeLength)

    # Run the genetic algorithm for a specified number of runs (generations)
    for run in range(maxRuns):

        # At each generation, compute the average and best fitness in the current population
        avg_fitness, best_fitness = evaluateFitness(population)
        message = f"        Generation {run}: average fitness {avg_fitness:.2f}, best fitness {best_fitness:.2f}\n"
        statistics.write(message)
        # print(message)

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


# Execute 30 runs and compute the average generation
def getAverageGenerations(populationSize, genomeLength, crossoverRate, mutationRate, statistics):

    totalGenerations = 0

    for run in range(30):
        message = f'   GA execution number: {run}. GA data:\n'
        statistics.write(message)
        print(f'GA execution number: {run}')
        generation = runGA(populationSize, genomeLength, crossoverRate, mutationRate, statistics)
        message = f'   Number of generations to get best genome at execution number {run}: {generation}\n\n'
        statistics.write(message)
        print(f'Number of generations to get best genome {generation}')
        totalGenerations += generation

    averageGenerations = totalGenerations / 30

    return averageGenerations


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    statistics_file = f'{root}/data/statistics.txt'

    # Record GA data
    with open(statistics_file, 'w') as statistics:
        # Run with crossover rate 0.7
        statistics.write(f'Crossover Rate 0.7\n')
        print(f'Crossover Rate 0.7')
        averageGenerations_07 = getAverageGenerations(50, 10, 0.7, 0.001, statistics)

        # Run with crossover rate 0
        statistics.write(f'Crossover Rate 0.0\n')
        print(f'\n\nCrossover Rate 0.0')
        averageGenerations_00 = getAverageGenerations(50, 10, 0, 0.001, statistics)

        statistics.write('Summary:\n')
        statistics.write(f'Average Generations for Crossover Rate 0.7 = {averageGenerations_07}\n')
        statistics.write(f'Average Generations for Crossover Rate 0.0 = {averageGenerations_00}\n')
