import random

class GeneticAlgorithm:
    """
    Genetic Algorithm designed to evolve a binary string to match a target.
    """

    def __init__(self, target_string, population_size=100, mutation_rate=0.1, crossover_rate=0.95):
        """
        Initializes the Genetic Algorithm with its parameters.

        Args:
            target_string (list[int]): The target binary string the algorithm will try to match.
            population_size (int): The number of individuals in the population per generation.
            mutation_rate (float): The probability (0.0 to 1.0) of a gene mutating.
            crossover_rate (float): The probability (0.0 to 1.0) that a crossover will occur for a pair of parents.
        """
        self.target_string = target_string
        self.target_length = len(target_string)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self._create_initial_population()

    def _create_individual(self):
        """
        Creates a single random individual (a binary string).

        Returns:
            list[int]: A list of 0s and 1s representing a chromosome.
        """
        return [random.randint(0, 1) for _ in range(self.target_length)]

    def _create_initial_population(self):
        """
        Creates the initial population of random individuals.

        Returns:
            list[list[int]]: A list of individuals.
        """
        return [self._create_individual() for _ in range(self.population_size)]

    def _calculate_fitness(self, individual):
        """
        Calculates the fitness of an individual by comparing it to the target string.
        The fitness score is the number of matching bits.

        Args:
            individual (list[int]): The chromosome to evaluate.

        Returns:
            int: The fitness score (higher is better).
        """
        score = 0
        for i in range(self.target_length):
            if individual[i] == self.target_string[i]:
                score += 1
        return score

    def _roulette_wheel_selection(self, fitness_scores):
        """
        Selects a single parent from the population using roulette wheel selection.
        Individuals with higher fitness have a higher probability of being selected.

        Args:
            fitness_scores (list[int]): A list of fitness scores for the entire population.

        Returns:
            list[int]: The selected parent individual.
        """
        total_fitness = sum(fitness_scores)
        
        # Handle the case where all fitness scores are 0 to avoid division by zero
        if total_fitness == 0:
            return random.choice(self.population)

        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        for i, fitness in enumerate(fitness_scores):
            current_sum += fitness
            if current_sum > selection_point:
                return self.population[i]
        
        # Fallback in case of floating point inaccuracies
        return self.population[-1]

    def _single_point_crossover(self, parent1, parent2):
        """
        Performs single-point crossover on two parents to create two children.

        Args:
            parent1 (list[int]): The first parent chromosome.
            parent2 (list[int]): The second parent chromosome.

        Returns:
            tuple[list[int], list[int]]: A tuple containing the two new child chromosomes.
        """
        # Decide if crossover should happen based on the crossover rate
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]  # Return copies of parents if no crossover

        # Choose a random crossover point (not at the ends)
        crossover_point = random.randint(1, self.target_length - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2

    def _mutate(self, individual):
        """
        Performs bit-flip mutation on an individual. Each bit has a chance
        to be flipped based on the mutation rate.

        Args:
            individual (list[int]): The chromosome to mutate.

        Returns:
            list[int]: The mutated chromosome.
        """
        mutated_individual = individual[:] # Create a copy to modify
        for i in range(self.target_length):
            if random.random() < self.mutation_rate:
                # Flip the bit
                mutated_individual[i] = 1 - mutated_individual[i]
        return mutated_individual

    def run(self, max_generations=500):
        """
        Runs the genetic algorithm for a specified number of generations or
        until the target string is found.

        Args:
            max_generations (int): The maximum number of generations to run.

        Returns:
            tuple: A tuple containing the best solution found, its fitness,
                   and the generation number it was found in.
        """
        print("Starting Genetic Algorithm...")
        print(f"Target: {''.join(map(str, self.target_string))}")

        for generation in range(max_generations):
            # 1. Calculate fitness for the entire population
            fitness_scores = [self._calculate_fitness(ind) for ind in self.population]

            # Find the best individual in the current generation
            best_fitness = 0
            best_individual = None
            for i in range(self.population_size):
                if fitness_scores[i] > best_fitness:
                    best_fitness = fitness_scores[i]
                    best_individual = self.population[i]
            
            # 2. Check for a solution
            if best_fitness == self.target_length:
                print(f"\nSolution found in generation {generation}!")
                return best_individual, best_fitness, generation
            
            print(f"Generation {generation:03d}: Best Fitness = {best_fitness}/{self.target_length} | "
                  f"Best Individual: {''.join(map(str, best_individual))}")

            # 3. Create the next generation
            new_population = []
            while len(new_population) < self.population_size:
                # 3a. Selection
                parent1 = self._roulette_wheel_selection(fitness_scores)
                parent2 = self._roulette_wheel_selection(fitness_scores)
                
                # 3b. Crossover
                child1, child2 = self._single_point_crossover(parent1, parent2)
                
                # 3c. Mutation
                mutated_child1 = self._mutate(child1)
                mutated_child2 = self._mutate(child2)
                
                new_population.append(mutated_child1)
                if len(new_population) < self.population_size:
                    new_population.append(mutated_child2)

            # Replace the old population with the new one
            self.population = new_population

        print(f"\nAlgorithm finished after {max_generations} generations.")
        final_fitness_scores = [self._calculate_fitness(ind) for ind in self.population]
        final_best_fitness = max(final_fitness_scores)
        final_best_individual = self.population[final_fitness_scores.index(final_best_fitness)]
        return final_best_individual, final_best_fitness, max_generations


def main():
    # --- Configuration ---
    TARGET_BINARY_STRING = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.1  # 1% chance to flip a bit
    CROSSOVER_RATE = 0.95 # 95% chance to perform crossover
    MAX_GENERATIONS = 500

    #Execution
    ga = GeneticAlgorithm(
        target_string=TARGET_BINARY_STRING,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE
    )
    
    best_solution, best_fitness, generation_found = ga.run(max_generations=MAX_GENERATIONS)
    
    # Results
    print("\nGenetic Algorithm Results ---")
    print(f"Target String:    {''.join(map(str, TARGET_BINARY_STRING))}")
    print(f"Best Solution Found: {''.join(map(str, best_solution))}")
    print(f"Fitness Score:       {best_fitness}/{len(TARGET_BINARY_STRING)}")
    print(f"Found in Generation: {generation_found}")


if __name__ == '__main__':
    main()