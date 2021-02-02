#!/usr/bin/env python3

# ----------------------------------------------------------------
# Copyright (c) 2020 Matthias Karlbauer, Sebastian Otte
# ----------------------------------------------------------------

__author__ = "Matthias Karlbauer, Sebastian Otte"


import numpy as np


class DEInterface :

    def create_empty_tensor(self) : 
        pass

    def create_individual(self) :
        pass

    def eval_fitness(self, candidate) :
        pass

    def generate_random_tensor(self) :
        pass

class DifferentialEvolution :



    def __init__(self, de_interface : DEInterface, pop_size = 5, F = 0.2, CR = 0.4) :
        print("init")
        self.de_interface = de_interface
        
        self.F = F
        self.CR = CR
        self.pop_size = pop_size

        self.initialize_population()
        self.evaluate_population()

    def initialize_population(self) :
        self.population = []

        for i in range(self.pop_size) :
            self.population.append(
                self.de_interface.create_individual()
            )

    def evaluate_population(self) :

        self.fitness = np.ones(self.pop_size) * float('inf')
        self.best_fitness = float('inf')
        self.best_i = -1

        for i in range(self.pop_size) :
            fitness_i = self.de_interface.eval_fitness(
                self.population[i]
            )
            self.fitness[i] = fitness_i
            if (fitness_i < self.best_fitness) :
                self.best_fitness = fitness_i
                self.best_i = i
    
    def generate_donor_tensor(self, i) :

        random_indices = np.random.choice(
            self.pop_size, 4, replace = False
        )
        index_shift = (self.pop_size - random_indices[0]) + i
        random_indices = (random_indices + index_shift) % self.pop_size

        _ ,r1, r2, r3 = random_indices

        donor = self.population[i] + self.F * (
            self.population[self.best_i] - self.population[r1]
        ) + self.F * (
            self.population[r2] - self.population[r3]
        )

        '''
        donor = self.population[r1] + self.F * (
            self.population[r2] - self.population[r3]
        )
        '''

        return donor

    def generate_trial_tensor(self, i, donor_tensor) :

        base_tensor = self.population[i]
        random_values = self.de_interface.generate_random_tensor()
        random_mask = random_values < self.CR

        trial_tensor = self.de_interface.copy_tensor(base_tensor)
        trial_tensor[random_mask] = donor_tensor[random_mask]
            
        return trial_tensor

    def get_best_solution(self) :
        if (self.best_i >= 0) :
            return self.de_interface.copy_tensor(self.population[self.best_i])
        else :
            return None

    def get_best_fitness(self) :
        return self.best_fitness

    def optimize_step(self) :

        for i in range(self.pop_size) :
            
            donor_tensor = self.generate_donor_tensor(i)
            trial_tensor = self.generate_trial_tensor(i, donor_tensor)

            fitness_i = self.de_interface.eval_fitness(
                trial_tensor
            )

            if (fitness_i < self.fitness[i]) :
                self.fitness[i] = fitness_i
                self.population[i] = trial_tensor

            if (fitness_i < self.best_fitness) :
                self.best_fitness = fitness_i
                self.best_i = i


class VectorInterface(DEInterface) :

    def copy_tensor(self, tensor) : 
        return np.copy(tensor)

    def create_individual(self) :
        return np.random.randn(2) * 0.1

    def eval_fitness(self, candidate) :
        x, y = candidate
        
        fxy = np.exp(-(
            ((x - 2)**2)/(20) + ((y + 3)**2)/(20)
        ))

        return 1.0 - fxy

    def generate_random_tensor(self) :
        return np.random.rand(2)


v = VectorInterface()
de = DifferentialEvolution(de_interface = v, pop_size = 5, F=0.4, CR=0.6)

generations = 1000

for g in range(generations) :
    de.optimize_step()
    print(de.get_best_fitness())




                    





