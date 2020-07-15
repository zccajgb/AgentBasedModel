import numpy as np
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp
from Receptor import Receptor
from Nanoparticle import Nanoparticle
from Ligand import Ligand
import pandas as pd
from numba import njit
from random import shuffle
from Visualiser import visualiser
from BaseAgent import BaseAgent


class Master(BaseAgent):
    ''' This class is in chanrge of controlling the system.
        We have the __init__ method, which is run when we first start up the calculation and sets up the system.

        Then we have two types of methods:
         i) ones which control the system - these start with letters
         ii) helper methods, these are bits of code that have been extracted into a method to simplify things or so they can be reused. These start with underscores _

        Control methods:
            i) create_... just creates the items in question
            ii) run: this runs the simulation
            iii) step: this moves the entire system forward one step
            iv) try_to_bind: this is called when a ligand and receptor are close to each other, and sees if they can form a bond.
    '''
    def __init__(self, dimension, binding_energy, time_unit, number_of_receptors, receptor_length, number_of_nanoparticles,
                 nanoparticle_radius, number_of_ligands, ligand_length, binding_distance, receptor_radius):
        self.agents = []  # create an empty list of agents
        self.dimension = dimension
        self.points = []
        self.binding_energy = binding_energy
        self.time_unit = time_unit
        self.number_of_receptors = number_of_receptors
        self.receptor_length = receptor_length
        self.number_of_nanoparticles = number_of_nanoparticles
        self.number_of_ligands = number_of_ligands
        self.nanoparticle_radius = nanoparticle_radius
        self.ligand_length = ligand_length
        self.binding_distance = binding_distance
        self.receptor_radius = receptor_radius
        self.collision = False
        self.coverage = [0]
        self.time = 0
        self.bound_nanoparticles = 0

    def create_nanoparticles_and_ligands(self): 
        total_radius = self.nanoparticle_radius + self.ligand_length  
        upper_limit = self.dimension - total_radius
        for i in range(self.number_of_nanoparticles):
            agent_id = f'Nanoparticle {i}'
            nanoparticle_cartesean_position = self._initialise_nanoparticle_position(total_radius, upper_limit)
            nanoparticle = Nanoparticle(agent_id, nanoparticle_cartesean_position, self.number_of_ligands, self.nanoparticle_radius,
                                        self.ligand_length, self.dimension, binding_energy=self.binding_energy,
                                        time_unit=self.time_unit)
            
            self.agents.append(nanoparticle) 

    def create_receptors(self):
        #TODO there is nothing that stops two receptors being created arbitrarily close to each other
        for i in range(self.number_of_receptors):  
            receptor_id = f'Receptor {i}'
            base_position = np.array([np.random.uniform(self.receptor_length, self.dimension - self.receptor_length), np.random.uniform(self.receptor_length, self.dimension - self.receptor_length), 0]) 
            receptor = Receptor(receptor_id, base_position, self.receptor_length, self.dimension, self.binding_energy, self.nanoparticle_radius, self.ligand_length)
            self.agents.append(receptor) 
    
    def run(self, steps):
        '''this method runs the simulation, looping through each step'''
        for i in range(steps):
            self.time += 1
            self.step()
            for agent in self.agents:
                if isinstance(agent, Nanoparticle):
                    if agent.bound:
                        self.bound_nanoparticles += 1
            self.coverage.append(self.calculate_surface_coverage(self.bound_nanoparticles))
            
        # visualiser(self)
        number_of_bound_receptors = 0
        for i in self.agents:
            if isinstance(i, Receptor) and i.bound is not None:
                number_of_bound_receptors += 1
        print(f'There are {self.bound_nanoparticles} nanoparticles bound to the surface')
        print(f'There are {number_of_bound_receptors} receptors bound to nanoparticles')

    def step(self):
        length_of_random_numbers = self.number_of_nanoparticles + 2 * self.number_of_receptors
        random_numbers = list(np.random.normal(size = (length_of_random_numbers, 3)))
        nanoparticle_positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        receptor_positions = [agent.position for agent in self.agents if isinstance(agent, Receptor)]

        max_seperation_for_receptor_to_react = self.nanoparticle_radius + self.ligand_length + self.binding_distance
        
        for agent in self.agents:
            if isinstance(agent, Nanoparticle):
                nanoparticles_except_current = [posn for posn in nanoparticle_positions if posn is not agent.position]
                agent.step(random_numbers.pop(), nanoparticles_except_current, receptor_positions)
            
            elif isinstance(agent, Receptor):
                receptors_except_current = [receptor_position for receptor_position in receptor_positions if receptor_position is not agent.position] 
                agent.step(random_numbers.pop(), random_numbers.pop(), receptors_except_current, nanoparticle_positions)
                
                is_receptor_bound = agent.bound is not None
                if is_receptor_bound: 
                    self.try_to_unbind(agent)
                else:
                    nanoparticles = [n for n in self.agents if isinstance(n, Nanoparticle)]
                    for n in nanoparticles:
                        is_receptor_close_enough_to_react = self.distance(agent.position, n.position) < max_seperation_for_receptor_to_react
                        if is_receptor_close_enough_to_react:
                            self.try_to_bind(n, agent)

    def try_to_bind(self, nanoparticle, receptor):
        if nanoparticle.agent_id == receptor.agent_id: return
        
        is_receptor_already_bound = receptor.bound is not None
        if is_receptor_already_bound: return

        for ligand in nanoparticle.ligands:
            is_ligand_already_bound = ligand.bound is not None
            if is_ligand_already_bound: continue

            is_too_far_away_to_bind = self.distance(ligand.position, receptor.position) > self.binding_distance
            if is_too_far_away_to_bind: continue
                
            if self._metropolis_algorithm_for_binding(self.binding_energy):
                ligand.bound = receptor
                receptor.bound = ligand
                nanoparticle.bound = True
                return

    def try_to_unbind(self, receptor):
        if self._metropolis_algorithm_for_unbinding(self.binding_energy):
            receptor.bound.bound = None
            receptor.bound = None

    def _initialise_nanoparticle_position(self, total_radius, upper_limit):
        while True:
            nanoparticle_cartesean_position = np.array([np.random.uniform(total_radius, upper_limit), np.random.uniform(total_radius, upper_limit),np.random.uniform(total_radius, upper_limit)])  # 3D cube cartesian system - rectangular coordinates
            if self._check_space_available_nanoparticle(nanoparticle_cartesean_position):
                break
        return nanoparticle_cartesean_position
        
    def _check_space_available(self, current_agent_position, current_agent_radius):
        ''' Returns true is there is space available to make the move, returns false otherwise '''
        nanoparticle_positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        min_allowed_separation_nanoparticles = self.nanoparticle_radius + self.ligand_length + current_agent_radius
      
        for i in nanoparticle_positions:
            seperation = self.distance(current_agent_position, i)
            is_nanoparticle_too_close = seperation < min_allowed_separation_nanoparticles
            if is_nanoparticle_too_close:
                return False
    
        receptor_positions = [agent.position for agent in self.agents if isinstance(agent, Receptor)]
        min_allowed_seperation_receptors = self.receptor_radius + current_agent_radius
        for i in receptor_positions:
            seperation = self.distance(current_agent_position, i)
            is_nanoparticle_too_close_to_receptors = seperation < min_allowed_seperation_receptors
            if is_nanoparticle_too_close_to_receptors:
                return False 
        
        return True

    def _calculate_surface_coverage(self, n):
        self.surface_coverage = n * ((4 * (self.nanoparticle_radius + self.ligand_length) ** 2) / self.dimension ** 2)
        return self.surface_coverage



