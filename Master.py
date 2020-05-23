"""
Next Steps
We have designed a simple agent based model, that doesnt really teach us anything. We can improve this in multiple ways to start building up something intelligent:

make it 3D
    have the movement be more realistic:
        rather than moving on a 1d grid, use physically accurate representations of movement, e.g. brownian motion
        look at the way we generate random numbers, does this reflect physics?

make the interaction criteria cleverer:
    instead of assuming two agents in the same position will interact, have two agents in a certain distance able to interact
    make a reaction dependent on a probability (i.e. to make a reaction 50% likely, generate a random number thats either 1 or zero)
        we can then make this probability physically realistic (see Metropolis algorithm)

have two different agents types, one representing the nanoparticle, one representing the surface
    initially have the surface stationary, and have the nanoparticle move

increase the complexity of the agents, to add in ligands and receptors
    now the nanoparticle doesnt bind to the surface, but the ligand binds with the receptors
    maybe keep the surface/nanoparticle stationary, but allow receptors to move (remembering they're bound to the surface by their tether so have restricted movement)
    maybe then allow the nanoparticle to move, so the ligands can move within their tether length

create more complex models with different ligands and different receptors

We also then need to think about how we get data out. Our current model just stops and prints out "Collision". This obviously isn't very useful.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp
from ReceptorModel import Receptor
from NanoparticleModel import Nanoparticle
from LigandModel import Ligand
import pandas as pd
from numba import njit
from random import shuffle
from Visualiser import visualiser


class Master:
    def __init__(self, dimension, binding_energy, time_unit, number_of_receptors, receptor_length, number_of_nanoparticles,
                 nanoparticle_radius, number_of_ligands, ligand_length, binding_distance):
        self.agents = []  # create an empty list of agents
        self.collision = False
        self.count = 0
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

    def create_nanoparticles_and_ligands(self):  # Adds nanoparticles into the system
        true_radius = self.nanoparticle_radius + self.ligand_length  # The maximum radius of the nanoparticle including the ligands
        upper_limit = self.dimension - true_radius
        for i in range(self.number_of_nanoparticles):  # loop from 0 to number of agents
            agent_id = f'Nanoparticle {i}'
            while True:
                nanoparticle_position_xyz = np.array(
                    [np.random.uniform(true_radius, upper_limit), np.random.uniform(true_radius, upper_limit),
                     np.random.uniform(true_radius, upper_limit)])  # 3D cube cartesian system - rectangular coordinates
                if self.is_space_available_nanoparticle(nanoparticle_position_xyz):
                    break
                else:
                    continue
            nanoparticle = Nanoparticle(agent_id, nanoparticle_position_xyz, self.number_of_ligands, self.nanoparticle_radius,
                                        self.ligand_length, self.dimension, binding_energy=self.binding_energy,
                                        time_unit=self.time_unit)
            # print(f'Created {agent_id}')
            self.agents.append(nanoparticle)  # add agent to list of agents

    def is_space_available_nanoparticle(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        nanoparticle_positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        separation = (self.nanoparticle_radius + self.ligand_length) * 2
        for i in nanoparticle_positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                count += 1
            else:
                return False
        if count == len(nanoparticle_positions):
            receptor_positions = [agent.position for agent in self.agents if isinstance(agent, Receptor)]
            if len(receptor_positions) > 0:
                separation2 = 7  # receptor must be at least a radius away from receptor
                count = 0
                for i in receptor_positions:
                    distance = self.distance(attempt, i)
                    if distance >= separation2:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                        count += 1
                    else:
                        return False  # Repulsive potential not exceede
                if count == len(receptor_positions):
                    return True  # Returns True when there is space available
            else:
                return True  # if no receptors in the system

    def create_receptors(self):  # Adds receptors into the system
        for i in range(self.number_of_receptors):  # loop from 0 to number of agents
            receptor_id = f'Receptor {i}'
            base_position = np.array([np.random.uniform(0, self.dimension), np.random.uniform(0, self.dimension), 0])  # 3D cube cartesian system - rectangular coordinates
            receptor = Receptor(receptor_id, base_position, self.receptor_length, self.dimension, self.binding_energy, self.nanoparticle_radius, self.ligand_length)
            self.agents.append(receptor)  # add agent to list of agents

    def is_space_available_receptor(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        receptor_positions = [agent.base_position for agent in self.agents if isinstance(agent, Receptor)]
        separation = self.number_of_receptors * 0.025
        for i in receptor_positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the receptor is a certain distance from the other nanoparticles
                count += 1
            else:
                return False
        if count == len(receptor_positions):
            nanoparticle_positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
            if len(nanoparticle_positions) > 0:
                separation2 = self.nanoparticle_radius + self.ligand_length  # nanoparticle must be at least a radius away from receptor
                count = 0
                for i in nanoparticle_positions:
                    distance = self.distance(attempt, i)
                    if distance >= separation2:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                        count += 1
                    else:
                        return False  # Repulsive potential not exceede
                if count == len(nanoparticle_positions):
                    return True  # Returns True when there is space available
            else:
                return True  # if no nanoparticles in the system

    def run(self, steps):
        self.coverage = [0]
        self.time = 0
        plateau_count = 0
        self.bound_nanoparticles = 0
        for i in range(steps):  # loop through number of steps
            self.time += 1
            self.step()  # run one step
            bound_nanoparticles = 0
            for agent in self.agents:
                if isinstance(agent, Nanoparticle):
                    if agent.bound is True:
                        bound_nanoparticles += 1  # Number of nanoparticles bound to the cell surface
            self.coverage.append(self.calculate_surface_coverage(bound_nanoparticles))
            '''if you wish experiment to stop at a plateau + visually see the interaction'''
            if i == steps - 1:  # or i == steps/2:
                visualiser(self)
        for agent in self.agents:
            if isinstance(agent, Nanoparticle):
                if agent.bound is True:
                    self.bound_nanoparticles += 1  # Number of nanoparticles bound to the cell surface
        self.calculate_surface_coverage(self.bound_nanoparticles)
        count1 = 0
        count2 = 0
        for i in self.agents:
            if isinstance(i, Nanoparticle) and i.bound is True:
                count1 += 1
            elif isinstance(i, Receptor) and i.bound is not None:
                count2 += 1
        print(f'There are {count1} nanoparticles bound to the surface')
        print(f'There are {count2} receptors bound to nanoparticles')

    def step(self):
        list_of_nanoparticle_arrays = list(np.random.normal(size=(self.number_of_nanoparticles, 3)))
        list_of_receptor_arrays = list(np.random.normal(size=(self.number_of_receptors, 3)))
        self.nanoparticle_list = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        self.receptor_list = [agent.position for agent in self.agents if isinstance(agent, Receptor)]
        max_dist_to_react = self.nanoparticle_radius + self.ligand_length + self.receptor_length  # Loop only entered if Nanoparticle is close enough, i.e. receptor have a base position where z = 0
        max_dist_to_react2 = self.nanoparticle_radius + self.ligand_length
        for agent in self.agents:  # loop through agents
            if isinstance(agent, Nanoparticle):
                nanoparticles_list = [nanoparticle_position for nanoparticle_position in self.nanoparticle_list if nanoparticle_position is not agent.position]  # List of the other nanoparticles' positions
                agent.step(self.nanoparticle_brownian(list_of_nanoparticle_arrays.pop()), nanoparticles_list, self.receptor_list)  # different movements for nanoparticle and its ligands
                if agent.position[2] < max_dist_to_react:  # Reaction only possible only if the ligand is close enough to the in the z-axis to the receptor
                    # print(f'{agent.agent_id} is interacting -------------------------------------------------------------------------------------------')
                    receptors = [x for x in self.agents if isinstance(x, Receptor) and x.bound is None]  # List of receptors that are not bound to a ligand
                    for r in receptors:
                        if max_dist_to_react2 <= self.distance(agent.position, r.position) < max_dist_to_react2 + 1:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(agent, r)  # if any agent meets interaction criteria with "agent"
            elif isinstance(agent, Receptor):
                receptors_list = [receptor_position for receptor_position in self.receptor_list if receptor_position is not agent.position]  # List of the other receptors' base positions
                agent.step(self.receptor_brownian(list_of_receptor_arrays.pop()), receptors_list, self.nanoparticle_list)  # move agent
                if agent.bound is None:  # Checks that the receptor isn't already bound to a ligand
                    nanoparticles = [n for n in self.agents if isinstance(n, Nanoparticle)]  # and (n.position[2] < max_dist_to_react))]  # List of Nanoparticles
                    for n in nanoparticles:  # loop through agents again
                        if self.distance(agent.position, n.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(n, agent)  # if any agent meets interaction criteria with "agent"
            else:
                print(False)

    def calculate_surface_coverage(self, n):
        self.surface_coverage = n * ((4 * (self.nanoparticle_radius + self.ligand_length) ** 2) / self.dimension ** 2)
        return self.surface_coverage

    def nanoparticle_brownian(self, array):
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (
                6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9)))) ** 0.5) * 1e9 * self.time_unit * array
        '''using viscosity of water and radius of particle = 50-100 nm; using brownian equations (1) and (3) pg8 + (79) pg 28'''
        return random_movement_cartesian

    def receptor_brownian(self, array):
        """look for the end of the receptors or 100 times smaller"""
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (
                6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9 / 100)))) ** 0.5) * 1e9 * self.time_unit * array
        '''using viscosity of water and radius of particle = 50-100 nm; using brownian equations (1) and (3) pg8 + (79) pg 28 from joe's workk'''
        r = (random_movement_cartesian[0] ** 2 + random_movement_cartesian[1] ** 2 + random_movement_cartesian[
            2] ** 2) ** 0.5
        θ = atan(random_movement_cartesian[1] / random_movement_cartesian[0])
        Φ = acos(random_movement_cartesian[2] / r)
        '''Equations from: https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates'''
        '''simply explained in https://mathworld.wolfram.com/SphericalCoordinates.html'''
        return np.array([r, θ, Φ])

    def interaction_criteria(self, nanoparticle, receptor):
        if nanoparticle.agent_id != receptor.agent_id:  # true if same id
            ligands = nanoparticle.ligands
            for ligand in ligands:
                if ligand.bound is None and receptor.bound is None:  # Can only bind one receptor to one ligand at a time
                    distance = self.distance(ligand.position, receptor.position)
                    if distance <= self.binding_distance:  # true if close position, false otherwise
                        if np.random.uniform(low=0, high=1) > exp(-self.binding_energy):
                            ligand.bound = receptor
                            receptor.bound = ligand
                            nanoparticle.bound = True
                            ligand.position = receptor.position
                            self.count += 1
                            break  # As receptor can only bind to one ligand
                        else:
                            continue

    @staticmethod
    @njit(fastmath=True)
    def distance(a, b):
        return linalg.norm(a - b)