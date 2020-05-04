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


# 20 micro meter for endothelium, width of a endthlial cell
# start with 1000 nano metres

# check if nanoparticles are too close
# if binding when they shouldn't be, i.e. are bound particles still bound when not near the surface


class MyModel:
    def __init__(self, dimension, binding_energy, time_unit):
        self.agents = []  # create an empty list of agents
        self.collision = False
        self.count = 0
        self.dimension = dimension
        self.points = []
        self.binding_energy = binding_energy
        self.time_unit = time_unit

    def create_nanoparticles_and_ligands(self, number_of_nanoparticles, number_of_ligands, nanoparticle_radius,
                                         ligand_length):  # Adds nanoparticles into the system
        self.number_of_nanoparticles = number_of_nanoparticles
        self.nanoparticle_radius = nanoparticle_radius
        self.nanoparticle_surface_area = 4 * pi * nanoparticle_radius ** 2
        self.ligand_length = ligand_length
        self.number_of_ligands = number_of_ligands
        true_radius = self.nanoparticle_radius + self.ligand_length  # The maximum radius of the nanoparticle including the ligands
        for i in range(number_of_nanoparticles):  # loop from 0 to number of agents
            agent_id = f'Nanoparticle {i}'
            upper_limit = self.dimension - true_radius
            count = 0
            while True:
                nanoparticle_position_xyz = np.array(
                    [np.random.uniform(true_radius, upper_limit), np.random.uniform(true_radius, upper_limit),
                     np.random.uniform(true_radius, upper_limit)])  # 3D cube cartesian system - rectangular coordinates
                if self.is_space_available_nanoparticle(nanoparticle_position_xyz):
                    break
                else:
                    continue
            nanoparticle = Nanoparticle(agent_id, nanoparticle_position_xyz, number_of_ligands, nanoparticle_radius,
                                        ligand_length, self.dimension, binding_energy=self.binding_energy,
                                        time_unit=self.time_unit)
            # print(f'Created {agent_id}')
            self.agents.append(nanoparticle)  # add agent to list of agents

    def is_space_available_nanoparticle(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        separation = (self.nanoparticle_radius + self.ligand_length) * 2
        for i in positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                count += 1
            else:
                return False
        if count == len(positions):
            return True  # Returns True when there is space available

    '''Repulsion when setting up system????'''  # ######################################################################

    # def is_space_available2(self, coordinates_list, attempt):
    #     separation = 2 * (self.nanoparticle_radius + self.ligand_length)  # nanoparticles can be touching
    #     max_closeness = separation + 0.5 * self.nanoparticle_radius  # van der Waals' radius = 0.5 x separation
    #     count = 0
    #     for i in coordinates_list:
    #         distance = self.distance(attempt, i)
    #         if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
    #             if distance <= max_closeness:  # Checks if close enough for repulsive potential
    #                 if np.random.uniform(low=0, high=1) < exp(-self.repulsive_potential(distance, max_closeness)):  # If there is repulsion then check  # tolerable_potential:
    #                     count += 1
    #             else:
    #                 count += 1
    #         elif distance < separation:
    #             print('Error', distance)
    #             break
    #     if count == len(coordinates_list):
    #         return True  # Returns True when there is space available

    def create_receptors(self, number_of_receptors, receptor_length):  # Adds receptors into the system
        self.receptor_length = receptor_length
        self.number_of_receptors = number_of_receptors
        # for i in range(number_of_receptors):  # loop from 0 to number of agents
        #     receptor_id = f'Receptor {i}'
        #     base_position = np.array([np.random.uniform(0, self.dimension), np.random.uniform(0, self.dimension), 0])  # 3D cube cartesian system - rectangular coordinates
        #     receptor = Receptor(receptor_id, base_position, receptor_length, self.dimension, binding_energy=self.binding_energy)  # create receptor
        # rows = np.linspace(0, int(self.dimension), int(self.number_of_receptors*0.1), endpoint=True).tolist()
        # x = []
        # for i in rows:
        #     x.append(np.full(int(self.number_of_receptors/int(len(rows))), int(i)))
        # y = []
        # for i in range(len(rows)):
        #     y.append(np.linspace(0, int(self.dimension), int(self.number_of_receptors*0.01), endpoint=True))
        # x1 = np.concatenate(x, axis=None)
        # y1 = np.concatenate(y, axis=None)
        # z1 = np.full(number_of_receptors, 0)
        # bases = []
        # for x2, y2, z2 in np.nditer([x1, y1, z1]):
        #     bases.append(np.array([x2, y2, z2]))
        # for i in range(number_of_receptors):  # loop from 0 to number of agents
        #     receptor_id = f'Receptor {i}'
        #     receptor = Receptor(receptor_id, bases.pop(), receptor_length, self.dimension, self.binding_energy,
        #                         self.nanoparticle_radius, self.ligand_length)  # create receptor
        #     self.agents.append(receptor)  # add agent to list of agents
        for i in range(number_of_receptors):  # loop from 0 to number of agents
            receptor_id = f'Receptor {i}'
            '''Random rather than ordered receptors'''
            while True:
                base_position = np.array([np.random.uniform(0, self.dimension), np.random.uniform(0, self.dimension),
                                          0])  # 3D cube cartesian system - rectangular coordinates
                if self.is_space_available_receptor(base_position):
                    break
                else:
                    continue
            receptor = Receptor(receptor_id, base_position, receptor_length, self.dimension, self.binding_energy,
                                self.nanoparticle_radius, self.ligand_length)  # create receptor
            # print(f'Created {receptor_id}')
            self.agents.append(receptor)  # add agent to list of agents

    def is_space_available_receptor(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        positions = [agent.base_position for agent in self.agents if isinstance(agent, Receptor)]
        separation = 7  # 0.01 * self.receptor_length
        for i in positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                count += 1
            else:
                return False
        if count == len(positions):
            return True  # Returns True when there is space available

    def run(self, steps):
        self.steric_potential = self.steric_potential()
        bound_number = []
        self.coverage = [0]
        self.time = 0
        plateau_count = 0
        self.bound_nanoparticles = 0
        for i in range(steps):  # loop through number of steps
            self.time += 1
            print(f'Running step {i} ---------------------------------------------------------------------------------')
            self.step()  # run one step
            bound_nanoparticles = 0
            for agent in self.agents:
                if isinstance(agent, Nanoparticle):
                    if agent.bound is True:
                        bound_nanoparticles += 1  # Number of nanoparticles bound to the cell surface
            self.coverage.append(self.calculate_surface_coverage(bound_nanoparticles))
            if i == steps - 1:  # or i == steps/2:
                self.visualiser()
            # self.visualiser()
            if self.time > 30000:
                if len(self.coverage) > 2:
                    if (self.coverage[-1] - self.coverage[-2]) < 0.005:
                        plateau_count += 1
                if plateau_count > 50:
                    self.visualiser()
                    break  # If data starts to plateau for 50 steps, then the experiment ends
        #     bound_number.append(bound_nanoparticles)  # List of number of bound nanoparticles per step
        # mean_bound = sum(bound_number) / len(bound_number)  # Mean number of bound nanoparticles throughout the experiment
        # self.calculate_surface_coverage(mean_bound)
        for agent in self.agents:
            if isinstance(agent, Nanoparticle):
                if agent.bound is True:
                    self.bound_nanoparticles += 1  # Number of nanoparticles bound to the cell surface
        self.calculate_surface_coverage(self.bound_nanoparticles)

    def step(self):
        list_of_nanoparticle_arrays = list(np.random.normal(size=(self.number_of_nanoparticles, 3)))
        list_of_receptor_arrays = list(np.random.normal(size=(self.number_of_receptors, 3)))
        self.nanoparticle_list = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        self.receptor_list = [agent.position for agent in self.agents if isinstance(agent, Receptor)]
        max_dist_to_react = self.nanoparticle_radius + self.ligand_length + self.receptor_length  # Loop only entered if Nanoparticle is close enough, i.e. receptor have a base position where z = 0
        max_dist_to_react2 = self.nanoparticle_radius + self.ligand_length
        for agent in self.agents:  # loop through agents
            if isinstance(agent, Nanoparticle):
                nanoparticles_list = [nanoparticle_position for nanoparticle_position in self.nanoparticle_list if
                                      nanoparticle_position is not agent.position]  # List of the other nanoparticles' positions
                agent.step(self.nanoparticle_brownian(list_of_nanoparticle_arrays.pop()), nanoparticles_list,
                           self.receptor_list)  # different movements for nanoparticle and its ligands
                if agent.position[
                    2] < max_dist_to_react:  # Reaction only possible only if the ligand is close enough to the in the z-axis to the receptor
                    # print(f'{agent.agent_id} is interacting -------------------------------------------------------------------------------------------')
                    receptors = [x for x in self.agents if isinstance(x,
                                                                      Receptor) and x.bound is None]  # List of receptors that are not bound to a ligand
                    for r in receptors:
                        # print(f'{agent.agent_id} is interacting -------------------------------------------------------------------------------------------')
                        # if linalg.norm(agent.position - r.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                        if max_dist_to_react2 <= self.distance(agent.position,
                                                               r.position) < max_dist_to_react2 + 1:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(agent, r)  # if any agent meets interaction criteria with "agent"
            elif isinstance(agent, Receptor):
                receptors_list = [receptor_position for receptor_position in self.receptor_list if
                                  receptor_position is not agent.position]  # List of the other receptors' base positions
                agent.step(self.receptor_brownian(list_of_receptor_arrays.pop()), receptors_list,
                           self.nanoparticle_list)  # move agent
                if agent.bound is None:  # Checks that the receptor isn't already bound to a ligand
                    nanoparticles = [n for n in self.agents if isinstance(n,
                                                                          Nanoparticle)]  # and (n.position[2] < max_dist_to_react))]  # List of Nanoparticles
                    for n in nanoparticles:  # loop through agents again
                        # print(f'{agent.agent_id} is interacting -------------------------------------------------------------------------------------------')
                        # if linalg.norm(agent.position - n.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                        if self.distance(agent.position,
                                         n.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(n, agent)  # if any agent meets interaction criteria with "agent"
            else:
                print(False)

    def calculate_surface_coverage(self, n):
        self.surface_coverage = n * ((4 * (self.nanoparticle_radius + self.ligand_length) ** 2) / self.dimension ** 2)
        return self.surface_coverage

    def visualiser(self):
        """Nanoparticles"""
        self.agent_nanoparticles_dictionary = {'Points': 'Bound'}
        for agent in self.agents:
            if isinstance(agent, Nanoparticle):
                self.agent_nanoparticles_dictionary[tuple(agent.position.tolist())] = agent.bound
        nanoparticles = [agent for agent in self.agents if isinstance(agent, Nanoparticle)]
        x = [i.position[0] for i in nanoparticles]
        y = [i.position[1] for i in nanoparticles]
        z = [i.position[2] for i in nanoparticles]
        bound_nanoparticles = [str(i.bound) for i in nanoparticles]
        dictionary = {'x': x, 'y': y, 'z': z, 'Bound': bound_nanoparticles}
        self.nanoparticles_df = pd.DataFrame(dictionary)
        x1 = []
        y1 = []
        z1 = []
        x2 = []
        y2 = []
        z2 = []
        for position, bound in self.agent_nanoparticles_dictionary.items():
            if bound is False:
                x1.append(position[0])
                y1.append(position[1])
                z1.append(position[2])
            if bound is True:
                x2.append(position[0])
                y2.append(position[1])
                z2.append(position[2])
        '''Receptors'''
        self.agent_receptors_dictionary = {'Points': 'Bound'}
        for agent in self.agents:
            if isinstance(agent, Receptor):
                self.agent_receptors_dictionary[tuple(agent.position.tolist())] = agent.bound
        receptors = [agent for agent in self.agents if isinstance(agent, Receptor)]
        x3 = [i.position[0] for i in receptors]
        y3 = [i.position[1] for i in receptors]
        z3 = [i.position[2] for i in receptors]
        bound_receptors = [str(i.bound) for i in receptors]
        dictionary = {'x': x3, 'y': y3, 'z': z3, 'Bound': bound_receptors}
        self.receptors_df = pd.DataFrame(dictionary)
        x4 = []
        y4 = []
        z4 = []
        x5 = []
        y5 = []
        z5 = []
        for position, bound in self.agent_receptors_dictionary.items():
            if bound is None:
                x4.append(position[0])
                y4.append(position[1])
                z4.append(position[2])
            elif bound is not (None or 'Bound'):
                x5.append(position[0])
                y5.append(position[1])
                z5.append(position[2])
        '''Ligands'''
        x6 = []
        y6 = []
        z6 = []
        for agent in self.agents:  # loop through agents
            if isinstance(agent, Nanoparticle):
                for i in agent.ligands:
                    x6.append(i.position[0])
                    y6.append(i.position[1])
                    z6.append(i.position[2])
        '''3D'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, y1, z1, c='Red', s=350 * (self.nanoparticle_radius / 50))  # Unbound
        ax.scatter(x2, y2, z2, c='Green', s=350 * (self.nanoparticle_radius / 50))  # Bound
        ax.scatter(x4, y4, z4, c='Red', s=self.receptor_length, marker='1')  # Unbound
        ax.scatter(x5, y5, z5, c='Green', s=self.receptor_length, marker='1')  # Bound
        # ax.scatter(x6, y6, z6, c='Blue', s=self.ligand_length*100, marker='1')  # Ligands
        ax.set_title('3D')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, self.dimension)
        ax.set_ylim(0, self.dimension)
        ax.set_zlim(0, self.dimension)
        plt.show()
        '''2D'''
        # plt.scatter(x1, y1, c='Red', s=2 * (self.nanoparticle_radius + self.ligand_length))
        # plt.scatter(x2, y2, c='Green', s=2 * (self.nanoparticle_radius + self.ligand_length))
        # plt.title('2D')
        # plt.xlabel('X Axis')
        # plt.ylabel('Y Axis')
        # plt.xlim(0, self.dimension)
        # plt.ylim(0, self.dimension)
        # plt.show()

    def nanoparticle_brownian(self, array):
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (
                6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9)))) ** 0.5) * 1e9 * self.time_unit * array
        '''using viscosity of water and radius of particle = 50-100 nm; using brownian equations (1) and (3) pg8 + (79) pg 28'''
        return random_movement_cartesian

    def receptor_brownian(self, array):
        """look for the end of the receptors or 100 times smaller"""
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (
                6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9 / 100)))) ** 0.5) * 1e9 * self.time_unit * array
        '''using viscosity of water and radius of particle = 50-100 nm; using brownian equations (1) and (3) pg8 + (79) pg 28'''
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
                    if distance <= 1:  # true if close position, false otherwise
                        # if np.allclose(ligand.position, receptor.position, 1):  # true if same position, false otherwise
                        # print(f'Collision with {ligand.agent_id} of {agent1.agent_id} and {agent2.agent_id}')
                        # if np.random.uniform(low=0, high=1) < (self.repulsive_potential(distance, 5 * self.ligand_length) + self.steric_potential):
                        if np.random.uniform(low=0, high=1) > exp(-self.binding_energy):
                            print(f'Reaction happened between {ligand.agent_id} of {nanoparticle.agent_id} and {receptor.agent_id}')
                            ligand.bound = receptor
                            receptor.bound = ligand
                            nanoparticle.bound = True
                            self.count += 1
                            break  # As receptor can only bind to one ligand
                        else:
                            continue

    @staticmethod
    @njit(fastmath=True)
    def distance(a, b):
        return linalg.norm(a - b)

    @staticmethod
    def repulsive_potential(distance, max_closeness):
        repulsive_potential = 4 * 1 * ((max_closeness / distance) ** 12 - (max_closeness / distance) ** 6)
        return repulsive_potential

    def steric_potential(self):
        receptor_tip_volume = (4 / 3) * pi * ((self.nanoparticle_radius * 1e-9 / 100) ** 3)
        area_per_ligand = (4 * pi * ((self.nanoparticle_radius * 1e-9) ** 2)) / self.number_of_ligands
        β = (1.38064852e-23 * 310.15) ** -1  # Boltzmann constant x Temperature (of human body 37°C = 310.15K)
        steric_potential = ((receptor_tip_volume * (
                (1 - (((self.ligand_length * 1e-9) / (self.receptor_length * 1e-9)) ** 2)) ** (9 / 4)))
                            / (area_per_ligand ** (3 / 2))) / β
        return steric_potential


# don't go over 10% volume fraction = 190 nanoparticles at 50 nm radius
# 2500 (≈ 10^3.4) ligands max based on 50nm radius nanoparticle and 2 nm ligand
# fix number of ligands, test for different numbers of ligands, plot data over time
# See what situation is at 500 steps where all nanoparticles are supposedly bound

# run with binding energy with 100 or 200 - so no bonds break - up to 1000
# run with binding energy of 0 - check no bonds form
# save the output data  ##############################
# breakpoint on method that calculates if things are two close
# method which compares particle positions to see if too close


# see if graph eqilibriates
# lower binding energy then plot
# Plot against number of receptors

# slow down
# rigid receptors and ligands?


# c = np.arange(50, 250, 50).tolist()
c = [1000]  # , 2000, 3000, 4000]
b = []


def experiment_variable():
    coverage = [0]
    for i in c:
        number_of_seconds = 250  # i.e. 1 hour = 3600 seconds
        my_model = MyModel(dimension=1000, binding_energy=2)
        my_model.create_nanoparticles_and_ligands(number_of_nanoparticles=190, number_of_ligands=int(round(10 ** 2, 0)),
                                                  nanoparticle_radius=50, ligand_length=7)
        my_model.create_receptors(number_of_receptors=i, receptor_length=100)  # 100 nm for receptor
        print(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
              f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands,\n'
              f'Ligand length {my_model.ligand_length} nm, {my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length')
        my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
        print(f'There were {my_model.count} reactions')
        print(f'The surface coverage is {my_model.surface_coverage}')
        coverage.append(my_model.surface_coverage)
    '''Surface coverage v Variable value'''
    plt.title(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
              f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.number_of_ligands} Ligands, {my_model.ligand_length} nm Ligand length \n'  # {my_model.number_of_receptors} Receptors,
              f'{my_model.receptor_length} nm Receptor length')
    plt.xlabel('Number of Receptors')
    plt.ylabel('End Surface Coverage')
    c.insert(0, 0)
    plt.plot(c, coverage)
    plt.show()


# experiment_variable()


def experiment_time():  # subplots
    for i in c:
        number_of_seconds = 60000  # i.e. 1 hour = 3600 seconds
        my_model = MyModel(dimension=1000, binding_energy=3, time_unit=10e-3)
        my_model.create_nanoparticles_and_ligands(number_of_nanoparticles=92, number_of_ligands=int(round(10 ** 2, 0)), nanoparticle_radius=50, ligand_length=7)  # 1-2 nm for ligand  # 95 particles
        my_model.create_receptors(number_of_receptors=1000, receptor_length=100)  # 100 nm for receptor
        print(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
              f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands,\n'
              f'Ligand length {my_model.ligand_length} nm, {my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length')
        my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
        print(f'There were {my_model.count} reactions')
        # print(f'The surface coverage is {my_model.surface_coverage}')
        # points = my_model.agent_position_dictionary
        '''Plot Time (Seconds) v surface coverage'''
        plt.subplot()
        # plt.title(f'{my_model.number_of_nanoparticles} Nanoparticles, 10^{i} Ligands, Ligand length {my_model.ligand_length} nm, {my_model.number_of_receptors} Receptors')
        plt.title(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
                  f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.number_of_ligands} Ligands, {my_model.ligand_length} nm Ligand length \n'
                  f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length')
        plt.xlabel('Time (milliseconds)')
        plt.ylabel('Surface Coverage')
        plt.plot(list(range(0, my_model.time + 1)), my_model.coverage,
                 label=f'Nanoparticle radius {my_model.nanoparticle_radius} nm')
        '''Plot log(Time) (Seconds) v surface coverage'''
        # plt.subplot()
        # plt.title(f'{my_model.number_of_nanoparticles} Nanoparticles, 10^{i} Ligands, {my_model.number_of_receptors} Receptors, Receptor length {my_model.receptor_length} nm')
        # plt.xlabel('log(Time (seconds))')  # For log x-axis
        # plt.ylabel('Surface Coverage')
        # plt.plot([np.log10(i) for i in list(range(1, my_model.time+1))], my_model.coverage, label=f'Ligand length {my_model.ligand_length} nm')  # For log x-axi
        '''Saving Data'''
        # dictionary of lists
        # dictionary = {'Time (Seconds)': list(range(1, my_model.time+1)), 'Surface Coverage': my_model.coverage}
        # dictionary = {'log(Time) (Seconds)': [np.log10(i) for i in list(range(1, my_model.time+1)], 'Surface Coverage': b}  # For log x-axis
        # df = pd.DataFrame(dictionary)
        # saving the DataFrame
        # df.to_csv('data12.CSV', index=False)
        # my_model.nanoparticles_df.to_csv('data12_nanoparticles.CSV', index=False)
        # my_model.receptors_df.to_csv('data12_receptors.CSV', index=False)

    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.5)
    plt.show()


experiment_time()


def stored_visualiser(dimensions, data):
    df = pd.read_csv(data)
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    for i in range(len(df.index)):
        a = str(df.iloc[i]['Bound'])
        if a == 'True':
            x1.append(df.iloc[i]['x'])
            y1.append(df.iloc[i]['y'])
            z1.append(df.iloc[i]['z'])
        if a == 'False':
            x2.append(df.iloc[i]['x'])
            y2.append(df.iloc[i]['y'])
            z2.append(df.iloc[i]['z'])
    '''3D'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, z1, c='Green', s=104)  # Bound
    ax.scatter(x2, y2, z2, c='Red', s=104)  # Unbound
    ax.set_title('3D')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, dimensions)
    ax.set_ylim(0, dimensions)
    ax.set_zlim(0, dimensions)
    plt.show()


# stored_visualiser(dimensions=1000, data='data11points.CSV')

'''Get graph from data'''
# df = pd.read_csv('data12.CSV', header=0)
# plt.title(f'Number of Ligands = 10^{2}')
# plt.xlabel('Time (Seconds)')
# plt.ylabel('Surface Coverage')
# plt.plot(df['Time (Seconds)'], df['Surface Coverage'])
# plt.show()
''''''

# attempts_for_mean = 1
# for i in a:
#     print(f'Number of Ligands = 10^{i}')
#     surface_coverage = 0
#     for j in range(attempts_for_mean):
#         my_model = MyModel(dimension=1000)
#         my_model.create_nanoparticles_and_ligands(number_of_nanoparticles=10, number_of_ligands=int(round(10 ** i, 0)), nanoparticle_radius=50, ligand_length=2)  # 1-2 nm for ligand
#         print('Created Nanoparticles')
#         my_model.create_receptors(number_of_receptors=1000, receptor_length=100)  # 100 nm for receptor
#         print('Created Receptors')
#         print('--Running model--')
#         my_model.run(steps=5)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         # print(f'The surface coverage is {my_model.surface_coverage}')
#         surface_coverage += my_model.surface_coverage
#     mean_surface_coverage = surface_coverage/attempts_for_mean
#     print(f'The mean surface coverage is {mean_surface_coverage}')
#     b.append(mean_surface_coverage)  # Mean surface coverage

#
# # dictionary of lists
# dictionary = {'Log(Number of Ligands)': a, 'Surface Coverage': b}
# df = pd.DataFrame(dictionary)
# # saving the DataFrame
# df.to_csv('data6.CSV', index=False)
# print(df)
#
# # Plot Log(ligands) v surface coverage
# plt.xlabel('Log(Number of Ligands)')
# plt.ylabel('Surface Coverage')
# plt.plot(a, b)
# plt.show()
#
# print('Log Bases: ', a)
# print('Surface Coverages: ', b)

'''Repulsive Potential data plot'''
# def repulsive_potential(distance, max_closeness):
#     potential = 4 * 10000 * (((max_closeness / distance) ** 12) - ((max_closeness / distance) ** 6))
#     return potential
#
# nanoparticle_radius = 52  # Radius + ligand
# separation = nanoparticle_radius * 2  # = 104 # separation between two Nanoparticles = Nanoparticle diameter including ligand = 104
# max_closeness_ = separation + 0.5 * nanoparticle_radius  # = 130  # Arbitrary value representing van der Waals' radius
#
# a = []
# # b = [140, 130, 120, 110, 100]
# b = np.arange(100, 500, 1).tolist()
#
# for i in b:
#     a.append(repulsive_potential(i, max_closeness_))
#
# plt.plot(b, a)
# plt.xlabel('Distance')
# plt.ylabel('Potential')
# plt.show()
