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



class MyModel:
    def __init__(self, dimension, binding_energy):
        self.agents = []  # create an empty list of agents
        self.collision = False
        self.count = 0
        self.dimension = dimension
        self.points = []
        self.binding_energy = binding_energy

    def create_nanoparticles_and_ligands(self, number_of_nanoparticles, number_of_ligands, nanoparticle_radius,
                                         ligand_length):  # Adds nanoparticles into the system
        self.number_of_nanoparticles = number_of_nanoparticles
        self.nanoparticle_radius = nanoparticle_radius
        self.nanoparticle_surface_area = 4 * pi * nanoparticle_radius ** 2
        self.ligand_length = ligand_length
        self.number_of_ligands = number_of_ligands
        true_radius = self.nanoparticle_radius + self.ligand_length  # The maximum radius of the nanoparticle including the ligands
        for i in range(number_of_nanoparticles):  to number of agents
            agent_id = f'Nanoparticle {i}' # loop from 0
            upper_limit = self.dimension - true_radius
            count = 0
            while True:
                nanoparticle_position_xyz = np.array([np.random.uniform(true_radius, upper_limit), np.random.uniform(true_radius, upper_limit), np.random.uniform(true_radius, upper_limit)])  # 3D cube cartesian system - rectangular coordinates
                if self.is_space_available_nanoparticle(nanoparticle_position_xyz):
                    break
                else:
                    continue
            nanoparticle = Nanoparticle(agent_id, nanoparticle_position_xyz, number_of_ligands, nanoparticle_radius,
                                        ligand_length, self.dimension, binding_energy=self.binding_energy)
            self.agents.append(nanoparticle)  # add agent to list of agents

    def is_space_available_nanoparticle(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        separation = (self.nanoparticle_radius + self.ligand_length) * 2
        for i in positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                count += 1
            elif distance < separation:
                return False
        if count == len(positions):
            return True  # Returns True when there is space available

    def create_receptors(self, number_of_receptors, receptor_length):  # Adds receptors into the system
        self.receptor_length = receptor_length
        self.number_of_receptors = number_of_receptors
        for i in range(number_of_receptors):  # loop from 0 to number of agents
            receptor_id = f'Receptor {i}'
            '''Random rather than ordered receptors'''
            while True:
                base_position = np.array([np.random.uniform(0, self.dimension), np.random.uniform(0, self.dimension), 0]) # 3D cube cartesian system - rectangular coordinates
                if self.is_space_available_receptor(base_position):
                    break
                else:
                    continue
            receptor = Receptor(receptor_id, base_position, receptor_length, self.dimension, binding_energy=self.binding_energy)  # create receptor
            self.agents.append(receptor)  # add agent to list of agents

    def is_space_available_receptor(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        positions = [agent.base_position for agent in self.agents if isinstance(agent, Receptor)]
        separation = 0.1 * self.receptor_length
        for i in positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                count += 1
        if count == len(positions):
            return True  # Returns True when there is space available

    def run(self, steps):
        self.steric_potential = self.steric_potential()
        self.coverage = [0]
        self.time = 0
        plateau_count = 0
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
            if i == steps-1:  # or i == steps/2:
                self.visualiser()
        
    def step(self):
        list_of_nanoparticle_arrays = list(np.random.normal(size=(self.number_of_nanoparticles, 3)))
        list_of_receptor_arrays = list(np.random.normal(size=(self.number_of_receptors, 3)))
        self.nanoparticle_list = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        self.receptor_list = [agent.base_position for agent in self.agents if isinstance(agent, Receptor)]
        for agent in self.agents:  # loop through agents
            if isinstance(agent, Nanoparticle):
                nanoparticles_list = [nanoparticle_position for nanoparticle_position in self.nanoparticle_list if nanoparticle_position is not agent.position]  # List of the other nanoparticles' positions
                agent.step(self.nanoparticle_brownian(list_of_nanoparticle_arrays.pop()), nanoparticles_list)  # different movements for nanoparticle and its ligands
                max_dist_to_react = self.nanoparticle_radius + self.ligand_length + self.receptor_length  # Loop only entered if Nanoparticle is close enough, i.e. receptor have a base position where z = 0
                if agent.position[2] < max_dist_to_react:  # Reaction only possible only if the ligand is close enough to the in the z-axis to the receptor
                    receptors = [x for x in self.agents if isinstance(x, Receptor) and x.bound is None]  # List of receptors that are not bound to a ligand
                    for r in receptors:
                        max_dist_to_react2 = self.nanoparticle_radius + self.ligand_length
                        if self.distance(agent.position, r.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(agent, r)  # if any agent meets interaction criteria with "agent"
            elif isinstance(agent, Receptor):
                receptors_list = [receptor_position for receptor_position in self.receptor_list if receptor_position is not agent.base_position]  # List of the other receptors' base positions
                agent.step(self.receptor_brownian(list_of_receptor_arrays.pop()), receptors_list)  # move agent
                if agent.bound is None:  # Checks that the receptor isn't already bound to a ligand
                    nanoparticles = [x for x in self.agents if isinstance(x, Nanoparticle)]  # List of Nanoparticles
                    for n in nanoparticles:  # loop through agents again
                        max_dist_to_react2 = self.nanoparticle_radius + self.ligand_length
                        if self.distance(agent.position, n.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(n, agent)  # if any agent meets interaction criteria with "agent"

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
        ax.scatter(x1, y1, z1, c='Red', s=350*(self.nanoparticle_radius/50))  # Unbound
        ax.scatter(x2, y2, z2, c='Green', s=350*(self.nanoparticle_radius/50))  # Bound
        ax.scatter(x4, y4, z4, c='Red', s=self.receptor_length, marker='1')  # Unbound
        ax.scatter(x5, y5, z5, c='Green', s=self.receptor_length, marker='1')  # Bound
        ax.set_title('3D')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, self.dimension)
        ax.set_ylim(0, self.dimension)
        ax.set_zlim(0, self.dimension)
        plt.show()

    def nanoparticle_brownian(self, array):
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9)))) ** 0.5) * 1e9 * array
        '''using viscosity of water and radius of particle = 50-100 nm; using brownian equations (1) and (3) pg8 + (79) pg 28'''
        return random_movement_cartesian

    def receptor_brownian(self, array):
        """look for the end of the receptors or 100 times smaller"""
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9 / 100)))) ** 0.5) * 1e9 * array
        '''using viscosity of water and radius of particle = 50-100 nm; using brownian equations (1) and (3) pg8 + (79) pg 28'''
        r = (random_movement_cartesian[0] ** 2 + random_movement_cartesian[1] ** 2 + random_movement_cartesian[2] ** 2) ** 0.5
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
                    if distance <= 1:
                        if np.random.uniform(low=0, high=1) > exp(-self.binding_energy):
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
        β = (1.38064852e-23 * 310.15)**-1  # Boltzmann constant x Temperature (of human body 37°C = 310.15K)
        steric_potential = ((receptor_tip_volume * ((1 - (((self.ligand_length * 1e-9) / (self.receptor_length * 1e-9)) ** 2)) ** (9 / 4)))
                             / (area_per_ligand ** (3 / 2))) / β
        return steric_potential


c = [200]
b = []



def experiment():
    for i in c:
        number_of_seconds = 3600  # i.e. 1 hour = 3600 seconds
        my_model = MyModel(dimension=1000, binding_energy=200)
        my_model.create_nanoparticles_and_ligands(number_of_nanoparticles=190, number_of_ligands=int(round(10 ** 2, 0)), nanoparticle_radius=50, ligand_length=7)  # 1-2 nm for ligand  # 95 particles
        my_model.create_receptors(number_of_receptors=1000, receptor_length=100)  # 100 nm for receptor
        print(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
              f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands,\n'
              f'Ligand length {my_model.ligand_length} nm, {my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length')
        my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
        print(f'There were {my_model.count} reactions')
        '''Plot Time (Seconds) v surface coverage'''
        plt.subplot()
        plt.title(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
                  f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.number_of_ligands} Ligands, {my_model.ligand_length} nm Ligand length \n'
                  f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Surface Coverage')
        plt.plot(list(range(0, my_model.time+1)), my_model.coverage, label=f'Nanoparticle radius {my_model.nanoparticle_radius} nm')

    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.5)
    plt.show()


experiment()


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