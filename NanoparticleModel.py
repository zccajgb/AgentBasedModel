import numpy as np
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp
from LigandModel import Ligand
from numba import njit


class Nanoparticle:  # inherits from agent
    def __init__(self, agent_id, nanoparticle_position_xyz, number_of_ligands, nanoparticle_radius, ligand_length,
                 dimension, binding_energy):
        self.number_of_ligands = number_of_ligands
        self.agent_id = agent_id
        self.position = nanoparticle_position_xyz
        self.ligands = []

        a = np.full(number_of_ligands, 0.5*nanoparticle_radius)  # r
        b = np.random.uniform(low=0, high=(2 * pi), size=number_of_ligands)  # θ
        c = np.random.uniform(low=0, high=pi, size=number_of_ligands)  # Φ
        bases = []
        for r, θ, Φ in np.nditer([a, b, c]):
            bases.append(np.array([r, θ, Φ]))

        d = np.random.uniform(low=0, high=ligand_length, size=number_of_ligands)  # r
        e = np.random.uniform(low=0, high=(2 * pi), size=number_of_ligands)  # θ
        f = np.random.uniform(low=0, high=(0.5 * pi), size=number_of_ligands)  # Φ
        tips = []
        for r, θ, Φ in np.nditer([d, e, f]):  # iterate through  r, θ, Φ
            tips.append(np.array([r, θ, Φ]))

        for i in range(number_of_ligands):
            base_array = bases.pop()
            tip_array = tips.pop()
            self.ligands.append(Ligand(i, agent_id, nanoparticle_position_xyz, nanoparticle_radius, ligand_length, base_array, tip_array, binding_energy))

        self.nanoparticle_radius = nanoparticle_radius
        self.ligand_length = ligand_length
        self.bound = False
        self.dimension = dimension
        self.binding_energy = binding_energy

    def step(self, value, agents_list):  # values keeps the everything moving according to brownian motion
        # """Convert spherical brownian coordinates into rectangular"""
        # x = value[0] * sin(value[2]) * cos(value[1])
        # y = value[0] * sin(value[2]) * sin(value[1])
        # z = value[0] * cos(value[2])
        # value_rectangular = np.array([x, y, z])
        # value_rectangular = self.convert_spherical_to_rectangular(value)
        attempt = self.position + value  # value_rectangular
        true_attempt = self.get_absolute_position(attempt)
        freedom = self.is_space_available(agents_list, true_attempt)
        list_of_ligand_arrays = list(np.random.normal(size=(self.number_of_ligands, 3)))
        n = 0
        for ligand in self.ligands:
            if ligand.bound is not None:  # i.e. a ligand is bound
                n += 1
        if n > 0:  # i.e. there are some ligands that have bonded
            if np.random.uniform(low=0, high=1) > (1 - exp(-self.binding_energy)) ** n:  # Bonds get broken
                if freedom:
                    self.bound = False
                    # print(f'{self.agent_id} broke it bonds')
                    self.position = true_attempt
                    # print(f"{self.agent_id} moved to position {self.position}")
                    for i in self.ligands:
                        if i.bound is not None:
                            # print(f'The bond between {i.agent_id} and {i.bound.agent_id} was broken')
                            i.bound.bound = None
                            i.bound = None
                        i.step(list_of_ligand_arrays.pop(), self.position)
                elif not freedom:
                    # print(f'{self.agent_id} stayed at {self.position}')
                    for i in self.ligands:
                        i.step(list_of_ligand_arrays.pop(), self.position)
            else:
                # print(f'{self.agent_id} remained bound to the surface at {self.position}')
                for i in self.ligands:
                    i.step(list_of_ligand_arrays.pop(), self.position)
        elif n == 0:  # i.e. no ligands bound
            self.bound = False  # If all the bonds were broken by receptors moving then this updates
            if freedom:
                self.position = true_attempt
                # print(f"{self.agent_id} moved to position {self.position}")
                for i in self.ligands:
                    i.step(list_of_ligand_arrays.pop(), self.position)
            elif not freedom:
                for i in self.ligands:
                    i.step(list_of_ligand_arrays.pop(), self.position)
                # print(f'{self.agent_id} stayed at {self.position}')
        # print(f'{self.agent_id} moved to {self.position}')
        return self.position

    def is_space_available(self, coordinates_list, attempt):
        separation = 2 * (self.nanoparticle_radius + self.ligand_length)  # nanoparticles can be touching
        max_closeness = separation + 0.5 * self.nanoparticle_radius  # van der Waals' radius = 0.5 x separation
        count = 0
        for i in coordinates_list:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                if distance <= max_closeness:  # Checks if close enough for repulsive potential
                    if np.random.uniform(low=0, high=1) < exp(-self.repulsive_potential(distance, max_closeness)):  # If there is repulsion then check  # tolerable_potential:
                        count += 1
                    else:
                        return False  # Repulsive potential not exceeded
                else:
                    count += 1
            elif distance < separation:
                return False
        if count == len(coordinates_list):
            return True  # Returns True when there is space available

    @staticmethod
    @njit(fastmath=True)
    def distance(a, b):
        return linalg.norm(a - b)

    @staticmethod
    def repulsive_potential(distance, max_closeness):
        potential = 4 * 1 * ((max_closeness / distance) ** 12 - (max_closeness / distance) ** 6)
        return potential

    def get_absolute_position(self, absolute_position):
        """Keeping nanoparticle in the system"""
        max_abs_position = self.dimension - (self.nanoparticle_radius + self.ligand_length)
        for i in range(3):
            # while not (self.nanoparticle_radius + self.ligand_length) <= absolute_position[i] <= max_abs_position:
            while not 0 <= absolute_position[i] <= max_abs_position:
                if absolute_position[i] < 0:
                    absolute_position[i] = abs(absolute_position[i])
                # if absolute_position[i] < (self.nanoparticle_radius + self.ligand_length):
                #     absolute_position[i] = absolute_position[i] + self.nanoparticle_radius + self.ligand_length
                if absolute_position[i] > max_abs_position:
                    recoil = absolute_position[i] - max_abs_position
                    absolute_position[i] = max_abs_position - recoil
                    # absolute_position[i] = absolute_position[i] % 2 * max_abs_position
                    # if absolute_position[i] > max_abs_position:
                    #     recoil = absolute_position[i] - max_abs_position
                    #     absolute_position[i] = max_abs_position - recoil

        return absolute_position
