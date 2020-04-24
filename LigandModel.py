import numpy as np
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp
from numba import njit
from ReceptorModel import Receptor


class Ligand:
    def __init__(self, agent_id, nanoparticle_id, nanoparticle_position, nanoparticle_radius, ligand_length, base_array, tip_array, binding_energy):
        self.agent_id = f'Ligand {agent_id}'
        """Nanoparticle ligand base spherical coordinates in the format(r,θ,Φ), distance of nanoparticle of radius 1 from the centre of the nanoparticle, on the surface of the nanoparticle"""
        ligand_base_position = base_array
        self.ligand_tip_position = tip_array
        self.temp_tip = None
        self.bound = None
        self.nanoparticle_position = nanoparticle_position
        self.nanoparticle_id = nanoparticle_id
        self.ligand_length = ligand_length
        self.nanoparticle_radius = nanoparticle_radius
        self.total_radius = nanoparticle_radius + ligand_length
        self.binding_energy = binding_energy

        '''For ligand base'''
        self.ligand_base_position = self.convert_spherical_to_rectangular(ligand_base_position)

        '''For ligand tip'''
        ligand_tip_xyz = self.convert_spherical_to_rectangular(self.ligand_tip_position)

        self.position = self.ligand_base_position + ligand_tip_xyz + nanoparticle_position

    @staticmethod
    @njit(fastmath=True)
    def convert_spherical_to_rectangular(array):
        x = array[0] * sin(array[2]) * cos(array[1])
        y = array[0] * sin(array[2]) * sin(array[1])
        z = array[0] * cos(array[2])  # z value of tip
        return np.array([x, y, z])

    @staticmethod
    @njit(fastmath=True)
    def distance(a, b):
        return linalg.norm(a - b)

    def step(self, value, nanoparticle_position):
        self.update_nanoparticle_position(nanoparticle_position)
        attempt = self.move(value, nanoparticle_position)
        if isinstance(self.bound, Receptor):
            distance = self.distance(self.bound.base_position, attempt)
            inside_radius = (distance <= self.bound.receptor_length)  # receptor length
            '''Returns True if inside and False if outside'''
            if inside_radius:
                self.position = attempt
                self.ligand_tip_position = self.temp_tip
                return self.position
            else:  
                if np.random.uniform(low=0, high=1) < exp(-self.binding_energy):  # Bond gets broken
                    self.position = attempt
                    self.ligand_tip_position = self.temp_tip
                    self.bound.bound = None
                    self.bound = None
                    return self.position
                else:
                    return self.position
        else:
            self.position = attempt
            self.ligand_tip_position = self.temp_tip
            return self.position

    def update_nanoparticle_position(self, position):
        self.nanoparticle_position = position
        return self.nanoparticle_position

    def move(self, value, nanoparticle_position):
        attempt_tip = self.ligand_tip_position + self.ligand_brownian(value)
        attempt = self.get_attempt_position(attempt_tip, nanoparticle_position)
        return attempt

    def get_attempt_position(self, attempt_tip, nanoparticle_position):
        """Keeping ligand_tip within a certain distance of the ligand_base"""
        while not 0 <= attempt_tip[0] <= self.ligand_length:
            if attempt_tip[0] < 0:
                attempt_tip[0] = abs(attempt_tip[0])
            if attempt_tip[0] > self.ligand_length:
                recoil = attempt_tip[0] - self.ligand_length
                attempt_tip[0] = self.ligand_length - recoil
        while not 0 <= attempt_tip[1] <= (2 * pi):
            if attempt_tip[1] < 0:
                attempt_tip[1] = abs(attempt_tip[1])
            if attempt_tip[1] > (2 * pi):
                attempt_tip[1] -= (2 * pi)
        while not 0 <= attempt_tip[2] <= (0.5 * pi):
            if attempt_tip[2] < 0:
                attempt_tip[2] = abs(attempt_tip[2])
            if attempt_tip[2] > (0.5 * pi):
                attempt_tip[2] -= (0.5 * pi)
        self.temp_tip = attempt_tip

        ligand_tip_xyz = self.convert_spherical_to_rectangular(attempt_tip)
        attempt = self.ligand_base_position + ligand_tip_xyz + nanoparticle_position
        return attempt

    def ligand_brownian(self, array):
        """look for the end of the receptors or 100 times smaller"""
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9 / 100)))) ** 0.5) * 100000 * array  ###   * by time step i.e. 0.001  =------------
        '''using viscosity of water and radius of particle = 50-100 nm; using brownian equations (1) and (3) pg8 + (79) pg 28'''
        r = (random_movement_cartesian[0] ** 2 + random_movement_cartesian[1] ** 2 + random_movement_cartesian[2] ** 2) ** 0.5
        θ = atan(random_movement_cartesian[1] / random_movement_cartesian[0])
        Φ = acos(random_movement_cartesian[2] / (random_movement_cartesian[0] ** 2 + random_movement_cartesian[1] ** 2 + random_movement_cartesian[2] ** 2) ** 0.5)
        '''Equations from: https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates'''
        return np.array([r, θ, Φ])

