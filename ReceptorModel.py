import numpy as np
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp
from numba import njit


class Receptor:
    def __init__(self, agent_id, base_position, receptor_length, dimension, binding_energy):
        self.agent_id = agent_id
        self.base_position = base_position
        '''spherical coordinates in the format(r,θ,Φ), for the space in a hemisphere of radius 1: r <= 1, 0<= θ <=2π, 0<= Φ <= 0.5π'''
        self.tip_position = np.array(
                [np.random.uniform(0, receptor_length), np.random.uniform(0, (2 * pi)), np.random.uniform(0, (0.5 * pi))])
        self.temp_tip = None
        self.bound = None
        self.receptor_length = receptor_length
        # '''Equations from: https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates'''
        tip_position_rectangular = self.convert_spherical_to_rectangular(self.tip_position)
        self.binding_energy = binding_energy
        '''Keeping receptor in the hemisphere'''
        for i in range(3):
            while not 0 <= tip_position_rectangular[i] <= self.receptor_length:
                if tip_position_rectangular[i] < 0:
                    tip_position_rectangular[i] = abs(tip_position_rectangular[i])
                if tip_position_rectangular[i] > self.receptor_length:
                    tip_position_rectangular[i] = tip_position_rectangular[i] % 2 * self.receptor_length
                    if tip_position_rectangular[i] > self.receptor_length:
                        recoil = tip_position_rectangular[i] - self.receptor_length
                        tip_position_rectangular[i] = self.receptor_length - recoil
        absolute_position = tip_position_rectangular + self.base_position
        self.position = absolute_position
        self.dimension = dimension

    @staticmethod
    @njit(fastmath=True)
    def convert_spherical_to_rectangular(array):
        x = array[0] * sin(array[2]) * cos(array[1])
        y = array[0] * sin(array[2]) * sin(array[1])
        z = array[0] * cos(array[2])  # z value of tip
        return np.array([x, y, z])

    '''Alternate step functions, one with and without movement'''
    '''With Movement'''
    def step(self, value, coordinates_list):
        attempt = self.move(value)
        freedom = self.is_space_available(coordinates_list)
        if freedom:
            if isinstance(self.bound, Ligand):
                distance = self.distance(self.bound.nanoparticle_position, attempt)
                inside_radius = (distance <= self.bound.total_radius)
                '''Returns True if inside and False if outside'''
                if inside_radius:
                    self.position = attempt
                    self.tip_position = self.temp_tip
                    self.base_position = self.temp_base
                    return self.position
                else:  # If movement outside radius
                    if np.random.uniform(low=0, high=1) < exp(-self.binding_energy):  # Bond gets broken
                        self.position = attempt
                        self.tip_position = self.temp_tip
                        self.base_position = self.temp_base
                        self.bound.bound = None
                        self.bound = None
                        return self.position
                    else:
                        return self.position
            else:
                self.position = attempt
                self.tip_position = self.temp_tip
                self.base_position = self.temp_base
                return self.position

    @staticmethod
    @njit(fastmath=True)
    def distance(a, b):
        return linalg.norm(a - b)
    '''Without Movement'''

    def move(self, value):
        attempt_tip = self.tip_position + value 
        '''Convert spherical brownian coordinates into rectangular just for x and y, z doesn't
        matter as always 0 as the receptor is not moving vertically, i.e. stuck to a surface'''
        x = value[0] * sin(value[2]) * cos(value[1])
        y = value[0] * sin(value[2]) * sin(value[1])
        value_rectangular = np.array([x, y, 0])
        attempt_base = self.base_position + value_rectangular
        attempt_base[2] = 0
        attempt = self.get_absolute_position(attempt_tip, attempt_base)
        return attempt

    def get_absolute_position(self, attempt_tip, attempt_base):  
        """Keeping receptor base in the system"""
        for i in range(2):
            upper_limit = self.dimension
            if attempt_base[i] < self.receptor_length:
                attempt_base[i] = abs(attempt_base[i])
            if attempt_base[i] > upper_limit:
                attempt_base[i] -= upper_limit * (attempt_base[i] // upper_limit)
        self.temp_base = attempt_base
        '''Keeping receptor tip in the hemisphere'''
        while not (0 <= attempt_tip[0] <= self.receptor_length):
            if attempt_tip[0] < 0:
                attempt_tip[0] = abs(attempt_tip[0])
            if attempt_tip[0] > self.receptor_length:
                attempt_tip[0] -= self.receptor_length * (attempt_tip[0] // self.receptor_length)
        while not (0 <= attempt_tip[1] <= (2 * pi)):
            if attempt_tip[1] < 0:
                attempt_tip[1] = abs(attempt_tip[1])
            if attempt_tip[1] > (2 * pi):
                attempt_tip[1] = (2 * pi) * (attempt_tip[1] // (2 * pi))
        while not (0 <= attempt_tip[2] <= (0.5 * pi)):
            if attempt_tip[2] < 0:
                attempt_tip[2] = abs(attempt_tip[2])
            if attempt_tip[2] > (0.5 * pi):
                attempt_tip[2] -= 0.5 * pi
        '''Keeping the tip in the system'''
        low_x = attempt_base[0] <= self.receptor_length 
        high_x = attempt_base[0] >= self.dimension - self.receptor_length
        low_y = attempt_base[1] <= self.receptor_length 
        high_y = attempt_base[1] >= self.dimension - self.receptor_length 
        if low_x and low_y:
            if 0.5 * pi < attempt_tip[1] <= pi:
                attempt_tip[1] -= 0.5*pi
            elif pi < attempt_tip[1] <= 1.5 * pi:
                attempt_tip[1] += pi
            elif 1.5 * pi < attempt_tip[1] < 2 * pi:
                attempt_tip[1] += 0.5 * pi
        elif low_x and high_y:
            if 0 < attempt_tip[1] <= 0.5 * pi:
                attempt_tip[1] -= 0.5 * pi
            elif 0.5 * pi < attempt_tip[1] <= pi:
                attempt_tip[1] += pi
            elif pi < attempt_tip[1] < 1.5 * pi:
                attempt_tip[1] += 0.5 * pi
        elif high_x and high_y:
            if 0 < attempt_tip[1] <= 0.5 * pi:
                attempt_tip[1] += pi
            elif 0.5 * pi < attempt_tip[1] < pi:
                attempt_tip[1] += 0.5 * pi
            elif 1.5 * pi < attempt_tip[1] <= 2 * pi:
                attempt_tip[1] -= 0.5 * pi
        elif high_x and low_y:
            if 0 < attempt_tip[1] <= 0.5 * pi:
                attempt_tip[1] += 0.5 * pi
            elif pi < attempt_tip[1] <= 1.5 * pi:
                attempt_tip[1] -= 0.5 * pi
            elif 1.5 * pi < attempt_tip[1] < 2 * pi:
                attempt_tip[1] -= pi
        elif low_x:
            if 0.5 * pi < attempt_tip[1] <= pi:
                attempt_tip[1] -= 0.5*pi
            elif pi < attempt_tip[1] < 1.5 * pi:
                attempt_tip[1] += 0.5 * pi
        elif high_x:
            if 0 <= attempt_tip[1] < 0.5 * pi:
                attempt_tip[1] += 0.5*pi
            elif 1.5 * pi < attempt_tip[1] <= 2 * pi:
                attempt_tip[1] -= 0.5 * pi
        elif low_y:
            if pi < attempt_tip[1] < 2 * pi:
                attempt_tip[1] -= pi
        elif high_y:
            if 0 < attempt_tip[1] < pi:
                attempt_tip[1] += pi
        self.temp_tip = attempt_tip
        tip_position_rectangular = self.convert_spherical_to_rectangular(attempt_tip)
        absolute_position = tip_position_rectangular + attempt_base 
        return absolute_position

    def is_space_available(self, coordinates_list):
        separation = 0.01 * self.receptor_length
        max_closeness = separation + 0.5 * separation  # van der Waals' radius = 0.5 x separation
        count = 0
        for i in coordinates_list:
            distance = self.distance(self.temp_base, i)
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
    def repulsive_potential(distance, max_closeness):
        potential = 4 * 1 * ((max_closeness / distance) ** 12 - (max_closeness / distance) ** 6)
        return potential


from LigandModel import Ligand
