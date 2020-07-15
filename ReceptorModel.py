from operator import is_
import numpy as np
from LigandModel import Ligand
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp

class Receptor:
    def __init__(self, agent_id, base_position, receptor_length, dimension, binding_energy, nanoparticle_radius, ligand_length):
        self.agent_id = agent_id
        self.base_position = base_position
        '''spherical coordinates in the format(r,theta,phi), for the space in a hemisphere of radius 1: r <= 1, 0<= theta <=2π, 0<= phi <= 0.5π'''
        self.tip_position = np.array(
                [np.random.uniform(0, receptor_length), np.random.uniform(0, (2 * pi)), np.random.uniform(0, (0.5 * pi))])
        self.temp_tip = None
        self.bound = None
        self.receptor_length = receptor_length
        """If numba package not installed"""
        # """Converting from spherical coordinates to rectangular coordinates for tip_position"""
        # x = self.tip_position[0] * sin(self.tip_position[2]) * cos(self.tip_position[1])
        # y = self.tip_position[0] * sin(self.tip_position[2]) * sin(self.tip_position[1])
        # z = self.tip_position[0] * cos(self.tip_position[2])
        # '''Equations from: https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates'''
        # tip_position_rectangular = np.array([x, y, z])
        tip_position_rectangular = self._convert_to_cartesean(self.tip_position)
        self.binding_energy = binding_energy
        self.nanoparticle_radius = nanoparticle_radius
        self.ligand_length = ligand_length
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
        absolute_position = tip_position_rectangular + self.base_position  # Adding base and tip position
        self.position = absolute_position
        self.dimension = dimension

    def _convert_to_cartesean(array):
        x = array[0] * sin(array[2]) * cos(array[1])
        y = array[0] * sin(array[2]) * sin(array[1])
        z = array[0] * cos(array[2])  # z value of tip
        return np.array([x, y, z])

    def _convert_to_spherical(array):
        r = (array[0] ** 2 + array[1] ** 2 + array[
            2] ** 2) ** 0.5
        theta = atan(array[1] / array[0])
        phi = acos(array[2] / r)
        return np.array([r, theta, phi])

    def step(self, value, receptors_list, nanoparticle_list):
        self.move(value)
        if self.is_space_available(receptors_list, nanoparticle_list):
            is_move_denied, is_bond_broken = _decide_on_move()
            if not is_move_denied:
                self._accept_move()
            if is_bond_broken:
                self.bound.bound = None
                self.bound = None
            return self.position

    def _decide_on_move(self):
        is_bound = isinstance(self.bound, Ligand)
        is_move_denied = False
        is_bond_broken = False
        if is_bound:
            distance = self._calculate_distance(self.bound.ligand_base_position, self.temp_position)
            has_moved_far_enough_to_break_bound = distance <= self.bound.ligand_length
            metropolis_result = self._metropolis_algorithm_for_bond_breaking()
            is_move_denied = is_bound and has_moved_far_enough_to_break_bound and not metropolis_result
            is_bond_broken = is_bound and has_moved_far_enough_to_break_bound and metropolis_result
        return is_move_denied, is_bond_broken

    def _accept_move(self):
            self.position = self.temp_position
            self.tip_position = self.temp_tip
            self.base_position = self.temp_base

    def _metropolis_algorithm_for_bond_breaking():
        return np.random.uniform(low=0, high=1) < exp(-self.binding_energy)

    def _calculate_distance(a, b):
        return linalg.norm(a - b)

    def move(self, amount_to_move):
        #TODO this shouldn't use the same value for both movements. The movement should probably be generated here.
        attempt_tip = self.tip_position + amount_to_move  # updates tip_position

        cartesean_amount_to_move = _convert_to_cartesean(amount_to_move)

        attempt_base = self.base_position + cartesean_amount_to_move
        attempt_base[2] = 0 #shouldn't leave the surface
        self.get_absolute_position(attempt_tip, attempt_base)

    def get_absolute_position(self, attempt_tip, attempt_base):  # Returns absolute position and maintains position in hemisphere
        """If numba package not installed"""
        # """Converting from spherical coordinates to rectangular coordinates for tip_position"""
        # x = attempt_tip[0] * sin(attempt_tip[2]) * cos(attempt_tip[1])
        # y = attempt_tip[0] * sin(attempt_tip[2]) * sin(attempt_tip[1])
        # z = attempt_tip[0] * cos(attempt_tip[2])
        # '''Equations from: https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates'''
        # tip_position_rectangular = np.array([x, y, z])6
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
        low_x = attempt_base[0] <= self.receptor_length  # x axis
        high_x = attempt_base[0] >= self.dimension - self.receptor_length
        low_y = attempt_base[1] <= self.receptor_length  # y axis
        high_y = attempt_base[1] >= self.dimension - self.receptor_length  # y axis
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
        attempt_tip_position_rectangular = self._convert_to_cartesean(attempt_tip)
        absolute_position = attempt_tip_position_rectangular + attempt_base  # Adding base and tip position
        self.temp_position = absolute_position

    def is_space_available(self, receptors_list, nanoparticle_list):
        separation1 = 7
        max_closeness1 = separation1 + 0.5 * separation1  # van der Waals' radius = 0.5 x separation
        count = 0
        for i in receptors_list:
            distance = self._calculate_distance(self.temp_position, i)
            if distance >= separation1:  # Checks that if the receptor is a certain distance from the other receptors
                if distance <= max_closeness1:  # Checks if close enough for repulsive potential
                    if np.random.uniform(low=0, high=1) < exp(-self._repulsive_potential(distance, max_closeness1)):  # If there is repulsion then check  # tolerable_potential:
                        count += 1
                    else:
                        return False  # Repulsive potential not exceeded
                else:
                    count += 1
            elif distance < separation1:
                return False
        if count == len(receptors_list):
            separation2 = self.nanoparticle_radius  # + self.ligand_length
            max_closeness2 = separation2 + 0.5 * separation2  # van der Waals' radius = 0.5 x separation
            count = 0
            for i in nanoparticle_list:
                distance = self._calculate_distance(self.temp_position, i)
                if distance >= separation2:  # Checks that if the receptor is a certain distance from the other nanoparticles
                    if distance <= max_closeness2:  # Checks if close enough for repulsive potential
                        if np.random.uniform(low=0, high=1) < exp(0 * -self._repulsive_potential(distance, max_closeness2)):  # If there is repulsion then check  # tolerable_potential:
                            count += 1
                        else:
                            return False  # Repulsive potential not exceeded
                    else:
                        count += 1
                elif distance < separation2:
                    return False
            if count == len(nanoparticle_list):
                return True  # Returns True when there is space available
    
    def _repulsive_potential(distance, max_closeness):
        potential = 4 * 1 * ((max_closeness / distance) ** 12 - (max_closeness / distance) ** 6)
        return potential



