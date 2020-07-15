from operator import is_
import numpy as np
from numpy.random import _bounded_integers
from LigandModel import Ligand
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp

class Receptor:
    def __init__(self, agent_id, base_position, receptor_length, dimension, binding_energy, nanoparticle_radius, ligand_length):
        self.agent_id = agent_id
        self.base_position = base_position
        self.binding_energy = binding_energy
        self.nanoparticle_radius = nanoparticle_radius
        self.ligand_length = ligand_length
        self.dimension = dimension
        self.receptor_length = receptor_length           
        self.temp_tip = None
        self.bound = None
        
        self.tip_position = np.array([np.random.uniform(0, receptor_length), np.random.uniform(0, (2 * pi)), np.random.uniform(0, (0.5 * pi))])
        tip_position_cartesean = self._convert_to_cartesean(self.tip_position)      
        absolute_position = tip_position_cartesean + self.base_position
        self.position = absolute_position

    def step(self, value, receptors_list, nanoparticle_list):
        self.move(value)
        if self.is_space_available(receptors_list, nanoparticle_list):
            is_move_denied, is_bond_broken = self._decide_on_move()
            if not is_move_denied:
                self._accept_move()
            if is_bond_broken:
                self.bound.bound = None
                self.bound = None
            return self.position

    def move(self, amount_to_move):
        #TODO this shouldn't use the same value for both movements. The movement should probably be generated here.
        new_tip_position = self.tip_position + amount_to_move

        cartesean_amount_to_move = self._convert_to_cartesean(amount_to_move)

        new_base_position = self.base_position + cartesean_amount_to_move
        new_base_position[2] = 0 #shouldn't leave the surface
        new_tip_position, new_base_position = self._apply_boundary_conditions(new_tip_position, new_base_position)
        self.temp_base = new_base_position
        self.temp_tip = new_tip_position
        self.temp_position = self._get_absolute_position(new_tip_position, new_base_position)
    
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

    def _metropolis_algorithm_for_bond_breaking(self):
        return np.random.uniform(low=0, high=1) < exp(-self.binding_energy)

    def _calculate_distance(a, b):
        return linalg.norm(a - b)

    def _is_in_range(position, boundaries):
        for (p, b) in zip(position, boundaries):
            if p < 0: return False
            if p > b: return False

    def _get_absolute_position(self, tip_position, base_position):
        attempt_tip_position_cartesean = self._convert_to_cartesean(tip_position)
        absolute_position = attempt_tip_position_cartesean + base_position  # Adding base and tip position
        return absolute_position

    def _apply_boundary_conditions(self, tip_position, base_position):
        boundaries = [self.dimension, self.dimension, self.dimension]
        is_base_position_valid = self._is_in_range(base_position, boundaries)
        if not is_base_position_valid:
            base_position = [self._reflective_boundary_condition(p, b, self.receptor_length) for (p,b) in zip(base_position, boundaries)]
        
        is_tip_in_radius = 0 <= tip_position[0] <= self.receptor_length
        if not is_tip_in_radius:
            tip_position[0] = self._reflective_boundary_condition(tip_position[0], self.receptor_length)
        
        tip_position[1] = tip_position[1] % (2*pi)

        is_tip_above_surface = 0 <= tip_position[2] <= (0.5 * pi)
        if not is_tip_above_surface:
            tip_position[2] = self._reflective_boundary_condition(tip_position[2], 0.5*pi)
        return tip_position, base_position
                   
    def _reflective_boundary_condition(position, boundary, offset=0):
        position = position - offset
        boundary = boundary - offset
        position = abs(position) % 2*boundary
        if position > boundary:
            position = 2* boundary - position
        return position + offset

    def _metropolis_algorithm_for_repulsion(self, seperation, min_allowed_separation):
        return np.random.uniform(low=0, high=1) < exp(-self._repulsive_potential(seperation, min_allowed_separation))

    def _check_space_available(self, receptors_list, nanoparticle_list):
        seperation_when_touching = 2 * self.receptor_radius
        min_allowed_separation_receptors = 1.5 * seperation_when_touching  # van der Waals' radius = 0.5 x separation
        is_move_okay_receptors = self._is_space_available(receptors_list, min_allowed_separation_receptors, self.temp_position)
           
        seperation_when_touching = self.nanoparticle_radius + self.receptor_radius
        min_allowed_separation_nanoparticles = 1.5 * seperation_when_touching
        is_move_okay_nanoparticles = self._is_space_available(nanoparticle_list, min_allowed_separation_nanoparticles, self.temp_position)
        return is_move_okay_nanoparticles and is_move_okay_receptors

    def _is_space_available(self, agents_list, min_allowed_seperation, current_position):
        for i in agents_list:
            seperation = self._calculate_distance(current_position, i)
            is_nanoparticle_too_close = seperation < min_allowed_seperation
            if is_nanoparticle_too_close:  # Checks that if the receptor is a certain distance from the other nanoparticles
                if not self._metropolis_algorithm_for_repulsion(seperation, min_allowed_seperation):
                    return False  
        return True
    
    def _repulsive_potential(distance, max_closeness):
        potential = 4 * 1 * ((max_closeness / distance) ** 12 - (max_closeness / distance) ** 6)
        return potential



