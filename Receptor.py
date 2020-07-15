from operator import is_
import numpy as np
from numpy.random import _bounded_integers
from Ligand import Ligand
from BaseAgent import BaseAgent

class Receptor(BaseAgent):
    def __init__(self, agent_id, base_position, receptor_length, dimension, binding_energy, nanoparticle_radius, ligand_length, cell_diffusion_coeff, receptor_radius, time_unit):
        self.agent_id = agent_id
        self.base_position = base_position
        self.binding_energy = binding_energy
        self.nanoparticle_radius = nanoparticle_radius
        self.ligand_length = ligand_length
        self.dimension = dimension
        self.receptor_length = receptor_length           
        self.receptor_radius = receptor_radius
        self.time_unit = time_unit
        self.temp_tip = None
        self.bound = None
        self.weighted_diffusion_coef_tip = ((2 * ((1.38064852e-23 * 310.15) / (6 * np.pi * 8.9e-4 * (self.receptor_radius * 1e-9)))) ** 0.5) * 1e9 * self.time_unit
        self.weighted_diffusion_coef_base = ((2 * cell_diffusion_coeff) ** 0.5) * 1e9 * self.time_unit
        
        self.tip_position = np.array([np.random.uniform(0, receptor_length), np.random.uniform(0, (2 * np.pi)), np.random.uniform(0, (0.5 * np.pi))])
        tip_position_cartesean = self._convert_to_cartesean(self.tip_position)      
        absolute_position = tip_position_cartesean + self.base_position
        self.position = absolute_position

    def step(self, random_array_base, random_array_tip, receptors_list, nanoparticle_list):
        
        self.move(random_array_base, random_array_tip)
        if self._check_space_available(receptors_list, nanoparticle_list):
            is_move_denied, is_bond_broken = self._decide_on_move()
            if not is_move_denied:
                self._accept_move()
            if is_bond_broken:
                self.bound.bound = None
                self.bound = None
            return self.position

    def move(self, random_array_base, random_array_tip):
        #TODO, I think the movement of the receptor needs to be constrained, I think it should always be fully exended
        # but I want to check this. I also don't think it should be able to move so its completely flat to the cell surface (e.g. phi=0.5pi)
        distance_to_move_tip = self._brownian_motion(self.weighted_diffusion_coef_tip, random_array_tip, spherical=True)
        distance_to_move_base = self._brownian_motion(self.weighted_diffusion_coef_base, random_array_base)
        new_tip_position = self.tip_position + distance_to_move_tip

        new_base_position = self.base_position + distance_to_move_base
        new_base_position[2] = 0 #shouldn't leave the surface
        new_tip_position, new_base_position = self._apply_boundary_conditions(new_tip_position, new_base_position)
        self.temp_base = new_base_position
        self.temp_tip = new_tip_position
        self.temp_position = self._get_absolute_position(new_tip_position, new_base_position)
    
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

    def _is_in_range(self, position, boundaries):
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
        
        tip_position[1] = tip_position[1] % (2*np.pi)

        is_tip_above_surface = 0 <= tip_position[2] <= (0.5 * np.pi)
        if not is_tip_above_surface:
            tip_position[2] = self._reflective_boundary_condition(tip_position[2], 0.5*np.pi)
        return tip_position, base_position
                   
    def _metropolis_algorithm_for_repulsion(self, seperation, min_allowed_separation):
        return np.random.uniform(low=0, high=1) < np.exp(-self._repulsive_potential(seperation, min_allowed_separation))

    def _check_space_available(self, receptors_list, nanoparticle_list):
        seperation_when_touching = 2 * self.receptor_radius
        min_allowed_separation_receptors = 1.5 * seperation_when_touching  # van der Waals' radius = 0.5 x separation
        is_move_okay_receptors = self._is_space_available(receptors_list, min_allowed_separation_receptors, self.temp_position)
           
        seperation_when_touching = self.nanoparticle_radius + self.receptor_radius
        min_allowed_separation_nanoparticles = 1.5 * seperation_when_touching
        is_move_okay_nanoparticles = self._is_space_available(nanoparticle_list, min_allowed_separation_nanoparticles, self.temp_position)
        return is_move_okay_nanoparticles and is_move_okay_receptors



