import numpy as np
from numpy.core.defchararray import splitlines
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp
from BaseAgent import BaseAgent

class Ligand(BaseAgent):
    def __init__(self, agent_id, nanoparticle_id, nanoparticle_position, nanoparticle_radius, ligand_length, base_array, tip_array, binding_energy, time_unit):
        self.agent_id = f'Ligand {agent_id}'
        self.base_position = base_array
        self.tip_position = tip_array
        self.nanoparticle_position = nanoparticle_position
        self.nanoparticle_id = nanoparticle_id
        self.ligand_length = ligand_length
        self.nanoparticle_radius = nanoparticle_radius
        self.total_radius = nanoparticle_radius + ligand_length
        self.binding_energy = binding_energy
        self.time_unit = time_unit
        self.temp_tip = None
        self.bound = None
        
        self.weighted_diffusion_coef = ((2 * ((1.38064852e-23 * 310.15) / (6 * pi * 8.9e-4 * (self.ligand_radius * 1e-9)))) ** 0.5) * 1e9 * self.time_unit
        self.base_position = self.convert_spherical_to_rectangular(self.base_position)        
        self.absolute_position = self._get_absolute_position(self.tip_position, self.nanoparticle_position)

    def step(self, value, nanoparticle_position):
        self._update_nanoparticle_position(nanoparticle_position)

        if self.bound:
            if self._metropolis_algorithm_for_unbinding(self.binding_energy):
                self.tip_position = self.temp_tip
                self.bound.bound = None
                self.bound = None
        else:
            self.move(value, nanoparticle_position)

    def move(self, value, nanoparticle_position):
        new_tip_position = self.tip_position + self.brownian_motion(value)
        new_tip_position[0] = self._reflective_boundary_condition(new_tip_position[0], self.ligand_length)
        new_absolute_position = self._get_absolute_position(new_tip_position, nanoparticle_position)
        self.tip_position = new_tip_position
        self.absolute_position = new_absolute_position

    def _update_nanoparticle_position(self, position):
        self.nanoparticle_position = position
        return self.nanoparticle_position

    def _get_absolute_position(self, ligand_tip_position, nanoparticle_position):
        ligand_tip_cartesean = self._convert_to_cartesean(self.tip_position)
        ligand_tip_cartesean = np.sign(self.base_position)*abs(ligand_tip_cartesean) #make sure on correct side
        absolute_position = self.base_position + ligand_tip_cartesean + nanoparticle_position
        return absolute_position


