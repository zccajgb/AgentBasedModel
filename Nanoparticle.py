from BaseAgent import BaseAgent
import numpy as np
from Ligand import Ligand



class Nanoparticle(BaseAgent):
    def __init__(self, agent_id, nanoparticle_position_xyz, number_of_ligands, nanoparticle_radius, ligand_length, dimension, binding_energy, ligand_radius, receptor_radius, time_unit):
        self.number_of_ligands = number_of_ligands
        self.agent_id = agent_id
        self.position = nanoparticle_position_xyz
        self.time_unit = time_unit
        self.ligands = []
        self.nanoparticle_radius = nanoparticle_radius
        self.ligand_length = ligand_length
        self.bound = False
        self.dimension = dimension
        self.binding_energy = binding_energy
        self.ligand_radius = ligand_radius
        self.receptor_radius = receptor_radius
        self.weighted_diffusion_coef = ((2 * ((1.38064852e-23 * 310.15) / (6 * np.pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9)))) ** 0.5) * 1e9 * time_unit
        self.create_ligands()

    def create_ligands(self):
        base_r_list = np.full(self.number_of_ligands, self.nanoparticle_radius)
        base_theta_list = np.random.uniform(low=0, high=(2 * np.pi), size=self.number_of_ligands)
        base_phi_list = np.random.uniform(low=0, high=np.pi, size=self.number_of_ligands)        
        bases = [np.array([r, theta, phi]) for r, theta, phi in np.nditer([base_r_list, base_theta_list, base_phi_list])] #reshape into one vector

        tip_r_list = np.random.uniform(low=0, high=self.ligand_length, size=self.number_of_ligands)
        tip_theta_list = np.random.uniform(low=0, high=(2 * np.pi), size=self.number_of_ligands)
        tip_phi_list = np.random.uniform(low=0, high=(0.5 * np.pi), size=self.number_of_ligands)
        tips = [np.array([r, theta, phi]) for r, theta, phi in np.nditer([tip_r_list, tip_theta_list, tip_phi_list])]

        for i in range(self.number_of_ligands):
            ligand = Ligand(i, self.agent_id, self.position, self.nanoparticle_radius, self.ligand_length, bases.pop(), tips.pop(), self.binding_energy, self.ligand_radius, self.time_unit)
            self.ligands.append(ligand)

    def move(self, random_array, nanoparticle_list, receptor_list):
        distance_to_move = self._brownian_motion(self.weighted_diffusion_coef, random_array)
        new_position = self.position + distance_to_move
        new_position = self._apply_boundary_conditions(new_position)
        if(self._check_space_available(nanoparticle_list, receptor_list, new_position)):
            self.position = new_position

    def step(self, random_array, nanoparticle_list, receptor_list):
        if not self.bound:
            self.move(random_array, nanoparticle_list, receptor_list)
        random_number_list = list(np.random.normal(size=(self.number_of_ligands, 3)))
        [x.step(random_number_list.pop(), self.position) for x in self.ligands]
        self.bound = any([ligand.bound is not None for ligand in self.ligands])
        return self.position

    def _check_space_available(self, nanoparticle_list, receptor_list, temp_position):
        separation_when_touching_nanoparticle = 2 * self.nanoparticle_radius
        min_allowed_separation_nanoparticles = 1.5* separation_when_touching_nanoparticle
        is_move_okay_nanoparticles = self._is_space_available(nanoparticle_list, min_allowed_separation_nanoparticles, temp_position)
        
        seperation_when_touching_receptor = self.nanoparticle_radius + self.receptor_radius
        min_allowed_separation_receptors = 1.5 * seperation_when_touching_receptor
        is_move_okay_receptors =  self._is_space_available(receptor_list, min_allowed_separation_receptors, temp_position)

        return is_move_okay_nanoparticles and is_move_okay_receptors

    def _apply_boundary_conditions(self, absolute_position):
        absolute_position = [self._reflective_boundary_condition(p, self.dimension, offset = self.nanoparticle_radius + self.ligand_length) for p in absolute_position]
        return absolute_position
