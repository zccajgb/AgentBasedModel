import numpy as np

class BaseAgent():
    def _convert_to_cartesean(self, array):
        x = array[0] * np.sin(array[2]) * np.cos(array[1])
        y = array[0] * np.sin(array[2]) * np.sin(array[1])
        z = array[0] * np.cos(array[2])  # z value of tip
        return np.array([x, y, z])

    def _convert_to_spherical(self, array):
        r = (array[0] ** 2 + array[1] ** 2 + array[2] ** 2) ** 0.5
        theta = np.arctan(array[1] / array[0])
        phi = np.arccos(array[2] / r)
        return np.array([r, theta, phi])

    def _calculate_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _repulsive_potential(self, distance, max_closeness):
        potential = 4 * 1 * ((max_closeness / distance) ** 12 - (max_closeness / distance) ** 6)
        return potential

    def _metropolis_algorithm_for_binding(self, binding_energy):
        return np.random.uniform(low=0, high=1) > np.exp(-binding_energy)
    
    def _metropolis_algorithm_for_unbinding(self, binding_energy):
        return np.random.uniform(low=0, high=1) < np.exp(-binding_energy)

    def _reflective_boundary_condition(self, position, boundary, offset=0):
        position = position - offset
        boundary = boundary - offset
        position = abs(position) % 2*boundary
        if position > boundary:
            position = 2* boundary - position
        return position + offset

    def _brownian_motion(self, weighted_diffusion_coef, random_array, spherical=False):
        random_movement =  weighted_diffusion_coef * random_array
        if spherical:
            random_movement = self._convert_to_spherical(random_movement)        
        return random_movement

    def _is_space_available(self, agents_list, min_allowed_seperation, current_position):
        for i in agents_list:
            seperation = self._calculate_distance(current_position, i)
            is_agent_too_close = seperation < min_allowed_seperation
            if is_agent_too_close:  # Checks that if the receptor is a certain distance from the other nanoparticles
                if not self._metropolis_algorithm_for_repulsion(seperation, min_allowed_seperation):
                    return False  
        return True
    