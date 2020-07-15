import unittest
import numpy as np
from Master import Master
from NanoparticleModel import Nanoparticle
from itertools import combinations

class Test_is_space_available_nanoparticle(unittest.TestCase):
 
    def test_two_particles_cannot_be_in_same_positions(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(2, 10, 10, 1)
        master.agents[0].position = np.array([1, 1, 1])
        position = np.array([1, 1, 1])

        self.assertFalse(master._check_space_available_nanoparticle(position))
    
    def test_two_particles_can_be_far_away(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(2, 10, 10, 1)
        master.agents[0].position = np.array([1, 1, 1])
        position = np.array([50, 50, 50])

        self.assertTrue(master._check_space_available_nanoparticle(position))

    def test_two_particles_cannot_be_touching(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(2, 10, 10, 1)
        master.agents[0].position = np.array([1, 1, 1]) 
        position = np.array([23-1e-10, 1, 1])
        self.assertFalse(master._check_space_available_nanoparticle(position))
        x = 22/np.sqrt(3) + 1 - 1e-10
        position = np.array([x, x, x])
        self.assertFalse(master._check_space_available_nanoparticle(position))

    def test_two_particles_can_be_really_close_without_touching(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(2, 10, 10, 1)
        master.agents[0].position = np.array([1, 1, 1]) 
        position = np.array([23, 1, 1])
        self.assertTrue(master._check_space_available_nanoparticle(position))
        x = 22/np.sqrt(3) + 1
        position = np.array([x, x, x])
        self.assertTrue(master._check_space_available_nanoparticle(position))
  
