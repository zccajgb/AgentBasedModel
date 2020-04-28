import unittest
import numpy as np
from Master import Master
from NanoparticleModel import Nanoparticle
from itertools import combinations

class Test_Create_Nanoparticles_And_Ligands(unittest.TestCase):
 
    def test_correct_number_of_nanoparticles_are_created(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(10, 1, 1, 1)
        self.assertEqual(10, master.number_of_nanoparticles)

        agents = [i for i in master.agents if isinstance(i, Nanoparticle)]
        self.assertEqual(10, len(agents))

    def test_correct_number_ligands_are_created(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(100, 10, 1, 1)
        self.assertEqual(master.ligand_length, 1)
        self.assertEqual(master.number_of_ligands, 10)
        ligands = [i.ligands for i in master.agents]
        for l in ligands:
            self.assertEqual(10, len(l))

    def test_no_nanoparticles_are_touching(self):
        master = Master(1000, 1)
        master.create_nanoparticles_and_ligands(100, 10, 10, 1)
        positions = [i.position for i in master.agents]
        combs = combinations(positions, 2)
        distance = [np.linalg.norm(a-b) for (a,b) in combs]
        self.assertTrue(all([i > 22 for i in distance]))

    def test_no_nanoparticles_outside_area(self):
        master = Master(1000, 1)
        master.create_nanoparticles_and_ligands(100, 10, 10, 1)
        positions = [i.position for i in master.agents]
        self.assertTrue(all([(i[0] < 1000 and i[1] < 1000 and i[2] < 1000) for i in positions]))
        self.assertTrue(all([(i[0] > 0 and i[1] > 0 and i[2] > 0) for i in positions]))

    def test_error_is_raised_if_too_many_nanoparticles_for_area(self):
        master = Master(10, 1)
        self.assertRaises(RuntimeError, master.create_nanoparticles_and_ligands, 5, 5, 1, 16)
