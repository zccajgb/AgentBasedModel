import unittest
import numpy as np
from Master import Master
from NanoparticleModel import Nanoparticle
from ReceptorModel import Receptor
from itertools import combinations

class Test_Create_Receptors(unittest.TestCase):
 
    def test_correct_number_of_receptors_are_created(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(10, 1, 1, 1)
        master.create_receptors(10, 1)
        self.assertEqual(10, master.number_of_receptors)

        agents = [i for i in master.agents if isinstance(i, Receptor)]
        self.assertEqual(10, len(agents))

    def test_all_receptors_have_correct_length(self):
        master = Master(100, 1)
        master.create_nanoparticles_and_ligands(10, 1, 1, 1)
        master.create_receptors(10, 1)
        [self.assertEqual(1, a.receptor_length) for a in master.agents if isinstance(a, Receptor)]

    def test_no_receptor_bases_are_touching(self):
        for _ in range(100):
            master = Master(100, 1)
            master.create_nanoparticles_and_ligands(10, 1, 1, 1)
            master.create_receptors(100, 1)
            positions = [a.base_position for a in master.agents if isinstance(a, Receptor)]
            combs = combinations(positions, 2)
            distance = [np.linalg.norm(a-b) for (a,b) in combs]
            [self.assertNotAlmostEqual(d, 0) for d in distance]

    def test_no_receptor_tips_are_touching(self):
        for _ in range(100):
            master = Master(100, 1)
            master.create_nanoparticles_and_ligands(10, 1, 1, 1)
            master.create_receptors(100, 1)
            positions = [a.tip_position for a in master.agents if isinstance(a, Receptor)]
            combs = combinations(positions, 2)
            distance = [np.linalg.norm(a-b) for (a,b) in combs]
            [self.assertNotAlmostEqual(d, 0) for d in distance]

    def test_no_receptor_bases_outside_area(self):
        for _ in range(100):
            master = Master(500, 1)
            master.create_nanoparticles_and_ligands(100, 10, 10, 1)
            positions = [i.base_position for i in master.agents if isinstance(i, Receptor)]
            self.assertTrue(all([(i[0] < 1000 and i[1] < 1000 and i[2] < 1000) for i in positions]))
            self.assertTrue(all([(i[0] > 0 and i[1] > 0 and i[2] > 0) for i in positions]))

    def test_no_receptor_tips_outside_area(self):
        for _ in range(100):
            master = Master(500, 1)
            master.create_nanoparticles_and_ligands(100, 10, 10, 1)
            master.create_receptors(100, 1)
            positions = [i.tip_position for i in master.agents if isinstance(i, Receptor)]
            master.create_receptors(100, 1)
            self.assertTrue(all([(i[0] < 1000 and i[1] < 1000 and i[2] < 1000) for i in positions]))
            self.assertTrue(all([(i[0] > 0 and i[1] > 0 and i[2] > 0) for i in positions]))

    def test_error_is_raised_if_too_many_receptors_for_area(self):
        master = Master(1, 1)
        self.assertRaises(RuntimeError, master.create_receptors, 100, 1)
