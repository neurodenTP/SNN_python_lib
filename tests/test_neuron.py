import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from neuron import Neuron, LIFNeuron, AdaptiveLIFNeuron
import numpy as np


class TestNeuron(unittest.TestCase):

    def test_neuron_init_valid_and_invalid_params(self):
        N = 10
        # Корректные параметры
        params_np = {
            'Ustart': np.zeros(N),
            'Istart': np.zeros(N),
            'Sstart': np.zeros(N, dtype=bool)
        }
        neuron = Neuron('test', N, params_np)
        self.assertTrue(np.all(neuron.U == 0))
        self.assertTrue(np.all(neuron.I == 0))
        self.assertTrue(np.all(neuron.S == False))

        # Корректные параметры
        params_single = {
            'Ustart': 1,
            'Istart': 0,
            'Sstart': False
        }
        neuron = Neuron('test', N, params_single)
        self.assertTrue(np.all(neuron.U == 1))
        self.assertTrue(np.all(neuron.I == 0))
        self.assertTrue(np.all(neuron.S == False))
        
        # Не все параметры указаны
        params_invalid = {
            'Ustart': 1,
            'Istart': 0,
        }
        with self.assertRaises(ValueError):
            neuron = Neuron('test', N, params_invalid)

        # Параметры с несоответствующей длиной массивов
        params_invalid_len = {
            'Ustart': np.zeros(N),
            'Istart': np.zeros(N + 1),
            'Sstart': np.zeros(N, dtype=bool)
        }
        with self.assertRaises(ValueError):
            Neuron('test', N, params_invalid_len)

    def test_lif_neuron_step_and_spike(self):
        N = 4
        params = {
            'Ustart': np.zeros(N),
            'Istart': np.zeros(N),
            'Sstart': np.zeros(N, dtype=bool),
            'Utay': np.full(N, 2.0),
            'Uth': np.full(N, 1.0),
            'Urest': np.zeros(N),
            'Itay': np.full(N, 2.0),
            'Imax': np.full(N, 1.0)
        }
        neuron = LIFNeuron('test', N, params)

        input_current = 0.4 * np.arange(N)
        neuron.step(1, input_current)

        expected_potential = np.array([0, 0.4, 0.8, 0])
        expected_current = np.array([0, 0, 0, 1])
        expected_spike = np.array([False, False, False, True])

        np.testing.assert_array_almost_equal(neuron.get_potential(), expected_potential)
        np.testing.assert_array_equal(neuron.get_current(), expected_current)
        np.testing.assert_array_equal(neuron.get_spike(), expected_spike)

        neuron.step(1, input_current)

        expected_potential2 = np.array([0, 0.4 * (2 - 1 / 2), 0, 0])
        expected_current2 = np.array([0, 0, 1, 1])
        expected_spike2 = np.array([False, False, True, True])

        np.testing.assert_array_almost_equal(neuron.get_potential(), expected_potential2)
        np.testing.assert_array_equal(neuron.get_current(), expected_current2)
        np.testing.assert_array_equal(neuron.get_spike(), expected_spike2)

    def test_lif_neuron_long_simulation_consistency(self):
        N = 1
        Utay = 100.
        dt = 1.
        time_step_amount = 100
        
        params = {
            'Ustart': np.array([1.0]),
            'Istart': np.array([0.0]),
            'Sstart': np.array([False]),
            'Utay': np.array([Utay]),
            'Uth': np.array([2.0]),
            'Urest': np.array([0.0]),
            'Itay': np.array([2.0]),
            'Imax': np.array([1.0])
        }
        neuron = LIFNeuron('test', N, params)

        for _ in range(time_step_amount):
            neuron.step(dt, 0)

        expected_val = np.exp(- time_step_amount * dt / Utay)
        actual_val = neuron.get_potential()[0]

        self.assertAlmostEqual(expected_val, actual_val, places=2)


if __name__ == '__main__':
    unittest.main()