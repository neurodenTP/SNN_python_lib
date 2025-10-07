import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from neuron import LIFNeuron
from layer import Layer

class TestLayerWithLIFNeuron(unittest.TestCase):
    def setUp(self):
        neuron_params = {
            'Ustart': 0.0,
            'Ioutstart': 0.0,
            'Utay': 10.0,
            'Uth': 1.0,
            'Urest': 0.0,
            'Itay': 10.0,
            'refractiontime': 5.0,
            'Iout_max': 1.0
        }
        self.layer = Layer("TestLayerLIF", 4, LIFNeuron, neuron_params)

    def test_initialization(self):
        self.assertEqual(len(self.layer.neurons), 4)
        for neuron in self.layer.neurons:
            self.assertEqual(neuron.U, 0.0)
            self.assertEqual(neuron.Iout, 0.0)

    def test_reset(self):
        for neuron in self.layer.neurons:
            neuron.U = 5.0
            neuron.Iout = 0.5
        self.layer.reset()
        for neuron in self.layer.neurons:
            self.assertEqual(neuron.U, 0.0)
            self.assertEqual(neuron.Iout, 0.0)

    def test_step_and_output(self):
        dt = 1.0
        inputs = [0.5, 0.1, 0.0, 0.2]
        self.layer.reset()
        self.layer.step(dt, inputs)
        outputs = self.layer.get_outputs()
        states = self.layer.get_states()
        spikes = self.layer.get_spikes()

        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(states), 4)
        self.assertEqual(len(spikes), 4)

        for i, neuron in enumerate(self.layer.neurons):
            self.assertAlmostEqual(outputs[i], neuron.Iout)
            self.assertAlmostEqual(states[i], neuron.U)
            self.assertEqual(spikes[i], neuron.is_spike)

    def test_step_input_length_mismatch(self):
        dt = 1.0
        with self.assertRaises(TypeError):
            self.layer.step(dt, 1.0)  # скаляр вместо списка/массива

        with self.assertRaises(ValueError):
            self.layer.step(dt, [0.1, 0.2])  # длина не совпадает с числом нейронов

if __name__ == '__main__':
    unittest.main()