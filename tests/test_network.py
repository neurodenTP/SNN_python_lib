import unittest
import numpy as np
from neuron import LIFNeuron
from synapse import Synapse
from network import Network


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.net = Network()

        # Создаем два слоя нейронов по 3 и 2 нейрона
        params = {
            'Ustart': 0.,
            'Istart': 0.,
            'Sstart': False,
            'Utay': 10.,
            'Uth': 1.,
            'Urest': 0.,
            'Itay': 100.,
            'Imax': 1.
        }

        self.neuron1 = LIFNeuron("layer1", 3, params)
        self.neuron2 = LIFNeuron("layer2", 2, params)

        self.net.add_neurons([self.neuron1, self.neuron2])

        # Создаем синапс между слоями
        weight = np.array([[0.5, 0.2, 0.0],
                           [0.1, 0.3, 0.4]])
        self.syn = Synapse("syn1", self.neuron1, self.neuron2, weight=weight)
        self.net.add_synapse(self.syn)

    def test_add_duplicate_neuron_raises(self):
        with self.assertRaises(ValueError):
            self.net.add_neuron(self.neuron1)

    def test_step_propagation(self):
        # Внешние токи для layer1
        I_external = {
            "layer1": np.array([0.3, 0.0, 0.5]),
            "layer2": np.array([0.0, 0.0])
        }

        self.net.step(0.1, I_external)

        # Проверяем, что neuron1 получил входные токи
        np.testing.assert_array_equal(self.neuron1.get_potential(), I_external["layer1"])
        # Проверяем, что neuron2 получил входы от синапса + внешние (здесь 0)
        expected_input_layer2 = self.syn.propagate(self.neuron1.get_current())
        np.testing.assert_array_equal(self.neuron2.get_current(), expected_input_layer2)

    def test_run_multiple_steps(self):
        inputs = {
            "layer1": np.array([
                [1.0, 0.0, 0.5],
                [0.1, 0.2, 0.3],
                [0.0, 0.0, 0.0]
            ]),
            "layer2": np.zeros((3, 2))
        }

        self.net.run(0.1, inputs)

        # После прогонки проверим, что значения текущие обновились
        self.assertEqual(self.neuron1.get_current().shape[0], 3)
        self.assertEqual(self.neuron2.get_current().shape[0], 2)

if __name__ == '__main__':
    unittest.main()