import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from network import Network
from layer import Layer
from neuron import LIFNeuron
from connection import Connection

class TestNetwork(unittest.TestCase):
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
        self.network = Network()
        # Создаем два слоя
        self.layer1 = Layer("Layer1", 2, LIFNeuron, neuron_params)
        self.layer2 = Layer("Layer2", 2, LIFNeuron, neuron_params)
        self.network.add_layer(self.layer1)
        self.network.add_layer(self.layer2)

        # Создаем соединение между слоями
        self.connection = Connection("Conn1", self.layer1, self.layer2)
        self.network.add_connection(self.connection)

    def test_add_same_layer_raises(self):
        with self.assertRaises(ValueError):
            self.network.add_layer(self.layer1)

    def test_add_connection_with_unknown_layer_raises(self):
        fake_layer = Layer("FakeLayer", 1, LIFNeuron, {})
        fake_conn = Connection("FakeConnection", fake_layer, self.layer2)
        with self.assertRaises(ValueError):
            self.network.add_connection(fake_conn)

    def test_add_same_connection_raises(self):
        with self.assertRaises(ValueError):
            self.network.add_connection(self.connection)

    def test_add_monitor_and_remove_monitor(self):
        class DummyMonitor:
            def __init__(self):
                self.name = "Monitor1"
                self.called = False
            def collect(self):
                self.called = True

        monitor = DummyMonitor()
        self.network.add_monitor(monitor)
        with self.assertRaises(ValueError):
            self.network.add_monitor(monitor)  # повторное добавление с таким же именем
        self.assertIn("Monitor1", self.network.monitors)
        self.network.remove_monitor("Monitor1")
        self.assertNotIn("Monitor1", self.network.monitors)
        self.network.remove_monitor("NonExistentMonitor")  # не должно вызывать ошибку

    def test_reset_resets_layers(self):
        # Установим в слоях значения, потом вызовем reset и проверим
        for layer in self.network.layers.values():
            for neuron in layer.neurons:
                neuron.U = 5.0
        self.network.reset()
        for layer in self.network.layers.values():
            for neuron in layer.neurons:
                self.assertEqual(neuron.U, 0.0)

    def test_step_and_run(self):
        dt = 1.0
        # Внешние входы для слоя1 и слоя2
        inputs = {
            'Layer1': np.array([0.1, 0.2]),
            'Layer2': np.array([0.0, 0.0])
        }
        self.network.step(dt, inputs)
        # Проверим, что потенциалы обновились
        for neuron in self.layer1.neurons:
            self.assertTrue(neuron.U > 0 or neuron.Iout > 0)
        for neuron in self.layer2.neurons:
            self.assertTrue(neuron.U >= 0)

        # Прогон по времени (например 3 шага для layer1)
        time_inputs = {
            'Layer1': np.array([[0.1, 0.0],
                                [0.2, 0.1],
                                [0.0, 0.2]])
        }
        # Запуск не должен выдавать ошибки
        self.network.run(dt, time_inputs)

    def test_run_raises_on_mismatched_timesteps(self):
        time_inputs = {
            'Layer1': np.array([[0.1, 0.0],
                                [0.2, 0.1]]),  # 2 timesteps
            'Layer2': np.array([[0.0, 0.1],
                                [0.1, 0.2],
                                [0.2, 0.3]])  # 3 timesteps
        }
        dt = 1.0
        with self.assertRaises(ValueError):
            self.network.run(dt, time_inputs)


if __name__ == '__main__':
    unittest.main()