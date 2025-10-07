import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from connection import Connection
from layer import Layer
from neuron import LIFNeuron

class DummyLearningRule:
    def __init__(self, params):
        pass

    def rule(self, weights, get_source_outputs, get_target_outputs):
        # Простое правило обучения для теста: добавить 0.1 ко всем весам
        return 0.1

class TestConnection(unittest.TestCase):
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
        self.source_layer = Layer("SourceLayer", 3, LIFNeuron, neuron_params)
        self.target_layer = Layer("TargetLayer", 2, LIFNeuron, neuron_params)

    def test_init_default_weights(self):
        conn = Connection("Conn1", self.source_layer, self.target_layer)
        self.assertEqual(conn.name, "Conn1")
        self.assertTrue(conn.weights.shape == (self.target_layer.neurons_num, self.source_layer.neurons_num))
        self.assertIsNone(conn.learning)

    def test_init_with_weights_and_learning(self):
        weights = np.ones((self.target_layer.neurons_num, self.source_layer.neurons_num))
        conn = Connection("Conn2", self.source_layer, self.target_layer, learning_class=DummyLearningRule, learning_params={},
                          weight_matrix=weights)
        self.assertTrue(np.array_equal(conn.weights, weights))
        self.assertIsNotNone(conn.learning)

    def test_init_weights_shape_error(self):
        weights_wrong_shape = np.ones((1,1))  # Не соответствует форме
        with self.assertRaises(ValueError):
            Connection("Conn3", self.source_layer, self.target_layer, weight_matrix=weights_wrong_shape)

    def test_propagate(self):
        conn = Connection("Conn4", self.source_layer, self.target_layer)
        # Установим выводы источника в известные значения
        for i, neuron in enumerate(self.source_layer.neurons):
            neuron.Iout = float(i+1)  # 1.0, 2.0, 3.0
        output = conn.propagate()
        expected = np.dot(conn.weights, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(output, expected)

    def test_update_weights(self):
        weights = np.zeros((self.target_layer.neurons_num, self.source_layer.neurons_num))
        conn = Connection("Conn5", self.source_layer, self.target_layer, learning_class=DummyLearningRule, learning_params={},
                          weight_matrix=weights)
        old_weights = conn.weights.copy()
        conn.update_weights(0.1)
        # weights должны измениться после применения правила обучения (на 0.1)
        self.assertTrue(np.all(conn.weights > old_weights))

    def test_reset_weights(self):
        conn = Connection("Conn6", self.source_layer, self.target_layer)
        old_weights = conn.weights.copy()
        conn.reset_weights()
        # Проверка, что веса обновились (сгенерированы заново)
        self.assertFalse(np.array_equal(old_weights, conn.weights))

        # Прямое задание новых весов
        new_w = np.ones_like(conn.weights)
        conn.reset_weights(new_w)
        self.assertTrue(np.array_equal(conn.weights, new_w))

        # Ошибка при неверной форме
        wrong_w = np.ones((1,1))
        with self.assertRaises(ValueError):
            conn.reset_weights(wrong_w)

if __name__ == '__main__':
    unittest.main()