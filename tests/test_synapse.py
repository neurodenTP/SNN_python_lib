import unittest
import numpy as np
from synapse import Synapse, SynapseSTDP, SynapseLTPf
from neuron import Neuron

class TestSynapse(unittest.TestCase):

    def setUp(self):
        # Создаем два слоя нейронов-заглушек для тестов с разным числом нейронов
        params = {
            'Ustart': 1,
            'Istart': 0,
            'Sstart': True
        }
        self.pre = Neuron('pre', 3, params)
        self.post = Neuron('post', 4, params)

    def test_random_weight_generation(self):
        syn = Synapse("syn1", self.pre, self.post)
        self.assertEqual(syn.weight.shape, (self.post.N, self.pre.N))
        # Проверка сгенерированных весов не на нули
        self.assertFalse(np.all(syn.weight == 0))

    def test_weight_init_and_shape_validation(self):
        valid_weight = np.ones((self.post.N, self.pre.N))
        syn = Synapse("syn2", self.pre, self.post, weight=valid_weight)
        np.testing.assert_array_equal(syn.weight, valid_weight)

        invalid_weight = np.ones((self.pre.N, self.post.N))  # неправильная форма
        with self.assertRaises(ValueError):
            Synapse("syn3", self.pre, self.post, weight=invalid_weight)

    def test_propagate(self):
        w = np.array([
            [1, 0, 2],
            [0, 0, 1],
            [3, -1, 0],
            [0, 2, 1]
        ])
        syn = Synapse("syn4", self.pre, self.post, weight=w)
        source_output = np.array([1, 2, 3])
        expected = w.dot(source_output)
        result = syn.propagate(source_output)
        np.testing.assert_array_equal(result, expected)

    def test_reset_weight(self):
        syn = Synapse("syn5", self.pre, self.post)
        old_weight = syn.weight.copy()

        # Сброс новых случайных весов
        syn.reset_weight()
        self.assertEqual(syn.weight.shape, (self.post.N, self.pre.N))
        self.assertFalse(np.all(syn.weight == old_weight))  # вероятно, веса разные

        # Сброс заданными весами
        new_w = np.ones((self.post.N, self.pre.N)) * 5
        syn.reset_weight(new_w)
        np.testing.assert_array_equal(syn.weight, new_w)

        # Некорректные размеры для нового веса
        invalid_w = np.ones((self.pre.N, self.post.N))
        with self.assertRaises(ValueError):
            syn.reset_weight(invalid_w)
            
    def test_STDP_update_weight(self):
        params_syn_STDP = {'Aplus': 0.1, 'Aminus': 0.1,
                           'Tpre': 20, 'Tpost': 20}
        w = np.array([
            [1, 0, 2],
            [0, 0, 1],
            [3, -1, 0],
            [0, 2, 1]
        ], dtype = 'float64')
        syn = SynapseSTDP("syn6", self.pre, self.post, 
                          weight=w, params=params_syn_STDP)
        syn.update_weight(1)
        
    def test_LTPf_update_weight(self):
        params_syn_LTPf = {'Aplus': 0.01, 'Aforgetting': 0.005,
                           'Tpre': 20}
        w = np.array([
            [1, 0, 2],
            [0, 0, 1],
            [3, -1, 0],
            [0, 2, 1]
        ], dtype = 'float64')
        syn = SynapseLTPf("syn7", self.pre, self.post, 
                          weight=w, params=params_syn_LTPf)
        syn.update_weight(1)
        print(syn.weight)
        

if __name__ == '__main__':
    unittest.main()