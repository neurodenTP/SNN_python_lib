

# import unittest
# import numpy as np
# from optimization import STDP, STDPWithDecay



# class TestSTDP(unittest.TestCase):
#     def setUp(self):
#         params = {
#             'A_plus': 0.05,
#             'A_minus': 0.03,
#             'tau_plus': 20.0,
#             'tau_minus': 20.0
#         }
#         self.stdp = STDP(params)

#     def test_rule_no_spikes(self):
#         weights = np.array([[0.5, 0.5], [0.5, 0.5]])
#         pre_spikes = [False, False]
#         post_spikes = [False, False]

#         def pre_outputs(): return pre_spikes
#         def post_outputs(): return post_spikes

#         new_weights = self.stdp.rule(weights.copy(), pre_outputs, post_outputs)
#         np.testing.assert_array_almost_equal(new_weights, weights)

#     def test_rule_with_spikes(self):
#         weights = np.array([[0.4, 0.4], [0.4, 0.4]])
#         pre_spikes = [True, False]
#         post_spikes = [True, False]

#         def pre_outputs(): return pre_spikes
#         def post_outputs(): return post_spikes

#         new_weights = self.stdp.rule(weights.copy(), pre_outputs, post_outputs)
#         # Вес между активным препринроном и постпринроном должен увеличиться на A_plus - A_minus
#         expected = weights.copy()
#         expected[0, 0] += params['A_plus'] - params['A_minus']
#         np.testing.assert_array_almost_equal(new_weights, expected)
#         # Значения должны быть в диапазоне [0,1]
#         self.assertTrue(np.all(new_weights >= 0))
#         self.assertTrue(np.all(new_weights <= 1))

# class TestSTDPWithDecay(unittest.TestCase):
#     def setUp(self):
#         params = {
#             'A_plus': 0.05,
#             'A_minus': 0.03,
#             'tau_plus': 20.0,
#             'tau_minus': 20.0,
#             'decay_rate': 0.1
#         }
#         self.stdp_decay = STDPWithDecay(params)

#     def test_rule_no_spikes(self):
#         weights = np.array([[0.5, 0.5], [0.5, 0.5]])
#         pre_spikes = [False, False]
#         post_spikes = [False, False]

#         def pre_outputs(): return pre_spikes
#         def post_outputs(): return post_spikes

#         new_weights = self.stdp_decay.rule(weights.copy(), pre_outputs, post_outputs)
#         expected = weights.copy() * (1 - params['decay_rate'])
#         np.testing.assert_array_almost_equal(new_weights, expected)

#     def test_rule_with_spikes(self):
#         weights = np.array([[0.4, 0.4], [0.4, 0.4]])
#         pre_spikes = [True, False]
#         post_spikes = [True, False]

#         def pre_outputs(): return pre_spikes
#         def post_outputs(): return post_spikes

#         new_weights = self.stdp_decay.rule(weights.copy(), pre_outputs, post_outputs)
#         expected = weights.copy()
#         expected[0, 0] += params['A_plus'] - params['A_minus']
#         expected *= (1 - params['decay_rate'])
#         np.testing.assert_array_almost_equal(new_weights, expected)
#         # Значения должны лежать в диапазоне [0,1]
#         self.assertTrue(np.all(new_weights >= 0))
#         self.assertTrue(np.all(new_weights <= 1))


# if __name__ == '__main__':
#     unittest.main()