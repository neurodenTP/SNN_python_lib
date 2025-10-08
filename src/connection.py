import numpy as np


# class Synapse:
#     def __init__(self, pre, post, weight,
#                  learning_class=None, learning_params=None):
#         self.pre = pre
#         self.post = post
        
#         if learning_class == None:
#             self.learning = None
#         else:
#             self.learning = learning_class(learning_params)

#         if weight is None:
#             self.weight = self._generate_random_weight()  
#         else:
#             self.weight = weight

#     def _generate_random_weight(self):
#         return np.random.normal(0, 1)
    
#     def propagate(self, source_output):
#         return self.weights * source_output

#     def update_weight(self, dt):
#         if self.learning is None:
#             return
#         self.weights = self.learning.rule(self.weight, 
#                                           self.pre.get_spike,
#                                           self.post.get_spike,
#                                           dt)
    
#     def reset_weights(self, new_weight=None):
#         if new_weight is None:
#             self.weight = self._generate_random_weight()
#         else:
#             self.weight = new_weight


class Connection:
    def __init__(self, name, source_layer, target_layer, weight_matrix=None,
                 learning_class=None, learning_params=None):
        self.name = name
        self.source_layer = source_layer
        self.target_layer = target_layer
        
        if learning_class == None:
            self.learning = None
        else:
            self.learning = learning_class(learning_params)

        if weight_matrix is None:
            self.weights = self._generate_random_weights()
        else:
            if weight_matrix.shape != (target_layer.neurons_num, source_layer.neurons_num):
                raise ValueError("Размер weight_matrix не соответствует количествам нейронов в слоях")
            self.weights = weight_matrix

    def _generate_random_weights(self):
        return np.random.normal(0, 1, (self.target_layer.neurons_num, 
                                       self.source_layer.neurons_num))

    def propagate(self, source_output):
        return np.dot(self.weights, source_output)

    def update_weights(self, dt):
        if self.learning is None:
            return
        self.weights = self.learning.rule(self.weights, 
                                          self.source_layer.get_spike,
                                          self.target_layer.get_spike,
                                          dt)

    def reset_weights(self, new_weights=None):
        if new_weights is None:
            self.weights = self._generate_random_weights()
        else:
            if new_weights.shape != self.weights.shape:
                raise ValueError("Форма new_weights должна совпадать с формой weights")
            self.weights = new_weights

        