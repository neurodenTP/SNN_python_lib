import numpy as np

class Synapse:
    """
    Класс синапса между двумя слоями нейронов.
    Хранит веса связей и реализует передачу сигнала.
    """

    def __init__(self, name: str, preNeuron, postNeuron,
                 weight: np.ndarray = None, params = None):
        self.name = name
        self.pre = preNeuron
        self.post = postNeuron
        self.weight_shape = (self.post.N, self.pre.N)
        self.params = params

        if weight is None:
            self.weight = self.generate_random_weight()
        else:
            if weight.shape != self.weight_shape:
                raise ValueError(f"Размер weight ({weight.shape}) не соответствует количествам нейронов в слоях ({self.weight_shape})")
            self.weight = weight

    def generate_random_weight(self) -> np.ndarray:
        """Генерация случайной матрицы весов с нормальным распределением."""
        return np.random.normal(0, 1, self.weight_shape)

    def propagate(self, pre_current: np.ndarray) -> np.ndarray:
        """
        Пропускает сигнал через синапс — умножает выходной вектор 
        предшествующего слоя на матрицу весов.
        """
        return np.dot(self.weight, pre_current)

    def get_weight(self) -> np.ndarray:
        return self.weight

    def update_weight(self, dt: float):
        """Обновление весов синапса (метод-заглушка)."""
        pass

    def reset_weight(self, new_weight: np.ndarray = None):
        """
        Сброс весов — либо генерация новой матрицы весов,
        либо обновление заданной матрицей new_weight.
        """
        if new_weight is None:
            self.weight = self.generate_random_weight()
        else:
            if new_weight.shape != self.weight_shape:
                raise ValueError(f"Форма new_weight ({new_weight.shape}) должна совпадать с формой weight ({self.weight_shape})")
            self.weight = new_weight
    
    def check_params(self, keys: list[str]):
        for key in keys:
            if key not in self.params:
                raise ValueError(f"В словаре params нет {key}")
            

class SynapseSTDP(Synapse):
    def __init__(self, name: str, preNeuron, postNeuron,
                 weight: np.ndarray = None, params = None):
        super().__init__(name, preNeuron, postNeuron, weight, params)
        
        self.check_params(['Aplus', 'Aminus', 'Tpre', 'Tpost'])
        self.a_plus = self.params['Aplus']
        self.a_minus = self.params['Aminus']
        self.tay_pre = self.params['Tpre']
        self.tay_post = self.params['Tpost']
        
        self.trace_pre = np.zeros(self.weight_shape[1])
        self.trace_post = np.zeros(self.weight_shape[0])
    
    def update_weight(self, dt: float):
        """
        Обновление весов синапса
        dd - dirac delta
        
        dtrace_ / dt = -trace_ / tay_ + Sum(dd(t-t_))
        
        dw/dt = Aplus * trace_pre * dd(t-t_post) * (1-w) -
                - Aminus * trace_post * dd(t-t_pre) * w
        
        """
        trace_pre = self.trace_pre
        trace_post = self.trace_post
        weight = self.weight
        spike_pre = self.pre.get_spike()
        spike_post = self.post.get_spike()
        
        trace_pre *= (1 - dt / self.tay_pre)
        trace_post *= (1 - dt / self.tay_pre)
        
        trace_pre += spike_pre
        trace_post += spike_post
        
        weight += self.a_plus * dt * spike_post[:, np.newaxis] * trace_pre * (1 - weight)
        weight -= self.a_minus * spike_pre * trace_post[:, np.newaxis] * dt * weight


class SynapseLTPf(Synapse):
    def __init__(self, name: str, preNeuron, postNeuron,
                 weight: np.ndarray = None, params = None):
        super().__init__(name, preNeuron, postNeuron, weight, params)
        
        self.check_params(['Aplus', 'Tpre', 'Aforgetting'])
        self.a_plus = self.params['Aplus']
        self.tay_pre = self.params['Tpre']
        self.a_forg = self.params['Aforgetting']
        
        self.trace_pre = np.zeros(self.weight_shape[1])
        
    def update_weight(self, dt: float):
        """
        Обновление весов синапса
        dd - dirac delta
        
        dtrace_pre / dt = -trace_pre / tay_pre + Sum(dd(t-t_pre))
        
        dw/dt = (Aplus * trace_pre * (1-w) - Aforg * w) * dd(t-t_post)
        """
        trace_pre = self.trace_pre
        weight = self.weight
        
        trace_pre *= (1 - dt / self.tay_pre)
        trace_pre += self.pre.get_spike()
        spike_post = self.post.get_spike()
        
        weight += (self.a_plus * trace_pre * spike_post[:, np.newaxis] * (1 - weight) - 
                   self.a_forg * spike_post[:, np.newaxis] * weight) * dt
        
        
# if __name__ == '__main__':
#     from neuron import Neuron
#     params = {
#         'Ustart': 1,
#         'Istart': 0,
#         'Sstart': True
#     }
#     pre = Neuron('pre', 3, params)
#     post = Neuron('post', 4, params)
    
#     params_syn_STDP = {'Aplus': 0.01, 'Aminus': 0.01,
#                        'Tpre': 20, 'Tpost': 20}
#     w = np.array([
#         [1, 0, 2],
#         [0, 0, 1],
#         [3, -1, 0],
#         [0, 2, 1]
#     ], dtype = 'float64')
#     syn = SynapseSTDP("syn6", pre, post, 
#                       weight=w, params=params_syn_STDP)
#     syn.update_weight(1)