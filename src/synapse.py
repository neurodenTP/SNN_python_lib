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
        
        self.check_params(['Aplus', 'Aminus', 'Tplus', 'Tminus'])
        self.a_plus = self.params['Aplus']
        self.a_minus = self.params['Aminus']
        self.tay_plus = self.params['Tplus']
        self.tay_minus = self.params['Tminus']
        
        self.pre_trace = np.zeros(self.weight_shape[1])
        self.post_trace = np.zeros(self.weight_shape[0])
    
    def update_weight(self, dt: float):
        """Обновление весов синапса"""
        
        """
        Матрично
        идем по пре, обновляем все следы * и прибавить если спайк
        идет по пост, обновляем все следы * и прибавить если спайк
        
        Итак имеем следы.
        
        умножаем спайки пост на след пре на А - прибавляем к весу
        умножааем спайки пре на след пост на А - вычитаем 
        """
        
class SynapseLTPf(Synapse):
    def __init__(self, name: str, preNeuron, postNeuron,
                 weight: np.ndarray = None, params = None):
        super().__init__(name, preNeuron, postNeuron, weight, params)
        
        self.check_params(['Aplus', 'Tplus', 'Aforgetting'])
        self.a_plus = self.params['Aplus']
        self.tay_plus = self.params['Tplus']
        self.a_forg = self.params['Aforgetting']
        
        self.pre_trace = np.zeros(self.weight_shape[1])
        
    def update_weight(self, dt: float):
        """Обновление весов синапса"""
        
        """
        Матрично
        идем по пре, обновляем все следы * и прибавить если спайк
        
        Итак имеем следы.
        
        умножаем спайки пост на след пре на А - прибавляем к весу
        вычитаем на фиксированное число
        """