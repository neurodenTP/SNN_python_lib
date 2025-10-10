import numpy as np

class Synapse:
    """
    Класс синапса между двумя слоями нейронов.
    Хранит веса связей и реализует передачу сигнала.
    """

    def __init__(self, name: str, preNeuron, postNeuron, weight: np.ndarray = None):
        self.name = name
        self.pre = preNeuron
        self.post = postNeuron
        self.weight_shape = (self.post.N, self.pre.N)

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