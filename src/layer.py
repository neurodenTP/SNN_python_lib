class Layer:
    def __init__(self, name, neurons_num, neuron_class, neuron_params):
        self.name = name
        self.neurons_num = neurons_num
        self.neuron = neuron_class(neuron_params)
        # Создаем словарь переменных
        self.var = self.neuron.generate(neurons_num)

    def reset(self):
        self.neuron.reset(self.var)

    def step(self, dt, Iin):
        """
        Выполнить один временной шаг для всего слоя.
        Iin - входной сигнал слоя.
        """
        if not hasattr(Iin, '__len__'):
            raise TypeError(f"Iin должен быть списком или массивом длины {self.neurons_num}, получено скалярное значение")
        if len(Iin) != self.neurons_num:
            raise ValueError(f"Длина входного сигнала Iin ({len(Iin)}) не соответствует количеству нейронов ({self.neurons_num})")

        self.neuron.step(self.var, dt, Iin)

    def get_outputs(self):
        # Получить текущие выходные токи всех нейронов слоя
        return 1.0 * self.var['I']

    def get_states(self):
        # Получить текущие мембранные потенциалы всех нейронов слоя
        return 1.0 * self.var['U']
    
    def get_spikes(self):
        # Получить состояния спайкования нейронов
        return 1.0 * self.var['S']