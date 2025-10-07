# import numpy as np

class Neuron:
    def __init__(self, params):
        self.params = params
        self.U = params.get('Ustart', 0.0)  # Мембранный потенциал
        self.Iout = params.get('Ioutstart', 0.0)  # Выходной ток

    def reset(self):
        self.U = self.params.get('Ustart', 0.0)
        self.Iout = self.params.get('Ioutstart', 0.0)

    def step(self, dt, Iin):
        raise NotImplementedError

    def get_Iout(self):
        return self.Iout

    def get_U(self):
        return self.U
    
    def get_is_spike(self):
        return self.is_spike


class LIFNeuron(Neuron):
    def __init__(self, params):
        """
        Инициализация LIF нейрона.

        Аргументы:
        params (dict): словарь параметров нейрона, включает:
            'Ustart' (float): начальное значение мембранного потенциала (по умолчанию 0.0)
            'Ioutstart' (float): начальный выходной ток (по умолчанию 0.0)
            'Utay' (float): постоянная времени мембранного потенциала (декадация), в тех же единицах времени (по умолчанию 10.0)
            'Uth' (float): порог спайка мембранного потенциала (по умолчанию 1.0)
            'Urest' (float): потенциал покоя мембраны после спайка (по умолчанию 0.0)
            'Itay' (float): постоянная времени выхода тока (декадация), в тех же единицах времени (по умолчанию 10.0)
            'refractiontime' (float): время рефрактерного периода, в тех же единицах времени (по умолчанию 5.0)
            'Iout_max' (float): максимальный выходной ток при спайке (по умолчанию 1.0)

        Атрибуты экземпляра:
        self.U (float): текущий мембранный потенциал
        self.Iout (float): текущий выходной ток
        self.is_spike (bool): флаг спайка в текущий шаг
        self.tr (int): счётчик оставшегося рефрактерного времени
        """
        super().__init__(params)
        self.itay = params.get('Itay', 10.0)
        self.utay = params.get('Utay', 10.0)
        self.uth = params.get('Uth', 1.0)
        self.urest = params.get('Urest', 0.0)
        self.iout_max = params.get('Iout_max', 1.0)
        self.refraction_time = params.get('refractiontime', 5.0)
        self.reset()

    def reset(self):
        self.U = self.params.get('Ustart', 0.0)
        self.Iout = self.params.get('Ioutstart', 0.0)
        self.is_spike = False
        self.tr = 0

    def step(self, dt, Iin):
        self.Iout *= (1 - dt / self.itay)
        self.is_spike = False
        
        if self.tr > 0:
            self.tr -= dt
            return

        self.U *= (1 - dt / self.utay)
        self.U += Iin

        if self.U >= self.uth:
            self.Iout = self.iout_max
            self.U = self.urest
            self.tr = self.refraction_time
            self.is_spike = True

            
class LIFAdaptiveNeuron(Neuron):
    def __init__(self, params):
        """
        Инициализация LIF Adaptive нейрона.

        Аргументы:
        params (dict): словарь параметров нейрона, включает:
            'Ustart' (float): начальное значение мембранного потенциала (по умолчанию 0.0)
            'Vstart' (float): потенциала восстановления (по умолчанию 0.0)
            'Ioutstart' (float): начальный выходной ток (по умолчанию 0.0)
            
            'Utay' (float): постоянная времени мембранного потенциала (декадация), в тех же единицах времени (по умолчанию 10.0)
            'Uth' (float): порог спайка мембранного потенциала (по умолчанию 1.0)
            'Vtay' (float): постоянная времени потенциала восстановления, в тех же единицах времени (по умолчанию 10.0)
            'Vstep' (float): шаг изменения потенциала восстановления при активации
            
            'Itay' (float): постоянная времени выхода тока (декадация), в тех же единицах времени (по умолчанию 10.0)
            'refractiontime' (float): время рефрактерного периода, в тех же единицах времени (по умолчанию 5.0)
            'Iout_max' (float): максимальный выходной ток при спайке (по умолчанию 1.0)

        Атрибуты экземпляра:
        self.U (float): текущий мембранный потенциал
        self.Iout (float): текущий выходной ток
        self.is_spike (bool): флаг спайка в текущий шаг
        self.tr (int): счётчик оставшегося рефрактерного времени
        """
        super().__init__(params)
        self.utay = params.get('Utay', 10.0)
        self.uth = params.get('Uth', 1.0)
        
        self.vtay = params.get('Vtay', 1000.0)
        self.vstep = params.get('Vstep', 0.1)
        
        self.itay = params.get('Itay', 10.0)
        self.iout_max = params.get('Iout_max', 1.0)
        self.refraction_time = params.get('refractiontime', 5.0)
        self.reset()

    def reset(self):
        self.U = self.params.get('Ustart', 0.0)
        self.V = self.params.get('Vstart', 0.0)
        self.Iout = self.params.get('Ioutstart', 0.0)
        self.is_spike = False
        self.tr = 0

    def step(self, dt, Iin):
        self.Iout *= (1 - dt / self.itay)
        self.V *= (1 - dt / self.vtay)
        self.is_spike = False
        
        if self.tr > 0:
            self.tr -= dt
            return

        self.U *= (1 - dt / self.utay)
        self.U += Iin
        

        if self.U >= self.uth:
            self.V -= self.vstep
            self.Iout = self.iout_max
            self.U = self.V
            self.tr = self.refraction_time
            self.is_spike = True