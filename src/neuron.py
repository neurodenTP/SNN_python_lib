import numpy as np

class Neuron:
    def __init__(self, name, N, params):
        self.N = N
        self.params = params
        
        for key in ['Ustart', 'Istart']:
            if key not in self.params:
                raise ValueError(f"В словаре params нет {key}")
            if not hasattr(self.params[key], '__len__'):
                self.params[key] = np.full(N, self.params[key])
            if len(self.params[key]) != N:
                raise ValueError(f"Длина параметра {key} = {len(self.params[key])} не соответствует количеству нейронов ({N})")    
        
        self.ustart = params['Ustart']
        self.istart = params['Istart']
        
        self.U = np.empty(N)
        self.I = np.empty(N)
        
        self.reset()

    def reset(self):
        self.U = self.ustart
        self.I = self.istart

    def step(self, dt, Iin):
        pass
        
    def get_potential(self):
        return self.U
    
    def get_current(self):
        return self.I


class LIFNeuron(Neuron):
    def __init__(self, params):
        """
        Инициализация LIF нейрона.

        Аргументы:
        params (dict): словарь параметров нейрона, включает:
            'Ustart' (float): начальное значение мембранного потенциала (по умолчанию 0.0)
            'Utay' (float): постоянная времени мембранного потенциала (декадация), в тех же единицах времени (по умолчанию 10.0)
            'Uth' (float): порог спайка мембранного потенциала (по умолчанию 1.0)
            'Urest' (float): потенциал покоя мембраны после спайка (по умолчанию 0.0)
            
            'Itay' (float): постоянная времени выхода тока (декадация), в тех же единицах времени (по умолчанию 10.0)
            'Imax' (float): максимальный выходной ток при спайке (по умолчанию 1.0)
            'Istart' (float): начальный выходной ток (по умолчанию 0.0)
        """
        super().__init__(params)
        
        self.utay = params.get('Utay', 10.0)
        self.uth = params.get('Uth', 1.0)
        self.urest = params.get('Urest', 0.0)
        self.ustart = params.get('Ustart', self.urest)
        
        self.itay = params.get('Itay', 10.0)
        self.imax = params.get('Iout_max', 1.0)
        self.istart = params.get('Istart', 0.0)


    def generate(self, N):
        var = {'U': np.full(N, self.ustart),
               'I': np.full(N, self.istart),
               'S': np.full(N, 0)}
        return var

    def reset(self, var):
        var['U'].fill(self.ustart)
        var['I'].fill(self.istart)
        var['S'].fill(0.0)

    def step(self, var, dt, Iin):
        var['U'] *= (1 - dt / self.utay)
        var['U'] += Iin
        
        ind_spike = np.where(var['U'] >= self.uth)
        ind_no_spike = np.where(var['U'] < self.uth)

        var['I'][ind_spike] = self.imax
        var['U'][ind_spike] = self.urest
        var['S'][ind_spike] = 1.
        
        var['I'][ind_no_spike] *= (1 - dt / self.itay)
        var['S'][ind_no_spike] = 0.


            
class AdaptiveLIFNeuron(Neuron):
    def __init__(self, params):
        """
        Инициализация LIF Adaptive нейрона.

        Аргументы:
        params (dict): словарь параметров нейрона, включает:
            'Utay' (float): постоянная времени мембранного потенциала (декадация), в тех же единицах времени (по умолчанию 10.0)
            'Uth' (float): порог спайка мембранного потенциала (по умолчанию 1.0)
            'Ustart' (float): начальное значение мембранного потенциала (по умолчанию 0.0)
            
            'Vtay' (float): постоянная времени потенциала восстановления, в тех же единицах времени (по умолчанию 10.0)
            'Vstep' (float): шаг изменения потенциала восстановления при активации
            'Vstart' (float): потенциала восстановления (по умолчанию 0.0)
            
            'Itay' (float): постоянная времени выхода тока (декадация), в тех же единицах времени (по умолчанию 10.0)
            'Imax' (float): максимальный выходной ток при спайке (по умолчанию 1.0)
            'Istart' (float): начальный выходной ток (по умолчанию 0.0)
        """
        super().__init__(params)
        
        self.utay = params.get('Utay', 10.0)
        self.uth = params.get('Uth', 1.0)
        self.ustart = params.get('Ustart', self.urest)
        
        self.vtay = params.get('Vtay', 1000.0)
        self.vstep = params.get('Vstep', 0.1)
        self.vstart = params.get('Vstart', 0.0)
        
        self.itay = params.get('Itay', 10.0)
        self.imax = params.get('Imax', 1.0)
        self.istart = params.get('Istart', 0.0)

    def generate(self, N):
        var = {'U': np.full(N, self.ustart),
               'V': np.full(N, self.vstart),
               'I': np.full(N, self.istart),
               'S': np.full(N, 0)}
        return var

    def reset(self, var):
        var['U'].fill(self.vstart)
        var['V'].fill(self.ustart)
        var['I'].fill(self.istart)
        var['S'].fill(0)

    def step(self, var, dt, Iin):
        var['U'] *= (1 - dt / self.utay)
        var['U'] += Iin
        
        ind_spike = np.where(var['U'] >= self.uth)
        ind_no_spike = np.where(var['U'] < self.uth)

        var['I'][ind_spike] = self.imax
        var['V'][ind_spike] -= self.vstep
        var['U'][ind_spike] = 1.0 * var['V'][ind_spike]
        var['S'][ind_spike] = 1.
        
        var['I'][ind_no_spike] *= (1 - dt / self.itay)
        var['V'][ind_no_spike] *= (1 - dt / self.vtay)
        var['S'][ind_no_spike] = 0.


if __name__ == '__main__':
    N = 4
    params = {'Ustart': np.zeros(N),
              'Istart': 0.}
    neuron = Neuron('kwa', N, params)
    print(neuron.get_current())
    print(neuron.get_potential())
    