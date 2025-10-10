import numpy as np

class Neuron:
    def __init__(self, name, N, params):
        """
        Инициализация базового нейрона.

        Аргументы:
        params (dict): словарь параметров нейрона, включает:
            'Ustart' (float): начальное значение мембранного потенциала
            'Istart' (float): начальный выходной ток
            'Sstart' (bool): начальное состояние
            
        """
        self.N = N
        self.params = params.copy()
        
        self.check_params(['Ustart', 'Istart', 'Sstart'])
        
        self.ustart = self.params['Ustart']
        self.istart = self.params['Istart']
        self.sstart = self.params['Sstart']
        
        self.U = np.empty(N)
        self.I = np.empty(N)
        
        self.U = self.ustart.copy()
        self.I = self.istart.copy()
        self.S = self.sstart.copy()

    def check_params(self, keys):
        N = self.N
        for key in keys:
            if key not in self.params:
                raise ValueError(f"В словаре params нет {key}")
            if not hasattr(self.params[key], '__len__'):
                self.params[key] = np.full(N, self.params[key])
            if len(self.params[key]) != N:
                raise ValueError(f"Длина параметра {key} = {len(self.params[key])} не соответствует количеству нейронов ({N})")    

    def reset(self):
        self.U = self.ustart.copy()
        self.I = self.istart.copy()
        self.S = self.sstart.copy()

    def step(self, dt, Iin):
        pass
        
    def get_potential(self):
        return self.U
    
    def get_current(self):
        return self.I
    
    def get_spike(self):
        return self.S


class LIFNeuron(Neuron):
    def __init__(self, name, N, params):
        """
        Инициализация LIF нейрона.

        Аргументы:
        params (dict): словарь параметров нейрона, включает:
            'Ustart' (float): начальное значение мембранного потенциала
            'Utay' (float): постоянная времени мембранного потенциала (декадация), в тех же единицах времени
            'Uth' (float): порог спайка мембранного потенциала
            'Urest' (float): потенциал покоя мембраны после спайка
            
            'Itay' (float): постоянная времени выхода тока (декадация), в тех же единицах времени
            'Imax' (float): максимальный выходной ток при спайке
            'Istart' (float): начальный выходной ток
            
            'Sstart' (bool): начальное состояние
            
        """
        super().__init__(name, N, params)
        
        self.check_params(['Utay', 'Uth', 'Urest',
                           'Itay', 'Imax'])
        
        self.utay = self.params['Utay']
        self.uth = self.params['Uth']
        self.urest = self.params['Urest']
        
        self.itay = self.params['Itay']
        self.imax = self.params['Imax']

    def step(self, dt, Iin):
        # self.U *= np.exp(- dt / self.utay) 
        self.U *= (1 - dt / self.utay)
        self.U += Iin
        
        self.S = self.U >= self.uth
        ind_spike = np.where(self.S)
        ind_no_spike = np.where(np.invert(self.S))

        self.I[ind_spike] = self.imax[ind_spike]
        self.U[ind_spike] = self.urest[ind_spike]
        
        # self.I[ind_no_spike] *= np.exp(- dt / self.itay[ind_no_spike])
        self.I[ind_no_spike] *= (1 - dt / self.itay[ind_no_spike])


            
class AdaptiveLIFNeuron(Neuron):
    def __init__(self, name, N, params):
        """
        Инициализация LIF Adaptive нейрона.

        Аргументы:
        params (dict): словарь параметров нейрона, включает:
            'Utay' (float): постоянная времени мембранного потенциала (декадация), в тех же единицах времени
            'Uth' (float): порог спайка мембранного потенциала
            'Ustart' (float): начальное значение мембранного потенциала
            
            'Vtay' (float): постоянная времени потенциала восстановления, в тех же единицах времени
            'Vstep' (float): шаг изменения потенциала восстановления при активации
            'Vstart' (float): потенциала восстановления
            
            'Itay' (float): постоянная времени выхода тока (декадация), в тех же единицах времени
            'Imax' (float): максимальный выходной ток при спайке
            'Istart' (float): начальный выходной ток
        """
        super().__init__(name, N, params)

        self.check_params(['Utay', 'Uth',
                           'Vtay', 'Vstep','Vstart',
                           'Itay', 'Imax'])
        
        
        self.utay = self.params['Utay']
        self.uth = self.params['Uth']
        
        self.vtay = self.params['Vtay']
        self.vstep = self.params['Vstep']
        self.vstart = self.params['Vstart'].copy()
        
        self.itay = self.params['Itay']
        self.imax = self.params['Imax']
        
        self.reset()

    def reset(self):
        self.U = self.ustart.copy()
        self.V = self.vstart.copy()
        self.I = self.istart.copy()
        self.S = self.sstart.copy()

    def step(self, dt, Iin):
        # self.U *= np.exp(- dt / self.utay)
        self.U *= (1 - dt / self.utay)
        self.U += Iin
        
        self.S = self.U >= self.uth
        ind_spike = np.where(self.S)
        ind_no_spike = np.where(np.invert(self.S))

        self.I[ind_spike] = self.imax[ind_spike]
        self.V[ind_spike] -= self.vstep[ind_spike]
        self.U[ind_spike] = self.V.copy()[ind_spike]
        
        # self.I[ind_no_spike] *= np.exp(- dt / self.itay[ind_no_spike])
        self.I[ind_no_spike] *= (1 - dt / self.itay[ind_spike])
        # self.U[ind_no_spike] *= np.exp(- dt / self.vtay[ind_spike])
        self.U[ind_no_spike] *= (1 - dt / self.vtay[ind_spike])


# if __name__ == '__main__':
    # N = 4
    # params = {'Ustart': np.zeros(N),
    #           'Istart': 0.,
    #           'Sstart': False}
    # neuron = Neuron('kwa', N, params)
    # print(neuron.get_current())
    # print(neuron.get_potential())
    # print(neuron.get_spike())
    
    # N = 4
    # params = {'Ustart': np.zeros(N),
    #           'Istart': 0.0,
    #           'Sstart': False,
    #           'Utay': 2., 'Uth': 1.0, 'Urest': 0.,
    #           'Itay': 100., 'Imax': 1.0}
    # neuron = LIFNeuron('kwa', N, params)
    
    # neuron.step(1, 0.4 * np.arange(0, N))
    # print(neuron.get_potential())
    # print(neuron.get_current())
    # print(neuron.get_spike())
    
    # neuron.step(1, 0.4 * np.arange(0, N))
    # print(neuron.get_potential())
    # print(neuron.get_current())
    # print(neuron.get_spike())
    
    # neuron.step(1, 0.4 * np.arange(0, N))
    # print(neuron.get_potential())
    # print(neuron.get_current())
    # print(neuron.get_spike())
    
    # N = 4
    # params = {'Ustart': 0.0,
    #           'Vstart': 0.0,
    #           'Istart': 0.0,
    #           'Sstart': 0.0,
    #           'Utay': 2., 'Uth': 1.0,
    #           'Vtay': 10., 'Vstep': 1.0,
    #           'Itay': 100., 'Imax': 1.0}
    # neuron = AdaptiveLIFNeuron('kwa', N, params)
    
    # neuron.step(1, 0.4 * np.arange(0, N))
    # print(neuron.get_potential())
    # print(neuron.V)
    # print(neuron.get_current())
    # print(neuron.get_spike())
    
    # neuron.step(1, 0.4 * np.arange(0, N))
    # print(neuron.get_potential())
    # print(neuron.V)
    # print(neuron.get_current())
    # print(neuron.get_spike())
    
    # neuron.step(1, 0.4 * np.arange(0, N))
    # print(neuron.get_potential())
    # print(neuron.V)
    # print(neuron.get_current())
    # print(neuron.get_spike())