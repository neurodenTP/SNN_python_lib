import queue
from neuron import Neuron
from network import Network
import numpy as np


class InputConstantData:
    """
    Класс для данных, загружаемых целиком (например, из файла или генерации).
    """
    def __init__(self, net:Network):
        self.current = dict.fromkeys(net.neurons.keys())
        self.neuron_size = {}
        for neuron_name in net.neurons:
            self.neuron_size[neuron_name] = net.neurons[neuron_name].N
        
        self.time = None
        self.dt = None
        self.time_size = None
    
    def import_data_from_file(self, sourse, param):
        data_raw = np.loadtxt(self.source)[param['start']:param['stop']:param['step']]    
        
        for neuron_name, current_column in param['current_column']:
            self.current[neuron_name] = param['signal_scaler'] * data_raw[:, current_column]
        
        time = param['time_scaler'] * data_raw[:, param['time_column']]
        self.dt = (time[-1] - time[0]) / (len(time) - 1)
        self.time_size = len(time)
        self.time = time
    
    def generate_time_grid(self, start, stop, dt):
        self.dt = dt
        self.time = np.arange(start, stop, dt)
        self.time_size = len(self.time)

    def generate_current_constant(self, neuron_name, current):
        t_size = self.time_size
        n_size = self.neuron_size[neuron_name]
        self.current[neuron_name] = np.full((t_size, n_size), current)
    
    def generate_current_poisson_intervals(self, neuron_name, tay):
        lb = tay / self.dt
        t_size = self.time_size
        n_size = self.neuron_size[neuron_name]
        self.current[neuron_name] = np.array([self.poisson_intervals_array(t_size, lb)
                                              for n in range(n_size)]).T
    
    def poisson_intervals_array(self, N, lambda_param, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        arr = np.zeros(N)
        positions = [0]
        
        current_pos = 0
        while current_pos < N:
            # Генерируем интервал из пуассоновского распределения
            interval = np.random.poisson(lambda_param)
            
            next_pos = current_pos + interval + 1
            
            if next_pos < N:
                positions.append(next_pos)
                current_pos = next_pos
            else:
                break
        
        for pos in positions:
            if pos < N:
                arr[pos] = 1.
        
        arr[0] = 0.
        
        return arr


# class InputStreamData:
#     """
#     Класс для потоковых данных.
#     Использует очередь для передачи данных другим потокам.
#     """
#     def __init__(self, stream_source=None):
#         self.stream_source = stream_source
#         self.data_queue = queue.Queue()
    
#     def import_data(self):
#         """
#         Прием данных из потока и помещение в очередь.
#         Метод должен быть реализован в наследниках.
#         """
#         raise NotImplementedError("Метод import_data должен быть реализован.")
    
#     def preprocess(self):
#         """
#         Обработка данных из очереди.
#         Может работать в отдельном потоке, вытягивая данные из self.data_queue.
#         """
#         # Пример: обработка элементов очереди, например фильтрация или агрегация.
#         pass
    
#     def add_chunk(self, chunk):
#         """
#         Добавление части данных в очередь.
#         """
#         self.data_queue.put(chunk)
