import numpy as np
import matplotlib.pyplot as plt
from neuron import Neuron
from synapse import Synapse


class Monitor():
    def __init__(self, name: str, objs, save_step: int = 1, max_points: int = None):
        """
        name: имя монитора
        objs: объект или список объект для мониторинга (Neuron/Synapse)
        save_step: сохранять данные каждый n-й вызов collect
        max_points: максимальное количество последних точек для хранения (None - без ограничения)
        """
        self.name = name
        if not isinstance(objs, list):
            objs = [objs]
        self.objs = objs
        self.save_step = save_step
        self.max_points = max_points
        self.counter = 0

        self.data = {obj.name: [] for obj in self.objs}
    
    def _request_data_from_obj(self, obj) -> np.ndarray:
        pass
    
    def get_data(self, obj_name) -> np.ndarray:
        if obj_name not in self.data:
            raise ValueError(f"Данные для {obj_name} не собраны")
        return self.data[obj_name]
    
    def collect(self):
        self.counter += 1
        if self.counter % self.save_step != 0:
            return
        for obj in self.objs:
            datum = self._request_data_from_obj(obj).copy()
            points = self.data[obj.name]
            points.append(datum)
            if self.max_points is not None and len(points) > self.max_points:
                points.pop(0)
    
    def clear(self):
        self.data = {obj.name: [] for obj in self.objs}


class MonitorNeuron(Monitor):    
    def _plot_line(self, layer_name, dt, xlabel, ylabel, title):
        data = np.array(self.get_data(layer_name))
        times = (self.counter - len(data) + np.arange(len(data))) * dt

        for i in range(len(data[0])):
            plt.plot(times, data[:, i], label=str(i))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
            
    def _plot_imshow(self, layer_name, dt, xlabel, ylabel, title):
        data = np.array(self.get_data(layer_name))
        times = (self.counter - len(data) + np.arange(len(data))) * dt
        plt.imshow(data.T, extent=[times[0], times[-1], 0, data.shape[1]], 
                   aspect='auto', origin='lower', interpolation='none')
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        

class MonitorPotential(MonitorNeuron):
    def _request_data_from_obj(self, neuron: Neuron) -> np.ndarray:
        return neuron.get_potential()

    def plot_line(self, layer_name, dt):
        self._plot_line(layer_name, dt, 'Время (мс)', 'Нейрон', 
                        f"Потенциалы слоя {layer_name}")
        
    def plot_imshow(self, layer_name, dt):
        self._plot_imshow(layer_name, dt, 'Время (мс)', 'Нейрон', 
                          f"Потенциалы слоя {layer_name}")

class MonitorCurrent(MonitorNeuron):
    def _request_data_from_obj(self, neuron: Neuron) -> np.ndarray:
        return neuron.get_current()
    
    def plot_line(self, layer_name, dt):
        self._plot_line(layer_name, dt, 'Время (мс)', 'Нейрон', 
                        f"Токи слоя {layer_name}")
        
    def plot_imshow(self, layer_name, dt):
        self._plot_imshow(layer_name, dt, 'Время (мс)', 'Нейрон', 
                          f"Токи слоя {layer_name}")


class MonitorSpike(MonitorNeuron):
    def _request_data_from_obj(self, neuron: Neuron) -> np.ndarray:
        outputs = neuron.get_spike()
        return [i for i, val in enumerate(outputs) if val]

    def plot_scatter(self, layer_name, dt):
        #TODO rewrite
        data = self.get_data(layer_name)
        
        N = 0
        for spikes in data:
            for spike in spikes:
                if spike > N:
                    N = 1 * spike

        data_np = [np.ones((0,2)) for i in range(N+1)]
        for t, spikes in enumerate(data):
            for spike in spikes:
                data_np[spike] = np.vstack((data_np[spike], np.array([[t*dt, spike]])))
                
        plt.figure()
        for data_one in data_np:
            plt.scatter(data_one[:,0], data_one[:,1])
            
        plt.xlabel('Время (мс)')
        plt.ylabel('Нейроны')
        plt.title(f"Спайки слоя {layer_name}")
        plt.show()
        

class MonitorWeigts(Monitor):
    def _request_data_from_obj(self, synapse: Synapse) -> np.ndarray:
        return synapse.get_weight()
    
    def plot_imshow(self, connection_name, dt):
        data = np.array(self.get_data(connection_name))
        arr = np.array(data)
        if arr.ndim == 3:
            # Среднее по времени
            weights = np.mean(arr, axis=0)
        else:
            weights = arr

        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Нейроны исходного слоя')
        plt.ylabel('Нейроны целевого слоя')
        # timespan = (self.counter - len(data), self.counter) if self.max_points else (0, len(data))
        plt.title(f"Весы соединения {connection_name}\nПоследние {len(data)} шагов (dt={dt})")
        plt.show()
        
    def plot_line(self, connection_name, dt):
        data = np.array(self.get_data(connection_name))
        times = (self.counter - len(data) + np.arange(len(data))) * dt
        for post in range(len(data[0])):
            for pre in range(len(data[0, 0])):
                plt.plot(times, data[:, post, pre], label=f"{pre} to {post}")
        
        plt.xlabel('Времяб (мс)')
        plt.ylabel('Веса')
        plt.legend()
        plt.title(f"Весы соединения {connection_name}\nПоследние {len(data)} шагов (dt={dt})")
        plt.show()