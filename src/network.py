import numpy as np
from neuron import Neuron
from synapse import Synapse
from monitor import Monitor

class Network:
    def __init__(self):
        self.neurons = {}    # словарь: имя слоя -> neuron
        self.synapses = {}   # словарь: имя соединения -> synapse
        self.monitors = {}


    """Операции с нейронами"""
    def add_neuron(self, neuron: Neuron):
        if neuron.name in self.neurons:
            raise ValueError(f"Нейрон с именем {neuron.name} уже существует")
        self.neurons[neuron.name] = neuron
        
    def add_neurons(self, neurons: list[Neuron]):
        for neuron in neurons:
            self.add_neuron(neuron)
            
    def remove_neurons(self, neuron_names: list[str] = None):
        if neuron_names is None:
            self.neurons.clear()
        else:
            for name in neuron_names:
                del self.neurons[name]
    
    def reset_neurons(self, neuron_names: list[str] = None):
        if neuron_names is None:
            for neuron in self.neurons.values():
                neuron.reset()
        else:
            for name in neuron_names:
                neuron[name].reset()
        
    """Операции с синасами"""
    def add_synapse(self, synapse: Synapse):
        if synapse.name in self.synapses:
            raise ValueError(f"Соединение с именем {synapse.name} уже существует")
        if (synapse.pre.name not in self.neurons or 
            synapse.post.name not in self.neurons):
            raise ValueError("Указанные слои должны существовать в сети")
        self.synapses[synapse.name] = synapse
        
    def add_synapses(self, synapses: list[Synapse]):
        for synapse in synapses:
            self.add_synapse(synapse)
    
    def remove_synapses(self, synapse_names: list[str] = None):
        if synapse_names is None:
            self.synapses.clear()
        else:
            for name in synapse_names:
                del self.synapses[name]
    
    """Операции с мониторами"""
    def add_monitor(self, monitor: Monitor):
        if monitor.name in self.monitors:
            raise ValueError(f"Монитор с именем {monitor.name} уже существует")
        self.monitors[monitor.name] = monitor
        
    def add_monitors(self, monitors: list[Monitor]):
        for monitor in monitors:
            self.add_monitor(monitor)   
        
    def remove_monitors(self, monitor_names: list[str] = None):
        if monitor_names is None:
            self.monitors.clear()
        else:
            for name in monitor_names:
                del self.monitors[name]

    def clear_monitors(self, monitor_names: list[str] = None):
        if monitor_names is None:
            for monitor in self.monitors.values():
                monitor.reset()
        else:
            for name in monitor_names:
                monitor[name].clear()


    """Расчетная часть"""
    def step(self, dt, I_external):
        """
        Шаг времени dt.
        I_external - словарь {имя_слоя: входные внешние токи (numpy массив)}
        """
        # Инициализируем входы нейронов каждого слоя внешними токами
        I_in = {}
        for neuron_name, neuron in self.neurons.items():
            # Создаем копии внешних токов, чтобы их модифицировать дальше
            I_in[neuron_name] = np.array(I_external.get(neuron_name, np.zeros(neuron.N)))
    
        # Добавляем входы от соединений
        for synapse in self.synapses.values():
            # Получаем выходные токи исходного слоя
            I_out_pre = np.array(synapse.pre.get_current())
            # Прогоняем их через веса соединения
            I_in_post = synapse.propagate(I_out_pre)
            # Складываем с текущими входными токами целевого слоя
            I_in[synapse.post.name] += I_in_post
    
        # Делаем шаг для каждого слоя с суммарным входом
        for neuron_name, neuron in self.neurons.items():
            neuron.step(dt, I_in[neuron_name])
    
        # Производим обучение для всех соединений
        for synapse in self.synapses.values():
            synapse.update_weight(dt)
                
        # Сбор данных мониторами
        for monitor in self.monitors.values():
            monitor.collect()

    def run(self, dt, inputs):
        """
        Прогон всей сети по временным шагам.
    
        Параметры:
        dt - шаг времени
        inputs - словарь {имя_слоя: np.array формы (num_steps, num_neurons)}
    
        Возвращает:
        outputs - словарь {имя_слоя: np.array выхода формы (num_steps, num_neurons)}
        """
        num_steps = None
        for inp in inputs.values():
            if num_steps is None:
                num_steps = inp.shape[0]
            elif inp.shape[0] != num_steps:
                raise ValueError("Все входы должны иметь одинаковое число временных шагов")
    
        for t in range(num_steps):
            inputs_t = {neuron: inputs[neuron][t] for neuron in inputs}
            self.step(dt, inputs_t)
