import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cProfile, pstats
import numpy as np

from neuron import LIFNeuron
from synapse import SynapseSTDP
from network import Network
from monitor import MonitorPotential, MonitorCurrent, MonitorSpike

net = Network()

# Добавляем слои нейронов
params_neuron_in = {
    'Ustart': 0.0,
    'Istart': 0.0,
    'Sstart': False,
    'Utay': 10., 'Uth': 1.0, 'Urest': 0.,
    'Itay': 10., 'Imax': 1.0
}

params_neuron_out = {
    'Ustart': 0.0,
    'Istart': 0.0,
    'Sstart': False,
    'Utay': 100., 'Uth': 1.0, 'Urest': 0.,
    'Itay': 100., 'Imax': 1.0
}

num_neuron_in = 100
num_neuron_out = 30

neuron_input =  LIFNeuron('input', num_neuron_in, params_neuron_in)
neuron_output =  LIFNeuron('output', num_neuron_out, params_neuron_out)

net.add_neurons([neuron_input, neuron_output])

# Добавляем связь между слоями
params_syn = {'Aplus': 0.01, 'Aminus': 0.01,
              'Tpre': 20, 'Tpost': 20}  
syn = SynapseSTDP('in_out', neuron_input, neuron_output, 
              weight=None, params = params_syn)


net.add_synapse(syn)

# Добавляем мониторы
monitor_U = MonitorPotential('U', [neuron_input, neuron_output])
monitor_Iout = MonitorCurrent('Iout', [neuron_input, neuron_output])
monitor_S = MonitorSpike('S', [neuron_input, neuron_output])

net.add_monitors([monitor_U, monitor_Iout, monitor_S])

# Задание внешних сигналов и расчет
t_steps = 10000
dt = 1

signal_in = np.array([[0.15 * (i + 1) for i in range(num_neuron_in)] for j in range(t_steps)])
bias_out = np.full((t_steps, num_neuron_out), 0.01)

model_input_current = {'input': signal_in,
                       'output': bias_out}

# pr = cProfile.Profile()
# pr.enable()
net.run(dt, model_input_current)
# pr.disable()

# stats = pstats.Stats(pr)
# stats.strip_dirs().sort_stats('cumulative').print_stats()