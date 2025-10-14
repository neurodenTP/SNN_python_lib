import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import matplotlib.pyplot as plt
import time

from neuron import LIFNeuron
from synapse import SynapseSTDP, SynapseLTPf
from network import Network
from monitor import MonitorSpike, MonitorWeigts
from data_io import poisson_intervals_array

net = Network()

# Добавляем слои нейронов
params_neuron_in = {
    'Ustart': 0.0,
    'Istart': 0.0,
    'Sstart': False,
    'Utay': 10., 'Uth': 1.0, 'Urest': 0.,
    'Itay': 10., 'Imax': 0.0
}

params_neuron_out = params_neuron_in

num_neuron_in = 10
num_neuron_out = 1

neuron_input =  LIFNeuron('input', num_neuron_in, params_neuron_in)
neuron_output =  LIFNeuron('output', num_neuron_out, params_neuron_out)

net.add_neurons([neuron_input, neuron_output])

monitor_S = MonitorSpike('S', [neuron_input, neuron_output])
net.add_monitors([monitor_S])

# Задание внешних сигналов
t_steps = 10000
dt = 1.0

freq_max_in = 100. / 1000
freq_min_in = 1. / 1000
freqs_in = np.linspace(freq_min_in, freq_max_in, num_neuron_in)
spike_itervals_in = 1. / freqs_in / dt

freq_out = 50. / 1000
spike_itervals_out = 1. / dt / freq_out

signal_in = np.array([poisson_intervals_array(t_steps, i) for i in spike_itervals_in]).T
signal_out = np.array([poisson_intervals_array(t_steps, spike_itervals_out )]).T

model_input_current = {'input': signal_in,
                       'output': signal_out}

# Визуализация спайков
net.run(dt, model_input_current)

monitor_S.plot_scatter('input', dt)
plt.show()
monitor_S.plot_scatter('output', dt)
plt.show()


# Добавляем связь между слоями STDP
weights_STDP = np.full((num_neuron_out, num_neuron_in), 0.5)

params_syn_STDP = {'Aplus': 0.01, 'Aminus': 0.01,
              'Tpre': 20, 'Tpost': 20}
syn_STDP = SynapseSTDP('in_out_STDP', neuron_input, neuron_output, 
                  weight=weights_STDP, params=params_syn_STDP)

net.add_synapse(syn_STDP)

monitor_W_STDP = MonitorWeigts('W_STDP', [syn_STDP])
net.add_monitors([monitor_W_STDP])


t0 = time.time()
net.run(dt, model_input_current)
print("running time STDP = ", time.time() - t0)

monitor_W_STDP.plot_line('in_out_STDP', dt)


# Удаляем STDP
net.remove_synapses(['in_out_STDP'])
net.remove_monitors(['W_STDP'])


# Добавляем связь между слоями LTPf
weights_LTPf = np.full((num_neuron_out, num_neuron_in), 0.5)

params_syn_LTPf = {'Aplus': 0.01, 'Aforgetting': 0.005,
                   'Tpre': 20}
syn_LTPf = SynapseLTPf('in_out_LTPf', neuron_input, neuron_output, 
                       weight=weights_LTPf, params=params_syn_LTPf)

net.add_synapse(syn_LTPf)

monitor_W_LTPf = MonitorWeigts('W_LTPf', [syn_LTPf])
net.add_monitors([monitor_W_LTPf])


t0 = time.time()
net.run(dt, model_input_current)
print("running time LTPf = ", time.time() - t0)

monitor_W_LTPf.plot_line('in_out_LTPf', dt)
