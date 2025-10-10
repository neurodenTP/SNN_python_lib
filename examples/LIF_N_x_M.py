import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import matplotlib.pyplot as plt

from neuron import LIFNeuron
from synapse import Synapse
from network import Network
from monitor import MonitorPotential, MonitorCurrent, MonitorSpike, MonitorWeigts

net = Network()

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

num_neuron_in = 2
num_neuron_out = 3

neuron_input =  LIFNeuron('input', num_neuron_in, params_neuron_in)
neuron_output =  LIFNeuron('output', num_neuron_out, params_neuron_out)

net.add_neurons([neuron_input, neuron_output])

weights = np.array([[0.5*(i==j) for i in range(num_neuron_in)] 
                    for j in range(num_neuron_out)])
syn = Synapse('in_out', neuron_input, neuron_output, 
              weight=weights)


net.add_synapse(syn)

monitor_U = MonitorPotential('U', [neuron_input, neuron_output])
monitor_Iout = MonitorCurrent('Iout', [neuron_input, neuron_output])

net.add_monitors([monitor_U, monitor_Iout])

t_steps = 100
dt = 1

signal_in = np.array([[0.15 * (i + 1) for i in range(num_neuron_in)] for j in range(t_steps)])
bias_out = np.full((t_steps, num_neuron_out), 0.01)

model_input_current = {'input': signal_in,
                       'output': bias_out}

net.run(dt, model_input_current)

monitor_U.plot_line('input', dt)
plt.show()
monitor_U.plot_line('output', dt)
plt.show()

# monitor_Iout.plot_line('input', dt)
# plt.show()
# monitor_Iout.plot_line('output', dt)
# plt.show()

# print(net.neurons['output'].get_potential())