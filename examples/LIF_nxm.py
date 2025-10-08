import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import matplotlib.pyplot as plt

from neuron import LIFNeuron
from layer import Layer
from connection import Connection
from network import Network
from monitor import PotentialMonitor, CurrentMonitor, SpikeMonitor
from monitor import MonitorConnection

net = Network()

params_neuron_in = {
    'Ustart': 0.0,
    'Ioutstart': 0.0,
    'Utay': 10.0,
    'Uth': 1.0,
    'Urest': 0.0,
    'Itay': 10.0,
    'refractiontime': 5.0,
    'Iout_max': 1.0
}

params_neuron_out = {
    'Ustart': 0.0,
    'Ioutstart': 0.0,
    'Utay': 100.0,
    'Uth': 1.0,
    'Urest': 0.0,
    'Itay': 10.0,
    'refractiontime': 5.0,
    'Iout_max': 1.0
}

num_neuron_in = 2
num_neuron_out = 3

layer_input =  Layer('input', num_neuron_in, LIFNeuron, params_neuron_in)
layer_output =  Layer('output', num_neuron_out, LIFNeuron, params_neuron_out)

net.add_layers([layer_input, layer_output])

weights = np.array([[0.5*(i==j) for i in range(num_neuron_in)] 
                    for j in range(num_neuron_out)])
conn = Connection('in_out', layer_input, layer_output, 
                  weight_matrix=weights)


net.add_connection(conn)

monitor_U = PotentialMonitor('U', [layer_input, layer_output])
monitor_Iout = CurrentMonitor('Iout', [layer_input, layer_output])

net.add_monitors([monitor_U, monitor_Iout])

N = 100
dt = 1

signal_in = np.array([[0.15 * (i + 1) for i in range(num_neuron_in)] for j in range(N)])
bias_out = np.full((N, num_neuron_out), 0.01)

model_input_current = {'input': signal_in,
                       'output': bias_out}

net.run(dt, model_input_current)

monitor_U.plot_line('input', dt)
plt.show()
monitor_U.plot_line('output', dt)
plt.show()

monitor_Iout.plot_line('input', dt)
plt.show()
monitor_Iout.plot_line('output', dt)
plt.show()
