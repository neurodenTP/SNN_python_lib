import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(dirname, '../src')))

# import numpy as np
import matplotlib.pyplot as plt

from neuron import LIFNeuron
from network import Network
from monitor import MonitorPotential, MonitorCurrent, MonitorSpike

from data_io import EMGSignalStateImporterFromFile

net = Network()

# Добавляем слои нейронов
params_neuron_in = {
    'Ustart': 0.0,
    'Istart': 0.0,
    'Sstart': False,
    'Utay': 20., 'Uth': 1.0, 'Urest': 0.,
    'Itay': 100., 'Imax': 1.0
}

num_neuron_in = 1

neuron_input =  LIFNeuron('input', num_neuron_in, params_neuron_in)

net.add_neuron(neuron_input)

# Добавляем мониторы
monitor_U = MonitorPotential('U', neuron_input)
monitor_Iout = MonitorCurrent('Iout', neuron_input)
monitor_S = MonitorSpike('S', neuron_input)

net.add_monitors([monitor_U, monitor_Iout, monitor_S])

# Задание внешних сигналов и расчет
params_data = {
    'signal_scaler': 0.05,
    'time_scaler': 1000,
    'start': 0,
    'stop': 10000,
    'step': 5,
    'lowcut': 0.001,
    'highcut': 0.01
    }

data = EMGSignalStateImporterFromFile(dirname + "/data_time_emg_state.txt", params_data)
data.import_data()
data.preprocess()
plt.plot(data.time, data.signal)
plt.plot(data.time, data.state)
plt.show()

model_input_current = {'input': data.signal}

net.run(data.dt, model_input_current)

# Визуализация
monitor_U.plot_line('input', data.dt)
plt.show()

monitor_Iout.plot_line('input', data.dt)
plt.plot(data.time, data.state)
plt.show()

monitor_S.plot_scatter('input', data.dt)
plt.show()