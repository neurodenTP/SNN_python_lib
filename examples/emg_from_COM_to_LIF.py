import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(dirname, '../src')))

# # import numpy as np
# import matplotlib.pyplot as plt

# from neuron import LIFNeuron
# from network import Network
# from monitor import MonitorPotential, MonitorCurrent, MonitorSpike

# from data_io import EMGSignalStateImporterFromFile

# net = Network()
# net.add_neuron(neuron_input)


# streaming_monitor_U = StreamingMonitorPotential()
# net.add_monitors([streaming_monitor_U])
# # Стриминг монитор специальный потому что нужно следить за тем, чтобы он записывал только когда массив не используется
# # График тоже может строиться в этом мониторе, в общем эта история создает себе отдельный поток для работы

# data = StreamingData()
# # Эта штука создает себе отдельный поток и очередь в которую пишет данные

# net.streamingrun(data)
# Эта штука создает себе отдельный поток в котором берет данные из очереди и обсчитывает

# Вообще для самих данных входных может отдельную структуру тоже сделать?
# Вместо словаря массивов. Так будет понятнее.
# И в этой структуре можно уже метод - импорт делать, например. 

from neuron import Neuron
from network import Network
from data_io_new import InputConstantData

net = Network()
param = {'Ustart': 0,
         'Istart': 0,
         'Sstart': 0}
neuron_1 = Neuron('1', 2, param)
neuron_2 = Neuron('2', 3, param)
net.add_neurons([neuron_1, neuron_2])
data = InputConstantData(net)
data.generate_time_grid(0, 1 ,0.1)
data.generate_current_constant('1', 0)
data.generate_current_poisson_intervals('2', 0.2)

print(data.time)
print(data.current)
print(data.dt)
print(data.time_size)
