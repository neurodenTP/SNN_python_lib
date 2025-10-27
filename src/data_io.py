import numpy as np
# import serial
from scipy.signal import butter, filtfilt


class DataImporter:
    def __init__(self, source: str, params: dict):
        self.source = source
        self.params = params.copy()
        self.time = None
        self.data = None

    def import_data(self):
        raise NotImplementedError

    def preprocess(self, data):
        raise NotImplementedError()


class DataImporterFromFile(DataImporter):
    def import_data(self):
        # Базовый импорт
        self.data = np.loadtxt(self.source)


class EMGStateImporterFromFile(DataImporterFromFile):
    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        self.state = None
        self.EMG_scaler = params['EMG_scaler']
        self.time_scaler = params['time_scaler']
        self.lowcut = params['lowcut']
        self.highcut = params['highcut']
        
        self.start = params['start']
        self.stop = params['stop']
        self.step = params['step']
        
        self.dt = None
        
    def import_data(self):
        data_raw = np.loadtxt(self.source)[self.start:self.stop:self.step]
        
        time = self.time_scaler * data_raw[:, 0]
        data = self.EMG_scaler * data_raw[:, 1]
        self.state = data_raw[:, 2]
        self.dt = (time[-1] - time[0]) / (len(time) - 1)
        
        data = self.preprocess(data)
        self.time = time
        self.data = data
    
    def preprocess(self, data):
        data_filtred = bandpass_filter(data, self.lowcut, self.highcut, 1./self.dt)
        data_abs = abs(data_filtred)
        return data_abs


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Создаёт коэффициенты фильтра полосового пропускания.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Применяет полосовой фильтр к входным данным.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=0)
    return y

def poisson_intervals_array(N, lambda_param, seed=None):
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