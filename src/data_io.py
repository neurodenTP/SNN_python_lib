import numpy as np
# import serial
from scipy.signal import butter, filtfilt


class DataImporter:
    def __init__(self, source: str, params: dict):
        self.source = source
        self.params = params.copy()

    def import_data(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError()


class EMGSignalStateImporterFromFile(DataImporter):
    def __init__(self, sourse: str, params: dict):
        super().__init__(sourse, params)
        self.time = None
        self.signal = None
        self.state = None
        self.dt = None
        
        self.signal_scaler = params['signal_scaler']
        self.time_scaler = params['time_scaler']
        
        self.lowcut = params['lowcut']
        self.highcut = params['highcut']
        
        self.start = params['start']
        self.stop = params['stop']
        self.step = params['step']

        
    def import_data(self):
        data_raw = np.loadtxt(self.source)[self.start:self.stop:self.step]    
        time = self.time_scaler * data_raw[:, 0]
        signal = self.signal_scaler * data_raw[:, 1]
        
        self.time = time
        self.signal =signal
        self.state = data_raw[:, 2]
        self.dt = (time[-1] - time[0]) / (len(time) - 1)
        
        
    def preprocess(self):
        signal = self.signal
        signal_filtred = bandpass_filter(signal, self.lowcut, self.highcut, 1./self.dt)
        signal_abs = abs(signal_filtred)
        self.signal = signal_abs


class EMGSignalStateImporterFromCOM(DataImporter):
    pass


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