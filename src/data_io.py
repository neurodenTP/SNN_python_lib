import numpy as np
import serial
import queue
import threading
from scipy.signal import butter, filtfilt


def read_emg_from_file(filename, delimiter=None):
    """
    Считает данные из файла, возвращая numpy массив.
    """
    data = np.loadtxt(filename, delimiter=delimiter)
    return data

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