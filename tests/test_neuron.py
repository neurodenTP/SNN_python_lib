import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from neuron import LIFNeuron, LIFAdaptiveNeuron
import numpy as np
import matplotlib.pyplot as plt


class TestLIFNeuron(unittest.TestCase):
    def setUp(self):
        params = {
            'Ustart': 0.0,
            'Uth': 1.0,
            'Urest': 0.0,
            'Utay': 10.0,
            'Ioutstart': 0.0,
            'Iout_max': 1.0,
            'Itay': 10.0,
            'refractiontime': 5.0
        }
        self.neuron = LIFNeuron(params)
    
    def test_initial_values(self):
        self.assertEqual(self.neuron.U, 0.0)
        self.assertEqual(self.neuron.Iout, 0.0)

    def test_reset(self):
        self.neuron.U = 5
        self.neuron.Iout = 0.5
        self.neuron.reset()
        self.assertEqual(self.neuron.U, 0.0)
        self.assertEqual(self.neuron.Iout, 0.0)

    def test_step_no_spike(self):
        self.neuron.reset()
        self.neuron.step(1, 0.1)
        self.assertLess(self.neuron.U, self.neuron.params['Uth'])
        self.assertEqual(self.neuron.Iout, 0)

    def test_step_spike(self):
        self.neuron.reset()
        self.neuron.step(1, 5.0)
        self.assertEqual(self.neuron.Iout, self.neuron.params.get('Iout_max', 1.0))
        self.assertEqual(self.neuron.U, self.neuron.params.get('Urest', 0.0))

    def test_refractory(self):
        self.neuron.reset()
        self.neuron.step(1, 5.0)  # spike
        self.assertTrue(self.neuron.is_spike)
        self.neuron.step(1, 5.0)  # refractory, no spike
        self.assertFalse(self.neuron.is_spike)
        self.assertEqual(self.neuron.Iout, self.neuron.params.get('Iout_max', 1.0) * 
                         (1 - 1 / self.neuron.itay))  # decaying output current

    def test_long_low_input(self):
        """Test neuron behavior on long time with small constant input"""
        self.neuron.reset()
        dt = 1.0
        time_steps = 200
        input_current = 0.05  # small input current
        spikes = 0
        
        U_values = []
        Iout_values = []
        
        for _ in range(time_steps):
            self.neuron.step(dt, input_current)
            U_values.append(self.neuron.U)
            Iout_values.append(self.neuron.Iout)
            if self.neuron.is_spike:
                spikes += 1
        # Expect few or no spikes with small input current
        self.assertLessEqual(spikes, 5)
        # Membrane potential should be below threshold or near resting
        self.assertLess(self.neuron.U, self.neuron.params['Uth'])
        
        # Построение графика
        plt.plot(U_values, label='Membrane potential U')
        plt.plot(Iout_values, label='Output current')
        plt.xlabel('Time step')
        plt.title('Neuron response to low constant input')
        plt.legend()
        plt.show()

    def test_long_high_input(self):
        """Test neuron behavior on long time with large constant input"""
        self.neuron.reset()
        dt = 1.0
        time_steps = 200
        input_current = 5.0  # large input current
        spikes = 0
        
        U_values = []
        Iout_values = []
        
        for _ in range(time_steps):
            self.neuron.step(dt, input_current)
            U_values.append(self.neuron.U)
            Iout_values.append(self.neuron.Iout)
            if self.neuron.is_spike:
                spikes += 1
        # Expect multiple spikes with large input
        self.assertGreater(spikes, time_steps // (self.neuron.refraction_time + 1))
        # After multiple spikes membrane potential should be at resting potential
        self.assertEqual(self.neuron.U, self.neuron.params['Urest'])
        
        # Построение графика
        plt.plot(U_values, label='Membrane potential U')
        plt.plot(Iout_values, label='Output current')
        plt.xlabel('Time step')
        plt.title('Neuron response to high constant input')
        plt.legend()
        plt.show()


class TestLIFAdaptiveNeuron(unittest.TestCase):
    def setUp(self):
        params = {
            'Ustart': 0.0,
            'Vstart': 0.0,
            'Ioutstart': 0.0,
            'Utay': 10.0,
            'Uth': 1.0,
            'Vtay': 100.0,
            'Vstep': 0.1,
            'Itay': 10.0,
            'refractiontime': 2.0,
            'Iout_max': 1.0
        }
        self.neuron = LIFAdaptiveNeuron(params)

    def test_initial_values(self):
        self.assertEqual(self.neuron.U, 0.0)
        self.assertEqual(self.neuron.V, 0.0)
        self.assertEqual(self.neuron.Iout, 0.0)
        self.assertFalse(self.neuron.is_spike)
        self.assertEqual(self.neuron.tr, 0)

    def test_reset(self):
        self.neuron.U = 5
        self.neuron.V = 2
        self.neuron.Iout = 0.5
        self.neuron.is_spike = True
        self.neuron.tr = 3
        self.neuron.reset()
        self.assertEqual(self.neuron.U, 0.0)
        self.assertEqual(self.neuron.V, 0.0)
        self.assertEqual(self.neuron.Iout, 0.0)
        self.assertFalse(self.neuron.is_spike)
        self.assertEqual(self.neuron.tr, 0)

    def test_step_no_spike(self):
        self.neuron.reset()
        self.neuron.step(1, 0.1)
        self.assertLess(self.neuron.U, self.neuron.uth)
        self.assertEqual(self.neuron.Iout, 0)
        self.assertFalse(self.neuron.is_spike)

    def test_step_spike_and_adaptation(self):
        self.neuron.reset()
        dt = 1
        # Поднять потенциал до порога, чтобы вызвать спайк
        self.neuron.step(dt, 2)
        self.assertTrue(self.neuron.is_spike)
        self.assertEqual(self.neuron.Iout, self.neuron.iout_max)
        v_after_spike = self.neuron.V
        self.assertEqual(self.neuron.U, self.neuron.V)

        # На следующем шаге V должен уменьшится на Vstep и потенциал сброситься на V
        self.neuron.step(dt, 0)
        self.assertEqual(self.neuron.V, v_after_spike * (1 - dt / self.neuron.vtay))
        # После спайка потенциал U равен V


        # Проверка рефрактерного периода, потенциал не меняется и спайков нет
        prev_U = self.neuron.U
        self.neuron.step(1, 5.0)
        self.assertFalse(self.neuron.is_spike)
        self.assertEqual(self.neuron.U, prev_U)

    def test_long_term_adaptation(self):
        self.neuron.reset()
        dt = 1
        time_steps = 300
        input_current = 1.0
        spikes = 0
        
        U_values = []
        V_values = []
        Iout_values = []
        
        for _ in range(time_steps):
            self.neuron.step(dt, input_current)
            U_values.append(self.neuron.U)
            V_values.append(self.neuron.V)
            Iout_values.append(self.neuron.Iout)
            if self.neuron.is_spike:
                spikes += 1
        self.assertGreater(spikes, 1)
        # V должен снизиться за счет адаптации
        self.assertLess(self.neuron.V, 0.0)
        
        # Построение графика
        plt.plot(U_values, label='Membrane potential U')
        plt.plot(V_values, label='Rest potential V')
        plt.plot(Iout_values, label='Output current')
        plt.xlabel('Time step')
        plt.title('Neuron response to constant input')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
