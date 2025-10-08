import numpy as np

#TODO Переписать правила

class STDP:
    def __init__(self, params):
        self.A_plus = params.get('A_plus', 0.01)
        self.A_minus = params.get('A_minus', 0.012)
        self.tau_plus = params.get('tau_plus', 20.0)
        self.tau_minus = params.get('tau_minus', 20.0)
        self.y_plus = None
        self.y_minus = None
        

    def rule(self, weights, pre_spikes, post_spikes, dt):
        weights_delta = np.zeros_like(weights)
        pre_spikes = pre_outputs_func()
        post_spikes = post_outputs_func()

        y_plus *= (1 - dt/self.tau_plus)
        y_minus *= (1 - dt/self.tau_minus)

        for i_post, post_val in enumerate(post_spikes):
            for j_pre, pre_val in enumerate(pre_spikes):
                if pre_val and post_val:
                    # Здесь предполагается, что pre_val и post_val - булевы или бинарные значения спайка
                    # Для упрощения: время не учитывается, просто коррекция при совпадении спайка
                    weights_delta[i_post, j_pre] += self.A_plus - self.A_minus

        weights += weights_delta
        np.clip(weights, 0, 1, out=weights)
        return weights

class LTPf