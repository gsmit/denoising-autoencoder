import torch
import numpy as np


class BaseMixer:
    """Base mixer class."""

    def __init__(self, proba=0.15):
        self.proba = proba
        self.shape = None
        self.torch = None

    def set_noise(self, proba):
        self.proba = proba

    def fit(self, tensor):
        assert type(tensor) == torch.Tensor or type(tensor) == np.array

        if type(tensor) == torch.Tensor:
            self.torch = True
        else:
            self.torch = False

        self.shape = tensor.shape

    def mix(self, tensor):
        pass
