"""
Name : optimization.py
Description : Contains various learning rate scheduler classes
Author : Blake Moody
Date : 10-25-2024
"""
from math import pi, cos

class CosineScheduler:
    """
    A class whose sole purpose is to determine the current learning rate of a model's optimizer at a given step in training

    Attributes
    ----------
    initial_lr : float
        The starting learning rate of the training scheduler (when number of steps are at their minimum)
    final_lr : float
        The final learning rate of the training scheduler (when the number of steps is equal or exceeds the maximum number of decay steps)
    max_steps : int
        The last step that the learning rate can continue to decrease
    """
    def __init__(self, initial_lr, final_lr, max_steps):
        """
        Parameters
        ----------
        initial_lr : float
        The starting learning rate of the training scheduler (when number of steps are at their minimum)
        final_lr : float
            The final learning rate of the training scheduler (when the number of steps is equal or exceeds the maximum number of decay steps)
        max_steps : int
            The last step that the learning rate can continue to decrease
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.max_steps = max_steps

    def __call__(self, step):
        """
        Returns the current learning rate given the current step

        Parameters
        ----------
        step : int
            The query step to derive the learning rate from
        """
        if step <= self.max_steps:
            return self.final_lr + (((self.initial_lr - self.final_lr) / 2) * (1 + cos(pi * step / self.max_steps)))

        return self.final_lr


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    num_epochs = 30
    scheduler = CosineScheduler(initial_lr=0.3, final_lr=0.01, max_steps=20)
    plt.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
    plt.show()