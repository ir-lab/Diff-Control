import numpy as np
import math

# from torch.optim.lr_scheduler import LambdaLR


class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0.0, warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup

    def step(self, t):
        if self.__warmup and t < self.__warmup:
            # lr = self.__lr_max / self.__warmup * t
            lr = self.__lr_max * pow(t / self.__warmup, 4)

        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (
                1 + np.cos(t / T_max * np.pi)
            )
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr