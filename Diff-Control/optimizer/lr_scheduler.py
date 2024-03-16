from __future__ import absolute_import
from __future__ import print_function

from .polynomial_decay import PolynomialDecay
from .cosine_decay_scheduler import CosineDecayLR
import torch
import numpy as np

AVAI_SCH = ['polynomial_decay', "single_step", 'cosine_decay', 'multisteplr']


def build_lr_scheduler(optimizer, lr_scheduler='polynomial_decay', base_lr=0.0001, max_decay_steps=1000,
                       end_learning_rate=0.0001, power=0.9):
    if lr_scheduler not in AVAI_SCH:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))

    if lr_scheduler == 'polynomial_decay':
        end_learning_rate = end_learning_rate if end_learning_rate != -1 else 0.1 * base_lr
        # print(end_learning_rate, max_decay_steps)

        scheduler = PolynomialDecay(optimizer, max_decay_steps=max_decay_steps, end_learning_rate=end_learning_rate,
                                    power=power)
    if  lr_scheduler == 'cosine_decay':
        scheduler = CosineDecayLR(optimizer, T_max=max_decay_steps, lr_init=0.1, lr_min=0.001, warmup=1)
    
    if lr_scheduler == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, np.arange(0 + 1, 200, 50), gamma=0.1
        )
    return scheduler
