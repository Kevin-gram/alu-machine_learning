#!/usr/bin/env python3
"""
Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    
    return np.diag(confusion) / np.sum(confusion, axis=1)

