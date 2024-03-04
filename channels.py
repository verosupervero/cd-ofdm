# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:32:04 2024

@author: 
"""
import numpy as np

def fadding_channel(N=int)->np.array:
    """Recibe una cantidad de subportadoras y devuelve un fadding channel de ese tamanio
    """
    
    A=np.random.rayleigh(N)
    ph=np.random.uniform(0,2*np.pi,N)
    channel=A*np.exp(1j*ph)
    
    return channel