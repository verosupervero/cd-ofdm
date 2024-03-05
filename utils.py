# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:24:26 2024

@author: Vero
"""
import numpy as np

def R(x:np.array, y:np.array=None) -> np.array:
    """
    Calcula la crosscorrelacion entre x e y
    Si se especifica un solo argumento, calcula
    la autocorrelacion de x

    Parameters
    ----------
    x : np.array(N)
        Primer vector
    y : np.array(N), optional
        DESCRIPTION. Segundo vector, si es None se toma x.

    Returns
    -------
    np.array(N,N)
        Matriz de correlacion o cross-correlacion
    """
    if y is None:
        y = x
    
    return x.reshape((-1,1)) * y.reshape((-1,1)).T.conj()

    