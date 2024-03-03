# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:42:36 2024

@author: C.FDANKO
"""
import numpy as np

def mod(par_data: np.array) -> np.array:
    """
    Convierte los simbolos paralelizados una señal OFDM (dominio del tiempo)
    Cada columna tiene tantos simbolos como subportadoras se deseen

    Parameters
    ----------
    par_data : np.array
        Matriz, cada columna sera un simbolo OFDM

    Returns
    -------
    np.array
        Matriz de N_subportadoras x M_simbolos OFDM

    """
    return np.fft.ifft(par_data, axis=0)

def demod(OFDM_data: np.array) -> np.array:
    """
    Convierte la señal OFDM a simbolos paralelizados
    Cada columna posee señales OFDM
    A los cuales se demodulan las subportadoras (tiempo->freq)

    Parameters
    ----------
    OFDM_data : np.array
        Cada columna es un simbolo OFDM

    Returns
    -------
    np.array
        Matriz de simbolos paralelizados

    """
    return np.fft.fft(OFDM_data, axis=0)

def test_mod():
    m = np.ones((10,2))
    res = np.zeros((10,2))
    res[0,0]=1
    res[0,1]=1
    assert np.all(mod(m) == res)

def addCP(OFDM_time, CP):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def removeCP(signal, CP):
    return signal[CP:]


class TXRX():
    """
    Clase para simular una transmision
    """
    
    
    
    def __init__(self):
        return 3
    