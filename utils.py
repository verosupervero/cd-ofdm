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


def add_block_pilots(M: np.array, amplitude, period) -> (np.array, np.array):
    """
    Agrega pilotos en bloque.

    Parameters
    ----------
    M : np.array
        Matriz de simbolos. Cada fila es una subportadora,y las columnas avanzan en el tiempo
    amplitude : TYPE
        Amplitud del piloto
    period : TYPE
        Periodo de piloto.

    Returns
    -------
    all_symb : TYPE
        Matriz con pilotos en bloque agregados.
        
    pilot_symbol : TYPE
        Matriz con pilotos en bloque agregados.
    """
    N = M.shape[0] #Cantidad subportadoras
    N_ODFM_sym = M.shape[1] #Cantidad simbolos
    
    # Agrego pilotos (en la frecuencia, todos unos)
    pilot_symbol = amplitude*np.ones(N, dtype=M.dtype)
    N_pilots = (N_ODFM_sym // (period-1))+1
    
    # En esta matriz van los simbolos mas los pilotos
    all_symb = np.zeros((N,N_ODFM_sym+N_pilots),dtype=M.dtype)
    
    data_idx=0
    for symb_idx in range(0,all_symb.shape[1]):
        # Si es un multiplo de pilot_period, mando un piloto
        if symb_idx%period == 0:
            all_symb[:,symb_idx] = pilot_symbol
        else:  
            all_symb[:,symb_idx] = M[:,data_idx]
            data_idx = data_idx+1
    
    return all_symb, pilot_symbol