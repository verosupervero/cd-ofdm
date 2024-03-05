# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:24:26 2024

@author: Vero
"""
import numpy as np
import scipy

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

def add_comb_pilots(M: np.array, amplitude, period) -> np.array:
    """
    Agrega pilotos en peine

    Parameters
    ----------
    M : np.array
        DESCRIPTION.
    amplitude : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.

    Returns
    -------
    all_symb : TYPE
        DESCRIPTION.

    """
    N = M.shape[0] #Cantidad subportadoras
    
    N_pilots = (N // (period-1))+1
    
    # En esta matriz van los simbolos mas los pilotos
    all_symb = np.zeros((N+N_pilots,M.shape[1]),dtype=M.dtype)
    peine = np.zeros((N+N_pilots,1),dtype=M.dtype)
    data_idx=0
    for f_idx in range(0,N+N_pilots):
        if f_idx%period == 0:
            # Si es un multiplo de pilot_period, mando un piloto
            all_symb[f_idx,:] = amplitude
            peine[f_idx] = amplitude # para devolver una muestra del peine
        else:  
            # Si no, copio la fila
            all_symb[f_idx,:] = M[data_idx,:]
            data_idx = data_idx+1
    
    return all_symb, peine

def add_noise(signal, SNR_dB):
    # Calcular la potencia de la señal transmitida

    if SNR_dB == -1:
        return signal

    signal_power = np.mean(np.abs(signal) ** 2)

    # Calcular la potencia del ruido a partir de la relación señal a ruido (SNR)
    noise_power = signal_power / (10 ** (SNR_dB / 10))

    # Generar ruido gaussiano con la potencia calculada
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Agregar ruido a la señal transmitida
    noise_signal = signal + noise

    return noise_signal

def add_comb_pilots(M: np.array, amplitude,period, N_pilots) -> np.array:
    """
    Estima el H por LS y el simbolo recibido

    Parameters
    ----------
    M : np.array
        DESCRIPTION.
    amplitude : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.

    Returns
    -------
    all_symb : TYPE
        DESCRIPTION.

    """
    N = M.shape[0] #Cantidad subportadoras
    # En esta matriz van los simbolos mas los pilotos
    all_symb = np.zeros((N+N_pilots,M.shape[1]),dtype=M.dtype)
    peine = np.zeros((N+N_pilots,1),dtype=M.dtype)
    data_idx=0
    for f_idx in range(0,N+N_pilots):
        if f_idx%period == 0:
            # Si es un multiplo de pilot_period, mando un piloto
            all_symb[f_idx,:] = amplitude
            peine[f_idx] = amplitude # para devolver una muestra del peine
        else:  
            # Si no, copio la fila
            all_symb[f_idx,:] = M[data_idx,:]
            data_idx = data_idx+1
    
    return all_symb, peine

def estimate_channel(received_pilots, pilot_symbols, N_portadoras, interpolation_method = 'linear'):
  
    H_est=np.divide(received_pilots, pilot_symbols) 

    if interpolation_method == 'spline':
        f= scipy.interpolate.interp1d(np.arange(0,len(H_est)), H_est, kind='cubic')  
    else:
        f= scipy.interpolate.interp1d(np.arange(0,len(H_est)), H_est, kind=interpolation_method)

    # Interpola los valores de H para las distintas portadoras.
    interpolated_H = f(np.linspace(0, len(H_est)-1, N_portadoras))
    return interpolated_H
