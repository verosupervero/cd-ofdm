# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:24:42 2024

@author: Vero
"""
import numpy as np
import QAM16 as qam
import channels
import math
import utils
import ofdm



def mod_comb(N:int, pilot_period:int, N_symb_ofdm:int) -> (np.array, np.array, np.complex128):
    """
    Crea N_symb_ofdm simbolos, con N subportadoras, agregando un piloto cada
    plot_period subportadoras.

    Parameters
    ----------
    N : int
        Cantidad de subportadoras
    pilot_period : int
        Cada esta cantidad de subportadoras va un piloto, equiespaciados.
    N_symb_ofdm: int
        Cantidad de simbolos ofdm (columnas) a generar

    Returns
    -------
    all_symb : np.array(N,N_symb_ofdm)
        Matriz de simbolos ofdm. Cada fila es una subportadora, y avanzan en
        el tiempo con las columnas hacia la derecha
    
    pilot_symbol : np.array(N,1)
        Piloto utilizado. Cero en las subportadoras que no hay piloto.
        
    pilot_amplitude : complex
        Valor de cada elemento del piloto
    """
    cant_pilotos = N // pilot_period #lastima que tengo que poner esto aca y en la funcion
    QAM_symb_len = (N-cant_pilotos)*N_symb_ofdm # cantidad de simbolos QAM a transmitir
    
    #  Simbolos
    Nbits = QAM_symb_len*qam.QAM_bits_per_symbol
    data_bits = np.random.randint(2,size=Nbits)
    
    # Convierto la serie de bits a una serie de simbolos qam
    data_qam = qam.bits_to_qam(data_bits)
    
    # Conversion serie a paralelo
    data_par = data_qam.reshape(N-cant_pilotos, -1)
    
    # Agrego pilotos
    all_symb, pilot_symbol = utils.add_comb_pilots(data_par, qam.QAM(0), pilot_period,cant_pilotos)
    
    pilot_amplitude = qam.QAM(0)
    
    return all_symb, pilot_symbol, pilot_amplitude, data_bits


def rx_comb(rx_freq: np.array, pilot_period:int):
    N = rx_freq.shape[0]
    N_rx = rx_freq.shape[1]

    # Prealoco matriz para los simbolos ofdm recuperados
    rx_fix = np.zeros(rx_freq.shape, dtype=rx_freq.dtype)

    indices = np.arange(0, N,pilot_period) 
    for idx in range(0,N_rx):
        H_interp = utils.interp_channel(rx_freq[indices,idx], indices, N)
        rx_fix[:,idx] = np.divide(rx_freq[:,idx], H_interp)

    # Saco los pilotos
    symb_idx = np.isin(np.arange(N), indices, invert=True)
    rx_fix_symb = rx_fix[symb_idx,:].reshape(-1)
    
    return rx_fix_symb


# obtengo bits
#rx_bits = qam.qam_to_bits(rx_fix_symb.reshape(-1))
