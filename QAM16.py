# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:38:43 2024

@author: fdanko
"""

import numpy as np
# Mapeo 16-QAM

QAM_bits_per_symbol = 4

QAM16 = np.array([# codigo de grey
         -3-3j, -3-1j, -3+3j, -3+1j,
         -1-3j, -1-1j, -1+3j, -1+1j, 
         3-3j,  3-1j,  3+3j,  3+1j,
         1-3j,  1-1j,  1+3j,  1+1j])

def QAM(bits: int) -> complex:
    """
    Convierte un numero de 4 bits en un simbolo 16-QAM.
    
    Parameters
    ----------
    bits : int
        Bits, en forma de entero (ej: 0100 -> 4).

    Returns
    -------
    complex
        Simbolo.

    """
    return QAM16[bits]

def unQAM(symb: complex) -> int:
    """
    Convierte un un simbolo 16-QAM en un numero de 4 bits.
    
    Parameters
    ----------
    symb : complex
        Simbolo 16-QAM.

    Returns
    -------
    int
        Bits, en forma de entero (ej: 0100 -> 4).

    """
    # A todos los puntos de la constelacion le resto mi simbolo
    # Tomo el valor absoluto para obtener la distancia de cada
    # simbolo de la constelacion a mi simbolo, y obtengo el indice
    # del simbolo mas cercano
    return np.argmin(abs(QAM16 - symb)) 
    

def test_qam():
    # Pruebo que pueda recuperar todos los simbolos
    # correr con pytest
    noise = 0.1-0.3j
    for n in range(0,16):
        assert unQAM(QAM(n)+noise) == n

def bits_to_qam(data_bits:np.array) -> np.array:
    """
    Parameters
    ----------
    data_bits : Array de bits .
    QAM_bits_per_symbol: bps QAM (log2(M))

    Returns
    -------
    Array de simbolos qam.

    """
    # a mi vector de bits lo reshapeo a una matriz de nx4, y lo doy vuelta
    # por como toma los bits packbits
    data_bits_grouped = np.fliplr(data_bits.reshape(-1, QAM_bits_per_symbol))
    # y la codifico en QAM. bitorder little llena con ceros adelante
    a = np.packbits(data_bits_grouped, axis=1, bitorder='little') 
    return np.apply_along_axis(QAM,0, a)

def qam_to_bits(qam_arr: np.array) -> np.array:
    """
    Parameters
    ----------
    qam_arr : np.array
        Array de simbolos 16QAM.

    Returns
    -------
    Array de bits.

    """
    # Convierto de qam a enteros representando bits
    # y convierto los enteros a array de bits
    return np.array([(np.unpackbits(
        np.array([unQAM(s)], dtype=np.uint8),
        bitorder='little')[0:4])[::-1] for s in qam_arr]).reshape(-1)

def test_qam_bits():
    np.random.seed(123)
    Nqam = 1000
    bits = np.random.randint(2,size=4*Nqam, dtype=np.uint8)
    assert np.all(bits == qam_to_bits(bits_to_qam(bits)))