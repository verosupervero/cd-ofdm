# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:54:47 2024

@author: 
"""

def QAM_demod(symbols):
    demapping_table = {v : k for k, v in mapping_table.items()}
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])

    # calculate distance of each RX point to each possible point
    dists = abs(symbols.reshape((-1,1)) - constellation.reshape((1,-1)))

    # for each element in QAM, choose the index in constellation
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)

    # get back the real constellation point
    hardDecision = constellation[const_index]

    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision])#, hardDecision

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time, CP_len):
    cp = OFDM_time[-CP_len:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def removeCP(signal, CP_len):
    return signal[CP_len:]

# Función para simular la transmisión por el canal
def simulate_channel_transmission(data_sequence, channel_response, SNR_dB=-1):
    # Aplicar la respuesta al impulso del canal mediante la convolución
    transmitted_signal = np.convolve(data_sequence, channel_response, mode='same')

    if SNR_dB == -1:
      return transmitted_signal

    # Calcular la potencia de la señal transmitida
    signal_power = np.mean(np.abs(transmitted_signal) ** 2)

    # Calcular la potencia del ruido a partir de la relación señal a ruido (SNR)
    noise_power = signal_power / (10 ** (SNR_dB / 10))

    # Generar ruido gaussiano con la potencia calculada
    noise = np.random.normal(0, np.sqrt(noise_power), len(transmitted_signal))

    # Agregar ruido a la señal transmitida
    received_signal = transmitted_signal + noise

    return received_signal