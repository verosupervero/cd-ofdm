# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:59:31 2024

@author: 
"""
#%% Inicializar sim
import numpy as np
import QAM16 as qam
import channels
import math
import utils

sz = lambda x: (np.size(x,0), np.size(x,1))

N = 128 # numero de subportadoras
pilot_period = 8 # un piloto cada esta cantidad de simbolos
cant_pilotos = N // pilot_period #lastima que tengo que poner esto aca y en la funcion

QAM_symb_len = (N-cant_pilotos)*20 # cantidad de simbolos QAM a transmitir
CP = N // 4 # prefijo ciclico
SNRdB = 30 #dB

# Para siempre generar los mimsos numeros aleatorios y tener repetibilidad
np.random.seed(123)


#  Simbolos
Nbits = QAM_symb_len*qam.QAM_bits_per_symbol
data_bits = np.random.randint(2,size=Nbits)

# Convierto la serie de bits a una serie de simbolos qam
data_qam = qam.bits_to_qam(data_bits)

# Conversion serie a paralelo
data_par = data_qam.reshape(N-cant_pilotos, -1)

# Agrego pilotos
all_symb, peine = utils.add_comb_pilots(data_par, qam.QAM(0), pilot_period,cant_pilotos)


pilot_amplitude = qam.QAM(0)
pilot_symbol=pilot_amplitude* np.ones(cant_pilotos, dtype=data_par.dtype)

#%% Convierto a ODFM
import ofdm
#tx_symb = ofdm.mod(all_symb)
#tx_pilot = ofdm.mod(all_symb[:,0])

#%% Canal
H = channels.fadding_channel(N) #Canal en frecuencia
# Prealoco matriz con simbolos recibidos
rx_symb = np.zeros(all_symb.shape, dtype=all_symb.dtype)

# Calculo de la varianza del ruido
# SNR = 10.log(P_S/P_N)
# 10^(SNR/10) = var(symb_QAM)/var(N)
# var(N) = var(symb_QAM)/10^(SNR/10)
#var_noise = qam.eps/pow(10, SNR/10)

for idx in range(0,rx_symb.shape[1]):
    # Vario levemente el canal (canal variante en el tiempo AR-1)
    #x = np.zeros(128)
   # x[0] = 1
    #H = np.fft.fft(x)
    H = 0.998*H + 0.002*channels.fadding_channel(N)
    rx_ofdm= utils.add_noise(all_symb[:,idx]*H,SNRdB)
    #ofdm_noise = math.sqrt(var_noise)*np.random.standard_normal(size=N)
    #rx_ofdm = all_symb[:,idx]*H + ofdm_noise
    rx_symb[:,idx] = ofdm.mod(rx_ofdm)


#%% Recepcion
Nport = np.size(rx_symb, axis=0) #son 128
N_rx = np.size(rx_symb, axis=1) # son 20
Ndata_rx = N_rx

# Prealoco filtro inverso y lo pongo plano
Hinv = np.zeros((Nport,1), dtype=rx_symb.dtype)
# Prealoco matriz para los simbolos ofdm recuperados
rx_fix_symb = np.zeros((Nport-cant_pilotos,Ndata_rx), dtype=rx_symb.dtype)

##Demodulo los simbolos recibidos
rx_freq = ofdm.demod(rx_symb)

for idx in range(0,N_rx):
    # Si es un multiplo de pilot_period, es un piloto
    indices = np.arange(0, N,pilot_period) 
    pilots_rx_data = rx_freq[np.isin(np.arange(0,N),indices),idx]
    Hinv= utils.estimate_channel(pilots_rx_data,pilot_symbol,N)
    all_carriers = Hinv * rx_freq[:,idx]
    rx_fix_symb[:,idx] = all_carriers[np.isin(np.arange(0,N),indices) == False]
    
# obtengo bits
rx_bits = qam.qam_to_bits(rx_fix_symb.reshape(-1))


# Calculo errores
Nerr = np.sum(rx_bits != data_bits)
Perr = Nerr / np.size(data_bits)

print(f"""Prob errrores: {Perr*100}%""")

#%% graficos
qam.plot_qam_constellation(rx_fix_symb)