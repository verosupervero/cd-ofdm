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
all_symb, pilot_symbol = utils.add_comb_pilots(data_par, qam.QAM(0), pilot_period,cant_pilotos)


pilot_amplitude = qam.QAM(0)

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
    # Vario levemente el canal
    # Le vario la ganancia y fase en general, y le sumo ruido
  #  Hgain = channels.fadding_channel(1)
   # Hnoise = channels.fadding_channel(N)
  #  H = (Hgain)*H + 0.01*Hnoise
    rx_ofdm = all_symb[:,idx]*H
    #ofdm_noise = math.sqrt(var_noise)*np.random.standard_normal(size=N)
    #rx_ofdm = all_symb[:,idx]*H + ofdm_noise
    rx_symb[:,idx],noise_power = utils.add_noise(ofdm.mod(rx_ofdm),SNRdB)


#%% Recepcion
Nport = np.size(rx_symb, axis=0) #son 128
N_rx = np.size(rx_symb, axis=1) # son 20
Ndata_rx = N_rx

# Prealoco matriz para los simbolos ofdm recuperados
rx_fix = np.zeros(rx_symb.shape, dtype=rx_symb.dtype)

##Demodulo los simbolos recibidos
rx_freq = ofdm.demod(rx_symb)


d_idx = 0      
indices_peine = np.arange(0, N,pilot_period) 
old_peine=pilot_symbol
for idx in range(0,N_rx):
    
        # PEINE - Trackeo cuanto vario el H desde el simbolo anterior hasta ahora
        Hpeine_pilotos = np.divide(rx_freq[indices_peine,d_idx].reshape(-1),
                                   pilot_symbol[indices_peine].reshape(-1))
        Hpeine_interp = utils.interp_channel(Hpeine_pilotos, indices_peine, N)
        
        # Actualizo la estimacion del simbolo actual
        # (COMENTAR ESTA LINEA PARA COMPARA SIN CORRECION X PEINE)
        rx_freq[:,d_idx] = rx_freq[:,d_idx] /Hpeine_interp #LS - peine
        
        # Para la proxima iteracion
        old_peine = rx_freq[:,d_idx]
        
        d_idx = d_idx +1  

# Saco los pilotos
symb_idx = np.isin(np.arange(N), indices_peine, invert=True)
rx_fix_symb = rx_fix[symb_idx,:]
# obtengo bits
rx_bits = qam.qam_to_bits(rx_fix_symb.reshape(-1))


# Calculo errores
Nerr = np.sum(rx_bits != data_bits)
Perr = Nerr / np.size(data_bits)

print(f"""BER: {Perr:.1E}""")

#%% graficos
qam.plot_qam_constellation(rx_fix_symb)