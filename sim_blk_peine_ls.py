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
import ofdm

sz = lambda x: (np.size(x,0), np.size(x,1))

N = 128 # numero de subportadoras
pilot_period_comb = 8 # un piloto cada esta cantidad de simbolos
cant_pilotos_comb = N // pilot_period_comb #lastima que tengo que poner esto aca y en la funcion
pilot_period_blk = 20 #pilotos bloque
QAM_symb_len = (N-cant_pilotos_comb)*200 # cantidad de simbolos QAM a transmitir
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
data_par = data_qam.reshape(N-cant_pilotos_comb, -1)

# Agrego pilotos peine
amp = qam.QAM(0)
all_symb_comb, pilot_symbol_comb = utils.add_comb_pilots(data_par, amp, pilot_period_comb,cant_pilotos_comb)

# Agrego pilotos bloque
all_symb, pilot_symbol = utils.add_block_pilots(all_symb_comb, amplitude=amp, period=pilot_period_blk)

#%% Convierto a ODFM
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
    Hgain = channels.fadding_channel(1)
    Hnoise = channels.fadding_channel(N)
    H = (Hgain)*H + 0.01*Hnoise
    rx_ofdm = all_symb[:,idx]*H
    #ofdm_noise = math.sqrt(var_noise)*np.random.standard_normal(size=N)
    #rx_ofdm = all_symb[:,idx]*H + ofdm_noise
    rx_symb[:,idx],noise_power = utils.add_noise(ofdm.mod(rx_ofdm),SNRdB)


#%% Recepcion BLK MMSE
Nport = np.size(rx_symb, axis=0)
N_rx = np.size(rx_symb, axis=1)
Npilots_rx_blk, idx_pilots_rx = utils.recover_Npilots_blk(N_rx, pilot_period_blk)
Ndata_rx_blk = N_rx-Npilots_rx_blk

# Prealoco filtro inverso y lo pongo plano
HMMSE = np.zeros((Nport,1), dtype=rx_symb.dtype)

rx_freq = ofdm.demod(rx_symb)

d_idx = 0

# Prealoco matriz para los simbolos ofdm recuperados, sin los pilotos blk
rx_fix_symb = np.zeros((Nport,Ndata_rx_blk), dtype=rx_symb.dtype)


# Prealoco matriz para los simbolos ofdm recuperados
#rx_fix = np.zeros((Nport,Ndata_rx), dtype=rx_symb.dtype)
indices_peine = np.arange(0, N,pilot_period_comb) 

#SNR_MFB = qam.eps / N0
for idx in range(0,N_rx):
    # Si es un multiplo de pilot_period, es un piloto
    if idx in idx_pilots_rx:
        #Estimacion canal LS, y lo invierto
        Y = rx_freq[:,idx]
        # Armo una matriz H tal que Y = H X + n, donde X,Y son simbolos ofdm
        #H=np.diag(np.divide(Y, pilot_symbol))
        #W_mmse = H.T.conj() @ np.linalg.inv((H @ H.T.conj()) + N0 * np.eye(N))
        H=np.divide(Y, pilot_symbol)
        # adaptado de https://www.sharetechnote.com/html/Communication_ChannelModel_MMSE.html
        W_mmse=np.diag(H.conj()/((abs(H)**2)+noise_power))
        old_peine = pilot_symbol_comb
    else:
        ##MMSE - bloques
        rx_fix_symb[:,d_idx] = W_mmse @ rx_freq[:,idx] #MMSE
        
        # PEINE - Trackeo cuanto vario el H desde el simbolo anterior hasta ahora
        Hpeine_pilotos = np.divide(rx_fix_symb[indices_peine,d_idx].reshape(-1),
                                   old_peine[indices_peine].reshape(-1))
        Hpeine_interp = utils.interp_channel(Hpeine_pilotos, indices_peine, N)
        
        # Actualizo la estimacion del simbolo actual
        # (COMENTAR ESTA LINEA PARA COMPARA SIN CORRECION X PEINE)
        rx_fix_symb[:,d_idx] = rx_fix_symb[:,d_idx] /Hpeine_interp #LS - peine
        
        # Para la proxima iteracion
        old_peine = rx_fix_symb[:,d_idx]
        
        d_idx = d_idx +1

# Saco los pilotos
symb_idx = np.isin(np.arange(N), indices_peine, invert=True)
rx = rx_fix_symb[symb_idx,:]
# obtengo bits
rx_bits = qam.qam_to_bits(rx.reshape(-1))


# Calculo errores
Nerr = np.sum(rx_bits != data_bits)
Perr = Nerr / np.size(data_bits)

print(f"""BER: {Perr:.1E}""")

#%% graficos
qam.plot_qam_constellation(rx_fix_symb)