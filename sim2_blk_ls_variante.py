# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:59:31 2024

@author: 
"""
#%% Inicializar sim
import numpy as np
import QAM16 as qam
import channels
import utils

sz = lambda x: (np.size(x,0), np.size(x,1))

N_subportadoras = 128 # numero de subportadoras
pilot_period = 20 # un piloto cada esta cantidad de simbolos
N_OFDM_sym = 1000
QAM_symb_len = N_subportadoras*N_OFDM_sym # cantidad de simbolos QAM a transmitir

SNRdB = 20 #dB

# Para siempre generar los mimsos numeros aleatorios y tener repetibilidad
np.random.seed(123)

#  Simbolos
Nbits = QAM_symb_len*qam.QAM_bits_per_symbol #cantidad de bits a generar
data_bits = np.random.randint(2,size=Nbits)

# Convierto la serie de bits a una serie de simbolos qam
data_qam = qam.bits_to_qam(data_bits)

# Conversion serie a paralelo
data_par = data_qam.reshape(N_subportadoras, -1)

# Agrego pilotos (en la frecuencia, todos unos)
pilot_amplitude = qam.QAM(0)

# Agrego pilotos (en la frecuencia, todos unos)
all_symb, pilot_symbol = utils.add_block_pilots(data_par, amplitude=pilot_amplitude, period=pilot_period)

#%% Convierto a ODFM
import ofdm
#tx_symb = ofdm.mod(all_symb)
#tx_pilot = ofdm.mod(all_symb[:,0])

#%% Canal
H = channels.fadding_channel(N_subportadoras) #Canal en frecuencia
# Prealoco matriz con simbolos recibidos
rx_symb = np.zeros(all_symb.shape, dtype=all_symb.dtype)

for idx in range(0,rx_symb.shape[1]):
    # Vario levemente el canal (canal variante en el tiempo AR-1)
    H = 0.998*H + 0.002*channels.fadding_channel(N_subportadoras)
    rx_ofdm = all_symb[:,idx]*H
    #ofdm_noise = math.sqrt(var_noise)*np.random.standard_normal(size=N)
    #rx_ofdm = all_symb[:,idx]*H + ofdm_noise
    rx_symb[:,idx],noise_power = utils.add_noise(ofdm.mod(rx_ofdm),SNRdB)

#%% Recepcion
Nport = np.size(rx_symb, axis=0)
N_rx = np.size(rx_symb, axis=1)
Npilots_rx, idx_pilots_rx = utils.recover_Npilots_blk(N_rx, pilot_period)
Ndata_rx = N_rx-Npilots_rx

# Prealoco filtro inverso y lo pongo plano
Hinv = np.zeros((Nport,1), dtype=rx_symb.dtype)
# Prealoco matriz para los simbolos ofdm recuperados
rx_fix_symb = np.zeros((Nport,Ndata_rx), dtype=rx_symb.dtype)

rx_freq = ofdm.demod(rx_symb)

d_idx = 0
for idx in range(0,N_rx):
    # Si es un multiplo de pilot_period, es un piloto
    if idx in idx_pilots_rx:
        #Estimacion canal LS, y lo invierto
        Hinv=np.divide(pilot_symbol, rx_freq[:,idx]) 
    else:
        rx_fix_symb[:,d_idx] = Hinv * rx_freq[:,idx]
        d_idx = d_idx +1

# obtengo bits
rx_bits = qam.qam_to_bits(rx_fix_symb.reshape(-1))

# Calculo errores
Nerr = np.sum(rx_bits != data_bits)
Perr = Nerr / np.size(data_bits)

print(f"""BER: {Perr:.1E}""")

#%% graficos
qam.plot_qam_constellation(rx_fix_symb)
