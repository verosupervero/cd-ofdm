# -*- coding: utf-8 -*-
"""
Transmite simbolos OFDM por un canal Fading variante en el tiempo
Usa pilotos en bloque y en frecuencia
@author: Vero
"""
import numpy as np
import QAM16 as qam
import channels
import utils
import ofdm
import sim_ofdm
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

np.random.seed(123)
N = 128 # numero de subportadoras
pilot_period_comb = 8 # un piloto cada esta cantidad de simbolos
pilot_period_blk = 20 #pilotos bloque
NsymbOFDM = 1000
SNRdB_vec = np.linspace(0,40,10) #dB

# Genero simbolos
all_symb, pilot_symbol_blk, pilot_symbol_comb,bits_tx = sim_ofdm.generar_simbolos_tx_blk_peine(N, pilot_period_comb, pilot_period_blk, NsymbOFDM)

# Canal inicial, igual para todos
Hinit = channels.fadding_channel(N) #Canal en frecuencia
ber={}
for snr in SNRdB_vec:
    np.random.seed(1111) # Para tener la misma dinamica de canal en todas
    # Paso la señal por mi canal
    # paso simbolos ofdm por canal, modulo ofdm->tiempo, agrego ruido, demodulo tiempo -> simbolos ofdm_rx
    rx_symb, noise_power = sim_ofdm.simular_canal_fading_variante(all_symb, snr, Hinit, Hexp=0, HRWalk=0.002)
    
    # Demodulo
    bits_rx, rx_fix_symb = sim_ofdm.sim_demodular(rx_symb, pilot_period_blk,
                                                  pilot_period_comb, 
                                                  pilot_symbol_blk,
                                                  pilot_symbol_comb,
                                                  noise_power,
                                                  use_comb=False,
                                                  use_blk=True)
   
    # Calculo errores
    Nerr = np.sum(bits_rx != bits_tx)
    Perr = Nerr / np.size(bits_tx)
    ber[snr] = Perr
    print(f"""SNR:{snr}, BER: {Perr:.1E}""")
    
#%% Creo un nuevo plot semilogarítmico
plt.figure(1)
plt.semilogy(list(ber.keys()), [ber[k] for k in ber.keys()])

# Ajusta las etiquetas y otros detalles si es necesario
# Guarda la figura como un archivo PNG
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
plt.savefig(f"""BER_SNR_blk_{timestamp}.png""")
# Muestra el gráfico
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.show()

#%% Ploteo la ultima constelacion
plt.figure(2)
qam.plot_qam_constellation(rx_fix_symb)
plt.title(f"""16-QAM SNR={snr}dB""")
plt.savefig(f"""Constelacion_SNR_blk_{timestamp}.png""")