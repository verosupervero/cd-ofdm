# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 03:49:34 2024

@author: Vero
"""
import numpy as np
import def_blk_peine_ls
import matplotlib.pyplot as plt
from datetime import datetime
import QAM16 as qam

snr_vec = np.linspace(-40,50,10)
ber_blk_peine_variante=[]
symbs_blk_peine_variante=[]

ber_blk_variante=[]
symbs_blk_variante=[]

def simular(snr_vec, hab_peine, variante):
    ber =[]
    for snr in snr_vec:
        x, symb = def_blk_peine_ls.sim(snr,hab_peine, variante)
        ber.append(x)
    return ber, symb

op_var = [False, True]
fig=1
for variante in op_var:
    var_sin_peine_ber, var_sin_peine_symb = simular(snr_vec,hab_peine=False, variante=variante)
    var_con_peine_ber, var_con_peine_symb = simular(snr_vec,hab_peine=True, variante=variante)

    #%% Creo un nuevo plot semilogarítmico
    plt.figure(fig)
    fig = fig+1
    plt.semilogy(snr_vec, var_sin_peine_ber, label="Blk")
    plt.semilogy(snr_vec, var_con_peine_ber, 'r', label="Blk y Peine")
    plt.legend()
    # Ajusta las etiquetas y otros detalles si es necesario
    # Guarda la figura como un archivo PNG
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    plt.title(f"""Canal {"variante" if variante else "no variante"}""")
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.savefig(f"""img/var{variante}_{timestamp}.png""")
    plt.show()
    
    #%% Ploteo la ultima constelacion
    plt.figure(fig)
    fig = fig+1
    qam.plot_qam_constellation(var_con_peine_symb)
    plt.title(f"""16-QAM SNR={snr_vec[-1]}dB Con Peine""")
    plt.savefig(f"""img/var{variante}_peine_{timestamp}.png""")
    #%% Ploteo la ultima constelacion
    plt.figure(fig)
    fig = fig+1
    qam.plot_qam_constellation(var_sin_peine_symb)
    plt.title(f"""16-QAM SNR={snr_vec[-1]}dB Sin Peine""")
    plt.savefig(f"""img/var{variante}_blk_{timestamp}.png""")
        

    