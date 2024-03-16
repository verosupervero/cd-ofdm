# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 03:49:34 2024

@author: Vero
"""
import numpy as np
import def_blk_peine_ls_copia
import matplotlib.pyplot as plt
from datetime import datetime
import QAM16 as qam

snr_vec = np.linspace(10,40,5)
ber_blk_peine_variante=[]
symbs_blk_peine_variante=[]

default_per_peine=8
default_per_bloque=20

ber_blk_variante=[]
symbs_blk_variante=[]

def simular_variacion_temporal(snr_vec, hab_peine, variante):
    ber =[]
    for snr in snr_vec:
        x, symb = def_blk_peine_ls_copia.sim(snr,default_per_peine,default_per_bloque,hab_peine, variante)
        ber.append(x)
    return ber, symb


def simular_variacion_bloques_peines(snr_vec, cant_peines,cant_bloques, hab_peine, variante):
    ber_snrs =[]
    for snr in snr_vec:
        print("Para",snr, "dB")
        ber_matrix = np.zeros((len(cant_peines), len(cant_bloques)))
        for i, a in enumerate(cant_peines):
            #print("Calculando para", snr, "con", a, "peines")
            for j, b in enumerate(cant_bloques):
                Perr, rx_fix_symb = def_blk_peine_ls_copia.sim(snr,a,b,True,True)
                ber_matrix[i, j] =  Perr
                #print("Calculando para", snr, "con", a, "peines y",b, "bloques","BER:",Perr)
            #print("Resultados para los distintas cantidades de bloques")
            #print(ber_matrix[i,:])
            
        print("Resultados para los distintas cantidades de peines y bloques con",snr,"dB")
        resultados=np.zeros((len(cant_peines)+1, len(cant_bloques)+1))
        resultados[0,0]=snr
        resultados[1:,0]=cant_peines
        resultados[0,1:]=cant_bloques
        resultados[1:, 1:] = ber_matrix   
        print(resultados)
        ber_snrs.append(ber_matrix)    
    return ber_snrs

op_var = [False, True]
fig=1
# for variante in op_var:
#     var_sin_peine_ber, var_sin_peine_symb = simular_variacion_temporal(snr_vec,hab_peine=False, variante=variante)
#     var_con_peine_ber, var_con_peine_symb = simular_variacion_temporal(snr_vec,hab_peine=True, variante=variante)

cant_peines=[2,8,16,32]
cant_bloques=[2,20,200]

ber_snrs=simular_variacion_bloques_peines(snr_vec, cant_peines,cant_bloques, hab_peine=True, variante=True)



    # #%% Creo un nuevo plot semilogarítmico
    # plt.figure(fig)
    # fig = fig+1
    # plt.semilogy(snr_vec, var_sin_peine_ber, label="Blk")
    # plt.semilogy(snr_vec, var_con_peine_ber, 'r', label="Blk y Peine")
    # plt.legend()
    # # Ajusta las etiquetas y otros detalles si es necesario
    # # Guarda la figura como un archivo PNG
    # now = datetime.now()
    # timestamp = now.strftime("%Y%m%d_%H%M%S")
    # plt.title(f"""Canal {"variante" if variante else "no variante"}""")
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('BER')
    # plt.savefig(f"""img/var{variante}_{timestamp}.png""")
    # plt.show()
    
    # #%% Ploteo la ultima constelacion
    # plt.figure(fig)
    # fig = fig+1
    # qam.plot_qam_constellation(var_con_peine_symb)
    # plt.title(f"""16-QAM SNR={snr_vec[-1]}dB Con Peine""")
    # plt.savefig(f"""img/var{variante}_peine_{timestamp}.png""")
    # #%% Ploteo la ultima constelacion
    # plt.figure(fig)
    # fig = fig+1
    # qam.plot_qam_constellation(var_sin_peine_symb)
    # plt.title(f"""16-QAM SNR={snr_vec[-1]}dB Sin Peine""")
    # plt.savefig(f"""img/var{variante}_blk_{timestamp}.png""")
        

    