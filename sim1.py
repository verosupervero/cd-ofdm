# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:59:31 2024

@author: 
"""
#%% Inicializar sim
import numpy as np
import QAM16 as qam

N = 128 # numero de subportadoras
pilot_period = 8 # un piloto cada esta cantidad de simbolos
QAM_symb_len = N*20 # cantidad de simbolos QAM a transmitir
CP = N // 4 # prefijo ciclico

# Para siempre generar los mimsos numeros aleatorios y tener repetibilidad
np.random.seed(123)

#  Simbolos
Nbits = QAM_symb_len*qam.QAM_bits_per_symbol
data_bits = np.random.randint(2,size=Nbits)

# Convierto la serie de bits a una serie de simbolos qam
data_qam = qam.bits_to_qam(data_bits)

# Conversion serie a paralelo
data_par = data_qam.reshape(N, -1)

# Agrego pilotos
pilot_qam = qam.QAM(1) # elijo un simbolo qam como piloto
pilot_symbol = np.ones(N) * pilot_qam

N_ODFM_sym = np.size(data_par,axis=1)
N_pilots = N_ODFM_sym // pilot_period + 1

# En esta matriz van los simbolos mas los pilotos
all_symb = np.zeros((N,N_ODFM_sym+N_pilots),dtype=pilot_qam.dtype)

symb_idx=0
for data_idx in range(0,N_ODFM_sym):
    # Si es un multiplo de pilot_period, mando un piloto
    if symb_idx%pilot_period == 0:
        all_symb[:,symb_idx] = pilot_symbol
        symb_idx=symb_idx+1
    
    all_symb[:,symb_idx] = data_par[:,data_idx]
    symb_idx = symb_idx+1


# Armo 





