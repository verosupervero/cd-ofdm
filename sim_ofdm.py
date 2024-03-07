# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:32:54 2024

@author: Vero
"""
import numpy as np
import QAM16 as qam
import channels
import utils
import ofdm

def generar_simbolos_tx_blk_peine(N:int, pilot_period_comb:int, pilot_period_blk:int, NsymbOFDM:int=1000) -> (np.array, np.array, np.array,np.array):
    """
    Genera la cantidad NsymbQAM de simbolos QAM, y arma simbolos OFDM.
    Agrega pilotos en bloque y peine.

    Parameters
    ----------
    N : int
        Cantidad subportadoras (incluyendo los peines).
    pilot_period_comb : int
        Periodo de los simbolos peine (cada cuantas subportadoras hay un piloto).
    pilot_period_blk : int
        Periodo de los simbolos bloque (cada cuantos simbolos OFDM uno es un piloto).
    NsymbQAM : int, optional
        Cantidad de simbolos OFDM a generar (sin incluir pilotos blk). Default is 1000.
    
    Returns
    -------
    all_symb : np.array(N, NsymbOFDM+Npilotos_blk)
        Matriz cuyas filas tienen los simbolos QAM para cada subportadoras y
        las columnas son los simbolos OFDM, avanzan hacia la derecha.
    pilot_symbol_blk : np.array(N,1)
        Simbolo utilizado como piloto bloque
    pilot_symbol_comb : np.array(N,1)
        Simbolo usado como peine. En las frecuencias que hay datos contiene 0.
    data_bits : np.array(N,1)
        Bits de datos
    """
    cant_pilotos_comb = N // pilot_period_comb
    QAM_symb_len = (N-cant_pilotos_comb)*NsymbOFDM # cantidad de simbolos QAM a transmitir

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
    all_symb, pilot_symbol_blk = utils.add_block_pilots(all_symb_comb, amplitude=amp, period=pilot_period_blk)
    
    return all_symb, pilot_symbol_blk, pilot_symbol_comb,data_bits

def simular_canal_fading_variante(all_symb, SNRdB,Hinit, Hexp:float=1.0, HRWalk:float=0.01):
    """
    Genero un canal fading variante en el tiempo.
    Con Hexp se controla la variacion temporal: 1-> varia mucho, 0->constante
    HRWalk es la amplitud de un ruido aditivo al canal. 
    Parameters
    ----------
    all_symb : TYPE
        DESCRIPTION.
    SNRdB : TYPE
        DESCRIPTION.
    Hinit : TYPE
        DESCRIPTION.
    Hexp : float, optional
        Variacion temporal. Entre 0.0 y 1.0. No validado. The default is 1.0.
    HRWalk : float, optional
        Amplitud de ruido aditivo. The default is 0.01.

    Returns
    -------
    rx_symb : TYPE
        DESCRIPTION.
    noise_power : TYPE
        DESCRIPTION.

    """
    N = all_symb.shape[0]
    
    # Canal inicial
    H = Hinit #Canal en frecuencia
    # Prealoco matriz con simbolos recibidos
    rx_symb = np.zeros(all_symb.shape, dtype=all_symb.dtype)

    # Para cada simbolo, vario levemente el canal, y lo aplico
    # Luego lo convierto a tiempo y le sumo ruido
    for idx in range(0,all_symb.shape[1]):
        # Le vario la ganancia y fase en general, y le sumo ruido
        Hgain = (1.0-Hexp) + Hexp*channels.fadding_channel(1) # Comportamiento exponenecial
        Hnoise = HRWalk*channels.fadding_channel(N) # Random walk
        
        # Vario canal
        H = Hgain*H + Hnoise
        
        # Aplico el canal (en la freq) al simb. ofdm
        rx_ofdm = all_symb[:,idx]*H
        
        # Convierto coef. ofdm a tiempo y sumo ruido
        rx_symb[:,idx],noise_power = utils.add_noise(ofdm.mod(rx_ofdm),SNRdB)
    
    return rx_symb, noise_power

def sim_demodular(rx_symb, pilot_period_blk, pilot_period_comb,pilot_symbol_blk,pilot_symbol_comb, noise_power, use_comb=True, use_blk=True):
    """
    Demodula la señal temporal OFDM rx_simb, ecualiza, y entrega los bits
    y los simbolos qam.
    use_comb y use_blk controlan el ecualizador por peine y pilotos respectivamente.

    Parameters
    ----------
    rx_symb : TYPE
        DESCRIPTION.
    pilot_period_blk : TYPE
        DESCRIPTION.
    pilot_period_comb : TYPE
        DESCRIPTION.
    pilot_symbol_blk : TYPE
        DESCRIPTION.
    pilot_symbol_comb : TYPE
        DESCRIPTION.
    noise_power : TYPE
        DESCRIPTION.
    use_comb : TYPE, optional
        Habilitar ecualizador peine LS. The default is True.
    use_blk : TYPE, optional
        Habilitar ecualizador bloque MMSE. The default is True.

    Returns
    -------
    rx_bits : TYPE
        DESCRIPTION.
    rx_fix_symb : TYPE
        DESCRIPTION.

    """
    N = rx_symb.shape[0]
    #%% Recepcion BLK MMSE
    Nport = np.size(rx_symb, axis=0)
    N_rx = np.size(rx_symb, axis=1)
    
    # Calculo de cantidad de pilotos blk
    Npilots_rx_blk, idx_pilots_rx = utils.recover_Npilots_blk(N_rx, pilot_period_blk)
    Ndata_rx_blk = N_rx-Npilots_rx_blk
    
    # Calculo las posiciones del peine
    indices_peine = np.arange(0, N,pilot_period_comb) 

    # Demodulo el canal
    rx_freq = ofdm.demod(rx_symb)
    
    # Prealoco matriz para los simbolos ofdm recuperados, sin los pilotos blk
    rx_fix_symb = np.zeros((Nport,Ndata_rx_blk), dtype=rx_symb.dtype)
    
    # Itero sobre los simbolos (columnas)
    d_idx = 0 # Indice simbolos
    for idx in range(0,N_rx): #indice simbolos+pilotos blk
        # Si es un multiplo de pilot_period, es un piloto
        if idx in idx_pilots_rx:
            if use_blk:
                #Estimacion canal LS, y lo invierto
                Y = rx_freq[:,idx]
                # Armo una matriz H tal que Y = H X + n, donde X,Y son simbolos ofdm
                #H=np.diag(np.divide(Y, pilot_symbol))
                #W_mmse = H.T.conj() @ np.linalg.inv((H @ H.T.conj()) + N0 * np.eye(N))
                H=np.divide(Y, pilot_symbol_blk)
                # adaptado de https://www.sharetechnote.com/html/Communication_ChannelModel_MMSE.html
                W_mmse=np.diag(H.conj()/((abs(H)**2)+noise_power))
            else:
                # Si ignoro los blk, paso la identidad en el filtro
                W_mmse = np.eye(N)
            old_peine = pilot_symbol_comb
        else:
            ##MMSE - bloques
            rx_fix_symb[:,d_idx] = W_mmse @ rx_freq[:,idx] #MMSE
            
            # PEINE - Trackeo cuanto vario el H desde el simbolo anterior hasta ahora
            Hpeine_pilotos = np.divide(rx_fix_symb[indices_peine,d_idx].reshape(-1),
                                       old_peine[indices_peine].reshape(-1))
            Hpeine_interp = utils.interp_channel(Hpeine_pilotos, indices_peine, N)
            
            # Actualizo la estimacion del simbolo actual
            if use_comb:    
                rx_fix_symb[:,d_idx] = rx_fix_symb[:,d_idx] /Hpeine_interp #LS - peine
            
            # Para la proxima iteracion
            old_peine = rx_fix_symb[:,d_idx]
            
            d_idx = d_idx +1

    # Saco los pilotos
    symb_idx = np.isin(np.arange(N), indices_peine, invert=True)
    rx = rx_fix_symb[symb_idx,:]
    # obtengo bits
    rx_bits = qam.qam_to_bits(rx.reshape(-1))
    
    return rx_bits,rx_fix_symb
