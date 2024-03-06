# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:24:26 2024

@author: Vero
"""
import numpy as np
import scipy

def R(x:np.array, y:np.array=None) -> np.array:
    """
    Calcula la crosscorrelacion entre x e y
    Si se especifica un solo argumento, calcula
    la autocorrelacion de x

    Parameters
    ----------
    x : np.array(N)
        Primer vector
    y : np.array(N), optional
        DESCRIPTION. Segundo vector, si es None se toma x.

    Returns
    -------
    np.array(N,N)
        Matriz de correlacion o cross-correlacion
    """
    if y is None:
        y = x
    
    return x.reshape((-1,1)) * y.reshape((-1,1)).T.conj()

def calc_Npilots_blk(Nsymb:int, period:int) -> (int, np.array):
    """
    Devuelve la cantidad de pilotos requeridos para N simbolos

    Parameters
    ----------
    N : int
        Cantidad de simbolos (sin pilotos)
    period : int
        Periodo de los pilotos

    Returns
    -------
    Npilots : TYPE
        Cantidad de pilotos requeridos
    idx : TYPE
        Array de indices de pilotos

    """
    
    Npilots = (Nsymb // (period-1))
    if (Nsymb%(period-1))!=0:
        # En este caso, no pongo un piloto al final si no hay mas datos
        Npilots = Npilots+1
    
    idx = np.arange(0,Nsymb+Npilots,period)    
    return Npilots, idx


def recover_Npilots_blk(total_len:int, period:int) -> (int, np.array):
    """
    Devuelve la cantidad de pilotos si hay total_len elementos

    Parameters
    ----------
    total_len : int
        Cantidad de elementos totales (data+pilotos).
    period : int
        Periodo de los pilotos

    Returns
    -------
    Npilots : TYPE
        Cantidad de pilotos
    idx : TYPE
        Array de indices de pilotos

    """
    
    Npilots = ((total_len-1) // period)+1
    idx = np.arange(0,total_len,period)    
    return Npilots, idx

def test_calc_Npilots_blk():
    test_cases = [(10,5,3),(9,5,3),(8,5,2),(2,5,1),(1,5,1),
                  (8,4,3),(6,4,2),(5,4,2),(4,4,2),(3,4,1),(1,4,1)]
    for tc in test_cases:
        res, idx = calc_Npilots_blk(tc[0],tc[1])
        print(tc, res)
        assert res == tc[2]

def test_recover_Npilots_blk():
    test_cases = [(10,5,3),(9,5,3),(8,5,2),(2,5,1),(1,5,1),
                  (8,4,3),(6,4,2),(5,4,2),(4,4,2),(3,4,1),(1,4,1)]
    for tc in test_cases:
        res, idx = recover_Npilots_blk(tc[0]+tc[2],tc[1])
        print(tc, res)
        assert res == tc[2]

def add_block_pilots(M: np.array, amplitude, period) -> (np.array, np.array):
    """
    Agrega pilotos en bloque.

    Parameters
    ----------
    M : np.array
        Matriz de simbolos. Cada fila es una subportadora,y las columnas avanzan en el tiempo
    amplitude : TYPE
        Amplitud del piloto
    period : TYPE
        Periodo de piloto.

    Returns
    -------
    all_symb : TYPE
        Matriz con pilotos en bloque agregados.
        
    pilot_symbol : TYPE
        Matriz con pilotos en bloque agregados.
    """
    N = M.shape[0] #Cantidad subportadoras
    N_ODFM_sym = M.shape[1] #Cantidad simbolos
    
    
    # Agrego pilotos (en la frecuencia, todos unos)
    pilot_symbol = amplitude*np.ones(N, dtype=M.dtype)
    N_pilots,index_pilots = calc_Npilots_blk(N_ODFM_sym,period)
    
    
    # En esta matriz van los simbolos mas los pilotos
    all_symb = np.zeros((N,N_ODFM_sym+N_pilots),dtype=M.dtype)
    
    data_idx=0
    for symb_idx in range(0,all_symb.shape[1]):
        # Si es un multiplo de pilot_period, mando un piloto
        if symb_idx%period == 0:
            all_symb[:,symb_idx] = pilot_symbol
        else:  
            all_symb[:,symb_idx] = M[:,data_idx]
            data_idx = data_idx+1
    
    return all_symb, pilot_symbol


    


def add_noise(signal, SNR_dB):
    # Calcular la potencia de la señal transmitida

    if SNR_dB == -1:
        return signal

    signal_power = np.mean(np.abs(signal) ** 2)

    # Calcular la potencia del ruido a partir de la relación señal a ruido (SNR)
    noise_power = signal_power / (10 ** (SNR_dB / 10))

    # Generar ruido gaussiano con la potencia calculada
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Agregar ruido a la señal transmitida
    noise_signal = signal + noise

    return noise_signal, noise_power

def add_comb_pilots(M: np.array, amplitude,period, N_pilots) -> np.array:
    """
    Estima el H por LS y el simbolo recibido

    Parameters
    ----------
    M : np.array
        DESCRIPTION.
    amplitude : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.

    Returns
    -------
    all_symb : TYPE
        DESCRIPTION.

    """
    N = M.shape[0] #Cantidad subportadoras
    # En esta matriz van los simbolos mas los pilotos
    all_symb = np.zeros((N+N_pilots,M.shape[1]),dtype=M.dtype)
    peine = np.zeros((N+N_pilots,1),dtype=M.dtype)
    data_idx=0
    for f_idx in range(0,N+N_pilots):
        if f_idx%period == 0:
            # Si es un multiplo de pilot_period, mando un piloto
            all_symb[f_idx,:] = amplitude
            peine[f_idx] = amplitude # para devolver una muestra del peine
        else:  
            # Si no, copio la fila
            all_symb[f_idx,:] = M[data_idx,:]
            data_idx = data_idx+1
 
    return all_symb, peine
    

def interp_channel(H_pilots: np.array, idx:np.array, N, interpolation_method:str = 'linear') -> np.array:
    """
    Interpolador de canal

    Parameters
    ----------
    H_pilots : np.array
        Muestras del canal en los valores de idx
    idx : np.array
        Indice de las muestras
    N : TYPE
        Largo del canal interpolado
    interpolation_method : str, optional
        Metodo de interpolacion segun scipy.interpolate.interp1d
        Por defecto: 'lineal'

    Returns
    -------
    H_interp
        Canal interpolado desde 0 a N

    """
    x = np.arange(N) #absisas
    
    # Interp no extrapola, asi que aplano los puntos que esten
    # antes del primer piloto o despues del ultimo
    x[(x<np.min(idx))] = np.min(idx)
    x[(x>np.max(idx))] = np.max(idx)
    f = scipy.interpolate.interp1d(idx, H_pilots, kind=interpolation_method)
    
    return f(x)

def test_interp_channel():
    # Canal real
    N=10
    Hreal = 10*np.arange(N) #ordenadas
    x=np.arange(N) #absisas
    
    # Pilotos
    idx=np.array([2,3,5,8])
    Hpilots=Hreal[idx]
    
    # Interpolo
    Hinterp = interp_channel(Hpilots, idx, N)
    assert (Hinterp[idx] == Hreal[idx]).all()
    
    # Los puntos a la izquerda del primer piloto tienen que aplanarse
    xinf = x[x<np.min(idx)]
    assert (Hinterp[xinf] ==  Hreal[np.min(idx)]).all()
    
    # Los puntos a la derecha del ultimo piloto tambien
    xsup = x[x>np.max(idx)]
    assert (Hinterp[xsup] ==  Hreal[np.max(idx)]).all()

def interp_channel(H_pilots: np.array, idx:np.array, N, interpolation_method:str = 'cubic') -> np.array:
    """
    Interpolador de canal

    Parameters
    ----------
    H_pilots : np.array
        Muestras del canal en los valores de idx
    idx : np.array
        Indice de las muestras
    N : TYPE
        Largo del canal interpolado
    interpolation_method : str, optional
        Metodo de interpolacion segun scipy.interpolate.interp1d
        Por defecto: 'lineal'

    Returns
    -------
    H_interp
        Canal interpolado desde 0 a N

    """
    x = np.arange(N) #absisas
    
    # Interp no extrapola, asi que aplano los puntos que esten
    # antes del primer piloto o despues del ultimo
    x[(x<np.min(idx))] = np.min(idx)
    x[(x>np.max(idx))] = np.max(idx)
    f = scipy.interpolate.interp1d(idx, H_pilots, kind=interpolation_method)
    
    return f(x)


def ajustar_channel(H_pilots: np.array, indices:np.array, N, interpolation_method:str = 'linear') -> np.array:
    """
    Interpolador de canal

    Parameters
    ----------
    H_pilots : np.array
        Muestras del canal en los valores de idx
    indices : np.array
        Indice de las muestras
    N : TYPE
        Largo del canal interpolado
    interpolation_method : str, optional
        Metodo de interpolacion segun scipy.interpolate.interp1d
        Por defecto: 'lineal'

    Returns
    -------
    H_interp
        Canal interpolado desde 0 a N

    """
    
    x = np.arange(N) #absisas
    
    # Interp no extrapola, asi que aplano los puntos que esten
    # antes del primer piloto o despues del ultimo
    x[(x<np.min(indices))] = np.min(indices)
    x[(x>np.max(indices))] = np.max(indices)
    
    # Encontrar el polinomio que ajusta los datos seleccionados
    grado_polinomio = len(H_pilots) - 1
    coeficientes_polinomio = np.polyfit(indices, H_pilots, grado_polinomio)

    # Crear el polinomio a partir de los coeficientes
    polinomio = np.poly1d(coeficientes_polinomio)

    # Evaluar el polinomio en todos los índices para interpolar los datos
    H_interp = polinomio(indices)
    
    return H_interp


