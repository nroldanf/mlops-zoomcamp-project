import math
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

def nextpow2(i):
    n = 1
    while n < i: 
        n *= 2
    return n

def hjorth_parameters(signal: np.ndarray):
    # The axis along which the difference is taken, default is the last axis.
    dv_sig = np.diff(signal)
    dv_sig2 = np.diff(signal, 2)
    s0 = np.mean(np.square(signal)) # variance or activity of the signal
    s1 = np.mean(np.square(dv_sig)) # second moment
    s2 = np.mean(np.square(dv_sig2))
    # Hjorth parameters calculation
    activity = s0
    mobility = s1 / s0
    complexity = np.sqrt((s2 / s1) - mobility)
    return activity, mobility, complexity

def higuchi_fractal_dimension(signal: np.ndarray, kmax: int):
    N = len(signal)
    lmk = np.zeros((kmax, kmax))
    for k in range(0, kmax):
        for m in range(0, k):
            lmki = 0
            for i in range(0, int(np.fix((N - m) / k))):
                lmki = lmki + np.abs(signal[m + i*k] - signal[m + (i - 1)*k])
            ng = (N - 1) / (np.fix((N - m) / k) * k)
            lmk[m, k] = (lmki * ng) / k
    
    
    lk = np.zeros((1, kmax))
    for k in range(0, kmax):
        lk[0, k] = np.sum(lmk[0:k, k], axis=0) / k
    lnlk = np.log(lk).flatten()
    lnk = np.log(1 / np.arange(0, kmax))
    print(lnk.shape, lnlk.shape)
    b = np.polyfit(lnk, lnlk, 1)
    hfd = b[0] # polinomial coefs
    return hfd

def freq_feats(
    signal: np.ndarray, 
    win_size: int, 
    win_overlap: int,
    psd_win_size: int,
    psd_win_overlap: int,
    nfft: int,
    fs: int,
    freqlims: list  
):
    '''
    :param signal: (n_samples, 1)
    '''
        
    length = len(signal)
    step = int(win_size - win_overlap) # step length
    nwins = math.floor((length - win_overlap) / step) # number of windows
    # variables initialization
    totalpower: np.ndarray = np.zeros((nwins,))
    powdelta: np.ndarray = np.zeros((nwins,))
    powtheta: np.ndarray = np.zeros((nwins,))
    powalpha: np.ndarray = np.zeros((nwins,))
    powbeta: np.ndarray = np.zeros((nwins,))
    powlowgamma: np.ndarray = np.zeros((nwins,))
    freq_half_power: np.ndarray = np.zeros((nwins,))
    
    # counters inits
    wincount = 0
    index = 0
    
    while (wincount < nwins):
        # Window movement and extraction
        win_signal = signal[index:index+win_size,]
        f, pxx = welch(
            x=win_signal,
            fs=fs,
            nperseg=psd_win_size,
            noverlap=psd_win_overlap,
            nfft=nfft,
        )
        # total power
        # If X is a multidimensional array, then find returns a column vector of the linear indices of the result.
        #  returns the first n indices corresponding to the nonzero elements in X.
        indmin = int(np.argwhere(f>=freqlims[0])[0])
        indmax = int(np.argwhere(f>=freqlims[1])[0])
        v_totalpower = np.sum(pxx[indmin:indmax]) # size; 1, nchann
        totalpower[wincount,] = v_totalpower
        # relative power
        # delta 0.5 - 4.0 Hz
        ind1 = int(np.argwhere(f>=0.5)[0])
        ind2 = int(np.argwhere(f>=4)[0])
        v_powdelta = np.sum(pxx[ind1:ind2])
        v_powdelta = v_powdelta / v_totalpower # element wise division
        powdelta[wincount,] = v_powdelta
        
        # relative power
        # delta 4 - 8 Hz
        ind1 = int(np.argwhere(f>=4)[0])
        ind2 = int(np.argwhere(f>=8)[0])
        v_powtheta = np.sum(pxx[ind1:ind2])
        v_powtheta = v_powtheta / v_totalpower # element wise division
        powtheta[wincount,] = v_powtheta
        
        # relative power
        # delta 8 - 12 Hz
        ind1 = int(np.argwhere(f>=8)[0])
        ind2 = int(np.argwhere(f>=12)[0])
        v_powalpha = np.sum(pxx[ind1:ind2])
        v_powalpha = v_powalpha / v_totalpower # element wise division
        powalpha[wincount,] = v_powalpha
        
        # relative power
        # delta 12 - 30 Hz
        ind1 = int(np.argwhere(f>=12)[0])
        ind2 = int(np.argwhere(f>=30)[0])
        v_powbeta = np.sum(pxx[ind1:ind2])
        v_powbeta = v_powbeta / v_totalpower # element wise division
        powbeta[wincount,] = v_powbeta
        
        # relative power
        # delta 30 - 50 Hz
        ind1 = int(np.argwhere(f>=30)[0])
        ind2 = int(np.argwhere(f>=50)[0])
        v_powlowgamma = np.sum(pxx[ind1:ind2])
        v_powlowgamma = v_powlowgamma / v_totalpower # element wise division
        powlowgamma[wincount,] = v_powlowgamma        

        index += step        
        wincount += 1
        
    return powdelta, powtheta, powalpha, powbeta, powlowgamma

def temp_feats(signal: np.ndarray, winsize: int, winoverlap: int):
        
    length = signal.shape[0]
    step = int(winsize - winoverlap) # step length
    nwins = math.floor((length - winoverlap) / step) # number of windows
    chann = signal.shape[1]
    
    # variables init
    mu = np.zeros((nwins, chann))
    sigma = np.zeros((nwins, chann))
    sk = np.zeros((nwins, chann))
    kurt = np.zeros((nwins, chann))
    
    act = np.zeros((nwins, chann))
    mob = np.zeros((nwins, chann))
    comp = np.zeros((nwins, chann))
    
    # counters inits
    wincount = 0
    index = 0
    
    while (wincount < nwins):
        # Window movement and extraction
        win_signal = signal[index:index+winsize, :]        
        # mean, variance, skewness and kurtosis
        mu[wincount, :] = np.mean(win_signal)
        sigma[wincount, :] = np.var(win_signal)
        sk[wincount, :] = skew(win_signal, axis=0)
        kurt[wincount, :] = kurtosis(win_signal, axis=0)
        
        # Hjörth Parameters
        [v_act, v_mob, v_comp] = hjorth_parameters(win_signal)
        act[wincount, :] = v_act
        mob[wincount, :] = v_mob
        comp[wincount, :] = v_comp
        
        index += step        
        wincount += 1
    return mu, sigma, sk, kurt, act, mob, comp