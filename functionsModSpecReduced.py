def checkModInputs(fs, frameSize, windowSize, modFrameSize, modWindowSize, window, sym=False):
    """
    This function checks the frame sizes and (a) throws an error if they are not suitable (e.g. if 
    the modulation window is not an integral multiple of the acoustic frame step) and (b) returns a
    warning message if the constant overlap-add (COLA) principle is not satisfied.

    Inputs:
        - fs: the sampling frequency of the speech files, form "16000"
        - frameSize: the step size of the acoustic windows in seconds, form "0.001"
        - windowSize: the acoustic window size in seconds, form "0.03"
        - modFrameSiza: the step size of the modulation windows in seconds, form "0.1"
        - modWindowSize: the modulation window size in seconds, form "1"
        - window: the window type to use for both acoustic and modulation domains which
          must be of the ones recognised by SciPy without additional variables, form "hamming" 
        - sym: whether a symmetric window should be used, form "False"

    Outputs:
        - acWin: the acoustic window as a numpy array
        - modWin: the modulation window as a numpy array
    """
    from scipy.signal.windows import get_window, hamming
    from scipy.signal import check_COLA

    w = round(frameSize * 10000) # This assumes never less than 0.1 ms steps
    x = round(windowSize * 10000)
    y = round(modFrameSize * 10000)
    z = round(modWindowSize * 10000)
    
    if (z%w) != 0:
        raise Exception("The modulation window size must be an integral multiple of the modulation frame sizes.")
    elif w > x:
        raise Exception("The acoustic window size must be greater than or equal to the acoustic frame size.")
    elif y > z:
        raise Exception("The modulation window size must be greater than or equal to the modulation frame size.")
    elif (modWindowSize*100000)/10 != round(z):
        raise Exception("The modulation window size must be a multiple of 0.01 s.")
    elif (modFrameSize*100000)/10 != round(y):
        raise Exception("The modulation frame size must be a multiple of 0.01 s.")

    print("Frame and window sizes checked, all OK.")

    # Testing COLA over modulation windows
    acWin = get_window(window, round(windowSize*fs), not sym)
    modWin = get_window(window, round(modWindowSize/frameSize), not sym)

    acCola = check_COLA(acWin, int(windowSize*fs), int(round(windowSize-frameSize, 4)*fs), tol=1e-10)
    if acCola == False:
        txt = "The constant overlap-add test (COLA) is not satisfied for the acoustic windows, "
        txt += "so the speech signal may not be perfectly reproducible."
        print(txt)
    else:
        print("The constant overlap-add test (COLA) is satisfied for the acoustic windows.")
    modCola = check_COLA(modWin, int(modWindowSize/frameSize), int((modWindowSize-modFrameSize)/frameSize), tol=1e-10)
    if modCola == False:
        txt = "The constant overlap-add test (COLA) is not satisfied for the modulation windows, "
        txt += "so the speech signal may not be perfectly reproducible."
        print(txt)
    else:
        print("The constant overlap-add test (COLA) is satisfied for the modulation windows.")

    return acWin, modWin

def generateFrameSizes(sig, frameSize, windowSize, modFrameSize, modWindowSize, fs, padFront=False, padEnd=False):
    """
    To set frame and window sizes.  This speeds up the overall computation by setting up zero matrices of the
    relevant sizes which can be amended by the subsequent functions (appending rows is much slower).
    Set padFront=True if reconstructing the signal as this should help remove transients.
    Set padEnd=True to pad out the speech signal to full modulation frames.

    Inputs:
        - sig: the speech signal to analyse
        - frameSize: the step size of the acoustic windows in seconds, form "0.001"
        - windowSize: the acoustic window size in seconds, form "0.03"
        - modFrameSiza: the step size of the modulation windows in seconds, form "0.1"
        - modWindowSize: the modulation window size in seconds, form "1"
        - fs: the sampling frequency of the speech files, form "16000"
        - padFront: whether to pad the front of the signal with zeroes, form="False"
        - PadEnd: whether to pad th end of the signal to ensure full modulation frames, form=False"

    Outputs:
        - nFrames: the number of acoustic frames that the speech signal is broken up into
        - nModFrames: the number of modulation frames that the speech signal is broken up into
        - sig: the lengthened signal, only returned if padEnd = True
        - nFramesPad: the lengthened number of acoustic frames, only returned if padEnd = True
    """
    import math
    import numpy as np

    time = len(sig)/fs

    nFrames = math.floor(round((time-windowSize+frameSize)/frameSize, 3))

    if padFront == False and padEnd == False:
        nModFrames = math.floor(round((time-(windowSize-frameSize)-(modWindowSize-modFrameSize))/modFrameSize, 3))
    elif padFront == False and padEnd == True:
        nModFrames = math.ceil(round((time-(windowSize-frameSize)-(modWindowSize-modFrameSize))/modFrameSize, 3))
        newSigLen = round(nModFrames*modFrameSize*fs) + round((modWindowSize-modFrameSize)*fs) + round((windowSize-frameSize)*fs)
        print("newSigLen is {}".format(newSigLen))
        sig = np.concatenate((sig, np.zeros(newSigLen-len(sig))), axis=0)
        nFramesPad = round((newSigLen - (windowSize-frameSize)*fs)/(frameSize*fs))
    elif padFront == True and padEnd == True:
        extraModFrames = 5
        nModFrames = extraModFrames + math.ceil(round((time-(windowSize-frameSize)-(modWindowSize-modFrameSize))/modFrameSize, 3))
        newSigLen = round(nModFrames*modFrameSize*fs) + round((modWindowSize-modFrameSize)*fs) + round((windowSize-frameSize)*fs)
        print("newSigLen is {}".format(newSigLen))
        sig = np.concatenate((np.zeros(round(extraModFrames*modFrameSize*fs)), sig, np.zeros(round(newSigLen-len(sig)-extraModFrames*modFrameSize*fs))), axis=0)
        nFramesPad = round((newSigLen - (windowSize-frameSize)*fs)/(frameSize*fs))
        print("nFramesPad is {}".format(nFramesPad))
    else:
        print("The code assumes padFront can only be true if padEnd is also true.")

    nModFrames = max(nModFrames, 0) # Floor at 0

    if padEnd == False:
        return nFrames, nModFrames
    else:
        return nFrames, nModFrames, sig, nFramesPad

def generateStage1FFT(fs, sig, frameSize, windowSize, nFrames, acWin, form="magnitude", phases=False, nFilts=40, nMFCCs=19):
    """
    This function takes an input speech signal and returns the STFT.
    Specify form="magnitude" to generate Hilbert transform later - the Hilbert transform only works on real signals.
    Note the scaling has been added as that is in SciPy STFT and seems necessary for signal reconstruction.
    
    Inputs:
        - fs: the sampling frequency of the speech files, form "16000"
        - sig: the speech signal to analyse
        - frameSize: the step size of the acoustic windows in seconds, form "0.001"
        - windowSize: the acoustic window size in seconds, form "0.03"
        - nFrames: the number of acoustic frames that the speech signal is broken up into
        - acWin: the acoustic window as a numpy array
        - form: a choice of "magnitude", "complex" or "real" which determines the content of fft1matrix
        - phases: whether to return the phase information as a separate matrix, form "True"
    Outputs:
        - p: the number of FFT points based on the acoustic window, form "48"
        - fft1matrix: a 2D matrix size [nFrames x p] that contains the FFT of each acoustic window
        - fft1matrixphases: the phase of each element, only returned if phases = True
    """
    import math
    from scipy.fftpack import fft
    #from scipy.signal.windows import hamming, hann
    import numpy as np
    from python_speech_features import fbank, mfcc, ssc

    if form == "fbank":
        p = nFilts
    elif form == "mfcc":
        p = nMFCCs
    else:
        p = round(windowSize*fs/2+1) # p is the number of acoustic frequency bins afters limiting with Nyquist

    fft1matrix = np.zeros((nFrames, p)) if form != "complex" else np.zeros((nFrames, p), dtype=np.complex_)

    scale1 = acWin.sum()

    if phases == True:
        fft1matrixphases = np.zeros((nFrames, p))
    
    for i in range(nFrames):
        sigExtract = np.array(sig[i*round(frameSize*fs):i*round(frameSize*fs)+round(windowSize*fs)]) # Ignoring pre-emphasis filter here
        sigWin = fft(sigExtract*acWin)/scale1
        nfft = 2**math.ceil(np.log2(round(windowSize*fs)))
        if len(sigExtract) == round(windowSize*fs):
            if form == "magnitude":
                fft1matrix[i, :] = np.absolute(sigWin)[:p]
            elif form == "complex":
                fft1matrix[i, :] = sigWin[:p]
            elif form == "real":
                fft1matrix[i, :] = np.real(sigWin)[:p]
            elif form == "fbank":
                fft1matrix[i, :] = fbank(sigExtract*acWin, samplerate=fs, winlen=windowSize, winstep=frameSize, nfilt=nFilts, \
                                    nfft=nfft, lowfreq=0, highfreq=int(fs/2), preemph=0)[0][:p]
            elif form == "mfcc":
                fft1matrix[i, :] = mfcc(sigExtract*acWin, samplerate=fs, winlen=windowSize, winstep=frameSize, numcep=nMFCCs, 
                                        nfilt=nFilts, nfft=nfft, lowfreq=0, highfreq=int(fs/2), preemph=0, ceplifter=22,
                                        appendEnergy=False)[0][:p]
            else:
                print("Form must be \"magnitude\", \"complex\", \"real\", \"fbank\" or \"mfcc\".")
            if phases == True:
                fft1matrixphases[i, :] = np.angle(fft(sigExtract*acWin))[:p]
    print("fft1matrix shape is {}".format(fft1matrix.shape))
    if phases == True:
        return p, fft1matrix, fft1matrixphases
    elif form == "fbank":
        freqs1 = ssc(np.array(sig[:round(windowSize*fs)]), samplerate=fs, winlen=windowSize, winstep=frameSize, nfilt=40, nfft=512, \
                                lowfreq=0, highfreq=round(fs/2), preemph=0)[0][:p]
        return p, fft1matrix, freqs1
    else:
        return p, fft1matrix

def generateStage2FFT(frameSize, modFrameSize, modWindowSize, nModFrames, p, fft1matrix, modWin, form="magnitude", nFilts=30, nMFCCs=15):
    """
    This function takes an input spectrogram and returns a tensor modulation spectrum.
    Note form can be "magnitude", "complex" or "real", which describes what is done to the data resulting from each FFT.
    
    Inputs:
        - frameSize: the step size of the acoustic windows in seconds, form "0.001"
        - modFrameSiza: the step size of the modulation windows in seconds, form "0.1"
        - modWindowSize: the modulation window size in seconds, form "1"
        - nModFrames: the number of modulation frames that the speech signal is broken up into
        - p: the number of FFT points based on the acoustic window, form "48"
        - fft1matrix: a 2D matrix size [nFrames x p] that contains the FFT of each acoustic window
        - modWin: the modulation window as a numpy array
        - form: a choice of "magnitude", "complex" or "real" which determines the content of fft2matrix
    Outputs:
        - q: the number of FFT points based on the modulation window, form "1000"
        - fft2matrix: a 2D matrix size [nModFrames x (p*q)] that has the flattened modulation spectrum
          per row
        - logfft2matrix: the decibel magnitude of fft2matrix
    """
    import math
    from scipy.fftpack import fft
    #from scipy.signal.windows import hamming, hann
    import numpy as np
    from python_speech_features import fbank, mfcc, ssc
    import warnings
    
    if form == "fbank":
        q = nFilts
    elif form == "mfcc":
        q = nMFCCs
    else:
        q = round(modWindowSize/(2*frameSize)+1) # After applying Nyquist
    fft2matrix = np.zeros((nModFrames, p, q)) if form != "complex" else np.zeros((nModFrames, p, q), dtype=np.complex_)
    logfft2matrix = np.zeros((nModFrames, p, q))

    scale2 = modWin.sum()

    for i in range(nModFrames):
        if i%100 == 0:
            print("{:,}".format(i), end="\r")
        for j in range(p):
            sigExtract2 = fft1matrix[i*int(modFrameSize/frameSize):i*int(modFrameSize/frameSize)\
                            +round(modWindowSize/frameSize), j]
            sigWin2 = fft(sigExtract2*modWin)/scale2
            nfft = 2**math.ceil(np.log2(round(modWindowSize/frameSize)))
            if form == "magnitude":
                fft2matrix[i, j, :] = np.absolute(sigWin2)[:q]
            elif form == "complex":
                fft2matrix[i, j, :] = sigWin2[:q]
            elif form == "real":
                fft2matrix[i, j, :] = np.real(sigWin2)[:q]
            elif form == "fbank":
                fft2matrix[i, j, :] = fbank(sigExtract2*modWin, samplerate=round(1/frameSize), winlen=modWindowSize, winstep=modFrameSize, nfilt=nFilts, \
                                     nfft=nfft, lowfreq=0, highfreq=round(1/(2*frameSize)), preemph=0)[0][:q]
            elif form == "mfcc":
                fft2matrix[i, j, :] = mfcc(sigExtract2*modWin, samplerate=round(1/frameSize), winlen=modWindowSize, winstep=modFrameSize, numcep=nMFCCs, 
                                        nfilt=nFilts, nfft=nfft, lowfreq=0, highfreq=round(1/(2*frameSize)), preemph=0, ceplifter=22,
                                        appendEnergy=False)[0][:q]
            else:
                print("Form must be \"magnitude\", \"complex\", \"real\", \"fbank\" or \"mfcc\".")

            if form != "complex":
                logfft2matrix[i, j, :] = 20*np.log10(fft2matrix[i, j, :])[:q]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    logfft2matrix[i, j, :] = 10*np.log10(fft2matrix[i, j, :]*np.conj(fft2matrix[i, j, :]))[:q]

    print("fft2matrix shape is {}".format(fft2matrix.shape))

    if form == "fbank":
        freqs2 = ssc(np.array(fft1matrix[0, :round(modWindowSize/frameSize)]), samplerate=(1/frameSize), winlen=modWindowSize, winstep=modFrameSize, nfilt=nFilts, nfft=2048, \
                            lowfreq=0, highfreq=round(1/(2*frameSize)), preemph=0)[0][:q]
        return q, fft2matrix, logfft2matrix, freqs2
    else:
        return q, fft2matrix, logfft2matrix
