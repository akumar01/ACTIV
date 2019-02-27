import numpy as np
from scipy.signal import convolve
import pdb

# Convolve with complex Morlet wavelets centered at the given 
# frequencies. Wavelet widths are automatically chosen to get
# the best time resolution given the necessary frequency resolution 
# to distinguish the target freqs
def calc_wvlt_transform(X, freqs):

    # Complex morlet wavelet
    cmorlet = lambda t, f, s: 1/np.sqrt(s * np.sqrt(np.pi)) * np.exp(-t**2/(2 * s**2)) * np.exp(1j * 2*np.pi*f*t)

    # Calculate maximum permissible bandwidth ()
    delta_f = np.min(np.diff(freqs))
    bwidth = 1/(2 * delta_f)
    Xtrans = []
    for i, f in enumerate(freqs):

        wvlt = cmorlet(np.linspace(-2*bwidth, 2*bwidth, 100), f, bwidth)

        Xtrans.append(convolve(wvlt, X, 'valid'))

    return np.array(Xtrans)
