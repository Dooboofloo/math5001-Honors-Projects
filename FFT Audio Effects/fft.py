import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from scipy.io.wavfile import read, write

SAMPLE_RATE = 44100

# WAVE FORMS

def sine_wave(freq, duration):
    '''Produces a sine wave given frequency, sample rate, and duration'''
    x = np.linspace(0, duration, SAMPLE_RATE * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return y

def noise(duration):
    '''Produces noise'''
    y = np.random.random(SAMPLE_RATE * duration)
    return y
    

# FILTERS

def hipass(audio, cutoffFreq, smoothing = 0.01):
    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth cutoff with sigmoid
    yf = yf * (1 / (1 + np.exp(-(1/smoothing)*(xf-cutoffFreq))))

    return irfft(yf)

def lowpass(audio, cutoffFreq, smoothing = 0.01):
    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth cutoff with sigmoid
    yf = yf * (1 - (1 / (1 + np.exp(-(1/smoothing)*(xf-cutoffFreq)))))

    return irfft(yf)

def bandpass(audio, centerFreq, smoothing = 100):
    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth band pass with guassian
    yf = yf * 2 / (1 + np.exp( (1 / (2 * (smoothing**2))) * ((xf - centerFreq)**2) ))

    return irfft(yf)

def bandstop(audio, centerFreq, smoothing = 100, distance = 4):
    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth band stop with overlapped lowpass and hipass
    yf = yf * (1 + (1 / (1 + np.exp(-(1/smoothing)*(xf-centerFreq-distance)))) - (1 / (1 + np.exp(-(1/smoothing)*(xf-centerFreq+distance)))))
    
    return irfft(yf)

def normalize(audio):
    '''Normalizes audio to 16-bit integer format'''
    return np.int16((audio / audio.max()) * 32767)

if __name__ == '__main__':
    duration = 5

    # ex_tone = sine_wave(440, duration)
    # higher = sine_wave(554.3653, duration)

    n = noise(duration)
    fn = bandstop(n, 4000, 100, 2000)

    yn = rfft(n)
    xn = rfftfreq(SAMPLE_RATE * duration, 1 / SAMPLE_RATE)

    yfn = rfft(fn)

    write("noise.wav", SAMPLE_RATE, normalize(n))
    write("bandpass.wav", SAMPLE_RATE, normalize(fn))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(xn[1:], np.abs(yn)[1:])
    ax2.plot(xn[1:], np.abs(yfn)[1:])
    
    ax1.set_title("Random Noise")
    ax2.set_title("Bandstop (freq=4000, smoothing=100, distance=2000)")

    fig.suptitle("Noise vs Bandstop")

    plt.show()