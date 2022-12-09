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
    '''Produces noise of specified duration'''
    y = np.random.random(SAMPLE_RATE * duration)
    return y
    

# FILTERS

def hipass(audio, cutoffFreq, smoothing = 0.01):
    '''Applies a high-pass filter to audio'''

    if len(audio.shape) == 2: # if audio is stereo
        # Apply filter separately to each channel
        r = np.zeros(audio.shape)
        r[:,0] = hipass(audio[:,0], cutoffFreq, smoothing)
        r[:,1] = hipass(audio[:,1], cutoffFreq, smoothing)
        return r

    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth cutoff with sigmoid
    yf = yf * (1 / (1 + np.exp(-(1/smoothing)*(xf-cutoffFreq))))

    return irfft(yf)

def lowpass(audio, cutoffFreq, smoothing = 0.01):
    '''Applies a low-pass filter to audio'''

    if len(audio.shape) == 2: # if audio is stereo
        # Apply filter separately to each channel
        r = np.zeros(audio.shape)
        r[:,0] = lowpass(audio[:,0], cutoffFreq, smoothing)
        r[:,1] = lowpass(audio[:,1], cutoffFreq, smoothing)
        return r

    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth cutoff with sigmoid
    yf = yf * (1 - (1 / (1 + np.exp(-(1/smoothing)*(xf-cutoffFreq)))))

    return irfft(yf)

def bandpass(audio, centerFreq, smoothing = 100):
    '''Applies a band-pass filter to audio'''

    if len(audio.shape) == 2: # if audio is stereo
        # Apply filter separately to each channel
        r = np.zeros(audio.shape)
        r[:,0] = bandpass(audio[:,0], centerFreq, smoothing)
        r[:,1] = bandpass(audio[:,1], centerFreq, smoothing)
        return r

    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth band pass with guassian
    yf = yf * 2 / (1 + np.exp( (1 / (2 * (smoothing**2))) * ((xf - centerFreq)**2) ))

    return irfft(yf)

def bandstop(audio, centerFreq, smoothing = 100, distance = 200):
    '''Applies a band-stop filter to audio'''

    if len(audio.shape) == 2: # if audio is stereo
        # Apply filter separately to each channel
        r = np.zeros(audio.shape)
        r[:,0] = bandstop(audio[:,0], centerFreq, smoothing, distance)
        r[:,1] = bandstop(audio[:,1], centerFreq, smoothing, distance)
        return r

    yf = rfft(audio)
    xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

    # smooth band stop with overlapped lowpass and hipass
    yf = yf * (1 + (1 / (1 + np.exp(-(1/smoothing)*(xf-centerFreq-distance)))) - (1 / (1 + np.exp(-(1/smoothing)*(xf-centerFreq+distance)))))
    
    return irfft(yf)

# Utility

def normalize(audio):
    '''Normalizes audio to 16-bit integer format'''
    return np.int16((audio / audio.max()) * 32767)



if __name__ == '__main__':
    duration = 5

    guitarSampleRate, guitar = read("./input/Guitar.wav")

    guitar = bandstop(guitar, 1300, 50, 900)

    write("bandstop_guitar_1500hz_50_800.wav", guitarSampleRate, normalize(guitar))