import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

N_FFT = 2048
S_RATE = 22050
MAX_LEN = 16

def read_audio_spectrum(filename, duration=MAX_LEN, n_fft=N_FFT, sr=S_RATE):
    x, sr = librosa.load(filename, duration=duration, sr=sr)
    S = librosa.stft(x, n_fft=n_fft)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, sr, p


def plot_spectrum(spectrum):
    spec_db = librosa.amplitude_to_db(spectrum, ref=np.max)
    librosa.display.specshow(spec_db)
    plt.show()


def spectrum_to_audio(spectrum, p=None, n_fft=N_FFT, rounds=64):
    if p is None:
        p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi

    spectrum = np.expm1(spectrum)
    for i in range(rounds):
        S = spectrum * np.exp(1j * p)
        wav = librosa.istft(S)
        p = np.angle(librosa.stft(wav, n_fft=n_fft))
    return wav
