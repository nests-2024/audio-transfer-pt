import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_specinv import griffin_lim

N_FFT = 2048
WIN_SIZE = 2048
S_RATE = 22050
MAX_LEN = 16


def read_audio_spectrum_pt(filename, duration=MAX_LEN, windowsize=WIN_SIZE, sr=S_RATE):
    x, sr = librosa.load(filename, duration=duration, sr=sr)
    x = torch.from_numpy(x)
    window = torch.hann_window(windowsize)
    S = torch.view_as_real(torch.stft(x, 2048, window=window, return_complex=True))
    mag = S.pow(2).sum(2).sqrt().log1p()
    return mag

def spectrum_to_audio_pt(spectrum, windowsize=WIN_SIZE, rounds=100, alpha=0.3):
    if torch.cuda.is_available():
        spectrum = spectrum.expm1().cuda()
        window = torch.hann_window(windowsize).cuda()
        rounds = max(8, min(rounds, 64))
    else:
        spectrum = spectrum.expm1()
        window = torch.hann_window(windowsize)
        rounds = max(1, min(rounds, 8))

    xhat = griffin_lim(spectrum, maxiter=rounds, alpha=alpha, window=window)
    return xhat


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


def spectrum_to_audio(spectrum, p=None, n_fft=N_FFT, rounds=128):
    if p is None:
        p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi

    spectrum = np.expm1(spectrum)
    for i in range(rounds):
        S = spectrum * np.exp(1j * p)
        wav = librosa.istft(S)
        p = np.angle(librosa.stft(wav, n_fft=n_fft))
    return wav
