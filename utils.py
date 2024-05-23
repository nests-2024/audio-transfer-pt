import io
import librosa
import matplotlib.pyplot as plt
import numpy as np

N_FFT = 1024
HOP_LEN = 256
WIN_LEN = 1024
S_RATE = 22050
MAX_LEN = 30


def audio_to_spectrum(x, sr, duration=MAX_LEN, hop_length=HOP_LEN, win_length=WIN_LEN):
    S = librosa.stft(x, n_fft=N_FFT, hop_length=hop_length, win_length=win_length)
    last_sample = int(duration * sr / hop_length)
    p = np.angle(S[:, :last_sample])
    s = np.log1p(np.abs(S[:, :last_sample]))
    s01 = (s - s.min()) / s.ptp()
    return s01, p


def read_audio_spectrum(filename, target_sr=S_RATE, duration=MAX_LEN):
    x, sr = librosa.load(filename, sr=target_sr)
    s, p = audio_to_spectrum(x, duration, sr)
    return s, p, sr


def plot_spectrum(spectrum):
    spec_db = librosa.amplitude_to_db(spectrum, ref=np.max)
    librosa.display.specshow(spec_db)
    plt.show()


def spectrum_to_audio(spectrum, p=None, rounds=64):
    if p is None:
        p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi

    spectrum = np.expm1(spectrum)
    for i in range(rounds):
        S = spectrum * np.exp(1j * p)
        wav = librosa.istft(S, hop_length=HOP_LEN, win_length=WIN_LEN)
        p = np.angle(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN))
    return wav


def spectrum_to_figure(spectrum):
    fig = plt.figure(figsize=(8, 2.5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.imshow(spectrum.clip(0, 1e3), aspect='auto')

    iw, ih = int(fig.bbox.bounds[2]), int(fig.bbox.bounds[3])

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=100)
    io_buf.seek(0)
    fig_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape(ih, iw, -1)
    io_buf.close()

    return fig_arr
