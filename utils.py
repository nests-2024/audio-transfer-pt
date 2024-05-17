import io
import librosa
import matplotlib.pyplot as plt
import numpy as np

N_FFT = 1024
HOP_LEN = 256
WIN_LEN = 1024
S_RATE = 22050
MAX_LEN = 16

def read_audio_spectrum(filename, duration=MAX_LEN, sr=S_RATE):
    x, sr = librosa.load(filename, sr=sr)
    S = librosa.stft(x, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
    last_sample = int(duration * sr / HOP_LEN)

    p = np.angle(S[:, :last_sample])
    S = np.log1p(np.abs(S[:, :last_sample]))
    return S, sr, p


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


#TODO: time domain modulation


def modulate_spectrum(carrier_s, content_s, carrier_p, carrier_sr=S_RATE):
    carrier_shape = carrier_s.shape
    carrier_avg = carrier_s.mean(axis=0)
    carrier_avg_s = np.expand_dims(carrier_avg, 0).repeat(carrier_shape[0], axis=0).clip(0, 1000)

    carrier_length = carrier_s.shape[1]
    content_length = content_s.shape[1]

    if content_length < carrier_length:
        m = (carrier_length // content_length) + 1
        content_s = np.tile(content_s, m)

    content_s = content_s[:, :carrier_length].clip(0, 1000)

    return np.power(carrier_avg_s * content_s, 1), carrier_sr, carrier_p
