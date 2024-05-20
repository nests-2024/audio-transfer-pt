import json
import librosa
import numpy as np
import torch

from transformers import VitsModel, AutoTokenizer

from utils import read_audio_spectrum

S_RATE = 22050

def get_json(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
    return d


class Impulse:
    def __init__(self, model_url, phrases_path):
        self.model = VitsModel.from_pretrained(model_url).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.phrases = get_json(phrases_path)

    def get_impulse(self, seed, target_sr=S_RATE):
        words = self.phrases[seed % len(self.phrases)]
        txt = " ''' ".join(words[:5])
        inputs = self.tokenizer(txt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        output = output.cpu().data.numpy().squeeze()
        output = librosa.resample(output, orig_sr=self.model.config.sampling_rate, target_sr=target_sr)
        return output, target_sr


class ImpulseSP(Impulse):
    def __init__(self):
        super().__init__("facebook/mms-tts-spa", "./txts/sp.json")


class ImpulsePT(Impulse):
    def __init__(self):
        super().__init__("facebook/mms-tts-por", "./txts/pt.json")


def average_spectrum_frequencies(spectrum):
    spectrum_shape = spectrum.shape
    spectrum_avg_1 = spectrum.mean(axis=0)
    spectrum_avg = np.expand_dims(spectrum_avg_1, 0).repeat(spectrum_shape[0], axis=0).clip(0, 1000)
    return spectrum_avg


def modulate_spectrum(content_s, style_s):
    content_s_avg = average_spectrum_frequencies(content_s)

    content_length = content_s.shape[1]
    style_length = style_s.shape[1]

    if style_length < content_length:
        m = (content_length // style_length) + 1
        style_s = np.tile(style_s, m)

    style_s = style_s[:, :content_length].clip(0, 1000)

    return (content_s_avg * style_s)

#TODO: time domain modulation
