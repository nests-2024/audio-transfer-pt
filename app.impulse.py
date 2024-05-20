import glob
import gradio as gr
import numpy as np
import soundfile as sf

from model import RandomCNN, run_transfer
from impulse import ImpulsePT, ImpulseSP, average_spectrum_frequencies, modulate_spectrum
from utils import audio_to_spectrum, read_audio_spectrum, spectrum_to_audio, spectrum_to_figure

NUM_INPUTS = 2

STRATEGY = ["Transfer", "Modulation"]

examples = [
    ["wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XC181063-PHINOV36_0101_LIMPO.mp3"],
    ["wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3"],
]

m_impulse = ImpulseSP()

def update_inputs(*minputs):
    mouts = []
    for ma,mi in zip(minputs[0::2], minputs[1::2]):
        if ma is not None:
            spectrum, _, _ = read_audio_spectrum(ma)
            img = spectrum_to_figure(spectrum)
            mouts.append(gr.Image(value=img))
        elif ma is None and mi is not None:
            mouts.append(gr.Image(value=None))
        else:
            mouts.append(gr.Image())

    return [*mouts]


def clear_outputs():
    return [
        gr.Audio(visible=True, value=None),
        gr.Image(visible=False, value=None),
        gr.Textbox(visible=False, value="")
    ]


def transfer_spectrum(content_s, style_s, with_avg=True):
    if with_avg:
        kx, ky = 17, 17
        content_w, style_w = 1, 1e11
        content_s = average_spectrum_frequencies(content_s) * 1e-4
    else:
        kx, ky = 17, 17
        content_w, style_w = 1, 1e14

    mcnn = RandomCNN(out_channels=392, kernel=(kx, ky), stride=(kx - 2, ky - 2))
    result = run_transfer(mcnn, content_s, style_s, num_steps=1000, content_weight=content_w, style_weight=style_w)

    result_spectrum = result.cpu().data.numpy().squeeze().clip(0, 1e3)
    return result_spectrum


def clicked(seed, strategy, *file_paths):
    non_none_paths = [p for p in file_paths if p is not None]

    if len(non_none_paths) < 1:
        return [gr.Audio(), gr.Image(), gr.Textbox()]

    content_audio, content_sr = m_impulse.get_impulse(seed)
    content_spectrum, content_p = audio_to_spectrum(content_audio, content_sr)

    style_spectrum, style_p = None, None
    style_sr = None
    filename = ""

    for style_path in non_none_paths:
        m_style_slug = style_path.split("/")[-1].split(" ")[0].split(".")[0][:32].split("-0-")[0]
        filename += m_style_slug + "-"

        m_style_s, m_style_p, m_style_sr = read_audio_spectrum(style_path)

        if style_spectrum is None:
            style_spectrum = m_style_s
            style_p = m_style_p
            style_sr = m_style_sr
        else:
            style_spectrum = np.concatenate((style_spectrum, m_style_s), axis=1)
            style_p = np.concatenate((style_p, m_style_p), axis=1)
            style_sr = m_style_sr

    if strategy == STRATEGY[0]:
        result_spectrum = transfer_spectrum(content_spectrum, style_spectrum)
    else:
        result_spectrum = modulate_spectrum(content_spectrum, style_spectrum)

    result_img = spectrum_to_figure(result_spectrum)
    result_wav = spectrum_to_audio(result_spectrum, rounds=128)

    filename = f"{filename[:-1]}-{strategy}-{seed}.wav"

    return [
        gr.Audio(visible=True, value=(content_sr, result_wav)),
        gr.Image(visible=True, value=result_img),
        gr.Textbox(visible=True, value=filename)
    ]


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(
        '''# Audio Texture Synthesis
           ### Using PyTorch with GPU support
        '''
    )
    minputs = []
    for i in range(NUM_INPUTS):
        with gr.Row():
            ma = gr.Audio(sources=["upload"], type="filepath", label=f"Style-{i}", visible=(i==0))
            mi = gr.Image(label=f"STFT-{i}", visible=(i==0), interactive=False, height=200)
            minputs.append(ma)
            minputs.append(mi)

    for ma in minputs[0::2]:
        ma.change(update_inputs, inputs=[*minputs], outputs=[*minputs[1::2]])

    seed_slide = gr.Slider(minimum=0, maximum=1023, step=1, value=0, label="Seed")
    type_radio = gr.Radio(choices=STRATEGY, value=STRATEGY[0], label="Strategy")
    result_but = gr.Button(value="Generate")

    with gr.Row():
        with gr.Column():
            result_wav = gr.Audio(label="Result", type="numpy", interactive=False, visible=True)
            result_name = gr.Textbox(label="name", visible=False)
        result_img = gr.Image(label="Result Spectogram", interactive=False, visible=False, height=200)

    type_radio.change(clear_outputs, inputs=[], outputs=[result_wav, result_img, result_name])
    result_but.click(clicked, inputs=[seed_slide, type_radio, *minputs[0::2]], outputs=[result_wav, result_img, result_name])


    '''
    gr.Examples(examples=examples, fn=clicked,
                inputs=[*minputs[0::2]],
                outputs=[result_wav, result_img, result_name],
                cache_examples=True)
    '''


if __name__ == "__main__":
   demo.launch(show_api=False, server_name="0.0.0.0", server_port=7863)
