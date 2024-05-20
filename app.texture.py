import glob
import gradio as gr
import numpy as np

from model import RandomCNN, run_transfer
from utils import read_audio_spectrum, spectrum_to_audio, spectrum_to_figure

NUM_INPUTS = 2

all_birds = sorted(glob.glob("./wavs/birds/**/*.mp3"))

examples = [
    ["wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XC181063-PHINOV36_0101_LIMPO.mp3"],
    ["wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3"],
]


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


def clicked(seed=-1, *file_paths):
    non_none_paths = [p for p in file_paths if p is not None]

    if len(non_none_paths) < 1:
        return [gr.Audio(), gr.Image(), gr.Textbox()]

    content_path = all_birds[int(seed) % len(all_birds)]
    content_s, content_p, content_sr = read_audio_spectrum(content_path)

    content_shape = content_s.shape
    content_avg = content_s.mean(axis=0)
    content_s = np.expand_dims(content_avg, 0).repeat(content_shape[0], axis=0)

    style_s = None
    filename = ""
    for style_path in non_none_paths:
        m_style_slug = style_path.split("/")[-1].split(" ")[0].split(".")[0][:32].split("-0-")[0]
        filename += m_style_slug + "-"
        m_style_s, _, _ = read_audio_spectrum(style_path)
        if style_s is None:
            style_s = m_style_s
        else:
            style_s = np.concatenate((style_s, m_style_s), axis=1)

    kx, ky = 17, 17
    content_weight = 0 if seed == -1 else 1e-1
    style_weight = 1e12 if seed == -1 else 1e10

    mcnn = RandomCNN(out_channels=392, kernel=(kx, ky), stride=(kx - 2, ky - 2))
    result = run_transfer(mcnn, content_s, style_s, num_steps=1500, content_weight=content_weight, style_weight=style_weight)
    result_spectrum = result.cpu().data.numpy().squeeze().clip(0, 1e3)

    result_img = spectrum_to_figure(result_spectrum)
    result_wav = spectrum_to_audio(result_spectrum)
    filename = f"{filename[:-1]}.wav"

    return [
        gr.Audio(visible=True, value=(content_sr, result_wav)),
        gr.Image(visible=True, value=result_img),
        gr.Textbox(visible=True, value=filename)
    ]


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(
        '''# Audio Texture Synthesis
           ### Based on: [[1]](https://arxiv.org/abs/1905.03637)
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

    seed_slide = gr.Slider(minimum=-1, maximum=1000, step=1, value=-1, label="Seed")
    result_but = gr.Button(value="Generate")

    with gr.Row():
        with gr.Column():
            result_wav = gr.Audio(label="Result", type="numpy", interactive=False, visible=True)
            result_name = gr.Textbox(label="name", visible=False)
        result_img = gr.Image(label="Result Spectogram", interactive=False, visible=False, height=200)
    
    result_but.click(clicked, inputs=[seed_slide, *minputs[0::2]], outputs=[result_wav, result_img, result_name])

    '''
    gr.Examples(examples=examples, fn=clicked,
                inputs=[*minputs[0::2]],
                outputs=[result_wav, result_img, result_name],
                cache_examples=True)
    '''


if __name__ == "__main__":
   demo.launch(show_api=False, server_name="0.0.0.0", server_port=7864)
