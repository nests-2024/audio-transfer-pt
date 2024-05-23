import gradio as gr
import numpy as np

from model import RandomCNN, run_transfer
from utils import read_audio_spectrum, spectrum_to_audio, spectrum_to_figure

NUM_INPUTS = 4

examples = [
    ["wavs/voices/boy.wav",
     "wavs/voices/girl.wav"],

    ["wavs/corpus/johntejada-1.wav",
     "wavs/target/beat-box-2.wav"],

    ["wavs/songs/imperial.mp3",
     "wavs/songs/usa.mp3"],

    ["wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XC181063-PHINOV36_0101_LIMPO.mp3",
     "wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3"],

    ["wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3",
     "wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XC181063-PHINOV36_0101_LIMPO.mp3"],
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


def concatenate_audios(paths):
    spectrum, p, sr = None, None, None
    filename = ""

    for path in paths:
        m_slug = path.split("/")[-1].split(" ")[0].split(".")[0][:32].split("-0-")[0]
        filename += m_slug + "-"

        m_s, m_p, m_sr = read_audio_spectrum(path)

        if spectrum is None:
            spectrum, p, sr = m_s, m_p, m_sr
        else:
            spectrum = np.concatenate((spectrum, m_s), axis=1)
            p = np.concatenate((p, m_p), axis=1)
            sr = m_sr

    return spectrum, p, sr, filename[:-1]


def clicked(*file_paths):
    content_paths = [p for p in file_paths[:NUM_INPUTS//2] if p is not None]
    style_paths = [p for p in file_paths[NUM_INPUTS//2:] if p is not None]

    if len(content_paths) < 1 or len(style_paths) < 1:
        return [gr.Audio(), gr.Image(), gr.Textbox()]

    content_s, content_p, content_sr, content_slug = concatenate_audios(content_paths)
    style_s, style_p, style_sr, style_slug = concatenate_audios(style_paths)

    kx, ky = 31, 21
    mcnn = RandomCNN(out_channels=768, kernel=(kx, ky), stride=(kx - 2, ky - 2))
    result = run_transfer(mcnn, content_s, style_s, num_steps=1500, content_weight=1, style_weight=1e12)

    result_spectrum = result.cpu().data.numpy().squeeze()
    result_img = spectrum_to_figure(result_spectrum)
    result_wav = spectrum_to_audio(result_spectrum, p=content_p, rounds=150)

    filename = f"c-{content_slug}_s-{style_slug}.wav"

    return [
        gr.Audio(visible=True, value=(content_sr, result_wav)),
        gr.Image(visible=True, value=result_img),
        gr.Textbox(visible=True, value=filename)
    ]


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(
        '''# Audio Style Transfer
           ### Based on: [[1]](https://arxiv.org/abs/1508.06576), [[2]](https://arxiv.org/abs/1710.11385) and [[3]](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
           ### Using PyTorch with GPU support
        '''
    )
    minputs = []
    for i in range(NUM_INPUTS):
        with gr.Row():
            mlabel = "Content" if i < NUM_INPUTS//2 else "Style"
            ma = gr.Audio(sources=["upload"], type="filepath", label=mlabel, visible=True)
            mi = gr.Image(visible=True, interactive=False, height=200)
            minputs.append(ma)
            minputs.append(mi)

    for ma in minputs[0::2]:
        ma.change(update_inputs, inputs=[*minputs], outputs=[*minputs[1::2]])

    result_but = gr.Button(value="Generate")
    with gr.Row():
        with gr.Column():
            result_wav = gr.Audio(label="Result", type="numpy", interactive=False, visible=True)
            result_name = gr.Textbox(label="name", visible=False)
        result_img = gr.Image(label="Result Spectogram", interactive=False, visible=False, height=200)
    
    result_but.click(clicked, inputs=[*minputs[0::2]], outputs=[result_wav, result_img, result_name])

    gr.Examples(examples=examples, fn=clicked,
                inputs=[*minputs[0::2]],
                outputs=[result_wav, result_img, result_name],
                cache_examples=True)

if __name__ == "__main__":
   demo.launch(show_api=False, server_name="0.0.0.0", server_port=7862)
