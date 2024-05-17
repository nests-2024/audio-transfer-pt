import gradio as gr
import numpy as np

from model import RandomCNN, run_transfer
from utils import read_audio_spectrum, spectrum_to_audio, spectrum_to_figure

NUM_INPUTS = 2

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


def clicked(*file_paths):
    content_path = file_paths[0]
    style_path = file_paths[1]

    if content_path is None or style_path is None:
        return [gr.Audio(), gr.Image(), gr.Textbox()]

    content_spectrum, content_sr, content_p = read_audio_spectrum(content_path)
    style_spectrum, style_sr, style_p = read_audio_spectrum(style_path)

    kx, ky = 17, 5
    mcnn = RandomCNN(out_channels=384, kernel=(kx, ky), stride=(kx - 2, ky - 2))
    result = run_transfer(mcnn, content_spectrum, style_spectrum, num_steps=1500, content_weight=1e-1, style_weight=1e10)

    result_spectrum = result.cpu().data.numpy().squeeze()
    result_img = spectrum_to_figure(result_spectrum)
    result_wav = spectrum_to_audio(result_spectrum, p=content_p, rounds=150)

    content_slug = content_path.split("/")[-1].split(" ")[0].split(".")[0][:32].split("-0-")[0]
    style_slug = style_path.split("/")[-1].split(" ")[0].split(".")[0][:32].split("-0-")[0]
    filename = f"c-{content_slug}_s-{style_slug}.wav"

    return [
        gr.Audio(visible=True, value=(content_sr, result_wav)),
        gr.Image(visible=True, value=result_img),
        gr.Textbox(visible=True, value=filename)
    ]


with gr.Blocks() as demo:
    gr.Markdown("# Audio Style Transfer\n### Following [this paper](https://arxiv.org/abs/1508.06576)\n### Using PyTorch with GPU support")
    minputs = []
    for i in range(NUM_INPUTS):
        with gr.Row():
            mlabel = "Content" if len(minputs) < 1 else "Style"
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

# allow_flagging="never",
# analytics_enabled=None

if __name__ == "__main__":
   demo.launch(show_api=False, server_name="0.0.0.0", server_port=7862)
